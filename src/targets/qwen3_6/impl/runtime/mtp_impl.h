#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

// One rank's window into ITS OWN MtpDecodeState, sliced to the round's batch. Both ranks are
// sliced by the same function so a shape mistake cannot apply to one device only -- which matters
// because rank 1's buffers live on the other GPU, where an out-of-bounds write is silent: a
// peer-side write that is exactly in bounds at batch 1 runs off the end of the buffer at batch 2
// with nothing on the local device noticing.
struct MtpRoundView {
    Tensor anchors;
    Tensor frontiers;
    Tensor budgets;
    Tensor current_extents;
    Tensor target_valid;
    Tensor current_drafts;
    Tensor target_rope;
    Tensor text_rows;
    Tensor mtp_rows;
    Tensor lanes;
    Tensor rope_deltas;
    Tensor verify_ids;
    Tensor target_positions;
    Tensor target_tokens;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor selected_hidden;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted;
    Tensor next_extents;
    Tensor alignment_ids;
    Tensor alignment_hidden;
    Tensor ar_hidden;
    Tensor next_hidden;
    Tensor ar_positions;
    Tensor ar_rope_positions;
    Tensor ar_valid_columns;
    Tensor next_drafts;
    Tensor proposal_logits;
    const ops::SamplingConfig* sampling = nullptr;
};

MtpRoundView slice_mtp_frame(qwen3_6::MtpDecodeState& frame, std::int32_t batch_size) {
    MtpRoundView out;
    out.anchors           = frame.anchors.slice(0, 0, batch_size);
    out.frontiers         = frame.base_frontiers.slice(0, 0, batch_size);
    out.budgets           = frame.remaining_budgets.slice(0, 0, batch_size);
    out.current_extents   = frame.current_extents.slice(0, 0, batch_size);
    out.target_valid      = frame.target_valid_columns.slice(0, 0, batch_size);
    out.current_drafts    = frame.current_drafts.slice(1, 0, batch_size);
    out.target_rope       = frame.target_rope_positions.slice(1, 0, batch_size);
    out.text_rows         = frame.text_kv_table_rows.slice(0, 0, batch_size);
    out.mtp_rows          = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
    out.lanes             = frame.lanes.slice(0, 0, batch_size);
    out.rope_deltas       = frame.rope_deltas.slice(0, 0, batch_size);
    out.verify_ids        = frame.verify_ids.slice(1, 0, batch_size);
    out.target_positions  = frame.target_positions.slice(1, 0, batch_size);
    out.target_tokens     = frame.target_argmax.slice(1, 0, batch_size);
    out.target_logits     = frame.target_logits.slice(2, 0, batch_size);
    out.target_hidden     = frame.target_hidden.slice(2, 0, batch_size);
    out.selected_hidden   = frame.target_continuation_hidden.slice(1, 0, batch_size);
    out.licensed_tokens   = frame.licensed_tokens.slice(1, 0, batch_size);
    out.licensed_counts   = frame.licensed_counts.slice(0, 0, batch_size);
    out.accepted          = frame.accepted_drafts.slice(0, 0, batch_size);
    out.next_extents      = frame.next_extents.slice(0, 0, batch_size);
    out.alignment_ids     = frame.alignment_ids.slice(1, 0, batch_size);
    out.alignment_hidden  = frame.alignment_hidden.slice(2, 0, batch_size);
    out.ar_hidden         = frame.ar_hidden.slice(1, 0, batch_size);
    out.next_hidden       = frame.next_hidden.slice(1, 0, batch_size);
    out.ar_positions      = frame.ar_positions.slice(0, 0, batch_size);
    out.ar_rope_positions = frame.ar_rope_positions.slice(0, 0, batch_size);
    out.ar_valid_columns  = frame.ar_valid_columns.slice(0, 0, batch_size);
    out.next_drafts       = frame.next_drafts.slice(0, 0, batch_size);
    out.proposal_logits   = frame.proposal_logits.slice(1, 0, batch_size);
    out.sampling          = frame.sampling;
    return out;
}

TargetVerifyFrameView verify_view(const MtpRoundView& v, const GdnReplayRecords* records) {
    return TargetVerifyFrameView{
        .ids             = v.verify_ids,
        .cache_positions = v.target_positions,
        .rope_positions  = v.target_rope,
        .valid_columns   = v.target_valid,
        .kv_table_rows   = v.text_rows,
        .lanes           = v.lanes,
        .target_hidden   = v.target_hidden,
        .target_logits   = v.target_logits,
        .target_tokens   = v.target_tokens,
        .drafts          = v.current_drafts,
        .current_extents = v.current_extents,
        .frontiers       = v.frontiers,
        .anchors         = v.anchors,
        .licensed_tokens = v.licensed_tokens,
        .licensed_counts = v.licensed_counts,
        .accepted_drafts = v.accepted,
        .selected_hidden = v.selected_hidden,
        .replay_records  = records,
        .sampling        = v.sampling,
    };
}

} // namespace

void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding, const Tensor* peer_previous_hidden) {
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    if (state.execution.peer != nullptr) {
        // tp2 bridge. The bridge resumes the MTP head from a RETAINED target hidden at the reuse
        // frontier. That hidden is bit-identical on both ranks (final_norm is replicated and the
        // target output is all-reduced), and each rank keeps its OWN copy: rank 0's in
        // `previous_hidden` (the per-lane tail/checkpoint store) and rank 1's in
        // `peer_previous_hidden` (PeerRuntime's mirror, kept in sync by copy_tail + the MTP decode
        // round's accepted-hidden scatter). The MTP stem consumes them per rank -- rank 0's fc
        // shard contracts the normalized EMBEDDING half, rank 1's the normalized HIDDEN half -- and
        // the tail's cached attention appends AND reads each rank's own MTP KV shard at `position`,
        // so the MTP KV frontier advances by one on both ranks in lockstep (the same A1 semantics
        // the tp1 bridge relies on). No cross-rank transfer of the hidden is needed: each rank
        // feeds its own row-parallel fc its own retained hidden.
        if (build_proposal) {
            // The AR proposal steps (drafts 1..extent-1) have no tp2 twin; the planner stages the
            // tp2 bridge with build_proposal == false (the initial MTP draft comes from the staged
            // prefill's final chunk, and the zero-suffix exact-hit case that would build a proposal
            // here is downgraded to a full reset at tp2).
            throw std::logic_error("tensor-parallel MTP bridge does not build an AR proposal");
        }
        if (peer_previous_hidden == nullptr) {
            throw std::logic_error("tensor-parallel MTP bridge requires the peer's retained hidden");
        }
        std::optional<TpExecution> tp = tp_execution(state.execution);
        if (!tp) {
            throw std::logic_error("tensor-parallel MTP bridge has no peer execution");
        }
        // The per-sequence MTP KV window is request state, so rank 1's comes from the PrefillContext
        // rather than the process-lifetime peer core -- exactly as the text prefill does.
        tp->mtp_kv = state.mtp_kv_peer;
        if (tp->mtp_kv.valid() != state.mtp_kv.valid()) {
            throw std::logic_error("tensor-parallel MTP KV windows disagree between ranks");
        }
        state.execution.work.reset();
        tp->work->reset();
        TextContext card(state.execution.device, state.execution.model, state.execution.work,
                         state.execution.rope_frequency, state.text_kv,
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk,
                         state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                         &*tp);
        configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                            state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);
        // The MTP KV row was bound once per sequence at bind_sequence_kv and persists for the whole
        // request, so each rank's io.backend_kv_table_row already names this sequence's MTP page.
        const ExecutionContext& bridge_ec = *tp->execution;
        for_each_rank(bridge_ec, [&](int rank) {
            Tensor position_view = card.io_for(rank).mtp->target_positions.slice(0, 0, 1);
            ops::set_i32_scalar(position_view, position, card.stream_for(rank));
        });
        std::array<Tensor, 2> positions{state.execution.io.mtp->target_positions.slice(0, 0, 1),
                                        state.execution.peer->io->mtp->target_positions.slice(0, 0, 1)};
        std::array<Tensor, 2> previous{previous_hidden, *peer_previous_hidden};
        std::array<Tensor, 2> mtp_hidden{state.execution.io.mtp->ar_hidden,
                                         state.execution.peer->io->mtp->ar_hidden};
        const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
        const ops::GqaExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
        card.mtp_forward_batch_tp2(next_token, previous, positions,
                                   {rope_position[0], rope_position[1], rope_position[2]},
                                   bridge_envelope, mtp_hidden, -1, nullptr, nullptr);
        return;
    }
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::GqaExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    if (!build_proposal) { return; }

    if (state.mtp_proposal_extent == 0 ||
        state.mtp_proposal_extent >
            static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.execution.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::GqaExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                 envelope, next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.execution.io.mtp->ar_hidden.data, next_hidden.data,
                                   state.execution.io.mtp->ar_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, state.execution.device.stream));
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
}


auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                           MtpGqaEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        qwen3_6::MtpDecodeState& frame = state.frame;
        const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));
        std::optional<TpExecution> tp = tp_execution(state.execution);
        if (tp) {
            // Rank 1 runs the round from ITS OWN copy of the same ingress record. Everything the
            // peer needs that is not in the ingress -- verify ids, target positions, the accepted
            // count, the next round's AR positions -- is DERIVED from it by the same deterministic
            // Ops, run again on device 1, rather than transferred: the only other inputs are the
            // gathered logits, which are bit-identical on both ranks.
            if (!tp->io->mtp_decode.has_value()) {
                throw std::logic_error("tensor-parallel MTP decode requires a peer frame");
            }
            // Rank 1 uploads ITS OWN ingress record, not rank 0's. The two differ in exactly one
            // field per row -- `sampling[row].token_counts`, which must name rank 1's penalty
            // counter lane. `speculative_accept_greedy_drafts` reads and atomically writes that
            // pointer in sampling mode, so handing rank 1 a pointer into rank 0's arena is an
            // illegal access without peer mapping and a silent double-increment with it. Every
            // other byte is identical, which is what keeps the two replicated accepts in step.
            const qwen3_6::MtpDecodeIngress* peer_ingress =
                state.execution.peer->mtp_host_ingress;
            if (peer_ingress == nullptr) {
                throw std::logic_error("tensor-parallel MTP decode requires a peer ingress record");
            }
            const CurrentDevice restore;
            CUDA_CHECK(cudaSetDevice(tp->device->device));
            CUDA_CHECK(cudaMemcpyAsync(tp->io->mtp_decode->ingress.data, peer_ingress,
                                       sizeof(qwen3_6::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                       tp->device->stream));
        }

        TextContext card(state.execution.device, state.execution.model, state.execution.work,
                         state.execution.rope_frequency, {}, state.execution.linear_attention,
                         state.execution.io, state.execution.prefill_hidden,
                         state.execution.prefill_chunk, 0, {}, &state.text_cache,
                         &state.mtp_cache, tp ? &*tp : nullptr);

        MtpRoundView v = slice_mtp_frame(frame, batch_size);

        if (!tp) {
            ops::speculative_prepare_verify_inputs(v.anchors, v.current_drafts, v.frontiers,
                                                   v.current_extents, v.verify_ids,
                                                   v.target_positions,
                                                   state.execution.device.stream);
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 verify_view(v, state.execution.replay_records),
                                 envelopes.target_verify);

            ops::mtp_prepare_next_round(v.verify_ids, v.anchors, v.accepted, v.frontiers,
                                        v.budgets, v.licensed_counts, v.rope_deltas,
                                        v.alignment_ids, v.next_extents, v.ar_positions,
                                        v.ar_rope_positions, v.ar_valid_columns,
                                        static_cast<std::int32_t>(state.text_cache.max_context()),
                                        state.execution.device.stream);
            card.mtp_forward_decode_batch(v.alignment_ids, v.target_hidden, v.target_positions,
                                          v.target_rope, v.licensed_counts, v.mtp_rows,
                                          envelopes.batch, v.alignment_hidden);
            ops::speculative_select_accepted_hidden(v.alignment_hidden, v.accepted, v.ar_hidden,
                                                    state.execution.device.stream);

            Tensor draft0 = v.next_drafts.slice(1, 0, 1).view({batch_size});
            card.mtp_propose_batch(v.ar_hidden, v.proposal_logits, draft0);
            for (std::uint32_t step = 0; step + 1 < k; ++step) {
                Tensor previous =
                    v.next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
                Tensor next = v.next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1)
                                  .view({batch_size});
                Tensor position =
                    v.ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view({1,
                                                                                      batch_size});
                Tensor rope = v.ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                  .view({1, batch_size});
                Tensor valid = v.ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                                   .view({batch_size});
                Tensor previous_batch    = previous.view({1, batch_size});
                Tensor hidden_batch      = v.ar_hidden.view({TextConfig::hidden, 1, batch_size});
                Tensor next_hidden_batch = v.next_hidden.view({TextConfig::hidden, 1, batch_size});
                card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                              v.mtp_rows, envelopes.ar[step], next_hidden_batch);
                card.mtp_propose_batch(v.next_hidden, v.proposal_logits, next);
                CUDA_CHECK(cudaMemcpyAsync(v.ar_hidden.data, v.next_hidden.data,
                                           v.ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                           state.execution.device.stream));
            }
        } else {
            const ExecutionContext& ec = *tp->execution;
            MtpRoundView p         = slice_mtp_frame(*tp->io->mtp_decode, batch_size);
            MtpRoundView* views[2] = {&v, &p};
            for_each_rank(ec, [&](int rank) {
                MtpRoundView& r = *views[static_cast<std::size_t>(rank)];
                ops::speculative_prepare_verify_inputs(r.anchors, r.current_drafts, r.frontiers,
                                                       r.current_extents, r.verify_ids,
                                                       r.target_positions, ec.dev[rank]->stream);
            });
            target_verify_accept(state.execution, state.continuation_hidden_store,
                                 state.peer_continuation_hidden_store, card,
                                 verify_view(v, state.execution.replay_records),
                                 verify_view(p, tp->replay_records), envelopes.target_verify);
            for_each_rank(ec, [&](int rank) {
                MtpRoundView& r = *views[static_cast<std::size_t>(rank)];
                ops::mtp_prepare_next_round(
                    r.verify_ids, r.anchors, r.accepted, r.frontiers, r.budgets, r.licensed_counts,
                    r.rope_deltas, r.alignment_ids, r.next_extents, r.ar_positions,
                    r.ar_rope_positions, r.ar_valid_columns,
                    static_cast<std::int32_t>(state.text_cache.max_context()),
                    ec.dev[rank]->stream);
            });
            card.mtp_forward_decode_batch(v.alignment_ids, {v.target_hidden, p.target_hidden},
                                          {v.target_positions, p.target_positions},
                                          {v.target_rope, p.target_rope},
                                          {v.licensed_counts, p.licensed_counts},
                                          {v.mtp_rows, p.mtp_rows}, envelopes.batch,
                                          {v.alignment_hidden, p.alignment_hidden});
            for_each_rank(ec, [&](int rank) {
                MtpRoundView& r = *views[static_cast<std::size_t>(rank)];
                ops::speculative_select_accepted_hidden(r.alignment_hidden, r.accepted, r.ar_hidden,
                                                        ec.dev[rank]->stream);
            });

            const std::array<Tensor, 2> proposal_logits = {v.proposal_logits, p.proposal_logits};
            Tensor draft0 = v.next_drafts.slice(1, 0, 1).view({batch_size});
            card.mtp_propose_batch({v.ar_hidden, p.ar_hidden}, proposal_logits, draft0);
            for (std::uint32_t step = 0; step + 1 < k; ++step) {
                Tensor previous =
                    v.next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
                Tensor next = v.next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1)
                                  .view({batch_size});
                std::array<Tensor, 2> position;
                std::array<Tensor, 2> rope;
                std::array<Tensor, 2> valid;
                std::array<Tensor, 2> hidden_batch;
                std::array<Tensor, 2> next_hidden_batch;
                for (std::size_t r = 0; r < 2; ++r) {
                    MtpRoundView& view = *views[r];
                    position[r] = view.ar_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                      .view({1, batch_size});
                    rope[r] = view.ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                  .view({1, batch_size});
                    valid[r] = view.ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                                   .view({batch_size});
                    hidden_batch[r] = view.ar_hidden.view({TextConfig::hidden, 1, batch_size});
                    next_hidden_batch[r] =
                        view.next_hidden.view({TextConfig::hidden, 1, batch_size});
                }
                Tensor previous_batch = previous.view({1, batch_size});
                card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                              {v.mtp_rows, p.mtp_rows}, envelopes.ar[step],
                                              next_hidden_batch);
                card.mtp_propose_batch({v.next_hidden, p.next_hidden}, proposal_logits, next);
                for_each_rank(ec, [&](int rank) {
                    MtpRoundView& r = *views[static_cast<std::size_t>(rank)];
                    CUDA_CHECK(cudaMemcpyAsync(r.ar_hidden.data, r.next_hidden.data,
                                               r.ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                               ec.dev[rank]->stream));
                });
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpGqaEnvelopes envelopes, DecodeGraphDefinition& definition) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    capture_graph(state, definition, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpGqaEnvelopes envelopes, DecodeGraphExecutable* executable) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
