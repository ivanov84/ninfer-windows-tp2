#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

void validate_sampling(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
}

ops::SamplingConfig translate_sampling(const ResolvedSamplingParameters& source) {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.seed              = source.seed;
    out.token_counts      = nullptr;
    return out;
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint64_t projected_service_work(const runtime::RequestPlanSummary& summary,
                                     std::uint32_t reuse_base, std::uint32_t prefill_chunk,
                                     std::size_t prefill_splits) noexcept {
    const std::uint32_t suffix = summary.prompt_tokens - reuse_base;
    const std::uint64_t prefill_units =
        suffix == 0
            ? 1ULL
            : 1ULL + (static_cast<std::uint64_t>(suffix) - 1ULL) / prefill_chunk + prefill_splits;
    const std::uint64_t decode_units =
        summary.effective_output_tokens == 0 ? 0ULL : summary.effective_output_tokens - 1ULL;
    return prefill_units + decode_units;
}

// NINFER_DEBUG_REUSE=1: per-request trace of the prefix-reuse decision so a full_reset
// can be attributed to the exact gate that produced it.
bool reuse_debug_enabled() noexcept {
    static const bool enabled =
        std::getenv("NINFER_DEBUG_REUSE") != nullptr &&
        std::strcmp(std::getenv("NINFER_DEBUG_REUSE"), "0") != 0;
    return enabled;
}

const char* reuse_path_name(ReusePath path) noexcept {
    switch (path) {
    case ReusePath::FullReset: return "full_reset";
    case ReusePath::AppendAtFrontier: return "append_frontier";
    case ReusePath::RestoreTurnCheckpoint: return "restore_turn";
    case ReusePath::RestoreResponseCheckpoint: return "restore_response";
    }
    return "?";
}

} // namespace

RequestBasePlan
ProgramImplCore::plan_request_base(const PreparedPromptData& prompt,
                                   const runtime::ResolvedExecutionOptions& options) {
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (prompt.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("prompt contains token outside the 248077-token domain");
        }
    }
    if (prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3ULL * prompt.token_ids.size()) {
        throw std::invalid_argument("prepared prompt token metadata has an invalid shape");
    }
    if (prompt.has_media() != !prompt.media_payloads.empty() ||
        prompt.media_payloads.size() != prompt.vision_items.size()) {
        throw std::invalid_argument("prepared prompt media payload is incomplete");
    }
    for (std::size_t i = 0; i < prompt.media_payloads.size(); ++i) {
        if (!prompt.media_payloads[i] ||
            prompt.media_payloads[i]->patch_elements !=
                prompt.vision_items[i].patch_count * kPreparedVisionPatchFeatures) {
            throw std::invalid_argument("prepared prompt media item payload has an invalid shape");
        }
    }
    if (prompt.has_media() && !vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    validate_sampling(options.sampling);

    auto base                             = std::make_unique<RequestBasePlanImpl>();
    base->summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.token_ids.size());
    base->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        capacity - base->summary.prompt_tokens + static_cast<std::uint32_t>(1);
    base->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason = options.requested_output_tokens <= capacity_output
                                               ? FinishReason::OutputLimit
                                               : FinishReason::ContextCapacity;
    base->summary.transient_alignment    = 1;
    base->summary.transient_bytes        = 0;
    base->sampling                       = translate_sampling(options.sampling);
    base->allow_prefix_reuse             = options.allow_prefix_reuse;
    const std::uint32_t reserved_context_tokens =
        base->summary.prompt_tokens + (base->summary.effective_output_tokens == 0
                                           ? 0U
                                           : base->summary.effective_output_tokens - 1U);
    base->text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    if (speculative_backend == SpeculativeBackend::Mtp) {
        const std::uint32_t mtp_tokens    = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(reserved_context_tokens) + draft_window - 1ULL));
        base->backend_kv_page_entitlement = pages_for_tokens(mtp_tokens);
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        base->backend_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    }
    base->summary.admission = runtime::AdmissionResources{
        .active_lanes     = 1,
        .main_kv_pages    = base->text_kv_page_entitlement,
        .backend_kv_pages = base->backend_kv_page_entitlement,
    };
    if (prompt.has_media()) {
        auto control =
            std::make_shared<qwen3_6::VisionControl>(qwen3_6::build_vision_control(prompt));
        std::size_t max_merged     = 0;
        std::uint32_t previous_end = 0;
        for (const qwen3_6::VisionItemControl& item : control->items) {
            if (item.scatter_indices.empty()) {
                throw std::invalid_argument("vision item has no Text consumer columns");
            }
            const auto first = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last  = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin =
                speculative_backend == SpeculativeBackend::Mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end = last + 1;
            if (begin < previous_end) {
                throw std::invalid_argument("vision item consumer spans overlap");
            }
            if (end > base->summary.prompt_tokens) {
                throw std::invalid_argument("vision item consumer span exceeds prompt");
            }
            if (schedule::VisionContext::workspace_bytes(item) > work.capacity()) {
                throw std::invalid_argument("vision item exceeds the Program workspace envelope");
            }
            previous_end = end;
            max_merged   = std::max(max_merged, item.merged_count);
        }
        base->vision_transient_bytes = schedule::VisionContext::output_transient_bytes(max_merged);
        base->vision_control         = std::move(control);
    }

    if (prompt.identity.rewrite_checkpoint) {
        const RewriteCheckpointSpec candidate = *prompt.identity.rewrite_checkpoint;
        if (candidate.frontier == 0 || candidate.frontier > base->summary.prompt_tokens) {
            throw std::invalid_argument(
                "rewrite checkpoint frontier must lie at or inside the prompt frontier");
        }
        base->rewrite_checkpoint = candidate;
    }
    const std::size_t cold_prefill_splits =
        (base->vision_control != nullptr ? base->vision_control->items.size() : 0ULL) +
        (base->rewrite_checkpoint &&
                 base->rewrite_checkpoint->frontier < base->summary.prompt_tokens
             ? 1ULL
             : 0ULL);
    base->summary.service_work_quanta =
        projected_service_work(base->summary, 0, prefill_chunk, cold_prefill_splits);
    return RequestBasePlan(std::move(base));
}

RequestPlan ProgramImplCore::plan_request_for_lane(std::uint32_t lane,
                                                   const PreparedPromptData& prompt,
                                                   const RequestBasePlan& base_plan) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    const RequestControl& request = requests[lane];
    const SequenceState& sequence = sequences[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("cannot plan a request while Program is active or pending");
    }
    if (base_plan.impl_ == nullptr) { throw std::logic_error("request base plan is empty"); }
    const RequestBasePlanImpl& base = *base_plan.impl_;

    auto plan                         = std::make_unique<RequestPlanImpl>();
    plan->summary                     = base.summary;
    plan->sampling                    = base.sampling;
    plan->text_kv_page_entitlement    = base.text_kv_page_entitlement;
    plan->backend_kv_page_entitlement = base.backend_kv_page_entitlement;

    const bool prefix_match_at_frontier =
        sequence.execution_frontier != 0 &&
        qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                        sequence.execution_frontier);
    if (reuse_debug_enabled()) {
        std::fprintf(stderr,
                     "[reuse-debug] lane=%u prompt=%u exec_frontier=%u ledger_size=%zu "
                     "allow_prefix_reuse=%d identity_reusable=%d retained=%d "
                     "prefix_match_at_frontier=%d tail_hidden_valid=%d mtp_kv_valid=%u\n",
                     lane, plan->summary.prompt_tokens, sequence.execution_frontier,
                     sequence.ledger.size(), base.allow_prefix_reuse ? 1 : 0,
                     prompt.identity.reusable ? 1 : 0, sequence.retained ? 1 : 0,
                     prefix_match_at_frontier ? 1 : 0, sequence.tail_hidden_valid ? 1 : 0,
                     sequence.mtp_kv_valid);
        if (!prefix_match_at_frontier) {
            const std::size_t frontier = sequence.execution_frontier;
            const std::size_t n = std::min<std::size_t>(prompt.token_ids.size(), frontier);
            std::size_t first_mismatch = n;
            for (std::size_t i = 0; i < n; ++i) {
                if (prompt.token_ids[i] != sequence.ledger[i]) { first_mismatch = i; break; }
            }
            const bool token_ids_match = first_mismatch == n;
            if (token_ids_match) {
                std::fprintf(stderr, "[reuse-debug]   tokens MATCH for %zu; identity mismatch "
                                     "(token_type/positions/vision)\n", n);
            } else {
                const std::size_t lo = first_mismatch > 4 ? first_mismatch - 4 : 0;
                std::fprintf(stderr, "[reuse-debug]   FIRST TOKEN MISMATCH at i=%zu (frontier=%zu)\n",
                             first_mismatch, frontier);
                for (std::size_t i = lo; i < first_mismatch + 4 && i < n; ++i) {
                    std::fprintf(stderr, "[reuse-debug]   i=%zu prompt=%lld ledger=%lld %s\n", i,
                                 (long long)prompt.token_ids[i], (long long)sequence.ledger[i],
                                 i == first_mismatch ? "<--" : "");
                }
            }
        }
        std::fflush(stderr);
    }

    if (base.allow_prefix_reuse && prompt.identity.reusable && sequence.retained) {
        const bool dflash_append_ready =
            speculative_backend != SpeculativeBackend::DFlash ||
            sequence.dflash_context_frontier == sequence.execution_frontier;
        if (sequence.execution_frontier != 0 && dflash_append_ready && prefix_match_at_frontier) {
            plan->reuse      = ReusePath::AppendAtFrontier;
            plan->reuse_base = sequence.execution_frontier;
        } else {
            const bool ckpt_valid     = sequence.rewrite_checkpoint.valid;
            const bool ckpt_frontier  = sequence.rewrite_checkpoint.frontier != 0;
            const bool ckpt_in_range  = sequence.rewrite_checkpoint.frontier <= prompt.token_ids.size();
            const bool ckpt_prefix_ok = ckpt_valid && ckpt_frontier && ckpt_in_range &&
                qwen3_6::detail::prefix_matches(prompt, sequence.ledger,
                                                sequence.prefix_identity,
                                                sequence.rewrite_checkpoint.frontier);
            if (reuse_debug_enabled()) {
                std::fprintf(stderr,
                             "[reuse-debug] ckpt: valid=%d frontier=%u kind=%d in_range=%d "
                             "prefix_ok=%d prompt_size=%zu ledger_size=%zu\n",
                             ckpt_valid ? 1 : 0, sequence.rewrite_checkpoint.frontier,
                             (int)sequence.rewrite_checkpoint.kind, ckpt_in_range ? 1 : 0,
                             ckpt_prefix_ok ? 1 : 0, prompt.token_ids.size(),
                             sequence.ledger.size());
                std::fflush(stderr);
            }
            if (ckpt_prefix_ok) {
                plan->reuse      = restore_path(sequence.rewrite_checkpoint.kind);
                plan->reuse_base = sequence.rewrite_checkpoint.frontier;
            }
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const bool append_ready =
            plan->reuse == ReusePath::AppendAtFrontier && sequence.tail_hidden_valid &&
            decoder->mtp_cache() != nullptr &&
            (plan->reuse_base == 0 || sequence.mtp_kv_valid >= plan->reuse_base - 1);
        const bool checkpoint_ready = is_rewrite_checkpoint_restore(plan->reuse) &&
                                      decoder->mtp_cache() != nullptr && plan->reuse_base != 0 &&
                                      sequence.mtp_kv_valid >= plan->reuse_base - 1;
        if (plan->reuse != ReusePath::FullReset && !append_ready && !checkpoint_ready) {
            plan->reuse      = ReusePath::FullReset;
            plan->reuse_base = 0;
        }
    }

    // Zero-suffix reuse has no tensor-parallel implementation. When the reusable prefix already
    // covers the whole prompt there is nothing to prefill, so the bonus token would be sampled
    // from a restored hidden state through the output head -- and that head is vocabulary-SPLIT at
    // tp2, so it needs both ranks' logits and a gather that this path has no place to run.
    //
    // Downgrade to a full reset rather than rejecting the request. The answer is identical (the
    // prompt is simply prefilled again instead of resumed); only the reuse saving is lost. The
    // alternative -- throwing from inside prefill execution -- is much worse than it sounds: an
    // exception there does not merely fail one request, it takes the executor down and leaves the
    // whole engine unusable, so a single client turn with prefix reuse enabled would brick a
    // served process. Deciding it HERE, while planning, keeps the failure mode out of execution
    // entirely. `sample_from_hidden`'s own tp2 throw stays as an unreachable backstop.
    if (tp != 1 && plan->reuse != ReusePath::FullReset &&
        plan->reuse_base >= plan->summary.prompt_tokens) {
        plan->reuse      = ReusePath::FullReset;
        plan->reuse_base = 0;
    }

    // MTP prefix reuse at tp2 runs the MTP BRIDGE, which drives the MTP head from the retained
    // target hidden of the reused frontier. Rank 1's MTP stem needs its OWN copy of that hidden
    // -- the stem's row-parallel fc contracts the normalized-HIDDEN half on device 1 -- so the
    // bridge consumes rank 0's per-lane store and the PeerRuntime mirror of it (kept in sync by
    // copy_tail and the MTP decode round's accepted-hidden scatter). The bridge's MTP forward
    // appends AND reads each rank's own MTP KV shard at the reuse frontier position (position-
    // addressed, so it overwrites the slot the original request already wrote -- idempotent),
    // which advances both ranks' MTP frontier in lockstep.
    //
    // The APPEND-AT-FRONTIER path is fully supported at tp2 and no longer downgraded.
    // The REWRITE-CHECKPOINT restores are also supported at tp2 for the non-zero-suffix case:
    // the rewrite-checkpoint hidden is only consumed by copy_tail when base == prompt_tokens
    // (zero-suffix), which the zero-suffix downgrade above already catches. For the non-zero-
    // suffix case the hidden is recomputed by the suffix prefill, so no peer mirror is needed.
    // The GDN state is restored on both ranks via copy_slot (see program_impl.h restore path).

    // Multimodal prefix reuse has no tensor-parallel implementation either, and for two reasons.
    // (1) The vision PLANNER does not split the item control across the reuse boundary: every
    // suffix use-span is required to map to an item the tp2 session can encode, and a reused
    // prefix would need the pre-frontier part of that metadata reconstructed per rank. (2) A text
    // suffix appended after a vision-carrying prefix inherits that prompt's nonzero rope_delta,
    // and the tp2 text path ropes plain suffixes with the un-offset positions -- so the reuse
    // would silently produce wrong MRoPE angles rather than a loud failure. Full re-prefill gives
    // the identical answer (the tp2 multimodal prefill is bit-faithful to tp1), only the reuse
    // saving is lost. Decided HERE, not in execution, so a single vision turn with prefix reuse
    // enabled cannot take the executor down; the executor-side tp2 throws stay as unreachable
    // backstops.
    if (tp != 1 && base.vision_control != nullptr && plan->reuse != ReusePath::FullReset) {
        plan->reuse      = ReusePath::FullReset;
        plan->reuse_base = 0;
    }

    if (is_rewrite_checkpoint_restore(plan->reuse) &&
        speculative_backend == SpeculativeBackend::DFlash &&
        (!dflash || !sequence.kv || !sequence.kv->backend ||
         sequence.dflash_context_frontier < plan->reuse_base)) {
        plan->reuse      = ReusePath::FullReset;
        plan->reuse_base = 0;
    }

    const std::optional<RewriteCheckpointSpec>& desired = base.rewrite_checkpoint;
    const bool existing_checkpoint_matches =
        desired && plan->reuse != ReusePath::FullReset && sequence.rewrite_checkpoint.valid &&
        sequence.rewrite_checkpoint.frontier == desired->frontier &&
        qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                        desired->frontier);
    if (!desired) {
        plan->rewrite_checkpoint_action = RewriteCheckpointAction::Drop;
    } else if (existing_checkpoint_matches) {
        plan->rewrite_checkpoint_action = sequence.rewrite_checkpoint.kind == desired->kind
                                              ? RewriteCheckpointAction::KeepExisting
                                              : RewriteCheckpointAction::ReclassifyExisting;
    } else if (desired->frontier > plan->reuse_base) {
        plan->rewrite_checkpoint_action  = RewriteCheckpointAction::CaptureNew;
        plan->rewrite_checkpoint_capture = desired;
    } else {
        // The selected continuation state is already past the desired boundary. It remains a
        // valid hit; do not replay an otherwise reusable prefix merely to materialize an older
        // auxiliary snapshot. A later request can still use the checkpoint currently retained.
        plan->rewrite_checkpoint_action = RewriteCheckpointAction::DeferCapture;
    }

    plan->summary.reusable_prompt_tokens = plan->reuse_base;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        if (plan->reuse == ReusePath::FullReset) {
            plan->prepare_mtp = true;
        } else if (plan->reuse == ReusePath::AppendAtFrontier) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        } else if (is_rewrite_checkpoint_restore(plan->reuse)) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        }
    }

    if (base.vision_control != nullptr) {
        VisionPrefillPlan vision;
        vision.control = base.vision_control;
        vision.uses.reserve(base.vision_control->items.size());
        for (std::size_t index = 0; index < base.vision_control->items.size(); ++index) {
            const qwen3_6::VisionItemControl& item = base.vision_control->items[index];
            const auto first          = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last           = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin = plan->prepare_mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end   = last + 1;
            if (end <= plan->reuse_base) { continue; }
            vision.uses.push_back(VisionUseSpan{begin, end, static_cast<std::uint32_t>(index)});
        }
        if (!vision.uses.empty()) {
            plan->summary.transient_alignment = 256;
            plan->summary.transient_bytes     = base.vision_transient_bytes;
            plan->vision                      = std::move(vision);
        }
    }

    const std::size_t prefill_splits =
        (plan->vision ? plan->vision->uses.size() : 0ULL) +
        (plan->rewrite_checkpoint_capture &&
                 plan->rewrite_checkpoint_capture->frontier < plan->summary.prompt_tokens
             ? 1ULL
             : 0ULL);
    plan->summary.service_work_quanta =
        projected_service_work(plan->summary, plan->reuse_base, prefill_chunk, prefill_splits);
    if (reuse_debug_enabled()) {
        std::fprintf(stderr,
                     "[reuse-debug] lane=%u FINAL reuse=%s reuse_base=%u prompt=%u "
                     "prefix_match_at_frontier=%d tail_hidden_valid=%d mtp_kv_valid=%u\n",
                     lane, reuse_path_name(plan->reuse), plan->reuse_base,
                     plan->summary.prompt_tokens, prefix_match_at_frontier ? 1 : 0,
                     sequence.tail_hidden_valid ? 1 : 0, sequence.mtp_kv_valid);
        std::fflush(stderr);
    }
    return RequestPlan(std::move(plan));
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
