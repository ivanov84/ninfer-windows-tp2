#!/usr/bin/env python3
"""
Prefix-reuse probe v2 for ninfer-serve (TP2 + MTP), JSONL-log based.

Why v2:
  * The stdout log is block-buffered on Windows when redirected to a file, so
    `[req N] done ...` lines are not visible while the server runs. The
    structured request log (--request-log-jsonl) flushes after EVERY record,
    so we parse that instead. Fields per request_done record:
      result.prefix_cache_hit_tokens, result.prefix_reuse_path,
      result.prompt_tokens, result.computed_prefill_tokens,
      timings_seconds.{ttft,prefill,decode,total}, speculative.*
  * An IDENTICAL resend of a finished request is a ZERO-SUFFIX reuse, which
    tp2 intentionally downgrades to full_reset (the bonus token is sampled
    through the vocab-split head, which has no tp2 path). So the expected
    append_frontier case is a GROWING CONVERSATION: each request = previous
    request's full prompt + previous answer + a new user turn. That is exactly
    the history the fix enables (AppendAtFrontier with MTP bridge).
  * The server must be started with --no-thinking so content is not empty
    (a 64-token budget spent on reasoning_content hides the echoed fact).

Request flow (single lane, max-concurrency 1):
  R1: [system, user Q1]                          -> expect full_reset,  cache=0
  R2: [system, user Q1, asst A1, user Q2]        -> expect append_frontier, cache>0
  R3: R2 + [asst A2, user Q3]                    -> append_frontier, cache >= R2.cache
  R4: R3 + [asst A3, user Q4] + tools + prompt_cache_key (OpenHands-style)
                                          -> append_frontier (if schema accepts
                                             the extras; 400 -> retry without them
                                             and record a WARN)

Usage:
    python cache_test.py --base http://localhost:8099 --key local --jsonl <path>

Exit 0 = all reuse/cache assertions passed.
"""
import argparse
import json
import sys
import time
import urllib.request

FACT = "The secret code is KX-7741-DELTA."
STABLE_SYSTEM = (
    "You are a precise assistant operating with a fixed toolset. " + FACT + " "
    "Answer the user's question in exactly one short line. "
    "First sentence of this block is a stable system prefix used by many requests.\n"
    + ("Context line: the sky is clear and the lab is quiet. " * 60)
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        },
    }
]

QUESTIONS = [
    "What is the secret code?",
    "What is the secret code, repeated twice?",
    "Is the sky clear, and what is the secret code?",
    "Give the secret code, then state whether the lab is quiet.",
]


def read_done_rows(path: str):
    """All request_done records (JSONL), in file order."""
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "request_done":
                    rows.append(rec)
    except OSError as exc:
        print(f"  (could not read jsonl log {path}: {exc})")
    return rows


def chat(base: str, key: str, messages, gen: int, extra=None, timeout: int = 300):
    body = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": gen,
        "temperature": 0.0,
        "stream": False,
    }
    if extra:
        body.update(extra)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    wall = time.perf_counter() - t0
    obj = json.loads(raw)
    if "choices" not in obj:
        return {
            "status": status,
            "wall_s": wall,
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish": "http_error",
            "content": json.dumps(obj)[:200],
        }
    choice = obj["choices"][0]
    usage = obj.get("usage", {})
    return {
        "status": status,
        "wall_s": wall,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": choice.get("finish_reason", "?"),
        "content": choice["message"]["content"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8099")
    ap.add_argument("--key", default="local")
    ap.add_argument("--jsonl", required=True,
                    help="path to the server's --request-log-jsonl file")
    ap.add_argument("--gen", type=int, default=64)
    args = ap.parse_args()

    rows = []
    # conversation history grows across requests (the point of the test)
    history = [{"role": "system", "content": STABLE_SYSTEM}]

    def run(name, question, expect_reuse, expect_cache_min, extra=None):
        before = read_done_rows(args.jsonl)
        r = chat(args.base, args.key, history, args.gen, extra=extra)
        logrow = None
        deadline = time.time() + 15
        while time.time() < deadline:
            after = read_done_rows(args.jsonl)
            if len(after) > len(before):
                logrow = after[-1]
                break
            time.sleep(0.5)
        rows.append((name, r, logrow, expect_reuse, expect_cache_min))
        print(f"\n=== {name} ===")
        print(f"  client wall={r['wall_s']:.2f}s status={r['status']} finish={r['finish']}")
        print(f"  usage prompt={r['prompt_tokens']} completion={r['completion_tokens']}")
        print(f"  content[:120]={r['content'][:120]!r}")
        if logrow:
            res = logrow.get("result", {})
            tim = logrow.get("timings_seconds", {})
            print(
                f"  LOG  reuse={res.get('prefix_reuse_path')} "
                f"cache={res.get('prefix_cache_hit_tokens')} "
                f"prompt={res.get('prompt_tokens')} computed_prefill={res.get('computed_prefill_tokens')} "
                f"ttft={tim.get('ttft', 0) * 1000:.0f}ms "
                f"prefill_s={tim.get('prefill', 0):.3f} decode_s={tim.get('decode', 0):.3f}"
            )
        else:
            print("  LOG  (no request_done record appeared in jsonl)")

    # ---- R1: cold conversation turn ------------------------------------------
    history.append({"role": "user", "content": QUESTIONS[0]})
    run("R1 cold turn", QUESTIONS[0], "full_reset", 0)

    # ---- R2..R4: growing history (each turn appends the previous answer) -----
    for idx in (1, 2, 3):
        last = rows[-1]
        history.append({"role": "assistant", "content": last[1]["content"]})
        history.append({"role": "user", "content": QUESTIONS[idx]})
        extras = None
        if idx == 3:
            # OpenHands-style extras: real tools + prompt_cache_key.
            extras = {"tools": TOOLS, "prompt_cache_key": "openhands-session-42"}
        run(f"R{idx + 1} growing history" + (" +OH-extras" if idx == 3 else ""),
            QUESTIONS[idx], "append_frontier", 1, extra=extras)

    # ---- verdict ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    ok = True
    for name, r, logrow, expect_reuse, expect_cache_min in rows:
        if logrow is None:
            print(f"  {name}: FAIL (no log record)")
            ok = False
            continue
        res = logrow["result"]
        reuse = res.get("prefix_reuse_path", "?")
        cache = int(res.get("prefix_cache_hit_tokens", 0))
        reuse_ok = reuse == expect_reuse
        cache_ok = cache >= max(expect_cache_min, 0)
        line = (f"  {name}: reuse={reuse} (want {expect_reuse}) "
                f"cache={cache} (want >={expect_cache_min}) -> ")
        line += "PASS" if (reuse_ok and cache_ok) else "FAIL"
        print(line)
        ok = ok and reuse_ok and cache_ok
    for name, r, logrow, _, _ in rows:
        if "KX-7741-DELTA" not in r["content"]:
            print(f"  {name}: WARN model did not echo the secret code -> {r['content'][:80]!r}")
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())