#!/usr/bin/env python3
"""
Minimal prefix-reuse probe for ninfer-serve (TP2 + MTP).

OpenHands-independent: talks straight to http://localhost:8080/v1 (OpenAI-compatible),
sends a large prompt, resends it, sends it with a small addition, then checks the
SERVER LOG for cache=/reuse=/ttft= on each request.

Usage:
    python tmp\\cache_test.py [--base http://localhost:8080] [--key local]
        [--log <path-to-ninfer-stderr.log>] [--ctx N] [--gen N] [--big N]

Exit code 0 = all correctness assertions passed (reuse + cache as expected).
The script prints a per-request table and a PASS/FAIL verdict.
"""
import argparse
import json
import re
import sys
import time
import urllib.request

# ---- prompt material ---------------------------------------------------------
# A stable, self-describing "system+history" block. Every request reuses it verbatim,
# so a correct prefix cache must hit on it. We also embed a fact the model must echo
# so we can sanity-check that generation is not garbage.
FACT = "The secret code is KX-7741-DELTA."
STABLE = (
    "You are a precise assistant. " + FACT + " "
    "Answer the user's question in exactly one short line. "
    "First sentence of this block is a stable system prefix used by many requests.\n"
    # pad to a few hundred tokens of stable context
    + ("Context line: the sky is clear and the lab is quiet. " * 60)
)


def build_prompt(addition: str) -> list:
    return [
        {"role": "system", "content": STABLE},
        {"role": "user", "content": addition},
    ]


def chat(base: str, key: str, messages: list, ctx: int, gen: int, timeout: int = 300):
    body = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": gen,
        "temperature": 0.0,
        "stream": False,
    }
    if ctx:
        body["max_context"] = ctx
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        status = resp.status
    wall = time.perf_counter() - t0
    obj = json.loads(raw)
    usage = obj.get("usage", {})
    content = obj["choices"][0]["message"]["content"]
    finish = obj["choices"][0].get("finish_reason", "?")
    return {
        "status": status,
        "wall_s": wall,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": finish,
        "content": content,
    }


LOG_RE = re.compile(
    r"\[req (?P<id>\d+)\] done finish=(?P<finish>\S+).*?prompt=(?P<prompt>\d+) "
    r"gen=(?P<gen>\d+) cache=(?P<cache>\d+) reuse=(?P<reuse>\S+) "
    r"ttft=(?P<ttft>\d+)ms prefill=(?P<prefill>[\d.]+) decode=(?P<decode>[\d.]+)"
)


def read_log(path: str):
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()
    except OSError as exc:
        print(f"  (could not read server log {path}: {exc})")
        return []


def last_req_entries(lines):
    entries = []
    for line in lines:
        m = LOG_RE.search(line)
        if m:
            entries.append(m.groupdict())
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--key", default="local")
    ap.add_argument("--log", default="", help="path to the ninfer stderr log to parse")
    ap.add_argument("--ctx", type=int, default=0, help="max_context (0 = server default)")
    ap.add_argument("--gen", type=int, default=64, help="max_tokens per request")
    ap.add_argument("--big", type=int, default=0, help="extra stable pad lines (0 = default ~300 tok)")
    args = ap.parse_args()

    global STABLE
    if args.big:
        STABLE = STABLE + ("More stable context. " * args.big)

    rows = []

    def run(name, addition, expect_reuse, expect_cache_min):
        before = last_req_entries(read_log(args.log))
        t0 = time.perf_counter()
        r = chat(args.base, args.key, build_prompt(addition), args.ctx, args.gen)
        wall = time.perf_counter() - t0
        after = last_req_entries(read_log(args.log))
        # the new request is the last entry that is not in `before`
        new = after[len(before):]
        logrow = new[-1] if new else None
        rows.append((name, r, wall, logrow, expect_reuse, expect_cache_min))
        print(f"\n=== {name} ===")
        print(f"  client wall={wall:.2f}s status={r['status']} finish={r['finish']}")
        print(f"  usage prompt={r['prompt_tokens']} completion={r['completion_tokens']}")
        print(f"  content[:120]={r['content'][:120]!r}")
        if logrow:
            print(
                f"  LOG  cache={logrow['cache']} reuse={logrow['reuse']} "
                f"ttft={logrow['ttft']}ms prefill={logrow['prefill']} decode={logrow['decode']}"
            )
        else:
            print("  LOG  (no matching request-done line found in server log)")

    # 1. cold: full prefill
    run("1 cold (full prefill)", "What is the secret code?", "full_reset", 0)
    # 2. identical resend: must reuse the stable prefix
    run("2 identical resend", "What is the secret code?", "append_frontier", 1)
    # 3. small addition: must reuse the stable prefix, compute only the new tail
    run("3 +small addition", "What is the secret code? And is the sky clear?", "append_frontier", 1)
    # 4. different question, same stable prefix: still reuses the prefix
    run("4 different question", "Repeat only the secret code.", "append_frontier", 1)

    # ---- verdict -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    ok = True
    for name, r, wall, logrow, expect_reuse, expect_cache_min in rows:
        if logrow is None:
            print(f"  {name}: FAIL (no server-log line)")
            ok = False
            continue
        reuse_ok = logrow["reuse"] == expect_reuse
        cache_ok = int(logrow["cache"]) >= expect_cache_min
        line = f"  {name}: reuse={logrow['reuse']} (want {expect_reuse}) " \
               f"cache={logrow['cache']} (want >={expect_cache_min}) -> "
        line += "PASS" if (reuse_ok and cache_ok) else "FAIL"
        print(line)
        ok = ok and reuse_ok and cache_ok
    # correctness: the echoed fact must be present in at least the resend answers
    for name, r, wall, logrow, _, _ in rows[1:]:
        if "KX-7741-DELTA" not in r["content"]:
            print(f"  {name}: WARN model did not echo the secret code -> {r['content'][:80]!r}")
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())