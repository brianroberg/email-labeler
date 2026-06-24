#!/usr/bin/env python3
"""Smoke-test mlx_lm.server continuous batching: N concurrent vs 1 request.

Run on the machine serving the model (or another tailnet box pointed at it).
With continuous batching working, N concurrent requests finish in roughly the
wall time of a single request; ~1x throughput gain means requests are
serializing. Dependency-free (stdlib only).
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Mirrors the local model's real job: Stage 2 label classification of a person
# email body (NEEDS_RESPONSE / FYI / LOW_PRIORITY). The person-vs-service call
# is the cloud model's job, not the local model's.
PROMPT = ("From: jane@example.com\nSubject: Re: budget review\n\n"
          "Hi — can you look over the Q3 numbers I sent and tell me if the travel "
          "line looks right before I submit it Friday? Thanks, Jane\n\n"
          "Classify this email as exactly one of: NEEDS_RESPONSE, FYI, LOW_PRIORITY.")


def post(url, max_tokens):
    payload = {"messages": [{"role": "user", "content": PROMPT}],
               "temperature": 0, "max_tokens": max_tokens}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read())
    return time.monotonic() - t0, body


def extract(body):
    """Best-effort text from an OpenAI-style response; never raises."""
    try:
        msg = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return f"<no choices: {json.dumps(body)[:80]}>"
    text = msg.get("content") or msg.get("reasoning_content") or ""
    return text.strip()[:30] or f"<no text; message keys={list(msg)}>"


def first_message(body):
    """Best-effort message object from an OpenAI-style response; never raises.

    Unlike body["choices"][0], this tolerates a present-but-empty choices array
    (the `[{}]` default of dict.get only applies when the key is absent), falling
    back to the whole body for display.
    """
    choices = body.get("choices") or [{}]
    first = choices[0] if isinstance(choices, list) and choices else {}
    return first.get("message", body) if isinstance(first, dict) else body


def call(url, max_tokens, i):
    dt, body = post(url, max_tokens)
    return i, dt, extract(body)


def main():
    p = argparse.ArgumentParser(description="Smoke-test mlx_lm.server continuous batching.")
    p.add_argument("-n", "--concurrency", type=int, default=4,
                   help="number of concurrent requests (default: 4)")
    p.add_argument("--url", default="http://127.0.0.1:8080",
                   help="server base URL (default: http://127.0.0.1:8080)")
    p.add_argument("--max-tokens", type=int, default=8,
                   help="max output tokens per request (default: 8)")
    args = p.parse_args()
    n = args.concurrency
    url = args.url.rstrip("/") + "/v1/chat/completions"

    print(f"Endpoint {url}  N={n}  max_tokens={args.max_tokens}")
    _, warm = post(url, args.max_tokens)        # warmup: compiles graph / loads model
    sample = first_message(warm)
    print(f"sample response message: {json.dumps(sample)[:200]}")
    base, _ = post(url, args.max_tokens)        # single-request baseline
    print(f"single request: {base:.2f}s")

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(lambda i: call(url, args.max_tokens, i), range(n)))
    wall = time.monotonic() - t0
    for i, dt, out in res:
        print(f"  req {i}: {dt:5.2f}s -> {out!r}")
    print(f"\n{n} concurrent wall: {wall:.2f}s   (serialized would be ~{base * n:.2f}s)")
    print(f"throughput gain: {base * n / wall:.1f}x   (~1x = NOT batching, ~{n}x = fully batched)")


if __name__ == "__main__":
    main()
