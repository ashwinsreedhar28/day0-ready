"""Throughput bench against a Serverless endpoint's OpenAI-compatible route (stdlib only).

Every request sends a fixed list of token ids as the prompt (so its length is
exact without a tokenizer), asks for max_tokens with ignore_eos, and streams.
Times are taken on the client, so they include Runpod's gateway:

    TTFT  first streamed text after the request was sent
    TPOT  (last text - first text) / (output tokens - 1)

Output tokens come from the final usage chunk (stream_options.include_usage).
If the gateway delivers a stream in a few large pieces instead of token by
token, TTFT and TPOT mean nothing; `chunks_per_token` in the summary shows it.
"""

import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = "day0-ready/0.1 (+https://github.com/ashwinsreedhar28/day0-ready)"


def prompts(n, length, vocab_size, seed=0):
    """n distinct prompts of `length` token ids, the same for a given seed.

    Ids come from [1000, vocab_size - 1000): most tokenizers keep special and
    control tokens at the two ends of the vocabulary.
    """
    if vocab_size < 4000:
        raise ValueError(f"vocab_size {vocab_size} is too small for this sampler")
    rng = random.Random(seed)
    return [[rng.randrange(1000, vocab_size - 1000) for _ in range(length)] for _ in range(n)]


def body(model, prompt, max_tokens):
    return {"model": model, "prompt": prompt, "max_tokens": max_tokens, "ignore_eos": True,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}


def stream_one(url, key, payload, timeout=900):
    """Send one streamed completion; returns timings and token counts (never raises)."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "Accept": "text/event-stream", "User-Agent": UA})
    t0 = time.monotonic()
    text_times, usage, error = [], None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    error = str(event["error"])[:300]
                    break
                if event.get("usage"):
                    usage = event["usage"]
                if any(c.get("text") for c in event.get("choices") or []):
                    text_times.append(time.monotonic())
    except urllib.error.HTTPError as e:
        error = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
    except Exception as e:  # noqa: BLE001  (a cut stream raises http.client.IncompleteRead; one request must not end the run)
        error = f"{type(e).__name__}: {e}"[:300]
    end = time.monotonic()
    out_tokens = (usage or {}).get("completion_tokens")
    if out_tokens is None:
        out_tokens = len(text_times)
    rec = {"ok": error is None and out_tokens == payload["max_tokens"], "error": error,
           "prompt_tokens": (usage or {}).get("prompt_tokens"), "out_tokens": out_tokens,
           "chunks": len(text_times), "e2e_s": round(end - t0, 3), "ttft_s": None, "tpot_s": None}
    if error is None and out_tokens != payload["max_tokens"]:
        rec["error"] = f"got {out_tokens} output tokens, asked for {payload['max_tokens']}"
    if text_times:
        rec["ttft_s"] = round(text_times[0] - t0, 4)
        if out_tokens and out_tokens > 1:
            rec["tpot_s"] = round((text_times[-1] - text_times[0]) / (out_tokens - 1), 5)
    return rec


def pct(values, q):
    """Nearest-rank percentile (q in 0..100); None for an empty list."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[max(0, math.ceil(q / 100 * len(vals)) - 1)]


def run_level(url, key, model, prompt_list, concurrency, max_tokens, timeout=900, send=None):
    """All prompts, at most `concurrency` in flight. Returns (summary, per-request records)."""
    send = send or stream_one
    payloads = [body(model, p, max_tokens) for p in prompt_list]
    t0 = time.monotonic()
    ex = ThreadPoolExecutor(max_workers=concurrency)
    try:
        recs = list(ex.map(lambda pl: send(url, key, pl, timeout), payloads))
    finally:
        # On an interrupt, drop the queued requests instead of sending them all first, so the
        # caller's scale-down runs now (requests already in flight finish on their own).
        ex.shutdown(wait=False, cancel_futures=True)
    wall = time.monotonic() - t0
    good = [r for r in recs if r["ok"]]
    out = sum(r["out_tokens"] for r in good)
    ms = lambda v: None if v is None else round(v * 1000, 1)  # noqa: E731
    summary = {
        "concurrency": concurrency, "requests": len(recs), "failed": len(recs) - len(good),
        "wall_s": round(wall, 2), "output_tokens": out,
        "output_tok_s": round(out / wall, 1) if wall > 0 else None,
        "ttft_p50_ms": ms(pct([r["ttft_s"] for r in good], 50)),
        "ttft_p99_ms": ms(pct([r["ttft_s"] for r in good], 99)),
        "tpot_p50_ms": ms(pct([r["tpot_s"] for r in good], 50)),
        "e2e_p50_s": pct([r["e2e_s"] for r in good], 50),
        "chunks_per_token": round(statistics.median(r["chunks"] / r["out_tokens"] for r in good), 3) if good else None,
        "errors": sorted({r["error"] for r in recs if r["error"]})[:5],
    }
    return summary, recs
