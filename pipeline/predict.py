"""Which GPU class holds a model at the sweep's bench load, from vLLM's own boot numbers.

    python3 pipeline/predict.py --weights-gib 15.27 --kv-bytes-per-token 147470 --name Qwen/Qwen3-8B
    python3 pipeline/predict.py --log data/smoke/logs/day0-flint-b779895_h200_flashinfer.txt
    python3 pipeline/predict.py --record data/sweep/<image tag>/S0.json
    python3 pipeline/predict.py --config config.json --weights-gib 15.27     # estimate, standard attention only

Two numbers decide fit, and vLLM prints both at every boot:
  weights   "Model loading took X GiB"
  KV/token  "Available KV cache memory: X GiB" / "KV cache size: N tokens"
They don't depend on the GPU: Qwen3-8B gave 147,470 B/token on an L40S and 147,480 on
an RTX PRO 6000 Blackwell MIG slice (pilot, Oct 28). So one boot anywhere (a smoke test)
predicts every class:

  KV tokens = (device memory x 0.92 - weights - overhead) / KV bytes per token

Overhead (activations, CUDA graphs, non-torch memory) measured 2.44 GiB for Qwen3-8B on
an L40S and 2.80 GiB for flint (74 GiB of weights) on an H200; 3 GiB is used.

For hybrid models (linear attention or Mamba layers next to attention) part of the cache is a
fixed state per sequence, not per token, so the per-token figure from a boot is approximate
away from the concurrency it was measured at.

A config.json estimate (2 x layers x KV heads x head dim x dtype bytes) is only offered
for plain full attention. MLA, hybrid and sliding-window models are "unknown": their
configs vary by architecture, which is where rules stop working on day 0.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sizing  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GIB = 2 ** 30
GMU = 0.92               # vLLM's default gpu_memory_utilization
OVERHEAD_GIB = 3.0
CONCURRENCY, BENCH_TOKENS = 30, 4096 + 512
KV_FLOOR = CONCURRENCY * BENCH_TOKENS   # 138,240, the sweep's fit check
SMOKE_LEN = 4096                        # the smoke test's MAX_MODEL_LEN
# Device memory as nvidia-smi reports it (MiB), for the type each class usually gets.
DEVICE_MIB = {"24gb": ("NVIDIA GeForce RTX 4090", 24564), "48gb": ("NVIDIA L40S", 46068),
              "80gb": ("NVIDIA H100 80GB HBM3", 81559), "141gb": ("NVIDIA H200", 143771),
              "180gb": ("NVIDIA B200", 183359)}
# vLLM's server default for max_num_batched_tokens on that device (arg_utils, v0.32.0).
BATCH_DEFAULT = {"24gb": 2048, "48gb": 2048, "80gb": 8192, "141gb": 8192, "180gb": 16384}
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float8_e4m3fn": 1}

_LOAD = re.compile(r"Model loading took ([\d.]+) GiB")
_KV_GIB = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
_KV_TOK = re.compile(r"KV cache size: ([\d,]+) tokens")


def from_numbers(weights_gib, kv_gib, kv_tokens):
    return {"weights_gib": weights_gib, "kv_bytes_per_token": round(kv_gib * GIB / kv_tokens), "source": "measured"}


def from_log(text):
    """Measured numbers from a worker or vLLM log; None if a line is missing."""
    load, kv, tok = _LOAD.findall(text), _KV_GIB.findall(text), _KV_TOK.findall(text)
    if not (load and kv and tok):
        return None
    return from_numbers(float(load[-1]), float(kv[-1]), int(tok[-1].replace(",", "")))


def from_record(rec):
    """Measured numbers from a sweep config record (any run's boot report that has them)."""
    for run in rec.get("runs", []):
        v = (run.get("boot") or {}).get("vllm") or {}
        if v.get("model_load_gib") and v.get("kv_cache_gib") and v.get("kv_cache_tokens"):
            return from_numbers(v["model_load_gib"], v["kv_cache_gib"], v["kv_cache_tokens"])
    return None


def from_config(cfg, weights_gib):
    """Estimate KV bytes/token from config.json for plain full attention; None when it can't."""
    c = cfg.get("text_config") or cfg
    special = [k for k in ("kv_lora_rank", "layer_types", "layers_block_type", "full_attention_interval",
                           "sliding_window_pattern", "attn_layer_indices") if k in c]
    if special or (c.get("sliding_window") and c.get("use_sliding_window", True) is not False):
        return None  # MLA, hybrid or sliding-window: KV per token isn't 2 x layers x heads x dim
    try:
        layers, heads = int(c["num_hidden_layers"]), int(c["num_attention_heads"])
        kv_heads = int(c.get("num_key_value_heads") or heads)
        head_dim = int(c.get("head_dim") or int(c["hidden_size"]) // heads)
    except (KeyError, TypeError, ValueError):
        return None
    dtype = str(c.get("torch_dtype") or cfg.get("torch_dtype") or c.get("dtype") or "bfloat16")
    nbytes = DTYPE_BYTES.get(dtype, 2)
    return {"weights_gib": weights_gib, "kv_bytes_per_token": 2 * layers * kv_heads * head_dim * nbytes,
            "source": "config estimate"}


def per_class(m, gmu=GMU, overhead_gib=OVERHEAD_GIB):
    """One row per GPU class: KV tokens, smoke and bench fit, context at 30 requests, $/h."""
    rows = []
    for cls, (device, mib) in DEVICE_MIB.items():
        kv_gib = mib / 1024 * gmu - m["weights_gib"] - overhead_gib
        tokens = int(kv_gib * GIB / m["kv_bytes_per_token"]) if kv_gib > 0 else 0
        rows.append({"gpu_class": cls, "device": device, "kv_gib": round(max(kv_gib, 0), 2), "kv_tokens": tokens,
                     "smoke_fits": tokens >= SMOKE_LEN, "bench_fits": tokens >= KV_FLOOR,
                     "context_at_30": tokens // CONCURRENCY, "usd_per_hr": sizing.PRICE_PER_HR[cls],
                     "batch_default": BATCH_DEFAULT[cls]})
    return rows


def predict(m, name=None):
    rows = per_class(m)
    bench = [r for r in rows if r["bench_fits"]]
    smoke = [r for r in rows if r["smoke_fits"]]
    pick = min(bench, key=lambda r: r["usd_per_hr"]) if bench else None
    configs = None
    if pick:
        configs = ["S0", "S1", "S2", "S3", "S4"] + (["S5"] if pick["batch_default"] < 8192 else []) + ["W"]
    return {"model": name, **m, "classes": rows,
            "smoke_class": smoke[0]["gpu_class"] if smoke else None,
            "sweep_class": pick["gpu_class"] if pick else None,
            "sweep_configs": configs,
            "note": None if pick else "doesn't hold 30 bench requests on one GPU: multi-GPU, or a smaller context",
            "neighbors": neighbors(m)}


def neighbors(m, data=None, tol=0.02):
    """Swept models with the same shape: weights and KV bytes/token within 2% (fine-tunes of one base)."""
    out = []
    for f in sorted(((data or ROOT / "data") / "sweep").glob("*/S0.json")):
        rec = json.loads(f.read_text())
        other = from_record(rec)
        if not other:
            continue
        if all(abs(other[k] - m[k]) <= tol * m[k] for k in ("weights_gib", "kv_bytes_per_token")):
            summary = f.parent / "summary.json"
            s = json.loads(summary.read_text()) if summary.exists() else {}
            out.append({"sweep": str(f.parent.relative_to(f.parents[2])), "model": s.get("model"),
                        "device": s.get("device"), "verdict": s.get("verdict"),
                        "recommended": s.get("recommended"), "best_so_far": s.get("best_so_far")})
    return out


def show(p):
    print(f"{p['model'] or 'model'}: {p['weights_gib']} GiB of weights, {p['kv_bytes_per_token']:,} KV bytes/token "
          f"({p['source']})")
    print(f"{'class':6} {'device':24} {'KV GiB':>7} {'KV tokens':>10} {'ctx@30':>7} {'smoke':>5} {'bench':>5} {'$/h':>5}")
    for r in p["classes"]:
        print(f"{r['gpu_class']:6} {r['device']:24} {r['kv_gib']:7.1f} {r['kv_tokens']:10,} {r['context_at_30']:7,} "
              f"{'yes' if r['smoke_fits'] else 'no':>5} {'yes' if r['bench_fits'] else 'no':>5} {r['usd_per_hr']:5.2f}")
    print(f"smoke on {p['smoke_class']}; sweep on {p['sweep_class']} with {p['sweep_configs']}" if p["sweep_class"]
          else f"smoke on {p['smoke_class']}; {p['note']}")
    for n in p["neighbors"]:
        print(f"same shape as {n['model']} ({n['sweep']}): {n['verdict']}, "
              f"recommended {n['recommended']}, best so far {n['best_so_far']}, on {n['device']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--weights-gib", type=float)
    ap.add_argument("--kv-bytes-per-token", type=int)
    ap.add_argument("--log", help="a worker or vLLM log with the boot lines")
    ap.add_argument("--record", help="a sweep config record (data/sweep/<tag>/<config>.json)")
    ap.add_argument("--config", help="config.json, for an estimate (needs --weights-gib)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.log:
        m = from_log(Path(a.log).read_text(errors="replace"))
    elif a.record:
        m = from_record(json.loads(Path(a.record).read_text()))
    elif a.config:
        if a.weights_gib is None:
            sys.exit("--config needs --weights-gib")
        m = from_config(json.loads(Path(a.config).read_text()), a.weights_gib)
        if m is None:
            sys.exit("this config isn't plain full attention (MLA, hybrid or sliding window): "
                     "boot it once and pass --log instead")
    elif a.weights_gib and a.kv_bytes_per_token:
        m = {"weights_gib": a.weights_gib, "kv_bytes_per_token": a.kv_bytes_per_token, "source": "given"}
    else:
        sys.exit("give --log, --record, --config + --weights-gib, or --weights-gib + --kv-bytes-per-token")
    if m is None:
        sys.exit("no weights / KV lines found")
    p = predict(m, a.name)
    print(json.dumps(p, indent=1)) if a.json else show(p)


if __name__ == "__main__":
    main()
