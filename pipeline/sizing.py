"""Pick the smallest single-GPU class that should hold a model, from its weight size.

Weight bytes come from the Hugging Face API: the safetensors parameter count per
dtype when the repo has it, otherwise the sizes of its weight files. A class
fits when the weights take at most HEADROOM of its VRAM; the rest is KV cache
at MAX_MODEL_LEN=4096 plus activations. Checked against the smoke runs so far:
Qwen3-8B (16.4 GB) and K2-Horizon-7B passed on 24 GB, and flint (79.6 GB) needs
a 141 GB card at least.
"""

import json
import math
import urllib.parse
import urllib.request

VRAM_GB = {"24gb": 24, "48gb": 48, "80gb": 80, "141gb": 141, "180gb": 180}
# Highest listed Serverless rate among the GPU types in each class (runpod.io/pricing,
# 2026-10-21), so cost estimates are upper bounds.
PRICE_PER_HR = {"24gb": 1.10, "48gb": 1.75, "80gb": 4.79, "141gb": 5.93, "180gb": 8.64}
HEADROOM = 0.80
DTYPE_BYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "U32": 4, "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
               "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BOOL": 1, "F4": 0.5, "F6_E2M3": 0.75,
               "F6_E3M2": 0.75}
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
UA = "day0-ready/0.1 (+https://github.com/ashwinsreedhar28/day0-ready)"


def hf_model_info(model):
    url = f"https://huggingface.co/api/models/{urllib.parse.quote(model, safe='/')}?blobs=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
        return json.loads(r.read())


def weights_gb(info):
    """Weight size in GB (1e9 bytes), or None if the API says nothing usable."""
    params = (info.get("safetensors") or {}).get("parameters") or {}
    if params:
        return sum(n * DTYPE_BYTES.get(dtype.upper(), 2) for dtype, n in params.items()) / 1e9
    sizes = [s.get("size") or 0 for s in info.get("siblings") or []
             if s.get("rfilename", "").endswith(WEIGHT_SUFFIXES) and "/" not in s.get("rfilename", "")]
    return sum(sizes) / 1e9 if any(sizes) else None


def pick_class(gb):
    """Smallest class whose VRAM holds the weights with headroom, or None (multi-GPU)."""
    for cls, vram in VRAM_GB.items():
        if gb <= HEADROOM * vram:
            return cls
    return None


def disk_gb(gb):
    """Container disk: weights land there without a network volume, plus the ~20 GB image."""
    return max(60, math.ceil(gb * 1.15 + 25))


def cost_usd(minutes, gpu_class):
    return round(minutes / 60 * PRICE_PER_HR[gpu_class], 2)
