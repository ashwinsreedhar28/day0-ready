"""CPU-only readiness checks, run inside a built worker image.

    docker run --rm --entrypoint python3 -v "$PWD/checks:/checks" -v "$PWD/out:/out" \
        IMAGE /checks/cpu_check.py MODEL_ID

No GPU and no weights: it fetches config and tokenizer files only. It catches
the startup failures seen on real endpoints in study/ before any GPU is spent:

  unsupported            vLLM can't resolve the architecture (Qwen3.5 on vLLM 0.16.0)
  transformers_too_old   transformers can't parse the config (Gemma 4 on v2.17.0)
  needs_remote_code      config ships custom code (K2-Horizon); retried with it on

Each step is recorded separately so one failure doesn't hide the others.
Writes /out/result.json and prints it as the last line.
"""

import json
import sys
import time
import traceback
from pathlib import Path


def step(result, name, fn):
    t0 = time.time()
    try:
        value = fn()
        result["steps"][name] = {"ok": True, "value": value, "s": round(time.time() - t0, 1)}
        return value
    except Exception as e:  # noqa: BLE001 - every failure is data here
        msg = f"{type(e).__name__}: {e}"
        result["steps"][name] = {"ok": False, "error": msg[:2000], "s": round(time.time() - t0, 1),
                                 "trace_tail": traceback.format_exc().splitlines()[-3:]}
        return None


def versions():
    import torch
    import transformers
    import vllm

    return {"vllm": vllm.__version__, "transformers": transformers.__version__, "torch": torch.__version__}


def hub_config(model):
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(model, "config.json")).read_text())
    return {"architectures": cfg.get("architectures") or [], "model_type": cfg.get("model_type"),
            "auto_map": sorted((cfg.get("auto_map") or {}).keys())}


def registry_has(archs):
    from vllm import ModelRegistry

    supported = set(ModelRegistry.get_supported_archs())
    return {a: a in supported for a in archs}


def transformers_config(model):
    from transformers import AutoConfig

    return type(AutoConfig.from_pretrained(model)).__name__


def vllm_model_config(model, remote):
    from vllm.config import ModelConfig

    mc = ModelConfig(model=model, trust_remote_code=remote)
    out = {"trust_remote_code": remote}
    for attr in ("architecture", "architectures", "model_impl", "runner_type"):
        if hasattr(mc, attr):
            out[attr] = str(getattr(mc, attr))
    return out


def tokenizer(model, remote):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=remote)
    return {"class": type(tok).__name__, "vocab": len(tok)}


def classify(r):
    s = r["steps"]
    if not s.get("hub_config", {}).get("ok"):
        err = s.get("hub_config", {}).get("error", "")
        return "gated_or_missing" if any(k in err for k in ("401", "403", "Gated", "gated", "404")) else "hub_error"
    if s.get("vllm_model_config", {}).get("ok") or s.get("vllm_model_config_remote", {}).get("ok"):
        native = any((s.get("registry") or {}).get("value", {}).values())
        remote = bool(s.get("vllm_model_config_remote", {}).get("ok"))
        kind = "native" if native else "fallback"
        return f"{kind}+remote_code" if remote else kind
    err = (s.get("vllm_model_config_remote") or s.get("vllm_model_config") or {}).get("error", "")
    if "does not recognize this architecture" in err or "Transformers does not recognize" in err:
        return "transformers_too_old"
    if "are not supported for now" in err:
        return "unsupported"
    return "error"


def main():
    model = sys.argv[1]
    r = {"model": model, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": {}}
    step(r, "versions", versions)
    cfg = step(r, "hub_config", lambda: hub_config(model))
    if cfg:
        step(r, "registry", lambda: registry_has(cfg["architectures"]))
        step(r, "transformers_config", lambda: transformers_config(model))
        step(r, "vllm_model_config", lambda: vllm_model_config(model, False))
        err = r["steps"]["vllm_model_config"].get("error", "")
        remote = bool(cfg["auto_map"]) and "trust_remote_code" in err
        if remote:
            step(r, "vllm_model_config_remote", lambda: vllm_model_config(model, True))
        step(r, "tokenizer", lambda: tokenizer(model, remote))
    r["verdict"] = classify(r)
    out = Path("/out")
    if out.is_dir():
        (out / "result.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
