"""Recipe -> worker env, checked on shapes taken from real vllm-project/recipes files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import recipes  # noqa: E402
import smoke  # noqa: E402

B200 = smoke.GPU_CLASSES["180gb"]
H200 = smoke.GPU_CLASSES["141gb"]

# Step-5-Preview (abridged): single-GPU-relevant settings mixed with deploy-only ones, and a
# Blackwell MoE backend that the single_node_tp strategy turns back off.
STEP5 = {
    "model": {"model_id": "stepfun-ai/Step-5-Preview", "docker_image": "vllm/vllm-openai:step-5-preview",
              "base_args": ["--data-parallel-backend", "mp", "--disable-custom-all-reduce", "--max-model-len", "1048576",
                            "--max-num-seqs", "64", "--gpu-memory-utilization", "0.85", "--enable-prefix-caching"],
              "base_env": {}},
    "variants": {"default": {"precision": "fp8"}},
    "hardware_overrides": {"blackwell": {"extra_args": ["--moe-backend", "deep_gemm_mega_moe"]}},
    "strategy_overrides": {"single_node_tp": {"hardware_overrides": {"blackwell": {"extra_args": []}}}},
}
# IFM K2-Horizon-375B (abridged): remote code in base_args, AMD-only override.
K2 = {
    "meta": {"hardware": {"h200": "verified", "mi355x": "verified"}},
    "model": {"base_args": ["--trust-remote-code", "--chat-template-content-format", "string", "--dtype", "bfloat16"],
              "base_env": {}},
    "hardware_overrides": {"amd": {"extra_args": ["--attention-backend", "ROCM_AITER_UNIFIED_ATTN"],
                                   "extra_env": {"VLLM_ROCM_USE_AITER": "1"}}},
}


def test_step5_keeps_startup_settings_and_drops_mega_moe_on_one_gpu():
    s = recipes.settings(STEP5, B200)
    assert s["env"] == {"GPU_MEMORY_UTILIZATION": "0.85"}
    assert "--max-model-len 1048576" in s["skipped"] and "--enable-prefix-caching" in s["skipped"]
    assert s["generation"] == "blackwell"


def test_k2_remote_code_is_noted_not_applied_and_amd_block_ignored():
    s = recipes.settings(K2, H200)
    assert s["env"] == {"DTYPE": "bfloat16"}
    assert any("--trust-remote-code" in n for n in s["notes"])


def test_generation_block_beats_nvidia_fallback_and_variant_overrides_win():
    r = {"model": {"base_args": ["--attention-backend", "FLASH_ATTN"], "base_env": {"A": "1"}},
         "hardware_overrides": {"nvidia": {"extra_args": ["--kv-cache-dtype", "fp8"]},
                                "blackwell": {"extra_args": ["--attention-backend", "FLASHINFER"], "extra_env": {"B": "2"}}},
         "variants": {"default": {"hardware_overrides": {"b200": {"extra_args": ["--enforce-eager"]}}}}}
    assert recipes.settings(r, B200)["env"] == {"ATTENTION_BACKEND": "FLASHINFER", "A": "1", "B": "2", "ENFORCE_EAGER": "true"}
    # 24gb mixes Ada and Ampere: no single generation, so only the brand-wide block applies.
    assert recipes.settings(r, smoke.GPU_CLASSES["24gb"])["env"] == {
        "ATTENTION_BACKEND": "FLASH_ATTN", "A": "1", "KV_CACHE_DTYPE": "fp8"}


def test_hardware_of_each_class():
    assert recipes.hardware(smoke.GPU_CLASSES["141gb"]) == ("hopper", "h200")
    assert recipes.hardware(smoke.GPU_CLASSES["180gb"]) == ("blackwell", "b200")
    assert recipes.hardware(smoke.GPU_CLASSES["80gb"]) == (None, None)  # A100 and H100
    assert recipes.hardware(["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL"]) == ("hopper", "h100")
    assert recipes.hardware(["NVIDIA L40S"]) == ("ada", "l40s")


def test_parse_args_shapes():
    assert recipes.parse_args(["--a", "1", "--b", "--c=x", "--d", '{"k": 1}']) == [
        ("--a", "1"), ("--b", None), ("--c", "x"), ("--d", '{"k": 1}')]


def test_fetch_tries_the_next_source_on_404(monkeypatch):
    import io
    import urllib.error

    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        if "recipes/main" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)
        return io.BytesIO(b"model:\n  base_args: ['--dtype', 'bfloat16']\n")

    monkeypatch.setattr(recipes.urllib.request, "urlopen", urlopen)
    r, src = recipes.fetch("o/m")
    assert r["model"]["base_args"] == ["--dtype", "bfloat16"] and src.endswith("vllm/main/recipes/models/o/m.yaml")
    assert len(calls) == 2
