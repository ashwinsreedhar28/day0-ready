"""GPU-class prediction from vLLM's boot numbers, checked against the boots we've recorded."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import predict  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# Pilot run 3, S0 warm-up on an L40S: 15.27 GiB of weights, 23.68 GiB of KV, 172,416 tokens.
QWEN3_8B = predict.from_numbers(15.27, 23.68, 172_416)
QWEN3_8B_CONFIG = {"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 36, "num_attention_heads": 32,
                   "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 4096, "torch_dtype": "bfloat16",
                   "max_position_embeddings": 40960}


def row(p, cls):
    return next(r for r in p["classes"] if r["gpu_class"] == cls)


def test_kv_bytes_per_token_is_the_same_on_two_devices():
    # Same model on an L40S and on an RTX PRO 6000 Blackwell MIG slice (pilot runs 1 and 3).
    blackwell = predict.from_numbers(15.27, 26.4, 192_208)
    assert QWEN3_8B["kv_bytes_per_token"] == pytest.approx(blackwell["kv_bytes_per_token"], rel=0.001)


def test_config_estimate_matches_the_measurement_for_plain_attention():
    est = predict.from_config(QWEN3_8B_CONFIG, 15.27)
    assert est["kv_bytes_per_token"] == 2 * 36 * 8 * 128 * 2 == 147_456
    assert est["kv_bytes_per_token"] == pytest.approx(QWEN3_8B["kv_bytes_per_token"], rel=0.001)


@pytest.mark.parametrize("extra", [{"kv_lora_rank": 512}, {"layer_types": ["linear_attention", "full_attention"]},
                                   {"sliding_window": 4096}, {"full_attention_interval": 4}])
def test_config_estimate_refuses_mla_hybrid_and_sliding_window(extra):
    assert predict.from_config({**QWEN3_8B_CONFIG, **extra}, 15.27) is None


def test_qwen3_8b_needs_48_gb_for_the_bench_and_24_gb_for_the_smoke_test():
    p = predict.predict(QWEN3_8B, "Qwen/Qwen3-8B")
    assert row(p, "24gb")["smoke_fits"] and not row(p, "24gb")["bench_fits"]   # smoke passed on 24 GB (Oct 21)
    assert p["smoke_class"] == "24gb" and p["sweep_class"] == "48gb"
    # Measured on an L40S: 172,416 tokens. 3 GiB of overhead vs 2.44 measured keeps the prediction just under.
    assert row(p, "48gb")["kv_tokens"] == pytest.approx(172_416, rel=0.03) and row(p, "48gb")["kv_tokens"] < 172_416
    assert "S5" in p["sweep_configs"]   # vLLM's default batch budget on a 48 GB card is 2,048


def test_flint_needs_an_h200():
    log = (ROOT / "data/smoke/logs/day0-flint-b779895_h200_flashinfer.txt").read_text(errors="replace")
    m = predict.from_log(log)
    assert m["weights_gib"] == 74.23
    p = predict.predict(m, "RecursiveSI/flint")
    assert not row(p, "80gb")["smoke_fits"]   # 74 GiB of weights leave nothing on an 80 GB card
    assert p["smoke_class"] == p["sweep_class"] == "141gb"
    assert row(p, "141gb")["kv_tokens"] == pytest.approx(486_702, rel=0.01)   # measured on the H200
    assert "S5" not in p["sweep_configs"]   # 8,192 is already vLLM's default there


def test_too_big_for_one_gpu():
    p = predict.predict({"weights_gib": 208.6, "kv_bytes_per_token": 100_000, "source": "given"}, "MiniMax-M3.1-Flash")
    assert p["sweep_class"] is None and p["smoke_class"] is None and "multi-GPU" in p["note"]


def test_from_record_reads_a_sweep_boot_report():
    rec = {"runs": [{"boot": None}, {"boot": {"vllm": {"model_load_gib": 15.27, "kv_cache_gib": 23.68,
                                                       "kv_cache_tokens": 172_416}}}]}
    assert predict.from_record(rec) == QWEN3_8B


def test_neighbors_are_models_of_the_same_shape(tmp_path):
    d = tmp_path / "sweep" / "day0-x--day0-sweep-abc"
    d.mkdir(parents=True)
    (d / "S0.json").write_text(json.dumps({"runs": [{"boot": {"vllm": {
        "model_load_gib": 15.27, "kv_cache_gib": 23.68, "kv_cache_tokens": 172_416}}}]}))
    (d / "summary.json").write_text(json.dumps({"model": "Qwen/Qwen3-8B", "device": "NVIDIA L40S",
                                                "verdict": "incomplete", "recommended": None, "best_so_far": "S0"}))
    finetune = {"weights_gib": 15.3, "kv_bytes_per_token": 147_456, "source": "config estimate"}
    n = predict.neighbors(finetune, data=tmp_path)
    assert [x["model"] for x in n] == ["Qwen/Qwen3-8B"] and n[0]["best_so_far"] == "S0"
    assert predict.neighbors({"weights_gib": 30.5, "kv_bytes_per_token": 147_456}, data=tmp_path) == []


def test_qwen3_14b_needs_two_24_gb_cards():
    # 14.8B parameters in bf16 = 29.5 GB (27.5 GiB); 40 layers x 8 KV heads x 128 x 2 bytes x 2 = 163,840 B/token.
    m = {"weights_gib": 27.5, "kv_bytes_per_token": 163_840, "source": "config estimate"}
    one, two = predict.predict(m, gpus=1), predict.predict(m, gpus=2)
    assert not row(one, "24gb")["smoke_fits"] and row(two, "24gb")["smoke_fits"]
    assert row(two, "24gb")["usd_per_hr"] == 2 * row(one, "24gb")["usd_per_hr"]
    assert 60_000 < row(two, "24gb")["kv_tokens"] < 80_000
