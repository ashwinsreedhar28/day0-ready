"""Smoke-test ladder against a fake Runpod API, replaying failures from study/."""

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import runpod_api as rp  # noqa: E402
import smoke  # noqa: E402


def fake_api(script):
    """script: one (job_status or None, output, [worker states per health poll]) per rung."""
    st = {"rung": -1, "polls": 0}

    def call(method, url, key, body=None):
        if url.endswith("/templates") and method == "POST":
            st["rung"] += 1
            st["polls"] = 0
            return 201, {"id": f"t{st['rung']}"}
        if url.endswith("/endpoints") and method == "POST":
            return 201, {"id": f"e{st['rung']}"}
        if url.endswith("/run"):
            return 200, {"id": "j"}
        status, output, states = script[st["rung"]]
        if url.endswith("/health"):
            st["polls"] += 1
            return 200, {"workers": states[min(st["polls"] - 1, len(states) - 1)]}
        if "/status/" in url:
            if status and st["polls"] >= len(states):
                return 200, {"status": status, "output": output, "error": output if status == "FAILED" else None}
            return 200, {"status": "IN_QUEUE"}
        return 204, None
    return call


def run_ladder(monkeypatch, script, env=None, gpu="24gb"):
    monkeypatch.setattr(rp, "call", fake_api(script))
    monkeypatch.setattr(smoke.time, "sleep", lambda s: None)
    clock = itertools.count(1_000_000, 30)
    monkeypatch.setattr(smoke.time, "time", lambda: next(clock))
    env, rungs = dict(env or {}), []
    while True:
        r = smoke.run_rung("k", "img", "m", env, gpu, "13.0", smoke.time.time() + 3600, f"t{len(rungs)}")
        rungs.append((r["outcome"], dict(env), gpu))
        if r["outcome"] == "pass":
            return rungs
        nxt = smoke.next_rung(env, gpu, r["outcome"], "TRUST_REMOTE_CODE" in env, detail=r.get("output") or "")
        if nxt is None:
            return rungs
        env, gpu = nxt


RUN = [{"running": 1}]


def test_first_request_oom_retries_with_headroom(monkeypatch):
    # Qwen3.5-4B on v.2.16.0: Triton autotune OOM on the first request at 0.95.
    rungs = run_ladder(monkeypatch, [("FAILED", {"error": {"message": "EngineCore encountered an issue."}}, RUN),
                                     ("COMPLETED", ["ok"], RUN)])
    assert rungs == [("first_request_failure", {}, "24gb"), ("pass", {"GPU_MEMORY_UTILIZATION": "0.85"}, "24gb")]


def test_silent_restart_loop_retries_with_remote_code(monkeypatch):
    # K2-Horizon on v2.28.0: ready -> running -> ready -> running -> ready, never unhealthy.
    loop = [{"ready": 1}, {"running": 1}, {"ready": 1}, {"running": 1}, {"ready": 1}]
    rungs = run_ladder(monkeypatch, [(None, None, loop), ("COMPLETED", ["ok"], RUN)])
    assert rungs == [("crash_loop", {}, "24gb"), ("pass", {"TRUST_REMOTE_CODE": "true"}, "24gb")]


def test_startup_oom_moves_up_a_gpu_class(monkeypatch):
    msg = {"error": {"message": "X ran out of GPU memory during startup.", "type": "startup_error"}}
    rungs = run_ladder(monkeypatch, [("FAILED", msg, RUN), ("COMPLETED", ["ok"], RUN)])
    assert rungs == [("startup_oom", {}, "24gb"), ("pass", {}, "48gb")]


def test_unsupported_architecture_stops(monkeypatch):
    msg = {"error": {"message": "Model architectures ['X'] are not supported", "type": "startup_error"}}
    assert run_ladder(monkeypatch, [("FAILED", msg, RUN)]) == [("startup_error", {}, "24gb")]


def test_remote_code_is_tried_once(monkeypatch):
    loop = [{"unhealthy": 1}, {"ready": 1}, {"unhealthy": 1}]
    rungs = run_ladder(monkeypatch, [(None, None, loop)], env={"TRUST_REMOTE_CODE": "true"})
    assert rungs == [("crash_loop", {"TRUST_REMOTE_CODE": "true"}, "24gb")]


def test_startup_oom_climbs_to_h200_then_b200_then_stops():
    assert smoke.next_rung({}, "80gb", "startup_oom", False) == ({}, "141gb")
    assert smoke.next_rung({}, "141gb", "startup_oom", False) == ({}, "180gb")
    assert smoke.next_rung({}, "180gb", "startup_oom", False) is None


def test_worker_remote_code_startup_error_retries_with_remote_code(monkeypatch):
    msg = {"error": {"type": "startup_error", "message": "IFM/K2-Horizon-7B ships its own modeling code on Hugging "
                     "Face (an auto_map in its config.json), and vLLM only runs that code with TRUST_REMOTE_CODE=true."}}
    rungs = run_ladder(monkeypatch, [("FAILED", msg, RUN), ("COMPLETED", ["ok"], RUN)])
    assert rungs == [("needs_remote_code", {}, "24gb"), ("pass", {"TRUST_REMOTE_CODE": "true"}, "24gb")]


def test_gpu_climb_stops_at_max_gpu():
    assert smoke.next_rung({}, "48gb", "startup_oom", False, max_gpu="48gb") is None
    assert smoke.next_rung({}, "24gb", "startup_oom", False, max_gpu="48gb") == ({}, "48gb")


def test_crash_loop_skips_remote_code_when_cpu_check_ruled_it_out():
    # flint on H200: native, no auto_map; the loop was a Blackwell-only MoE kernel.
    assert smoke.next_rung({"ATTENTION_BACKEND": "FLASHINFER"}, "141gb", "crash_loop", False,
                           remote_code_possible=False) is None
    assert smoke.next_rung({}, "24gb", "needs_remote_code", False, remote_code_possible=False) == (
        {"TRUST_REMOTE_CODE": "true"}, "24gb")


def test_cpu_verdict_lookup_by_pinned_tag():
    assert smoke.cpu_verdict("ghcr.io/ashwinsreedhar28/worker-vllm:day0-flint-b779895") == "native"
    assert smoke.cpu_verdict("ghcr.io/ashwinsreedhar28/worker-vllm:day0-v0.32.0--fix-remote-code-startup-error-c1b457f") == "native+remote_code"
    assert smoke.cpu_verdict("runpod/worker-v1-vllm:v2.27.2") is None


def test_ci_never_runs_repo_code_without_approval():
    assert smoke.next_rung({}, "24gb", "needs_remote_code", False, allow_remote_code=False) is None
    assert smoke.next_rung({}, "24gb", "crash_loop", False, allow_remote_code=False) is None
    # Settings that don't run repo code still apply.
    assert smoke.next_rung({}, "24gb", "first_request_failure", False, allow_remote_code=False) == (
        {"GPU_MEMORY_UTILIZATION": "0.85"}, "24gb")


def test_spend_counts_only_the_last_24h(tmp_path):
    import json
    from datetime import datetime, timezone
    (tmp_path / "a.json").write_text(json.dumps({"rungs": [
        {"started": "2026-10-21T21:20:40+00:00", "gpu_class": "180gb", "minutes": 13.1, "est_cost_usd": 1.89},
        {"started": "2026-10-21T20:12:01+00:00", "gpu_class": "24gb", "outcome": "error"},  # template 401, no minutes
        {"started": "2026-10-19T20:00:00+00:00", "gpu_class": "141gb", "minutes": 30, "est_cost_usd": 2.97}]}))
    (tmp_path / "b.json").write_text(json.dumps({"rungs": [  # older result without est_cost_usd
        {"started": "2026-10-21T20:34:00+00:00", "gpu_class": "141gb", "minutes": 31.3}]}))
    now = datetime(2026, 10, 21, 22, 0, tzinfo=timezone.utc)
    assert smoke.spent_last_24h(now, tmp_path) == round(1.89 + 3.09, 2)


def test_passing_rung_records_a_redeployable_body(monkeypatch):
    rungs = run_ladder(monkeypatch, [("COMPLETED", ["ok"], RUN)])
    assert rungs[0][0] == "pass"
    body = smoke.template_body("img", "m/x", {"ATTENTION_BACKEND": "FLASHINFER"}, 150)
    assert body["env"] == {"MODEL_NAME": "m/x", "MAX_MODEL_LEN": "4096", "ATTENTION_BACKEND": "FLASHINFER"}
    assert smoke.endpoint_body("180gb", "13.0")["gpuTypeIds"] == ["NVIDIA B200"]


def test_too_large_for_max_gpu_is_recorded_not_run(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setattr(smoke.rp, "load_key", lambda f=None: "k")
    monkeypatch.setattr(smoke.sizing, "hf_model_info", lambda m: {"safetensors": {"parameters": {"BF16": 39_785_287_456}}})
    monkeypatch.setattr(smoke.rp, "call", lambda *a, **k: pytest.fail("no API call for a skipped run"))
    monkeypatch.setattr(sys, "argv", ["smoke", "--image", "ghcr.io/x/w:day0-flint-abc1234", "--model", "RecursiveSI/flint",
                                      "--gpu", "auto", "--max-gpu", "48gb"])
    assert smoke.main() == 1
    r = json.loads((tmp_path / "data" / "smoke" / "day0-flint-abc1234.json").read_text())
    assert r["skipped"]["reason"] == "needs_larger_gpu" and r["weights_gb"] == 79.6 and r["rungs"] == []


def test_budget_stops_before_spending(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setattr(smoke.rp, "load_key", lambda f=None: "k")
    monkeypatch.setattr(smoke, "spent_last_24h", lambda: 9.80)
    monkeypatch.setattr(smoke.recipes, "fetch", lambda m: (None, None))
    monkeypatch.setattr(smoke.rp, "call", lambda *a, **k: pytest.fail("over budget: no endpoint"))
    monkeypatch.setattr(sys, "argv", ["smoke", "--image", "ghcr.io/x/w:day0-q-abc1234", "--model", "Qwen/Qwen3-8B",
                                      "--gpu", "24gb", "--disk-gb", "60", "--budget-usd", "10"])
    assert smoke.main() == 1
    r = json.loads((tmp_path / "data" / "smoke" / "day0-q-abc1234.json").read_text())
    assert r["skipped"]["reason"] == "budget"



def test_gpu_types_override_reaches_the_endpoint(monkeypatch):
    # Hopper only: the 80gb class starts with A100s, where vLLM picks another attention backend.
    bodies = []

    def call(method, url, key, body=None):
        if method == "POST" and url.endswith("/endpoints"):
            bodies.append(body)
            return 500, "stop after the endpoint request"
        return 201, {"id": "t"}

    monkeypatch.setattr(rp, "call", call)
    r = smoke.run_rung("k", "img", "m", {}, "80gb", "13.0", 0, "t", 150, ["NVIDIA H100 80GB HBM3"])
    assert bodies[0]["gpuTypeIds"] == ["NVIDIA H100 80GB HBM3"] and r["outcome"] == "error"
    assert smoke.next_rung({}, "80gb", "startup_oom", False, max_gpu="80gb") is None  # no climb past the list


def _model_code(cause):
    # worker-vllm #363's startup_error, as the job returns it.
    return {"error": {"type": "startup_error", "message": (
        f"RecursiveSI/flint failed inside vLLM's implementation of the model during startup: {cause}. The model's "
        "code requires something this worker's GPU or settings don't provide, so restarting this worker will not "
        "change it. Check the model card and vLLM's recipe for the attention backend (ATTENTION_BACKEND), GPU "
        "generation or dtype it needs.")}}


FLINT_ATTN = _model_code("AssertionError: Flint requires FlashInfer attention")
FLINT_MOE = _model_code("RuntimeError: Flint BF16 decode MoE requires the FlashInfer TRTLLM BF16 MoE backend "
                        "without padding (got UnquantizedMoeBackend.TRITON)")


def test_flint_climbs_to_flashinfer_then_b200(monkeypatch):
    # The Oct 21 flint launch, without the human: H200 picks FA3, then the BF16 MoE wants a TRTLLM kernel.
    rungs = run_ladder(monkeypatch, [("FAILED", FLINT_ATTN, RUN), ("FAILED", FLINT_MOE, RUN), ("COMPLETED", ["ok"], RUN)],
                       gpu="141gb")
    assert rungs == [("model_code", {}, "141gb"), ("model_code", {"ATTENTION_BACKEND": "FLASHINFER"}, "141gb"),
                     ("pass", {"ATTENTION_BACKEND": "FLASHINFER"}, "180gb")]


def test_model_code_blackwell_respects_max_gpu():
    detail = smoke.json.dumps(FLINT_MOE)
    env = {"ATTENTION_BACKEND": "FLASHINFER"}
    assert smoke.next_rung(env, "141gb", "model_code", False, max_gpu="141gb", detail=detail) is None
    assert smoke.model_code_hint(detail, "141gb", "141gb") == "needs a Blackwell GPU (180gb); --max-gpu is 141gb"
    assert smoke.next_rung(env, "180gb", "model_code", False, detail=detail) is None  # already on B200


def test_model_code_without_a_known_fix_stops():
    detail = smoke.json.dumps(_model_code("NotImplementedError: Foo needs bfloat16 weights"))
    assert smoke.next_rung({}, "24gb", "model_code", False, detail=detail) is None
    # The attention fix is tried once: same requirement again means the backend didn't help.
    detail = smoke.json.dumps(FLINT_ATTN)
    assert smoke.next_rung({"ATTENTION_BACKEND": "FLASHINFER"}, "24gb", "model_code", False, detail=detail) is None


def test_model_code_reads_only_the_cause_not_the_advice():
    # The worker's advice mentions ATTENTION_BACKEND and "GPU generation"; neither is a requirement.
    detail = smoke.json.dumps(_model_code("ValueError: Foo only supports head_dim 128"))
    assert smoke.classify_failure("FAILED", smoke.json.loads(detail), 0) == "model_code"
    assert smoke.next_rung({}, "24gb", "model_code", False, detail=detail) is None


def test_recipe_settings_seed_each_rung_and_set_wins(monkeypatch, tmp_path):
    import json
    recipe = {"model": {"base_args": ["--dtype", "bfloat16", "--max-model-len", "262144"], "base_env": {}},
              "hardware_overrides": {"blackwell": {"extra_args": ["--moe-backend", "flashinfer_trtllm"]}},
              "variants": {"default": {"precision": "bf16"}}}
    envs = []

    def call(method, url, key, body=None):
        if url.endswith("/templates") and method == "POST":
            envs.append(body["env"])
            return 201, {"id": "t"}
        if url.endswith("/endpoints") and method == "POST":
            return 201, {"id": "e"}
        if url.endswith("/run"):
            return 200, {"id": "j"}
        if url.endswith("/health"):
            return 200, {"workers": {"running": 1}}
        if "/status/" in url:
            out = {"error": {"type": "startup_error", "message": "X ran out of GPU memory during startup."}}
            return 200, ({"status": "FAILED", "error": out} if len(envs) == 1 else {"status": "COMPLETED", "output": ["ok"]})
        return 204, None

    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setattr(smoke.rp, "load_key", lambda f=None: "k")
    monkeypatch.setattr(smoke.rp, "call", call)
    monkeypatch.setattr(smoke.time, "sleep", lambda s: None)
    monkeypatch.setattr(smoke.recipes, "fetch", lambda m: (recipe, "https://example/recipe.yaml"))
    monkeypatch.setattr(sys, "argv", ["smoke", "--image", "ghcr.io/x/w:day0-r-abc1234", "--model", "o/r", "--gpu", "141gb",
                                      "--max-gpu", "180gb", "--disk-gb", "60", "--set", "DTYPE=float16"])
    assert smoke.main() == 0
    # H200 has no hopper block; B200 adds the blackwell MoE backend. --set DTYPE beats the recipe's bfloat16.
    assert {k: v for k, v in envs[0].items() if k not in ("MODEL_NAME", "MAX_MODEL_LEN")} == {"DTYPE": "float16"}
    assert {k: v for k, v in envs[1].items() if k not in ("MODEL_NAME", "MAX_MODEL_LEN")} == {
        "DTYPE": "float16", "MOE_BACKEND": "flashinfer_trtllm"}
    r = json.loads((tmp_path / "data" / "smoke" / "day0-r-abc1234.json").read_text())
    assert r["recipe"]["source"] == "https://example/recipe.yaml" and "--max-model-len 262144" in r["recipe"]["skipped"]
    assert r["deploy"]["template"]["env"]["MOE_BACKEND"] == "flashinfer_trtllm"
