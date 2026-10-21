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
        nxt = smoke.next_rung(env, gpu, r["outcome"], "TRUST_REMOTE_CODE" in env)
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
