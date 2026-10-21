"""Board cells and data/ready.json from recorded smoke results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "detect"))
import board  # noqa: E402

FLINT = {"image": "ghcr.io/a/worker-vllm:day0-flint-b779895", "model": "RecursiveSI/flint", "passed": True,
         "passing_env": {"ATTENTION_BACKEND": "FLASHINFER"}, "passing_gpu_class": "180gb",
         "finished_at": "2026-10-21T21:33:51+00:00",
         "rungs": [{"env": {"ATTENTION_BACKEND": "FLASHINFER"}, "gpu_class": "180gb", "disk_gb": 150,
                    "minutes": 13.1, "outcome": "pass"}]}
FLINT_BUILD = {"vllm_tag": "flint", "model": "RecursiveSI/flint", "image": "ghcr.io/a/worker-vllm:day0-flint",
               "image_pinned": FLINT["image"], "digest": "sha256:d3c8"}


def test_cells():
    assert board.smoke_text(FLINT) == "pass: flint, 180gb, ATTENTION_BACKEND=FLASHINFER, 13.1 min"
    assert board.smoke_text({"model": "a/b", "passed": False, "rungs": [],
                             "skipped": {"reason": "needs_larger_gpu", "detail": "needs 141gb; --max-gpu is 48gb"}}) \
        == "not run: needs 141gb; --max-gpu is 48gb"
    assert "review the repo's code" in board.smoke_text(
        {"model": "IFM/K2-Horizon-7B", "passed": False, "rungs": [{"outcome": "needs_remote_code"}]})
    assert board.smoke_text({"model": "a/b", "passed": False, "rungs": [{"outcome": "startup_error"}],
                             "stopped": {"reason": "budget", "detail": "x"}}) == "fail: b, startup_error (stopped: budget)"


def test_ready_lists_only_launch_models_on_board_images():
    qwen_standin = {**FLINT, "image": "ghcr.io/a/worker-vllm:day0-step-5-preview", "model": "Qwen/Qwen3-8B"}
    branch = {**FLINT, "image": "ghcr.io/a/worker-vllm:day0-flint--fix-x-04233af"}
    smoke = {"day0-flint": FLINT, "day0-step-5-preview": qwen_standin, "day0-flint--fix-x": branch}
    builds = {FLINT["image"]: FLINT_BUILD,
              qwen_standin["image"]: {"model": "stepfun-ai/Step-5-Preview", "image": qwen_standin["image"]}}
    entries = board.ready_entries(smoke, builds)
    assert [e["model"] for e in entries] == ["RecursiveSI/flint"]
    e = entries[0]
    assert e["digest"] == "sha256:d3c8" and e["vllm_tag"] == "flint"
    assert e["deploy"]["template"]["env"]["ATTENTION_BACKEND"] == "FLASHINFER"
    assert e["deploy"]["template"]["containerDiskInGb"] == 150
    assert e["deploy"]["endpoint"]["gpuTypeIds"] == ["NVIDIA B200"]
