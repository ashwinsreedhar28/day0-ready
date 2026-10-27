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


SWEEP = {"image": FLINT["image"], "model": "RecursiveSI/flint", "gpu_type": "NVIDIA B200", "verdict": "ok",
         "recommended": "S3", "fast_start": "S2", "finished_at": "2027-01-12T00:00:00+00:00", "stopped": None,
         "configs": [{"config": "S2", "engine_init_s_p50": 30.1, "output_tok_s_c30": 900.0},
                     {"config": "S3", "engine_init_s_p50": 41.2, "output_tok_s_c30": 1510.4}],
         "recommended_block": {"config": "S3", "env": {"MODEL_NAME": "RecursiveSI/flint", "MAX_NUM_SEQS": "32"}}}


def test_sweep_cell():
    assert board.sweep_text(SWEEP) == ("S3: 41.2 s engine init, 1510 tok/s at C=30; "
                                       "fast start S2: 30.1 s, 900 tok/s (NVIDIA B200)")
    assert board.sweep_text({**SWEEP, "recommended": None, "verdict": "inconclusive",
                             "stopped": {"reason": "inconclusive"}}) == "inconclusive (stopped: inconclusive)"


def test_ready_gains_the_sweeps_recommended_block():
    builds = {FLINT["image"]: FLINT_BUILD}
    e = board.ready_entries({"day0-flint": FLINT}, builds, {FLINT["image"]: SWEEP})[0]
    assert e["recommended"]["config"] == "S3"
    assert e["env"] == {"ATTENTION_BACKEND": "FLASHINFER"}  # the day-0 block stays as it was
    other_model = {**SWEEP, "model": "Qwen/Qwen3-8B"}
    assert "recommended" not in board.ready_entries({"day0-flint": FLINT}, builds, {FLINT["image"]: other_model})[0]
    assert "recommended" not in board.ready_entries({"day0-flint": FLINT}, builds)[0]


def test_latest_sweep_per_image(tmp_path):
    for name, finished in (("a", "2027-01-12T00:00:00+00:00"), ("b", "2027-01-13T00:00:00+00:00")):
        (tmp_path / "sweep" / name).mkdir(parents=True)
        (tmp_path / "sweep" / name / "summary.json").write_text(
            __import__("json").dumps({**SWEEP, "finished_at": finished, "recommended": name}))
    assert board.load_sweeps(tmp_path)[FLINT["image"]]["recommended"] == "b"
