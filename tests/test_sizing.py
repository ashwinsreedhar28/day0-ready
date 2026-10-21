"""GPU class from weight size, against the smoke runs recorded so far."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import sizing  # noqa: E402


@pytest.mark.parametrize("params,cls", [
    ({"BF16": 8_190_735_360}, "24gb"),        # Qwen/Qwen3-8B: passed on 24 GB
    ({"BF16": 7_615_616_512}, "24gb"),        # IFM/K2-Horizon-7B: passed on 24 GB
    ({"BF16": 39_785_287_456}, "141gb"),      # RecursiveSI/flint: 79.6 GB
    ({"BF16": 30_532_122_624}, "80gb"),       # a 30B dense model: 61 GB
    ({"F8_E4M3": 30_000_000_000, "BF16": 1_000_000_000}, "48gb"),  # fp8 halves it
    ({"BF16": 405_000_000_000}, None),        # multi-GPU
])
def test_pick_class(params, cls):
    assert sizing.pick_class(sizing.weights_gb({"safetensors": {"parameters": params}})) == cls


def test_falls_back_to_weight_file_sizes():
    info = {"siblings": [{"rfilename": "model-00001-of-00002.safetensors", "size": 10_000_000_000},
                         {"rfilename": "model-00002-of-00002.safetensors", "size": 6_000_000_000},
                         {"rfilename": "original/consolidated.pth", "size": 16_000_000_000},  # duplicate copy
                         {"rfilename": "tokenizer.json", "size": 11_000_000}]}
    assert sizing.weights_gb(info) == 16.0


def test_unknown_size():
    assert sizing.weights_gb({"siblings": [{"rfilename": "README.md"}]}) is None


def test_disk_and_cost():
    assert sizing.disk_gb(16.4) == 60
    assert sizing.disk_gb(79.6) == 117
    assert sizing.cost_usd(13.1, "180gb") == 1.89
