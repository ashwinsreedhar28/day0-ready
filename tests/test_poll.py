"""Detector logic against tag and repo names seen in study/."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("poll", Path(__file__).resolve().parent.parent / "detect" / "poll.py")
poll = importlib.util.module_from_spec(spec)
spec.loader.exec_module(poll)


@pytest.mark.parametrize("tag", ["v0.32.0", "v0.32.0-cu130", "latest-x86_64", "nightly", "nightly-aarch64",
                                 "cu134-nightly-1b2a666391f2a24228a1436b64d547c34cc6b6ce", "cu129-nightly",
                                 "nightly-x86_64-dev-20261012-59e75f4", "cu134-nightly-aarch64-dev-20261017",
                                 "vllm-minimax-m3-perf-x86_64-13.0.1-8b00f41", "minimax-m3-perf-arm64-13.0.1-7a67223"])
def test_release_and_nightly_tags_are_not_launches(tag):
    assert poll.NOT_A_LAUNCH.match(tag)


@pytest.mark.parametrize("tag,family", [
    ("step-5-preview-arm64-cu130", "step-5-preview"),
    ("gemma4-0409-x86_64-cu130", "gemma4"),
    ("deepseekv41-flash-0909-cu129-amd64", "deepseekv41-flash"),
    ("qwen3_5-021926-aarch64-cu129", "qwen3_5"),
    ("mimov25-cu129-x86_64", "mimov25"),
    ("muse-spark-open", "muse-spark-open"),
    ("hy3-preview-x86_64-cu129", "hy3-preview"),
    ("glm53-flash-arm64-cu129", "glm53-flash"),
])
def test_tag_family(tag, family):
    assert poll.tag_family(tag) == family


REPOS = [{"id": i, "created": "2026-01-01"} for i in [
    "stepfun-ai/Step-5-Preview", "stepfun-ai/Step-5-Preview-BF16", "meta-models/Muse-Spark-Open-816B-A42B",
    "meta-models/Muse-Glimmer-30B", "google/gemma-4-31B-it", "google/gemma-4-E2B-it", "moonshotai/Kimi-K3",
    "deepseek-ai/DeepSeek-V4.1-Flash", "zai-org/GLM-5.3-Flash", "zai-org/GLM-5.3", "tencent/Hy4-preview",
    "Qwen/Qwen3.5-397B-A17B", "XiaomiMiMo/MiMo-V2.5-Pro", "poolside/Laguna-XS.2", "MiniMaxAI/MiniMax-M3"]]


@pytest.mark.parametrize("family,repo", [
    ("step-5-preview", "stepfun-ai/Step-5-Preview"),
    ("muse-spark-open", "meta-models/Muse-Spark-Open-816B-A42B"),
    ("muse-glimmer", "meta-models/Muse-Glimmer-30B"),
    ("kimi-k3", "moonshotai/Kimi-K3"),
    ("deepseekv41-flash", "deepseek-ai/DeepSeek-V4.1-Flash"),
    ("glm53-flash", "zai-org/GLM-5.3-Flash"),
    ("hy4-preview", "tencent/Hy4-preview"),
    ("qwen3_5", "Qwen/Qwen3.5-397B-A17B"),
    ("mimov25", "XiaomiMiMo/MiMo-V2.5-Pro"),
    ("laguna", "poolside/Laguna-XS.2"),
    ("minimax-m3", "MiniMaxAI/MiniMax-M3"),
])
def test_match_repo(family, repo):
    assert poll.match_repo(family, REPOS) == repo


def test_no_match_for_unrelated_family():
    assert poll.match_repo("reflection-beam", REPOS) is None


@pytest.mark.parametrize("repo", ["unsloth/Muse-Glimmer-30B-GGUF", "meta-models/Muse-Glimmer-30B-assistant",
                                  "Qwen/Qwen3.6-35B-A3B-FP8", "z-lab/Muse-Glimmer-30B-DFlash2"])
def test_derivatives_filtered(repo):
    assert poll.DERIVATIVE.search(repo)


FLINT_SEARCH = [  # HF search for "flint", sorted by likes (shape of the real 2026-10-21 response)
    {"id": "RecursiveSI/flint", "createdAt": "2026-10-21T14:20:58.000Z", "likes": 123},
    {"id": "unsloth/flint-GGUF", "createdAt": "2026-10-20T03:26:14.000Z", "likes": 28},
    {"id": "RecursiveSI/flint-int4", "createdAt": "2026-10-21T13:51:27.000Z", "likes": 2},
    {"id": "TechnoBaptist/flint", "createdAt": "2026-10-21T18:33:29.000Z", "likes": 0},
    {"id": "himankhanda/flint-bert-base", "createdAt": "2026-10-17T17:09:53.000Z", "likes": 0},
    {"id": "Cargile/flint-473", "createdAt": "2026-07-12T04:26:14.000Z", "likes": 9},
]


def test_search_fallback_finds_unlisted_lab():
    hit = poll.search_repo("flint", "2026-10-21T16:09:51Z", search=lambda q: FLINT_SEARCH)
    assert hit == ("RecursiveSI/flint", 123)


def test_search_fallback_ignores_repos_far_from_the_tag_date():
    old_only = [r for r in FLINT_SEARCH if r["id"] == "Cargile/flint-473"]
    assert poll.search_repo("flint", "2026-10-21T16:09:51Z", search=lambda q: old_only) is None


def test_watchlist_match_wins_over_search(monkeypatch):
    monkeypatch.setattr(poll, "search_repo", lambda *a, **k: ("someone/step-5-preview", 999))
    assert poll.resolve_model("step-5-preview", "2026-10-15T00:51:17Z", REPOS) == {
        "model": "stepfun-ai/Step-5-Preview", "match": "watchlist"}
