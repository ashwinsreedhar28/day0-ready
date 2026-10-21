# day0-ready

Goal: when an open-weight model drops, a tested Runpod Serverless vLLM worker for it exists within hours, not weeks.

`study/` measures the gap today. For 25 new-architecture releases in 2026, the median wait from public weights to a published `runpod/worker-v1-vllm` image whose vLLM lists the architecture is **24 days**. vLLM itself usually publishes a model-named image (`vllm/vllm-openai:gemma4`, `step-5-preview`, ...) on release day: 16 of the 18 such images were pushed the same day. The worker can't build from those images today because of a version check, fixed in [ashwinsreedhar28/worker-vllm `fix/version-guard-any-base-tag`](https://github.com/ashwinsreedhar28/worker-vllm/tree/fix/version-guard-any-base-tag).

## Pipeline

| stage | what | GPU | status |
|---|---|---|---|
| detect | poll vLLM model-named image tags and a Hugging Face lab watchlist | no | planned |
| build | worker-vllm `FROM vllm/vllm-openai:<day-0 tag>`, pushed to `ghcr.io` | no | `.github/workflows/build-day0.yml` |
| CPU checks | inside the built image: architecture resolution, transformers config, tokenizer | no | `checks/cpu_check.py` |
| smoke test | one request on a Serverless endpoint, fallback ladder: default → `GPU_MEMORY_UTILIZATION=0.85` → `TRUST_REMOTE_CODE=true` | yes, capped | from `study/scripts/validate_proxy.py` |
| pre-warm | torch.compile cache on a network volume per (model, GPU) | yes | planned |
| board | per model: release time, image, checks, smoke test, time to ready | no | planned |

## Run a build

Actions → **Build day-0 worker** → Run workflow, with a `vllm_tag` (e.g. `muse-spark-open`) and the Hugging Face `model` to check. The result lands in the run summary and as `ghcr.io/ashwinsreedhar28/worker-vllm:day0-<tag>`.
