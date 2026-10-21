# day0-ready

Goal: when an open-weight model drops, a tested Runpod Serverless vLLM worker for it exists within hours, not weeks.

`study/` measures the gap today. In a hand-picked sample of 25 new-architecture releases in 2026, the median wait from public weights to a published `runpod/worker-v1-vllm` image whose vLLM lists the architecture is **24 days** (over the 23 with a vLLM final release; a registry proxy that can overstate the real serving delay). vLLM itself usually publishes a model-named image (`vllm/vllm-openai:gemma4`, `step-5-preview`, ...) on release day: 16 of the 18 such images were pushed the same day. The worker can't build from those images today because of a version check, fixed in [ashwinsreedhar28/worker-vllm `fix/version-guard-any-base-tag`](https://github.com/ashwinsreedhar28/worker-vllm/tree/fix/version-guard-any-base-tag).

First launch through the pipeline: [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) (repo created 2026-10-21 14:21 UTC). vLLM pushed `vllm/vllm-openai:flint` at 16:09, the worker image built from it was pushed at 20:05, and it answered a chat request on a Runpod Serverless B200 at 21:33 the same day, with one model-specific setting (`ATTENTION_BACKEND=FLASHINFER`). It can't run on an H200 in vLLM: its BF16 MoE needs a Blackwell-only kernel ([result and logs](data/smoke/day0-flint-b779895.json)). The detector wasn't live when flint launched, so this isn't a time-to-ready measurement yet.

## Pipeline

| stage | what | GPU | status |
|---|---|---|---|
| detect | every 30 min: new model-named `vllm/vllm-openai` tags, matched to a Hugging Face repo (24-lab watchlist, then HF search) | no | `.github/workflows/detect.yml`, `detect/poll.py`; starts a build per new launch |
| build | worker-vllm `FROM vllm/vllm-openai:<day-0 tag>`, pushed to `ghcr.io` | no | `.github/workflows/build-day0.yml` |
| CPU checks | inside the built image: architecture resolution, transformers config, vLLM `ModelConfig`, tokenizer | no | `checks/cpu_check.py`, run by the build |
| smoke test | one request on a Serverless endpoint; on failure: `GPU_MEMORY_UTILIZATION=0.85` (first-request OOM), `TRUST_REMOTE_CODE=true` (remote-code error, or a crash loop the CPU check didn't rule out), next GPU class (startup OOM, capped by `--max-gpu`) | yes, capped | `pipeline/smoke.py`, run by hand with a Runpod key |
| pre-warm | torch.compile cache on a network volume per (model, GPU) | yes | planned |
| board | per launch: tag push time, model, build, CPU verdict, smoke test, tag → worker image | no | [`data/board.md`](data/board.md), regenerated on every recorded build |

## Run a build

Actions → **Build day-0 worker** → Run workflow, with a `vllm_tag` (e.g. `muse-spark-open`) and the Hugging Face `model` to check. The result lands in the run summary and as `ghcr.io/ashwinsreedhar28/worker-vllm:day0-<tag>`.

## Run a smoke test

Needs a Runpod API key (kept out of the repo, e.g. in `~/.runpod.env`). Spend is capped by `--max-minutes` and `--max-gpu`; endpoints are scaled to zero on exit and deleted on a pass.

```
python3 pipeline/smoke.py --env-file ~/.runpod.env \
  --image ghcr.io/ashwinsreedhar28/worker-vllm:day0-<tag>-<sha> --model <org/repo> \
  --gpu 24gb --max-gpu 48gb [--set KEY=VALUE ...]
```

The result goes to `data/smoke/<image tag>.json` and shows up on the board.
