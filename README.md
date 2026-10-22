# day0-ready

Goal: when an open-weight model drops, a tested Runpod Serverless vLLM worker for it exists within hours, not weeks.

`study/` measures the gap today. In a hand-picked sample of 25 new-architecture releases in 2026, the median wait from public weights to a published `runpod/worker-v1-vllm` image whose vLLM lists the architecture is **24 days** (over the 23 with a vLLM final release; a registry proxy that can overstate the real serving delay). vLLM itself usually publishes a model-named image (`vllm/vllm-openai:gemma4`, `step-5-preview`, ...) on release day: 16 of the 18 such images were available on or before the recorded release date. The worker can't build from those images today because of a version check, fixed in [ashwinsreedhar28/worker-vllm `fix/version-guard-any-base-tag`](https://github.com/ashwinsreedhar28/worker-vllm/tree/fix/version-guard-any-base-tag).

First launch through the pipeline: [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) (repo created 2026-10-21 14:21 UTC). vLLM pushed `vllm/vllm-openai:flint` at 16:09, the worker image built from it was pushed at 20:05, and it answered a chat request on a Runpod Serverless B200 at 21:33 the same day, with one model-specific setting (`ATTENTION_BACKEND=FLASHINFER`). It can't run on an H200 in vLLM: its BF16 MoE needs a Blackwell-only kernel ([result and logs](data/smoke/day0-flint-b779895.json)). The detector wasn't live when flint launched, so this isn't a time-to-ready measurement yet.

## Pipeline

| stage | what | GPU | status |
|---|---|---|---|
| detect | every 30 min: new model-named `vllm/vllm-openai` tags, matched to a Hugging Face repo (24-lab watchlist, then HF search) | no | `.github/workflows/detect.yml`, `detect/poll.py`; starts a build per new launch |
| build | worker-vllm `FROM vllm/vllm-openai:<day-0 tag>`, pushed to `ghcr.io`. Default worker branch: [`day0-integration`](https://github.com/ashwinsreedhar28/worker-vllm/tree/day0-integration), upstream main plus the version-guard fix (#360) and model-code startup errors (#363) | no | `.github/workflows/build-day0.yml` |
| CPU checks | inside the built image: architecture resolution, transformers config, vLLM `ModelConfig`, tokenizer | no | `checks/cpu_check.py`, run by the build |
| smoke test | one request on a Serverless endpoint, GPU class picked from the weight size, starting from the model's vLLM recipe settings when one exists; on failure: `GPU_MEMORY_UTILIZATION=0.85` (first-request OOM), next GPU class (startup OOM), the attention backend the model's code asks for, or the B200 class when it asks for a Blackwell-only kernel (model-code errors, #363), `TRUST_REMOTE_CODE=true` only when allowed; GPU climbs capped by `max_gpu` | yes, capped | `.github/workflows/smoke.yml` (`pipeline/smoke.py`, `pipeline/recipes.py`), started by every dispatched build |
| pre-warm | torch.compile cache on a network volume per (model, GPU) | yes | planned |
| board | per launch: tag push time, model, build, CPU verdict, smoke test, tag → worker image | no | [`data/board.md`](data/board.md), regenerated on every poll and every recorded build |
| deploy recipe | per launch whose own model passed: the exact template and endpoint request bodies, image digest, GPU class, env | no | [`data/ready.json`](data/ready.json) |

## Run a build

Actions → **Build day-0 worker** → Run workflow, with a `vllm_tag` (e.g. `muse-spark-open`) and the Hugging Face `model` to check. The result lands in the run summary and as `ghcr.io/ashwinsreedhar28/worker-vllm:day0-<tag>`.

## Automatic smoke tests

After a dispatched build pushes an image and its CPU checks resolve the model, `build-day0.yml` starts `smoke.yml` on the pinned image with `gpu=auto, max_gpu=48gb`. Spend limits:

- Classes above 48 GB never start automatically. Run `smoke.yml` by hand with a larger `max_gpu`; those runs use the `gpu-large` environment, which can require an approval.
- `max_minutes` (default 40) across all attempts, one worker per endpoint, endpoints scaled to zero on exit.
- A rolling 24 h budget (`SMOKE_BUDGET_USD` repo variable, default $10) over the recorded runs, using the highest listed Serverless rate in each class.
- A model that needs `TRUST_REMOTE_CODE` stops there; rerun with `allow_remote_code` after reading the repo's code.

Setup: the repo secret `RUNPOD_API_KEY` (a key used only for this), and optionally required reviewers on the `gpu-large` environment.

## Run a smoke test by hand

Needs a Runpod API key (kept out of the repo, e.g. in `~/.runpod.env`). Spend is capped by `--max-minutes` and `--max-gpu`; endpoints are scaled to zero on exit and deleted on a pass.

```
python3 pipeline/smoke.py --env-file ~/.runpod.env \
  --image ghcr.io/ashwinsreedhar28/worker-vllm:day0-<tag>-<sha> --model <org/repo> \
  --gpu 24gb --max-gpu 48gb [--set KEY=VALUE ...]
```

The result goes to `data/smoke/<image tag>.json` and shows up on the board.

### Recipe settings

When vLLM has a recipe for the model (`models/<org>/<repo>.yaml` in [vllm-project/recipes](https://github.com/vllm-project/recipes), or under `recipes/` in vllm-project/vllm), the smoke test starts from its single-GPU settings: attention and MoE backends, KV-cache dtype, dtype, quantization, memory utilization, loading flags, and the recipe's env vars, with the GPU generation's override block applied (Hopper on the 141gb class, Blackwell on 180gb). Parallelism, context length, parsers and opt-in features are left for the real deploy and listed under `recipe.skipped` in the result. `--set` and the ladder's changes win over the recipe, and `--trust-remote-code` in a recipe is only noted. `--recipe off` skips it.

A recipe doesn't always exist on launch day: flint had none on recipes.vllm.ai a day later. The model-code rung covers that case. It reads the requirement in worker-vllm's startup error (#363) and retries with what it names, which is how flint gets from H200 to a passing B200 without a person (`tests/test_smoke.py::test_flint_climbs_to_flashinfer_then_b200`).
