# day0-ready

Goal: when an open-weight model drops, a tested Runpod Serverless vLLM worker for it exists within hours, not weeks.

## Status (Oct 22, 2026)

- **Running since Oct 21, 19:28 UTC.** Every 30 minutes the detector looks for new model-named `vllm/vllm-openai` tags, builds worker-vllm from each one, runs CPU checks inside the image, and smoke-tests it on a Runpod Serverless endpoint (automatically on GPUs up to 48 GB; larger ones are started by hand). Polls are dispatched by an external cron because GitHub's own schedule never fired. Current state: [data/board.md](data/board.md).
- **First launch: Flint.** [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), a 40B MoE from Recursive, released Oct 21. vLLM pushed `vllm/vllm-openai:flint` at 16:09 UTC, the worker image built from it was pushed at 20:05, and it answered a chat request on a Serverless B200 at 21:33 with one setting, `ATTENTION_BACKEND=FLASHINFER` ([result](data/smoke/day0-flint-b779895.json)). That run was driven by hand before the detector was live, so it isn't a time-to-ready measurement. With the Oct 21 image Flint can't start on H200 (its BF16 MoE asks for a Blackwell-only kernel); on Oct 22 vLLM's Flint PR added a fallback to standard layers on other GPUs, not yet tested here.
- **Under load on one B200** ([bench/flint-b200](bench/flint-b200/README.md)): At 8K context Flint NVFP4 decodes at 1.76 ms/token for a single stream and reaches 5,406 output tok/s with 64 concurrent requests. In BF16 against Qwen3.6-35B-A3B on the same setup, Flint is 1.3–1.4× faster for a single stream, and the two are even at 32K context with 32+ concurrent requests.
- **Upstream.** worker-vllm: [#361](https://github.com/runpod-workers/worker-vllm/pull/361) merged (models that need `TRUST_REMOTE_CODE` answer with a clear error instead of restart-looping), [#360](https://github.com/runpod-workers/worker-vllm/pull/360) (build from day-0 images) and [#363](https://github.com/runpod-workers/worker-vllm/pull/363) (clear errors when a model's own code needs a specific backend or GPU) open. vLLM: [#63019](https://github.com/vllm-project/vllm/issues/63019) and [#63020](https://github.com/vllm-project/vllm/issues/63020), both from the B200 benchmark; #63020's two startup bugs are fixed on vLLM's Flint PR [#62938](https://github.com/vllm-project/vllm/pull/62938).

## Why: the gap today

`study/` measures it. In a hand-picked sample of 25 new-architecture releases in 2026, the median wait from public weights to a published `runpod/worker-v1-vllm` image whose vLLM lists the architecture is **24 days** (over the 23 with a vLLM final release; a registry proxy that can overstate the real serving delay). vLLM itself usually publishes a model-named image (`vllm/vllm-openai:gemma4`, `step-5-preview`, ...) on release day: 16 of the 18 such images were available on or before the recorded release date. The published worker can't build from those images because of a version check; [worker-vllm #360](https://github.com/runpod-workers/worker-vllm/pull/360) fixes that, and this pipeline builds from a branch that includes it.

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
| optimal config | after a pass: a bounded sweep (startup plan, eager, `max_num_seqs`, memory utilization, batch budget) ranked on engine init time and output tokens/s at the worker's full load, giving a recommended env block per model by day 1–2 | yes, capped | spec: [`docs/optimal-config.md`](docs/optimal-config.md); runner: `pipeline/sweep.py`, `sweep.yml`; pilot pending |

## Optimal config

The smoke test records the first configuration that answers a request, not the best one: flint's recipe is one setting at a 4K context with no network volume. [`docs/optimal-config.md`](docs/optimal-config.md) specifies the next step, a sweep that runs after a pass, and `pipeline/sweep.py` runs it:

- 6–7 configurations, one change at a time: the startup plan (`VLLM_ENABLE_STARTUP_PLAN`), eager mode, `MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`, and the batch budget where vLLM's default is small, then the changes that beat the startup-plan config on their own, combined.
- Each gets 3 cold starts, ranked on vLLM's own engine-init time from the worker's boot report (`DAY0_BOOT_REPORT=1`, on worker-vllm's `day0-sweep` branch for now). It also gets one throughput run at 30 concurrent requests through the endpoint's OpenAI route, ranked on output tokens/s.
- Only configurations that fit count: no out-of-memory relaunch, no failed or preempted requests, and enough KV cache for 30 requests.
- Each sweep has a worker-minute cap, and the runner checks the account balance before it starts.

Actions → **Optimal-config sweep** → Run workflow, with the pinned image and the model. Results land in `data/sweep/<image tag>/` (one file per config, `summary.json`, and two JSONL tables for a lakehouse), and a recommendation becomes a `recommended` env block in `data/ready.json` and a column on the board. `python3 pipeline/sweep.py --plan ...` prints the configs and the budget without calling Runpod, and `python3 pipeline/predict.py --log <worker log>` picks the GPU class from one boot's weights and KV bytes per token. The first run is a pilot on Qwen3-8B on one 48 GB GPU, to check the runner before any real launch.

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
