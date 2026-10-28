# day0-ready

Goal: when an open-weight model drops, a tested Runpod Serverless vLLM worker for it exists within hours, not weeks.

## Status (Oct 28, 2026): parked until January 2027

- **Still running, at no cost:** every 30 minutes the detector looks for new model-named `vllm/vllm-openai` tags (dispatched by an external cron, since GitHub's own schedule never fired), builds worker-vllm from each one and runs CPU checks inside the image. Current state: [data/board.md](data/board.md).
- **Off until January:** GPU smoke tests after a build. Set the repo variable `AUTO_SMOKE=true` to turn them back on; anything else that uses a GPU is started by hand.
- **Flint, the first launch end to end.** [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) (40B MoE, Oct 21): vLLM pushed `vllm/vllm-openai:flint` at 16:09 UTC, the worker image was pushed at 20:05, and it answered on a Serverless B200 at 21:33 with `ATTENTION_BACKEND=FLASHINFER` ([result](data/smoke/day0-flint-b779895.json)); driven by hand, so not a time-to-ready measurement. On one B200 ([bench/flint-b200](bench/flint-b200/README.md)) Flint NVFP4 decodes at 1.76 ms/token for one stream and 5,406 output tok/s at 64 concurrent requests (8K context).
- **MiniMax-M3.1-Flash:** detected 19 s after vLLM pushed its image and built in 15 minutes, unattended. Its 233 GiB needs 2 GPUs; two smoke tests on 2× H200 didn't get it to answer ([docs/multi-gpu.md](docs/multi-gpu.md)).
- **Optimal config:** the sweep's pilot, Qwen3-8B on one L40S, recommends the startup plan plus eager mode: vLLM engine init 22.9 → 2.6 s at 420 output tok/s with 30 concurrent requests, 98% of the startup plan alone ([docs/optimal-config.md](docs/optimal-config.md)).
- **Upstream.** worker-vllm: [#361](https://github.com/runpod-workers/worker-vllm/pull/361) merged (models that need `TRUST_REMOTE_CODE` answer with a clear error instead of restart-looping); [#360](https://github.com/runpod-workers/worker-vllm/pull/360) (build from day-0 images) and [#363](https://github.com/runpod-workers/worker-vllm/pull/363) (clear errors when a model's code needs a specific backend or GPU) open. vLLM: [#63019](https://github.com/vllm-project/vllm/issues/63019), [#63020](https://github.com/vllm-project/vllm/issues/63020) (its two startup bugs fixed on vLLM's Flint PR [#62938](https://github.com/vllm-project/vllm/pull/62938)).

**January:** turn `AUTO_SMOKE` and `SWEEP_AFTER_SMOKE` on (a sweep after each single-GPU pass); a third MiniMax try with the weights staged on a network volume; pre-warm the compile cache per model and GPU; measure tag-to-tested-endpoint on the next single-GPU launch.

## Why: the gap today

`study/` measures it. In a hand-picked sample of 25 new-architecture releases in 2026, the median wait from public weights to a published `runpod/worker-v1-vllm` image whose vLLM lists the architecture is **24 days** (over the 23 with a vLLM final release; a registry proxy that can overstate the real serving delay). vLLM itself usually publishes a model-named image (`vllm/vllm-openai:gemma4`, `step-5-preview`, ...) on release day: 16 of the 18 such images were available on or before the recorded release date. The published worker can't build from those images because of a version check; [worker-vllm #360](https://github.com/runpod-workers/worker-vllm/pull/360) fixes that, and this pipeline builds from a branch that includes it.

## Pipeline

| stage | what | GPU | where |
|---|---|---|---|
| detect | every 30 min: new model-named `vllm/vllm-openai` tags, matched to a Hugging Face repo (24-lab watchlist, then HF search); starts a build per new launch | no | `detect.yml`, `detect/poll.py` |
| build | worker-vllm `FROM vllm/vllm-openai:<day-0 tag>`, pushed to `ghcr.io`, from [`day0-integration`](https://github.com/ashwinsreedhar28/worker-vllm/tree/day0-integration): upstream main plus #360, #363 and the boot report | no | `build-day0.yml` |
| CPU checks | inside the built image: architecture resolution, transformers config, vLLM `ModelConfig`, tokenizer | no | `checks/cpu_check.py` |
| smoke test | one request on a Serverless endpoint, starting from the model's vLLM recipe when one exists, climbing a ladder of fixes on failure (below); 2+ GPUs with `--gpu-count` | yes, capped | `smoke.yml`, `pipeline/smoke.py`, `pipeline/recipes.py` |
| board | per launch: tag push time, model, build, CPU verdict, smoke test, tag → worker image | no | [`data/board.md`](data/board.md) |
| deploy recipe | per launch whose own model passed: template and endpoint request bodies, image digest, GPU class, env | no | [`data/ready.json`](data/ready.json) |
| optimal config | after a pass: startup plan, eager mode, `MAX_NUM_SEQS`, memory utilization and batch budget, ranked on engine init and output tokens/s at the worker's full load; one recommended env block per model | yes, capped | `sweep.yml`, `pipeline/sweep.py`, [spec](docs/optimal-config.md) |
| pre-warm | torch.compile cache on a network volume per model and GPU | yes | planned |

## Run a build

Actions → **Build day-0 worker** → Run workflow, with a `vllm_tag` (e.g. `muse-spark-open`) and the Hugging Face `model`. The result lands in the run summary and as `ghcr.io/ashwinsreedhar28/worker-vllm:day0-<tag>`.

## Smoke tests

With `AUTO_SMOKE=true`, a build whose CPU checks resolve the model starts `smoke.yml` on the pinned image with `gpu=auto, max_gpu=48gb`. Spend limits:

- Classes above 48 GB never start automatically; run `smoke.yml` by hand with a larger `max_gpu` (the `gpu-large` environment, which can require an approval).
- `max_minutes` (default 40) across all attempts, one worker per endpoint, endpoints scaled to zero on exit and deleted on a pass.
- A rolling 24 h budget (`SMOKE_BUDGET_USD`, default $10) over the recorded runs, at the highest listed Serverless rate in each class; `--min-balance` refuses to start below an account balance.
- A model that needs `TRUST_REMOTE_CODE` stops there; rerun with `allow_remote_code` after reading the repo's code.

Setup: the repo secret `RUNPOD_API_KEY` (a key used only for this), and optionally required reviewers on `gpu-large`.

By hand (key kept out of the repo, e.g. in `~/.runpod.env`):

```
python3 pipeline/smoke.py --env-file ~/.runpod.env \
  --image ghcr.io/ashwinsreedhar28/worker-vllm:day0-<tag>-<sha> --model <org/repo> \
  --gpu 24gb --max-gpu 48gb [--set KEY=VALUE ...]
```

The result goes to `data/smoke/<image tag>.json` and the board, with the device the worker actually got (a 48 GB type id can be served by a MIG slice of an RTX PRO 6000 Blackwell). `--gpu-count 2` with an explicit `--gpu` runs one worker on 2 GPUs with `TENSOR_PARALLEL_SIZE=2`. For a model in the hundreds of gigabytes, `pipeline/stage.py` first downloads the weights onto a network volume with a CPU pod and checks the bytes against the Hub's file sizes, and `--volume-id` boots the smoke test from that volume ([docs/multi-gpu.md](docs/multi-gpu.md)).

The ladder on failure: `GPU_MEMORY_UTILIZATION=0.85` after a first-request OOM, the next GPU class after a startup OOM, the attention backend the model's code asks for, or the B200 class when it asks for a Blackwell-only kernel (worker-vllm #363's startup errors), and `TRUST_REMOTE_CODE=true` only when allowed. GPU climbs are capped by `max_gpu`.

### Recipe settings

When vLLM has a recipe for the model (`models/<org>/<repo>.yaml` in [vllm-project/recipes](https://github.com/vllm-project/recipes), or under `recipes/` in vllm-project/vllm), the smoke test starts from its single-GPU settings: attention and MoE backends, KV-cache dtype, dtype, quantization, memory utilization, loading flags and env vars, with the GPU generation's override block applied (Hopper on 141gb, Blackwell on 180gb). Parallelism, context length, parsers and opt-in features are left for the real deploy and listed under `recipe.skipped`. `--set` and the ladder win over the recipe; `--recipe off` skips it.

A recipe doesn't always exist on launch day: flint had none a day later. The model-code rung covers that: it reads the requirement in worker-vllm's startup error and retries with what it names, which is how flint gets from H200 to a passing B200 without a person (`tests/test_smoke.py::test_flint_climbs_to_flashinfer_then_b200`).

## Optimal-config sweep

Actions → **Optimal-config sweep** → Run workflow, with the pinned image and the model. Results land in `data/sweep/<image tag>/` (one file per config, `summary.json`, two JSONL tables for a lakehouse); a recommendation becomes a `recommended` block in `data/ready.json` and a column on the board. `python3 pipeline/sweep.py --plan ...` prints the configs and budget without calling Runpod; `python3 pipeline/predict.py --log <worker log>` picks the GPU class from one boot's weights and KV bytes per token. The image needs the worker's boot report (`DAY0_BOOT_REPORT=1`, in `day0-integration` since Oct 28). With the repo variable `SWEEP_AFTER_SMOKE=true`, a passing smoke test starts the sweep itself. Method, rules and the pilot's numbers: [docs/optimal-config.md](docs/optimal-config.md).
