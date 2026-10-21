# day0-lag study

How long after an open-weight model goes public does the Runpod Serverless vLLM worker's published image include a vLLM that recognizes the model's architecture, and which step holds it up?

Window: models made public between 2026-01-01 and 2026-10-21. 25 rows have a new architecture; 10 control rows reuse an architecture vLLM already had. Data as of 2026-10-21.

## What is measured

For each model, two events:

- **Upstream:** the first `vllm/vllm-openai:vX.Y.Z` final image whose `vllm/model_executor/models/registry.py` lists the architecture class from the model's `config.json`.
- **Worker:** the first moment the latest published `runpod/worker-v1-vllm` release image pins a vLLM tag whose registry lists that class. Support is checked against the registry at that exact pin, not by version ordering.

Registry membership is a proxy. The endpoint runs below show where it's right and where it isn't. "Days" means the UTC date of the image push minus the lab-local release date, so each figure carries about ±1 day of noise. Events are ordered by full push timestamp.

## Result

For a new architecture, the median wait from public release to a published worker image whose vLLM pin lists the architecture is **24 days** (mean 28.5, min 2, max 74, n=23). Two rows have no vLLM final release listing them yet and are censored at ≥6 days: Step-5-Preview and Muse Spark Open.

| step | median | mean | max | at 0 days |
|---|---|---|---|---|
| upstream: release → first vLLM final image listing the arch | 14 d | 11.9 | 27 (GLM-5.3-Flash) | 1 / 23 |
| worker: that image → latest published worker image pins a vLLM listing it | 11 d | 16.6 | 55 (Qwen3.5-397B) | 0 / 23 |

Most of the upstream half is release cadence, not missing code. On vLLM's main branch, the support PR merged a median of 0 days from release (min −41 for North-Mini-Code, max +8). vLLM also publishes model-named images for big launches, such as `vllm/vllm-openai:gemma4`, `deepseekv4` and `step-5-preview`. 18 of the 25 new-architecture models had one, and **16 of those 18 were available on or before the recorded release date** (15 pushed on the release's UTC date, Step-3.7-Flash's the evening before). Muse Glimmer's came a day later and GLM-5.3-Flash's 14 days later. The worker couldn't be built from those images because of a version check; that's fixed on a branch (see the top-level README).

The worker half has the long tail, and it's the half Runpod controls:

- **Freezes.** The worker image pinned vLLM 0.11.0 from November 2025 until 2026-02-12, and 0.16.0 from 2026-03-05 to 2026-05-01. On 245 of the 294 days of 2026 so far, a newer vLLM final image had been published than the one the latest worker image pinned. On those days the oldest such release was a median 20 days old (max 84), and the worker was a median 3 finals behind (max 8).
- **Rollbacks.** 6 of 23 models lost support after first getting it. For MiMo-V2.5-Pro, Laguna-XS.2, Mellum 2 and North-Mini-Code, the gap was 47 days: v2.22.4 rolled back to vLLM 0.20.2 on Jun 12, and v2.23.0 restored support on Jul 29. For Qwen3.8-Flash-Next and Hy4-preview it was 7 days: v2.27.1 rolled back from 0.29.0 to 0.28.0 on Sep 21, and v2.28.0 restored support on Sep 28.
- **Tags with no image.** `v.2.15.0` (note the extra dot) doesn't match the `release.yml` tag trigger, and no image exists for it. `v2.29.0` (vLLM 0.31.0, tagged Oct 6) has no image on Docker Hub either. A CI fix for runner disk space landed Oct 7, and nothing has been re-tagged since. The newest published image is **v2.28.0 (vLLM 0.30.0)**.

In the controls, 9 of 10 had a worker image listing their architecture on release day. The exception is Qwen3.6-35B-A3B, which waited 15 days because the Qwen3.5 architecture it reuses wasn't yet in the worker's frozen 0.16.0 pin.

## Checked on real endpoints

Six worker images were deployed on Runpod Serverless (single 24–48 GB GPU, `MAX_MODEL_LEN=4096`), each sent one chat request, and torn down (`scripts/validate_proxy.py`, results in `data/proxy_validation.csv`, worker logs in `data/logs/`). For each model, the table predicts the older image fails and the newer one loads.

| model | image | table | observed | cause |
|---|---|---|---|---|
| Qwen3.5-4B | `v2.14.0` (vLLM 0.16.0) | fail | fail | `Qwen3_5ForConditionalGeneration ... not supported for now` |
| Qwen3.5-4B | `v.2.16.0` (vLLM 0.17.1) | load | load, but not at defaults | default `GPU_MEMORY_UTILIZATION` (0.95) left no room for Triton autotuning of the linear-attention kernel: out of memory on the first request. Loaded and answered at 0.85. |
| Gemma 4 E4B | `v2.17.0` (vLLM 0.18.1) | fail | fail | the image's transformers doesn't know model type `gemma4` |
| Gemma 4 E4B | `v2.18.0` (vLLM 0.19.1) | load | load | answered "ok" |
| K2-Horizon-7B | `v2.27.2` (vLLM 0.28.0) | fail | **load** | with `TRUST_REMOTE_CODE=true`; vLLM 0.28.0 has no native K2-Horizon class, so it ran the repo's own modeling code through the Transformers backend |
| K2-Horizon-7B | `v2.28.0` (vLLM 0.30.0) | load | load | needs `TRUST_REMOTE_CODE=true` |

What this means for the table:

- The failure side holds. Every predicted failure failed at startup, for architecture or config reasons.
- "Listed in the registry" can still fail at default settings (Qwen3.5), so a day-0 pipeline needs a fallback ladder, not one try.
- The table overstates the wait for models that ship their own modeling code. K2-Horizon was probably servable on its release day: the latest worker image then, `v2.26.0`, pins the same vLLM 0.28.0. 8 of the 25 new-architecture repos ship modeling code (`auto_map` with `AutoModel*` in `data/hf_configs.csv`): Kimi-K2.5, Kimi-K3, Step-3.5-Flash, Step-3.7-Flash, MiMo-V2.5-Pro, Laguna-XS.2, Ling-3.0-flash and K2-Horizon. Whether the fallback works for each, and how fast it is, isn't measured.
- Muse Spark Open shows the same blind spot from the vLLM side. vLLM's main branch has Muse Spark kernels, parsers and a release-day image, but no Muse Spark class in its registry; it appears to load through the Transformers backend.
- 6 of the 25 configs were written with an unreleased transformers (`transformers_version` ending in `.dev0` or `rc0`: GLM-4.7-Flash, GLM-5, Qwen3.5, Gemma 4, Muse Glimmer, Qwen3.8-Flash-Next). For those, the image's transformers matters as much as its vLLM, as the Gemma 4 failure on `v2.17.0` shows.

## Compile cache (from the code; no GPU run)

Both vLLM compile-cache layouts key on `VllmConfig.compute_hash()`, which includes `vllm.__version__` by default (checked at v0.23.0, v0.28.0 and v0.32.0):

- The backend cache lives at `VLLM_CACHE_ROOT/torch_compile_cache/<hash of env, config, traced code, compiler>`.
- The AOT cache lives at `VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile/<hash of env, config, model>`. This is the path PR #351's cache hits use.

So the first start on a vLLM version the volume hasn't cached before misses and compiles from scratch. A version cached earlier can still hit, and an explicit `compilation_config.cache_dir` overrides the generated path. The worker's `compile_cache.py` never prunes old directories, so each new version adds one per model and config.

PR #351 measured 108–133 s uncached vs 68–71 s cached on one setup (4090, Qwen3-8B). The 37–65 s difference shows the size of the cost; it was not measured across an upgrade. PR #351 itself (commit `7b8958e`) is only in `v2.29.0`, which has no published image.

## Caveats

- Even existing-architecture launches (GLM-5.1, GLM-5.2) got model-named images, for kernels and tool/reasoning parsers. Registry membership doesn't capture that.
- "Latest published worker image" is the newest release tag on Docker Hub `runpod/worker-v1-vllm`. I can't see which tag the console's quick deploy uses, or any internal images.
- Release dates use the weights-public date for staged releases (API first, weights later: Kimi-K3, MiniMax-M3, Step-5-Preview, Muse Spark Open). Ling-3.0-flash uses Aug 4, the date of the first public activity on its HF repo; news reports say Aug 5–7. HF `createdAt` isn't used, because repos are often created privately weeks earlier (Kimi-K2.5: 26 days).
- vLLM and worker image times are Docker Hub `tag_last_pushed`. A later re-push of a tag would make it look later than it shipped. Per-tag registry reads catch cherry-picks: Gemma 4 is in v0.19.0 even though that branch was cut Mar 29.
- The model set is hand-picked: major-lab text-generation releases from vLLM's 2026 new-architecture list, plus Meta's two Muse releases, which were added after a gap check. Day-0 images were found by tag-name substring search, so a missing name doesn't prove no image existed.
- Endpoint checks cover three small models only. The large models (Kimi K3, DeepSeek V4, GLM-5, Muse Spark Open) need multi-GPU endpoints.

## Files

- `data/models.csv`: per-model inputs (repo, architecture, release date and source, vLLM PR and merge date)
- `data/hf_configs.csv`: `auto_map` and `transformers_version` from each new-architecture repo's `config.json`
- `data/vllm_registry_by_tag.json`: architecture classes at every vLLM final tag (`scripts/vllm_registry_by_tag.py`)
- `data/vllm_images.csv`: Docker Hub push time of each `vllm/vllm-openai:vX.Y.Z`
- `data/worker_tags.csv`: vLLM pin at every worker-vllm git tag
- `data/worker_images.csv`: Docker Hub push time of each worker release image
- `data/vllm_day0_images.csv`: earliest model-named `vllm/vllm-openai` tag per model
- `data/lag_table.csv`, `data/summary.txt`: output of `python3 scripts/build_table.py`
- `data/proxy_validation.csv`, `data/logs/`: the endpoint runs

To regenerate: clone vllm-project/vllm into `src/vllm`, run `python3 scripts/vllm_registry_by_tag.py src/vllm`, then `python3 scripts/build_table.py`.
