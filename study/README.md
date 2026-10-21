# day0-lag

How long after an open-weight model goes public does the Runpod Serverless vLLM worker's published image include a vLLM that recognizes the model's architecture, and which step holds it up?

Window: models made public between 2026-01-01 and 2026-10-21. 23 rows have a new architecture; 10 control rows reuse an architecture vLLM already had. Data as of 2026-10-21.

## What is measured

For each model, two events:

- **Upstream:** the first `vllm/vllm-openai:vX.Y.Z` final image whose `vllm/model_executor/models/registry.py` lists the architecture class from the model's `config.json`.
- **Worker:** the first moment the latest published `runpod/worker-v1-vllm` release image pins a vLLM tag whose registry lists that class. Support is checked against the registry at that exact pin, not by version ordering.

Registry membership is necessary for serving but not sufficient. Nothing here builds a worker, loads weights or runs inference. "Days" means the UTC date of the image push minus the lab-local release date, so each figure carries about ±1 day of noise. Events are ordered by full push timestamp.

## Result

For a new architecture, the median wait from public release to a published worker image whose vLLM pin lists the architecture is **24.5 days** (mean 29.0, min 2, max 74, n=22). Step-5-Preview has no vLLM final release yet, 6 days after release, so it is censored at ≥6.

The wait splits into two roughly equal halves:

| step | median | mean | max | at 0 days |
|---|---|---|---|---|
| upstream: release → first vLLM final image listing the arch | 13 d | 11.7 | 27 (GLM-5.3-Flash) | 1 / 22 |
| worker: that image → latest published worker image pins a vLLM listing it | 12.5 d | 17.3 | 55 (Qwen3.5-397B) | 0 / 22 |

Most of the upstream half is release cadence, not missing code. On vLLM's main branch, the support PR merged a median of 0 days from release (min −41 for North-Mini-Code, max +8). vLLM also publishes model-named images for big launches, such as `vllm/vllm-openai:gemma4`, `deepseekv4` and `step-5-preview`. 16 of the 23 new-arch models had one, and **15 of those 16 were pushed on release day**; GLM-5.3-Flash's came 14 days later. So upstream already publishes day-0 images. The worker isn't built from them today.

The worker half has the long tail, and it is the half Runpod controls:

- **Freezes.** The worker image pinned vLLM 0.11.0 from November 2025 until 2026-02-12, and 0.16.0 from 2026-03-05 to 2026-05-01. On 245 of the 294 days of 2026 so far, a newer vLLM final image had been published than the one the latest worker image pinned. On those days the oldest such release was a median 20 days old (max 84), and the worker was a median 3 finals behind (max 8).
- **Rollbacks.** 6 of 22 models lost support after first getting it. For MiMo-V2.5-Pro, Laguna-XS.2, Mellum 2 and North-Mini-Code, the gap was 47 days: v2.22.4 rolled back to vLLM 0.20.2 on Jun 12, and v2.23.0 (0.23.0) restored support on Jul 29. For Qwen3.8-Flash-Next and Hy4-preview, it was 7 days: v2.27.1 rolled back from 0.29.0 to 0.28.0 on Sep 21, and v2.28.0 (0.30.0) restored support on Sep 28.
- **Tags with no image.** `v.2.15.0` (note the extra dot) doesn't match the `release.yml` tag trigger, and no image exists for it. `v2.29.0` (vLLM 0.31.0, tagged Oct 6) has no image on Docker Hub either. A CI fix for runner disk space landed Oct 7, and nothing has been re-tagged since. The newest published image is **v2.28.0 (vLLM 0.30.0)**. Two vLLM finals have been published since, and main now pins 0.32.0.

In the controls, 9 of 10 had a worker image listing their architecture on release day. The exception is Qwen3.6-35B-A3B, which waited 15 days: the Qwen3.5 architecture it reuses wasn't yet in the worker's frozen 0.16.0 pin.

## Compile cache (Phase 2, from the code; no GPU run)

Both vLLM compile-cache layouts key on `VllmConfig.compute_hash()`, which includes `vllm.__version__` by default (checked at v0.23.0, v0.28.0 and v0.32.0):

- The backend cache lives at `VLLM_CACHE_ROOT/torch_compile_cache/<hash of env, config, traced code, compiler>`.
- The AOT cache lives at `VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile/<hash of env, config, model>`. This is the path PR #351's cache hits use.

So the first start on a vLLM version the volume hasn't cached before misses and compiles from scratch. A version cached earlier can still hit, and an explicit `compilation_config.cache_dir` overrides the generated path. The worker's `compile_cache.py` never prunes old directories, so each new version adds one per model and config.

PR #351 measured 108–133 s uncached vs 68–71 s cached on one setup (4090, Qwen3-8B). The 37–65 s difference shows the size of the cost; it was not measured across an upgrade.

The PR #351 change (commit `7b8958e`) is only in `v2.29.0`, which has no published image.

## What this points to

1. **Let the worker build from vLLM's model-named images.** `VLLM_VERSION` is already a build arg. Dockerfile line 25, `test "$(python3 -c 'import vllm; print(vllm.__version__)')" = "${VLLM_VERSION#v}"`, rejects any non-semver base tag such as `step-5-preview`. It was added Aug 27 (#335) to catch the bitsandbytes plugin silently downgrading vLLM. Keep the check for `vX.Y.Z` tags and skip or relax it otherwise. That would make a day-0 build possible for the 15 launches that had a release-day image. Whether each of those builds loads and serves its model is untested.
2. **Bump cadence plus a regression gate.** Most of the worker tail is freezes and rollbacks, not slow bumps. `serverless-model-tests.yml` already exists. Next, read what it covers and whether it would have caught the 0.22.1 and 0.29.0 regressions before release.
3. **Pre-warm the compile cache when a vLLM version ships**, for each (popular model, vLLM version, GPU arch), since each new version starts with an empty cache.

## Caveats

- Even existing-arch launches (GLM-5.1, GLM-5.2) got model-named images, for kernels and tool/reasoning parsers. Registry membership doesn't capture that.
- "Latest published worker image" is the newest release tag on Docker Hub `runpod/worker-v1-vllm`. I can't see which tag the console's quick deploy uses or any internal images.
- Release dates use the weights-public date for staged releases (API first, weights later: Kimi-K3, MiniMax-M3, Step-5-Preview). Ling-3.0-flash uses Aug 4, the date of the first public activity on its HF repo; news reports say Aug 5–7. HF `createdAt` isn't used, because repos are often created privately weeks earlier (Kimi-K2.5: 26 days).
- vLLM and worker image times are Docker Hub `tag_last_pushed`. A later re-push of a tag would make it look later than it shipped. Per-tag registry reads catch cherry-picks: Gemma 4 is in v0.19.0 even though that branch was cut Mar 29.
- The model set is hand-picked: major-lab text-generation releases from vLLM's 2026 new-architecture list. Day-0 images were found by tag-name substring search, so a missing name doesn't prove no image existed.
- The transformers-backend fallback (`--model-impl transformers`) wasn't measured.

## Files

- `data/models.csv`: per-model inputs (repo, architecture, release date and source, vLLM PR and merge date)
- `data/vllm_registry_by_tag.json`: architecture classes at every vLLM final tag (`scripts/vllm_registry_by_tag.py`)
- `data/vllm_images.csv`: Docker Hub push time of each `vllm/vllm-openai:vX.Y.Z`
- `data/worker_tags.csv`: vLLM pin at every worker-vllm git tag
- `data/worker_images.csv`: Docker Hub push time of each worker release image
- `data/vllm_day0_images.csv`: earliest model-named `vllm/vllm-openai` tag per model
- `data/lag_table.csv`, `data/summary.txt`: output of `python3 scripts/build_table.py` (per-model status, lags, outage interval)

To regenerate: clone vllm-project/vllm into `src/vllm`, run `python3 scripts/vllm_registry_by_tag.py src/vllm`, then `python3 scripts/build_table.py`.
