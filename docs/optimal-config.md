# Optimal-config sweep (spec)

Status: spec v1, 2026-10-27; pilot complete 2026-10-28. The runner (`pipeline/sweep.py`) covers S0–S5 and W; Q and L come later. The pilot (Qwen3-8B on one L40S, run 7) finished all seven configs and recommends S2, the startup plan plus eager mode: engine init 22.9 → 2.6 s at 98% of S1's throughput. Sweeps of real day-0 launches, B200 and multi-GPU start in January 2027.

## Why

The day-0 smoke test records the **first** configuration that answers a request, not the best one. Flint's entry in `data/ready.json` is `ATTENTION_BACKEND=FLASHINFER` at `MAX_MODEL_LEN=4096`, with no network volume, on whichever GPU in a class list was free. That proves the model serves. It says nothing about how fast a worker starts or how much it serves once started.

This sweep runs **after** a smoke pass and produces a recommended configuration for each model by day 1–2: an env block the worker reads, plus the measurements behind it.

## Scope

- One sweep = one model × one GPU type × one worker image digest.
- It starts only after the model's smoke test passed on that image.
- The image has to carry the worker's boot report ([below](#boot-report)). Until that is on `day0-integration`, the worker branch the board's images come from, a sweep uses a `day0-sweep` build of the same vLLM tag, and its S0 warm-up doubles as that image's smoke test.
- Single GPU. Multi-GPU is described at the end and waits for multi-GPU smoke tests.
- It measures speed, not output quality. Configurations that change outputs (quantization, KV-cache precision, speculative decoding) are out of scope unless the model's publisher ships the variant (config Q).

## What is swept, and what isn't

Two things vary. One is a gate.

- **Cold start**: how long vLLM takes to get from launch to ready. On Serverless every scale-from-zero pays it. The ranked part is vLLM's engine init (memory profiling, KV-cache setup, warm-up and CUDA-graph capture), because that is what the settings below change. Weight loading is reported but not ranked: it depends on the network volume and the host's page cache, not the config (5.8–293 s for the same file in [vllm-coldstart](https://github.com/ashwinsreedhar28/vllm-coldstart) step 2).
- **Throughput at the worker's full load**: output tokens/s with as many requests in flight as one worker accepts (`MAX_CONCURRENCY`, 30 by default in worker-vllm's `main.py`). Serverless never sends one worker more than that, so it's the load point that matters.
- **Fit** (gate): with `MAX_MODEL_LEN` at the target context, a configuration either holds 30 requests of the bench shape (4,096 in, 512 out) at once, or it's excluded. The checks are in [Measurements](#fit).

Not swept, and why:

| knob | why it's left out |
|---|---|
| loader strategy (`runai_streamer`, `prefetch`, `eager` load) | inside page-cache noise in vllm-coldstart step 2 |
| `RUNPOD_SKIP_AUTO_SYSTEM_CHECKS` | worker-vllm runs `vllm serve` as a subprocess, so the SDK's CUDA check in the handler can't force vLLM to spawn; it matters only for in-process handlers |
| KV-cache fp8, quantization, speculative decoding | change outputs; need a quality eval this spec doesn't have |
| `TENSOR_PARALLEL_SIZE` > 1 for a model that fits one GPU | more GPUs per worker is a cost decision, not a config one |
| compile cache on the volume | a precondition (worker-vllm #351), on in every config |

## Fixed conditions (every config)

- The same image, by digest. The same single GPU type id on the endpoint (not a class list): vLLM's compile cache and startup plan are keyed by device, so a run on a different GPU than the warm-up misses both. Runpod can serve a type id on another device (the pilot asked for an RTX 6000 Ada and got a worker reporting "NVIDIA RTX PRO 6000 Blackwell Server Edition"), so the runner records the device the first warm-up reports and rejects any later start on a different one. The device is nvidia-smi's name plus the MIG profile when the worker got a slice (pilot runs 1 and 4: 2g.48gb slices on L40S and RTX 6000 Ada type ids). The REST API doesn't take MIG type ids for an endpoint (run 6), so a slice can't be requested directly.
- A network volume in a data center that has that GPU. Weights live under a shared `HF_HOME` on the volume. Each config gets its own `VLLM_CACHE_ROOT` on the volume, so one config's compile cache or startup plan can't leak into another's (vLLM's startup-plan key ignores `gpu_memory_utilization`, vllm#60143).
- `MAX_MODEL_LEN` = the model's recipe `context_length`, capped at 32,768. Without a recipe, the config's `max_position_embeddings`, same cap.
- Endpoint: `workersMax` toggled between 0 and 1 by the runner, `idleTimeout` 30 s, FlashBoot off, `MAX_CONCURRENCY` 30. Cold starts don't depend on the idle timeout (the runner sets `workersMax` to 0 between them); it's there so an endpoint left behind scales itself down.
- `DAY0_BOOT_REPORT=1` (see [Boot report](#boot-report)).

## Configurations

One change at a time from S1, the way vllm-coldstart's configs A–F were run, then one run that combines the winners.

| id | env on top of the fixed conditions | what it tests |
|---|---|---|
| **S0** | the smoke test's passing env (e.g. flint: `ATTENTION_BACKEND=FLASHINFER`) | day-0 settings under sweep conditions |
| **S1** | S0 + `VLLM_ENABLE_STARTUP_PLAN=1` | reuse the saved memory profile (4.6 s off engine init for Qwen3-8B in vllm-coldstart) |
| **S2** | S1 + `ENFORCE_EAGER=true` | skip CUDA-graph capture: faster start, slower decode |
| **S3** | S1 + `MAX_NUM_SEQS=32` | vLLM captures CUDA graphs for batches up to `min(2 × max_num_seqs, 512)` (1,024 on B200), and its default `max_num_seqs` is 256 (1,024 on H100, H200 and B200; A100 80GB keeps 256), so the default captures the full 512 or 1,024. 32 still covers `MAX_CONCURRENCY` 30, with 64 as the largest graph: less to capture, less memory held for graphs |
| **S4** | S1 + `GPU_MEMORY_UTILIZATION=0.95` | more KV cache vs less headroom (vLLM's default is 0.92; 0.95 is the Hub template's default and what made Qwen3.5 run out of memory on its first request in the lag study) |
| **S5** (conditional) | S1 + `MAX_NUM_BATCHED_TOKENS=8192` | vLLM's server default is 2,048 tokens per step on GPUs under 70 GB and on A100, so a 4,096-token prompt takes two steps; 8,192 is what it already uses on H100/H200 (16,384 on B200). Runs only when that default is below 8,192: from the launch log's `max_num_batched_tokens=` line, or, when vLLM doesn't print it (v0.32.0 didn't in the pilot), from vLLM's per-GPU defaults and the memory the worker reports |
| Q (optional) | S1 with the recipe's vendor-official FP8/NVFP4 checkpoint | fit and speed of the official quantized variant; quality not evaluated |
| L (optional) | S1 + `MAX_NUM_BATCHED_TOKENS` = 2× the launch-log default, benched with 32K-token prompts | long-context models only |
| **W** | S1 + every change that wins on its own (defined below) | confirms the changes combine |

Q runs only when the model's recipe lists a variant from the model's own publisher. L runs only when a person asks for it, for a model whose recipe has a `long_context` variant (recipes keep long context opt-in).

A change **wins on its own** when its config fits and beats S1 on one ranked number by more than the margin (engine init by the noise band in [Picking the recommendation](#picking-the-recommendation), or C=30 output tokens/s by more than 5%) without losing on the other by more than the same margin. W runs only when two or more changes win; with one, that config already is W.

## Measurements

### Boot report

A worker started with `DAY0_BOOT_REPORT=1` keeps a small record of its own boot: the times it launched vLLM and saw it healthy, whether it had to relaunch vLLM with a smaller memory footprint, and these lines from vLLM's log:

- `Model loading took X GiB memory and Y seconds`
- `init engine (profile, create kv cache, warmup model) took Z s`
- `torch.compile took … s`, `Graph capturing finished in … secs`
- `Available KV cache memory: … GiB`, `GPU KV cache size: N tokens`, `max_num_batched_tokens=…`

A job with input `{"day0_boot_report": true}` returns that record instead of running a request. Without it the numbers that matter are only in the console's worker logs. This is on worker-vllm's `day0-sweep` branch, off by default and tested, to be merged into `day0-integration` before the January sweeps. The same branch makes the route passthrough return non-JSON replies as text, so a `{"route": "/metrics"}` job returns vLLM's Prometheus counters.

### Cold start

Per config: one warm-up start, discarded (it downloads nothing new but fills the config's compile cache and startup plan), then **3 measured cold starts**. Before each, the runner sets `workersMax` to 0, waits until no worker is left, sets it back to 1, and sends the boot-report job.

- **Ranked:** engine init seconds (median of 3).
- **Reported:** model loading seconds, vLLM launch → healthy seconds, and the job's `delayTime` and `executionTime` from the Serverless API. worker-vllm starts the Serverless SDK only after vLLM is healthy, so `delayTime` is the cold start a user sees: scheduling, image pull, container start and the whole vLLM boot. `delayTime` varied 13.8–79.1 s across 26 runs in vllm-coldstart, more than any config changes, so it's context, not a ranking.

### Throughput

Per config, once, on a warm worker. Back-to-back requests keep it up; there's no `workersMin`, so nothing stays on if the runner dies.

- The prompts are 128 distinct sequences of fixed token ids (random, seeded, the same set for every config), so every request is exactly 4,096 input tokens without a tokenizer, and no prompt repeats on a worker, so prefix caching never hits. Each asks for 512 output tokens with `ignore_eos`, streamed.
- The first warm-up's worker gets one short streamed request that checks the route takes token-id prompts and streams token by token. If it doesn't, the sweep stops before paying for anything else.
- **C=1**, 8 requests (prompts 1–8), streamed through the endpoint's OpenAI-compatible route (`/openai/v1/completions`), the way users call it: TTFT and TPOT p50 (time per output token).
- **C=30**, 120 requests (prompts 9–128) as queued jobs through the Serverless jobs API, not streamed; the worker runs 30 at a time (`MAX_CONCURRENCY`): **output tokens/s** (ranked), failed requests, and vLLM's own mean TTFT and TPOT from its histograms. The window runs from the first submit to the last job's end (submit time + `delayTime` + `executionTime`). Not the OpenAI route: in the pilot, 30 concurrent streams through it ended after about 118 of their 512 tokens while vLLM's counter shows it generated nearly all of them (see [Pilot](#pilot)), so a streamed C=30 measures the route, not the config.
- After each level, the runner reads vLLM's `/metrics` through the worker's route passthrough (`{"route": "/metrics"}`) and cross-checks generated tokens and preemptions against the client's count.
- C=1's TTFT and TPOT include Runpod's gateway, so they're reported, not ranked.

### Fit

A configuration fits only if all three hold:

1. The boot report shows no relaunch with a smaller memory footprint. The worker does that silently when vLLM runs out of memory at startup, changing two settings the sweep didn't ask for.
2. The C=30 run has 0 failed requests and vLLM reports 0 preemptions.
3. vLLM's KV-cache size is at least 30 × 4,608 = 138,240 tokens, so 30 requests of the bench shape fit at once.

A configuration that doesn't fit is recorded as such and can't be recommended.

## Picking the recommendation

1. Candidates: the S and W configurations that fit. Q and L (a different checkpoint, a different prompt shape) are reported beside the result, never as candidates.
2. Throughput floor: 95% of the best C=30 output tokens/s among candidates.
3. Among candidates at or above the floor, recommend the one with the lowest median engine init. Differences smaller than the noise band count as ties: the larger of 2 s and the spread (max − min) of S1's three runs. Ties go to the config with fewer env changes from S0.
4. If the fastest-starting candidate overall is below the throughput floor (usually S2, eager), record it next to the recommendation as a **fast-start profile**, with both numbers. Scale-from-zero-heavy endpoints may prefer it.
5. If S1's three engine-init times spread by more than 20% of their median, the model is marked **inconclusive** and nothing is recommended. That points to a host or volume problem, and more runs would just spend money on noise.

6. A sweep that stopped early (budget, no GPU, an error) compared fewer configs than planned: its verdict is **incomplete**, and it reports its best config so far instead of a recommendation.

Throughput is one run per config, and the result says so.

## Picking the GPU class before a sweep

`pipeline/predict.py` picks the class from two numbers vLLM prints at every boot, so one smoke test anywhere predicts every class: the weights ("Model loading took X GiB") and KV bytes per token (available KV GiB ÷ KV tokens). KV tokens on a class = (device memory × 0.92 − weights − 3 GiB) ÷ KV bytes per token; the sweep goes on the cheapest class where that reaches 138,240. Checked against the recorded boots: Qwen3-8B predicts 168,333 tokens on an L40S (measured 172,416) and doesn't fit the bench on 24 GB; flint predicts 484,828 on an H200 (measured 486,702) and doesn't fit on 80 GB. KV bytes per token didn't depend on the device (147,470 on the L40S, 147,480 on the Blackwell slice). A `config.json` estimate is offered only for plain full attention; MLA, hybrid and sliding-window configs need a boot. Two models with the same weights and KV bytes per token (within 2%) count as the same shape, so a fine-tune can start from its base model's sweep.

## Stop rules and budget

- Per config: 1 warm-up + 3 cold starts + 1 throughput run. A config that fails to start gets one more cold start; if that fails too, it's recorded as failed and the sweep moves on. Failed configs' endpoints are deleted like the rest; the record keeps their boot report and the worker's startup-error text.
- If S1 fails to start, the sweep stops: S2–S5 and W all build on it.
- If no worker gets past `throttled` within 10 minutes, the start waits 2 minutes and tries once more, then the sweep stops (no GPU of that type free in the volume's data center). If health showed no worker at all, not even `throttled`, the retry goes to a new endpoint with the same env and cache root (pilot runs 2 and 3 both stalled that way on S1's fourth start). A start that never got a worker costs nothing.
- A step starts only if its worst case (a start that times out, then scale-down) fits before `--max-hours`.
- A workflow step that always runs, after the sweep, deletes anything left with the `day0-sweep-` prefix (endpoints, templates, volumes), even after a cancelled or crashed run.
- The runner estimates spend as worker-alive minutes × the class's highest listed Serverless rate (`pipeline/sizing.py`) and stops when the next step would pass the cap.

| GPU | rough cost per config | cap per model |
|---|---|---|
| ≤ 48 GB (e.g. Qwen3-8B on an L40S) | ~20 min, ~$0.60 | 180 GPU-min (~$5) |
| 80 GB | ~25 min | 180 GPU-min (~$14) |
| 141 GB (H200) | ~25 min | 180 GPU-min (~$18) |
| 180 GB (B200; flint's first loads on an H200 took 154–190 s) | ~30 min | 240 GPU-min (~$35) |
| multi-GPU | above × GPU count | per-sweep approval |

- The network volume is billed monthly: delete it when the sweep is done, or keep one per data center for all sweeps there.

## Outputs

- `data/sweep/<image tag>/<config>.json`: every run (boot report, job timings, bench results, metrics deltas, cost).
- `data/sweep/<image tag>/summary.json`: one row per config, the fit verdict, the recommendation and any fast-start profile.
- `data/sweep/<image tag>/config_runs.jsonl` and `config_summary.jsonl`: the two lakehouse tables below, one JSON object per row.
- `data/ready.json`: each entry gains a `recommended` block next to the day-0 one. It's the same shape as the template env the worker already reads, so it can be pasted into a Runpod template:

```json
"recommended": {
  "config": "S3",
  "image": "ghcr.io/ashwinsreedhar28/worker-vllm:day0-<tag>-<sha>",
  "digest": "sha256:…",
  "gpu_type": "NVIDIA L40S",
  "gpu_count": 1,
  "env": {"MODEL_NAME": "Qwen/Qwen3-8B", "MAX_MODEL_LEN": "32768",
          "VLLM_ENABLE_STARTUP_PLAN": "1", "MAX_NUM_SEQS": "32"},
  "measured": {"engine_init_s_p50": 0.0, "output_tok_s_c30": 0.0},
  "sweep": "data/sweep/<image tag>/summary.json"
}
```

- `data/board.md`: a "recommended" column with the config id and its two headline numbers.
- For [serverless-lakehouse](https://github.com/ashwinsreedhar28/serverless-lakehouse):
  - silver `config_runs`, one row per cold start or throughput run: `sweep_id, model, image_digest, gpu_type, config_id, run_kind (warmup|cold|bench), run_index, started_at, delay_s, execution_s, process_to_healthy_s, launch_to_healthy_s, engine_init_s, model_load_s, kv_cache_tokens, relaunched, concurrency, requests, failed, output_tok_s, ttft_p50_ms, tpot_p50_ms, est_cost_usd`
  - gold `config_summary`, one row per model × config: `sweep_id, model, gpu_type, config_id, env_json, fits, engine_init_s_p50, engine_init_s_min, engine_init_s_max, model_load_s_p50, output_tok_s_c30, tpot_p50_ms_c1, recommended, profile (recommended|fast_start|none), verdict (ok|inconclusive|incomplete|failed)`

## Multi-GPU (January, after multi-GPU smoke tests exist)

Multi-GPU smoke tests now exist as `smoke.py --gpu-count N`, with one proof of concept; see [`docs/multi-gpu.md`](multi-gpu.md).

MiniMax-M3.1-Flash is the live example: 224 GB of weights, more than one 180 GB B200 holds. The detector, which polls every 30 minutes, caught it 19 s after vLLM pushed its image (a poll happened to be running) and built a worker in 15 minutes, but there is no smoke test, let alone a sweep, for it yet.

- **GPU count:** the smallest tensor-parallel size in {2, 4, 8} where the weights are at most 80% of total GPU memory **and** the KV floor above still fits. The 80% rule alone would pick 2× H200 for MiniMax (224 ≤ 225.6) and leave ~35 GB for everything else.
- `TENSOR_PARALLEL_SIZE` must equal the endpoint's `gpuCount`. The worker doesn't infer it, and a mismatch costs a full boot plus an out-of-memory relaunch. The size must divide the model's attention and KV head counts.
- To verify before spending: the Serverless `gpuCount` limits for that GPU type, shared-memory size for NCCL on Serverless workers, and whether the volume's data center has that many GPUs of the type together.
- Configs shrink to S0–S3. Cost scales with GPU count (prices are per GPU): at TP=2 on B200 each worker-minute costs ~$0.29, so each multi-GPU sweep gets its own approval and cap.
- If no single 8-GPU node fits, the result is "official quantized variant (Q) or not a Serverless model", not a sweep.

## Known issues that affect the numbers

- vllm#60143: the startup-plan key ignores `gpu_memory_utilization`. Per-config cache roots work around it.
- worker-vllm's out-of-memory relaunch sets `MAX_NUM_BATCHED_TOKENS=8192`, which **raises** the budget on GPUs under 70 GB (vLLM's default there is 2,048). Worth a worker-vllm issue on its own; the fit gate excludes any config that relaunched.
- If `VLLM_CACHE_ROOT` on the volume isn't writable, worker-vllm compiles into the container disk instead. Boot report v2 flags it (`compile_cache_fallback`), and the runner rejects such a start as not comparable.
- worker-vllm's README lists `GPU_MEMORY_UTILIZATION` 0.9 and the Hub template 0.95; vLLM's own default, which the day-0 builds use, is 0.92.

## Pilot

Qwen3-8B (16.4 GB of weights) on one 48 GB GPU type, S0–S5 + W, capped at 270 GPU-min (~$8 at the 48 GB class's $1.75/h): the table's 180, plus 90 for first-run costs (the weights download onto a new volume, a rerun if the runner breaks partway). Qwen3-8B on a 24 GB card can't hold 30 × 4,608 tokens of KV cache, so a 4090 would fail fit. Qwen3-8B also has the most existing reference numbers (worker-vllm #351, vllm-coldstart). The pilot checks that the runner, the boot report and the fit checks work. It doesn't recommend a config for a launch.

Image `day0-v0.32.0--day0-sweep-3b31d89` (vLLM v0.32.0), `MAX_MODEL_LEN` 32,768, S0 = defaults.

**Run 1** (Oct 28, 00:15 UTC; 5 worker-min, ~$0.15). The catalog put the new volume where only "NVIDIA RTX 6000 Ada Generation" was in stock; the worker's nvidia-smi reported "NVIDIA RTX PRO 6000 Blackwell Server Edition", and vLLM saw about 48 GB (26.4 GiB of KV cache at 0.92 after 15.27 GiB of weights). The pin check stopped the sweep there. What it confirmed: the OpenAI route takes token-id prompts and streams token by token (16 chunks for 16 tokens, TTFT 364 ms), and the boot report comes back filled in (first warm-up: engine init 117.8 s, of which 89 s was CUDA-graph capture; KV cache 192,208 tokens). Change: with one requested type, record the device the worker reports and hold every later start to it.

**Run 2** (Oct 28, 00:23 UTC; 21.8 worker-min, ~$0.64), pinned to "NVIDIA L40S", device "NVIDIA L40S":

| | S0 (defaults) | S1 (+ startup plan) |
|---|---|---|
| engine init, 3 cold starts | 19.85 / 19.89 / 19.92 s | 11.17 / 10.93 s (third start got no GPU) |
| of which graph capture | 13 s | 7 s |
| model loading | 39.5 s, then 6.1 / 7.1 s | 6.6 / 6.6 s |
| `delayTime` p50 | 59.4 s | 162.6 s |
| KV cache | 172,416–179,664 tokens (floor 138,240) | 169,648 tokens |
| C=1 (streamed) | 42.0 tok/s, TPOT p50 22.7 ms, TTFT p50 582 ms | |
| C=30 (streamed then) | 116 of 120 failed | |

- The startup plan cut engine init by 8.9 s (19.9 → 11.0 s), about twice the 4.6 s vllm-coldstart measured for the same model on a 4090.
- Engine init repeats to within 0.07 s on one device, so the noise band is its 2 s floor. Model loading doesn't (39.5 s on the first cold start, ~6–7 s after), which is why it isn't ranked.
- C=30 through the OpenAI route: 116 of 120 streams ended after 114–121 of 512 tokens with no usage chunk, but vLLM's counters over the window show 60,077 tokens generated and all 491,520 prompt tokens read, so vLLM did the work and the streams were cut between the worker and the client. vLLM counted 91 successful requests, so it also aborted 29, consistent with the downstream connections closing. C=30 now goes through the jobs API instead.
- vLLM v0.32.0 didn't print the `max_num_batched_tokens=` line, so S5 would have been skipped. It now falls back to vLLM's per-GPU defaults (2,048 on an L40S).
- S1's third cold start saw no worker at all for 10 minutes (not even "throttled") and the sweep stopped as `no_gpu`. The runner now sets `workersMax` again after 3 minutes with no worker state, and a start that stops the sweep is kept in the record.

**Run 3** (Oct 28, 01:22 UTC; 21.2 worker-min, ~$0.62), image `day0-v0.32.0--day0-sweep-8cf6da4` (boot report v2), pinned to "NVIDIA L40S" (46,068 MiB, one device, no MIG), C=30 through the jobs API:

| | S0 (defaults) | S1 (+ startup plan) |
|---|---|---|
| engine init, 3 cold starts | 19.89 / 18.22 / 18.23 s | 12.29 / 12.07 s (fourth start never scheduled) |
| C=1 (streamed) | 41.9 tok/s, TPOT p50 22.6 ms, TTFT p50 648 ms | |
| C=30 (jobs API) | **429.9 tok/s**, 0 of 120 failed, 0 preemptions | |
| C=30, vLLM's own means | TTFT 4.06 s, TPOT 59.2 ms | |
| C=30 per request | e2e p50 74.4 s, of which 41.6 s queued | |
| fit | yes (KV 172,416 tokens) | |

- The jobs-API bench works: vLLM's counter (61,440 tokens, 120 successful requests) matches the jobs' usage exactly.
- S1 again saved ~6 s of engine init (18.2 → 12.2 s).
- S1's fourth start saw no worker at all for two 10-minute waits, exactly as in run 2, while S0's endpoint did five starts (one waited 8.5 min "throttled"). The cause is unknown; the runner now retries such a start on a new endpoint.
- The summary said "recommended: S0" because S0 was the only config that finished. A stopped sweep is now marked incomplete instead.

**Run 4** (Oct 28, 02:41 UTC; 67.6 worker-min, ~$1.97), pinned to "NVIDIA L40S". Both warm-ups ran on a real L40S (engine init 69.8 s with an empty compile cache), but both of S0's first cold starts went to a 2g.48gb MIG slice of an RTX PRO 6000 Blackwell (engine init 181.6 and 143.3 s, KV 192,208–199,296 tokens). The device check rejected them, so S0 failed its first cold start twice. S1's cold starts never answered and ran into the 25-minute start limit twice. Changes: cold starts get 15 minutes (`--cold-minutes`); warm-ups keep 25 for the download.

**Runs 5 and 5b** (Oct 28, 05:12 and 05:15 UTC; $0), pinned to "NVIDIA RTX A6000" to avoid slices. Run 5: no data center with network volumes listed the A6000 in stock. Run 5b: one did (US-KS-2), but no worker appeared in two 10-minute waits, and the sweep stopped as `no_gpu`.

What the four runs say about the 48 GB tier: Runpod's pricing lists "L40, L40S, 6000 Ada, MIG 48GB" as one $1.75/h tier, so an L40S or RTX 6000 Ada type id can be served on a MIG slice by design. The tier without slices (A6000, A40, $1.22/h) had no stock that night. On a slice, engine init was 7–10× an L40S's. Changes: the device check compares nvidia-smi's name plus the MIG profile (a whole RTX PRO 6000 and a slice report the same name). `--data-center` pins a new volume's data center.

**Run 6** (Oct 28, 05:43 UTC; $0): pinned to "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb", which the catalog lists in stock. `POST /endpoints` answered 400: `gpuTypeIds` only takes the REST API's own enum, which has no RTX PRO 6000 MIG ids. So a slice can be detected (the boot report's MIG line) but not requested, and the runner now refuses a MIG `--gpu-type` before creating anything.

**Run 7** (Oct 28, 05:48–08:42 UTC; 129.7 worker-min, ~$3.78), pinned to "NVIDIA L40S" with the volume in US-NC-1. Every start that got a worker ran on an L40S (46,068 MiB, no MIG); one of S4's cold starts saw no worker and was retried on a new endpoint. Verdict **ok, recommended S2**.

| config | change from S0 | engine init, 3 cold starts (s) | p50 | C=30 tok/s | C=1 TPOT p50 | KV tokens |
|---|---|---|---|---|---|---|
| S0 | none | 23.66 / 22.93 / 22.75 | 22.93 | 427.0 | 22.6 ms | 179,664 |
| S1 | startup plan | 14.98 / 14.19 / 16.10 | 14.98 | 428.7 | 22.6 ms | 169,648 |
| **S2** | S1 + `ENFORCE_EAGER=true` | 2.62 / 2.61 / 2.63 | **2.62** | 420.4 | 23.1 ms | 181,680 |
| S3 | S1 + `MAX_NUM_SEQS=32` | 9.28 / 8.65 / 8.46 | 8.65 | 428.5 | 22.6 ms | 174,160 |
| S4 | S1 + `GPU_MEMORY_UTILIZATION=0.95` | 12.92 / 12.78 / 12.41 | 12.78 | 432.7 | 22.5 ms | 179,360 |
| S5 | S1 + `MAX_NUM_BATCHED_TOKENS=8192` | 13.08 / 12.85 / 13.61 | 13.08 | 420.3 | 22.6 ms | 168,960 |
| W | S2 + S3 + S4 | 2.52 / 2.54 / 2.56 | 2.54 | 421.0 | 23.1 ms | 192,320 |

All seven fit: 0 failed requests and 0 preemptions at C=30, KV cache above the 138,240-token floor.

- Selection: the noise band is 2.0 s (S1's spread was 1.91 s, under the 2 s floor) and the throughput floor is 411.1 tok/s (95% of S4's 432.7). S2, S3 and S4 beat S1 by more than the band above the floor; S5 (1.9 s faster) didn't. W combines the three and comes in at 2.54 s, inside the band of S2's 2.62 s, so the tie goes to S2, the config with fewer changes.
- Eager mode skips CUDA-graph capture: engine init 22.9 → 2.6 s. It costs 1.9% of S1's throughput at C=30 (420.4 vs 428.7 tok/s) and 0.5 ms of TPOT for a single stream.
- `MAX_NUM_SEQS=32` captures graphs for fewer batch sizes: 8.65 s with no throughput loss. It's the pick when decode speed matters more than the last 6 s.
- The startup plan saved 7.95 s (22.93 → 14.98 s), against ~6 s in runs 2 and 3.
- Engine init is the part of a cold start a config controls. On the same runs `delayTime` went from 47 to 515 s and model loading from the volume from 7 to 54 s on cold starts, so one cold start varies far more than any config changes it.
- Run 4's MIG slices didn't come back. Nothing in run 7 prevented them; the device check would have rejected them.

Pilot total: ~$7.16 of the $8 (runs 1–7). Records: run 7 in `data/sweep/day0-v0.32.0--day0-sweep-8cf6da4/`, runs 2–6 under `data/sweep/_runs/`, run 7's log in `data/sweep/_logs/`.

## Decisions

- D1: throughput is measured through the Serverless endpoint, not on a pod: same worker, same concurrency cap, same gateway users get.
- D2: one prompt shape, 4,096 in / 512 out (config L adds 32K for long-context models).
- D3: the selection rule above. It started as "best throughput within 5 s of the best cold start" and moved to engine init with a measured noise band, because client-side timings can't resolve 5 s.
- D4: target context = recipe `context_length` capped at 32,768 (the smoke test's 4,096 isn't a serving config).
- D5: a network volume for every config.
- D6: config Q only for a variant from the model's own publisher, flagged as not quality-checked.
- D7: rank on engine-side times from the worker's boot report, not on client-side wall clock.
- D8: Q and L are reported, never recommended; they change the checkpoint or the workload, not just settings.
