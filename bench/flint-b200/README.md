# Flint on one B200: vLLM load sweep

RecursiveSI/flint served by vLLM on a single NVIDIA B200 (Runpod pod), measured with `vllm bench serve` at 8K and 32K input, 1,024 output tokens, concurrency 1 to 64. Run on 2026-10-22 between 01:23 and 02:12 UTC (Oct 21, 9:23–10:12 pm ET), the day vLLM published its Flint image.

## Results

NVFP4 checkpoint (`RecursiveSI/flint-nvfp4-gguf`). TTFT and TPOT in ms, throughput in tokens/s. Every request generated exactly 1,024 tokens (`--ignore-eos`).

| input | concurrency | prompts | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | output tok/s | total tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 8,192 | 1 | 8 | 138 | 153 | 1.76 | 1.86 | 1.79 | 523 | 4,703 |
| 8,192 | 8 | 32 | 507 | 638 | 3.27 | 3.49 | 3.19 | 2,124 | 19,118 |
| 8,192 | 32 | 128 | 1,200 | 2,027 | 6.58 | 7.38 | 5.89 | 4,130 | 37,169 |
| 8,192 | 64 | 256 | 1,886 | 3,860 | 9.92 | 11.39 | 8.34 | 5,406 | 48,656 |
| 32,768 | 1 | 8 | 459 | 507 | 1.93 | 2.00 | 1.90 | 420 | 13,852 |
| 32,768 | 8 | 32 | 1,646 | 2,575 | 4.81 | 5.90 | 3.98 | 1,247 | 41,144 |
| 32,768 | 32 | 128 | 3,782 | 9,492 | 14.43 | 17.46 | 9.04 | 1,760 | 58,093 |
| 32,768 | 64 | 256 | 2,692 | 19,009 | 30.25 | 31.95 | 14.98 | 1,941 | 64,046 |

BF16 checkpoint (`RecursiveSI/flint`), concurrency 1 only:

| input | concurrency | prompts | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | output tok/s | total tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 8,192 | 1 | 8 | 126 | 151 | 2.23 | 2.27 | 2.23 | 424 | 3,817 |
| 32,768 | 1 | 8 | 474 | 502 | 2.33 | 2.37 | 2.35 | 357 | 11,776 |

All 10 runs completed every request (0 failed). Raw results: `results/*.json`; each run's console output: `results/*.log`. `python3 summarize.py results` reprints the table.

### Against vLLM's published figures

vLLM's day-0 announcement for Flint gives GB200 numbers at batch 1: 2.06 ms/token end to end at 100K context; forward pass only, 1.78 ms NVFP4 and 2.34 ms BF16; 145 ms for an 8K-token prefill. (Quoted from the announcement on X, `x.com/vllm_project/status/2112948309350580235`, which we couldn't open directly.) Here, on a B200 at batch 1:

- NVFP4 TPOT p50 1.76 ms at 8K and 1.93 ms at 32K; BF16 2.23 ms and 2.33 ms. TPOT here is measured through the HTTP server, like vLLM's 2.06 ms end-to-end figure, but at 8K and 32K context rather than 100K, and on B200 rather than GB200.
- TTFT at 8K: 138 ms (NVFP4) and 126 ms (BF16), against 145 ms for vLLM's 8K prefill. TTFT includes HTTP and tokenization.

### What the sweep shows

- **8K context scales.** From concurrency 1 to 64, output throughput rises 10.3× (523 → 5,406 tok/s) while each stream still decodes at a median 9.92 ms/token (about 100 tok/s) and the median first token arrives in 1.9 s.
- **32K context saturates around concurrency 32.** Going from 32 to 64 adds 10% output throughput (1,760 → 1,941 tok/s) while median TPOT doubles (14.4 → 30.3 ms) and TTFT p99 doubles (9.5 → 19.0 s). Total throughput is 64K tok/s, almost all of it prompt tokens.
- **The KV cache was not the limit.** In the 32K, concurrency-64 run the server log shows up to 64 requests running and 25 waiting, KV-cache usage at most 43.8%, and no preemptions. The likely limit is prefill: with chunked prefill each engine step has a 16,384-token budget (`max_num_batched_tokens`, the default here), so a 32K prompt takes at least two steps, and those steps are shared with every decoding request. A larger budget was not tested.
- **At batch 1, NVFP4 takes 17–21% less time per output token than BF16** (1.76 vs 2.23 ms at 8K; 1.93 vs 2.33 ms at 32K). Prefill is about the same (8K: 138 vs 126 ms; 32K: 459 vs 474 ms).

## Setup

| | |
|---|---|
| GPU | 1× NVIDIA B200, 183,359 MiB, driver 595.91.07; Runpod Secure Cloud on-demand pod |
| Image | `vllm/vllm-openai:flint`, index digest `sha256:be337371b88c770a25a7c029ecc1096dde132889853609b28093ddabc002b0bd` (linux/amd64 `sha256:81e0d5248b2f7373bc63891f4cbdb38c9b6a795b89d72b2921fc1f47e9a68989`), pushed 2026-10-21 16:10 UTC |
| vLLM | 0.32.0rc1.dev348+ge7c846767, torch 2.14.0+cu130, CUDA 13.0 |
| NVFP4 checkpoint | `RecursiveSI/flint-nvfp4-gguf` → `flint-nvfp4.gguf` (21.84 GiB in GPU memory once loaded), config and tokenizer from `RecursiveSI/flint` |
| BF16 checkpoint | `RecursiveSI/flint` |
| Kernels (from the server log) | FlashInfer attention, decode on `trtllm-gen` (sm100); NVFP4 GEMM `FlashInferCuteDslNvFp4W4A16LinearKernel`; TRT-LLM fused MoE (NVFP4) / `TrtLlmBf16ExpertsMonolithic` (BF16) |
| KV cache | NVFP4: 139.72 GiB, 1,410,480 tokens. BF16: 87.52 GiB, 883,506 tokens. `gpu_memory_utilization` default 0.92; prefix caching and chunked prefill on (defaults) |

Same settings as the Oct 21 Serverless pass (`ATTENTION_BACKEND=FLASHINFER`, one GPU), except `--max-model-len 36864` (was 4096) so a 32,768-token prompt plus 1,024 output tokens fits.

### Serve commands

NVFP4, after `pip install gguf` (see "Problems found"). `serve.sh nvfp4` runs this; it was updated after the run to match what was typed by hand on the pod, and the server log's `non-default args` line shows these arguments:

```
python3 vllm_gguf.py serve /root/models/flint-nvfp4.gguf --tokenizer RecursiveSI/flint --hf-config-path RecursiveSI/flint \
  --served-model-name flint --attention-backend FLASHINFER --max-model-len 36864 --host 127.0.0.1 --port 8000
```

BF16:

```
vllm serve RecursiveSI/flint --served-model-name flint --attention-backend FLASHINFER --max-model-len 36864 --host 127.0.0.1 --port 8000
```

`serve.sh bf16` runs the BF16 command (`serve-bf16.cmd`).

### Bench command

Flags were checked against `vllm bench serve --help=all` from the same image (`bench-help-all.txt`) before running. For each input length L ∈ {8192, 32768} and concurrency c ∈ {1, 8, 32, 64}:

```
vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model flint --tokenizer RecursiveSI/flint \
  --dataset-name random --random-input-len L --random-output-len 1024 --ignore-eos \
  --num-prompts max(8, 4c) --max-concurrency c --seed L+c \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
  --save-result --result-dir results --result-filename flint-<ckpt>-inL-out1024-cc.json
```

- `--random-range-ratio` was left at its default, 0.0, so every prompt is exactly L tokens and every output exactly 1,024 (confirmed by `total_input_tokens` and `total_output_tokens` in each JSON).
- Each run uses its own seed. With one shared seed, vLLM's random prompts for 32K start with the same tokens as the 8K ones, and the prefix cache (on by default) would skip part of the prefill. The server's prefix-cache hit rate stayed at 0.5%.
- Request rate is unlimited, so the client keeps exactly c requests in flight.
- `--num-warmups` stayed at 0. Each run starts with the tool's single test request, which isn't measured, and CUDA graphs were captured at server start.

## Cost

Pod `bsh9fjju86vxwy` (1× B200, $7.99/hr on demand) ran 49 minutes, about **$6.56** at the hourly rate, including the 10.75 GB image pull, both checkpoint downloads and finding the two workarounds below. It was stopped afterwards, not terminated.

## Problems found

1. **vLLM can't start a local GGUF in this build.** `maybe_override_with_speculators()` (vllm/transformers_utils/config.py) reads the model path as a JSON config, so `vllm serve /root/models/flint-nvfp4.gguf` fails with "It looks like the config file at '/root/models/flint-nvfp4.gguf' is not a valid JSON file." `vllm_gguf.py` skips that check for `.gguf` paths and otherwise runs the normal CLI.
2. **The `vllm/vllm-openai:flint` image is missing the `gguf` package** that its own Flint loader requires: "Loading Flint NVFP4 GGUF checkpoints requires the `gguf` package (pip install gguf)."
3. **Bogus load timer.** The NVFP4 server log reports "Loading weights took 197518.83 seconds"; a few lines later it says "Model loading took 21.84 GiB memory and 23.915460 seconds", and the engine started 21 s before that first line.

## Limits

- Random token prompts, not real code; `--ignore-eos` forces full 1,024-token outputs.
- One run per point, no repeats; small prompt counts at low concurrency (8 at c=1).
- One B200 on one host; TTFT at high concurrency includes time queued for prefill.

## Files

- `results/` — 10 result JSONs and the console output of each run
- `serve-nvfp4.log`, `serve-bf16.log`, `serve-bf16.cmd` — server logs and the BF16 command
- `bench-help-all.txt` — `vllm bench serve --help=all` from the image
- `sweep-nvfp4.out` — the NVFP4 sweep's console output
- `pod.py` (create/status/stop/terminate the pod), `serve.sh`, `sweep.sh`, `summarize.py`, `vllm_gguf.py`
