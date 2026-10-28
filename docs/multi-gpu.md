# Multi-GPU smoke tests

Status: 2026-10-28. `smoke.py --gpu-count N` exists; a 2-GPU proof of concept passed (below). Automatic multi-GPU smoke tests and multi-GPU sweeps wait for January.

## Why

The launches that matter most are often too big for one GPU. MiniMax-M3.1-Flash (224 GB of weights) was detected 19 s after vLLM pushed its image and built in 15 minutes, then recorded as "needs multi-GPU" with no GPU test, because one B200 holds 180 GB.

## What a multi-GPU smoke test does

- `--gpu-count N` (2, 4 or 8) sets the endpoint's `gpuCount` to N and the worker's `TENSOR_PARALLEL_SIZE` to N. worker-vllm doesn't infer the size from the GPUs it gets, and a mismatch costs a full boot. N has to divide the model's attention and KV head counts.
- `--gpu` names the per-GPU class and must be explicit; the Runpod prices are per GPU, so the cost cap and budget checks multiply by N.
- The smoke test asks the worker for its boot report next to the request, so the result records the device(s) it ran on. A 48 GB type id can be served by a MIG slice of an RTX PRO 6000 Blackwell (sweep pilot, Oct 28), and tensor parallel across MIG slices isn't expected to work, so the device list matters more here than on one GPU.
- In `smoke.yml`, any `gpu_count` above 1 runs in the `gpu-large` environment, like the 80 GB and larger classes.

## Choosing N

`pipeline/predict.py --gpus N` gives KV tokens per class with N GPUs: `N × (device memory × 0.92 − 3 GiB) − weights`, divided by the model's KV bytes per token. The smoke test needs room for one 4,096-token request; a sweep needs 30 × 4,608 = 138,240 tokens. NCCL buffers come on top of the 3 GiB per GPU; the PoC checks how much.

## Proof of concept

Qwen3-14B (29.5 GB of weights, so it doesn't fit one 24 GB card) on 2 GPUs of the 24 GB class with `TENSOR_PARALLEL_SIZE=2`, one request, capped at 30 minutes. It answers three platform questions in one boot: whether a Serverless worker gets 2 GPUs of one type, whether NCCL works with the worker's shared memory, and whether worker-vllm passes the size through. `predict.py` expects ~69,700 KV tokens on 2× RTX 4090.

**Result (Oct 28, 04:44 UTC): pass in 4.3 min, ~$0.16.** The worker got 2× NVIDIA GeForce RTX 4090 (24,564 MiB each, no MIG), vLLM came up with `TENSOR_PARALLEL_SIZE=2` in 170 s from launch (engine init 80.9 s), and the chat request answered. So a Serverless worker does get 2 GPUs of one type, NCCL works under the worker's shared memory without NVLink, and worker-vllm passes the size through.

Under tensor parallel vLLM prints memory per GPU: 13.88 GiB of weights and 6.67 GiB of KV per card, 87,408 tokens for the pair. Scaled by 2 that's 27.8 GiB and 163,872 B/token, which matches the config's 2 × 40 × 8 × 128 × 2 = 163,840. `predict.py` expected 69,720 tokens (20% under: it charges 3 GiB of overhead per GPU, the PoC needed ~1.5); `from_numbers(..., gpus=N)` scales a TP boot.

## MiniMax-M3.1-Flash on 2× H200

The launch this was for: 224 GB of mxfp8 weights, image `day0-minimax-m31-flash-x86_64-cu134--day0-sweep-8cf6da4`, recipe setting `BLOCK_SIZE=128`, `TENSOR_PARALLEL_SIZE=2` on the 141gb class, capped at 45 minutes.

**Result (Oct 28, 04:50–05:27 UTC): no pass; `crash_loop` after 36.7 min, ~$7.26.** The worker got 2 GPUs and reached "running". vLLM started at 05:18 with TP=2, NCCL initialized, and Hopper kernels were picked for the MoE and sparse attention, then the worker restarted until the cap. The container log we have ends during weight loading, so the cause isn't known yet. 28 of the 36.7 minutes passed before vLLM started (scheduling, the image pull and possibly the weights; the log doesn't separate them).

## Not built yet

- Picking N automatically from the weight size (`--gpu auto` stays single-GPU).
- The ladder for multi-GPU failures (NCCL errors, a head count N doesn't divide).
- Sweeps on multi-GPU workers (configs S0–S3 only, per-sweep approval; see `docs/optimal-config.md`).
