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

**Try 1 (Oct 28, 04:50–05:27 UTC): no pass; `crash_loop` after 36.7 min, ~$7.26.** The worker got 2 GPUs and reached "running". vLLM started twice (05:09 and 05:18) with TP=2; both times NCCL initialized and Hopper kernels were picked for the MoE and sparse attention. It wasn't a crash: the worker log has no traceback, and both boots end with worker-vllm's "Received signal 15, shutting down vLLM", a stop from outside.

| | vLLM start | weights downloaded | stopped | after start |
|---|---|---|---|---|
| boot 1 | 05:09:07 | 442.6 s (232.94 GiB) | 05:17:34, shard 3 of 48 | 8 min 28 s |
| boot 2 | 05:18:19 | 426.6 s, all of it again | 05:26:40, shard 7 of 48 | 8 min 22 s |

- Each boot downloads the checkpoint onto the container disk at ~0.53 GiB/s, and a restart starts over, so no boot gets further than the first.
- Both stops landed ~8.4 min after vLLM started. Runpod's docs only describe the 10-minute execution timeout, and worker-vllm takes no job before vLLM is healthy, so that isn't it. A fixed ~8-minute startup limit looked likely, but in try 2 a boot ran 17 min, so the cause is still open.
- The smoke test called it `crash_loop` because it counts a worker leaving "running" as a crash; here those were two external stops during startup.

**Try 2 (Oct 28, 06:20–06:57 UTC): no pass; `crash_loop` after 32.6 min, ~$6.45.** The plan was to put the weights on a 300 GB network volume first with a CPU pod (`pipeline/stage.py`), so the worker only loads them. In US-NC-1 the 8- and 4-vCPU pods had no capacity; a 2-vCPU pod ($0.06/h) got one.

- Staging reported done after 247 s with 17 MB on the volume. The model's cache directory had grown to 29 GiB and then shrank, so the weights never landed and the worker downloaded them again, as in try 1. The likely cause is the download path (huggingface_hub 2.x goes through hf_xet), not verified. Staging has to check the bytes on the volume against the repo's size before it says done.
- The worker showed "running" for 8.6 min (06:31:30–06:40:08), then again for 17 min (06:40:23–06:57:25). The smoke test counts a change from running to idle/ready as a crash and stopped at the second one. "ready" may have been the worker coming up; the cleanup deleted the endpoint, so its log is gone.
- Fixed since (no GPU spent):
  - `pipeline/stage.py` now counts only finished files in the cache's `blobs/` against the sizes the Hub lists for the files it asked for, and says done only when they match. The pod downloads over plain HTTP (`HF_HUB_DISABLE_XET=1`), and the runner checks the pod's numbers again.
  - The smoke test's crash rule now reads the job's status first. A worker leaving "running" counts as a restart only while the job is still queued; with the job `IN_PROGRESS` the worker is serving it, since worker-vllm takes jobs only once vLLM is healthy. Each rung records a worker-state timeline, the minute the job was picked up, and the job's status when the rung stopped.
  - `stage.py --cleanup --keep-endpoints` detaches the volume from a failed smoke endpoint and keeps the endpoint at 0 workers, so its log stays readable; if Runpod won't let the volume go, the endpoint is deleted so no volume keeps billing.
- A third try needs approval (~$8 at the 40-minute cap).

## Not built yet

- Picking N automatically from the weight size (`--gpu auto` stays single-GPU).
- The ladder for multi-GPU failures (NCCL errors, a head count N doesn't divide).
- Sweeps on multi-GPU workers (configs S0–S3 only, per-sweep approval; see `docs/optimal-config.md`).
