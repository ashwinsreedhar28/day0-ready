# Multi-GPU smoke tests

Status (Oct 28, 2026, parked until January): `smoke.py --gpu-count N` works; a 2-GPU proof of concept passed. MiniMax-M3.1-Flash on 2× H200 didn't come up in two tries; the fixes those tries called for are in, and a third try waits for January. Automatic multi-GPU smoke tests and multi-GPU sweeps aren't built.

## Why

The launches that matter most are often too big for one GPU. MiniMax-M3.1-Flash was detected 19 s after vLLM pushed its image and built in 15 minutes, then recorded as "needs multi-GPU": its 224 GB of weights is more than one 180 GB B200 holds.

## How it works

- `--gpu-count N` (2, 4 or 8) sets the endpoint's `gpuCount` and the worker's `TENSOR_PARALLEL_SIZE` to N. worker-vllm doesn't infer the size, and a mismatch costs a full boot. N has to divide the model's attention and KV head counts.
- `--gpu` names the per-GPU class and must be explicit. Prices are per GPU, so the cost cap and budget checks multiply by N; in `smoke.yml` any `gpu_count` above 1 runs in the `gpu-large` environment.
- The worker's boot report records the device(s) it got. A 48 GB type id can be served by a MIG slice of an RTX PRO 6000 Blackwell, and tensor parallel across slices isn't expected to work.
- Choosing N: `pipeline/predict.py --gpus N` gives KV tokens per class as `N × (device memory × 0.92 − 3 GiB) − weights`, divided by the model's KV bytes per token. A smoke test needs room for one 4,096-token request; a sweep needs 30 × 4,608 = 138,240 tokens.
- Weights in the hundreds of gigabytes: `pipeline/stage.py` downloads them onto a network volume with a CPU pod first, and `smoke.py --volume-id` boots from that volume.

## Proof of concept: Qwen3-14B on 2× RTX 4090

**Pass, Oct 28 04:44 UTC, 4.3 min, ~$0.16.** Qwen3-14B (29.5 GB of weights, too big for one 24 GB card) with `TENSOR_PARALLEL_SIZE=2`: the worker got 2× RTX 4090 (24,564 MiB each, no MIG), vLLM was up 170 s after launch (engine init 80.9 s), and the request answered. So a Serverless worker gets 2 GPUs of one type, NCCL works under the worker's shared memory without NVLink, and worker-vllm passes the size through.

Under tensor parallel vLLM prints memory per GPU: 13.88 GiB of weights and 6.67 GiB of KV per card, 87,408 tokens for the pair. Scaled by 2 that's 163,872 B/token, matching the config's 2 × 40 × 8 × 128 × 2 = 163,840. `predict.py` expected 69,720 tokens, 20% under, because it charges 3 GiB of overhead per GPU where the PoC needed ~1.5; `from_numbers(..., gpus=N)` scales a TP boot. Record: `data/smoke/_runs/qwen3-14b-2x4090-poc.json`.

## MiniMax-M3.1-Flash on 2× H200

Image `day0-minimax-m31-flash-x86_64-cu134--day0-sweep-8cf6da4`, mxfp8 weights (232.94 GiB on disk), the recipe's `BLOCK_SIZE=128`, `TENSOR_PARALLEL_SIZE=2`. Records: `data/smoke/_runs/minimax-2xh200-try*.json`; try 1's worker log and try 2's runner log in `data/smoke/_logs/`.

| | weights | what happened | outcome |
|---|---|---|---|
| try 1, 04:50–05:27 UTC | downloaded to the container disk by each boot (442.6 s, then 426.6 s) | both boots stopped from outside ("Received signal 15", no traceback) ~8.4 min after vLLM started, a few of 48 shards into loading | `crash_loop` at 36.7 min, ~$7.26 |
| try 2, 06:20–06:57 UTC | staging on a 300 GB volume said done with 17 MB there, so the worker downloaded them again | worker "running" 8.6 min, then 17 min; the smoke test counted both changes back to idle/ready as crashes | `crash_loop` at 32.6 min, ~$6.45 |

What's known: NCCL came up with TP=2 and Hopper kernels were picked for the MoE and sparse attention in both boots of try 1. What isn't: who stopped try 1's boots (not the 10-minute execution timeout, since worker-vllm takes no job before vLLM is healthy; and try 2 had a 17-min boot, so not a fixed ~8-minute limit either), and whether try 2's worker was serving the request when the smoke test gave up. Try 2's log went with its endpoint.

Fixed since, no GPU spent:

- `stage.py` says done only when the finished files in the cache's `blobs/` add up to the sizes the Hub lists for the files it asked for, downloads over plain HTTP (`HF_HUB_DISABLE_XET=1`; the hf_xet path is the likely cause of try 2's empty volume, not verified), and the runner checks the pod's numbers again. It tries 8-, 4- and 2-vCPU pods, only in data centers `POST /pods` accepts.
- The smoke test reads the job's status before its crash rule: a worker leaving "running" counts as a restart only while the job is still queued. Each rung records a worker-state timeline, the minute the job was picked up, and the job status when it stopped.
- `stage.py --cleanup --keep-endpoints` keeps a failed smoke endpoint at 0 workers with the volume detached, so its log stays readable; if the volume won't detach, the endpoint is deleted so nothing keeps billing.

Third try (January, ~$8 at a 40-minute cap): `stage.py --gpu-types "NVIDIA H200,NVIDIA H200 NVL"`, then `smoke.py --gpu 141gb --gpu-count 2 --volume-id <volume>`, then `stage.py --cleanup` (`--keep-endpoints` if it fails).

## Not built yet

- Picking N automatically from the weight size (`--gpu auto` stays single-GPU).
- The ladder for multi-GPU failures (NCCL errors, a head count N doesn't divide).
- Sweeps on multi-GPU workers (S0–S3 only, per-sweep approval; see [optimal-config.md](optimal-config.md)).
