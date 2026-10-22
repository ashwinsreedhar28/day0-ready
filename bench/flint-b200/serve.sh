#!/usr/bin/env bash
# On the pod: serve one model with vLLM. Usage: serve.sh nvfp4 | bf16 | qwen
#   nvfp4  Flint, RecursiveSI/flint-nvfp4-gguf     served as "flint"
#   bf16   Flint, RecursiveSI/flint                served as "flint"
#   qwen   Qwen/Qwen3.6-35B-A3B (BF16), text only   served as "qwen3.6"
# Flint keeps the Oct 21 Serverless settings (FlashInfer attention, 1 GPU); Qwen uses
# vLLM's defaults. All use max-model-len 36864 so a 32768-token prompt plus 1024
# output tokens fits.
set -euo pipefail
which=${1:-nvfp4}
mkdir -p /root/models /root/bench
common=(--max-model-len 36864 --host 127.0.0.1 --port 8000)
case "$which" in
  nvfp4)
    python3 -c "from huggingface_hub import hf_hub_download as d; print(d('RecursiveSI/flint-nvfp4-gguf', 'flint-nvfp4.gguf', local_dir='/root/models'))"
    # The flint image lacks the gguf package its Flint GGUF loader needs, and its
    # `vllm serve` can't start from a local .gguf path (see vllm_gguf.py).
    pip install -q gguf
    launch=(python3 "$(dirname "$(readlink -f "$0")")/vllm_gguf.py" serve /root/models/flint-nvfp4.gguf
            --tokenizer RecursiveSI/flint --hf-config-path RecursiveSI/flint
            --served-model-name flint --attention-backend FLASHINFER) ;;
  bf16)
    launch=(vllm serve RecursiveSI/flint --served-model-name flint --attention-backend FLASHINFER) ;;
  qwen)
    # Text-only benchmark: no image or video inputs, so vLLM skips profiling the vision encoder.
    launch=(vllm serve Qwen/Qwen3.6-35B-A3B --served-model-name qwen3.6 --limit-mm-per-prompt '{"image":0,"video":0}') ;;
  *) echo "usage: serve.sh nvfp4|bf16|qwen"; exit 1 ;;
esac
pkill -f "vllm serve|vllm_gguf.py serve" 2>/dev/null || true
# Wait for a previous server to release the GPU, or vLLM sizes its KV cache from less memory.
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${used:-0}" -lt 1000 ] && break
  sleep 2
done
cmd=("${launch[@]}" "${common[@]}")
printf '%q ' "${cmd[@]}" | tee /root/bench/serve-$which.cmd; echo
nohup "${cmd[@]}" > /root/bench/serve-$which.log 2>&1 &
pid=$!
for i in $(seq 1 240); do
  if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" 2>/dev/null; then
    echo "ready after $((i * 5)) s"
    grep -iE "Maximum concurrency|KV cache memory|GPU KV cache size|attention backend|MoE backend" /root/bench/serve-$which.log | tail -6
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "vLLM exited during startup:"; tail -40 /root/bench/serve-$which.log; exit 1
  fi
  sleep 5
done
echo "not ready after 20 min"; tail -40 /root/bench/serve-$which.log; exit 1
