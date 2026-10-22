#!/usr/bin/env bash
# On the pod: serve Flint with vLLM. Usage: serve.sh nvfp4 | bf16
# Same settings as the Oct 21 Serverless pass (FlashInfer attention, 1 GPU), except
# max-model-len 36864 so a 32768-token prompt plus 1024 output tokens fits.
set -euo pipefail
which=${1:-nvfp4}
mkdir -p /root/models /root/bench
if [ "$which" = nvfp4 ]; then
  python3 -c "from huggingface_hub import hf_hub_download as d; print(d('RecursiveSI/flint-nvfp4-gguf', 'flint-nvfp4.gguf', local_dir='/root/models'))"
  # The flint image lacks the gguf package its Flint GGUF loader needs, and its
  # `vllm serve` can't start from a local .gguf path (see vllm_gguf.py).
  pip install -q gguf
  launch=(python3 "$(dirname "$(readlink -f "$0")")/vllm_gguf.py" serve /root/models/flint-nvfp4.gguf
          --tokenizer RecursiveSI/flint --hf-config-path RecursiveSI/flint)
else
  launch=(vllm serve RecursiveSI/flint)
fi
pkill -f "vllm serve|vllm_gguf.py serve" 2>/dev/null || true
sleep 3
cmd=("${launch[@]}" --served-model-name flint --attention-backend FLASHINFER
     --max-model-len 36864 --host 127.0.0.1 --port 8000)
echo "${cmd[*]}" | tee /root/bench/serve-$which.cmd
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
