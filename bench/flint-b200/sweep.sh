#!/usr/bin/env bash
# On the pod: run the load sweep against the running server.
# Usage: sweep.sh <label> <concurrency...>, e.g. sweep.sh nvfp4 1   then   sweep.sh nvfp4 8 32 64
# Random prompts, fixed lengths, --ignore-eos so every request emits all 1024 tokens.
set -euo pipefail
label=$1; shift
out=/root/bench/results
mkdir -p "$out"
help=$(vllm bench serve --help=all 2>&1)
for f in --backend --base-url --model --tokenizer --dataset-name --random-input-len --random-output-len \
         --ignore-eos --num-prompts --max-concurrency --seed --percentile-metrics --metric-percentiles \
         --save-result --result-dir --result-filename; do
  grep -q -- "$f" <<<"$help" || { echo "flag $f is not in 'vllm bench serve --help'; stopping"; exit 1; }
done
for in_len in 8192 32768; do
  for c in "$@"; do
    n=$(( c * 4 < 8 ? 8 : c * 4 ))
    name="flint-${label}-in${in_len}-out1024-c${c}.json"
    echo "== $label  input $in_len  output 1024  concurrency $c  prompts $n"
    vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model flint --tokenizer RecursiveSI/flint \
      --dataset-name random --random-input-len "$in_len" --random-output-len 1024 --ignore-eos \
      --num-prompts "$n" --max-concurrency "$c" --seed "$((in_len + c))" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
      --save-result --result-dir "$out" --result-filename "$name" > "$out/${name%.json}.log" 2>&1 \
      || { echo "bench failed:"; tail -30 "$out/${name%.json}.log"; exit 1; }
  done
done
python3 /root/bench/summarize.py "$out"
