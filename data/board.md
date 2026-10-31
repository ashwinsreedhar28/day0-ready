# Day-0 board

Last poll: 2026-10-31T05:07:22Z. Rows: launches vLLM published a model-named image for.

| vLLM day-0 image | pushed (UTC) | model | worker build | CPU checks | GPU smoke test | recommended config | tag → worker image |
|---|---|---|---|---|---|---|---|
| `mistral-large-4` | 2026-10-29 14:22 | mistralai/Mistral-Large-4-1T-A52B | [success](https://github.com/ashwinsreedhar28/day0-ready/actions/runs/40893342906) → `worker-vllm:day0-mistral-large-4` | **fallback**: loads via vLLM's Transformers backend |  |  | 0.5 h |
| `minimax-m31-flash-x86_64-cu134` | 2026-10-23 01:07 | MiniMaxAI/MiniMax-M3.1-Flash | [success](https://github.com/ashwinsreedhar28/day0-ready/actions/runs/39882185176) → `worker-vllm:day0-minimax-m31-flash-x86_64-cu134` | **native**: vLLM resolves the architecture natively | not run: 224 GB of weights is more than one 180 GB GPU holds |  | 0.3 h |
| `flint` | 2026-10-21 16:09 | RecursiveSI/flint (HF search, 127 likes) | [success](https://github.com/ashwinsreedhar28/day0-ready/actions/runs/39665308085) → `worker-vllm:day0-flint` | **native**: vLLM resolves the architecture natively | pass: flint, 180gb, ATTENTION_BACKEND=FLASHINFER, 13.1 min |  | backfill |
| `reflection-beam` | 2026-10-16 08:16 | no match | pending |  |  |  |  |
| `muse-spark-open` | 2026-10-15 17:17 | meta-models/Muse-Spark-Open-816B-A42B | pending |  |  |  |  |
| `step-5-preview` | 2026-10-15 00:51 | stepfun-ai/Step-5-Preview | [success](https://github.com/ashwinsreedhar28/day0-ready/actions/runs/39718801214) → `worker-vllm:day0-step-5-preview` | **native**: vLLM resolves the architecture natively | image check with a stand-in model (pass: Qwen3-8B, 24gb, defaults, 7.1 min) |  | backfill |
| `v0.32.0` |  | Qwen/Qwen3-8B | [success](https://github.com/ashwinsreedhar28/day0-ready/actions/runs/39665268200) → `worker-vllm:day0-v0.32.0` | **native**: vLLM resolves the architecture natively |  |  |  |
