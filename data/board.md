# Day-0 board

Last poll: 2026-10-21T19:51:45Z. Rows: launches vLLM published a model-named image for.

| vLLM day-0 image | pushed (UTC) | model | worker build | CPU checks | GPU smoke test | tag → worker image |
|---|---|---|---|---|---|---|
| `flint` | 2026-10-21 16:09 | RecursiveSI/flint (search (127 likes)) | pending |  |  |  |
| `reflection-beam` | 2026-10-16 08:16 | no match | pending |  |  |  |
| `muse-spark-open` | 2026-10-15 17:17 | meta-models/Muse-Spark-Open-816B-A42B | pending |  |  |  |
| `step-5-preview` | 2026-10-15 00:51 | stepfun-ai/Step-5-Preview | [success](https://github.com/ashwinsreedhar28/day0-ready/actions/runs/39661568095) → `worker-vllm:day0-step-5-preview` | **native**: vLLM resolves the architecture natively | pass: Qwen3-8B, 24gb, defaults, 6.1 min | backfill |
