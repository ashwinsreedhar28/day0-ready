"""Print the result JSONs in a directory as one markdown table."""
import json
import re
import sys
from pathlib import Path

rows = []
for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else ".").glob("*-in*-out*-c*.json")):
    m = re.match(r"([\w.]+?)-(\w+)-in(\d+)-out(\d+)-c(\d+)\.json", f.name)
    if not m:
        continue
    r = json.loads(f.read_text())
    g = lambda *ks: next((r[k] for k in ks if r.get(k) is not None), float("nan"))  # noqa: E731
    rows.append((m[1], m[2], int(m[3]), int(m[5]), r.get("completed"), r.get("num_prompts", "?"),
                 g("p50_ttft_ms", "median_ttft_ms"), g("p99_ttft_ms"), g("p50_tpot_ms", "median_tpot_ms"), g("p99_tpot_ms"),
                 g("p50_itl_ms", "median_itl_ms"), g("output_throughput"), g("total_token_throughput")))
rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
print("| model | ckpt | input | conc | done | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | ITL p50 ms | out tok/s | total tok/s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for mdl, c, i, k, done, n, t50, t99, p50, p99, itl, out, tot in rows:
    print(f"| {mdl} | {c} | {i} | {k} | {done}/{n} | {t50:.0f} | {t99:.0f} | {p50:.2f} | {p99:.2f} | {itl:.2f} | {out:.0f} | {tot:.0f} |")
