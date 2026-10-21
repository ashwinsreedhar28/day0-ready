"""Render data/board.md from data/state.json and data/results/*.json.

One row per launch: when vLLM pushed its day-0 image, the Hugging Face model
it maps to, whether the worker built from it, the CPU-check verdict, and the
hours from vLLM's tag to a pushed worker image.
"""

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

VERDICT_NOTE = {
    "native": "vLLM resolves the architecture natively",
    "native+remote_code": "native, needs TRUST_REMOTE_CODE=true",
    "fallback": "loads via vLLM's Transformers backend",
    "fallback+remote_code": "Transformers backend with the repo's code; needs TRUST_REMOTE_CODE=true",
    "unsupported": "vLLM in this image can't resolve the architecture",
    "transformers_too_old": "transformers in this image can't parse the config",
    "gated_or_missing": "config not readable (gated repo; needs an HF token with access)",
    "hub_error": "Hugging Face request failed",
    "error": "check failed; see run",
}


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def main():
    state = json.loads((DATA / "state.json").read_text()) if (DATA / "state.json").exists() else {"seen": {}}
    results = {}
    for f in sorted((DATA / "results").glob("*.json")) if (DATA / "results").exists() else []:
        r = json.loads(f.read_text())
        results[r["vllm_tag"]] = r

    smoke = {}  # image name (without -sha) -> latest smoke result
    for f in sorted((DATA / "smoke").glob("*.json")) if (DATA / "smoke").exists() else []:
        sm = json.loads(f.read_text())
        name = sm["image"].rsplit(":", 1)[-1]
        name = re.sub(r"-[0-9a-f]{7,}$", "", name) if name.count("-") > 1 else name
        if name not in smoke or sm["finished_at"] > smoke[name]["finished_at"]:
            smoke[name] = sm

    def smoke_cell(r):
        sm = r and smoke.get(r["image"].rsplit(":", 1)[-1])
        if not sm:
            return ""
        last = sm["rungs"][-1] if sm["rungs"] else {}
        how = ", ".join(f"{k}={v}" for k, v in (sm.get("passing_env") or {}).items()) or "defaults"
        model = sm["model"].split("/")[-1]
        if sm["passed"]:
            return f"pass: {model}, {sm['passing_gpu_class']}, {how}, {last.get('minutes')} min"
        return f"fail: {model}, {last.get('outcome')}"

    rows = []
    for fam, ev in state["seen"].items():
        if ev.get("skipped"):
            continue
        label = ev.get("model") and (ev["model"] + (f" ({ev['match']})" if ev.get("match", "").startswith("search") else ""))
        rows.append((ev.get("vllm_tag_pushed") or "", fam, ev.get("vllm_tag"), label,
                     results.pop(ev.get("vllm_tag"), None), ev.get("first_seen")))
    for tag, r in results.items():  # manual runs with no detector event
        rows.append(("", tag, tag, r.get("model"), r, None))

    lines = ["# Day-0 board", "",
             f"Last poll: {state.get('last_poll', 'never')}. Rows: launches vLLM published a model-named image for.", "",
             "| vLLM day-0 image | pushed (UTC) | model | worker build | CPU checks | GPU smoke test | tag → worker image |",
             "|---|---|---|---|---|---|---|"]
    for pushed, fam, tag, model, r, first_seen in sorted(rows, key=lambda row: row[0], reverse=True):
        if r is None:
            build, verdict, lag = "pending", "", ""
        else:
            build = f"[{r['build']}]({r['run_url']})" + (f" → `{r['image'].split('/')[-1]}`" if r.get("pushed") == "success" else "")
            verdict = f"**{r.get('verdict') or '-'}**" + (f": {VERDICT_NOTE[r['verdict']]}" if r.get("verdict") in VERDICT_NOTE else "")
            lag = ""
            if pushed and first_seen and (ts(first_seen) - ts(pushed)).total_seconds() > 2 * 3600:
                lag = "backfill"  # detected long after vLLM pushed the tag; not a live measurement
            elif pushed and r.get("pushed") == "success":
                lag = f"{(ts(r['finished_at']) - ts(pushed)).total_seconds() / 3600:.1f} h"
        lines.append(f"| `{tag}` | {pushed[:16].replace('T', ' ')} | {model or 'no match'} | {build} | {verdict} | {smoke_cell(r)} | {lag} |")
    (DATA / "board.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
