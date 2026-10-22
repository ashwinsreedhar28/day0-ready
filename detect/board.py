"""Render data/board.md from data/state.json and data/results/*.json.

One row per launch: when vLLM pushed its day-0 image, the Hugging Face model
it maps to, whether the worker built from it, the CPU-check verdict, and the
hours from vLLM's tag to a pushed worker image.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "pipeline"))
import smoke as smoke_mod  # noqa: E402  (template_body / endpoint_body for data/ready.json)

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


def smoke_text(sm):
    """One board cell for a smoke result."""
    last = sm["rungs"][-1] if sm.get("rungs") else {}
    model = sm["model"].split("/")[-1]
    if sm.get("passed"):
        how = ", ".join(f"{k}={v}" for k, v in (sm.get("passing_env") or {}).items()) or "defaults"
        return f"pass: {model}, {sm['passing_gpu_class']}, {how}, {last.get('minutes')} min"
    skipped = sm.get("skipped") or (sm.get("stopped") if not sm.get("rungs") else None)
    if skipped:
        return f"not run: {skipped['detail']}"
    if last.get("outcome") == "needs_remote_code":
        return f"{model} needs TRUST_REMOTE_CODE: review the repo's code, then rerun with allow_remote_code"
    stopped = f" (stopped: {sm['stopped']['reason']})" if sm.get("stopped") else ""
    return f"fail: {model}, {last.get('outcome')}{stopped}"


def ready_entries(smoke, builds_by_image):
    """Known-good deploys: one per launch whose own model passed on its board image.

    Left out: branch test builds ("--" in the name) and stand-in models (step-5-preview's
    Qwen3-8B run proved the image serves, not that Step-5-Preview does).
    """
    out = []
    for name, sm in sorted(smoke.items()):
        build = builds_by_image.get(sm["image"]) or {}
        if not sm.get("passed") or "--" in name or build.get("model") != sm["model"]:
            continue
        last = sm["rungs"][-1]
        deploy = sm.get("deploy") or {
            "template": smoke_mod.template_body(sm["image"], sm["model"], last["env"], last.get("disk_gb", 60)),
            "endpoint": smoke_mod.endpoint_body(last["gpu_class"], "13.0")}
        out.append({"model": sm["model"], "vllm_tag": build.get("vllm_tag"), "image": sm["image"],
                    "digest": build.get("digest") or (sm.get("image_build") or {}).get("digest"),
                    "gpu_class": last["gpu_class"], "env": last["env"], "verified_at": sm["finished_at"],
                    "result": f"data/smoke/{sm['image'].rsplit(':', 1)[-1]}.json", "deploy": deploy})
    return out


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
        if r.get("model") and sm["model"] != r["model"]:  # e.g. Step-5-Preview needs multi-GPU; a small model checked the image
            return f"image check with a stand-in model ({smoke_text(sm)})"
        return smoke_text(sm)

    rows = []
    for fam, ev in state["seen"].items():
        if ev.get("skipped"):
            continue
        m = re.match(r"search \((\d+) likes\)", ev.get("match") or "")
        label = ev.get("model") and (ev["model"] + (f" (HF search, {m.group(1)} likes)" if m else ""))
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
    builds = {}
    for f in (DATA / "results").rglob("*.json") if (DATA / "results").exists() else []:
        r = json.loads(f.read_text())
        for k in ("image", "image_pinned"):
            if r.get(k):
                builds[r[k]] = r
    (DATA / "ready.json").write_text(json.dumps(ready_entries(smoke, builds), indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
