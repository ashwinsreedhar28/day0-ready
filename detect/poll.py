"""Detect new open-weight launches that vLLM has published a day-0 image for.

Two sources:
  1. vllm/vllm-openai tags on Docker Hub. vLLM pushes a model-named tag
     (step-5-preview, muse-spark-open, gemma4, ...) for big launches, usually
     on release day. Each new tag family is a launch event.
  2. New repos from a watchlist of labs on Hugging Face, used to attach the
     model id the tag is for (and to see launches that have no day-0 tag).

    python3 detect/poll.py                 # update data/state.json, print new events
    python3 detect/poll.py --since 2026-10-01 --dry-run

Stdlib only, so it runs on a bare Actions runner. Writes $GITHUB_OUTPUT
`new_events` (JSON list) when that variable is set.
"""

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "state.json"
UA = {"User-Agent": "day0-ready/0.1 (+https://github.com/ashwinsreedhar28/day0-ready)", "Accept": "application/json"}

HUB_TAGS = "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100&ordering=last_updated"
HF_MODELS = "https://huggingface.co/api/models?author={org}&sort=createdAt&direction=-1&limit=30"
HF_SEARCH = "https://huggingface.co/api/models?search={q}&sort=likes&direction=-1&limit=30"
SEARCH_WINDOW_DAYS = (14, 3)  # repo created at most 14 days before / 3 days after vLLM's tag

# Labs whose launches the study tracked, plus a few that ship open weights often.
WATCHLIST = [
    "Qwen", "deepseek-ai", "zai-org", "moonshotai", "google", "meta-models", "MiniMaxAI",
    "stepfun-ai", "XiaomiMiMo", "tencent", "inclusionAI", "thinkingmachines", "IFM",
    "CohereLabs", "JetBrains", "poolside", "mistralai", "nvidia", "LGAI-EXAONE", "Cloudflare",
    "openai", "microsoft", "ibm-granite", "allenai",
]

# Tags that are releases or builds, not model launches.
NOT_A_LAUNCH = re.compile(
    r"^(v?\d+\.\d+\.\d+.*|latest.*|nightly.*|cu\d+-nightly.*|.*-nightly-.*|.*-dev-\d{8}.*|test-.*|dev-.*|"
    r"vllm-.*|.*-perf-.*|[0-9a-f]{40})$")
# Per-arch / per-CUDA / date variants of one launch tag, stripped to find its family.
VARIANT = re.compile(r"(-(x86[-_]64|amd64|arm64|aarch64|aarc64|aarch|cu\d{3}|\d{4,6}|[0-9a-f]{7,8}))+$")
# Quantized or converted repos, not the launch itself.
DERIVATIVE = re.compile(r"(gguf|awq|gptq|fp8|nvfp4|mxfp4|int4|int8|mlx|bnb|exl2|-pte|executorch|onnx|"
                        r"dflash|eagle|-mtp|-draft|assistant)", re.I)


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def tag_family(tag):
    """'gemma4-0409-x86_64-cu130' -> 'gemma4', 'step-5-preview-arm64-cu130' -> 'step-5-preview'."""
    return VARIANT.sub("", tag)


def hub_launch_tags(pages=3):
    """{family: {"tag": best tag to build FROM, "pushed": earliest push}} from recent tags."""
    fams, url = {}, HUB_TAGS
    for _ in range(pages):
        data = get_json(url)
        for t in data.get("results", []):
            name, pushed = t["name"], t.get("tag_last_pushed") or t.get("last_updated")
            if NOT_A_LAUNCH.match(name):
                continue
            fam = tag_family(name)
            if not fam or NOT_A_LAUNCH.match(fam):
                continue
            f = fams.setdefault(fam, {"tag": None, "pushed": pushed, "variants": []})
            f["variants"].append(name)
            f["pushed"] = min(f["pushed"], pushed)
        url = data.get("next")
        if not url:
            break
    for fam, f in fams.items():
        # Build FROM the multi-arch family tag if it exists, else an amd64 variant.
        amd = [v for v in f["variants"] if re.search(r"x86[-_]64|amd64", v)]
        f["tag"] = fam if fam in f["variants"] else (sorted(amd, key=len)[0] if amd else sorted(f["variants"], key=len)[0])
    return fams


def hf_new_repos(orgs=WATCHLIST):
    repos = []
    for org in orgs:
        try:
            for m in get_json(HF_MODELS.format(org=urllib.parse.quote(org))):
                if not DERIVATIVE.search(m["id"]):
                    repos.append({"id": m["id"], "created": m.get("createdAt"), "gated": m.get("gated", False)})
        except Exception as e:  # noqa: BLE001 - one bad org shouldn't stop the poll
            print(f"warn: {org}: {e}", file=sys.stderr)
    return repos


def match_repo(family, repos):
    """Best HF repo for a vLLM tag family: the family's normalized name, minus a
    trailing date, must be a prefix of the repo name. Prefer the shortest repo
    name (the base launch, not a size variant), then the newest."""
    key = norm(re.sub(r"[-_]\d{4,6}$", "", family))
    if len(key) < 4:
        return None
    hits = [r for r in repos if norm(r["id"].split("/")[1]).startswith(key)]
    if not hits:
        return None
    return sorted(hits, key=lambda r: (len(r["id"]), r.get("created") or ""), reverse=False)[0]["id"]


def resolve_model(family, pushed, repos):
    model = match_repo(family, repos)
    if model:
        return {"model": model, "match": "watchlist"}
    hit = search_repo(family, pushed)
    if hit:
        return {"model": hit[0], "match": f"search ({hit[1]} likes)"}
    return {"model": None}


def search_repo(family, pushed, search=None):
    """Fallback when no watchlist lab matches (e.g. `flint` -> RecursiveSI/flint,
    a lab nobody had listed): search Hugging Face for the family name and take the
    most-liked base repo whose name starts with it and that was created near
    vLLM's tag push. Returns (repo_id, likes) or None."""
    q = re.sub(r"[-_]\d{4,6}$", "", family)
    key = norm(q)
    if len(key) < 4:
        return None
    try:
        found = search(q) if search else get_json(HF_SEARCH.format(q=urllib.parse.quote(q)))
    except Exception as e:  # noqa: BLE001
        print(f"warn: search {q}: {e}", file=sys.stderr)
        return None
    t = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
    lo, hi = t - timedelta(days=SEARCH_WINDOW_DAYS[0]), t + timedelta(days=SEARCH_WINDOW_DAYS[1])
    hits = []
    for m in found:
        created = m.get("createdAt")
        if (DERIVATIVE.search(m["id"]) or not created
                or not norm(m["id"].split("/")[1]).startswith(key)
                or not lo <= datetime.fromisoformat(created.replace("Z", "+00:00")) <= hi):
            continue
        hits.append((m.get("likes", 0), m["id"]))
    return max(hits)[::-1] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="only report tag families first pushed on/after this date (YYYY-MM-DD)")
    ap.add_argument("--dry-run", action="store_true", help="don't write data/state.json")
    a = ap.parse_args()

    state = json.loads(STATE.read_text()) if STATE.exists() else {"seen": {}}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fams = hub_launch_tags()
    repos = hf_new_repos()

    new = []
    for fam, f in sorted(fams.items(), key=lambda kv: kv[1]["pushed"]):
        if fam in state["seen"]:
            continue
        if a.since and f["pushed"][:10] < a.since:
            state["seen"][fam] = {"first_seen": now, "skipped": "before --since", "vllm_tag": f["tag"]}
            continue
        ev = {"family": fam, "vllm_tag": f["tag"], "vllm_tag_pushed": f["pushed"], "first_seen": now}
        ev.update(resolve_model(fam, f["pushed"], repos))
        state["seen"][fam] = ev
        new.append(ev)

    # Launches seen earlier without a model: the repo may have gone public since.
    for fam, ev in state["seen"].items():
        if ev.get("skipped") or ev.get("model") or ev in new:
            continue
        found = resolve_model(fam, ev["vllm_tag_pushed"], repos)
        if found.get("model"):
            ev.update(found, matched_at=now)
            new.append(ev)

    state["last_poll"] = now
    if not a.dry_run:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    print(json.dumps(new, indent=1))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"new_events={json.dumps(new)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
