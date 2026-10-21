"""For each vLLM final release tag, record its commit date and the set of
architecture class names in vllm/model_executor/models/registry.py.
Checking the file at each tag (not `git tag --contains`) catches support that
was cherry-picked onto a release branch under a different SHA."""
import json, re, subprocess, sys
from pathlib import Path

VLLM = Path(sys.argv[1] if len(sys.argv) > 1 else "src/vllm")
OUT = Path("data/vllm_registry_by_tag.json")
REG = "vllm/model_executor/models/registry.py"
TAG_RE = re.compile(r"^v0\.(\d+)\.(\d+)$")

def git(*a):
    return subprocess.run(["git", "-C", str(VLLM), *a], capture_output=True, text=True, check=True).stdout

tags = [t for t in git("tag").split() if TAG_RE.match(t) and int(TAG_RE.match(t)[1]) >= 11]
out = {}
for t in sorted(tags, key=lambda t: tuple(map(int, TAG_RE.match(t).groups()))):
    date = git("log", "-1", "--format=%cs", t).strip()
    src = git("show", f"{t}:{REG}")
    classes = sorted(set(re.findall(r'^\s*"([A-Za-z0-9_]+)":\s*\(', src, re.M)))
    out[t] = {"date": date, "n_classes": len(classes), "classes": classes}
    print(t, date, len(classes), file=sys.stderr)
OUT.write_text(json.dumps(out, indent=1))
