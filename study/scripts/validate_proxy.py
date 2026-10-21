"""Check the lag table's proxy on real Serverless endpoints.

The table counts a model as servable once the worker image's pinned vLLM lists
its architecture. This script deploys real runpod/worker-v1-vllm images on a
4090 and sends one chat request, for pairs where the table predicts the older
image fails and the newer one loads.

Usage (from the repo root):
  python3 scripts/validate_proxy.py --check          # auth + connectivity, creates nothing
  python3 scripts/validate_proxy.py --dry-run        # print what would be created
  python3 scripts/validate_proxy.py                  # run all pairs (default 3 at a time)
  python3 scripts/validate_proxy.py --only gemma     # run pairs whose model contains "gemma"
  python3 scripts/validate_proxy.py --cleanup        # delete every day0-* endpoint and template

Auth: RUNPOD_API_KEY from the environment, or --env-file path/to/.env.
The key is never printed.

Cost guard: max 1 worker per endpoint, 5 s idle timeout, and a per-run
timeout (default 25 min). When a run finishes, its endpoint is scaled to 0
workers. Loaded runs are deleted. Runs that failed or timed out are kept at 0
workers, which costs nothing, so their worker logs stay in the console.
--cleanup removes them.
"""
import argparse, csv, json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REST = "https://rest.runpod.io/v1"
JOBS = "https://api.runpod.ai/v2"
# 24 GB+ cards, in priority order. The first canary on 4090s alone never got a
# worker. RTX 5090 is left out because Blackwell support in old vLLM pins would
# add failures unrelated to the architecture.
GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40",
        "NVIDIA RTX A6000", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"]
NO_WORKER_MIN = 12  # give up early if no worker has started by then
PREFIX = "day0-"
OUT = Path(__file__).resolve().parent.parent / "data" / "proxy_validation.csv"

# (model, worker image tag, pinned vLLM, table's prediction, minimum host CUDA)
# Images through v2.18.0 build on nvidia/cuda:12.9.1; v2.24.0+ build on
# vllm/vllm-openai, whose default CUDA_VERSION is 13.0.3.
PAIRS = [
    ("Qwen/Qwen3.5-4B",       "v2.14.0",  "0.16.0", "fail", "12.9"),
    ("Qwen/Qwen3.5-4B",       "v.2.16.0", "0.17.1", "load", "12.9"),
    ("google/gemma-4-E4B-it", "v2.17.0",  "0.18.1", "fail", "12.9"),
    ("google/gemma-4-E4B-it", "v2.18.0",  "0.19.1", "load", "12.9"),
    ("IFM/K2-Horizon-7B",     "v2.27.2",  "0.28.0", "fail", "13.0"),
    ("IFM/K2-Horizon-7B",     "v2.28.0",  "0.30.0", "load", "13.0"),
]
JOB_INPUT = {"input": {"messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                       "sampling_params": {"max_tokens": 8, "temperature": 0}}}


def load_key(env_file):
    if env_file:
        for line in Path(env_file).expanduser().read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if line.startswith("RUNPOD_API_KEY="):
                os.environ["RUNPOD_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set (export it or pass --env-file)")
    return key


def call(method, url, key, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          # Cloudflare in front of the API rejects urllib's default
                                          # User-Agent with error 1010
                                          "User-Agent": "day0-lag/1.0 (+https://github.com/ashwinsreedhar28)",
                                          "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:500]


def items(resp, key):
    """List responses may come back as a bare list or wrapped in an object."""
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        return resp.get(key) or resp.get("data") or []
    return []


def log(tag, msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} [{tag}] {msg}", flush=True)


def run_pair(pair, key, timeout_min, stamp, extra_env=None):
    model, image, vllm, expect, cuda = pair
    tag = f"{model.split('/')[1]}@{image}"
    name = f"{PREFIX}{model.split('/')[1].lower()}-{image.replace('.', '')}-{stamp}"[:60]
    row = {"model": model, "worker_image": image, "pinned_vllm": vllm, "expected": expect,
           "observed": "", "job_status": "", "output": "", "delay_s": "", "exec_s": "", "wall_min": "",
           "endpoint_id": "", "kept_for_logs": "", "extra_env": " ".join(f"{k}={v}" for k, v in (extra_env or {}).items()),
           "note": ""}
    t0 = time.time()
    st, tpl = call("POST", f"{REST}/templates", key, {
        "name": name, "imageName": f"runpod/worker-v1-vllm:{image}", "isServerless": True,
        "containerDiskInGb": 50, "env": {"MODEL_NAME": model, "MAX_MODEL_LEN": "4096", **(extra_env or {})}})
    if not 200 <= st < 300:
        row.update(observed="error", note=f"template create {st}: {tpl}")
        log(tag, row["note"])
        return row
    st, ep = call("POST", f"{REST}/endpoints", key, {
        "templateId": tpl["id"], "name": name, "computeType": "GPU", "gpuTypeIds": GPUS, "gpuCount": 1,
        "workersMin": 0, "workersMax": 1, "idleTimeout": 5, "executionTimeoutMs": 600000,
        "flashboot": False, "minCudaVersion": cuda})
    if not 200 <= st < 300:
        row.update(observed="error", note=f"endpoint create {st}: {ep}")
        log(tag, row["note"])
        call("DELETE", f"{REST}/templates/{tpl['id']}", key)
        return row
    eid = ep["id"]
    row["endpoint_id"] = eid
    log(tag, f"endpoint {eid} created, submitting job")
    st, job = call("POST", f"{JOBS}/{eid}/run", key, JOB_INPUT)
    if not 200 <= st < 300:
        row.update(observed="error", note=f"run {st}: {job}")
    else:
        jid, last, last_workers, seen_workers = job["id"], None, None, set()
        while time.time() - t0 < timeout_min * 60:
            time.sleep(15)
            hst, h = call("GET", f"{JOBS}/{eid}/health", key)
            if 200 <= hst < 300 and isinstance(h, dict):
                workers = {k: v for k, v in (h.get("workers") or {}).items() if v}
                seen_workers |= set(workers)
                if workers != last_workers:
                    last_workers = workers
                    log(tag, f"workers {workers or '{}'} after {(time.time() - t0) / 60:.1f} min")
            if time.time() - t0 > NO_WORKER_MIN * 60 and seen_workers <= {"throttled"}:
                call("POST", f"{JOBS}/{eid}/cancel/{jid}", key)
                row.update(job_status="NO_WORKER",
                           note=f"no worker started in {NO_WORKER_MIN} min; states seen: {sorted(seen_workers) or 'none'}")
                break
            st, s = call("GET", f"{JOBS}/{eid}/status/{jid}", key)
            if not 200 <= st < 300:
                continue
            if s.get("status") != last:
                last = s.get("status")
                log(tag, f"{last} after {(time.time() - t0) / 60:.1f} min")
            if last in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                out = s.get("output") if last == "COMPLETED" else (s.get("error") or s.get("output"))
                row.update(job_status=last, output=json.dumps(out)[:300],
                           delay_s=round((s.get("delayTime") or 0) / 1000), exec_s=round((s.get("executionTime") or 0) / 1000))
                break
        else:
            call("POST", f"{JOBS}/{eid}/cancel/{jid}", key)
            row.update(job_status="NO_RESULT",
                       note=f"no result in {timeout_min} min; worker states seen: {sorted(seen_workers) or 'none'}")
        row["observed"] = {"COMPLETED": "load", "NO_WORKER": "no_worker"}.get(row["job_status"], "fail")
    row["wall_min"] = round((time.time() - t0) / 60, 1)
    call("PATCH", f"{REST}/endpoints/{eid}", key, {"workersMax": 0, "workersMin": 0})
    if row["observed"] in ("load", "no_worker"):  # nothing in the logs worth keeping
        call("DELETE", f"{REST}/endpoints/{eid}", key)
        call("DELETE", f"{REST}/templates/{tpl['id']}", key)
        row["kept_for_logs"] = "no"
    else:
        row["kept_for_logs"] = f"yes, scaled to 0: https://console.runpod.io/serverless/user/endpoint/{eid}"
    verdict = ("inconclusive: no GPU capacity" if row["observed"] == "no_worker"
               else "as predicted" if row["observed"] == expect else "DIFFERS FROM TABLE")
    log(tag, f"observed={row['observed']} expected={expect} ({verdict}) in {row['wall_min']} min")
    return row


def cleanup(key):
    st, eps = call("GET", f"{REST}/endpoints", key)
    if not 200 <= st < 300:
        sys.exit(f"list endpoints {st}: {eps}")
    for ep in items(eps, "endpoints"):
        if ep.get("name", "").startswith(PREFIX):
            call("PATCH", f"{REST}/endpoints/{ep['id']}", key, {"workersMax": 0, "workersMin": 0})
            print("delete endpoint", ep["id"], ep["name"], call("DELETE", f"{REST}/endpoints/{ep['id']}", key)[0])
    st, tpls = call("GET", f"{REST}/templates", key)
    for t in (items(tpls, "templates") if 200 <= st < 300 else []):
        if t.get("name", "").startswith(PREFIX):
            print("delete template", t["id"], t["name"], call("DELETE", f"{REST}/templates/{t['id']}", key)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated substrings of the model id")
    ap.add_argument("--image", default="", help="exact worker image tag, e.g. v.2.16.0")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra worker env var for every run, e.g. GPU_MEMORY_UTILIZATION=0.85")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--timeout-min", type=int, default=25)
    a = ap.parse_args()
    wanted = [w.strip().lower() for w in a.only.split(",")]
    pairs = [p for p in PAIRS if any(w in p[0].lower() for w in wanted) and (not a.image or p[1] == a.image)]
    extra_env = dict(e.split("=", 1) for e in a.env)
    if a.dry_run:
        for p in pairs:
            print(f"{p[0]:24} runpod/worker-v1-vllm:{p[1]:9} vLLM {p[2]}  expect {p[3]}  min cuda {p[4]}")
        return
    key = load_key(a.env_file)
    if a.check:
        st, eps = call("GET", f"{REST}/endpoints", key)
        print(f"GET /v1/endpoints -> {st}" + (f", {len(items(eps, 'endpoints'))} existing endpoints" if 200 <= st < 300 else f": {eps}"))
        return
    if a.cleanup:
        return cleanup(key)
    stamp = datetime.now(timezone.utc).strftime("%m%d%H%M")
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        rows = list(ex.map(lambda p: run_pair(p, key, a.timeout_min, stamp, extra_env), pairs))
    # Merge with earlier runs, which may have been written with fewer columns
    old_rows = list(csv.DictReader(open(OUT))) if OUT.exists() and OUT.stat().st_size else []
    fields = list(rows[0]) + [k for r in old_rows for k in r if k not in rows[0]]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(fields)), restval="")
        w.writeheader()
        w.writerows(old_rows + rows)
    print(f"\n{'model':24} {'image':9} {'vLLM':7} {'expected':8} {'observed':8} wall_min")
    for r in rows:
        print(f"{r['model']:24} {r['worker_image']:9} {r['pinned_vllm']:7} {r['expected']:8} {r['observed']:8} {r['wall_min']}"
              + ("   <-- no worker, rerun" if r["observed"] == "no_worker"
                 else "" if r["observed"] == r["expected"] else "   <-- differs"))
    print(f"\nappended to {OUT}")


if __name__ == "__main__":
    main()
