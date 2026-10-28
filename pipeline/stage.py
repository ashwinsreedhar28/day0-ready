"""Put a Hugging Face model's weights on a new network volume with a CPU pod, before any GPU boots.

    python3 pipeline/stage.py --model MiniMaxAI/MiniMax-M3.1-Flash --data-centers US-NC-1,CA-MTL-4 \
        --volume-gb 300 --out stage.json
    python3 pipeline/stage.py --cleanup          # every day0-stage- pod, smoke endpoint on such a volume, volume

Why: a Serverless worker that downloads 233 GiB onto its container disk was stopped ~8.4 min after
vLLM started, twice, a few shards into loading, and each restart downloaded everything again
(MiniMax-M3.1-Flash on 2x H200, Oct 28; docs/multi-gpu.md). With the weights already on a network
volume a boot only loads them.

The pod mounts the volume at /runpod-volume, the worker image's BASE_PATH, and downloads into
/runpod-volume/huggingface-cache/hub, the image's HF_HUB_CACHE, so vLLM in the worker finds the
files where it looks. The pod serves its progress as status.json on port 8000 through Runpod's HTTP
proxy; the runner polls that, then deletes the pod. Only names starting with day0-stage- are ever
deleted.
"""

import argparse
import base64
import inspect
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_api as rp  # noqa: E402

PREFIX = "day0-stage-"
SMOKE_PREFIX = "day0-smoke-"
HUB_CACHE = "/runpod-volume/huggingface-cache/hub"   # worker-vllm's HF_HUB_CACHE with a volume
IMAGE = "python:3.12-slim"
CPU_FLAVORS = ["cpu3c", "cpu5c", "cpu3g", "cpu5g", "cpu3m", "cpu5m"]
# POST /pods takes only these dataCenterIds (REST spec, Oct 28); the H200 data centers CA-MTL-4,
# EUR-IS-4 and US-CO-1 aren't among them, so a CPU pod can't stage weights there.
POD_DATA_CENTERS = {"EU-RO-1", "CA-MTL-1", "EU-SE-1", "US-IL-1", "EUR-IS-1", "EU-CZ-1", "US-TX-3", "EUR-IS-2",
                    "US-KS-2", "US-GA-2", "US-WA-1", "US-TX-1", "CA-MTL-3", "EU-NL-1", "US-TX-4", "US-CA-2",
                    "US-NC-1", "OC-AU-1", "US-DE-1", "EUR-IS-3", "CA-MTL-2", "AP-JP-1", "EUR-NO-1", "EU-FR-1",
                    "US-KS-3", "US-GA-1", "AP-IN-1", "US-MD-1", "US-MO-1", "US-MO-2"}
# vLLM loads safetensors; skip duplicate formats a repo may also carry.
IGNORE = ["original/*", "*.pth", "*.pt", "*.bin", "*.gguf", "*.onnx", "*.msgpack", "*.h5"]

# The pod's completion check, shared with the tests: a download only counts when the bytes in the
# cache's blobs/ match what the Hub lists for the files we asked for. Oct 28 (MiniMax try 2): the first
# version reported done after 247 s with 17 MB on the volume, because it trusted snapshot_download
# returning without an error.
def blob_bytes(repo_dir):
    """Bytes of finished files in a Hugging Face cache repo's blobs/ (not *.incomplete)."""
    import os
    total = 0
    for root, _, files in os.walk(os.path.join(repo_dir, "blobs")):
        for name in files:
            if not name.endswith(".incomplete"):
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
    return total


def expected_bytes(siblings, ignore):
    """Total size of the repo files snapshot_download fetches: (filename, size) pairs minus ignore patterns."""
    import fnmatch
    return sum(size or 0 for name, size in siblings if not any(fnmatch.fnmatch(name, p) for p in ignore))


def complete(have, want):
    return bool(want) and have >= want * 0.999


# Runs inside the pod (python:3.12-slim): install huggingface_hub, download, report progress.
# HF_HUB_DISABLE_XET=1 keeps the download on plain HTTP straight into the volume's cache; the hf_xet
# path is the likely cause of the Oct 28 failure (not verified).
POD_MAIN = r'''
import functools, http.server, json, os, subprocess, sys, threading, time
ST = "/status"
os.makedirs(ST, exist_ok=True)
state = {"state": "installing", "t0": time.time()}

def write(**kw):
    state.update(kw, at=time.time())
    with open(ST + "/status.tmp", "w") as f:
        json.dump(state, f)
    os.replace(ST + "/status.tmp", ST + "/status.json")

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ST)
threading.Thread(target=http.server.ThreadingHTTPServer(("", 8000), handler).serve_forever, daemon=True).start()
write()
try:
    p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"],
                       capture_output=True, text=True, timeout=900)
    if p.returncode:
        raise RuntimeError("pip: " + (p.stderr or p.stdout)[-600:])
    from huggingface_hub import HfApi, snapshot_download
    model, ignore = os.environ["MODEL"], json.loads(os.environ["IGNORE"])
    info = HfApi().model_info(model, files_metadata=True)
    want = expected_bytes([(f.rfilename, f.size) for f in info.siblings or []], ignore)
    repo = os.path.join(os.environ["HF_HUB_CACHE"], "models--" + model.replace("/", "--"))
    done = {}

    def run():
        try:
            done["path"] = snapshot_download(model, revision=info.sha, ignore_patterns=ignore, max_workers=16)
        except Exception as e:
            done["error"] = repr(e)[:800]

    t = threading.Thread(target=run)
    t.start()
    write(state="downloading", started=time.time(), expected=want, revision=info.sha)
    while t.is_alive():
        write(state="downloading", bytes=blob_bytes(repo))
        t.join(15)
    have = blob_bytes(repo)
    if "error" in done:
        write(state="error", detail=done["error"], bytes=have)
    elif not complete(have, want):
        write(state="error", detail=f"download returned with {have} of {want} bytes in blobs/", bytes=have)
    else:
        write(state="done", bytes=have, path=done["path"], seconds=round(time.time() - state["started"]))
except Exception as e:
    write(state="error", detail=repr(e)[:800])
while True:
    time.sleep(3600)
'''
POD_SCRIPT = "\n\n".join(inspect.getsource(f) for f in (blob_bytes, expected_bytes, complete)) + POD_MAIN


def log(msg):
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def pod_body(name, volume_id, dc, model, vcpu=8):
    """POST /v1/pods body: a CPU pod on the volume that runs POD_SCRIPT."""
    return {"name": name, "computeType": "CPU", "cpuFlavorIds": CPU_FLAVORS, "cpuFlavorPriority": "availability",
            "vcpuCount": vcpu, "dataCenterIds": [dc], "networkVolumeId": volume_id,
            "volumeMountPath": "/runpod-volume", "containerDiskInGb": 20, "imageName": IMAGE, "ports": ["8000/http"],
            # No spaces in any element, in case the command array is joined into one string.
            "dockerStartCmd": ["python3", "-c",
                               "exec(__import__('base64').b64decode(__import__('os').environ['STAGE_PY']))"],
            "env": {"STAGE_PY": base64.b64encode(POD_SCRIPT.encode()).decode(), "MODEL": model,
                    "HF_HUB_CACHE": HUB_CACHE, "HF_HUB_DISABLE_XET": "1", "IGNORE": json.dumps(IGNORE)}}


def status(pod_id, timeout=15):
    """The pod's status.json through Runpod's HTTP proxy; None until it answers."""
    url = f"https://{pod_id}-8000.proxy.runpod.net/status.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": rp.UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def delete_pod(key, pod_id):
    st, r = rp.call("DELETE", f"{rp.REST}/pods/{pod_id}", key)
    log(f"pod {pod_id} " + ("deleted" if rp.ok(st) else f"NOT deleted: {st} {str(r)[:160]}"))
    return rp.ok(st)


def delete_volume(key, vid, tries=6):
    for i in range(tries):
        st, r = rp.call("DELETE", f"{rp.REST}/networkvolumes/{vid}", key)
        if rp.ok(st):
            log(f"volume {vid} deleted")
            return True
        log(f"volume {vid} not deleted yet: {st} {str(r)[:160]}")
        time.sleep(20 * (i + 1))
    return False


def stage(key, model, dcs, gb, max_minutes, vcpu=8, poll=20, sleep=time.sleep, clock=time.time):
    """Create the volume and the pod in the first data center that takes both; wait for the download.

    Returns a result dict. On "done" the volume is kept for the GPU boot (the caller deletes it);
    on anything else the pod and the volume are deleted here.
    """
    res = {"model": model, "state": "no_data_center", "tried": []}
    for dc in dcs:
        if dc not in POD_DATA_CENTERS:
            res["tried"].append({"dc": dc, "pod": "not a pod data center in the REST API"})
            log(f"{dc}: no CPU pods through the REST API; skipped")
            continue
        name = f"{PREFIX}{int(clock())}"
        st, v = rp.call("POST", f"{rp.REST}/networkvolumes", key, {"name": name, "size": gb, "dataCenterId": dc})
        if not rp.ok(st):
            res["tried"].append({"dc": dc, "volume": f"{st} {str(v)[:160]}"})
            log(f"{dc}: volume {st}")
            continue
        # Oct 28, 06:18 UTC: no 8-vCPU CPU pod in either H200 data center ("no longer any instances
        # available"), so smaller sizes are tried before giving up on a data center.
        for n in sorted({vcpu, 4, 2}, reverse=True):
            st, pod = rp.call("POST", f"{rp.REST}/pods", key, pod_body(name, v["id"], dc, model, n))
            if rp.ok(st) and isinstance(pod, dict) and pod.get("id"):
                res["vcpu"] = n
                break
            log(f"{dc}: {n}-vCPU pod {st} {str(pod)[:120]}")
        if not rp.ok(st) or not isinstance(pod, dict) or not pod.get("id"):
            res["tried"].append({"dc": dc, "pod": f"{st} {str(pod)[:160]}"})
            delete_volume(key, v["id"])
            continue
        res.update(data_center=dc, volume_id=v["id"], pod_id=pod["id"], cost_per_hr=pod.get("costPerHr"))
        log(f"{dc}: volume {v['id']} ({gb} GB), CPU pod {pod['id']} (${pod.get('costPerHr')}/h)")
        break
    else:
        return res

    t0, last, s = clock(), None, None
    while clock() - t0 < max_minutes * 60:
        sleep(poll)
        s = status(res["pod_id"])
        if s and s.get("state") != last or (s and int(clock() - t0) % 120 < poll):
            gib = (s.get("bytes") or 0) / 2 ** 30
            log(f"pod: {s.get('state')}, {gib:.1f} GiB on the volume")
            last = s.get("state")
        if s and s.get("state") in ("done", "error"):
            break
    res["state"] = (s or {}).get("state") if s and s.get("state") in ("done", "error") else "timeout"
    res["minutes"] = round((clock() - t0) / 60, 1)
    if s:
        res.update(bytes=s.get("bytes"), expected=s.get("expected"), revision=s.get("revision"),
                   download_s=s.get("seconds"), detail=s.get("detail"))
        if res["state"] == "done" and not complete(res["bytes"] or 0, res["expected"] or 0):
            res.update(state="error", detail=f"pod said done with {res['bytes']} of {res['expected']} bytes")
    if res.get("cost_per_hr"):
        res["est_cost_usd"] = round(res["cost_per_hr"] * res["minutes"] / 60, 3)
    delete_pod(key, res["pod_id"])
    if res["state"] != "done":
        delete_volume(key, res["volume_id"])
        res["volume_deleted"] = True
    return res


def cleanup(key, volume_id=None, keep_endpoints=False):
    """Delete day0-stage- pods, then day0-smoke- endpoints on a day0-stage- volume (scaled to 0 first),
    their templates, then the volumes. With volume_id, only that volume and what uses it.

    keep_endpoints (after a failed smoke test): detach the volume from each smoke endpoint and keep
    the endpoint at 0 workers, which costs nothing, so the console still shows its worker log
    (MiniMax try 2's log went with its endpoint). If Runpod still refuses to delete the volume, the
    endpoint is deleted after all: a kept volume would bill. Returns the number of things left."""
    left = 0
    st, vl = rp.call("GET", f"{rp.REST}/networkvolumes", key)
    if not rp.ok(st):
        log(f"cleanup: volume list failed ({st} {str(vl)[:120]}); nothing deleted")
        return 1
    vols = [v for v in rp.items(vl, "networkVolumes")
            if str(v.get("name", "")).startswith(PREFIX) and (volume_id is None or v.get("id") == volume_id)]
    ids = {v["id"] for v in vols}
    st, pods = rp.call("GET", f"{rp.REST}/pods", key)
    if not rp.ok(st):
        log(f"cleanup: pod list failed ({st} {str(pods)[:120]})")
        left += 1
    for p in rp.items(pods, "pods") if rp.ok(st) else []:
        if str(p.get("name", "")).startswith(PREFIX) and (volume_id is None or
                                                           (p.get("networkVolume") or {}).get("id") in ids):
            left += not delete_pod(key, p["id"])
    st, eps = rp.call("GET", f"{rp.REST}/endpoints", key)
    if not rp.ok(st):
        log(f"cleanup: endpoint list failed ({st} {str(eps)[:120]})")
        left += 1
    on_vol = []
    for e in rp.items(eps, "endpoints") if rp.ok(st) else []:
        on = {e.get("networkVolumeId"), *(e.get("networkVolumeIds") or [])}
        if str(e.get("name", "")).startswith(SMOKE_PREFIX) and on & ids:
            on_vol.append(e)
    kept = []
    for e in on_vol:
        rp.call("PATCH", f"{rp.REST}/endpoints/{e['id']}", key, {"workersMin": 0, "workersMax": 0})
        rp.call("POST", f"{rp.JOBS}/{e['id']}/purge-queue", key)
        if keep_endpoints:
            st2, r = rp.call("PATCH", f"{rp.REST}/endpoints/{e['id']}", key,
                             {"networkVolumeId": None, "networkVolumeIds": []})
            log(f"endpoint {e['id']} ({e.get('name')}) kept at 0 workers for its log; volume detach "
                + ("accepted" if rp.ok(st2) else f"refused: {st2} {str(r)[:120]}"))
            kept.append(e)
            continue
        left += not delete_endpoint(key, e)
    for v in vols:
        if delete_volume(key, v["id"], tries=2 if kept else 6):
            continue
        if kept:   # the detach didn't take: the endpoint goes, so the volume can
            for e in kept:
                left += not delete_endpoint(key, e)
            kept = []
        left += not delete_volume(key, v["id"])
    return left


def delete_endpoint(key, e):
    """Delete a smoke endpoint and its template. Runpod refuses until the workers are gone, so this
    retries for up to 5 minutes."""
    st, r = None, None
    for _ in range(20):
        st, r = rp.call("DELETE", f"{rp.REST}/endpoints/{e['id']}", key)
        if rp.ok(st):
            break
        time.sleep(15)
    log(f"endpoint {e['id']} ({e.get('name')}) " + ("deleted" if rp.ok(st) else f"NOT deleted: {st} {str(r)[:120]}"))
    if rp.ok(st) and e.get("templateId"):
        rp.call("DELETE", f"{rp.REST}/templates/{e['templateId']}", key)
    return rp.ok(st)


def data_centers_with(avail, gpu_types):
    """Data centers that list one of the types in stock (the catalog query only covers ones with
    standard network volumes) and can run a CPU pod, most stock first."""
    import sweep  # noqa: PLC0415 - only for the catalog helpers
    have = [(max(sweep.LEVELS[g[t]] for t in sweep.types_in(g, gpu_types)), dc)
            for dc, g in (avail or {}).items() if sweep.types_in(g, gpu_types) and dc in POD_DATA_CENTERS]
    return [dc for _, dc in sorted(have, key=lambda x: (-x[0], x[1]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--data-centers", help="comma-separated, tried in order")
    ap.add_argument("--gpu-types", help="instead of --data-centers: every data center listing one of these "
                    "(comma-separated type ids) in stock, most first")
    ap.add_argument("--volume-gb", type=int, default=300)
    ap.add_argument("--max-minutes", type=float, default=45, help="limit for the download")
    ap.add_argument("--vcpu", type=int, default=8)
    ap.add_argument("--out", help="write the result JSON here")
    ap.add_argument("--cleanup", action="store_true", help="delete day0-stage- pods, their smoke endpoints, volumes")
    ap.add_argument("--volume-id", help="with --cleanup: only this volume and what uses it")
    ap.add_argument("--keep-endpoints", action="store_true",
                    help="with --cleanup: keep smoke endpoints at 0 workers (logs stay readable), detach the volume")
    ap.add_argument("--env-file")
    a = ap.parse_args()
    key = rp.load_key(a.env_file)
    if a.cleanup:
        sys.exit(1 if cleanup(key, a.volume_id, a.keep_endpoints) else 0)
    if not a.model or not (a.data_centers or a.gpu_types):
        ap.error("--model and one of --data-centers / --gpu-types are required")
    if a.data_centers:
        dcs = [d.strip() for d in a.data_centers.split(",") if d.strip()]
    else:
        import sweep  # noqa: PLC0415
        dcs = data_centers_with(sweep.availability(key), [t.strip() for t in a.gpu_types.split(",") if t.strip()])
        log(f"data centers with {a.gpu_types} in stock: {dcs or 'none'}")
    res = stage(key, a.model, dcs, a.volume_gb, a.max_minutes, a.vcpu)
    log(f"result: {json.dumps(res)}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    sys.exit(0 if res["state"] == "done" else 1)


if __name__ == "__main__":
    main()
