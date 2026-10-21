"""GPU smoke test for a built day-0 worker image, with a fallback ladder.

    python3 pipeline/smoke.py --image ghcr.io/ashwinsreedhar28/worker-vllm:day0-gemma4 \
        --model google/gemma-4-E4B-it [--gpu 24gb] [--max-minutes 40] [--env-file ~/.env]

Deploys the image as a Serverless endpoint (max 1 worker), sends one chat
request, and tears it down. If a rung fails in a way a setting can fix, it
moves to the next rung, mirroring the failures seen in study/:

  remote code    config ships custom code (CPU check says *+remote_code, or the
                 worker crash-loops on a trust_remote_code error)  -> TRUST_REMOTE_CODE=true
  first-request  engine dies on the first request (Qwen3.5: Triton autotune
  OOM            had no headroom at GPU_MEMORY_UTILIZATION=0.95)   -> 0.85
  startup OOM    the worker answers with a startup_error about GPU memory -> next GPU class

A worker that keeps restarting with no answer is treated as an unrecognised
startup failure; its logs are in the console (endpoint kept at 0 workers).

Writes data/smoke/<image tag>.json. Cost guard: --max-minutes across all rungs,
one worker at a time, 5 s idle timeout, endpoints scaled to 0 on every exit path.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_api as rp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "day0-smoke-"
GPU_CLASSES = {
    # Priority order inside a class. RTX 5090 left out: old vLLM pins may lack Blackwell support.
    "24gb": ["NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40",
             "NVIDIA RTX A6000", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"],
    "48gb": ["NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40", "NVIDIA RTX A6000"],
    "80gb": ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3",
             "NVIDIA H100 NVL"],
    # A ~40B model in bf16 (~80 GB of weights, e.g. RecursiveSI/flint) needs one of these.
    "141gb": ["NVIDIA H200", "NVIDIA H200 NVL"],
    "180gb": ["NVIDIA B200"],
}
NEXT_GPU = {"24gb": "48gb", "48gb": "80gb", "80gb": "141gb", "141gb": "180gb"}
JOB_INPUT = {"input": {"messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                       "sampling_params": {"max_tokens": 8, "temperature": 0}}}
CRASH_SIGNALS = 2  # unhealthy states or running->not-running drops before a rung is a crash loop


def log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def classify_failure(job_status, output, crash_signals):
    text = json.dumps(output) if output is not None else ""
    if job_status == "FAILED" and "startup_error" in text and ("GPU memory" in text or "KV cache" in text):
        return "startup_oom"
    if job_status == "FAILED" and ("EngineCore encountered an issue" in text or "out of memory" in text.lower()):
        return "first_request_failure"
    # Raw vLLM error, or the worker's startup_error once fix/remote-code-startup-error lands.
    if "trust_remote_code" in text.lower() or "custom code" in text or "own modeling code" in text:
        return "needs_remote_code"
    if job_status == "FAILED" and "startup_error" in text:
        return "startup_error"
    if crash_signals >= CRASH_SIGNALS:
        return "crash_loop"
    return "no_result"


def run_rung(key, image, model, env, gpu_class, min_cuda, deadline, label, disk_gb=60):
    rung = {"env": env, "gpu_class": gpu_class, "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    name = f"{PREFIX}{label}-{int(time.time())}"[:60]
    st, tpl = rp.call("POST", f"{rp.REST}/templates", key, {
        "name": name, "imageName": image, "isServerless": True, "containerDiskInGb": disk_gb,
        "env": {"MODEL_NAME": model, "MAX_MODEL_LEN": "4096", **env}})
    if not rp.ok(st):
        return {**rung, "outcome": "error", "detail": f"template {st}: {tpl}"}
    st, ep = rp.call("POST", f"{rp.REST}/endpoints", key, {
        "templateId": tpl["id"], "name": name, "computeType": "GPU", "gpuTypeIds": GPU_CLASSES[gpu_class],
        "gpuCount": 1, "workersMin": 0, "workersMax": 1, "idleTimeout": 5, "executionTimeoutMs": 600000,
        "flashboot": False, "minCudaVersion": min_cuda})
    if not rp.ok(st):
        rp.call("DELETE", f"{rp.REST}/templates/{tpl['id']}", key)
        return {**rung, "outcome": "error", "detail": f"endpoint {st}: {ep}"}
    eid = ep["id"]
    rung["endpoint_id"] = eid
    t0, status, output, crash_signals, last_workers, seen = time.time(), None, None, 0, None, set()
    try:
        st, job = rp.call("POST", f"{rp.JOBS}/{eid}/run", key, JOB_INPUT)
        if not rp.ok(st):
            return {**rung, "outcome": "error", "detail": f"run {st}: {job}"}
        jid = job["id"]
        log(f"[{label}] endpoint {eid}, env {env or '{}'}, {gpu_class}")
        while time.time() < deadline:
            time.sleep(15)
            hst, h = rp.call("GET", f"{rp.JOBS}/{eid}/health", key)
            if rp.ok(hst) and isinstance(h, dict):
                workers = {k: v for k, v in (h.get("workers") or {}).items() if v}
                if workers != last_workers:
                    # A worker that dies and restarts shows up either as unhealthy
                    # or as running -> ready/idle with the job still queued (study/
                    # K2-Horizon on v2.28.0 never went unhealthy).
                    if workers.get("unhealthy") or ((last_workers or {}).get("running") and not workers.get("running")):
                        crash_signals += 1
                    seen |= set(workers)
                    last_workers = workers
                    log(f"[{label}] workers {workers or '{}'} at {(time.time() - t0) / 60:.1f} min")
            if crash_signals >= CRASH_SIGNALS:
                status = "CRASH_LOOP"
                break
            st, s = rp.call("GET", f"{rp.JOBS}/{eid}/status/{jid}", key)
            if rp.ok(st) and s.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                status, output = s["status"], (s.get("output") if s["status"] == "COMPLETED" else s.get("error") or s.get("output"))
                rung.update(delay_s=round((s.get("delayTime") or 0) / 1000), exec_s=round((s.get("executionTime") or 0) / 1000))
                break
        if status is None:
            status = "DEADLINE"
        if status not in ("COMPLETED", "FAILED"):
            rp.call("POST", f"{rp.JOBS}/{eid}/cancel/{jid}", key)
    finally:
        rp.call("PATCH", f"{rp.REST}/endpoints/{eid}", key, {"workersMax": 0, "workersMin": 0})
    rung.update(job_status=status, output=json.dumps(output)[:600] if output is not None else None,
                minutes=round((time.time() - t0) / 60, 1), worker_states=sorted(seen))
    if status == "COMPLETED":
        rung["outcome"] = "pass"
        rp.call("DELETE", f"{rp.REST}/endpoints/{eid}", key)
        rp.call("DELETE", f"{rp.REST}/templates/{tpl['id']}", key)
    else:
        rung["outcome"] = classify_failure(status, output, crash_signals)
        rung["logs"] = f"https://console.runpod.io/serverless/user/endpoint/{eid}"
    log(f"[{label}] {rung['outcome']} ({status}) in {rung['minutes']} min")
    return rung


def next_rung(env, gpu_class, outcome, remote_code_tried, max_gpu="180gb"):
    """The ladder: which setting to change after a failure, or None to stop."""
    if outcome in ("needs_remote_code", "crash_loop") and not remote_code_tried:
        return {**env, "TRUST_REMOTE_CODE": "true"}, gpu_class
    if outcome == "first_request_failure" and env.get("GPU_MEMORY_UTILIZATION") != "0.85":
        return {**env, "GPU_MEMORY_UTILIZATION": "0.85"}, gpu_class
    order = list(GPU_CLASSES)
    if (outcome == "startup_oom" and gpu_class in NEXT_GPU
            and order.index(NEXT_GPU[gpu_class]) <= order.index(max_gpu)):
        return env, NEXT_GPU[gpu_class]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="24gb", choices=GPU_CLASSES)
    ap.add_argument("--max-gpu", default="48gb", choices=GPU_CLASSES,
                    help="largest class a startup OOM may climb to (H200/B200 cost several times a 4090)")
    ap.add_argument("--min-cuda", default="13.0", help="vllm/vllm-openai images build on CUDA 13.0")
    ap.add_argument("--remote-code", action="store_true", help="start with TRUST_REMOTE_CODE=true (CPU check said so)")
    ap.add_argument("--max-minutes", type=int, default=40, help="wall-clock cap across all rungs")
    ap.add_argument("--disk-gb", type=int, default=60,
                    help="container disk; weights download here without a network volume (~2x params in GB for bf16, plus ~20 for the image)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra worker env for every rung, e.g. --set ATTENTION_BACKEND=FLASHINFER (repeatable)")
    ap.add_argument("--env-file")
    a = ap.parse_args()
    if any("=" not in s for s in a.set):
        ap.error("--set takes KEY=VALUE")
    key = rp.load_key(a.env_file)

    deadline = time.time() + a.max_minutes * 60
    env, gpu = dict(s.split("=", 1) for s in a.set), a.gpu
    if a.remote_code:
        env["TRUST_REMOTE_CODE"] = "true"
    label = a.image.rsplit(":", 1)[-1].replace(".", "")[:30]
    rungs = []
    while time.time() < deadline:
        r = run_rung(key, a.image, a.model, env, gpu, a.min_cuda, deadline, f"{label}-{len(rungs)}", a.disk_gb)
        rungs.append(r)
        if r["outcome"] == "pass":
            break
        nxt = next_rung(env, gpu, r["outcome"], "TRUST_REMOTE_CODE" in env, a.max_gpu)
        if nxt is None:
            break
        env, gpu = nxt

    result = {"image": a.image, "model": a.model, "passed": bool(rungs) and rungs[-1]["outcome"] == "pass",
              "passing_env": rungs[-1]["env"] if rungs and rungs[-1]["outcome"] == "pass" else None,
              "passing_gpu_class": rungs[-1]["gpu_class"] if rungs and rungs[-1]["outcome"] == "pass" else None,
              "rungs": rungs, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = ROOT / "data" / "smoke" / f"{a.image.rsplit(':', 1)[-1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    log(f"{'PASS' if result['passed'] else 'FAIL'} after {len(rungs)} rung(s) -> {out.relative_to(ROOT)}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
