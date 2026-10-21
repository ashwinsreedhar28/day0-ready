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

Writes data/smoke/<image tag>.json; a pass also records the exact template and
endpoint bodies (data/ready.json is built from them). Cost guard: --max-minutes
across all rungs, one worker at a time, 5 s idle timeout, endpoints scaled to 0
on every exit path, and in CI (.github/workflows/smoke.yml) --gpu auto,
--max-gpu, --budget-usd and --no-remote-code.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_api as rp  # noqa: E402
import sizing  # noqa: E402

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


def template_body(image, model, env, disk_gb):
    """POST /v1/templates body (minus the name); recorded so a passing rung can be redeployed as is."""
    return {"imageName": image, "isServerless": True, "containerDiskInGb": disk_gb,
            "env": {"MODEL_NAME": model, "MAX_MODEL_LEN": "4096", **env}}


def endpoint_body(gpu_class, min_cuda):
    """POST /v1/endpoints body (minus name and templateId), with the smoke test's cost limits."""
    return {"computeType": "GPU", "gpuTypeIds": GPU_CLASSES[gpu_class], "gpuCount": 1, "workersMin": 0,
            "workersMax": 1, "idleTimeout": 5, "executionTimeoutMs": 600000, "flashboot": False,
            "minCudaVersion": min_cuda}


def run_rung(key, image, model, env, gpu_class, min_cuda, deadline, label, disk_gb=60):
    rung = {"env": env, "gpu_class": gpu_class, "disk_gb": disk_gb,
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    name = f"{PREFIX}{label}-{int(time.time())}"[:60]
    st, tpl = rp.call("POST", f"{rp.REST}/templates", key, {"name": name, **template_body(image, model, env, disk_gb)})
    if not rp.ok(st):
        return {**rung, "outcome": "error", "detail": f"template {st}: {tpl}"}
    st, ep = rp.call("POST", f"{rp.REST}/endpoints", key, {
        "templateId": tpl["id"], "name": name, **endpoint_body(gpu_class, min_cuda)})
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
    minutes = round((time.time() - t0) / 60, 1)
    rung.update(job_status=status, output=json.dumps(output)[:600] if output is not None else None,
                minutes=minutes, worker_states=sorted(seen),
                # Upper bound: includes queue and throttled time, which isn't billed.
                est_cost_usd=sizing.cost_usd(minutes, gpu_class))
    if status == "COMPLETED":
        rung["outcome"] = "pass"
        rp.call("DELETE", f"{rp.REST}/endpoints/{eid}", key)
        rp.call("DELETE", f"{rp.REST}/templates/{tpl['id']}", key)
    else:
        rung["outcome"] = classify_failure(status, output, crash_signals)
        rung["logs"] = f"https://console.runpod.io/serverless/user/endpoint/{eid}"
    log(f"[{label}] {rung['outcome']} ({status}) in {rung['minutes']} min")
    return rung


def next_rung(env, gpu_class, outcome, remote_code_tried, max_gpu="180gb", remote_code_possible=True,
              allow_remote_code=True):
    """The ladder: which setting to change after a failure, or None to stop.

    A crash loop is only retried with remote code when the CPU check didn't
    rule it out: flint (no auto_map) crash-looped on a Blackwell-only MoE
    kernel, and a TRUST_REMOTE_CODE rung would have been 30 min of H200 for nothing.
    allow_remote_code=False (CI's default) never runs a repo's own code: the
    run stops at needs_remote_code for a person to review the code first.
    """
    if allow_remote_code and not remote_code_tried and (
            outcome == "needs_remote_code" or (outcome == "crash_loop" and remote_code_possible)):
        return {**env, "TRUST_REMOTE_CODE": "true"}, gpu_class
    if outcome == "first_request_failure" and env.get("GPU_MEMORY_UTILIZATION") != "0.85":
        return {**env, "GPU_MEMORY_UTILIZATION": "0.85"}, gpu_class
    order = list(GPU_CLASSES)
    if (outcome == "startup_oom" and gpu_class in NEXT_GPU
            and order.index(NEXT_GPU[gpu_class]) <= order.index(max_gpu)):
        return env, NEXT_GPU[gpu_class]
    return None


def cpu_verdict(image):
    """The CPU-check verdict recorded for this exact image tag, if the pipeline built it."""
    for f in (ROOT / "data" / "results").rglob("*.json"):
        r = json.loads(f.read_text())
        if image in (r.get("image"), r.get("image_pinned")):
            return r.get("verdict")
    return None


def spent_last_24h(now=None, smoke_dir=None):
    """Estimated spend of recorded smoke rungs that started in the last 24 h."""
    now = now or datetime.now(timezone.utc)
    total = 0.0
    for f in (smoke_dir or ROOT / "data" / "smoke").glob("*.json"):
        for r in json.loads(f.read_text()).get("rungs") or []:
            started = datetime.fromisoformat(r["started"]) if r.get("started") else None
            if started and (now - started).total_seconds() < 86400 and r.get("gpu_class") in sizing.PRICE_PER_HR:
                total += r.get("est_cost_usd", sizing.cost_usd(r.get("minutes") or 0, r["gpu_class"]))
    return round(total, 2)


def write_result(result, image):
    out = ROOT / "data" / "smoke" / f"{image.rsplit(':', 1)[-1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="24gb", choices=[*GPU_CLASSES, "auto"],
                    help="auto: smallest class that holds the weights (Hugging Face API), see sizing.py")
    ap.add_argument("--max-gpu", default="48gb", choices=GPU_CLASSES,
                    help="largest class a startup OOM may climb to (H200/B200 cost several times a 4090)")
    ap.add_argument("--min-cuda", default="13.0", help="vllm/vllm-openai images build on CUDA 13.0")
    ap.add_argument("--remote-code", action="store_true", help="start with TRUST_REMOTE_CODE=true (CPU check said so)")
    ap.add_argument("--no-remote-code", action="store_true",
                    help="never set TRUST_REMOTE_CODE; stop at needs_remote_code for a person to review the code")
    ap.add_argument("--budget-usd", type=float,
                    help="skip or stop when recorded spend in the last 24 h plus this rung's worst case would exceed it")
    ap.add_argument("--max-minutes", type=int, default=40, help="wall-clock cap across all rungs")
    ap.add_argument("--disk-gb", type=int,
                    help="container disk; weights download here without a network volume (default: from the weight size, min 60)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra worker env for every rung, e.g. --set ATTENTION_BACKEND=FLASHINFER (repeatable)")
    ap.add_argument("--env-file")
    a = ap.parse_args()
    if any("=" not in s for s in a.set):
        ap.error("--set takes KEY=VALUE")
    if a.remote_code and a.no_remote_code:
        ap.error("--remote-code and --no-remote-code conflict")
    key = rp.load_key(a.env_file)

    order = list(GPU_CLASSES)
    base = {"image": a.image, "model": a.model}
    gb = None
    if a.gpu == "auto" or a.disk_gb is None:
        try:
            gb = sizing.weights_gb(sizing.hf_model_info(a.model))
        except Exception as e:  # noqa: BLE001 - sizing is best effort; the ladder still climbs on OOM
            log(f"weight size unavailable ({e})")
    if gb is not None:
        base["weights_gb"] = round(gb, 1)
    gpu = a.gpu
    if gpu == "auto":
        gpu = sizing.pick_class(gb) if gb is not None else "24gb"
        log(f"weights {base.get('weights_gb', '?')} GB -> {gpu or 'multi-GPU'}")
    skip = None
    if gpu is None:
        skip = {"reason": "needs_multi_gpu", "detail": f"{gb:.0f} GB of weights is more than one 180 GB GPU holds"}
    elif order.index(gpu) > order.index(a.max_gpu):
        skip = {"reason": "needs_larger_gpu", "detail": f"needs {gpu}; --max-gpu is {a.max_gpu}"}
    if skip:
        out = write_result({**base, "passed": False, "skipped": skip, "rungs": [],
                            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, a.image)
        log(f"NOT RUN: {skip['detail']} -> {out.relative_to(ROOT)}")
        return 1
    disk = a.disk_gb or (sizing.disk_gb(gb) if gb is not None else 60)

    deadline = time.time() + a.max_minutes * 60
    env = dict(s.split("=", 1) for s in a.set)
    if a.remote_code:
        env["TRUST_REMOTE_CODE"] = "true"
    verdict = cpu_verdict(a.image)
    remote_possible = verdict is None or "remote_code" in verdict
    if verdict:
        log(f"CPU check verdict for this image: {verdict}")
    label = a.image.rsplit(":", 1)[-1].replace(".", "")[:30]
    rungs, stopped = [], None
    while time.time() < deadline:
        if a.budget_usd is not None:
            spent = spent_last_24h() + sum(r.get("est_cost_usd", 0) for r in rungs)
            worst = sizing.cost_usd((deadline - time.time()) / 60, gpu)
            if spent + worst > a.budget_usd:
                stopped = {"reason": "budget", "detail": f"${spent:.2f} spent in the last 24 h + up to "
                           f"${worst:.2f} for a {gpu} rung > ${a.budget_usd:.2f} budget"}
                log(f"stopping: {stopped['detail']}")
                break
        r = run_rung(key, a.image, a.model, env, gpu, a.min_cuda, deadline, f"{label}-{len(rungs)}", disk)
        rungs.append(r)
        if r["outcome"] == "pass":
            break
        nxt = next_rung(env, gpu, r["outcome"], "TRUST_REMOTE_CODE" in env, a.max_gpu, remote_possible,
                        allow_remote_code=not a.no_remote_code)
        if nxt is None:
            break
        env, gpu = nxt

    passed = bool(rungs) and rungs[-1]["outcome"] == "pass"
    result = {**base, "passed": passed,
              "passing_env": rungs[-1]["env"] if passed else None,
              "passing_gpu_class": rungs[-1]["gpu_class"] if passed else None,
              "rungs": rungs, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if passed:
        last = rungs[-1]
        result["deploy"] = {"template": template_body(a.image, a.model, last["env"], last["disk_gb"]),
                            "endpoint": endpoint_body(last["gpu_class"], a.min_cuda)}
    if stopped:
        result["skipped" if not rungs else "stopped"] = stopped
    out = write_result(result, a.image)
    log(f"{'PASS' if passed else 'FAIL'} after {len(rungs)} rung(s) -> {out.relative_to(ROOT)}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
