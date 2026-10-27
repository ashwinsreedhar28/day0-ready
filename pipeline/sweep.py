"""Optimal-config sweep: one model, one GPU type, one image (docs/optimal-config.md).

    python3 pipeline/sweep.py --image ghcr.io/ashwinsreedhar28/worker-vllm:day0-v0.32.0--day0-sweep-<sha> \
        --model Qwen/Qwen3-8B --gpu-class 48gb --volume-id <id> --s0-env '{}' [--cap-minutes 180]
    python3 pipeline/sweep.py ... --plan       # configs, env and budget; no API calls
    python3 pipeline/sweep.py --preflight      # balance and network volumes; starts nothing

The image must carry worker-vllm's boot report (DAY0_BOOT_REPORT, branch
day0-sweep): engine-side start times come from it. A sweep without it stops
after the first warm-up.

Per config (S0-S5, then W): an endpoint pinned to one GPU type on the network
volume, one warm-up start (discarded), 3 measured cold starts, and one bench on
a warm worker (C=1 and C=30 through /openai/v1/completions, vLLM's /metrics
read before and after). S5 runs only when vLLM's default batch budget is under
8,192 tokens; W only when two or more single changes beat S1. Q and L are not
in this runner yet.

Spend: worker-alive minutes (from scale-up until no worker is left, an upper
bound) x the class's highest listed rate (sizing.py). Before every step the
runner checks that spent + the step's estimate stays under --cap-minutes, and it
stops after S1 when S1's three engine-init times spread by more than 20%.
Endpoints are scaled to 0 on every exit path; ones whose config failed are
kept (at 0 workers) so their logs stay readable in the console.

Writes data/sweep/<image tag>/: <config>.json per config, summary.json, and
config_runs.jsonl / config_summary.jsonl for serverless-lakehouse.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_client  # noqa: E402
import recipes  # noqa: E402
import runpod_api as rp  # noqa: E402
import sizing  # noqa: E402
from smoke import GPU_CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "day0-sweep-"
GRAPHQL = "https://api.runpod.io/graphql"
CATALOG = "https://api.runpod.io/v2/catalog/datacenters?include=GPU_AVAILABILITY"
LEVELS = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
NO_GPU_MINUTES = 10  # a start whose worker never gets past "throttled" in this long has no GPU

CONCURRENCY = 30                       # worker-vllm's MAX_CONCURRENCY default
PROMPT_TOKENS, OUTPUT_TOKENS = 4096, 512
KV_FLOOR = CONCURRENCY * (PROMPT_TOKENS + OUTPUT_TOKENS)  # 138,240 tokens
CONTEXT_CAP = 32768
C1_REQUESTS, C30_REQUESTS = 8, 120
COLD_RUNS = 3
FLOOR = 0.95                           # of the best C=30 output tokens/s
NOISE_MIN_S = 2.0
INCONCLUSIVE = 0.20                    # S1 spread / median
WIN_TOK = 0.05                         # throughput margin for "wins on its own"
EST_MIN = {"cold": 6.0, "bench": 12.0}  # step estimates until one has been measured

STARTUP_PLAN = {"VLLM_ENABLE_STARTUP_PLAN": "1"}
CHANGES = {"S2": {"ENFORCE_EAGER": "true"}, "S3": {"MAX_NUM_SEQS": "32"},
           "S4": {"GPU_MEMORY_UTILIZATION": "0.95"}, "S5": {"MAX_NUM_BATCHED_TOKENS": "8192"}}
ORDER = ["S0", "S1", "S2", "S3", "S4", "S5", "W"]
SWEEP_ONLY = {"DAY0_BOOT_REPORT", "VLLM_CACHE_ROOT", "MAX_CONCURRENCY"}
METRICS = ("vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:num_preemptions_total",
           "vllm:request_success_total")
# GPU type -> the class of its own VRAM. Classes are "at least" lists (24gb also names 48 GB
# cards), so a type takes the largest class that lists it.
VRAM_CLASS = {t: cls for cls in ("24gb", "48gb", "80gb", "141gb", "180gb") for t in GPU_CLASSES[cls]}


class Stop(Exception):
    """The sweep can't go on (budget, deadline, missing boot report)."""

    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason, self.detail = reason, detail


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- configs

def config_env(cid, s0_env, winners=()):
    """The config's own env (on top of the fixed conditions)."""
    if cid == "S0":
        return dict(s0_env)
    env = {**s0_env, **STARTUP_PLAN}
    if cid in CHANGES:
        env.update(CHANGES[cid])
    elif cid == "W":
        for w in winners:
            env.update(CHANGES[w])
    elif cid != "S1":
        raise ValueError(f"unknown config {cid}")
    return env


def n_changes(env, s0_env):
    return sum(1 for k in set(env) | set(s0_env) if env.get(k) != s0_env.get(k))


def fixed_env(model, max_model_len, sweep_id, cid):
    return {"MODEL_NAME": model, "MAX_MODEL_LEN": str(max_model_len), "MAX_CONCURRENCY": str(CONCURRENCY),
            "DAY0_BOOT_REPORT": "1", "VLLM_CACHE_ROOT": f"/runpod-volume/day0-sweep/{sweep_id}/{cid}"}


def s5_applies(s0_env, boots):
    """(runs?, why): S5 only where the launch log shows a default budget under 8,192."""
    if "MAX_NUM_BATCHED_TOKENS" in s0_env:
        return False, "S0 already sets MAX_NUM_BATCHED_TOKENS"
    seen = [b["vllm"].get("max_num_batched_tokens") for b in boots if b and b.get("vllm")]
    seen = [v for v in seen if v]
    if not seen:
        return False, "no max_num_batched_tokens line in S1's boot reports"
    return (min(seen) < 8192), f"vLLM's default here is {min(seen)}"


def max_model_len(recipe, hf_config):
    """Recipe context_length, else the config's max_position_embeddings; capped at 32,768."""
    n = ((recipe or {}).get("model") or {}).get("context_length")
    if not n:
        cfg = hf_config.get("text_config") or hf_config
        n = cfg.get("max_position_embeddings") or hf_config.get("max_position_embeddings")
    return min(int(n), CONTEXT_CAP) if n else CONTEXT_CAP


def vocab_size(hf_config):
    cfg = hf_config.get("text_config") or hf_config
    return cfg.get("vocab_size") or hf_config.get("vocab_size")


# ---------------------------------------------------------------- reading results

def boot_of(output):
    """The boot report from a job's output (a list when the worker aggregates its stream)."""
    for item in output if isinstance(output, list) else [output]:
        if isinstance(item, dict) and "day0_boot_report" in item:
            return item["day0_boot_report"]
    return None


def start_ok(boot):
    return bool(boot) and not boot.get("startup_error") and boot.get("vllm", {}).get("engine_init_s") is not None


_METRIC = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)")


def parse_metrics(text):
    """Prometheus text -> {name: sum over label sets} for the counters the sweep reads."""
    out = {}
    for line in (text or "").splitlines():
        m = _METRIC.match(line.strip())
        if m and m.group(1) in METRICS:
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
            except ValueError:
                pass
    return out


def metric_delta(before, after):
    if before is None or after is None:
        return None
    return {k.removeprefix("vllm:").removesuffix("_total"): round(after.get(k, 0.0) - before.get(k, 0.0), 1)
            for k in METRICS}


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def config_stats(rec):
    """Ranked and gate numbers for one config record."""
    cold = [r for r in rec.get("runs", []) if r["kind"] == "cold" and r.get("ok")]
    inits = [r["boot"]["vllm"]["engine_init_s"] for r in cold]
    bench = rec.get("bench") or {}
    c30, c1 = bench.get("c30") or {}, bench.get("c1") or {}
    pre30 = (bench.get("metrics") or {}).get("c30")
    boots = [r["boot"] for r in rec.get("runs", []) if r.get("boot")] + ([bench["boot"]] if bench.get("boot") else [])
    kv = [b["vllm"].get("kv_cache_tokens") for b in boots if b.get("vllm", {}).get("kv_cache_tokens")]
    checks = {
        "no_relaunch": None if not boots else not any(b.get("relaunches") for b in boots),
        "no_failed_or_preempted": None if not c30 else (
            c30.get("failed") == 0 and pre30 is not None and pre30.get("num_preemptions") == 0),
        "kv_floor": None if not kv else min(kv) >= KV_FLOOR,
    }
    return {
        "engine_init_s": inits, "engine_init_s_p50": median(inits),
        "engine_init_s_min": min(inits) if inits else None, "engine_init_s_max": max(inits) if inits else None,
        "model_load_s_p50": median(r["boot"]["vllm"].get("model_load_s") for r in cold),
        "delay_s_p50": median(r.get("delay_s") for r in cold),
        "output_tok_s_c30": c30.get("output_tok_s"), "tpot_p50_ms_c1": c1.get("tpot_p50_ms"),
        "kv_cache_tokens": min(kv) if kv else None, "fit_checks": checks,
        "fits": len(inits) == COLD_RUNS and all(v is True for v in checks.values()),
    }


def noise_band(s1):
    """(band seconds, spread, inconclusive?) from S1's three engine-init times."""
    inits = s1.get("engine_init_s") or []
    if len(inits) < COLD_RUNS:
        return None, None, None
    spread = max(inits) - min(inits)
    return max(NOISE_MIN_S, spread), round(spread, 2), spread > INCONCLUSIVE * statistics.median(inits)


def winners(stats, band):
    """Single changes (S2-S5) that beat S1 on one ranked number without losing on the other."""
    s1 = stats.get("S1")
    if not s1 or not s1["fits"] or band is None:
        return []
    out = []
    for cid in CHANGES:
        c = stats.get(cid)
        if not c or not c["fits"]:
            continue
        faster = s1["engine_init_s_p50"] - c["engine_init_s_p50"] > band
        slower = c["engine_init_s_p50"] - s1["engine_init_s_p50"] > band
        more = c["output_tok_s_c30"] > (1 + WIN_TOK) * s1["output_tok_s_c30"]
        less = c["output_tok_s_c30"] < (1 - WIN_TOK) * s1["output_tok_s_c30"]
        if (faster and not less) or (more and not slower):
            out.append(cid)
    return out


def recommend(stats, envs, s0_env, band):
    """{'recommended', 'fast_start', 'floor_tok_s', 'verdict'} from per-config stats."""
    cands = {c: s for c, s in stats.items() if s["fits"] and c in ORDER}
    if not cands or band is None:
        return {"recommended": None, "fast_start": None, "floor_tok_s": None, "verdict": "failed"}
    floor = FLOOR * max(s["output_tok_s_c30"] for s in cands.values())
    above = {c: s for c, s in cands.items() if s["output_tok_s_c30"] >= floor}
    best = min(s["engine_init_s_p50"] for s in above.values())
    tied = [c for c, s in above.items() if s["engine_init_s_p50"] - best < band]
    pick = min(tied, key=lambda c: (n_changes(envs[c], s0_env), above[c]["engine_init_s_p50"], ORDER.index(c)))
    fastest = min(cands, key=lambda c: (cands[c]["engine_init_s_p50"], ORDER.index(c)))
    fast = fastest if cands[fastest]["output_tok_s_c30"] < floor else None
    return {"recommended": pick, "fast_start": fast, "floor_tok_s": round(floor, 1), "verdict": "ok"}


# ---------------------------------------------------------------- Runpod calls

def workers(key, eid):
    st, h = rp.call("GET", f"{rp.JOBS}/{eid}/health", key)
    return {k: v for k, v in ((h or {}).get("workers") or {}).items() if v} if rp.ok(st) and isinstance(h, dict) else {}


def wait_cold(key, eid, timeout=600, settle=3):
    """True once health shows no workers for `settle` polls in a row (10 s apart)."""
    t, quiet = time.time(), 0
    while time.time() - t < timeout:
        quiet = quiet + 1 if not workers(key, eid) else 0
        if quiet >= settle:
            return True
        time.sleep(10)
    return False


def set_workers(key, eid, max_=None, min_=None, timeout=90):
    """PATCH the worker limits and poll until GET shows them (the API is eventually consistent)."""
    body = {k: v for k, v in (("workersMax", max_), ("workersMin", min_)) if v is not None}
    rp.call("PATCH", f"{rp.REST}/endpoints/{eid}", key, body)
    t = time.time()
    while time.time() - t < timeout:
        st, e = rp.call("GET", f"{rp.REST}/endpoints/{eid}", key)
        if rp.ok(st) and isinstance(e, dict) and all(e.get(k) == v for k, v in body.items()):
            return True
        time.sleep(3)
    return False


def run_job(key, eid, job_input, timeout, poll=5, watch=None):
    """(status, output, status body) for one queued job; cancels it on a timeout.

    watch, a dict, collects the worker states health showed while waiting
    ("states") and when a worker other than a throttled one first appeared
    ("worker_at"): a start that never got past throttled had no GPU to run on.
    """
    for _ in range(20):  # the jobs API sees a new workersMax a few seconds after REST does
        st, job = rp.call("POST", f"{rp.JOBS}/{eid}/run", key, {"input": job_input})
        if rp.ok(st) or "ENDPOINT_PAUSED" not in str(job):
            break
        time.sleep(3)
    if not rp.ok(st):
        return f"SUBMIT_{st}", str(job)[:300], {}
    jid, t0 = job["id"], time.time()
    while time.time() - t0 < timeout:
        time.sleep(poll)
        if watch is not None:
            states = set(workers(key, eid))
            watch.setdefault("states", set()).update(states)
            if states - {"throttled"} and "worker_at" not in watch:
                watch["worker_at"] = time.time()
            if "worker_at" not in watch and time.time() - t0 > watch.get("give_up_s", float("inf")):
                rp.call("POST", f"{rp.JOBS}/{eid}/cancel/{jid}", key)
                return "NO_GPU", None, {}
        st, s = rp.call("GET", f"{rp.JOBS}/{eid}/status/{jid}", key)
        if rp.ok(st) and s.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return s["status"], s.get("output") if s["status"] == "COMPLETED" else s.get("error") or s.get("output"), s
    rp.call("POST", f"{rp.JOBS}/{eid}/cancel/{jid}", key)
    return "DEADLINE", None, {}


def read_metrics(key, eid):
    status, out, _ = run_job(key, eid, {"route": "/metrics"}, timeout=180, poll=2)
    if status != "COMPLETED":
        return None
    text = "".join(x for x in (out if isinstance(out, list) else [out]) if isinstance(x, str))
    return parse_metrics(text) or None


def balance(key):
    st, r = rp.call("POST", GRAPHQL, key, {"query": "query { myself { clientBalance } }"})
    try:
        return float(r["data"]["myself"]["clientBalance"])
    except (TypeError, KeyError, ValueError):
        return None


def availability(key):
    """{data center id: {GPU type id: level}} from Runpod's catalog (levels NONE/LOW/MEDIUM/HIGH)."""
    st, r = rp.call("GET", CATALOG, key)
    if not rp.ok(st) or not isinstance(r, dict):
        return None
    return {dc["id"]: {g["id"]: g.get("availability") for g in dc.get("gpuAvailability") or []}
            for dc in r.get("dataCenters") or []}


def types_in(dc_avail, gpu_types):
    """The class's types this data center has in stock, most available first, ties in class order."""
    have = [t for t in gpu_types if LEVELS.get((dc_avail or {}).get(t), 0)]
    return sorted(have, key=lambda t: (-LEVELS[dc_avail[t]], gpu_types.index(t)))


def pick_volume(volumes, avail, gpu_types, need_gb):
    """(volume, types) for the existing volume whose data center has the most of the class in stock."""
    best = None
    for v in volumes:
        have = types_in((avail or {}).get(v.get("dataCenterId")), gpu_types)
        if have and (v.get("size") or 0) >= need_gb:
            score = LEVELS[avail[v["dataCenterId"]][have[0]]]
            if best is None or score > best[0]:
                best = (score, v, have)
    return (best[1], best[2]) if best else (None, None)


def pick_data_center(avail, gpu_types):
    """(data center, types) with the most of the class in stock, for a new volume."""
    ranked = sorted(((LEVELS[(a or {})[types_in(a, gpu_types)[0]]], dc) for dc, a in (avail or {}).items()
                     if types_in(a, gpu_types)), reverse=True)
    return (ranked[0][1], types_in(avail[ranked[0][1]], gpu_types)) if ranked else (None, None)


def hf_config(model):
    headers = {"User-Agent": rp.UA}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    url = f"https://huggingface.co/{model}/resolve/main/config.json"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------- the sweep

class Sweep:
    def __init__(self, key, a, max_len, vocab, out_dir):
        self.key, self.a, self.max_len, self.vocab, self.out = key, a, max_len, vocab, out_dir
        self.sweep_id = f"{a.image.rsplit(':', 1)[-1]}-{a.gpu_class}-{time.strftime('%Y%m%d%H%M', time.gmtime())}"
        self.gpu_type = a.gpu_type          # pinned after the first warm-up when None
        self.gpu_class = VRAM_CLASS.get(a.gpu_type, a.gpu_class) if a.gpu_type else a.gpu_class
        self.minutes = 0.0
        self.step_minutes = {"cold": [], "bench": []}
        self.deadline = time.time() + a.max_hours * 3600
        self.records, self.envs = {}, {}
        self.prompts = bench_client.prompts(C1_REQUESTS + C30_REQUESTS, PROMPT_TOKENS, vocab, seed=a.seed)
        self.probe = bench_client.prompts(1, 64, vocab, seed=a.seed + 1)[0]
        self.stopped = None

    def spent_usd(self):
        return sizing.cost_usd(self.minutes, self.gpu_class)

    def guard(self, kind):
        est = max(self.step_minutes[kind] or [EST_MIN[kind]])
        if self.minutes + est > self.a.cap_minutes:
            raise Stop("budget", f"{self.minutes:.0f} worker-min spent; the next {kind} step (~{est:.0f} min) "
                                 f"would pass the {self.a.cap_minutes} min cap")
        if time.time() + est * 60 > self.deadline:
            raise Stop("deadline", f"--max-hours {self.a.max_hours} reached")

    def charge(self, kind, t0):
        m = max(0.0, (time.time() - t0) / 60)
        self.minutes += m
        self.step_minutes[kind].append(m)
        return round(m, 2), sizing.cost_usd(m, self.gpu_class)

    def create(self, cid, env, gpu_types):
        name = f"{PREFIX}{cid}-{int(time.time())}"
        tpl_env = {**fixed_env(self.a.model, self.max_len, self.sweep_id, cid), **env}
        st, tpl = rp.call("POST", f"{rp.REST}/templates", self.key, {
            "name": name, "imageName": self.a.image, "isServerless": True,
            "containerDiskInGb": self.a.disk_gb, "env": tpl_env})
        if not rp.ok(st):
            raise Stop("api", f"template {st}: {tpl}")
        st, ep = rp.call("POST", f"{rp.REST}/endpoints", self.key, {
            "templateId": tpl["id"], "name": name, "computeType": "GPU", "gpuTypeIds": gpu_types, "gpuCount": 1,
            "workersMin": 0, "workersMax": 0, "idleTimeout": 5, "executionTimeoutMs": 900000, "flashboot": False,
            "networkVolumeId": self.a.volume_id, "minCudaVersion": self.a.min_cuda})
        if not rp.ok(st):
            rp.call("DELETE", f"{rp.REST}/templates/{tpl['id']}", self.key)
            raise Stop("api", f"endpoint {st}: {ep}")
        return ep["id"], tpl["id"], tpl_env

    def delete(self, eid, tid):
        rp.call("PATCH", f"{rp.REST}/endpoints/{eid}", self.key, {"workersMax": 0, "workersMin": 0})
        wait_cold(self.key, eid, timeout=300)
        rp.call("DELETE", f"{rp.REST}/endpoints/{eid}", self.key)
        rp.call("DELETE", f"{rp.REST}/templates/{tid}", self.key)

    def cold_start(self, eid, kind, index):
        """Scale 0 -> 1, wait for the boot-report job, scale back to 0 and wait until no worker is left."""
        self.guard("cold")
        t0 = time.time()
        rec = {"kind": kind, "index": index, "started_at": now_iso()}
        if not set_workers(self.key, eid, max_=1):
            log("workersMax=1 did not show up in 90 s; submitting anyway")
        watch = {"give_up_s": min(NO_GPU_MINUTES, self.a.start_minutes) * 60}
        status, out, s = run_job(self.key, eid, {"day0_boot_report": True}, timeout=self.a.start_minutes * 60,
                                 watch=watch)
        boot = boot_of(out) if status == "COMPLETED" else None
        rec.update(job_status=status, delay_s=round((s.get("delayTime") or 0) / 1000, 1) if s else None,
                   execution_s=round((s.get("executionTime") or 0) / 1000, 1) if s else None, boot=boot,
                   ok=start_ok(boot), worker_states=sorted(watch.get("states", ())),
                   no_gpu=status == "NO_GPU" or ("worker_at" not in watch and status != "COMPLETED"))
        if not boot:
            rec["output"] = json.dumps(out)[:400] if out is not None else None
        set_workers(self.key, eid, max_=0)
        if not wait_cold(self.key, eid):
            log("worker still up 10 min after scale-down")
        # Billing starts with a worker; a start that only ever showed "throttled" cost nothing.
        rec["minutes"], rec["est_cost_usd"] = self.charge("cold", time.time() if rec["no_gpu"] else watch.get("worker_at", t0))
        v = (boot or {}).get("vllm", {})
        log(f"  {kind} {index}: {status}, delay {rec['delay_s']} s, engine init {v.get('engine_init_s')} s, "
            f"load {v.get('model_load_s')} s, KV {v.get('kv_cache_tokens')} tok, gpu {(boot or {}).get('gpu')}, "
            f"{rec['minutes']} min")
        return rec

    def start_with_retry(self, eid, kind, index):
        """A start that fails gets one more try (stop rule); returns the runs made."""
        first = self.cold_start(eid, kind, index)
        if first["ok"]:
            return [first]
        if first["no_gpu"]:
            raise Stop("no_gpu", f"no worker got a GPU in {min(NO_GPU_MINUTES, self.a.start_minutes):.0f} min "
                                 f"(states seen: {first['worker_states'] or 'none'}); the volume's data center may "
                                 f"not have the GPU type free")
        if first["job_status"] == "COMPLETED" and not first["boot"]:
            raise Stop("no_boot_report", "the image answered without a boot report; build it from "
                                         "worker-vllm's day0-sweep branch")
        log(f"  {kind} {index} failed; one more try")
        return [first, self.cold_start(eid, kind, f"{index}-retry")]

    def bench(self, eid, model):
        self.guard("bench")
        t0 = time.time()
        rec = {"started_at": now_iso()}
        try:
            set_workers(self.key, eid, max_=1, min_=1)
            status, out, _ = run_job(self.key, eid, {"day0_boot_report": True}, timeout=self.a.start_minutes * 60)
            rec["boot"] = boot_of(out) if status == "COMPLETED" else None
            if not start_ok(rec["boot"]):
                rec["error"] = f"bench worker did not start ({status})"
                return rec
            url = f"{rp.JOBS}/{eid}/openai/v1/completions"
            probe = bench_client.stream_one(url, self.key, bench_client.body(model, self.probe, 16))
            rec["probe"] = probe
            if not probe["ok"]:
                rec["error"] = f"probe request failed: {probe['error']}"
                return rec
            rec["stream_buffered"] = probe["chunks"] < 0.5 * probe["out_tokens"]
            m0 = read_metrics(self.key, eid)
            rec["c1"], _ = bench_client.run_level(url, self.key, model, self.prompts[:C1_REQUESTS], 1, OUTPUT_TOKENS)
            m1 = read_metrics(self.key, eid)
            rec["c30"], _ = bench_client.run_level(url, self.key, model, self.prompts[C1_REQUESTS:], CONCURRENCY,
                                                   OUTPUT_TOKENS)
            m2 = read_metrics(self.key, eid)
            rec["metrics"] = {"c1": metric_delta(m0, m1), "c30": metric_delta(m1, m2)}
            gen = (rec["metrics"]["c30"] or {}).get("generation_tokens")
            rec["metrics_match"] = None if gen is None else abs(gen - rec["c30"]["output_tokens"]) <= max(
                1.0, 0.01 * rec["c30"]["output_tokens"])
            log(f"  bench: C=1 TPOT {rec['c1']['tpot_p50_ms']} ms; C=30 {rec['c30']['output_tok_s']} tok/s, "
                f"{rec['c30']['failed']} failed, preemptions {(rec['metrics']['c30'] or {}).get('num_preemptions')}")
        finally:
            set_workers(self.key, eid, max_=0, min_=0)
            wait_cold(self.key, eid)
            rec["minutes"], rec["est_cost_usd"] = self.charge("bench", t0)
        return rec

    def run_config(self, cid, env):
        log(f"{cid}: {env or '{}'}")
        rec = {"config": cid, "env": env, "n_changes": n_changes(env, self.a.s0_env), "runs": [],
               "started_at": now_iso()}
        self.records[cid], self.envs[cid] = rec, env
        types = [self.gpu_type] if self.gpu_type else self.a.gpu_types
        eid, tid, rec["template_env"] = self.create(cid, env, types)
        rec["endpoint_id"], keep = eid, False
        try:
            runs = self.start_with_retry(eid, "warmup", 0)
            rec["runs"] += runs
            if not runs[-1]["ok"]:
                rec["failed"], keep = "warm-up start failed twice", True
                return rec
            if not self.gpu_type:
                got = runs[-1]["boot"].get("gpu")
                if got not in types:
                    raise Stop("gpu", f"worker reported GPU {got!r}, not one of {types}")
                self.gpu_type, self.gpu_class = got, VRAM_CLASS.get(got, self.a.gpu_class)
                log(f"  pinned to {got}")
                if len(types) > 1:  # same template; a new endpoint limited to that one type
                    rp.call("PATCH", f"{rp.REST}/endpoints/{eid}", self.key, {"workersMax": 0})
                    rp.call("DELETE", f"{rp.REST}/endpoints/{eid}", self.key)
                    rp.call("DELETE", f"{rp.REST}/templates/{tid}", self.key)
                    eid, tid, _ = self.create(cid, env, [got])
                    rec["endpoint_id"] = eid
            rec["gpu_type"] = self.gpu_type
            for i in range(1, COLD_RUNS + 1):
                runs = self.start_with_retry(eid, "cold", i)
                rec["runs"] += runs
                self.save(cid)
                if not runs[-1]["ok"]:
                    rec["failed"], keep = f"cold start {i} failed twice", True
                    return rec
            rec["bench"] = self.bench(eid, self.a.model)
            if rec["bench"].get("error"):
                keep = True
            return rec
        finally:
            rec["finished_at"] = now_iso()
            rec["stats"] = config_stats(rec)
            if keep or self.a.keep:
                rp.call("PATCH", f"{rp.REST}/endpoints/{eid}", self.key, {"workersMax": 0, "workersMin": 0})
                rec["logs"] = f"https://console.runpod.io/serverless/user/endpoint/{eid}"
            else:
                self.delete(eid, tid)
            self.save(cid)
            log(f"{cid} done: fits {rec['stats']['fits']}, engine init p50 {rec['stats']['engine_init_s_p50']} s, "
                f"C=30 {rec['stats']['output_tok_s_c30']} tok/s; {self.minutes:.0f} worker-min, ~${self.spent_usd()}")

    def save(self, cid):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / f"{cid}.json").write_text(json.dumps(self.records[cid], indent=1) + "\n")

    def run(self):
        order = [c for c in ORDER if c in self.a.configs]
        try:
            for cid in order:
                if cid == "S5":
                    s1_boots = [r["boot"] for r in self.records.get("S1", {}).get("runs", []) if r.get("boot")]
                    runs, why = s5_applies(self.a.s0_env, s1_boots)
                    if not runs:
                        log(f"S5 skipped: {why}")
                        self.records["S5"] = {"config": "S5", "skipped": why}
                        continue
                if cid == "W":
                    band, _, _ = noise_band(self.stats().get("S1") or {})
                    win = winners(self.stats(), band)
                    if len(win) < 2:
                        why = f"{len(win)} single change(s) beat S1 ({', '.join(win) or 'none'})"
                        log(f"W skipped: {why}")
                        self.records["W"] = {"config": "W", "skipped": why, "winners": win}
                        continue
                    self.run_config("W", config_env("W", self.a.s0_env, win))
                    self.records["W"]["winners"] = win
                    continue
                self.run_config(cid, config_env(cid, self.a.s0_env))
                if cid == "S1":
                    if self.records["S1"].get("failed"):
                        raise Stop("s1_failed", "S1 did not start; S2-S5 and W all build on it")
                    _, spread, inconclusive = noise_band(self.records["S1"]["stats"])
                    if inconclusive:
                        raise Stop("inconclusive", f"S1's engine-init times spread {spread} s, over 20% of the median")
        except Stop as e:
            self.stopped = {"reason": e.reason, "detail": e.detail}
            log(f"stopped: {e.reason}: {e.detail}")
        return self.summary()

    def stats(self):
        return {c: r["stats"] for c, r in self.records.items() if r.get("stats")}

    def summary(self):
        stats = self.stats()
        # S1 sets the noise band; if S1 never ran or failed, S0's own spread stands in.
        band_from = "S1" if noise_band(stats.get("S1") or {})[0] is not None else "S0"
        band, spread, inconclusive = noise_band(stats.get(band_from) or {})
        rec = recommend(stats, self.envs, self.a.s0_env, band)
        if inconclusive:
            rec = {**rec, "recommended": None, "fast_start": None, "verdict": "inconclusive"}
        out = {
            "sweep_id": self.sweep_id, "model": self.a.model, "image": self.a.image, "digest": image_digest(self.a.image),
            "gpu_type": self.gpu_type, "gpu_class": self.gpu_class, "volume_id": self.a.volume_id,
            "max_model_len": self.max_len, "s0_env": self.a.s0_env, "kv_floor_tokens": KV_FLOOR,
            "bench": {"prompt_tokens": PROMPT_TOKENS, "output_tokens": OUTPUT_TOKENS, "c1_requests": C1_REQUESTS,
                      "c30_requests": C30_REQUESTS, "seed": self.a.seed},
            "noise_band_from": band_from, "spread_s": spread, "noise_band_s": band, **rec,
            "winners": self.records.get("W", {}).get("winners"),
            "configs": [{"config": c, "skipped": r.get("skipped"), "failed": r.get("failed"),
                         "env": r.get("env"), **(r.get("stats") or {})} for c, r in self.records.items()],
            "worker_minutes": round(self.minutes, 1), "est_cost_usd": self.spent_usd(),
            "cap_minutes": self.a.cap_minutes, "stopped": self.stopped, "finished_at": now_iso(),
        }
        if rec["recommended"]:
            out["recommended_block"] = recommended_block(out, self.records[rec["recommended"]])
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
        write_tables(self.out, out, self.records)
        return out


def volume_gb(model):
    """Room for the weights (Hugging Face API size) plus per-config compile caches."""
    gb = sizing.weights_gb(sizing.hf_model_info(model)) or 60
    return int(gb * 1.25) + 15


def image_digest(image):
    for f in (ROOT / "data" / "results").rglob("*.json"):
        r = json.loads(f.read_text())
        if image in (r.get("image"), r.get("image_pinned")) and r.get("digest"):
            return r["digest"]
    return None


def recommended_block(summary, rec):
    """The env block for data/ready.json: what a Runpod template needs, without the sweep-only settings."""
    s = rec["stats"]
    env = {k: v for k, v in rec["template_env"].items() if k not in SWEEP_ONLY}
    return {"config": rec["config"], "image": summary["image"], "digest": summary["digest"],
            "gpu_type": summary["gpu_type"], "gpu_count": 1, "env": env,
            "measured": {"engine_init_s_p50": s["engine_init_s_p50"], "output_tok_s_c30": s["output_tok_s_c30"]},
            "fast_start": summary.get("fast_start"),
            "sweep": f"data/sweep/{summary['image'].rsplit(':', 1)[-1]}/summary.json"}


def write_tables(out_dir, summary, records):
    """Silver config_runs and gold config_summary rows, one JSON object per line."""
    base = {"sweep_id": summary["sweep_id"], "model": summary["model"]}
    runs = []
    for cid, rec in records.items():
        for r in rec.get("runs", []):
            b = r.get("boot") or {}
            v = b.get("vllm") or {}
            runs.append({**base, "image_digest": summary["digest"], "gpu_type": rec.get("gpu_type"),
                         "config_id": cid, "run_kind": r["kind"], "run_index": r["index"],
                         "started_at": r["started_at"], "delay_s": r.get("delay_s"),
                         "execution_s": r.get("execution_s"), "process_to_healthy_s": b.get("process_to_healthy_s"),
                         "launch_to_healthy_s": b.get("launch_to_healthy_s"), "engine_init_s": v.get("engine_init_s"),
                         "model_load_s": v.get("model_load_s"), "kv_cache_tokens": v.get("kv_cache_tokens"),
                         "relaunched": bool(b.get("relaunches")) if b else None, "concurrency": None,
                         "requests": None, "failed": None, "output_tok_s": None, "ttft_p50_ms": None,
                         "tpot_p50_ms": None, "est_cost_usd": r.get("est_cost_usd")})
        bench = rec.get("bench") or {}
        for i, level in enumerate(("c1", "c30"), 1):
            lv = bench.get(level)
            if lv:
                runs.append({**base, "image_digest": summary["digest"], "gpu_type": rec.get("gpu_type"),
                             "config_id": cid, "run_kind": "bench", "run_index": i, "started_at": bench["started_at"],
                             "delay_s": None, "execution_s": None, "process_to_healthy_s": None,
                             "launch_to_healthy_s": None, "engine_init_s": None, "model_load_s": None,
                             "kv_cache_tokens": None, "relaunched": None, "concurrency": lv["concurrency"],
                             "requests": lv["requests"], "failed": lv["failed"], "output_tok_s": lv["output_tok_s"],
                             "ttft_p50_ms": lv["ttft_p50_ms"], "tpot_p50_ms": lv["tpot_p50_ms"],
                             "est_cost_usd": bench.get("est_cost_usd") if i == 2 else None})
    gold = []
    for row in summary["configs"]:
        if row.get("skipped"):
            continue
        cid = row["config"]
        profile = "recommended" if cid == summary["recommended"] else "fast_start" if cid == summary["fast_start"] else "none"
        gold.append({**base, "gpu_type": summary["gpu_type"], "config_id": cid,
                     "env_json": json.dumps(row.get("env") or {}, sort_keys=True), "fits": row.get("fits"),
                     "engine_init_s_p50": row.get("engine_init_s_p50"), "engine_init_s_min": row.get("engine_init_s_min"),
                     "engine_init_s_max": row.get("engine_init_s_max"), "model_load_s_p50": row.get("model_load_s_p50"),
                     "output_tok_s_c30": row.get("output_tok_s_c30"), "tpot_p50_ms_c1": row.get("tpot_p50_ms_c1"),
                     "recommended": cid == summary["recommended"], "profile": profile, "verdict": summary["verdict"]})
    (out_dir / "config_runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))
    (out_dir / "config_summary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gold))


# ---------------------------------------------------------------- CLI

def list_volumes(key):
    st, vols = rp.call("GET", f"{rp.REST}/networkvolumes", key)
    return rp.items(vols, "networkVolumes") if rp.ok(st) else []


def preflight(key, gpu_class):
    bal = balance(key)
    print(f"balance: {'unknown' if bal is None else f'${bal:.2f}'}")
    avail = availability(key)
    for v in list_volumes(key):
        have = types_in((avail or {}).get(v.get("dataCenterId")), GPU_CLASSES[gpu_class])
        print(f"volume {v.get('id')}: {v.get('name')}, {v.get('size')} GB, {v.get('dataCenterId')}; "
              f"{gpu_class} in stock there: {have or 'none'}")
    if avail is None:
        print("GPU availability: catalog request failed")
    for dc, a in sorted((avail or {}).items()):
        have = types_in(a, GPU_CLASSES[gpu_class])
        if have:
            print(f"{dc}: " + ", ".join(f"{t} {a[t]}" for t in have))


def plan(a, max_len):
    est = len(a.configs) * (1 + COLD_RUNS) * EST_MIN["cold"] + len(a.configs) * EST_MIN["bench"]
    print(f"model {a.model}, image {a.image}, {a.gpu_type or a.gpu_class}, volume {a.volume_id}")
    print(f"MAX_MODEL_LEN {max_len}; KV floor {KV_FLOOR:,} tokens; bench {PROMPT_TOKENS} in / {OUTPUT_TOKENS} out, "
          f"C=1 x {C1_REQUESTS}, C={CONCURRENCY} x {C30_REQUESTS}")
    for cid in [c for c in ORDER if c in a.configs]:
        print(f"  {cid}: {'S1 + every single change that beats S1' if cid == 'W' else config_env(cid, a.s0_env)}")
    cls = VRAM_CLASS.get(a.gpu_type, a.gpu_class) if a.gpu_type else a.gpu_class
    print(f"rough worst case before skips: {est:.0f} worker-min (~${sizing.cost_usd(est, cls)}); "
          f"cap {a.cap_minutes} min (~${sizing.cost_usd(a.cap_minutes, cls)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true", help="print balance and network volumes, then exit")
    ap.add_argument("--plan", action="store_true", help="print configs and budget without calling Runpod")
    ap.add_argument("--image")
    ap.add_argument("--model")
    ap.add_argument("--gpu-class", default="48gb", choices=list(GPU_CLASSES))
    ap.add_argument("--gpu-type", help="pin this Runpod GPU type id; default: the first type in the class that "
                                       "starts the S0 warm-up")
    ap.add_argument("--volume-id", default="auto",
                    help="network volume id; auto: an existing one in a data center with the class in stock")
    ap.add_argument("--create-volume", action="store_true",
                    help="with --volume-id auto and no fitting volume: create one, delete it at the end")
    ap.add_argument("--volume-gb", type=int, help="size for a new volume (default: from the weight size)")
    ap.add_argument("--s0-env", type=json.loads, help='the smoke test\'s passing env as JSON, e.g. \'{}\'')
    ap.add_argument("--from-smoke", help="data/smoke/<image tag>.json to take S0's env from")
    ap.add_argument("--configs", default=",".join(ORDER))
    ap.add_argument("--cap-minutes", type=float, default=180, help="worker-minute cap for the whole sweep")
    ap.add_argument("--max-hours", type=float, default=4.5, help="wall-clock limit")
    ap.add_argument("--start-minutes", type=float, default=25, help="limit for one start, download included")
    ap.add_argument("--max-model-len", type=int, help="override the recipe / config context length")
    ap.add_argument("--vocab-size", type=int)
    ap.add_argument("--disk-gb", type=int, default=40)
    ap.add_argument("--min-cuda", default="13.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="keep every endpoint (scaled to 0)")
    ap.add_argument("--min-balance", type=float, default=None,
                    help="refuse to start below this balance (default: the cap's cost + $5)")
    ap.add_argument("--env-file")
    a = ap.parse_args()

    if a.preflight:
        preflight(rp.load_key(a.env_file), a.gpu_class)
        return
    if not (a.image and a.model):
        sys.exit("--image and --model are required")
    if a.from_smoke:
        sm = json.loads(Path(a.from_smoke).read_text())
        if not sm.get("passed") or sm.get("model") != a.model:
            sys.exit(f"{a.from_smoke} is not a pass for {a.model}")
        a.s0_env = sm.get("passing_env") or {}
    if a.s0_env is None:
        sys.exit("give S0's env: --from-smoke data/smoke/<tag>.json or --s0-env '{...}'")
    a.configs = [c.strip() for c in a.configs.split(",") if c.strip()]
    if a.gpu_type and a.gpu_type not in VRAM_CLASS:
        sys.exit(f"unknown GPU type {a.gpu_type!r}; known: {sorted(VRAM_CLASS)}")

    cfg = hf_config(a.model) if not (a.max_model_len and a.vocab_size) else {}
    max_len = a.max_model_len or max_model_len(recipes.fetch(a.model)[0], cfg)
    vocab = a.vocab_size or vocab_size(cfg)
    if not vocab:
        sys.exit("no vocab_size in the model's config.json; pass --vocab-size")
    if a.plan:
        plan(a, max_len)
        return
    key = rp.load_key(a.env_file)
    cls = VRAM_CLASS.get(a.gpu_type, a.gpu_class) if a.gpu_type else a.gpu_class
    need = a.min_balance if a.min_balance is not None else sizing.cost_usd(a.cap_minutes, cls) + 5
    bal = balance(key)
    if bal is None or bal < need:
        sys.exit(f"balance {'unknown' if bal is None else f'${bal:.2f}'}; this sweep needs at least ${need:.2f}")
    log(f"balance ${bal:.2f}; cap {a.cap_minutes} worker-min (~${sizing.cost_usd(a.cap_minutes, cls)})")

    a.gpu_types = [a.gpu_type] if a.gpu_type else list(GPU_CLASSES[a.gpu_class])
    created = None
    if a.volume_id in (None, "auto"):
        need = a.volume_gb or volume_gb(a.model)
        avail = availability(key)
        vol, have = pick_volume(list_volumes(key), avail, a.gpu_types, need)
        if vol:
            a.volume_id, a.gpu_types = vol["id"], have
            log(f"volume {vol['id']} in {vol['dataCenterId']}, which has {have} in stock")
        elif a.create_volume:
            dc, have = pick_data_center(avail, a.gpu_types)
            if not dc:
                sys.exit(f"no data center lists {a.gpu_types} in stock")
            st, v = rp.call("POST", f"{rp.REST}/networkvolumes", key,
                            {"name": f"{PREFIX}{int(time.time())}", "size": need, "dataCenterId": dc})
            if not rp.ok(st):
                sys.exit(f"network volume {st}: {v}")
            a.volume_id, a.gpu_types, created = v["id"], have, v["id"]
            log(f"created a {need} GB volume {v['id']} in {dc} ({have} in stock); deleted at the end")
        else:
            sys.exit(f"no volume of {need}+ GB in a data center with {a.gpu_types} in stock; "
                     "pass --create-volume or --volume-id")

    out_dir = ROOT / "data" / "sweep" / a.image.rsplit(":", 1)[-1]
    try:
        s = Sweep(key, a, max_len, vocab, out_dir).run()
    finally:
        if created and not a.keep:
            st, r = rp.call("DELETE", f"{rp.REST}/networkvolumes/{created}", key)
            log(f"volume {created} deleted" if rp.ok(st) else
                f"volume {created} NOT deleted ({st} {str(r)[:120]}): delete it in the console (billed monthly)")
    log(f"verdict {s['verdict']}: recommended {s['recommended']}, fast start {s['fast_start']}; "
        f"{s['worker_minutes']} worker-min, ~${s['est_cost_usd']}" + (f"; stopped: {s['stopped']['reason']}" if s["stopped"] else ""))


if __name__ == "__main__":
    main()
