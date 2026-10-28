"""Optimal-config sweep: selection rules, fit gate, metrics, and a full run against a fake Runpod API."""

import json
import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bench_client  # noqa: E402
import runpod_api as rp  # noqa: E402
import sweep  # noqa: E402

S0 = {"ATTENTION_BACKEND": "FLASHINFER"}


def boot(init, kv=180_000, relaunches=(), mnbt=2048, load=7.5, gpu="NVIDIA L40S", error=False):
    return {"version": 1, "gpu": gpu, "relaunches": list(relaunches), "startup_error": error,
            "launch_to_healthy_s": init + load + 10, "process_to_healthy_s": init + load + 14,
            "vllm": {"engine_init_s": init, "model_load_s": load, "kv_cache_tokens": kv,
                     "max_num_batched_tokens": mnbt}}


def record(inits, tok, failed=0, preempt=0, **kw):
    runs = [{"kind": "warmup", "index": 0, "ok": True, "boot": boot(inits[0] + 20, **kw)}]
    runs += [{"kind": "cold", "index": i + 1, "ok": True, "boot": boot(x, **kw), "delay_s": 60.0}
             for i, x in enumerate(inits)]
    return {"runs": runs, "bench": {"c30": {"output_tok_s": tok, "failed": failed}, "c1": {"tpot_p50_ms": 30.0},
                                    "metrics": {"c30": {"num_preemptions": preempt}}}}


# ---------------------------------------------------------------- configs

def test_config_env_builds_on_s1():
    assert sweep.config_env("S0", S0) == S0
    assert sweep.config_env("S1", S0) == {**S0, "VLLM_ENABLE_STARTUP_PLAN": "1"}
    assert sweep.config_env("S3", S0)["MAX_NUM_SEQS"] == "32"
    w = sweep.config_env("W", S0, ["S3", "S5"])
    assert w == {**S0, "VLLM_ENABLE_STARTUP_PLAN": "1", "MAX_NUM_SEQS": "32", "MAX_NUM_BATCHED_TOKENS": "8192"}
    assert sweep.n_changes(w, S0) == 3
    with pytest.raises(ValueError):
        sweep.config_env("Q", S0)


def test_fixed_env_gives_each_config_its_own_cache_root():
    a = sweep.fixed_env("Qwen/Qwen3-8B", 32768, "sid", "S1")
    b = sweep.fixed_env("Qwen/Qwen3-8B", 32768, "sid", "S4")
    assert a["VLLM_CACHE_ROOT"] != b["VLLM_CACHE_ROOT"]  # vllm#60143: the plan key ignores GPU_MEMORY_UTILIZATION
    assert a["DAY0_BOOT_REPORT"] == "1" and a["MAX_CONCURRENCY"] == "30" and a["MAX_MODEL_LEN"] == "32768"


def test_s5_only_where_the_default_budget_is_small():
    assert sweep.s5_applies({}, [boot(25, mnbt=2048)])[0] is True
    assert sweep.s5_applies({}, [boot(25, mnbt=8192)])[0] is False
    assert sweep.s5_applies({"MAX_NUM_BATCHED_TOKENS": "4096"}, [boot(25)])[0] is False
    # v0.32.0 didn't print the line in the pilot: fall back to vLLM's per-GPU defaults.
    runs, why = sweep.s5_applies({}, [{"vllm": {}}], "NVIDIA L40S")
    assert runs is True and "2048" in why and "log line was missing" in why
    assert sweep.s5_applies({}, [{"vllm": {}}], "NVIDIA H100 80GB HBM3")[0] is False
    assert sweep.s5_applies({}, [{"vllm": {}}], "NVIDIA A100 80GB PCIe")[0] is True   # A100 keeps 2,048
    assert sweep.s5_applies({}, [], None)[0] is False


def test_default_batch_budget_uses_the_device_the_worker_saw():
    whole = {"gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "gpu_memory_mib": [97887], "gpu_devices": []}
    assert sweep.default_batch_budget(whole, "NVIDIA RTX 6000 Ada Generation") == 8192
    sliced = {**whole, "gpu_devices": ["GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition", "MIG 2g.48gb Device 0:"]}
    assert sweep.default_batch_budget(sliced, "NVIDIA RTX 6000 Ada Generation") == 2048  # MIG: the type's class
    assert sweep.default_batch_budget({"gpu_memory_mib": [183359]}, "NVIDIA B200") == 16384


def test_target_context():
    assert sweep.max_model_len({"model": {"context_length": 40960}}, {}) == 32768  # Qwen3-8B's recipe
    assert sweep.max_model_len({"model": {"context_length": 8192}}, {}) == 8192
    assert sweep.max_model_len(None, {"max_position_embeddings": 16384}) == 16384
    assert sweep.max_model_len(None, {"text_config": {"max_position_embeddings": 262144}}) == 32768
    assert sweep.vocab_size({"text_config": {"vocab_size": 262144}}) == 262144


def test_kv_floor_is_thirty_bench_requests():
    assert sweep.KV_FLOOR == 30 * (4096 + 512) == 138_240


# ---------------------------------------------------------------- results

def test_boot_of_reads_the_aggregated_stream():
    rep = boot(25)
    assert sweep.boot_of([{"day0_boot_report": rep}]) == rep
    assert sweep.boot_of({"day0_boot_report": rep}) == rep
    assert sweep.boot_of([{"error": {"message": "The boot report is off."}}]) is None
    assert sweep.start_ok(rep) and not sweep.start_ok(boot(25, error=True)) and not sweep.start_ok({"vllm": {}})


METRICS_TEXT = """# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="Qwen/Qwen3-8B"} 0.0
vllm:generation_tokens_total{engine="0",model_name="Qwen/Qwen3-8B"} 61440.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="Qwen/Qwen3-8B"} 120.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="Qwen/Qwen3-8B"} 1.0
vllm:prompt_tokens_total{engine="0",model_name="Qwen/Qwen3-8B"} 491520.0
vllm:generation_tokens_created{engine="0",model_name="Qwen/Qwen3-8B"} 1.7e+09
"""


def test_metrics_sum_label_sets_and_take_deltas():
    m = sweep.parse_metrics(METRICS_TEXT)
    assert m["vllm:request_success_total"] == 121.0
    assert m["vllm:generation_tokens_total"] == 61440.0
    assert "vllm:generation_tokens_created" not in m
    before = {**m, "vllm:generation_tokens_total": 1440.0}
    d = sweep.metric_delta(before, m)
    assert d["generation_tokens"] == 60000.0 and d["num_preemptions"] == 0.0
    assert sweep.metric_delta(None, m) is None


# ---------------------------------------------------------------- fit and selection

def test_fit_gate():
    assert sweep.config_stats(record([25, 25.5, 26], 700))["fits"] is True
    relaunched = sweep.config_stats(record([25, 25.5, 26], 700, relaunches=["oom"]))
    assert relaunched["fits"] is False and relaunched["fit_checks"]["no_relaunch"] is False
    assert sweep.config_stats(record([25, 25.5, 26], 700, preempt=3))["fits"] is False
    assert sweep.config_stats(record([25, 25.5, 26], 700, failed=1))["fits"] is False
    small = sweep.config_stats(record([25, 25.5, 26], 700, kv=100_000))
    assert small["fits"] is False and small["kv_cache_tokens"] == 100_000
    no_metrics = record([25, 25.5, 26], 700)
    no_metrics["bench"]["metrics"] = {"c30": None}
    assert sweep.config_stats(no_metrics)["fits"] is False  # preemptions unknown: can't be recommended
    two = record([25, 25.5], 700)
    assert sweep.config_stats(two)["fits"] is False  # needs 3 cold starts


def test_bench_boot_counts_for_the_gate():
    rec = record([25, 25.5, 26], 700)
    rec["bench"]["boot"] = boot(25, relaunches=["oom"])
    assert sweep.config_stats(rec)["fits"] is False


def test_noise_band_and_inconclusive():
    band, spread, bad = sweep.noise_band({"engine_init_s": [25.0, 25.5, 26.0]})
    assert (band, spread, bad) == (2.0, 1.0, False)
    band, spread, bad = sweep.noise_band({"engine_init_s": [20.0, 23.0, 28.0]})
    assert (band, bad) == (8.0, True)  # 8 s > 20% of 23 s
    assert sweep.noise_band({"engine_init_s": [25.0]}) == (None, None, None)


def stats_for(table):
    return {c: sweep.config_stats(record(inits, tok)) for c, (inits, tok) in table.items()}


TABLE = {
    "S0": ([30.0, 30.2, 30.5], 700),
    "S1": ([25.0, 25.4, 25.8], 700),
    "S2": ([12.0, 12.1, 12.3], 500),   # eager: starts fast, serves less
    "S3": ([21.0, 21.1, 21.4], 705),   # smaller graph capture
    "S4": ([25.2, 25.5, 25.9], 702),
    "S5": ([25.1, 25.3, 25.6], 760),   # bigger batch budget
}


def test_winners():
    st = stats_for(TABLE)
    band = sweep.noise_band(st["S1"])[0]
    assert sweep.winners(st, band) == ["S3", "S5"]


def test_recommendation_with_floor_ties_and_fast_start():
    table = {**TABLE, "W": ([21.0, 21.2, 21.3], 765)}
    st = stats_for(table)
    envs = {c: sweep.config_env(c, S0, ["S3", "S5"]) for c in table}
    r = sweep.recommend(st, envs, S0, sweep.noise_band(st["S1"])[0])
    assert r["floor_tok_s"] == round(0.95 * 765, 1)
    assert r["recommended"] == "W"     # S3 starts as fast but is under the floor
    assert r["fast_start"] == "S2"
    assert r["verdict"] == "ok"


def test_ties_go_to_fewer_changes():
    table = {"S0": ([25.0, 25.3, 25.5], 700), "S1": ([24.0, 24.2, 24.5], 701), "S3": ([23.5, 23.6, 24.0], 702)}
    st = stats_for(table)
    envs = {c: sweep.config_env(c, {}) for c in table}
    # Inside the 2 s band all three tie; S0 changes nothing.
    assert sweep.recommend(st, envs, {}, 2.0)["recommended"] == "S0"
    assert sweep.recommend(st, envs, {}, 2.0)["fast_start"] is None


def test_nothing_fits():
    st = {"S0": sweep.config_stats(record([25, 25, 25], 700, kv=1000))}
    assert sweep.recommend(st, {"S0": {}}, {}, 2.0)["verdict"] == "failed"


# ---------------------------------------------------------------- full run, fake API

class FakeRunpod:
    """Templates, endpoints, health and jobs, with timings that depend on the config's env."""

    def __init__(self, boot_missing=False, no_gpu=False, flaky_health=False, ghost=None):
        self.templates, self.endpoints, self.jobs, self.deleted = {}, {}, {}, []
        self.metric_reads, self.boot_missing, self.no_gpu, self.n = {}, boot_missing, no_gpu, 0
        self.flaky_health, self.health_calls = flaky_health, 0
        self.alive = {}       # endpoint -> health polls left before its worker is gone after max -> 0
        self.volumes = {}
        self.openai_jobs, self.job_index = {}, {}
        self.ghost, self.boots_by_ep = ghost or {}, {}   # endpoint -> starts after which it never schedules

    def env(self, eid):
        return self.templates[self.endpoints[eid]["templateId"]]["env"]

    def is_ghost(self, eid):
        return eid in self.ghost and self.boots_by_ep.get(eid, 0) >= self.ghost[eid]

    def boot(self, eid):
        self.boots_by_ep[eid] = self.boots_by_ep.get(eid, 0) + 1
        env = self.env(eid)
        init = 30.0
        if env.get("VLLM_ENABLE_STARTUP_PLAN"):
            init = 25.0
        if env.get("ENFORCE_EAGER"):
            init = 12.0
        if env.get("MAX_NUM_SEQS") == "32":
            init = 21.0
        self.n += 1
        kv = 195_000 if env.get("GPU_MEMORY_UTILIZATION") == "0.95" else 180_000
        return boot(init + (self.n % 3) * 0.3, kv=kv, mnbt=int(env.get("MAX_NUM_BATCHED_TOKENS", 2048)))

    def tok_s(self, eid):
        env = self.env(eid)
        tok = 500.0 if env.get("ENFORCE_EAGER") else 700.0
        if env.get("MAX_NUM_BATCHED_TOKENS"):
            tok += 60
        if env.get("MAX_NUM_SEQS"):
            tok += 5
        return tok

    def call(self, method, url, key, body=None):
        path = url.split("/v1", 1)[-1] if url.startswith(rp.REST) else url.split("/v2", 1)[-1]
        if url == sweep.GRAPHQL:
            return 200, {"data": {"myself": {"clientBalance": 70.0}}}
        if method == "POST" and path == "/templates":
            tid = f"t{len(self.templates)}"
            self.templates[tid] = dict(body, id=tid)
            return 201, {"id": tid}
        if method == "POST" and path == "/endpoints":
            eid = f"e{len(self.endpoints)}"
            self.endpoints[eid] = dict(body, id=eid)
            return 201, {"id": eid}
        if method == "GET" and path == "/endpoints":
            return 200, list(self.endpoints.values())
        if method == "GET" and path == "/templates":
            return 200, list(self.templates.values())
        if method == "GET" and path == "/networkvolumes":
            return 200, list(self.volumes.values())
        m = re.fullmatch(r"/endpoints/(\w+)", path)
        if m and method == "PATCH":
            ep = self.endpoints[m.group(1)]
            if body.get("workersMax") == 0 and ep.get("workersMax", 0) > 0 and not self.is_ghost(ep["id"]):
                self.alive[ep["id"]] = 1   # the worker shows up in one more health poll
            ep.update(body)
            return 200, ep
        if m and method == "GET":
            return 200, self.endpoints[m.group(1)]
        if m and method == "DELETE":
            ep = self.endpoints[m.group(1)]
            if ep["workersMax"] or ep["workersMin"] or self.alive.get(ep["id"]):
                return 400, "endpoint has workers"   # Runpod: workers must be at 0 first
            self.deleted.append(path)
            return 204, None
        if method == "DELETE":
            self.deleted.append(path)
            self.volumes.pop(path.rsplit("/", 1)[-1], None)
            return 204, None
        m = re.fullmatch(r"/(\w+)/health", path)
        if m:
            self.health_calls += 1
            if self.flaky_health and self.health_calls % 2:
                return 0, "ConnectionResetError"
            eid = m.group(1)
            if self.is_ghost(eid):
                return 200, {"workers": {}}
            if self.alive.get(eid):
                self.alive[eid] -= 1
                return 200, {"workers": {"running": 1}}
            on = self.endpoints[eid]["workersMax"] > 0
            state = "throttled" if self.no_gpu else "idle"
            return 200, {"workers": {state: 1} if on else {}}
        m = re.fullmatch(r"/(\w+)/run", path)
        if m:
            jid = f"j{len(self.jobs)}"
            self.jobs[jid] = (m.group(1), body["input"])
            if body["input"].get("openai_route"):
                assert body["input"]["openai_input"]["stream"] is False
                assert "stream_options" not in body["input"]["openai_input"]
                n = self.openai_jobs[m.group(1)] = self.openai_jobs.get(m.group(1), 0) + 1
                self.job_index[jid] = n - 1
            return 200, {"id": jid}
        m = re.fullmatch(r"/(\w+)/status/(\w+)", path)
        if m:
            eid, job_input = self.jobs[m.group(2)]
            if self.no_gpu:
                return 200, {"status": "IN_QUEUE"}
            if job_input.get("route") == "/metrics":
                n = self.metric_reads[eid] = self.metric_reads.get(eid, 0) + 1
                gen = {1: 16, 2: 16 + 8 * 512, 3: 16 + 128 * 512}[n]
                done = {1: 1, 2: 9, 3: 129}[n]
                text = (f'vllm:generation_tokens_total{{engine="0"}} {gen}.0\n'
                        f'vllm:num_preemptions_total{{engine="0"}} 0.0\n'
                        f'vllm:request_time_per_output_token_seconds_sum{{engine="0"}} {done * 0.03}\n'
                        f'vllm:request_time_per_output_token_seconds_count{{engine="0"}} {done}\n'
                        f'vllm:time_to_first_token_seconds_sum{{engine="0"}} {done * 0.5}\n'
                        f'vllm:time_to_first_token_seconds_count{{engine="0"}} {done}\n')
                return 200, {"status": "COMPLETED", "output": [text]}
            if job_input.get("openai_route"):
                # 30 at a time on the worker: batch b ends at (b + 1) x the time one batch of 512-token
                # answers takes at this config's throughput.
                i = self.job_index[m.group(2)]
                batch = 30 * job_input["openai_input"]["max_tokens"] / self.tok_s(eid)
                return 200, {"status": "COMPLETED", "delayTime": (i // 30) * batch * 1000,
                             "executionTime": batch * 1000,
                             "output": [{"usage": {"completion_tokens": job_input["openai_input"]["max_tokens"]}}]}
            if self.is_ghost(eid):
                return 200, {"status": "IN_QUEUE"}
            if self.boot_missing:
                return 200, {"status": "COMPLETED", "output": [{"error": {"message": "unknown input"}}]}
            return 200, {"status": "COMPLETED", "delayTime": 61000, "executionTime": 40,
                         "output": [{"day0_boot_report": self.boot(eid)}]}
        if "/cancel/" in path or path.endswith("/purge-queue"):
            return 200, {}
        raise AssertionError(f"unexpected call {method} {url}")


def args(**kw):
    a = dict(image="ghcr.io/a/worker-vllm:day0-v0.32.0--day0-sweep-3b31d89", model="Qwen/Qwen3-8B",
             gpu_class="48gb", gpu_type=None, volume_id="vol1", s0_env={}, configs=list(sweep.ORDER),
             cap_minutes=270, max_hours=4.5, start_minutes=25, cold_minutes=15, disk_gb=40, min_cuda="13.0", seed=0, keep=False,
             gpu_types=list(sweep.GPU_CLASSES["48gb"]))
    a.update(kw)
    return types.SimpleNamespace(**a)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    def make(**kw):
        api = FakeRunpod(**kw)
        monkeypatch.setattr(rp, "call", api.call)
        monkeypatch.setattr(sweep.time, "sleep", lambda s: None)
        monkeypatch.setattr(sweep, "image_digest", lambda image: "sha256:abc")

        def run_level(url, key, model, prompt_list, concurrency, max_tokens, **_):
            eid = url.split("/v2/")[1].split("/")[0]
            assert len({tuple(p[:8]) for p in prompt_list}) == len(prompt_list)  # distinct prompts
            tok = api.tok_s(eid) if concurrency == 30 else 30.0
            return {"concurrency": concurrency, "requests": len(prompt_list), "failed": 0,
                    "output_tokens": len(prompt_list) * max_tokens, "output_tok_s": tok,
                    "ttft_p50_ms": 900.0, "ttft_p99_ms": 2000.0, "tpot_p50_ms": 30.0}, []

        monkeypatch.setattr(bench_client, "run_level", run_level)
        monkeypatch.setattr(bench_client, "stream_one", lambda url, key, payload, timeout=900: {
            "ok": True, "error": None, "out_tokens": payload["max_tokens"], "chunks": payload["max_tokens"]})
        return api
    return make


def test_full_sweep(fake, tmp_path):
    api = fake()
    s = sweep.Sweep("k", args(), 32768, 151936, tmp_path).run()
    assert s["gpu_type"] == "NVIDIA L40S" and s["gpu_class"] == "48gb"
    assert s["winners"] == ["S3", "S5"]
    assert s["recommended"] == "W" and s["fast_start"] == "S2" and s["verdict"] == "ok"
    assert s["stopped"] is None
    block = s["recommended_block"]
    assert block["env"] == {"MODEL_NAME": "Qwen/Qwen3-8B", "MAX_MODEL_LEN": "32768", "VLLM_ENABLE_STARTUP_PLAN": "1",
                            "MAX_NUM_SEQS": "32", "MAX_NUM_BATCHED_TOKENS": "8192"}
    assert block["gpu_type"] == "NVIDIA L40S" and block["digest"] == "sha256:abc"
    # The first endpoint offered the whole class; every later one is pinned to the GPU the warm-up got.
    eps = list(api.endpoints.values())
    assert len(eps[0]["gpuTypeIds"]) == 4 and all(e["gpuTypeIds"] == ["NVIDIA L40S"] for e in eps[1:])
    assert all(e["networkVolumeId"] == "vol1" and e["flashboot"] is False and e["idleTimeout"] == sweep.IDLE_TIMEOUT_S for e in eps)
    # Every endpoint and template is gone at the end.
    assert sum(p.startswith("/endpoints/") for p in api.deleted) == len(api.endpoints)
    assert sum(p.startswith("/templates/") for p in api.deleted) == len(api.templates)
    # Files: one per config, the summary, the two lakehouse tables.
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [f"{c}.json" for c in sweep.ORDER] + ["summary.json", "config_runs.jsonl", "config_summary.jsonl"])
    s3 = json.loads((tmp_path / "S3.json").read_text())
    assert [r["kind"] for r in s3["runs"]] == ["warmup", "cold", "cold", "cold"]
    assert s3["bench"]["metrics_match"] is True and s3["bench"]["metrics"]["c30"]["num_preemptions"] == 0
    gold = [json.loads(x) for x in (tmp_path / "config_summary.jsonl").read_text().splitlines()]
    assert [g["profile"] for g in gold if g["profile"] != "none"] == ["fast_start", "recommended"]
    silver = [json.loads(x) for x in (tmp_path / "config_runs.jsonl").read_text().splitlines()]
    assert {r["run_kind"] for r in silver} == {"warmup", "cold", "bench"}


def test_lakehouse_columns_match_the_spec(fake, tmp_path):
    fake()
    sweep.Sweep("k", args(configs=["S0", "S1"]), 32768, 151936, tmp_path).run()
    spec = (Path(__file__).resolve().parent.parent / "docs" / "optimal-config.md").read_text()

    def cols(table):
        m = re.search(rf"`{table}`, one row per [^:]+: `([^`]+)`", spec)
        return [c.split(" ")[0] for c in m.group(1).split(", ")]

    silver = json.loads((tmp_path / "config_runs.jsonl").read_text().splitlines()[0])
    gold = json.loads((tmp_path / "config_summary.jsonl").read_text().splitlines()[0])
    assert list(silver) == cols("config_runs")
    assert list(gold) == cols("config_summary")


def test_stops_without_a_boot_report(fake, tmp_path):
    api = fake(boot_missing=True)
    s = sweep.Sweep("k", args(), 32768, 151936, tmp_path).run()
    assert s["stopped"]["reason"] == "no_boot_report" and s["recommended"] is None
    assert s["verdict"] == "incomplete"
    assert len(api.jobs) == 1  # no retry: the image can't answer, a second start would say the same


def test_stops_when_no_gpu_is_free(fake, tmp_path):
    fake(no_gpu=True)
    s = sweep.Sweep("k", args(start_minutes=0.001, cold_minutes=0.001), 32768, 151936, tmp_path).run()
    assert s["stopped"]["reason"] == "no_gpu" and "two waits" in s["stopped"]["detail"]
    s0 = json.loads((tmp_path / "S0.json").read_text())
    assert [r["retry"] for r in s0["runs"]] == [False, True]  # one more wait before giving up
    assert s["worker_minutes"] == 0.0  # a throttled start never had a worker to pay for


def test_budget_guard(fake, tmp_path):
    fake()
    sw = sweep.Sweep("k", args(cap_minutes=30), 32768, 151936, tmp_path)
    sw.minutes = 25.0
    with pytest.raises(sweep.Stop) as e:
        sw.guard("cold")  # 25 + the 6 min first estimate > 30
    assert e.value.reason == "budget"
    sw.minutes = 10.0
    sw.guard("cold")


def test_inconclusive_s1_stops_the_sweep(fake, tmp_path, monkeypatch):
    api = fake()
    inits = iter([40.0, 20.0, 26.0, 33.0] * 3)  # S1's cold starts spread 13 s around 26 s
    orig = api.boot

    def noisy(eid):
        b = orig(eid)
        if api.env(eid).get("VLLM_ENABLE_STARTUP_PLAN"):
            b["vllm"]["engine_init_s"] = next(inits)
        return b

    api.boot = noisy
    s = sweep.Sweep("k", args(), 32768, 151936, tmp_path).run()
    assert s["stopped"]["reason"] == "inconclusive" and s["verdict"] == "inconclusive"
    assert s["recommended"] is None
    assert [c["config"] for c in s["configs"]] == ["S0", "S1"]


# ---------------------------------------------------------------- where to run

AVAIL = {"US-TX-3": {"NVIDIA GeForce RTX 4090": "HIGH", "NVIDIA RTX A6000": "LOW"},
         "EU-RO-1": {"NVIDIA L40S": "MEDIUM", "NVIDIA RTX A6000": "HIGH"},
         "US-KS-2": {"NVIDIA L40S": "NONE"}}
TYPES = list(sweep.GPU_CLASSES["48gb"])


def test_types_in_a_data_center():
    assert sweep.types_in(AVAIL["EU-RO-1"], TYPES) == ["NVIDIA RTX A6000", "NVIDIA L40S"]  # most in stock first
    assert sweep.types_in(AVAIL["US-KS-2"], TYPES) == []
    assert sweep.types_in(None, TYPES) == []


def test_pick_volume_prefers_stock_and_needs_room():
    vols = [{"id": "v1", "dataCenterId": "US-TX-3", "size": 100}, {"id": "v2", "dataCenterId": "EU-RO-1", "size": 50},
            {"id": "v3", "dataCenterId": "EU-RO-1", "size": 10}]
    vol, have = sweep.pick_volume(vols, AVAIL, TYPES, 36)
    assert vol["id"] == "v2" and have[0] == "NVIDIA RTX A6000"
    assert sweep.pick_volume(vols[:1], AVAIL, ["NVIDIA L40S"], 36) == (None, None)


def test_pick_data_center_for_a_new_volume():
    assert sweep.pick_data_center(AVAIL, ["NVIDIA L40S"]) == ("EU-RO-1", ["NVIDIA L40S"])
    assert sweep.pick_data_center({"X": {"NVIDIA L40S": "NONE"}}, ["NVIDIA L40S"]) == (None, None)


# ---------------------------------------------------------------- failure paths (audit, Oct 27)

def test_flaky_health_never_reads_as_cold(fake, tmp_path):
    api = fake(flaky_health=True)
    s = sweep.Sweep("k", args(configs=["S0", "S1"]), 32768, 151936, tmp_path).run()
    assert s["stopped"] is None
    assert sum(p.startswith("/endpoints/") for p in api.deleted) == len(api.endpoints)


def test_workers_unknown_on_api_error(monkeypatch):
    monkeypatch.setattr(rp, "call", lambda *a, **k: (0, "RemoteDisconnected"))
    assert sweep.workers("k", "e") is None


def test_broken_openai_route_stops_after_the_first_warm_up(fake, tmp_path, monkeypatch):
    api = fake()
    monkeypatch.setattr(bench_client, "stream_one", lambda url, key, payload, timeout=900: {
        "ok": False, "error": "HTTP 400: prompt must be a string", "out_tokens": 0, "chunks": 0})
    s = sweep.Sweep("k", args(), 32768, 151936, tmp_path).run()
    assert s["stopped"]["reason"] == "bench_route" and "prompt must be a string" in s["stopped"]["detail"]
    assert len([j for j in api.jobs.values() if j[1].get("day0_boot_report")]) == 1  # one start paid for
    assert sum(p.startswith("/endpoints/") for p in api.deleted) == len(api.endpoints)


def test_an_error_mid_bench_still_writes_the_summary_and_cleans_up(fake, tmp_path, monkeypatch):
    api = fake()

    def boom(*a, **k):
        raise RuntimeError("socket closed")

    monkeypatch.setattr(bench_client, "run_level", boom)
    s = sweep.Sweep("k", args(), 32768, 151936, tmp_path).run()
    assert s["stopped"]["reason"] == "error" and "socket closed" in s["stopped"]["detail"]
    assert (tmp_path / "summary.json").exists() and (tmp_path / "S0.json").exists()
    assert all(e["workersMax"] == 0 and e["workersMin"] == 0 for e in api.endpoints.values())
    assert sum(p.startswith("/endpoints/") for p in api.deleted) == len(api.endpoints)


def test_interrupt_unwinds_through_cleanup(fake, tmp_path, monkeypatch):
    api = fake()
    calls = {"n": 0}
    real = sweep.run_job

    def interrupted(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("signal 15")
        return real(*a, **k)

    monkeypatch.setattr(sweep, "run_job", interrupted)
    s = sweep.Sweep("k", args(), 32768, 151936, tmp_path).run()
    assert s["stopped"]["reason"] == "interrupted"
    assert sum(p.startswith("/endpoints/") for p in api.deleted) == len(api.endpoints)


def test_a_reused_worker_is_not_a_cold_start(fake, tmp_path):
    api = fake()
    orig = api.boot

    def same_process(eid):
        b = orig(eid)
        b["process_start"] = 1000.0   # every answer from the same worker process
        return b

    api.boot = same_process
    pinned = args(configs=["S0"], gpu_type="NVIDIA L40S", gpu_types=["NVIDIA L40S"])  # one endpoint throughout
    s = sweep.Sweep("k", pinned, 32768, 151936, tmp_path).run()
    s0 = json.loads((tmp_path / "S0.json").read_text())
    assert s0["runs"][1].get("reused_worker") is True and s0["failed"]
    assert s["recommended"] is None


def test_startup_error_text_is_kept(fake, tmp_path):
    api = fake()
    orig = api.boot

    def failed(eid):
        b = orig(eid)
        b["startup_error"], b["vllm"] = True, {}
        return b

    api.boot = failed
    sweep.Sweep("k", args(configs=["S0"]), 32768, 151936, tmp_path).run()
    s0 = json.loads((tmp_path / "S0.json").read_text())
    assert s0["failed"] == "warm-up start failed twice"
    assert [r["retry"] for r in s0["runs"]] == [False, True]
    assert all("startup_error_output" in r for r in s0["runs"])


def test_missing_counter_fails_the_preemption_check():
    before = {"vllm:generation_tokens_total": 0.0}
    after = {"vllm:generation_tokens_total": 512.0}
    d = sweep.metric_delta(before, after)
    assert d["num_preemptions"] is None and d["generation_tokens"] == 512.0
    rec = record([25, 25.5, 26], 700)
    rec["bench"]["metrics"]["c30"] = d
    assert sweep.config_stats(rec)["fit_checks"]["no_failed_or_preempted"] is False


def test_cleanup_touches_only_sweep_names(fake):
    api = fake()
    api.endpoints = {"e0": {"id": "e0", "name": "day0-sweep-S1-1", "templateId": "t0", "workersMax": 1, "workersMin": 1},
                     "e1": {"id": "e1", "name": "my-prod-endpoint", "templateId": "t1", "workersMax": 2, "workersMin": 1}}
    api.templates = {"t0": {"id": "t0", "name": "day0-sweep-S1-1", "env": {}}, "t1": {"id": "t1", "name": "prod", "env": {}}}
    api.volumes = {"v0": {"id": "v0", "name": "day0-sweep-1", "dataCenterId": "X"},
                   "v1": {"id": "v1", "name": "hf-cache", "dataCenterId": "X"}}
    assert sweep.cleanup("k") == 0
    assert api.deleted == ["/endpoints/e0", "/templates/t0", "/networkvolumes/v0"]
    assert api.endpoints["e1"]["workersMin"] == 1  # untouched


def test_a_worker_swap_mid_bench_voids_the_counters(fake, tmp_path):
    api = fake()
    orig = api.boot
    starts = iter(range(10_000))

    def fresh_process(eid):
        b = orig(eid)
        b["process_start"] = float(next(starts))   # every boot report from a new process
        return b

    api.boot = fresh_process
    s = sweep.Sweep("k", args(configs=["S0"]), 32768, 151936, tmp_path).run()
    s0 = json.loads((tmp_path / "S0.json").read_text())
    assert s0["bench"]["worker_changed"] is True and s0["bench"]["metrics"]["c30"] is None
    assert s["configs"][0]["fits"] is False


def test_a_type_served_on_another_device_is_recorded_and_held(fake, tmp_path):
    """Oct 28 pilot: one requested type, a different nvidia-smi name; later starts must match it."""
    api = fake()
    orig = api.boot
    names = iter(["NVIDIA RTX PRO 6000 Blackwell Server Edition"] * 2 + ["NVIDIA RTX 6000 Ada Generation"]
                 + ["NVIDIA RTX PRO 6000 Blackwell Server Edition"] * 50)

    def served(eid):
        b = orig(eid)
        b["gpu"] = next(names)
        return b

    api.boot = served
    one = args(configs=["S0"], gpu_types=["NVIDIA RTX 6000 Ada Generation"])
    s = sweep.Sweep("k", one, 32768, 151936, tmp_path).run()
    assert s["gpu_type"] == "NVIDIA RTX 6000 Ada Generation"
    assert s["device"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    s0 = json.loads((tmp_path / "S0.json").read_text())
    cold = [r for r in s0["runs"] if r["kind"] == "cold"]
    assert cold[1].get("wrong_device") == "NVIDIA RTX 6000 Ada Generation" and cold[2]["retry"] is True
    assert s["configs"][0]["fits"] is True  # three good cold starts on the one device


def test_several_types_and_an_unknown_device_stop(fake, tmp_path):
    api = fake()
    orig = api.boot
    api.boot = lambda eid: {**orig(eid), "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}
    s = sweep.Sweep("k", args(configs=["S0"]), 32768, 151936, tmp_path).run()
    assert s["stopped"]["reason"] == "gpu" and "--gpu-type" in s["stopped"]["detail"]


def test_jobs_level_counts_short_and_failed_jobs(fake, monkeypatch):
    api = fake()
    api.endpoints["e9"] = {"id": "e9", "templateId": "t9", "workersMax": 1, "workersMin": 0}
    api.templates["t9"] = {"id": "t9", "env": {}}
    real = api.call

    def call(method, url, key, body=None):
        st, r = real(method, url, key, body)
        m = re.search(r"/status/(j\d+)$", url)
        if m and api.job_index.get(m.group(1)) == 3:
            return 200, {"status": "FAILED", "error": "worker restarted"}
        if m and api.job_index.get(m.group(1)) == 4:
            return 200, {**r, "output": [{"usage": {"completion_tokens": 118}}]}
        return st, r

    monkeypatch.setattr(rp, "call", call)
    summary, recs = sweep.jobs_level("k", "e9", "m", bench_client.prompts(60, 8, 32000), 512)
    assert summary["requests"] == 60 and summary["failed"] == 2 and summary["output_tokens"] == 58 * 512
    # two batches of 30 at the fake S0 rate of 700 tok/s; the two bad jobs count as no tokens
    assert summary["output_tok_s"] == pytest.approx(58 * 512 / (2 * 30 * 512 / 700), rel=0.01)
    assert summary["via"] == "jobs API, not streamed"
    assert any("worker restarted" in (r["error"] or "") for r in recs)


def test_route_probe_is_checked_on_a_retried_warm_up(fake, tmp_path, monkeypatch):
    api = fake()
    orig = api.boot
    calls = {"n": 0}

    def first_fails(eid):
        calls["n"] += 1
        b = orig(eid)
        if calls["n"] == 1:
            b["startup_error"], b["vllm"] = True, {}
        return b

    api.boot = first_fails
    monkeypatch.setattr(bench_client, "stream_one", lambda url, key, payload, timeout=900: {
        "ok": False, "error": "HTTP 400", "out_tokens": 0, "chunks": 0})
    s = sweep.Sweep("k", args(), 32768, 151936, tmp_path).run()
    assert s["stopped"]["reason"] == "bench_route"


def test_an_endpoint_that_never_schedules_is_replaced(fake, tmp_path):
    """Pilot runs 2-3: S1's fourth start saw no worker at all; a new endpoint takes over."""
    api = fake(ghost={"e1": 3})   # e1 is S1's endpoint (type pinned, so no pin re-create)
    pinned = args(gpu_type="NVIDIA L40S", gpu_types=["NVIDIA L40S"], start_minutes=0.01, cold_minutes=0.01)
    s = sweep.Sweep("k", pinned, 32768, 151936, tmp_path).run()
    s1 = json.loads((tmp_path / "S1.json").read_text())
    assert s1["recreated_from"] == ["e1"] and s1["endpoint_id"] != "e1"
    assert [(r["kind"], r["index"], r["retry"], r["ok"]) for r in s1["runs"]] == [
        ("warmup", 0, False, True), ("cold", 1, False, True), ("cold", 2, False, True),
        ("cold", 3, False, False), ("cold", 3, True, True)]
    assert s["stopped"] is None and s["verdict"] == "ok" and s["recommended"]
    assert sum(p.startswith("/endpoints/") for p in api.deleted) == len(api.endpoints)


def test_a_stopped_sweep_reports_best_so_far_not_a_recommendation(fake, tmp_path):
    api = fake(ghost={"e1": 3})
    real = api.is_ghost
    api.is_ghost = lambda eid: real(eid) or (eid == "e2")   # the replacement never schedules either
    pinned = args(gpu_type="NVIDIA L40S", gpu_types=["NVIDIA L40S"], start_minutes=0.01, cold_minutes=0.01)
    s = sweep.Sweep("k", pinned, 32768, 151936, tmp_path).run()
    assert s["stopped"]["reason"] == "no_gpu" and s["verdict"] == "incomplete"
    assert s["recommended"] is None and s["best_so_far"] == "S0" and "recommended_block" not in s


def test_cold_starts_get_the_shorter_limit(fake, tmp_path):
    fake()
    sweep.Sweep("k", args(configs=["S0"], start_minutes=25, cold_minutes=15), 32768, 151936, tmp_path).run()
    s0 = json.loads((tmp_path / "S0.json").read_text())
    assert [r["limit_min"] for r in s0["runs"]] == [25, 15, 15, 15]
