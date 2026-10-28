"""Staging weights on a network volume with a CPU pod (MiniMax retry, Oct 28)."""

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import smoke  # noqa: E402
import stage  # noqa: E402


class FakeAPI:
    def __init__(self, volume_fails=(), pod_fails=(), states=("installing", "downloading", "done"), vcpu_fails=()):
        self.volume_fails, self.pod_fails, self.vcpu_fails = set(volume_fails), set(pod_fails), set(vcpu_fails)
        self.states = list(states)
        self.calls, self.volumes, self.pods, self.endpoints = [], {}, {}, {}
        self.delete_refusals = {}
        self.n = 0

    def call(self, method, url, key, body=None):
        path = url.split("/v1", 1)[-1]
        self.calls.append((method, path, body))
        if method == "POST" and path == "/networkvolumes":
            if body["dataCenterId"] in self.volume_fails:
                return 400, "no capacity"
            vid, self.n = f"v{self.n}", self.n + 1
            self.volumes[vid] = {**body, "id": vid}
            return 200, {"id": vid}
        if method == "POST" and path == "/pods":
            if body["dataCenterIds"][0] in self.pod_fails or body["vcpuCount"] in self.vcpu_fails:
                return 500, "There are no instances currently available"
            pid, self.n = f"p{self.n}", self.n + 1
            self.pods[pid] = {**body, "id": pid, "networkVolume": {"id": body["networkVolumeId"]}}
            return 200, {"id": pid, "costPerHr": 0.24}
        if method == "GET" and path == "/networkvolumes":
            return 200, list(self.volumes.values())
        if method == "GET" and path == "/pods":
            return 200, list(self.pods.values())
        if method == "GET" and path == "/endpoints":
            return 200, list(self.endpoints.values())
        if method == "DELETE":
            kind, oid = path.strip("/").split("/")
            if self.delete_refusals.get(oid, 0):
                self.delete_refusals[oid] -= 1
                return 400, "in use"
            {"networkvolumes": self.volumes, "pods": self.pods, "endpoints": self.endpoints}.get(kind, {}).pop(oid, None)
            return 204, None
        return 200, {}


def setup(monkeypatch, api):
    monkeypatch.setattr(stage.rp, "call", api.call)
    monkeypatch.setattr(stage.time, "sleep", lambda s: None)
    it = iter(api.states)
    last = {}

    def status(pod_id, timeout=15):
        try:
            last["s"] = {"state": next(it), "bytes": 250 * 2 ** 30, "expected": 250 * 2 ** 30, "seconds": 600}
        except StopIteration:
            pass
        return last.get("s")

    monkeypatch.setattr(stage, "status", status)


def clock():
    t = [0.0]

    def now():
        t[0] += 20
        return t[0]
    return now


def test_pod_body_mounts_the_volume_where_the_worker_looks():
    b = stage.pod_body("day0-stage-1", "v0", "US-NC-1", "MiniMaxAI/MiniMax-M3.1-Flash")
    assert b["computeType"] == "CPU" and b["networkVolumeId"] == "v0" and b["dataCenterIds"] == ["US-NC-1"]
    assert b["volumeMountPath"] == "/runpod-volume"   # worker-vllm's BASE_PATH
    assert b["env"]["HF_HUB_CACHE"] == "/runpod-volume/huggingface-cache/hub"   # its HF_HUB_CACHE
    assert "8000/http" in b["ports"]
    assert all(" " not in x for x in b["dockerStartCmd"])
    script = base64.b64decode(b["env"]["STAGE_PY"]).decode()
    compile(script, "pod", "exec")   # the pod script at least parses
    assert script == stage.POD_SCRIPT and "*.safetensors" not in json.loads(b["env"]["IGNORE"])


def test_stage_keeps_the_volume_and_deletes_the_pod_when_done(monkeypatch):
    api = FakeAPI()
    setup(monkeypatch, api)
    res = stage.stage("k", "m/x", ["US-NC-1"], 300, 45, sleep=lambda s: None, clock=clock())
    assert res["state"] == "done" and res["volume_id"] in api.volumes and res["data_center"] == "US-NC-1"
    assert api.pods == {}
    assert res["est_cost_usd"] is not None


def test_stage_moves_on_when_a_data_center_has_no_cpu_pods(monkeypatch):
    api = FakeAPI(pod_fails={"US-NC-1"})
    setup(monkeypatch, api)
    res = stage.stage("k", "m/x", ["US-NC-1", "US-TX-3"], 300, 45, sleep=lambda s: None, clock=clock())
    assert res["state"] == "done" and res["data_center"] == "US-TX-3"
    assert list(api.volumes) == [res["volume_id"]]   # the first data center's volume was deleted
    assert res["tried"][0]["dc"] == "US-NC-1"


def test_stage_deletes_everything_on_error_or_timeout(monkeypatch):
    for states in (("downloading", "error"), ("downloading",) * 500):
        api = FakeAPI(states=states)
        setup(monkeypatch, api)
        res = stage.stage("k", "m/x", ["US-NC-1"], 300, 5, sleep=lambda s: None, clock=clock())
        assert res["state"] in ("error", "timeout") and res["volume_deleted"]
        assert api.volumes == {} and api.pods == {}


def test_no_data_center_takes_a_volume(monkeypatch):
    api = FakeAPI(volume_fails={"US-NC-1"})
    setup(monkeypatch, api)
    res = stage.stage("k", "m/x", ["US-NC-1"], 300, 45, sleep=lambda s: None, clock=clock())
    assert res["state"] == "no_data_center" and not api.pods


def test_cleanup_deletes_the_smoke_endpoint_before_its_volume(monkeypatch):
    api = FakeAPI()
    api.volumes = {"v9": {"id": "v9", "name": "day0-stage-9"}, "vx": {"id": "vx", "name": "someone-else"}}
    api.pods = {"p9": {"id": "p9", "name": "day0-stage-9", "networkVolume": {"id": "v9"}},
                "px": {"id": "px", "name": "my pod"}}
    api.endpoints = {"e9": {"id": "e9", "name": "day0-smoke-minimax-0", "networkVolumeId": "v9", "templateId": "t9"},
                     "ex": {"id": "ex", "name": "prod", "networkVolumeId": "vx"}}
    api.delete_refusals = {"e9": 2}   # workers still going away
    monkeypatch.setattr(stage.rp, "call", api.call)
    monkeypatch.setattr(stage.time, "sleep", lambda s: None)
    assert stage.cleanup("k") == 0
    assert set(api.volumes) == {"vx"} and set(api.pods) == {"px"} and set(api.endpoints) == {"ex"}
    deletes = [p for m, p, _ in api.calls if m == "DELETE"]
    assert deletes.index("/endpoints/e9") < deletes.index("/networkvolumes/v9")
    assert "/templates/t9" in deletes


def test_smoke_endpoint_mounts_the_staged_volume():
    assert "networkVolumeId" not in smoke.endpoint_body("141gb", "13.0")
    b = smoke.endpoint_body("141gb", "13.0", ["NVIDIA H200"], 2, volume_id="v0")
    assert b["networkVolumeId"] == "v0" and b["gpuCount"] == 2 and b["gpuTypeIds"] == ["NVIDIA H200"]


def test_data_centers_with_the_gpu_in_stock_most_first():
    avail = {"US-NC-1": {"NVIDIA H200": "LOW"}, "CA-MTL-4": {"NVIDIA H200": "MEDIUM"},
             "EU-RO-1": {"NVIDIA L40S": "HIGH"}, "US-CO-1": {"NVIDIA H200": "NONE", "NVIDIA H200 NVL": "LOW"}}
    # CA-MTL-4 and US-CO-1 have H200s but take no CPU pods through the REST API
    assert stage.data_centers_with(avail, ["NVIDIA H200", "NVIDIA H200 NVL"]) == ["US-NC-1"]
    assert stage.data_centers_with(None, ["NVIDIA H200"]) == []


def test_a_data_center_without_pods_is_skipped_without_a_volume(monkeypatch):
    api = FakeAPI()
    setup(monkeypatch, api)
    res = stage.stage("k", "m/x", ["CA-MTL-4", "US-NC-1"], 300, 45, sleep=lambda s: None, clock=clock())
    assert res["data_center"] == "US-NC-1"
    assert not any(b and b.get("dataCenterId") == "CA-MTL-4" for m, p, b in api.calls if p == "/networkvolumes")


def test_cleanup_reports_a_failed_list_instead_of_passing(monkeypatch):
    def fail(method, url, key, body=None):
        return 500, "boom"
    monkeypatch.setattr(stage.rp, "call", fail)
    assert stage.cleanup("k") == 1


def test_budget_counts_archived_smoke_runs(tmp_path):
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 10, 28, 6, 0, tzinfo=timezone.utc)
    rung = {"started": (now - timedelta(hours=1)).isoformat(), "gpu_class": "141gb", "est_cost_usd": 7.26}
    (tmp_path / "_runs").mkdir()
    (tmp_path / "_runs" / "try1.json").write_text(json.dumps({"rungs": [rung]}))
    (tmp_path / "_logs").mkdir()
    (tmp_path / "_logs" / "stage.json").write_text(json.dumps({"state": "done", "est_cost_usd": 0.2}))
    (tmp_path / "list.json").write_text("[]")
    assert smoke.spent_last_24h(now, tmp_path) == 7.26


def test_smaller_pods_are_tried_before_the_next_data_center(monkeypatch):
    api = FakeAPI(vcpu_fails={8})
    setup(monkeypatch, api)
    res = stage.stage("k", "m/x", ["US-NC-1"], 300, 45, sleep=lambda s: None, clock=clock())
    assert res["state"] == "done" and res["vcpu"] == 4
    sizes = [b["vcpuCount"] for m, p, b in api.calls if m == "POST" and p == "/pods"]
    assert sizes == [8, 4]


def test_done_needs_the_bytes_the_hub_lists():
    # Oct 28: "done" after 247 s with 17,032,957 bytes on the volume, for a 233 GiB checkpoint.
    sib = [("model-00001-of-00048.safetensors", 5 * 2 ** 30), ("config.json", 4_000),
           ("original/consolidated.pth", 9 * 2 ** 30), ("pytorch_model.bin", 7 * 2 ** 30)]
    want = stage.expected_bytes(sib, stage.IGNORE)
    assert want == 5 * 2 ** 30 + 4_000   # .pth under original/ and .bin are skipped
    assert not stage.complete(17_032_957, want) and stage.complete(want, want) and not stage.complete(0, 0)


def test_blob_bytes_skips_unfinished_downloads(tmp_path):
    blobs = tmp_path / "models--m--x" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "aaa").write_bytes(b"x" * 100)
    (blobs / "bbb.incomplete").write_bytes(b"x" * 1000)
    (tmp_path / "models--m--x" / "snapshots").mkdir()
    assert stage.blob_bytes(str(tmp_path / "models--m--x")) == 100


def test_pod_script_carries_the_check_and_skips_xet():
    b = stage.pod_body("day0-stage-1", "v0", "US-NC-1", "m/x")
    script = base64.b64decode(b["env"]["STAGE_PY"]).decode()
    assert "def blob_bytes" in script and "def complete" in script and "files_metadata=True" in script
    assert b["env"]["HF_HUB_DISABLE_XET"] == "1"


def test_the_runner_rejects_a_done_without_the_bytes(monkeypatch):
    api = FakeAPI()
    monkeypatch.setattr(stage.rp, "call", api.call)
    monkeypatch.setattr(stage.time, "sleep", lambda s: None)
    monkeypatch.setattr(stage, "status", lambda pid, timeout=15: {"state": "done", "bytes": 17_032_957,
                                                                 "expected": 250 * 2 ** 30})
    res = stage.stage("k", "m/x", ["US-NC-1"], 300, 45, sleep=lambda s: None, clock=clock())
    assert res["state"] == "error" and "17032957" in res["detail"]
    assert api.volumes == {} and api.pods == {}


def keep_setup(monkeypatch, detach_works):
    api = FakeAPI()
    api.volumes = {"v9": {"id": "v9", "name": "day0-stage-9"}}
    api.endpoints = {"e9": {"id": "e9", "name": "day0-smoke-minimax-0", "networkVolumeId": "v9", "templateId": "t9"}}
    real = api.call

    def call(method, url, key, body=None):
        path = url.split("/v1", 1)[-1]
        if method == "PATCH" and body and "networkVolumeIds" in body and detach_works:
            api.endpoints["e9"]["networkVolumeId"] = None
        if method == "DELETE" and path == "/networkvolumes/v9" and api.endpoints.get("e9", {}).get("networkVolumeId"):
            api.calls.append((method, path, body))
            return 400, "volume in use"
        return real(method, url, key, body)

    monkeypatch.setattr(stage.rp, "call", call)
    monkeypatch.setattr(stage.time, "sleep", lambda s: None)
    return api


def test_keep_endpoints_keeps_a_failed_endpoint_when_the_volume_detaches(monkeypatch):
    api = keep_setup(monkeypatch, detach_works=True)
    assert stage.cleanup("k", keep_endpoints=True) == 0
    assert api.volumes == {} and "e9" in api.endpoints


def test_keep_endpoints_still_deletes_the_volume_when_detach_doesnt_take(monkeypatch):
    api = keep_setup(monkeypatch, detach_works=False)
    assert stage.cleanup("k", keep_endpoints=True) == 0
    assert api.volumes == {} and api.endpoints == {}   # no volume left billing
