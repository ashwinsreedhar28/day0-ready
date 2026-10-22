"""Create, inspect, stop or terminate the Flint benchmark pod (Runpod REST v1).

    python3 bench/flint-b200/pod.py create | start | status | stop | terminate --yes

Uses RUNPOD_API_KEY from the environment. The pod runs vllm/vllm-openai:flint with
sshd added at start (custom images don't run it), reachable on its public IP.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import runpod_api as rp  # noqa: E402

STATE = Path(__file__).with_name(".pod.json")
IMAGE = "vllm/vllm-openai:flint"
START = (
    "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server >/dev/null"
    " && mkdir -p ~/.ssh /run/sshd && chmod 700 ~/.ssh"
    ' && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && echo "$SSH_PUBLIC_KEY" >> ~/.ssh/authorized_keys'
    " && chmod 600 ~/.ssh/authorized_keys && service ssh start && sleep infinity"
)


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    key = rp.load_key(None)
    s = state()
    if cmd == "create":
        if s.get("id"):
            sys.exit(f"pod {s['id']} already recorded in {STATE}; stop/terminate it first")
        body = {
            "name": "flint-bench-b200", "imageName": IMAGE, "gpuTypeIds": ["NVIDIA B200"], "gpuCount": 1,
            "cloudType": "SECURE", "interruptible": False, "containerDiskInGb": 200,
            "ports": ["22/tcp"], "env": {"SSH_PUBLIC_KEY": (Path.home() / ".ssh/runpod.pub").read_text().strip()},
            "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [START],
        }
        st, p = rp.call("POST", f"{rp.REST}/pods", key, body)
        if not rp.ok(st):
            sys.exit(f"create failed: {st} {p}")
        s = {"id": p["id"], "costPerHr": p.get("costPerHr"), "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        STATE.write_text(json.dumps(s))
        print(f"created pod {s['id']} at ${s['costPerHr']}/hr; billing starts now. Next: pod.py status")
    elif cmd == "start":  # restart a stopped pod; its container disk comes back empty
        st, p = rp.call("POST", f"{rp.REST}/pods/{s['id']}/start", key)
        if not rp.ok(st):
            sys.exit(f"start failed: {st} {p}")
        s["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        STATE.write_text(json.dumps(s))
        print(f"started pod {s['id']}; billing again from now (status counts from here). Next: pod.py status")
    elif cmd == "status":
        st, p = rp.call("GET", f"{rp.REST}/pods/{s['id']}", key)
        if not rp.ok(st):
            sys.exit(f"status failed: {st} {p}")
        ip, ports = p.get("publicIp"), p.get("portMappings") or {}
        mins = (datetime.now(timezone.utc) - datetime.fromisoformat(s["created"])).total_seconds() / 60
        cost = (s.get("costPerHr") or p.get("costPerHr") or 0) * mins / 60
        print(f"pod {s['id']}: {p.get('desiredStatus')}, image {p.get('imageName')}, ${p.get('costPerHr')}/hr, "
              f"{mins:.0f} min since create (~${cost:.2f})")
        if ip and ports.get("22"):
            print(f"ssh -i ~/.ssh/runpod -o StrictHostKeyChecking=accept-new -p {ports['22']} root@{ip}")
            (STATE.with_name(".ssh_target")).write_text(f"-p {ports['22']} root@{ip}\n")
        else:
            print("no public IP / SSH port yet (still pulling the image); run status again in a minute")
    elif cmd == "stop":
        st, p = rp.call("POST", f"{rp.REST}/pods/{s['id']}/stop", key)
        mins = (datetime.now(timezone.utc) - datetime.fromisoformat(s["created"])).total_seconds() / 60
        print(f"stop: {st} {p if not rp.ok(st) else ''} after {mins:.0f} min (~${(s.get('costPerHr') or 0) * mins / 60:.2f})")
    elif cmd == "terminate":
        if "--yes" not in sys.argv:
            sys.exit("terminate deletes the pod and its disk; rerun with --yes")
        st, p = rp.call("DELETE", f"{rp.REST}/pods/{s['id']}", key)
        print(f"terminate: {st} {p if not rp.ok(st) else ''}")
        if rp.ok(st):
            STATE.unlink()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
