"""Minimal Runpod REST + Serverless job client (stdlib only).

Shapes checked against docs.runpod.io: creates return 201, deletes 204, and
Cloudflare in front of the API rejects urllib's default User-Agent (error 1010).
"""

import http.client
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
JOBS = "https://api.runpod.ai/v2"
UA = "day0-ready/0.1 (+https://github.com/ashwinsreedhar28/day0-ready)"


def load_key(env_file=None):
    if env_file:
        for line in Path(env_file).expanduser().read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith("RUNPOD_API_KEY="):
                os.environ["RUNPOD_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set (export it or pass --env-file)")
    return key


def call(method, url, key, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json",
        "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:500]
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException, ValueError) as e:
        # Dropped connections and non-JSON bodies come back as status 0 too: callers check ok().
        return 0, f"{type(e).__name__}: {e}"[:500]


def ok(status):
    return 200 <= status < 300


def items(resp, key):
    """List responses may come back as a bare list or wrapped in an object."""
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        return resp.get(key) or resp.get("data") or []
    return []


def balance(key):
    """Account balance in USD, or None if it can't be read (callers refuse to spend then)."""
    st, r = call("POST", GRAPHQL, key, {"query": "query { myself { clientBalance } }"})
    try:
        return float(r["data"]["myself"]["clientBalance"])
    except (TypeError, KeyError, ValueError):
        return None
