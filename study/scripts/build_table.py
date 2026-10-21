"""Build the day-0 lag table.

Measured proxy, per model:
  upstream  public release -> first vllm/vllm-openai:vX.Y.Z image whose
            registry.py lists the model's architecture class
  worker    -> first moment the latest published runpod/worker-v1-vllm
            release image pins a vLLM whose registry lists that class
"Lists the class" is necessary for serving, not sufficient (parsers, kernels,
hardware are not checked).

Conventions:
  * Events are ordered by full UTC push timestamps; only events at or before
    the cutoff (end of AS_OF, UTC) count.
  * Day counts are UTC calendar date of the event minus the release date
    (the lab's local date). Expect +/-1 day of noise.
  * Nothing is available before the release date, so availability that
    predates it is clamped to the release date.
  * Support at a worker pin is checked by registry membership at that exact
    vLLM tag, not by version ordering.

Inputs (data/): models.csv, vllm_registry_by_tag.json, vllm_images.csv,
worker_tags.csv, worker_images.csv, vllm_day0_images.csv
Outputs: data/lag_table.csv, data/summary.txt
"""
import csv, json, statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data"
AS_OF = date(2026, 10, 21)
CUTOFF = datetime(2026, 10, 21, 23, 59, 59, tzinfo=timezone.utc)
WINDOW_START = date(2026, 1, 1)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def rows(name):
    return list(csv.DictReader(open(D / name)))


registry = json.loads((D / "vllm_registry_by_tag.json").read_text())
vllm_pub = sorted((ts(r["pushed_utc"]), r["vllm_tag"]) for r in rows("vllm_images.csv"))
vllm_pub = [(t, tag) for t, tag in vllm_pub if t <= CUTOFF]
vllm_pub_at = {tag: t for t, tag in vllm_pub}

pins = {r["git_tag"]: "v" + r["vllm_pin"] for r in rows("worker_tags.csv")}
# (push time, worker image tag, pinned vLLM tag), chronological, cut off at CUTOFF
worker = sorted((ts(r["pushed_utc"]), r["image_tag"], pins[r["image_tag"]]) for r in rows("worker_images.csv"))
worker = [w for w in worker if w[0] <= CUTOFF]
day0 = {r["model"]: r for r in rows("vllm_day0_images.csv")}


def supports(vllm_tag, archs):
    return any(a in registry[vllm_tag]["classes"] for a in archs)


def latest_worker_at(t):
    cur = None
    for w in worker:
        if w[0] <= t:
            cur = w
    return cur


def first_vllm(archs):
    for t, tag in vllm_pub:
        if supports(tag, archs):
            return t, tag
    return None


def worker_history(archs, start):
    """Walk worker image pushes from `start`. Returns (first_support, outages)
    where first_support is (time, image) or None and outages is a list of
    (lost_time, lost_image, restored_time or None, restored_image or None)."""
    events = [(start, latest_worker_at(start))] + [(w[0], w) for w in worker if w[0] > start]
    first, outages, state = None, [], False
    for t, img in events:
        ok = img is not None and supports(img[2], archs)
        if ok and first is None:
            first = (t, img)
        if first is not None and state and not ok:
            outages.append([t, img, None, None])
        if first is not None and not state and ok and outages and outages[-1][2] is None:
            outages[-1][2], outages[-1][3] = t, img
        state = ok
    return first, outages


def days(t, rel):
    return (t.date() - rel).days


out = []
for m in rows("models.csv"):
    archs = [a for a in m["architectures"].split(";") if a]
    rel = date.fromisoformat(m["public_release_date"])
    rel_t = datetime(rel.year, rel.month, rel.day, tzinfo=timezone.utc)
    r = {"name": m["name"], "group": m["group"], "repo_id": m["repo_id"], "architecture": archs[0],
         "public_release": rel.isoformat(), "release_confidence": m["confidence"],
         "vllm_pr": m["vllm_pr"], "vllm_merge": m["vllm_merge_date"],
         "vllm_merge_minus_release_days": (date.fromisoformat(m["vllm_merge_date"]) - rel).days if m["vllm_merge_date"] else ""}
    if m["name"] in day0:
        img = day0[m["name"]]
        r.update(day0_image=f"vllm/vllm-openai:{img['earliest_model_tag']}",
                 day0_image_lag_days=max(0, days(ts(img["pushed_utc"]), rel)))
    v = first_vllm(archs)
    if v is None:
        r.update(status="censored: no vLLM release yet",
                 upstream_lag_days=f">={(AS_OF - rel).days}", total_lag_days=f">={(AS_OF - rel).days}")
        out.append(r)
        continue
    v_t = max(v[0], rel_t)
    r.update(vllm_release=v[1], vllm_published=v[0].date().isoformat(), upstream_lag_days=days(v_t, rel))
    first, outages = worker_history(archs, v_t)
    if first is None:
        r.update(status="censored: no worker image yet",
                 worker_lag_days=f">={(AS_OF - v_t.date()).days}", total_lag_days=f">={(AS_OF - rel).days}")
    else:
        r.update(status="served", worker_image=f"{first[1][1]} (vLLM {first[1][2][1:]})",
                 worker_date=first[0].date().isoformat(),
                 worker_lag_days=(first[0].date() - v_t.date()).days,
                 total_lag_days=days(first[0], rel))
    if outages:
        o = outages[0]
        r.update(outage_from=f"{o[0].date()} {o[1][1]} (vLLM {o[1][2][1:]})",
                 outage_to=f"{o[2].date()} {o[3][1]}" if o[2] else "ongoing",
                 outage_days=(o[2].date() - o[0].date()).days if o[2] else f">={(AS_OF - o[0].date()).days}",
                 n_outages=len(outages))
    out.append(r)

cols = ["name", "group", "repo_id", "architecture", "public_release", "release_confidence", "status",
        "vllm_pr", "vllm_merge", "vllm_merge_minus_release_days", "vllm_release", "vllm_published",
        "upstream_lag_days", "worker_image", "worker_date", "worker_lag_days", "total_lag_days",
        "outage_from", "outage_to", "outage_days", "n_outages", "day0_image", "day0_image_lag_days"]
with open(D / "lag_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in sorted(out, key=lambda r: (r["group"], r["public_release"])):
        w.writerow({c: r.get(c, "") for c in cols})


def stat(rs, key):
    xs = [r[key] for r in rs if isinstance(r.get(key), int)]
    cens = [f"{r['name']} {r[key]}" for r in rs if str(r.get(key, "")).startswith(">=")]
    if not xs:
        return f"  {key}: none"
    return (f"  {key}: n={len(xs)} median={statistics.median(xs)} mean={statistics.mean(xs):.1f} "
            f"min={min(xs)} max={max(xs)} zero={sum(x == 0 for x in xs)}"
            + (f" | censored: {', '.join(cens)}" if cens else ""))


lines = [f"as of {AS_OF} (events cut off at {CUTOFF.isoformat()})"]
for g in ("new_arch", "control"):
    rs = [r for r in out if r["group"] == g]
    lines.append(f"\n[{g}] n={len(rs)}  status: " +
                 ", ".join(f"{s}={sum(r.get('status') == s for r in rs)}" for s in sorted({r.get('status') for r in rs})))
    for key in ("upstream_lag_days", "worker_lag_days", "total_lag_days", "outage_days",
                "vllm_merge_minus_release_days", "day0_image_lag_days"):
        lines.append(stat(rs, key))

# Worker staleness: for each UTC day in the window, compare the latest worker
# image's pin (end of day) with vLLM final images published by end of day.
day, n_days, newer_days, ages, behind = WINDOW_START, 0, 0, [], []
while day <= AS_OF:
    eod = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc)
    pin = latest_worker_at(eod)[2]
    newer = [t for t, tag in vllm_pub if t <= eod and tuple(map(int, tag[1:].split("."))) > tuple(map(int, pin[1:].split(".")))]
    n_days += 1
    newer_days += bool(newer)
    ages.append((day - min(newer).date()).days if newer else 0)
    behind.append(len(newer))
    day += timedelta(1)
lines.append(f"\n[worker staleness {WINDOW_START}..{AS_OF}, {n_days} days]")
lines.append(f"  days a newer vLLM final image was published than the latest worker pin: {newer_days}")
lines.append(f"  age of oldest such release (0 when none): median={statistics.median(ages)} max={max(ages)}")
lines.append(f"  vLLM finals behind: median={statistics.median(behind)} max={max(behind)}")
(D / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
