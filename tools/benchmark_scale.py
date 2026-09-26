"""NFR 2: 20,000 sound-event records and several users at once, measured over HTTP.

    .venv/bin/python tools/benchmark_scale.py            # ~10 min; writes reports/scale.json

The run builds a throwaway database (``database/init_db.py`` plus 20,000 synthetic events
spread over 60 days, every class, severity, status, quality and source), serves it with
gunicorn as the README recommends (one worker, eight threads), and sends real HTTP requests from signed-in
users: the events list, each search filter, the Events, Dashboard and Analytics pages, the
dashboard APIs and the CSV export. It then repeats the four-client 30-second upload test
from ``tools/benchmark_latency.py`` against the same server. The demo database is never
touched, and the synthetic rows carry ``scale-test`` as their location.

Pass line, fixed before the first run: with 10 users at once, the 95th-percentile response
of every page and API read is at most 1 s and no request fails. The CSV export is reported
separately: an all-rows export must be refused (the app caps exports at 10,000 events) and the
largest allowed export must finish within 10 s. Results are written whether or not they pass.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_EVENTS = 20_000
LEVELS = (1, 10, 20)
ROUNDS = 3
WORKERS, THREADS = 1, 8  # one worker: live-window streaks are per process
READ_P95_LIMIT_S = 1.0
LOGIN_SPACING_S = 3.5
EXPORT_LIMIT_S = 10.0

CLASSES = ["Machinery Fault", "Glass Breaking", "Alarm or Siren", "Vehicle Horn", "Animal Sound",
           "Gunshot", "Panic Scream", "Aggression", "Person Asking for Help", "Background Noise"]
SEVERITIES = ["Informational", "Low", "Medium", "High", "Critical"]
STATUSES = ["Classified", "Uncertain", "Alert Generated", "Manual Review", "Reviewed", "Closed"]
CONSISTENCY = ["Strong Match", "Acceptable Match", "Weak Match", "Model Disagreement",
               "Uncertain Result"]
QUALITY = ["Good", "Acceptable", "Poor"]

# (name, path, capability). Every role can list its own events; the rest follow the grants.
READS = [
    ("events_list", "/api/events?per_page=50", "view_own_events"),
    ("filter_class", "/api/events?sound_class=Gunshot", "view_own_events"),
    ("filter_severity", "/api/events?severity=Critical", "view_own_events"),
    ("filter_status", "/api/events?status=Manual%20Review", "view_own_events"),
    ("filter_consistency", "/api/events?consistency_status=Model%20Disagreement", "view_own_events"),
    ("filter_quality", "/api/events?quality=Poor", "view_own_events"),
    ("filter_source", "/api/events?source=microphone", "view_own_events"),
    ("filter_dates", "/api/events?date_from=2026-08-10&date_to=2026-08-20", "view_own_events"),
    ("filter_confidence", "/api/events?confidence_min=0.9", "view_own_events"),
    ("filter_text", "/api/events?q=SST-2026-07-28-0012", "view_own_events"),
    ("filter_combined", "/api/events?sound_class=Glass%20Breaking&severity=High"
                        "&date_from=2026-08-01&date_to=2026-09-01", "view_own_events"),
    ("page_events", "/events?sound_class=Aggression", "view_own_events"),
    ("page_dashboard", "/dashboard", "view_own_events"),
    ("api_dashboard_summary", "/api/dashboard/summary", "view_dashboards"),
    ("api_timeline", "/api/dashboard/timeline", "view_dashboards"),
    ("page_analytics", "/analytics", "view_analytics"),
    ("api_analytics_classes", "/api/analytics/classes", "view_analytics"),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    values = sorted(values)
    return {"n": len(values), "median_s": round(statistics.median(values), 4),
            "p95_s": round(values[max(0, int(round(0.95 * len(values))) - 1)], 4),
            "max_s": round(values[-1], 4)}


# database --------------------------------------------------------------------------

def seed(db_path: Path) -> dict:
    from sqlalchemy import func, select

    from src.db import create_engine_for, format_audio_id, make_session_factory, session_scope
    from src.models import AudioFile, Event, ModelVersion, User

    engine = create_engine_for(str(db_path))
    factory = make_session_factory(engine)
    rng = np.random.default_rng(20260927)
    started = time.perf_counter()
    with session_scope(factory) as s:
        users = s.execute(select(User).order_by(User.id)).scalars().all()
        versions = {m.model_name: m.id for m in s.execute(
            select(ModelVersion).where(ModelVersion.is_active.is_(True))).scalars()}
        base_day = dt.datetime(2026, 7, 28)  # 60 days ending 26 Sep
        for chunk in range(0, N_EVENTS, 2000):
            audio = [AudioFile(
                audio_id=format_audio_id(i + 1, base_day),
                filename=f"scale_test_{i:05d}.wav",
                stored_path=f"scale-test/{i:05d}.wav",
                sha256=f"{i:064x}",
                size_bytes=int(rng.integers(40_000, 2_000_000)),
                duration_sec=float(rng.uniform(1, 30)),
                sample_rate=16000, channels=1, bit_depth=16,
                source="microphone" if i % 5 == 0 else "upload",
                created_by_id=users[i % len(users)].id,
            ) for i in range(chunk, min(chunk + 2000, N_EVENTS))]
            s.add_all(audio)
            s.flush()
            events = []
            for a in audio:
                i = int(a.filename[11:16])
                top = float(rng.uniform(0.3, 1.0))
                events.append(Event(
                    audio_file_id=a.id, source=a.source,
                    status=STATUSES[i % len(STATUSES)],
                    predicted_class=CLASSES[i % len(CLASSES)],
                    severity=SEVERITIES[(i // 3) % len(SEVERITIES)],
                    consistency_status=CONSISTENCY[(i // 7) % len(CONSISTENCY)],
                    confidence_difference=float(rng.uniform(0, 0.6)),
                    top_confidence=top,
                    quality_verdict=QUALITY[(i // 11) % len(QUALITY)],
                    requires_manual_review=(i % 4 == 0),
                    python_model_version_id=versions.get("python"),
                    gtm_model_version_id=versions.get("gtm"),
                    location="scale-test",
                    created_by_id=a.created_by_id,
                    created_at=base_day + dt.timedelta(minutes=i * 4.32),
                ))
            s.bulk_save_objects(events)
        s.flush()
        counts = {"events": s.scalar(select(func.count()).select_from(Event)),
                  "audio_files": s.scalar(select(func.count()).select_from(AudioFile)),
                  "users": len(users)}
    engine.dispose()
    return {**counts, "seconds": round(time.perf_counter() - started, 1),
            "db_file_mb": round(db_path.stat().st_size / 1e6, 1)}


# HTTP ------------------------------------------------------------------------------

def sign_in(base: str, username: str, password: str) -> tuple[requests.Session, str]:
    s = requests.Session()
    page = s.get(f"{base}/login", timeout=30).text
    token = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
    r = s.post(f"{base}/session", data={"username": username, "password": password,
                                        "csrf_token": token}, timeout=30)
    if r.status_code != 200 or "/login" in r.url:
        raise RuntimeError(f"sign-in failed for {username}: {r.status_code} {r.url}")
    return s, token


def session_pool(base: str, accounts: list[dict], size: int) -> list[tuple[requests.Session, list]]:
    """Sign ``size`` users in once, spaced to stay under the login rate limits
    (config/auth.json: 20 per IP and 10 per username per minute; every request here comes
    from 127.0.0.1). The load levels then reuse them, as signed-in users would."""
    from src.auth import ROLE_GRANTS

    pool = []
    for n in range(size):
        account = accounts[n % len(accounts)]
        s, _ = sign_in(base, account["username"], account["password"])
        grants = ROLE_GRANTS.get(account["role"], frozenset())
        pool.append((s, [r for r in READS if r[2] in grants]))
        time.sleep(LOGIN_SPACING_S)
    return pool


def read_load(base: str, pool: list, users: int) -> dict:
    timings: dict[str, list[float]] = {}
    errors: list[str] = []
    lock = threading.Lock()
    sessions = pool[:users]
    barrier = threading.Barrier(users)

    def worker(s: requests.Session, mix: list[tuple[str, str, str]]) -> None:
        barrier.wait()
        for _ in range(ROUNDS):
            for name, path, _cap in mix:
                t0 = time.perf_counter()
                try:
                    r = s.get(f"{base}{path}", timeout=60)
                    ok = r.status_code == 200
                    detail = f"{name}: HTTP {r.status_code}"
                except requests.RequestException as exc:
                    ok, detail = False, f"{name}: {exc}"
                elapsed = time.perf_counter() - t0
                with lock:
                    timings.setdefault(name, []).append(elapsed)
                    if not ok:
                        errors.append(detail)

    wall = time.perf_counter()
    threads = [threading.Thread(target=worker, args=pair) for pair in sessions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall
    everything = [v for values in timings.values() for v in values]
    per_endpoint = {name: summary(values) for name, values in sorted(timings.items())}
    worst = max(per_endpoint.items(), key=lambda kv: kv[1].get("p95_s", 0))
    return {"users": users, "requests": len(everything), "errors": len(errors),
            "error_samples": errors[:5], "wall_clock_s": round(wall, 2),
            "requests_per_s": round(len(everything) / wall, 1),
            "all_reads": summary(everything),
            "worst_endpoint_p95": {"endpoint": worst[0], "p95_s": worst[1]["p95_s"]},
            "per_endpoint": per_endpoint}


def export_timing(base: str, admin: dict) -> dict:
    """The export refuses more than 10,000 rows (src/api/reports_api.py), so time both the
    refusal of an all-rows export and the largest export it allows (one date range)."""
    s, _ = sign_in(base, admin["username"], admin["password"])
    out = {}
    for name, query in (("all_rows", ""), ("date_range", "?date_from=2026-07-28&date_to=2026-08-24")):
        t0 = time.perf_counter()
        r = s.get(f"{base}/api/export/events.csv{query}", timeout=300)
        elapsed = time.perf_counter() - t0
        out[name] = {"status": r.status_code, "seconds": round(elapsed, 2),
                     "rows": max(0, r.text.count("\n") - 1) if r.status_code == 200 else 0,
                     "bytes": len(r.content),
                     "message": r.json().get("error", {}).get("message") if r.status_code != 200 else None}
    out["within_limit"] = (out["all_rows"]["status"] == 413 and out["date_range"]["status"] == 200
                           and out["date_range"]["seconds"] <= EXPORT_LIMIT_S)
    return out


def upload_load(base: str, admin: dict) -> dict:
    from tools.benchmark_latency import test_audio, wav_bytes

    rng = np.random.default_rng(1)
    clip = test_audio(30.0)

    def upload(s: requests.Session, token: str) -> float:
        body = wav_bytes(clip * rng.uniform(0.8, 0.95))
        t0 = time.perf_counter()
        r = s.post(f"{base}/api/audio/upload", files={"file": ("scale.wav", body, "audio/wav")},
                   headers={"X-CSRF-Token": token}, timeout=300)
        if r.status_code != 201:
            raise RuntimeError(f"upload failed: {r.status_code} {r.text[:200]}")
        return time.perf_counter() - t0

    s, token = sign_in(base, admin["username"], admin["password"])
    for _ in range(WORKERS * 2):  # every worker loads its models on its first request
        upload(s, token)
    sequential = [upload(s, token) for _ in range(3)]
    concurrent: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        ws, wt = sign_in(base, admin["username"], admin["password"])
        for _ in range(2):
            t = upload(ws, wt)
            with lock:
                concurrent.append(t)

    wall = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall
    return {"clip_s": 30, "sequential": summary(sequential),
            "four_clients": {**summary(concurrent), "wall_clock_s": round(wall, 2)}}


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="sst-scale-"))
    db_path = tmp / "scale.db"
    env = {**os.environ, "SST_DB_PATH": str(db_path), "SST_STORAGE_DIR": str(tmp / "storage"),
           "SST_PRODUCTION": "0", "SST_LOAD_MODELS": "1",
           # One PyTorch thread per physical core: 8 threads on this 4-core, 8-thread laptop
           # made a 30 s upload 40 % slower than 4 did.
           "SST_TORCH_THREADS": str(max(1, (os.cpu_count() or 2) // 2))}
    subprocess.run([sys.executable, "database/init_db.py"], cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL)
    print("seeding 20,000 synthetic events ...", flush=True)
    seeded = seed(db_path)
    print(json.dumps(seeded), flush=True)

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = (tmp / "gunicorn.log").open("w")
    server = subprocess.Popen(
        [str(Path(sys.executable).with_name("gunicorn")), "-w", str(WORKERS), "--threads", str(THREADS),
         "--timeout", "300", "-b", f"127.0.0.1:{port}", "src.app:create_app()"],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                if requests.get(f"{base}/api/health/ready", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise SystemExit(f"server not ready; see {tmp / 'gunicorn.log'}")

        accounts = json.loads((ROOT / "database" / "seed_credentials.json").read_text())["users"]
        admin = next(a for a in accounts if a["role"] == "administrator")
        print(f"signing in {max(LEVELS)} users ...", flush=True)
        pool = session_pool(base, accounts, max(LEVELS))
        for s, mix in pool[:len(accounts)]:  # warm every worker's templates and query caches
            for _name, path, _cap in mix:
                s.get(f"{base}{path}", timeout=60)

        levels = []
        for users in LEVELS:
            print(f"{users} concurrent users ...", flush=True)
            result = read_load(base, pool, users)
            print(f"  p95 {result['all_reads']['p95_s']} s, worst {result['worst_endpoint_p95']}, "
                  f"errors {result['errors']}", flush=True)
            levels.append(result)
        print("CSV export: all rows (refused) and the largest allowed ...", flush=True)
        export = export_timing(base, admin)
        print(f"  {export}", flush=True)
        print("30 s uploads, sequential and 4 clients ...", flush=True)
        uploads = upload_load(base, admin)
        print(f"  {uploads}", flush=True)
    finally:
        server.terminate()
        server.wait(timeout=60)

    at_ten = next(level for level in levels if level["users"] == 10)
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": {"cpu_threads": os.cpu_count(), "platform": platform.platform(),
                    "python": platform.python_version()},
        "server": f"gunicorn -w {WORKERS} --threads {THREADS} (the README's multi-user command)",
        "database": {"engine": "SQLite", **seeded,
                     "note": "synthetic rows, location 'scale-test'; no audio files behind them"},
        "pass_line": {"read_p95_s_at_10_users": READ_P95_LIMIT_S, "errors": 0,
                      "csv_export_s": EXPORT_LIMIT_S},
        "reads_per_request": len(READS), "rounds_per_user": ROUNDS,
        "levels": levels,
        "csv_export": export,
        "uploads_30s": uploads,
        "meets_read_line_at_10_users": all(v["p95_s"] <= READ_P95_LIMIT_S
                                           for v in at_ten["per_endpoint"].values())
                                       and at_ten["errors"] == 0,
        "meets_export_line": export["within_limit"],
    }
    (ROOT / "reports" / "scale.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({k: doc[k] for k in ("meets_read_line_at_10_users", "meets_export_line")}))


if __name__ == "__main__":
    main()
