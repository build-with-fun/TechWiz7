"""NFR 1 latency benchmark over HTTP with both real models.

    python tools/benchmark_latency.py            # ~2 min; writes reports/performance.json

Targets (SRS 1.7): a clip of up to 30 s within 8 s, a live window within 3 s. Measures:

* 30 s uploads one at a time, after a warm-up (the first, cold request is reported separately)
* 2 s live windows, one after another
* the same 30 s upload from 4 clients at once

Uses a temporary database. Times are per HTTP request, including decoding, both models,
the rules and the database write.
"""

from __future__ import annotations

import base64
import io
import json
import os
import platform
import statistics
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def wav_bytes(samples: np.ndarray, rate: int = 16000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return out.getvalue()


def test_audio(seconds: float) -> np.ndarray:
    """Test-split recordings joined up to the requested length."""
    import csv

    import librosa

    with (ROOT / "audio_dataset" / "manifest_with_split.csv").open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["dataset_split"] == "test"]
    parts, total = [], 0
    for r in sorted(rows, key=lambda r: r["audio_id"])[::15]:
        y, _ = librosa.load(str(ROOT / "audio_dataset" / r["filename"]), sr=16000, mono=True)
        parts.append(y)
        total += y.size
        if total >= seconds * 16000:
            break
    return np.concatenate(parts)[: int(seconds * 16000)].astype(np.float32)


def summary(values: list[float]) -> dict:
    values = sorted(values)
    return {"n": len(values), "median_s": round(statistics.median(values), 3),
            "p95_s": round(values[max(0, int(0.95 * len(values)) - 1)], 3),
            "max_s": round(values[-1], 3)}


def main() -> None:
    from src.app import create_app
    from src.auth import hash_password
    from src.models import User

    tmp = Path(tempfile.mkdtemp(prefix="sst-bench-"))
    app = create_app(SST_DB_PATH=str(tmp / "bench.db"), SST_STORAGE_DIR=str(tmp / "storage"),
                     SST_CSRF_ENABLED=False)
    with app.config["SST_SESSION_FACTORY"]() as session:
        session.add(User(username="bench", email="bench@example.test", display_name="bench",
                         password_hash=hash_password("BenchPass123!"), role="administrator"))
        session.commit()
        uid = session.query(User).filter_by(username="bench").one().id

    def client():
        c = app.test_client()
        with app.test_request_context("/"):
            with c.session_transaction() as cookie:
                cookie["_user_id"] = str(uid)
                cookie["_fresh"] = True
                cookie["_id"] = app.login_manager._session_identifier_generator()
        return c

    rng = np.random.default_rng(0)
    base = test_audio(30.0)

    def unique_clip() -> bytes:
        # Tiny random gain so the duplicate check doesn't skip the analysis.
        return wav_bytes(base * rng.uniform(0.8, 0.95))

    def upload(c) -> float:
        started = time.perf_counter()
        r = c.post("/api/audio/upload", data={"file": (io.BytesIO(unique_clip()), "bench.wav")},
                   content_type="multipart/form-data")
        elapsed = time.perf_counter() - started
        if r.status_code != 201:
            raise SystemExit(f"upload failed: {r.status_code} {r.get_json()}")
        return elapsed

    c = client()
    cold = upload(c)
    uploads = [upload(c) for _ in range(6)]

    session_id = c.post("/api/live/sessions", json={"consent_ack": True}).get_json()["data"]["id"]
    windows = []
    for seq in range(1, 16):
        start = int(rng.integers(0, base.size - 32000))
        payload = {"seq": seq, "sample_rate": 16000, "duration_sec": 2.0,
                   "audio_b64": base64.b64encode(wav_bytes(base[start:start + 32000])).decode()}
        t0 = time.perf_counter()
        r = c.post(f"/api/live/sessions/{session_id}/windows", json=payload)
        windows.append(time.perf_counter() - t0)
        if r.status_code != 200:
            raise SystemExit(f"live window failed: {r.status_code} {r.get_json()}")

    concurrent: list[float] = []
    lock = threading.Lock()

    def worker():
        cc = client()
        for _ in range(2):
            t = upload(cc)
            with lock:
                concurrent.append(t)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    wall = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall

    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": {"cpu_threads": os.cpu_count(), "platform": platform.platform(),
                    "python": platform.python_version()},
        "targets": {"upload_30s_clip_s": 8.0, "live_window_s": 3.0},
        "cold_start_first_upload_s": round(cold, 3),
        "upload_30s_sequential": summary(uploads),
        "live_window_2s_sequential": summary(windows),
        "upload_30s_4_clients": {**summary(concurrent), "wall_clock_s": round(wall, 2),
                                 "uploads": len(concurrent)},
    }
    doc["meets_upload_target"] = doc["upload_30s_sequential"]["max_s"] <= 8.0
    doc["meets_live_target"] = doc["live_window_2s_sequential"]["max_s"] <= 3.0
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "performance.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
