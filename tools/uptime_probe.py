"""NFR 5 availability probe: poll the readiness endpoint and record uptime.

    .venv/bin/python tools/uptime_probe.py https://<owner>-<space>.hf.space
    .venv/bin/python tools/uptime_probe.py http://127.0.0.1:5055 --interval 60 --hours 8
    .venv/bin/python tools/uptime_probe.py --summary      # summarise what was recorded

Every ``--interval`` seconds it requests ``/api/health/ready`` and appends one line to
``reports/uptime.jsonl``: time, HTTP status, response time and whether the app was ready
(database up and both models loaded). A probe counts as up only when the status is 200
within ``--timeout`` seconds. ``--summary`` prints and writes ``reports/uptime_summary.json``
with the uptime percentage against the SRS target of 99 %. Stop it with Ctrl+C; runs can be
resumed and are summarised together.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "reports" / "uptime.jsonl"
SUMMARY = ROOT / "reports" / "uptime_summary.json"
TARGET = 0.99


def probe(base: str, timeout: float) -> dict:
    started = time.perf_counter()
    row = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "base": base}
    try:
        r = requests.get(f"{base}/api/health/ready", timeout=timeout)
        row["status"] = r.status_code
        row["up"] = r.status_code == 200
    except requests.RequestException as exc:
        row["status"] = None
        row["up"] = False
        row["error"] = type(exc).__name__
    row["seconds"] = round(time.perf_counter() - started, 3)
    return row


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"probes": 0}
    up = sum(1 for r in rows if r["up"])
    outages, current = [], None
    for r in rows:
        if not r["up"] and current is None:
            current = {"from": r["at"], "probes": 0}
        if not r["up"]:
            current["probes"] += 1
            current["to"] = r["at"]
        elif current is not None:
            outages.append(current)
            current = None
    if current is not None:
        outages.append(current)
    uptime = up / len(rows)
    return {"probes": len(rows), "up": up, "uptime": round(uptime, 5),
            "target": TARGET, "meets_target": uptime >= TARGET,
            "first": rows[0]["at"], "last": rows[-1]["at"],
            "bases": sorted({r["base"] for r in rows}), "outages": outages}


def read_log() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", nargs="?", default="")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--hours", type=float, default=0.0, help="stop after this long (0 = until Ctrl+C)")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if not args.summary:
        if not args.base:
            parser.error("give the app's base URL, or --summary")
        LOG.parent.mkdir(parents=True, exist_ok=True)
        stop_at = time.time() + args.hours * 3600 if args.hours else float("inf")
        try:
            while time.time() < stop_at:
                row = probe(args.base.rstrip("/"), args.timeout)
                with LOG.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"{row['at']}  {'UP  ' if row['up'] else 'DOWN'}  {row['status']}  {row['seconds']} s",
                      flush=True)
                time.sleep(max(0.0, args.interval - row["seconds"]))
        except KeyboardInterrupt:
            pass

    doc = summarise(read_log())
    SUMMARY.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in doc.items() if k != "outages"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
