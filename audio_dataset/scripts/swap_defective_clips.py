#!/usr/bin/env python3
"""Move bad originals (under 0.5 s or quieter than -50 dBFS RMS) to data/dataset_overflow/,
drop their manifest rows and report the free slots. Nothing is deleted.

Backfill afterwards:

    python audio_dataset/scripts/acquire_corpus.py
    python data/topup_synthetic.py
    python audio_dataset/build_split.py && python audio_dataset/verify_split.py

    python audio_dataset/scripts/swap_defective_clips.py --check   # report only
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
AD = REPO / "audio_dataset"
OVERFLOW = REPO / "data" / "dataset_overflow"
MIN_DUR, SILENCE_RMS = 0.5, 10 ** (-50.0 / 20.0)

FROZEN = ["audio_id", "filename", "class_label", "source", "source_url",
          "licence", "author", "date_fetched", "duration_sec", "sampling_rate",
          "channels", "recording_environment", "recording_device",
          "approximate_distance", "original_or_augmented", "parent"]


def classify(path: Path) -> str | None:
    """'short' | 'silent' | None."""
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return "silent"
    mono = y.mean(axis=1)
    if (len(mono) / sr) < MIN_DUR:
        return "short"
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    return "silent" if rms <= SILENCE_RMS else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    args = ap.parse_args()

    rows = list(csv.DictReader((AD / "manifest_with_split.csv").open(newline="", encoding="utf-8")))
    fields = list(rows[0].keys())
    bad: dict[str, tuple[dict, str]] = {}
    for r in rows:
        if r["original_or_augmented"] != "original":
            continue
        p = AD / r["filename"]
        if not p.exists():
            continue
        verdict = classify(p)
        if verdict:
            bad[r["audio_id"]] = (r, verdict)
    print(f"defective originals: {len(bad)}")
    print("  by verdict:", dict(Counter(v for _, v in bad.values())))
    print("  by class:", dict(Counter(r["class_label"] for r, _ in bad.values())))
    print("  by split:", dict(Counter(r["dataset_split"] for r, _ in bad.values())))

    if args.check:
        return 0

    freed = Counter()
    for aid, (r, verdict) in sorted(bad.items()):
        src = AD / r["filename"]
        dst = OVERFLOW / f"{aid}.wav"
        OVERFLOW.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dst))
        freed[r["class_label"]] += 1

    kept = [r for r in rows if r["audio_id"] not in bad]
    for name in ("manifest_with_split.csv", "manifest.csv"):
        p = AD / name
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(kept)
    print(f"removed {len(bad)} rows; kept {len(kept)}")
    print("slots freed per class:", dict(freed))
    (REPO / "data" / "swap_defective_report.json").write_text(
        json.dumps({"removed": sorted(bad), "freed": dict(freed)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
