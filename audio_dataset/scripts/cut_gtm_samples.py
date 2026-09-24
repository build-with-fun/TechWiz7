#!/usr/bin/env python3
"""
SonicSentinel AI -- Google Teachable Machine sample cutter.

Owner: omar (GTM evidence).  Authority: SRS Step 5, deliverable 5, integrity rules.

WHY THIS EXISTS
---------------
SRS Step 5: the Python model and the Teachable Machine model must be trained on the
SAME underlying recordings. GTM cannot take a 7-second field recording; it takes
short samples. So this cuts short mono samples OUT of the exact recordings the
Python model trains on, and gives every sample a manifest row of its own.

WHAT IT GUARANTEES
------------------
1. **Train only.** A sample is cut only from a parent the frozen split
   (`data/splits/split.json`, owner: lorena) has assigned to `train`. If a parent is
   in val or test this script ABORTS. This is the difference between an honest
   accuracy number and a fantasy one.
2. **Lineage.** Every sample carries `parent_audio_id`, `segment_start_sec`,
   `segment_end_sec`, so `build_split.py` classifies it as a `segment` and forces it
   into its parent's split -- validation and test material therefore can never
   appear in the GTM training set even by accident.
3. **Same Audio ID lineage.** The sample id is `<parent_audio_id>S<n>`, so a reviewer
   can go from a GTM sample back to the exact original recording and its licence.
4. **Deterministic.** Sample positions are fixed fractions of the parent duration,
   not random: re-running produces byte-identical output, so the GTM upload can be
   reproduced.
5. **It does not assign a split.** `dataset_split` is emitted EMPTY, always.

Usage
-----
    .venv/bin/python audio_dataset/scripts/cut_gtm_samples.py --check
    .venv/bin/python audio_dataset/scripts/cut_gtm_samples.py --segments-per-recording 3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_DATASET = SCRIPT_DIR.parent
REPO_ROOT = AUDIO_DATASET.parent

MANIFEST = AUDIO_DATASET / "manifest.csv"
SPLIT = REPO_ROOT / "data" / "splits" / "split.json"
OUT_DIR = AUDIO_DATASET / "gtm_samples"
OUT_ROWS = AUDIO_DATASET / "manifests" / "gtm_segment_rows.csv"

FROZEN_COLUMNS = [
    "audio_id", "filename", "class_label", "source", "source_url", "licence", "author",
    "date_fetched", "duration_sec", "sampling_rate", "channels", "recording_environment",
    "recording_device", "approximate_distance", "original_or_augmented", "parent_audio_id",
    "segment_start_sec", "segment_end_sec", "sha256", "dataset_split",
]
EXTRA_COLUMNS = ["parent_class_label", "gtm_batch"]

SEGMENT_SEC = 2.0          # every GTM sample is exactly this long when the parent allows
MIN_SEGMENT_SEC = 0.6      # a shorter parent still yields a usable (if short) sample
TARGET_RATE = 16000
TARGET_CHANNELS = 1

# fixed fractions of the parent duration at which samples start (deterministic)
START_FRACTIONS = (0.10, 0.50, 0.85, 0.30, 0.70)


class CutError(Exception):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise CutError(f"ffprobe failed on {path}: {out.stderr.strip()[:200]}")
    return float(json.loads(out.stdout)["format"]["duration"])


def cut(src: Path, start: float, length: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{start:.4f}", "-t", f"{length:.4f}", "-i", str(src),
        "-ac", str(TARGET_CHANNELS), "-ar", str(TARGET_RATE),
        "-c:a", "pcm_s16le", "-sample_fmt", "s16", str(dest),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not dest.exists() or dest.stat().st_size < 512:
        raise CutError(f"ffmpeg failed for {dest.name}: {r.stderr.strip()[:200]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cut GTM training samples from the Python model's train recordings.")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-rows", type=Path, default=OUT_ROWS)
    ap.add_argument("--segments-per-recording", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        raise SystemExit(f"no manifest: {args.manifest} (run the assembler first)")
    if not args.split.exists():
        raise SystemExit(f"no frozen split: {args.split} (run build_split.py first)")

    with args.manifest.open(newline="", encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    by_id = {r["audio_id"]: r for r in rows}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    assign = split["assignments"]

    # ---- select parents: originals in the TRAIN split ------------------------------
    train_originals = [aid for aid, a in assign.items()
                       if a["split"] == "train" and a["role"] == "original"]
    if not train_originals:
        raise SystemExit("no originals assigned to train -- is the split built?")

    # hard guarantee: nothing here may be val/test
    illegal = [aid for aid in train_originals if assign[aid]["split"] != "train"]
    if illegal:  # defensive; cannot happen given the filter above, but never trust it
        raise CutError(f"refusing: {len(illegal)} parents are not in train")

    by_class: dict[str, list[str]] = defaultdict(list)
    for aid in train_originals:
        by_class[assign[aid]["class_label"]].append(aid)

    print(f"train originals: {len(train_originals)} across {len(by_class)} classes")
    print(f"cutting up to {args.segments_per_recording} samples per recording "
          f"({SEGMENT_SEC}s, {TARGET_RATE} Hz, mono)")

    out_rows: list[dict] = []
    today = date.today().isoformat()
    failures: list[str] = []
    per_class_count: Counter = Counter()
    n_fracs = min(args.segments_per_recording, len(START_FRACTIONS))

    for cls in sorted(by_class):
        slug = cls.lower().replace(" ", "_")
        for aid in sorted(by_class[cls]):
            rec = by_id.get(aid)
            if rec is None:
                failures.append(f"{aid}: not in manifest")
                continue
            src = AUDIO_DATASET / rec["filename"]
            if not src.exists():
                failures.append(f"{aid}: source file missing ({rec['filename']})")
                continue
            dur = probe(src)
            length = SEGMENT_SEC if dur >= SEGMENT_SEC + 0.3 else max(MIN_SEGMENT_SEC, dur * 0.8)
            starts: list[float] = []
            for frac in START_FRACTIONS[:n_fracs]:
                s = round(max(0.0, min(frac * dur, dur - length)), 3)
                if s not in starts:
                    starts.append(s)
            if not starts:
                starts = [0.0]

            for n, start in enumerate(starts, start=1):
                sid = f"{aid}S{n}"
                dest = args.out_dir / slug / f"{sid}.wav"
                if not args.check:
                    try:
                        cut(src, start, length, dest)
                    except CutError as exc:
                        failures.append(f"{sid}: {exc}")
                        continue
                end = round(start + length, 3)
                row = {c: "" for c in FROZEN_COLUMNS + EXTRA_COLUMNS}
                row.update({
                    "audio_id": sid,
                    "filename": str(dest.relative_to(AUDIO_DATASET)),
                    "class_label": cls,
                    "source": f"derived:segment of {aid} (FSD50K/procedural parent)",
                    "source_url": rec.get("source_url", ""),
                    "licence": rec.get("licence", ""),
                    "author": rec.get("author", ""),
                    "date_fetched": today,
                    "duration_sec": f"{length:.3f}",
                    "sampling_rate": str(TARGET_RATE),
                    "channels": str(TARGET_CHANNELS),
                    "recording_environment": rec.get("recording_environment", ""),
                    "recording_device": rec.get("recording_device", ""),
                    "approximate_distance": rec.get("approximate_distance", ""),
                    "original_or_augmented": "augmented",
                    "parent_audio_id": aid,
                    "segment_start_sec": f"{start:.3f}",
                    "segment_end_sec": f"{end:.3f}",
                    "sha256": "" if args.check else sha256_of(dest),
                    "dataset_split": "",          # build_split.py owns this. never here.
                    "parent_class_label": cls,
                    "gtm_batch": "gtm_train_v1",
                })
                out_rows.append(row)
            per_class_count[cls] += len(starts)

    print(f"\ncut {len(out_rows)} samples")
    for cls in sorted(per_class_count):
        print(f"  {cls:<24} {per_class_count[cls]:>4}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for f in failures[:10]:
            print(f"  {f}")

    if args.check:
        print("\n--check: nothing written.")
        return 0

    args.out_rows.parent.mkdir(parents=True, exist_ok=True)
    with args.out_rows.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FROZEN_COLUMNS + EXTRA_COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {args.out_rows}")
    print(f"samples under {args.out_dir}")
    print("\nNext: re-run assemble_manifest.py then build_split.py --strict so the samples "
          "inherit their parents' splits, then verify_dataset.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
