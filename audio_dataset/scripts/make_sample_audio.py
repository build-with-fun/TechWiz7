#!/usr/bin/env python3
"""Regenerate sample_audio/ (SRS deliverable: permitted sample audio).

One test-split clip per class + failure-path clips (silence, quiet, clipped, invalid).
Test-split provenance means no sample was ever seen in training.
"""
from __future__ import annotations
import csv, shutil, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "audio_dataset" / "manifest_with_split.csv"
OUT = REPO / "sample_audio"


def main() -> int:
    import csv
    rows = list(csv.DictReader(MANIFEST.open()))
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["dataset_split"] == "test":
            by_class[r["class_label"]].append(r)
    if len(by_class) != 10:
        raise SystemExit(f"expected 10 classes in the test split, got {len(by_class)}")

    OUT.mkdir(exist_ok=True)
    for cls, rs in sorted(by_class.items()):
        r = sorted(rs, key=lambda x: x["audio_id"])[0]
        shutil.copy2(REPO / "audio_dataset" / r["filename"], OUT / (cls.replace(" ", "_").lower() + ".wav"))

    sr = 16000
    t = np.arange(3 * sr) / sr
    sf.write(OUT / "silence.wav", np.zeros(3 * sr, dtype="float32"), sr, subtype="PCM_16")
    sf.write(OUT / "low_quality_quiet_tone.wav",
             (0.0008 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr, subtype="PCM_16")
    sf.write(OUT / "clipped_loud_tone.wav",
             (0.99 * np.sign(np.sin(2 * np.pi * 150 * t))).astype("float32"), sr, subtype="PCM_16")
    (OUT / "invalid.notaudio").write_bytes(b"this is not an audio file")
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
