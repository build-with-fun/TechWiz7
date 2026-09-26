#!/usr/bin/env python3
"""Regenerate sample_audio/ (SRS deliverable: permitted sample audio).

One test-split clip per class + failure-path clips (silence, quiet, clipped, invalid).
Test-split provenance means no sample was ever seen in training.
"""
from __future__ import annotations
import shutil
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
    provenance = []
    for cls, rs in sorted(by_class.items()):
        r = sorted(rs, key=lambda x: x["audio_id"])[0]
        name = cls.replace(" ", "_").lower() + ".wav"
        shutil.copy2(REPO / "audio_dataset" / r["filename"], OUT / name)
        provenance.append({"file": name, "audio_id": r["audio_id"], "class_label": cls,
                           "dataset_split": r["dataset_split"], "source": r["source"],
                           "licence": r["licence"], "author": r["author"],
                           "source_url": r["source_url"]})
    # Record each sample's source, so its licence and test-split origin can be checked.
    with (OUT / "SOURCES.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(provenance[0]))
        writer.writeheader()
        writer.writerows(provenance)

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
