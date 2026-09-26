"""Measure the two-stage near-duplicate check (FR lxxiv) on real recordings.

    python tools/evaluate_near_duplicates.py      # ~3 min; writes reports/near_duplicates.json

Positives: each chosen test recording is re-encoded to 64 kbit/s MP3, turned down 12 dB,
trimmed to its middle 70 %, and mixed with pink noise at 20 dB SNR; each copy should be
recognised as a near-duplicate of its source.
Hard negatives: for the same recordings, the three clips from OTHER source recordings
whose perceptual fingerprints are most similar. These are the clips most likely to be
falsely flagged, so they are the fair test of a threshold.

Both stages run on the output of the serving preprocessor, as they do in the app.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audio_dataset"))

from audio_preprocessing.pipeline import AudioPipeline  # noqa: E402
from augmentation.transforms import add_noise  # noqa: E402
from build_split import source_group  # noqa: E402
from python_models.preprocess_cache import load_cached  # noqa: E402
from src.services.pipeline import (audio_fingerprint, fingerprint_similarity,  # noqa: E402
                                   spectral_match)

PER_CLASS = 6
RATE = 16000


def mp3(y: np.ndarray) -> np.ndarray:
    import librosa
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        wav, out = Path(tmp) / "a.wav", Path(tmp) / "a.mp3"
        sf.write(wav, y, RATE, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
                        "-b:a", "64k", str(out)], check=True)
        return librosa.load(str(out), sr=RATE, mono=True)[0]


def variants(y: np.ndarray, rng) -> dict[str, np.ndarray]:
    n = y.size
    return {
        "mp3_64k": mp3(y),
        "minus_12dB": (y * 10 ** (-12 / 20)).astype(np.float32),
        "trim_middle_70pct": y[int(n * 0.15): int(n * 0.85)],
        "pink_noise_20dB": add_noise(y, RATE, rng, snr_db=20.0),
    }


def main() -> None:
    import librosa

    with (ROOT / "audio_dataset" / "manifest_with_split.csv").open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["original_or_augmented"] == "original"]
    by_id = {r["audio_id"]: r for r in rows}
    pipeline = AudioPipeline()
    prints = {r["audio_id"]: audio_fingerprint(load_cached(r["audio_id"]), RATE) for r in rows}

    chosen, seen = [], {}
    for r in sorted(rows, key=lambda r: r["audio_id"]):
        if r["dataset_split"] == "test" and seen.get(r["class_label"], 0) < PER_CLASS:
            seen[r["class_label"]] = seen.get(r["class_label"], 0) + 1
            chosen.append(r)

    rng = np.random.default_rng(3)
    positives, negatives = [], []
    for r in chosen:
        source = load_cached(r["audio_id"])
        raw, _ = librosa.load(str(ROOT / "audio_dataset" / r["filename"]), sr=RATE, mono=True)
        for name, copy in variants(raw, rng).items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pre = pipeline.preprocess_samples(copy, RATE, origin="upload")
            if pre.rejected:
                positives.append({"audio_id": r["audio_id"], "variant": name, "rejected": True})
                continue
            copy_print = audio_fingerprint(pre.samples, RATE)
            stage1 = fingerprint_similarity(prints[r["audio_id"]], copy_print)
            # Where the true source ranks among all 3,000 stored clips by fingerprint
            # similarity: the app only runs stage 2 on the top SHORTLIST candidates.
            others = [fingerprint_similarity(copy_print, fp) or 0.0
                      for other, fp in prints.items() if other != r["audio_id"] and fp]
            rank = 1 + sum(o > (stage1 or 0.0) for o in others)
            positives.append({"audio_id": r["audio_id"], "variant": name, "stage1": stage1,
                              "rank": rank,
                              "stage2": spectral_match(pre.samples, RATE, source, RATE)})
        group = source_group(r)
        others = sorted(((fingerprint_similarity(prints[r["audio_id"]], fp) or 0.0, other)
                         for other, fp in prints.items()
                         if fp and source_group(by_id[other]) != group), reverse=True)[:3]
        for stage1, other in others:
            negatives.append({"audio_id": r["audio_id"], "other": other, "stage1": stage1,
                              "same_class": by_id[other]["class_label"] == r["class_label"],
                              "stage2": spectral_match(source, RATE, load_cached(other), RATE)})

    stage1_threshold = 0.92
    table = []
    for t in (0.75, 0.8, 0.85, 0.9, 0.95):
        pos_ok = [p for p in positives if not p.get("rejected")]
        tp = sum(1 for p in pos_ok if (p["stage1"] or 0) >= stage1_threshold and p["stage2"] >= t)
        fp = sum(1 for n in negatives if n["stage1"] >= stage1_threshold and n["stage2"] >= t)
        table.append({"stage2_threshold": t, "recall": tp / len(pos_ok),
                      "hard_negative_false_positives": fp, "hard_negatives": len(negatives)})
    shortlist = []
    for k in (3, 5, 10):
        pos_ok = [p for p in positives if not p.get("rejected")]
        shortlist.append({"top_k": k, "stage2_threshold": 0.9,
                          "recall": sum(p["rank"] <= k and p["stage2"] >= 0.9 for p in pos_ok)
                          / len(pos_ok),
                          "hard_negative_false_positives": sum(n["stage2"] >= 0.9 for n in negatives)})
    by_variant = {}
    for p in positives:
        if not p.get("rejected"):
            by_variant.setdefault(p["variant"], []).append(p)
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recordings": len(chosen), "positives": len(positives), "hard_negatives": len(negatives),
        "stage1_only_false_positive_rate": sum(n["stage1"] >= stage1_threshold for n in negatives)
        / len(negatives),
        "stage1_recall": {k: sum((p["stage1"] or 0) >= stage1_threshold for p in v) / len(v)
                          for k, v in by_variant.items()},
        "stage1_missing_by_variant": {k: sum(p["stage1"] is None for p in v) for k, v in by_variant.items()},
        "stage2_median_by_variant": {k: float(np.median([p["stage2"] for p in v]))
                                     for k, v in by_variant.items()},
        "stage2_hard_negative_max": max(n["stage2"] for n in negatives),
        "stage2_hard_negative_p95": float(np.percentile([n["stage2"] for n in negatives], 95)),
        "thresholds": table, "shortlist_policy": shortlist, "positives_detail": positives, "negatives_detail": negatives,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "near_duplicates.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if not k.endswith("detail")}, indent=2))


if __name__ == "__main__":
    main()
