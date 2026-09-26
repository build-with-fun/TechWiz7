"""Run every manifest recording through the serving preprocessor once and keep the result.

Both the transfer-learning trainer and the Teachable Machine import builder read from
this cache. That is the point of it: the waveform a model is trained on comes out of the
same ``AudioPipeline`` object the Flask app calls on an upload, so there is no second,
slightly different copy of resampling / noise reduction / trimming to drift apart.

    python -m python_models.preprocess_cache            # all 3,000 originals, ~7 min
    python -m python_models.preprocess_cache --split test

Output: ``data/cache/pre16k/<audio_id>.npy`` (float32, 16 kHz mono) and
``data/cache/pre16k/index.csv`` with the quality verdict and any rejection reason.
The cache is ignored by Git; delete the folder to rebuild it from scratch.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "audio_dataset" / "manifest_with_split.csv"
CACHE = ROOT / "data" / "cache" / "pre16k"

# Preprocessing variants compared on the validation split (26 Sep). "current" is the
# serving configuration at the time; "gentle" halves the noise gate and trims only what
# is 45 dB below the peak, to keep the decay tail of impulsive sounds.
VARIANTS = {
    "current": {},
    "gentle": {"noise_reduction_strength": 0.4, "silence_trim_top_db": 45.0},
}


def cache_dir(variant: str = "current") -> Path:
    return CACHE if variant == "current" else CACHE.parent / f"pre16k_{variant}"
INDEX_FIELDS = ["audio_id", "class_label", "dataset_split", "n_samples", "sample_rate",
                "quality", "rejected", "reason", "preprocessing_version"]


def load_manifest(split: str | None = None) -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["original_or_augmented"] == "original"]
    return [r for r in rows if split is None or r["dataset_split"] == split]


def cached_path(audio_id: str, variant: str = "current") -> Path:
    return cache_dir(variant) / f"{audio_id}.npy"


def load_cached(audio_id: str, variant: str = "current") -> np.ndarray:
    return np.load(cached_path(audio_id, variant))


def read_index(variant: str = "current") -> dict[str, dict[str, str]]:
    path = cache_dir(variant) / "index.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["audio_id"]: r for r in csv.DictReader(fh)}


def build(split: str | None = None, *, force: bool = False,
          variant: str = "current") -> dict[str, dict[str, str]]:
    from audio_preprocessing import config as cfg_mod
    from audio_preprocessing.pipeline import AudioPipeline

    out = cache_dir(variant)
    out.mkdir(parents=True, exist_ok=True)
    index = read_index(variant)
    pipeline = AudioPipeline(audio_cfg={**cfg_mod.audio_config(), **VARIANTS[variant]})
    rows = load_manifest(split)
    started = time.time()
    for n, row in enumerate(rows, 1):
        audio_id = row["audio_id"]
        if not force and audio_id in index and cached_path(audio_id, variant).exists():
            continue
        path = ROOT / "audio_dataset" / row["filename"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pre = pipeline.preprocess_file(path)
                rejected = bool(getattr(pre, "rejected", False))
                reason = (pre.rejection_reason or "") if rejected else ""
                samples = np.asarray(pre.samples, dtype=np.float32)
                quality = (pre.quality or {}).get("verdict", "")
                version = (pre.preprocessing or {}).get("version", "")
                rate = int(pre.sample_rate)
            except Exception as exc:  # noqa: BLE001 - recorded per clip, run continues
                rejected, reason, samples, quality, version, rate = True, str(exc), None, "Unusable", "", 0
        if samples is not None and samples.size:
            np.save(cached_path(audio_id, variant), samples)
        index[audio_id] = {
            "audio_id": audio_id, "class_label": row["class_label"],
            "dataset_split": row["dataset_split"],
            "n_samples": str(0 if samples is None else samples.size), "sample_rate": str(rate),
            "quality": quality, "rejected": str(rejected), "reason": reason,
            "preprocessing_version": version,
        }
        if n % 250 == 0:
            print(f"  {n}/{len(rows)} clips, {time.time() - started:.0f}s", flush=True)

    with (out / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for audio_id in sorted(index):
            writer.writerow(index[audio_id])
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--force", action="store_true", help="recompute cached clips")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="current")
    args = parser.parse_args()
    index = build(args.split, force=args.force, variant=args.variant)
    rejected = [r for r in index.values() if r["rejected"] == "True"]
    summary = {"clips": len(index), "rejected": len(rejected),
               "rejected_ids": [r["audio_id"] for r in rejected][:50]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
