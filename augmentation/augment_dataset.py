"""Write augmented copies of TRAINING recordings, with lineage, never counted as originals.

    python -m augmentation.augment_dataset --copies 2          # ~4,200 files, ~10 min

Rules (SRS "Hint" section and FR xix):

* only rows with ``dataset_split == train`` are read, and each copy inherits that split;
* every copy gets its own ID ``<parent>-A<n>``, ``original_or_augmented=augmented`` and
  ``parent_audio_id=<parent>``, so the manifest can never count it as an original;
* the recipe name and the integer seed are recorded, so any single copy can be rebuilt
  bit-for-bit with ``regenerate(row)``.

Copies are written as 16 kHz PCM WAV under ``audio_dataset/augmented/`` (ignored by Git
like the originals) and listed in ``audio_dataset/manifests/augmented_rows.csv``.
The noise recipe mixes in a real Background Noise *training* clip when one is available,
because synthetic pink noise alone is easier to ignore than a real street or HVAC bed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from augmentation.transforms import TRAINING_RECIPES, add_noise  # noqa: E402

MANIFEST = ROOT / "audio_dataset" / "manifest_with_split.csv"
OUT_DIR = ROOT / "audio_dataset" / "augmented"
ROWS = ROOT / "audio_dataset" / "manifests" / "augmented_rows.csv"
RATE = 16000
FIELDS = ["audio_id", "filename", "class_label", "parent_audio_id", "dataset_split",
          "original_or_augmented", "recipe", "seed", "duration_sec", "sampling_rate",
          "channels", "sha256", "source", "licence"]


def load_mono(path: Path, rate: int = RATE) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=rate, mono=True)
    return y.astype(np.float32)


def seed_for(parent: str, copy: int) -> int:
    # Derived from the ID rather than a running counter, so adding or removing rows
    # elsewhere in the manifest does not change any existing copy.
    return int(hashlib.sha256(f"{parent}:{copy}".encode()).hexdigest()[:8], 16)


def augment_one(y: np.ndarray, recipe: str, seed: int,
                noise_bank: list[np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if recipe == "noise" and noise_bank:
        bed = noise_bank[int(rng.integers(0, len(noise_bank)))]
        return add_noise(y, RATE, rng, snr_db=float(rng.uniform(8, 20)), noise=bed)
    return TRAINING_RECIPES[recipe](y, RATE, rng)


def regenerate(row: dict[str, str]) -> np.ndarray:
    """Rebuild one augmented copy from its manifest row (used by the tests)."""
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        originals = {r["audio_id"]: r for r in csv.DictReader(fh)}
    parent = originals[row["parent_audio_id"]]
    y = load_mono(ROOT / "audio_dataset" / parent["filename"])
    return augment_one(y, row["recipe"], int(row["seed"]), _noise_bank(originals))


def _noise_bank(originals: dict[str, dict[str, str]], limit: int = 40) -> list[np.ndarray]:
    ids = sorted(a for a, r in originals.items()
                 if r["class_label"] == "Background Noise" and r["dataset_split"] == "train")
    bank = []
    for audio_id in ids[:limit]:
        path = ROOT / "audio_dataset" / originals[audio_id]["filename"]
        if path.exists():
            bank.append(load_mono(path))
    return bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment training recordings only")
    parser.add_argument("--copies", type=int, default=2, help="copies per training clip")
    args = parser.parse_args()

    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        originals = {r["audio_id"]: r for r in csv.DictReader(fh)}
    train = [r for r in originals.values()
             if r["dataset_split"] == "train" and r["original_or_augmented"] == "original"]
    recipes = sorted(TRAINING_RECIPES)
    bank = _noise_bank(originals)
    written = []
    for n, row in enumerate(sorted(train, key=lambda r: r["audio_id"]), 1):
        y = load_mono(ROOT / "audio_dataset" / row["filename"])
        for copy in range(1, args.copies + 1):
            seed = seed_for(row["audio_id"], copy)
            recipe = recipes[seed % len(recipes)]
            # Background Noise copies with added noise are still background noise, but
            # a noise-on-noise copy teaches nothing, so give those a different recipe.
            if recipe == "noise" and row["class_label"] == "Background Noise":
                recipe = "reverb"
            out = augment_one(y, recipe, seed, bank)
            audio_id = f"{row['audio_id']}-A{copy}"
            rel = Path("augmented") / row["class_label"].lower().replace(" ", "_") / f"{audio_id}.wav"
            path = ROOT / "audio_dataset" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, out, RATE, subtype="PCM_16")
            written.append({
                "audio_id": audio_id, "filename": str(rel), "class_label": row["class_label"],
                "parent_audio_id": row["audio_id"], "dataset_split": "train",
                "original_or_augmented": "augmented", "recipe": recipe, "seed": seed,
                "duration_sec": round(out.size / RATE, 3), "sampling_rate": RATE,
                "channels": 1, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source": f"augmentation of {row['audio_id']}", "licence": row["licence"],
            })
        if n % 300 == 0:
            print(f"  {n}/{len(train)} training clips augmented", flush=True)

    ROWS.parent.mkdir(parents=True, exist_ok=True)
    with ROWS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(written)
    print(f"wrote {len(written)} augmented rows to {ROWS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
