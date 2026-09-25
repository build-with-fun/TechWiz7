"""Build Teachable Machine audio imports from train-only WAV segments.

Teachable Machine imports its own sample ZIP format (``samples.json`` plus WebM),
not a ZIP of WAV files. Each exported entry keeps the original training-parent ID
in the accompanying index so the frozen validation and test splits remain separate.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "gtm_model/upload_package"
OUTPUT = PACKAGE / "tm_imports"
SAMPLES_PER_CLASS = 32
TARGET_RATE = 44100
FRAMES = 43
BINS = 232
HOP = 1024
FFT = 2048


def spectrum(wav_bytes: bytes) -> tuple[list[list[float]], bytes]:
    audio, rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    from math import gcd

    divisor = gcd(rate, TARGET_RATE)
    y = resample_poly(audio, TARGET_RATE // divisor, rate // divisor).astype(np.float32)
    y = np.pad(y[: TARGET_RATE], (0, max(0, TARGET_RATE - len(y))))
    # WebAudio's analyser uses a 2048-point Blackman window; each 1024-sample
    # hop supplies one frequency column. Values are dB, as in Download Samples.
    window = np.blackman(FFT).astype(np.float32)
    padded = np.pad(y, (FFT - HOP, FFT))
    starts = np.arange(FRAMES) * HOP
    blocks = np.stack([padded[start:start + FFT] for start in starts])
    amplitudes = np.abs(np.fft.rfft(blocks * window, axis=1)[:, :BINS]) / FFT
    db = 20 * np.log10(np.maximum(amplitudes, 1e-16))
    db = np.clip(db, -320, 0).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, y, TARGET_RATE, format="WAV", subtype="PCM_16")
    webm = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-c:a", "libopus", "-b:a", "64k", "-f", "webm", "pipe:1"],
        input=buffer.getvalue(), capture_output=True, check=True,
    ).stdout
    return db.tolist(), webm


def main() -> None:
    with (ROOT / "audio_dataset/manifest.csv").open(newline="", encoding="utf-8") as handle:
        train_ids = {row["audio_id"] for row in csv.DictReader(handle)
                     if row["dataset_split"] == "train"}
    with (ROOT / "audio_dataset/manifests/gtm_segment_rows.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_class: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in sorted(rows, key=lambda value: value["audio_id"]):
        parent = row["parent_audio_id"]
        if parent not in train_ids:
            raise ValueError(f"GTM sample {row['audio_id']} comes from a non-training clip")
        by_class[row["class_label"]].setdefault(parent, row)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    index = []
    with (PACKAGE / "index.csv").open(newline="", encoding="utf-8") as handle:
        packages = list(csv.DictReader(handle))
    for package in packages:
        label = package["class"]
        chosen = list(by_class[label].values())[:SAMPLES_PER_CLASS]
        if len(chosen) != SAMPLES_PER_CLASS:
            raise ValueError(f"Only {len(chosen)} independent parents for {label}")
        samples = []
        source = ROOT / package["zip"]
        target = OUTPUT / source.name
        with zipfile.ZipFile(source) as original, zipfile.ZipFile(
            target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as result:
            for number, row in enumerate(chosen, 1):
                frames, webm = spectrum(original.read(Path(row["filename"]).name))
                webm_name = f"sample-{number}.webm"
                result.writestr(webm_name, webm)
                samples.append({"frequencyFrames": frames, "blob": None,
                                "startTime": 0, "endTime": 1.0,
                                "recordingDuration": 1.0, "blobFilePath": webm_name})
            result.writestr("samples.json", json.dumps(samples, separators=(",", ":")))
        index.append({"class": label, "zip": str(target.relative_to(ROOT)),
                      "sample_count": len(chosen),
                      "segment_ids": [row["audio_id"] for row in chosen],
                      "parent_ids": [row["parent_audio_id"] for row in chosen]})
        print(f"{label}: {len(chosen)} samples -> {target.name}", flush=True)
    (OUTPUT / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
