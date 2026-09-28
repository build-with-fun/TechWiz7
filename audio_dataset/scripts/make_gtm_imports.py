"""Build Teachable Machine audio imports from the training split.

TM's audio uploader only accepts the ZIP format its own "Download Samples" button makes:
``samples.json`` with browser-FFT frames plus a WebM per sample (used only for playback).

Each training recording goes through the app's preprocessing, is resampled to 44.1 kHz,
and gives its loudest one-second window (same rule as inference). ``--windows-per-clip``
adds more non-overlapping windows. ``--max-per-class`` picks recordings in hash order,
not id order, because low ids all come from FSD50K.

Earlier runs: v1 used 32 clips per class (0.28 test accuracy); v2 took 140 per class in
id order (0.46), mostly FSD50K. This version spreads the cap across sources.

Outputs:

* ``gtm_model/upload_package/tm_imports/<class>.zip`` and ``index.json`` (lineage);
* ``audio_dataset/gtm_samples/<class>/<parent>S<k>.wav``: the exact one-second windows
  TM trains on (44.1 kHz), so anyone can listen to them;
* ``audio_dataset/manifests/gtm_segment_rows.csv``: one row per window with its parent
  recording, class and offset. Offsets are in the preprocessed (trimmed) signal.

Only training recordings are used; every parent is checked against the split first.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python_models.preprocess_cache import load_cached, read_index  # noqa: E402
from src.inference.gtm_predictor import (  # noqa: E402
    GtmFrontendConfig,
    compute_spectrogram,
    window_starts,
)

PACKAGE = ROOT / "gtm_model" / "upload_package"
OUTPUT = PACKAGE / "tm_imports"
MANIFEST = ROOT / "audio_dataset" / "manifest_with_split.csv"
SAMPLES = ROOT / "audio_dataset" / "gtm_samples"
ROWS = ROOT / "audio_dataset" / "manifests" / "gtm_segment_rows.csv"
ROW_FIELDS = [
    "audio_id", "filename", "class_label", "source", "source_url", "licence", "author",
    "date_fetched", "duration_sec", "sampling_rate", "channels", "recording_environment",
    "recording_device", "approximate_distance", "original_or_augmented", "parent_audio_id",
    "segment_start_sec", "segment_end_sec", "sha256", "dataset_split", "gtm_batch", "notes",
]
BATCH = "tm-v3"
SLUG = {
    "Machinery Fault": "machinery_fault", "Glass Breaking": "glass_breaking",
    "Alarm or Siren": "alarm_or_siren", "Vehicle Horn": "vehicle_horn",
    "Animal Sound": "animal_sound", "Gunshot": "gunshot", "Panic Scream": "panic_scream",
    "Aggression": "aggression", "Person Asking for Help": "person_asking_for_help",
    "Background Noise": "background_noise",
}


def ranked_windows(samples: np.ndarray, rate: int, config: GtmFrontendConfig,
                   count: int) -> list[tuple[float, np.ndarray]]:
    """``(start_sec, window)`` for the loudest second, then the next loudest non-overlapping ones."""
    y = np.asarray(samples, dtype="float32")
    if rate != config.sample_rate:
        import librosa

        y = librosa.resample(y, orig_sr=rate, target_sr=config.sample_rate).astype("float32")
    n = int(round(config.window_sec * config.sample_rate))
    chosen: list[int] = []
    for start in window_starts(y, n):
        if len(chosen) == count:
            break
        if all(abs(start - c) >= n for c in chosen):
            chosen.append(start)
    return [(s / config.sample_rate, y[s:s + n]) for s in chosen]


def to_webm(window: np.ndarray, rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, window, rate, format="WAV", subtype="PCM_16")
    return subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-c:a", "libopus", "-b:a", "48k", "-f", "webm", "pipe:1"],
        input=buffer.getvalue(), capture_output=True, check=True,
    ).stdout


def hash_order(audio_ids: list[str]) -> list[str]:
    return sorted(audio_ids, key=lambda a: hashlib.sha256(f"tm-v3:{a}".encode()).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only Teachable Machine imports")
    parser.add_argument("--windows-per-clip", type=int, default=1)
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="cap samples per class (0 = no cap)")
    parser.add_argument("--out", default=str(OUTPUT),
                        help="where to write the ZIPs and index.json")
    parser.add_argument("--no-evidence", action="store_true",
                        help="skip the WAV copies and gtm_segment_rows.csv (for trial runs)")
    args = parser.parse_args()
    output = Path(args.out).resolve()

    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        manifest = {r["audio_id"]: r for r in csv.DictReader(fh)}
    cache = read_index()
    frontend = GtmFrontendConfig.load(ROOT / "gtm_model" / "frontend_config.json")
    # TM stores raw dB frames and normalises them itself.
    raw_frontend = dataclasses.replace(frontend, normalize_mode="none")

    by_class: dict[str, list[str]] = defaultdict(list)
    for audio_id, row in manifest.items():
        if row["dataset_split"] == "train" and row["original_or_augmented"] == "original":
            by_class[row["class_label"]].append(audio_id)
    for label, parents in by_class.items():
        by_class[label] = hash_order(parents)
        bad = [p for p in parents if manifest[p]["dataset_split"] != "train"]
        if bad:
            raise ValueError(f"{label}: non-training parents {bad[:3]}")

    output.mkdir(parents=True, exist_ok=True)
    index, rows = [], []
    today = date.today().isoformat()
    for label, parents in sorted(by_class.items()):
        samples, lineage = [], []
        target = output / f"{SLUG[label]}.zip"
        wav_dir = SAMPLES / SLUG[label]
        if not args.no_evidence:
            wav_dir.mkdir(parents=True, exist_ok=True)
            for stale in wav_dir.glob("*.wav"):  # only this script writes here
                stale.unlink()
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            for parent in parents:
                if args.max_per_class and len(samples) >= args.max_per_class:
                    break
                info = cache.get(parent)
                if not info or info["rejected"] == "True":
                    continue  # the app would reject it too
                wave = load_cached(parent)
                windows = ranked_windows(wave, int(info["sample_rate"]), frontend, args.windows_per_clip)
                for k, (start_sec, window) in enumerate(windows, 1):
                    frames = compute_spectrogram(window, raw_frontend)
                    name = f"sample-{len(samples) + 1}.webm"
                    archive.writestr(name, to_webm(window, frontend.sample_rate))
                    # One list of 232 values per frame (43 frames). A flat list imports fine
                    # but hangs TM at "Preparing training data".
                    samples.append({"frequencyFrames": frames.astype(float).tolist(),
                                    "blob": None, "startTime": 0, "endTime": 1.0,
                                    "recordingDuration": 1.0, "blobFilePath": name})
                    segment_id = f"{parent}S{k}"
                    lineage.append({"sample": name, "segment_id": segment_id,
                                    "parent_audio_id": parent, "window": k,
                                    "start_sec_in_preprocessed": round(start_sec, 3)})
                    if args.no_evidence:
                        continue
                    wav_path = wav_dir / f"{segment_id}.wav"
                    sf.write(wav_path, window, frontend.sample_rate, subtype="PCM_16")
                    src = manifest[parent]
                    rows.append({
                        "audio_id": segment_id,
                        "filename": str(wav_path.relative_to(ROOT / "audio_dataset")),
                        "class_label": label,
                        "source": f"derived:TM window of {parent}",
                        "source_url": src.get("source_url", ""), "licence": src.get("licence", ""),
                        "author": src.get("author", ""), "date_fetched": today,
                        "duration_sec": f"{len(window) / frontend.sample_rate:.3f}",
                        "sampling_rate": str(frontend.sample_rate), "channels": "1",
                        "recording_environment": src.get("recording_environment", ""),
                        "recording_device": src.get("recording_device", ""),
                        "approximate_distance": src.get("approximate_distance", ""),
                        "original_or_augmented": "augmented", "parent_audio_id": parent,
                        "segment_start_sec": f"{start_sec:.3f}",
                        "segment_end_sec": f"{start_sec + frontend.window_sec:.3f}",
                        "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                        "dataset_split": "train", "gtm_batch": BATCH,
                        "notes": "offsets are in the preprocessed (trimmed) signal; split inherited from the parent",
                    })
            archive.writestr("samples.json", json.dumps(samples, separators=(",", ":")))
        index.append({"class": label, "zip": str(target.relative_to(ROOT)),
                      "sample_count": len(samples),
                      "distinct_parents": len({x["parent_audio_id"] for x in lineage}),
                      "windows_per_clip": args.windows_per_clip,
                      "parent_order": "sha256('tm-v3:' + audio_id)",
                      "samples": lineage})
        print(f"{label}: {len(samples)} samples from "
              f"{index[-1]['distinct_parents']} training recordings -> {target.name}",
              flush=True)

    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    if args.no_evidence:
        return
    with ROWS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} evidence rows -> {ROWS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
