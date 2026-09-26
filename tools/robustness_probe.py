"""Robustness and confusable-sound probes through the real analysis pipeline.

    python tools/robustness_probe.py                   # 15 test clips per class per condition
    python tools/robustness_probe.py --per-class 45    # every test clip (slow on a laptop)

Part A degrades held-out TEST recordings in the ways SRS 1.8 rule 9 says the hidden
test may: background noise, echo, low volume, a different device, distance, a partial
event, an overlapping second event, and re-encoding. Each condition is scored for both
models. These are probes: they show how the models degrade and are never reported as
test accuracy. The degraded audio is scored and discarded; nothing is written back into
the dataset.

Part B plays ESC-50 categories that are NOT in our training data but are easy to confuse
with a critical class: fireworks, door knocks and clapping (against Gunshot), laughing
and a crying baby (against Panic Scream), and church bells (against Alarm). There is no
correct class for these. What matters is whether the product raises a confident critical
alert or sends the clip to a person.

Outputs: reports/robustness.json and reports/ROBUSTNESS.md.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from augmentation import transforms as T  # noqa: E402

RATE = 16000
SEED = 7


def mp3_roundtrip(y: np.ndarray, sr: int, rng) -> np.ndarray:
    """Re-encode through 64 kbit/s MP3 and decode again, the way a phone upload might."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        wav, mp3 = Path(tmp) / "a.wav", Path(tmp) / "a.mp3"
        sf.write(wav, y, sr, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
                        "-b:a", "64k", str(mp3)], check=True)
        import librosa

        out, _ = librosa.load(str(mp3), sr=sr, mono=True)
    return out.astype(np.float32)


def conditions(noise_bank: list[np.ndarray], overlap_bank: dict[str, list[np.ndarray]]):
    def overlap(y, sr, rng, label):
        others = [c for c in overlap_bank if c != label]
        pick = overlap_bank[others[int(rng.integers(0, len(others)))]]
        return T.overlay(y, pick[int(rng.integers(0, len(pick)))], rng, ratio_db=-6.0)

    def real_noise(snr):
        def fn(y, sr, rng, label):
            bed = noise_bank[int(rng.integers(0, len(noise_bank)))]
            return T.add_noise(y, sr, rng, snr_db=snr, noise=bed)
        return fn

    return {
        "clean": lambda y, sr, rng, label: y,
        "background_noise_10dB": real_noise(10.0),
        "background_noise_0dB": real_noise(0.0),
        "low_volume_-30dB": lambda y, sr, rng, label: (y * 10 ** (-30 / 20)).astype(np.float32),
        "echo_rt60_0.8s": lambda y, sr, rng, label: T.reverberate(y, sr, rng, rt60=0.8, wet=0.5),
        "phone_device": lambda y, sr, rng, label: T.device(y, sr, rng, kind="phone"),
        "distant_20m": lambda y, sr, rng, label: T.distance(y, sr, rng, metres=20.0),
        "partial_50pct": lambda y, sr, rng, label: T.partial(y, sr, rng, keep=0.5),
        "overlap_-6dB": overlap,
        "mp3_64kbps": lambda y, sr, rng, label: mp3_roundtrip(y, sr, rng),
    }


def load(path: Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=RATE, mono=True)
    return y.astype(np.float32)


def analyse(pipeline, y: np.ndarray) -> dict:
    pipeline.tracker.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipeline.analyse_samples(y, RATE, origin="upload")


def part_a(pipeline, per_class: int) -> dict:
    with (ROOT / "audio_dataset" / "manifest_with_split.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    test = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["audio_id"]):
        if r["dataset_split"] == "test" and r["original_or_augmented"] == "original":
            test[r["class_label"]].append(r)
    chosen = [r for label in sorted(test) for r in test[label][:per_class]]
    # Noise beds and overlap sources come from the TRAINING split, so a test clip is
    # never mixed with another test clip.
    train = [r for r in rows if r["dataset_split"] == "train"]
    noise_bank = [load(ROOT / "audio_dataset" / r["filename"])
                  for r in train if r["class_label"] == "Background Noise"][:25]
    overlap_bank = defaultdict(list)
    for r in train:
        if len(overlap_bank[r["class_label"]]) < 5:
            overlap_bank[r["class_label"]].append(load(ROOT / "audio_dataset" / r["filename"]))

    results = {}
    for name, fn in conditions(noise_bank, overlap_bank).items():
        rng = np.random.default_rng(SEED)
        py_ok = gtm_ok = rejected = reviewed = crit_total = crit_py = 0
        started = time.time()
        for r in chosen:
            y = fn(load(ROOT / "audio_dataset" / r["filename"]), RATE, rng, r["class_label"])
            rec = analyse(pipeline, y)
            if not rec.get("ok"):
                rejected += 1
                continue
            label = r["class_label"]
            py_ok += rec["predictions"]["python"]["predicted_class"] == label
            gtm_ok += rec["predictions"]["gtm"]["predicted_class"] == label
            reviewed += bool(rec["review"]["required"])
            if label in pipeline.store.critical_classes():
                crit_total += 1
                crit_py += rec["predictions"]["python"]["predicted_class"] == label
        n = len(chosen)
        results[name] = {
            "clips": n, "rejected_as_unusable": rejected,
            "python_accuracy": py_ok / n, "gtm_accuracy": gtm_ok / n,
            "python_critical_recall": crit_py / crit_total if crit_total else None,
            "sent_to_review": reviewed, "seconds": round(time.time() - started, 1),
        }
        print(f"{name:24s} py={py_ok / n:.3f} gtm={gtm_ok / n:.3f} "
              f"rejected={rejected} review={reviewed}", flush=True)
    return {"per_class": per_class, "conditions": results}


CONFUSABLES = {
    "fireworks": "Gunshot", "door_wood_knock": "Gunshot", "clapping": "Gunshot",
    "laughing": "Panic Scream", "crying_baby": "Panic Scream", "church_bells": "Alarm or Siren",
}


def part_b(pipeline, per_category: int) -> dict:
    import pandas as pd
    import soundfile as sf

    frames = [pd.read_parquet(p) for p in
              sorted((ROOT / "audio_dataset" / "raw_downloads").glob("esc50-train-*.parquet"))]
    if not frames:
        return {"skipped": "ESC-50 parquet files not present locally"}
    esc = pd.concat(frames)
    critical = set(pipeline.store.critical_classes())
    out = {}
    for category, risk in CONFUSABLES.items():
        rows = esc[esc.category == category].sort_values("filename").head(per_category)
        predicted, alerts, confident_critical, reviewed = Counter(), 0, 0, 0
        for _, row in rows.iterrows():
            y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if y.ndim == 2:
                y = y.mean(axis=1)
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=RATE)
            rec = analyse(pipeline, y)
            if not rec.get("ok"):
                predicted["(rejected)"] += 1
                continue
            cls = rec["predictions"]["python"]["predicted_class"]
            predicted[cls] += 1
            reviewed += bool(rec["review"]["required"])
            alerts += bool(rec["alert"]["eligible"])
            if cls in critical and not rec["review"]["required"]:
                confident_critical += 1
        out[category] = {"clips": len(rows), "easily_confused_with": risk,
                         "python_predictions": dict(predicted.most_common()),
                         "sent_to_review": reviewed,
                         "critical_without_review": confident_critical,
                         "alert_eligible": alerts}
        print(f"{category:16s} {dict(predicted.most_common(3))} review={reviewed} "
              f"critical_without_review={confident_critical}", flush=True)
    return out


def write_markdown(doc: dict) -> None:
    a = doc["part_a"]["conditions"]
    lines = [
        "# Robustness and confusable-sound probes",
        "",
        f"Generated {doc['generated_at'][:19]} UTC by `tools/robustness_probe.py`. "
        f"Python model `{doc['models']['python']}`, Teachable Machine "
        f"`{doc['models']['gtm']}`. These are probes on degraded or out-of-set audio, "
        "**not** test accuracy.",
        "",
        f"## A. Degraded test recordings ({doc['part_a']['per_class']} per class)",
        "",
        "| Condition | Python acc. | GTM acc. | Python critical recall | Rejected | Sent to review |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in a.items():
        crit = "n/a" if r["python_critical_recall"] is None else f"{r['python_critical_recall']:.2f}"
        lines.append(f"| {name} | {r['python_accuracy']:.2f} | {r['gtm_accuracy']:.2f} | "
                     f"{crit} | {r['rejected_as_unusable']} | {r['sent_to_review']} |")
    b = doc["part_b"]
    lines += ["", "## B. Sounds the models never saw (ESC-50 categories outside our classes)", ""]
    if "skipped" in b:
        lines.append(b["skipped"])
    else:
        lines += ["| Category | Risk | Clips | Python's choices | Sent to review | Critical class without review |",
                  "|---|---|---:|---|---:|---:|"]
        for cat, r in b.items():
            top = ", ".join(f"{k} {v}" for k, v in list(r["python_predictions"].items())[:3])
            lines.append(f"| {cat} | {r['easily_confused_with']} | {r['clips']} | {top} | "
                         f"{r['sent_to_review']} | {r['critical_without_review']} |")
    (ROOT / "reports" / "ROBUSTNESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=15)
    parser.add_argument("--per-category", type=int, default=40)
    parser.add_argument("--skip-a", action="store_true")
    args = parser.parse_args()

    from src.services.pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline.load()
    desc = pipeline.models.describe()
    doc = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "models": {"python": f"{desc['python'].get('model_name')} v{desc['python'].get('model_version')}",
                      "gtm": f"v{desc['gtm'].get('model_version')}"},
           "part_a": {"per_class": 0, "conditions": {}} if args.skip_a else part_a(pipeline, args.per_class),
           "part_b": part_b(pipeline, args.per_category)}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "robustness.json").write_text(json.dumps(doc, indent=2) + "\n")
    write_markdown(doc)


if __name__ == "__main__":
    main()
