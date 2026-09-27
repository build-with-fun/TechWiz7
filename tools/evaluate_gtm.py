"""Score the exported TM model on a balanced test subset (server-side path only)."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from audio_preprocessing.pipeline import AudioPipeline
from src.inference.contract import AudioSource
from src.inference.gtm_predictor import GtmModelPredictor


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Score a Teachable Machine export")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="val to compare candidate exports; test only for the chosen one")
    parser.add_argument("--gtm-dir", default=str(ROOT / "gtm_model"),
                        help="folder with metadata.json, frontend_config.json and gtm_model.h5")
    parser.add_argument("--out", default=None, help="metrics file (default: gtm_metrics.json "
                        "in --gtm-dir for test, gtm_val_metrics.json for val)")
    parser.add_argument("--aggregation", choices=["loudest", "energy_weighted"], default=None,
                        help="override window_aggregation from frontend_config.json (for val comparisons)")
    args = parser.parse_args()
    gtm_dir = Path(args.gtm_dir)
    metrics_path = Path(args.out) if args.out else gtm_dir / (
        "gtm_metrics.json" if args.split == "test" else "gtm_val_metrics.json")
    with (ROOT / "audio_dataset/manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset_split"] == args.split]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["class_label"]].append(row)
    labels = sorted(groups)
    chosen = [row for label in labels for row in sorted(
        groups[label], key=lambda item: item["audio_id"]
    )]
    if len(chosen) != 450 or any(len(groups[label]) != 45 for label in labels):
        raise ValueError(f"Expected 45 held-out clips per class, got {len(chosen)} total")

    model = GtmModelPredictor.load(gtm_dir)
    if args.aggregation:
        model.frontend.window_aggregation = args.aggregation
    preprocessor = AudioPipeline()
    truth, predicted = [], []
    results = []
    for row in chosen:
        path = ROOT / "audio_dataset" / row["filename"]
        output = model.predict(AudioSource.from_path(path), preprocessor)
        truth.append(row["class_label"])
        predicted.append(output.predicted_class)
        results.append({"audio_id": row["audio_id"], "truth": row["class_label"],
                        "prediction": output.predicted_class,
                        "confidence": round(output.confidence, 6)})
        if len(results) % 25 == 0 or len(results) == len(chosen):
            print(f"Scored {len(results)}/{len(chosen)} held-out clips", flush=True)

    critical = ["Gunshot", "Glass Breaking", "Panic Scream", "Aggression",
                "Person Asking for Help"]
    per_class_recall = recall_score(truth, predicted, labels=labels, average=None,
                                    zero_division=0)
    recall_by_class = dict(zip(labels, [float(x) for x in per_class_recall]))
    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "protocol": f"all 45 {args.split} clips per class of split v2; only training "
                    "recordings were used to train the model",
        "n_clips": len(chosen), "model_version": model.model_version,
        "frontend_verified": model.verified,
        "window_aggregation": model.frontend.window_aggregation,
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "critical_macro_recall": sum(recall_by_class[name] for name in critical) / len(critical),
        "recall_by_class": recall_by_class,
        "labels": labels,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "samples": results,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in
                      ("n_clips", "accuracy", "macro_f1", "critical_macro_recall",
                       "frontend_verified")}, indent=2))


if __name__ == "__main__":
    main()
