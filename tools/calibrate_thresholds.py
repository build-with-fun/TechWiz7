"""Pick the decision thresholds on the validation split.

    python tools/calibrate_thresholds.py            # report only
    python tools/calibrate_thresholds.py --apply    # also write them to config/thresholds.json

Both served models score the 450 validation clips, then the app's own consistency and
review code runs for every setting in GRID. A clip is accepted when no review condition
fires. Rule, fixed in advance: accept as many clips as possible while at least
TARGET_PRECISION of the accepted ones (and of the accepted critical ones) are right. Ties
go to the stricter setting. Test clips are never used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python_models.preprocess_cache import load_cached, load_manifest, read_index  # noqa: E402
from python_models.train_transfer import class_config, emb_dir  # noqa: E402
from src.inference.consistency import classify_consistency  # noqa: E402
from src.inference.contract import PredictionResult, normalise_confidences  # noqa: E402
from src.inference.gtm_predictor import GtmModelPredictor  # noqa: E402
from src.services.config import ConfigStore  # noqa: E402
from src.services.pipeline import evaluate_review  # noqa: E402

TARGET_PRECISION = 0.97
GRID = {
    "confidence.min_confidence": [0.4, 0.5, 0.6, 0.7],
    "confidence.low_confidence_band": [0.4, 0.5, 0.6, 0.7, 0.75],
    "confidence.top_two_margin_min": [0.1, 0.15, 0.2],
    "consistency.confident_agreement_min": [0.5, 0.6, 0.7, 0.8],
}
EMBEDDING_PREFIX = {"ast": "ast", "clap": "clap", "panns": "cnn14"}


class GridStore(ConfigStore):
    """The real config, with thresholds.json swapped for one grid point."""

    def __init__(self, thresholds: dict) -> None:
        super().__init__()
        self._grid_thresholds = thresholds

    def thresholds(self) -> dict:
        return self._grid_thresholds


def backbones_of(feature_version: str) -> list[str]:
    parts = feature_version.split(":", 1)[1].split("+") if feature_version.startswith("ensemble:") \
        else [feature_version]
    return [next(b for prefix, b in EMBEDDING_PREFIX.items() if p.startswith(prefix)) for p in parts]


def python_scores(model_dir: Path, ids: list[str]) -> tuple[list[str], np.ndarray]:
    """Served Python model's scores from the cached embeddings."""
    estimator = joblib.load(model_dir / "model.joblib")
    version = json.loads((model_dir / "feature_config.json").read_text())["feature_version"]
    variant = json.loads((model_dir / "model_meta.json").read_text()).get("preprocessing_variant", "current")
    X = np.stack([np.concatenate([np.load(emb_dir(variant, b) / f"{a}.npy") for b in backbones_of(version)])
                  for a in ids]).astype(np.float64)
    return [str(c) for c in estimator.classes_], np.asarray(estimator.predict_proba(X))


def result(name: str, classes: list[str], scores) -> PredictionResult:
    confidences = normalise_confidences([float(v) for v in scores], classes)
    top = max(confidences.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return PredictionResult(model_name=name, model_version="calibration", predicted_class=top,
                            confidence=confidences[top], confidences=confidences)


def set_path(doc: dict, dotted: str, value: float) -> None:
    section, key = dotted.split(".")
    doc[section][key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python-dir", default=str(ROOT / "python_models" / "best"))
    parser.add_argument("--gtm-dir", default=str(ROOT / "gtm_model"))
    parser.add_argument("--apply", action="store_true", help="write the chosen values to config/thresholds.json")
    args = parser.parse_args()

    class_names, critical = class_config()
    rows = [r for r in load_manifest("val")]
    index = read_index()
    rows = [r for r in rows if index.get(r["audio_id"], {}).get("rejected") != "True"]
    ids = [r["audio_id"] for r in rows]
    truth = [r["class_label"] for r in rows]
    quality = [index[a]["quality"] for a in ids]

    py_classes, py_proba = python_scores(Path(args.python_dir), ids)
    gtm = GtmModelPredictor.load(args.gtm_dir)
    py_results, gtm_results = [], []
    for n, (audio_id, scores) in enumerate(zip(ids, py_proba), 1):
        py_results.append(result("python", py_classes, scores))
        audio = SimpleNamespace(samples=load_cached(audio_id), sample_rate=int(index[audio_id]["sample_rate"]))
        gtm_results.append(gtm.predict_from_preprocessed(audio))
        if n % 150 == 0:
            print(f"scored {n}/{len(ids)} validation recordings", flush=True)

    base = json.loads((ROOT / "config" / "thresholds.json").read_text(encoding="utf-8"))
    table = []
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        point = dict(zip(keys, values))
        # Skip settings that would let a model below the floor skip review.
        if point["confidence.low_confidence_band"] < point["confidence.min_confidence"] or \
                point["consistency.confident_agreement_min"] < point["confidence.min_confidence"]:
            continue
        thresholds = copy.deepcopy(base)
        for key, value in zip(keys, values):
            set_path(thresholds, key, value)
        store = GridStore(thresholds)
        accepted = correct = crit_accepted = crit_correct = agree = 0
        statuses: dict[str, int] = {}
        for py, gt, actual, verdict in zip(py_results, gtm_results, truth, quality):
            comparison = classify_consistency(py, gt, thresholds)
            statuses[comparison.consistency_status] = statuses.get(comparison.consistency_status, 0) + 1
            agree += comparison.classes_agree
            review = evaluate_review(
                store=store, python_class=py.predicted_class, python_confidence=py.confidence,
                gtm_class=gt.predicted_class, gtm_confidence=gt.confidence,
                classes_agree=comparison.classes_agree,
                confidence_difference=comparison.confidence_difference,
                top_two_margin=comparison.top_two_margin_python,
                consistency_status=comparison.consistency_status, quality=verdict,
                overlapping=comparison.overlapping, secondary_detection=comparison.secondary_detection,
                predicted_class=py.predicted_class)
            if review["required"]:
                continue
            accepted += 1
            correct += py.predicted_class == actual
            if py.predicted_class in critical:
                crit_accepted += 1
                crit_correct += py.predicted_class == actual
        table.append({
            **dict(zip(keys, values)),
            "accepted_share": round(accepted / len(ids), 4),
            "accepted_precision": round(correct / accepted, 4) if accepted else None,
            "accepted_critical": crit_accepted,
            "accepted_critical_precision": round(crit_correct / crit_accepted, 4) if crit_accepted else None,
            "accepted_wrong": accepted - correct,
            "status_counts": statuses,
        })

    def eligible(row: dict) -> bool:
        return (row["accepted_precision"] or 0) >= TARGET_PRECISION and \
            (row["accepted_critical_precision"] is None or row["accepted_critical_precision"] >= TARGET_PRECISION)

    ranked = sorted((r for r in table if eligible(r)),
                    key=lambda r: (-r["accepted_share"], *(-r[k] for k in keys)))
    chosen = ranked[0] if ranked else None
    configured = {k: base[k.split(".")[0]].get(k.split(".")[1]) for k in keys}
    current = next((r for r in table if all(r[k] == configured[k] for k in keys)), None)
    py_pred = [r.predicted_class for r in py_results]
    gtm_pred = [r.predicted_class for r in gtm_results]
    doc = {
        "split": "val", "n": len(ids),
        "python_model": str(Path(args.python_dir).relative_to(ROOT)),
        "gtm_model": gtm.model_version,
        "python_accuracy": round(float(np.mean([p == t for p, t in zip(py_pred, truth)])), 4),
        "gtm_accuracy": round(float(np.mean([p == t for p, t in zip(gtm_pred, truth)])), 4),
        "agreement_rate": round(float(np.mean([p == g for p, g in zip(py_pred, gtm_pred)])), 4),
        "accuracy_when_models_agree": round(float(np.mean(
            [p == t for p, g, t in zip(py_pred, gtm_pred, truth) if p == g])), 4),
        "rule": f"max accepted share with accepted precision >= {TARGET_PRECISION} overall and "
                f"for critical predictions; ties -> stricter thresholds",
        "grid": GRID,
        "configured_before": {"values": configured, "result": current},
        "chosen": chosen,
        "table": sorted(table, key=lambda r: -r["accepted_share"]),
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "threshold_calibration.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({k: doc[k] for k in ("python_accuracy", "gtm_accuracy", "agreement_rate",
                                           "accuracy_when_models_agree", "configured_before", "chosen")},
                     indent=2))
    if args.apply and chosen:
        updated = json.loads((ROOT / "config" / "thresholds.json").read_text(encoding="utf-8"))
        for key in keys:
            set_path(updated, key, chosen[key])
        (ROOT / "config" / "thresholds.json").write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        print("config/thresholds.json updated")


if __name__ == "__main__":
    main()
