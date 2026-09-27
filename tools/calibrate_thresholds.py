"""Check the confidence threshold on the validation split.

    python tools/calibrate_thresholds.py

For each ``confidence.min_confidence`` value it shows how many clips would be decided
automatically, how accurate those are, and how many critical clips would be decided
wrongly. Uses python_models/best/ and the cached embeddings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python_models.train_transfer import class_config, load_split  # noqa: E402


def main() -> None:
    best = ROOT / "python_models" / "best"
    model = joblib.load(best / "model.joblib")
    meta = json.loads((best / "model_meta.json").read_text(encoding="utf-8"))
    _, critical = class_config()
    _, X, y = load_split(meta.get("preprocessing_variant", "current"), "val",
                         backbone=meta.get("backbone", "cnn14"))
    proba = model.predict_proba(X)
    order = np.asarray(model.classes_).astype(str)
    pred, conf = order[proba.argmax(1)], proba.max(1)
    rows = []
    for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        auto = conf >= t
        crit = np.isin(y, critical)
        rows.append({
            "min_confidence": t,
            "auto_decided_share": round(float(auto.mean()), 3),
            "accuracy_of_auto_decisions": round(float((pred[auto] == y[auto]).mean()), 3) if auto.any() else None,
            "accuracy_sent_to_review": round(float((pred[~auto] == y[~auto]).mean()), 3) if (~auto).any() else None,
            "critical_recordings_wrongly_auto_decided": int((auto & crit & (pred != y)).sum()),
        })
    configured = json.loads((ROOT / "config" / "thresholds.json").read_text())["confidence"]["min_confidence"]
    doc = {"split": "val", "n": int(len(y)), "configured_min_confidence": configured, "table": rows}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "threshold_calibration.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
