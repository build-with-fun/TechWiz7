"""Score the served Python model on the train, validation and test splits.

    python tools/split_results.py

Uses python_models/best/ and the cached embeddings, so nothing is retrained. Writes
python_models/metrics/served_model_split_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python_models.train_transfer import class_config, load_split  # noqa: E402


def main() -> None:
    best = ROOT / "python_models" / "best"
    model = joblib.load(best / "model.joblib")
    meta = json.loads((best / "model_meta.json").read_text(encoding="utf-8"))
    classes, critical = class_config()
    doc = {"model": meta.get("model_name"), "version": meta.get("model_version"), "splits": {}}
    # An ensemble lists several backbones; its features are their embeddings joined.
    backbones = meta.get("backbones") or [meta.get("backbone", "cnn14")]
    for split in ("train", "val", "test"):
        parts = [load_split(meta.get("preprocessing_variant", "current"), split, backbone=b)
                 for b in backbones]
        if any(part[0] != parts[0][0] for part in parts):
            raise SystemExit(f"{split}: the embedding caches list different recordings")
        X, y = np.hstack([part[1] for part in parts]), parts[0][2]
        order = np.asarray(model.classes_).astype(str)
        pred = order[model.predict_proba(X).argmax(1)]
        p, r, f, n = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
        per_class = {c: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                         "f1": round(float(f[i]), 3), "support": int(n[i])}
                     for i, c in enumerate(classes)}
        doc["splits"][split] = {
            "n": int(len(y)),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, labels=classes, average="macro")), 4),
            "critical_recall": round(float(np.mean([per_class[c]["recall"] for c in critical])), 4),
            "per_class": per_class,
        }
        s = doc["splits"][split]
        print(f"{split:5s} n={s['n']:4d}  acc {s['accuracy']:.3f}  macro-F1 {s['macro_f1']:.3f}  "
              f"critical recall {s['critical_recall']:.3f}")
    out = ROOT / "python_models" / "metrics" / "served_model_split_results.json"
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
