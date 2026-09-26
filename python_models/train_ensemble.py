"""Soft-voting ensemble experiment across embedding families (val-selected, test-scored once).

Protocol (identical rules to train_transfer.py):

* ``fit`` sees only the 2,100 training originals (plus augmented copies with --augmented);
* the blend weight is a hyper-parameter, so it is chosen on the 450 validation recordings by
  the same ``selection_score`` criterion (0.5*macro_f1 + 0.5*critical_recall);
* the 450 test recordings are scored exactly once, after the weight is written down.

Two members, each the *selected* configuration of its own family:
  * CNN14 (PANNs) embeddings -> the classifier family that won on validation for that cache;
  * AST embeddings           -> the classifier family that won on validation for that cache.

Members are blended as ``w * p_ast + (1 - w) * p_cnn14`` with probabilities aligned to the
configured class order, so the blend can never map a score onto the wrong name.

This is an EXPERIMENT. ``python_models/best`` is only replaced by a blend if the blend wins
on validation by more than the 0.005 tie band AND it can be served under the saved-bundle
contract (a single feature vector per clip). A blend of two different embedding dimensions
cannot be, so a cross-family blend is reported here as a result and the served bundle stays
a single-backbone model unless a same-backbone blend wins.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python_models.train_transfer import (  # noqa: E402
    BACKBONES, METRICS, SEED, build_estimator, class_config, emb_dir, load_split,
    run_tag, score, selection_score,
)
from src.training.evaluation import compute_metrics  # noqa: E402

WEIGHTS = [round(w, 2) for w in np.arange(0.0, 1.01, 0.1)]


def _pick(selection_doc: dict) -> tuple[str, dict]:
    return selection_doc["selected"]["family"], selection_doc["selected"]["params"]


def aligned_proba(estimator, X, class_names) -> np.ndarray:
    """predict_proba re-sorted into the configured class order."""
    proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    order = [str(c) for c in estimator.classes_]
    idx = [order.index(c) for c in class_names]
    return proba[:, idx]


def load_selection(backbone: str, variant: str, augmented: bool) -> dict:
    path = METRICS / f"transfer_selection_{run_tag(backbone, variant, augmented)}.json"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing: run `select --backbone {backbone}"
            f"{' --augmented' if augmented else ''}` first")
    return json.loads(path.read_text())


def members(variant: str, augmented: bool):
    """Fit each family's selected configuration and return fitted estimators + configs."""
    out = {}
    for backbone in sorted(BACKBONES):
        sel = load_selection(backbone, variant, augmented)
        family, params = _pick(sel)
        _, Xtr, ytr = load_split(variant, "train", include_augmented=augmented, backbone=backbone)
        est = build_estimator(family, params)
        est.fit(Xtr, ytr)
        out[backbone] = {"estimator": est, "family": family, "params": params,
                         "selection": sel["selected"]}
        print(f"member {backbone}: {family} {params} fitted on {len(ytr)} train rows", flush=True)
    return out


def blend(p_ast: np.ndarray, p_cnn14: np.ndarray, w: float) -> np.ndarray:
    return w * p_ast + (1.0 - w) * p_cnn14


def evaluate(variant: str, augmented: bool, *, out_tag: str) -> dict:
    class_names, critical = class_config()
    fitted = members(variant, augmented)

    _, Xva_ast, yva_ast = load_split(variant, "val", backbone="ast")
    _, Xva_cnn, yva_cnn = load_split(variant, "val", backbone="cnn14")
    yva = list(yva_ast)
    assert list(yva_cnn) == yva, "val label order differs between embedding caches"

    pva = {"ast": aligned_proba(fitted["ast"]["estimator"], Xva_ast, class_names),
           "cnn14": aligned_proba(fitted["cnn14"]["estimator"], Xva_cnn, class_names)}

    rows = []
    for w in WEIGHTS:
        blended = blend(pva["ast"], pva["cnn14"], w)
        pred = [class_names[i] for i in blended.argmax(axis=1)]
        res = compute_metrics(yva, pred, class_names, critical,
                              model_name=f"blend(w_ast={w:.1f})", split="val")
        rows.append({"w_ast": w, "val_accuracy": res.accuracy, "val_macro_f1": res.macro_f1,
                     "val_critical_recall": res.critical_recall,
                     "selection_score": selection_score(res.macro_f1, res.critical_recall)})
        print(f"blend w_ast={w:.1f} acc={res.accuracy:.4f} f1={res.macro_f1:.4f} "
              f"crit={res.critical_recall:.4f}", flush=True)

    # Single-member reference points, so the blend is only preferred if it really beats them.
    for backbone, Xva in (("ast", Xva_ast), ("cnn14", Xva_cnn)):
        pred = [class_names[i] for i in pva[backbone].argmax(axis=1)]
        res = compute_metrics(yva, pred, class_names, critical,
                              model_name=f"{backbone}_solo", split="val")
        rows.append({"w_ast": 1.0 if backbone == "ast" else 0.0,
                     "val_accuracy": res.accuracy, "val_macro_f1": res.macro_f1,
                     "val_critical_recall": res.critical_recall,
                     "selection_score": selection_score(res.macro_f1, res.critical_recall),
                     "member": f"{backbone}_solo"})
        print(f"{backbone}_solo acc={res.accuracy:.4f} f1={res.macro_f1:.4f} "
              f"crit={res.critical_recall:.4f}", flush=True)

    rows.sort(key=lambda r: -r["selection_score"])
    best = rows[0]
    w_best = best["w_ast"]

    doc = {"variant": variant, "augmented": augmented,
           "members": {k: {"family": v["family"], "params": v["params"],
                           "val": {m: v["selection"][m] for m in
                                   ("val_accuracy", "val_macro_f1", "val_critical_recall")}}
                      for k, v in fitted.items()},
           "criterion": "0.5*val_macro_f1 + 0.5*val_critical_recall",
           "chosen_w_ast": w_best, "chosen_val": best, "all_weights": rows,
           "decided_at": datetime.now(timezone.utc).isoformat()}
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / f"ensemble_selection_{out_tag}.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"chosen blend weight w_ast={w_best} (val selection_score={best['selection_score']:.4f})")
    return doc


def final_test(variant: str, augmented: bool, *, out_tag: str) -> dict:
    """Score the frozen test split once with the val-chosen weight. Test is touched one time."""
    class_names, critical = class_config()
    chosen = json.loads((METRICS / f"ensemble_selection_{out_tag}.json").read_text())
    w = chosen["chosen_w_ast"]

    fitted = members(variant, augmented)
    ids_te = {}
    p_te = {}
    yte = None
    for backbone in sorted(BACKBONES):
        ids, Xte, y = load_split(variant, "test", backbone=backbone)
        ids_te[backbone] = ids
        p_te[backbone] = aligned_proba(fitted[backbone]["estimator"], Xte, class_names)
        if yte is None:
            yte = list(y)
        else:
            assert list(y) == yte, "test label order differs between embedding caches"
    assert ids_te["ast"] == ids_te["cnn14"], "test id order differs between embedding caches"
    assert len(yte) == 450, f"expected 450 test rows, found {len(yte)}"

    blended = blend(p_te["ast"], p_te["cnn14"], w)
    pred = [class_names[i] for i in blended.argmax(axis=1)]
    res = compute_metrics(yte, pred, class_names, critical,
                          model_name=f"ensemble_blend(w_ast={w})", split="test")
    print(f"TEST ensemble w_ast={w} acc={res.accuracy:.4f} f1={res.macro_f1:.4f} "
          f"crit={res.critical_recall:.4f} {res.critical_recall_by_class}", flush=True)

    # Solo members on test, for the honest before/after table. These are the same two
    # families whose test scores are already recorded by their own `final` runs; recomputed
    # here only so the table is self-consistent under one code path.
    solo = {}
    for backbone in sorted(BACKBONES):
        s = [class_names[i] for i in p_te[backbone].argmax(axis=1)]
        r = compute_metrics(yte, s, class_names, critical,
                            model_name=f"{backbone}_solo", split="test")
        solo[backbone] = {"accuracy": r.accuracy, "macro_f1": r.macro_f1,
                          "critical_recall": r.critical_recall,
                          "critical_recall_by_class": r.critical_recall_by_class}
        print(f"TEST {backbone}_solo acc={r.accuracy:.4f} f1={r.macro_f1:.4f} "
              f"crit={r.critical_recall:.4f}", flush=True)

    out = {"chosen_w_ast": w, "variant": variant, "augmented": augmented,
           "ensemble": res.to_dict(), "solo": solo,
           "scored_at": datetime.now(timezone.utc).isoformat()}
    (METRICS / f"ensemble_test_{out_tag}.json").write_text(json.dumps(out, indent=2) + "\n")

    import csv
    with (METRICS / f"ensemble_test_predictions_{out_tag}.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["audio_id", "actual", "predicted", "confidence"] + class_names)
        for aid, actual, p in zip(ids_te["ast"], yte, blended):
            writer.writerow([aid, actual, class_names[int(p.argmax())], f"{p.max():.4f}"]
                            + [f"{v:.4f}" for v in p])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["select", "final"])
    parser.add_argument("--variant", default="current")
    parser.add_argument("--augmented", action="store_true")
    args = parser.parse_args()
    tag = run_tag("ensemble", args.variant, args.augmented)
    if args.stage == "select":
        evaluate(args.variant, args.augmented, out_tag=tag)
    else:
        final_test(args.variant, args.augmented, out_tag=tag)


if __name__ == "__main__":
    main()
