"""Train the served Python model: AST, CLAP and CNN14 embeddings, averaged.

    python -m python_models.train_ensemble select
    python -m python_models.train_ensemble final --out python_models/best

Each member is a logistic regression on one embedding (averaging needs calibrated scores,
which an MLP doesn't give). Its C comes from 5-fold cross-validation on the training
recordings, grouped by source. Validation then picks which members to use, by the usual
score (0.5 x macro-F1 + 0.5 x critical recall; within 0.005 the smaller set wins). ``final``
scores test once and saves the bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python_models.preprocess_cache import VARIANTS  # noqa: E402
from python_models.train_transfer import (  # noqa: E402
    METRICS, SEED, class_config, load_split, run_tag, selection_score,
)
from src.training.evaluation import compute_metrics  # noqa: E402

MEMBERS = ("ast", "clap", "cnn14")


C_GRID = (0.003, 0.01, 0.03, 0.1)
CV_FOLDS = 5


def source_groups(ids: list[str]) -> list[str]:
    """Source recording of each clip, so slices of one upload stay in one fold."""
    with (ROOT / "audio_dataset" / "manifest_with_split.csv").open(newline="", encoding="utf-8") as fh:
        rows = {r["audio_id"]: r for r in csv.DictReader(fh)}
    return [rows[a].get("source_url") or a for a in ids]


def logreg(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def choose_c(variant: str, backbone: str, class_names: list[str], critical: list[str]) -> dict:
    """Best C for one member, by grouped cross-validation on train."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedGroupKFold

    ids, X, y = load_split(variant, "train", backbone=backbone)
    folds = list(StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
                 .split(X, y, source_groups(ids)))

    def fold_proba(C, train, held):
        est = logreg(C).fit(X[train], y[train])
        return held, aligned_proba(est, X[held], class_names)

    rows = []
    for C in C_GRID:
        proba = np.zeros((len(y), len(class_names)))
        for held, p in Parallel(n_jobs=CV_FOLDS)(delayed(fold_proba)(C, tr, te) for tr, te in folds):
            proba[held] = p
        pred = [class_names[i] for i in proba.argmax(axis=1)]
        res = compute_metrics(list(y), pred, class_names, critical, model_name=f"{backbone} C={C}", split="train_cv")
        rows.append({"C": C, "cv_accuracy": res.accuracy, "cv_macro_f1": res.macro_f1,
                     "cv_critical_recall": res.critical_recall,
                     "cv_score": selection_score(res.macro_f1, res.critical_recall)})
        print(f"cv {backbone:6s} C={C:<6} acc={res.accuracy:.4f} score={rows[-1]['cv_score']:.4f}", flush=True)
    best = max(rows, key=lambda r: (round(r["cv_score"] / 0.005), -r["C"]))
    return {"C": best["C"], "cv": rows}


def fit_members(variant: str, chosen_c: dict) -> dict:
    out = {}
    for backbone in MEMBERS:
        ids, X, y = load_split(variant, "train", backbone=backbone)
        C = chosen_c[backbone]
        out[backbone] = {"estimator": logreg(C).fit(X, y), "family": "logreg",
                         "params": {"C": C}, "train_ids": ids}
        print(f"member {backbone}: logreg C={C} on {len(y)} rows", flush=True)
    return out


def aligned_proba(estimator, X, class_names) -> np.ndarray:
    proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    order = [str(c) for c in estimator.classes_]
    return proba[:, [order.index(c) for c in class_names]]


def split_proba(fitted: dict, variant: str, split: str, class_names: list[str]):
    ids, y, proba = None, None, {}
    for backbone in MEMBERS:
        b_ids, X, b_y = load_split(variant, split, backbone=backbone)
        if ids is None:
            ids, y = b_ids, list(b_y)
        elif b_ids != ids:
            raise SystemExit(f"{split}: {backbone} embeddings list different recordings")
        proba[backbone] = aligned_proba(fitted[backbone]["estimator"], X, class_names)
    return ids, y, proba


def candidate_sets() -> list[tuple[str, ...]]:
    return [combo for r in range(1, len(MEMBERS) + 1)
            for combo in itertools.combinations(MEMBERS, r)]


def select(variant: str) -> dict:
    class_names, critical = class_config()
    cv = {b: choose_c(variant, b, class_names, critical) for b in MEMBERS}
    fitted = fit_members(variant, {b: cv[b]["C"] for b in MEMBERS})
    _, yva, pva = split_proba(fitted, variant, "val", class_names)
    rows = [score_candidate(combo, pva, yva, class_names, critical) for combo in candidate_sets()]
    rows.sort(key=lambda r: (-round(r["selection_score"] / 0.005), len(r["members"])))
    doc = {"variant": variant,
           "members": {b: {"family": "logreg", "C": cv[b]["C"], "cv": cv[b]["cv"]} for b in MEMBERS},
           "criterion": "member C: grouped 5-fold CV on train; member set: 0.5*val_macro_f1 + "
                        "0.5*val_critical_recall, equal weights, ties (0.005) -> fewer members",
           "selected": rows[0], "all": rows,
           "decided_at": datetime.now(timezone.utc).isoformat()}
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / f"ensemble_selection_{run_tag('ensemble', variant, False)}.json").write_text(
        json.dumps(doc, indent=2) + "\n")
    print("selected:", "+".join(rows[0]["members"]))
    return doc


def score_candidate(combo, pva, yva, class_names, critical) -> dict:
    blended = np.mean([pva[b] for b in combo], axis=0)
    pred = [class_names[i] for i in blended.argmax(axis=1)]
    res = compute_metrics(yva, pred, class_names, critical, model_name="+".join(combo), split="val")
    row = {"members": list(combo), "val_accuracy": res.accuracy,
           "val_macro_f1": res.macro_f1, "val_critical_recall": res.critical_recall,
           "val_critical_recall_by_class": res.critical_recall_by_class,
           "selection_score": selection_score(res.macro_f1, res.critical_recall)}
    print(f"{'+'.join(combo):16s} acc={res.accuracy:.4f} f1={res.macro_f1:.4f} "
          f"crit={res.critical_recall:.4f} score={row['selection_score']:.4f}", flush=True)
    return row


def final(variant: str, out_dir: Path) -> None:
    """Refit the chosen members, score test once and save the bundle."""
    from feature_extraction import ensemble_embeddings
    from src.inference.ensemble import EnsembleMember, SoftVotingEnsemble
    from src.inference.predictor import save_bundle

    tag = run_tag("ensemble", variant, False)
    selection = json.loads((METRICS / f"ensemble_selection_{tag}.json").read_text())
    chosen = list(selection["selected"]["members"])
    class_names, critical = class_config()
    fitted = fit_members(variant, {b: selection["members"][b]["C"] for b in MEMBERS})
    ids_te, yte, pte = split_proba(fitted, variant, "test", class_names)
    if len(yte) != 450:
        raise SystemExit(f"expected 450 test embeddings, found {len(yte)}")

    results = {}
    for combo in [tuple(chosen)] + [(b,) for b in MEMBERS if (b,) != tuple(chosen)]:
        blended = np.mean([pte[b] for b in combo], axis=0)
        pred = [class_names[i] for i in blended.argmax(axis=1)]
        top2 = [[class_names[i] for i in np.argsort(p)[::-1][:2]] for p in blended]
        res = compute_metrics(yte, pred, class_names, critical, model_name="+".join(combo),
                              split="test", top2=top2)
        results["+".join(combo)] = res
        print(f"TEST {'+'.join(combo):16s} acc={res.accuracy:.4f} f1={res.macro_f1:.4f} "
              f"crit={res.critical_recall:.4f} {res.critical_recall_by_class}", flush=True)
    chosen_res = results["+".join(chosen)]
    chosen_proba = np.mean([pte[b] for b in chosen], axis=0)

    members, start = [], 0
    for backbone in chosen:
        width = len(ensemble_embeddings.backbone_module(backbone).embedding_columns())
        members.append(EnsembleMember(backbone, fitted[backbone]["estimator"], start, start + width))
        start += width
    ensemble = SoftVotingEnsemble(members, class_names)

    train_ids = fitted[chosen[0]]["train_ids"]
    name = "ensemble(" + "+".join(chosen) + ")"
    meta = {
        "model_family": "transfer_learning_ensemble",
        "backbones": chosen,
        "members": {b: {"family": fitted[b]["family"], "params": fitted[b]["params"],
                        "embedding_version": ensemble_embeddings.backbone_module(b).EMBEDDING_VERSION}
                    for b in chosen},
        "combination": "unweighted mean of member probabilities",
        "preprocessing_variant": variant,
        "preprocessing_overrides": VARIANTS[variant],
        "selection": selection["selected"],
        "selection_criterion": selection["criterion"],
        "n_train_rows": len(train_ids),
        "train_ids_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
        "seed": SEED,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_results": {k: {m: getattr(v, m) for m in ("accuracy", "macro_f1", "critical_recall")}
                         for k, v in results.items()},
    }
    save_bundle(ensemble, out_dir, class_names=class_names,
                feature_version=ensemble_embeddings.embedding_version(chosen),
                feature_columns=ensemble_embeddings.embedding_columns(chosen),
                model_name=name, model_version=f"3.1.0-{tag}", metrics=chosen_res.to_dict(),
                metadata=meta)
    (METRICS / f"ensemble_test_{tag}.json").write_text(json.dumps(
        {"selected_members": chosen, "results": {k: v.to_dict() for k, v in results.items()}},
        indent=2) + "\n")
    with (METRICS / f"ensemble_test_predictions_{tag}.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["audio_id", "actual", "predicted", "confidence"] + class_names)
        for aid, actual, p in zip(ids_te, yte, chosen_proba):
            writer.writerow([aid, actual, class_names[int(p.argmax())], f"{p.max():.4f}"]
                            + [f"{v:.4f}" for v in p])
    print(f"saved {name} to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["select", "final"])
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="current")
    parser.add_argument("--out", default="python_models/best_ensemble")
    args = parser.parse_args()
    if args.stage == "select":
        select(args.variant)
    else:
        final(args.variant, ROOT / args.out)


if __name__ == "__main__":
    main()
