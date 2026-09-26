"""Train the transfer-learning Python model: pretrained embeddings + a small classifier.

    python -m python_models.train_transfer embed  --backbone ast --splits train val test
    python -m python_models.train_transfer select --backbone ast
    python -m python_models.train_transfer final  --backbone ast --out python_models/best

Backbones: ``cnn14`` (PANNs, feature_extraction/embeddings.py) and ``ast`` (Audio
Spectrogram Transformer, feature_extraction/ast_embeddings.py).

Protocol (the same rules the classical trainer follows):

* ``fit`` only ever sees the 2,100 training recordings, plus their augmented copies
  when ``--augmented`` is given (copies inherit their parent's split, see
  ``augmentation/augment_dataset.py``);
* every hyper-parameter and the classifier family are chosen on the 450 validation
  recordings, by the criterion in ``selection_score``;
* the 450 test recordings are embedded and scored once, by ``final``, after the choice is
  written down in ``python_models/metrics/transfer_selection.json``.

Selection criterion, decided before looking at test numbers: mean of validation macro-F1
and validation critical-class recall. Accuracy alone would let a model trade a missed
gunshot for an extra correct vehicle horn, which is the wrong trade for this product.
Ties within 0.005 go to the faster model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python_models.preprocess_cache import VARIANTS, load_cached, load_manifest, read_index  # noqa: E402

METRICS = ROOT / "python_models" / "metrics"
SEED = 20260926


BACKBONES = {"cnn14": "panns", "ast": "ast"}


def emb_dir(variant: str, backbone: str = "cnn14") -> Path:
    return ROOT / "data" / "cache" / f"{BACKBONES[backbone]}_{variant}"


def backbone_module(backbone: str):
    if backbone == "ast":
        from feature_extraction import ast_embeddings as mod
    else:
        from feature_extraction import embeddings as mod
    return mod


def make_embedder(backbone: str):
    threads = int(os.environ.get("SST_TORCH_THREADS", "8"))
    if backbone == "ast":
        return backbone_module(backbone).AstEmbedder(threads=threads)
    return backbone_module(backbone).PannsEmbedder(threads=threads)


def class_config() -> tuple[list[str], list[str]]:
    doc = json.loads((ROOT / "config" / "classes.json").read_text(encoding="utf-8"))
    return [c["name"] for c in doc["classes"]], list(doc["critical_classes"])


# embeddings

def augmented_rows() -> list[dict[str, str]]:
    path = ROOT / "audio_dataset" / "manifests" / "augmented_rows.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def embed(variant: str, splits: list[str], *, include_augmented: bool = False,
          backbone: str = "cnn14") -> None:
    import warnings

    from audio_preprocessing import config as cfg_mod
    from audio_preprocessing.pipeline import AudioPipeline

    out = emb_dir(variant, backbone)
    out.mkdir(parents=True, exist_ok=True)
    embedder = make_embedder(backbone)
    index = read_index(variant)
    todo = [r["audio_id"] for r in load_manifest() if r["dataset_split"] in splits]
    started = time.time()
    for n, audio_id in enumerate(todo, 1):
        target = out / f"{audio_id}.npy"
        if target.exists() or index.get(audio_id, {}).get("rejected") == "True":
            continue
        np.save(target, embedder.embed(load_cached(audio_id, variant)).astype(np.float32))
        if n % 250 == 0:
            print(f"  {n}/{len(todo)} embedded, {time.time() - started:.0f}s", flush=True)

    if include_augmented and "train" in splits:
        # Augmented copies are raw audio, so they go through the serving preprocessor
        # first, exactly like an upload would.
        pipeline = AudioPipeline(audio_cfg={**cfg_mod.audio_config(), **VARIANTS[variant]})
        rows = augmented_rows()
        for n, row in enumerate(rows, 1):
            target = out / f"{row['audio_id']}.npy"
            if target.exists():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pre = pipeline.preprocess_file(ROOT / "audio_dataset" / row["filename"])
            if getattr(pre, "rejected", False):
                continue
            np.save(target, embedder.embed(pre.samples, int(pre.sample_rate)).astype(np.float32))
            if n % 500 == 0:
                print(f"  {n}/{len(rows)} augmented copies embedded", flush=True)
    print(f"embeddings in {out.relative_to(ROOT)} ({time.time() - started:.0f}s)")


def load_split(variant: str, split: str, *, include_augmented: bool = False,
               backbone: str = "cnn14"):
    rows = [r for r in load_manifest(split)]
    if include_augmented and split == "train":
        rows += [r for r in augmented_rows() if r["dataset_split"] == "train"]
    ids, X, y = [], [], []
    for r in rows:
        path = emb_dir(variant, backbone) / f"{r['audio_id']}.npy"
        if not path.exists():
            continue
        ids.append(r["audio_id"])
        X.append(np.load(path))
        y.append(r["class_label"])
    return ids, np.asarray(X, dtype=np.float64), np.asarray(y)


# candidates

def candidates() -> list[tuple[str, dict, object]]:
    """(family, params, unfitted estimator). Every estimator exposes predict_proba."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    out = []
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        out.append(("logreg", {"C": C}, make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))))
    for C in (1.0, 3.0, 10.0):
        out.append(("svm_rbf", {"C": C}, make_pipeline(
            StandardScaler(),
            SVC(C=C, gamma="scale", probability=True, class_weight="balanced",
                random_state=SEED))))
    for alpha in (1e-3, 1e-2, 1e-1):
        out.append(("mlp", {"hidden": 512, "alpha": alpha}, make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(512,), alpha=alpha, max_iter=400,
                          early_stopping=True, validation_fraction=0.1,
                          random_state=SEED))))
    return out


def selection_score(macro_f1: float, critical_recall: float) -> float:
    return 0.5 * macro_f1 + 0.5 * critical_recall


def score(estimator, X, y, class_names, critical, *, split: str, name: str):
    from src.training.evaluation import compute_metrics

    proba = estimator.predict_proba(X)
    order = list(estimator.classes_)
    pred = [order[i] for i in proba.argmax(axis=1)]
    top2 = [[order[i] for i in np.argsort(p)[::-1][:2]] for p in proba]
    return compute_metrics(list(y), pred, class_names, critical, model_name=name,
                           split=split, top2=top2), proba


def select(variant: str, include_augmented: bool, backbone: str = "cnn14") -> dict:
    class_names, critical = class_config()
    _, Xtr, ytr = load_split(variant, "train", include_augmented=include_augmented, backbone=backbone)
    _, Xva, yva = load_split(variant, "val", backbone=backbone)
    if len(Xtr) < 2000 or len(Xva) < 400:
        raise SystemExit(f"embeddings incomplete: train={len(Xtr)} val={len(Xva)}; run `embed` first")
    rows = []
    for family, params, est in candidates():
        t0 = time.time()
        est.fit(Xtr, ytr)
        fit_s = time.time() - t0
        t0 = time.time()
        res, _ = score(est, Xva, yva, class_names, critical, split="val", name=family)
        per_clip_ms = (time.time() - t0) / len(Xva) * 1000
        rows.append({"family": family, "params": params, "val_accuracy": res.accuracy,
                     "val_macro_f1": res.macro_f1, "val_critical_recall": res.critical_recall,
                     "val_critical_recall_by_class": res.critical_recall_by_class,
                     "val_min_critical_recall": min(res.critical_recall_by_class.values()),
                     "selection_score": selection_score(res.macro_f1, res.critical_recall),
                     "fit_seconds": round(fit_s, 2), "predict_ms_per_clip": round(per_clip_ms, 3)})
        print(f"{family:8s} {json.dumps(params):32s} acc={res.accuracy:.4f} "
              f"f1={res.macro_f1:.4f} crit={res.critical_recall:.4f} fit={fit_s:.1f}s", flush=True)
    rows.sort(key=lambda r: (-round(r["selection_score"] / 0.005), r["predict_ms_per_clip"]))
    best_by_family = {}
    for r in rows:
        best_by_family.setdefault(r["family"], r)
    doc = {"backbone": backbone, "variant": variant, "augmented": include_augmented,
           "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
           "criterion": "0.5*val_macro_f1 + 0.5*val_critical_recall; ties (0.005) -> faster",
           "selected": rows[0], "best_by_family": best_by_family, "all": rows,
           "decided_at": datetime.now(timezone.utc).isoformat()}
    METRICS.mkdir(parents=True, exist_ok=True)
    tag = run_tag(backbone, variant, include_augmented)
    (METRICS / f"transfer_selection_{tag}.json").write_text(json.dumps(doc, indent=2) + "\n")
    print("selected:", rows[0]["family"], rows[0]["params"])
    return doc


def run_tag(backbone: str, variant: str, include_augmented: bool) -> str:
    # CNN14 runs keep their original file names so the earlier evidence still resolves.
    prefix = "" if backbone == "cnn14" else f"{backbone}_"
    return f"{prefix}{variant}{'_aug' if include_augmented else ''}"


def build_estimator(family: str, params: dict):
    for fam, p, est in candidates():
        if fam == family and p == params:
            return est
    raise KeyError((family, params))


def final(variant: str, include_augmented: bool, out_dir: Path, backbone: str = "cnn14") -> None:
    """Refit the chosen configuration on train, score test once, save the bundle."""
    import joblib  # noqa: F401  (save_bundle uses it; import here to fail early)

    from src.inference.predictor import save_bundle

    mod = backbone_module(backbone)
    tag = run_tag(backbone, variant, include_augmented)
    selection = json.loads((METRICS / f"transfer_selection_{tag}.json").read_text())
    class_names, critical = class_config()
    ids_tr, Xtr, ytr = load_split(variant, "train", include_augmented=include_augmented,
                                  backbone=backbone)
    ids_te, Xte, yte = load_split(variant, "test", backbone=backbone)
    if len(Xte) != 450:
        raise SystemExit(f"expected 450 test embeddings, found {len(Xte)}")

    results = {}
    for family, choice in selection["best_by_family"].items():
        est = build_estimator(family, choice["params"])
        est.fit(Xtr, ytr)
        res, proba = score(est, Xte, yte, class_names, critical, split="test",
                           name=f"{backbone}+{family}")
        results[family] = res.to_dict()
        print(f"TEST {family:8s} acc={res.accuracy:.4f} f1={res.macro_f1:.4f} "
              f"crit={res.critical_recall:.4f} {res.critical_recall_by_class}", flush=True)
        if family == selection["selected"]["family"]:
            chosen, chosen_res, chosen_proba = est, res, proba

    train_hash = hashlib.sha256("\n".join(sorted(ids_tr)).encode()).hexdigest()
    name = f"{backbone}_embedding+{selection['selected']['family']}"
    if backbone == "ast":
        pretrained = {"pretrained_checkpoint": mod.AST_MODEL, "pretrained_revision": mod.AST_REVISION}
    else:
        pretrained = {"pretrained_checkpoint": mod.CHECKPOINT_NAME,
                      "pretrained_sha256": mod.CHECKPOINT_SHA256}
    meta = {
        "model_family": "transfer_learning",
        "backbone": backbone,
        "embedding_version": mod.EMBEDDING_VERSION,
        **pretrained,
        "preprocessing_variant": variant,
        "preprocessing_overrides": VARIANTS[variant],
        "augmented_training_copies": include_augmented,
        "selection": selection["selected"],
        "selection_criterion": selection["criterion"],
        "n_train_rows": len(ids_tr),
        "train_ids_sha256": train_hash,
        "seed": SEED,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_results_by_family": {k: {m: v[m] for m in ("accuracy", "macro_f1", "critical_recall")}
                                   for k, v in results.items()},
    }
    save_bundle(chosen, out_dir, class_names=class_names, feature_version=mod.EMBEDDING_VERSION,
                feature_columns=mod.embedding_columns(), model_name=name,
                model_version=f"2.0.0-{tag}", metrics=chosen_res.to_dict(), metadata=meta)
    (METRICS / f"transfer_test_{tag}.json").write_text(json.dumps(
        {"selected_family": selection["selected"]["family"], "results": results}, indent=2) + "\n")
    order = list(chosen.classes_)
    with (METRICS / f"transfer_test_predictions_{tag}.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["audio_id", "actual", "predicted", "confidence"] + order)
        for aid, actual, p in zip(ids_te, yte, chosen_proba):
            writer.writerow([aid, actual, order[int(p.argmax())], f"{p.max():.4f}"]
                            + [f"{v:.4f}" for v in p])
    print(f"saved {name} to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN14 transfer-learning trainer")
    parser.add_argument("stage", choices=["embed", "select", "final"])
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="current")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="cnn14")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--augmented", action="store_true")
    parser.add_argument("--out", default="python_models/best_transfer")
    args = parser.parse_args()
    if args.stage == "embed":
        embed(args.variant, args.splits, include_augmented=args.augmented, backbone=args.backbone)
    elif args.stage == "select":
        select(args.variant, args.augmented, args.backbone)
    else:
        final(args.variant, args.augmented, ROOT / args.out, args.backbone)


if __name__ == "__main__":
    main()
