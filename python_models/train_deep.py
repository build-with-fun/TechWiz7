"""Deep model training for SonicSentinel AI -- SRS Step 7, FR xxiii-xxvi.

Owner: nadia.

WHAT THIS IS
------------
The deep-model twin of ``python_models/train_classical.py``.  Both scripts run the SAME
8-step protocol against the SAME frozen manifest and the SAME 254-column feature matrix, so
a row in one comparison table is directly comparable to a row in the other.

The 8 steps, identical in intent to the classical script:

    1. load the frozen split        (dataset.load_all_splits -- the split is NEVER recomputed)
    2. build the feature matrix     (tuning.build_feature_matrix, one shared cache)
    3. attach data to the protocol  (tuning.TuningProtocol, disjointness asserted)
    4. select on validation         (protocol.select -- split_used_for_selection="val")
    5. cross-validate the winner    (tuning.cross_validate_selected)
    6. refit on train+val, test once (tuning.finalize -- test scored exactly once)
    7. save the bundle              (save_bundle -> model.joblib, loadable by the app)
    8. verdict                      (result.meets_floors -- printed whether or not it passes)

WHY THE SAME PATH MATTERS
-------------------------
The SRS asks for "at least three models trained and compared" and for a comparison report
between the Python model and the Teachable Machine model.  Comparing a CNN against a
support-vector machine is only meaningful if both were selected on the same validation rows
and scored on the same untouched test rows through the same harness -- which is what
``tuning.py`` enforces.  ``select`` refuses to run with ``split_used_for_selection`` set to
anything but a training-side split, and ``finalize`` re-checks every test row's provenance,
so a test clip cannot quietly become a training clip either here or in the classical run.

THE DEEP CANDIDATES (python_models/deep.py)
-------------------------------------------
    cnn1d     1-D CNN over the mel frequency axis
    crnn      CNN + bidirectional GRU over frequency, recurrent pooling
    transfer  frozen ImageNet MobileNetV3Small backbone + a new head on our spectra

All three consume ``(n, 254)`` -- the locked vector the app's inference path feeds an
estimator -- and all three expose the sklearn-shaped surface (``fit`` / ``predict_proba`` /
``classes_``) the harness and ``save_bundle`` expect.  No network access is needed at train
or inference time: the ImageNet weights are cached under ``~/.keras/models`` and travel
inside the saved bundle.

TIME BUDGET
-----------
This is a CPU box with four threads and a competition deadline, so the grids are small by
design -- one or two points per axis.  A sweep that does not finish produces no model at all,
which is worse than a single well-chosen point per family.  ``--epochs`` scales every model's
training time linearly, so a shorter run is one flag away when the clock is short.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = "audio_dataset/manifest_with_split.csv"
METRICS_DIR = "python_models/metrics"
BEST_DIR = "python_models/best_model"
#: Shared with train_classical.py.  A feature cache keyed by audio id + feature version, so
#: the deep and classical runs do not extract the same 3,000 files twice.
FEATURE_CACHE = "python_models/.cache/features_{version}.json"
MODEL_VERSION = "1.0.0"

import python_models.deep as deep  # noqa: E402  (module import after constants for readability)
from python_models import dataset  # noqa: E402
from python_models import tuning  # noqa: E402


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def load_classes_config() -> tuple[list[str], list[str]]:
    """``(class_names, critical_classes)`` from config/classes.json -- the same file the app reads.

    Duplicating the list here would mean a class added to the config trains a model that
    cannot predict it, and the label encoder at load time would then disagree with the app.
    """
    path = REPO_ROOT / "config" / "classes.json"
    if not path.exists():
        raise SystemExit(f"config/classes.json not found at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    names = [str(c["name"]) for c in config["classes"]]
    critical = [str(c) for c in config.get("critical_classes", [])]
    if not names:
        raise SystemExit("config/classes.json defines no classes")
    return names, critical


def load_floors() -> dict[str, float]:
    """The SRS floors from config/thresholds.json, so they sit auditable next to the results."""
    path = REPO_ROOT / "config" / "thresholds.json"
    floors = {"accuracy": 0.85, "macro_f1": 0.80, "critical_recall": 0.85}
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        model_floors = config.get("model_floors") or {}
        for key in floors:
            if key in model_floors:
                floors[key] = float(model_floors[key])
    return floors


# --------------------------------------------------------------------------------------
# Candidate grid
# --------------------------------------------------------------------------------------


def _expand_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product of a param grid, as a list of dicts (small grids only)."""
    import itertools

    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*[grid[k] for k in keys])]


def _param_tag(params: dict[str, Any]) -> str:
    """Short tag for the comparison table, e.g. ``[channels=32]``."""
    if not params:
        return ""
    inner = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"[{inner}]"


def build_candidate_grid(
    names: list[str],
    class_names: list[str],
    critical: list[str],
    *,
    weight_boost: float = 2.0,
    with_weighted: bool = True,
    epochs_override: int | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, dict[str, float]]]:
    """Every ``(name, params)`` the deep run will fit, plus the weights each uses.

    Mirrors ``train_classical.build_candidate_grid`` exactly, so the deep and classical
    comparison tables have the same shape and both can be asked "did weighting help?".
    """
    weights = deep.sample_weights_for(class_names, {})  # placeholder, replaced below
    cw = {c: weight_boost for c in critical}
    weight_map: dict[str, dict[str, float]] = {}
    grid: list[tuple[str, dict[str, Any]]] = []

    for name in names:
        spec = deep.get_candidate(name)
        for params in _expand_grid(spec.param_grid):
            resolved = dict(params)
            if epochs_override is not None:
                resolved["epochs"] = int(epochs_override)
            base = {"_candidate": name, **resolved}
            tag = f"{name}{_param_tag(resolved)}"
            grid.append((tag, base))
            if with_weighted:
                wtag = f"{tag}+cw"
                grid.append((wtag, {**base, "_weighted": True}))
                weight_map[wtag] = cw
    return grid, weight_map


# --------------------------------------------------------------------------------------
# Fit / score callables handed to the harness
# --------------------------------------------------------------------------------------


def _fit_callable(X, y, params, seed, class_weights):
    """Build and fit one deep candidate.

    ``_candidate`` travels inside ``params`` (underscore keys never reach the estimator's
    constructor), so the harness sees a uniform ``fit(X, y) -> estimator`` for a CNN, a GRU
    and a MobileNet backbone alike.
    """
    name = str(params.get("_candidate") or "")
    if not name:
        raise ValueError(
            "params must carry '_candidate' so the zoo knows which estimator to build; "
            "got keys: " + ", ".join(sorted(params))
        )
    resolved = {k: v for k, v in params.items() if not k.startswith("_")}
    spec = deep.get_candidate(name)
    estimator = deep.build_estimator(spec, resolved, seed=seed)
    return deep.fit_estimator(estimator, X, y, class_weights=class_weights)


def _predict_proba_callable(estimator, X) -> np.ndarray:
    return np.asarray(estimator.predict_proba(X))


# --------------------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------------------


def _base_name(tag: str) -> str:
    return tag.split("[")[0].replace("+cw", "")


def print_result(result, floors) -> None:
    print(f"  accuracy          : {result.accuracy:.4f}  (floor {floors['accuracy']:.2f})")
    print(f"  macro F1          : {result.macro_f1:.4f}  (floor {floors['macro_f1']:.2f})")
    print(
        f"  critical recall   : {result.critical_recall:.4f}  "
        f"(floor {floors['critical_recall']:.2f})"
    )
    print(f"  macro precision   : {result.macro_precision:.4f}")
    print(f"  macro recall      : {result.macro_recall:.4f}")
    print(f"  top-2 accuracy    : {result.top2_accuracy:.4f}")
    worst = result.worst_classes(3)
    if worst:
        print("  weakest classes   :")
        for entry in worst:
            print(
                f"    {entry['class_name']:<26} recall={entry['recall']:.3f} "
                f"f1={entry['f1']:.3f} support={entry['support']}"
            )


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _per_candidate_metrics(
    protocol, selection, class_names: list[str], critical: list[str]
) -> list[dict[str, Any]]:
    """One row per candidate for the comparison CSV -- what was tried and how it scored.

    The deep models have no feature importances (a CNN's first-layer filters are not a
    per-column attribution), so that column is reported as empty rather than fabricated:
    ``classical.feature_importances`` makes the same choice for models that expose nothing.
    """
    rows: list[dict[str, Any]] = []
    for tag, trial in protocol.trials_by_candidate.items():
        entry = {
            "candidate": tag,
            "family": _base_name(tag),
            "weighted": tag.endswith("+cw"),
            "backend": deep.get_candidate(_base_name(tag)).backend,
            "spec_notes": deep.get_candidate(_base_name(tag)).notes,
            "params": {k: v for k, v in trial.params.items() if not k.startswith("_")},
            "seed": trial.seed,
            "fit_seconds": round(float(trial.fit_seconds), 2),
            "val_macro_f1": round(float(trial.val_macro_f1), 4),
            "val_accuracy": round(float(trial.val_accuracy), 4),
            "val_critical_recall": round(float(trial.val_critical_recall), 4),
            "feature_importances": {},
        }
        rows.append(entry)
    return rows


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]], floors) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate",
        "family",
        "backend",
        "weighted",
        "params",
        "seed",
        "fit_seconds",
        "val_macro_f1",
        "val_accuracy",
        "val_critical_recall",
        "feature_importances",
    ]
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "params": json.dumps(row.get("params", {}), default=str)})
    print(f"[save] comparison -> {path}")


def _plot_confusion(matrix, class_names, out_path: Path, title: str) -> None:
    """Confusion matrix of the winner on the test split, saved next to the metrics."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print(f"[save] matplotlib unavailable, skipping {out_path.name}")
        return
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value:
                ax.text(
                    j,
                    i,
                    str(int(value)),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=7,
                )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[save] confusion  -> {out_path}")


def _measure_latency(estimator, X: np.ndarray, repeats: int = 5) -> dict[str, float]:
    """Single-row inference latency, measured the same way as the classical script.

    The SRS gives live inference a 3 s budget and an upload 8 s; a deep model that wins on
    macro-F1 but takes 900 ms a row is a deployment problem worth seeing in the metrics.
    """
    return deep.inference_latency_ms(estimator, X, repeats=repeats)


def _rel(path: Path) -> str:
    """Repo-relative path for console output."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def deep_save(estimator, out_dir, **kwargs):
    """``save_bundle`` is the one writer the app's loader reads; this keeps a single format."""
    from src.inference.predictor import save_bundle

    return save_bundle(estimator, out_dir, **kwargs)


# --------------------------------------------------------------------------------------
# Money path
# --------------------------------------------------------------------------------------


def run_training(
    manifest_path: str,
    *,
    candidate_names: list[str] | None = None,
    seeds: tuple[int, ...] = (0,),
    weight_boost: float = 2.0,
    with_weighted: bool = True,
    cv_folds: int = 3,
    epochs_override: int | None = None,
    tag: str = "",
) -> dict[str, Any]:
    """Full train/compare/save run for the deep family. Returns the summary dict it writes.

    ``cv_folds`` defaults to 3, not the classical script's 5: a CRNN fold costs seconds and a
    MobileNet fold costs tens of seconds on this CPU, and CV is a variance estimate on the
    winner -- three folds still says something while leaving time to actually finish.
    """
    class_names, critical = load_classes_config()
    floors = load_floors()
    names = candidate_names or list(deep.DEEP_CANDIDATES)

    print("=" * 78)
    print("SonicSentinel AI -- deep model training (SRS Step 7)")
    print("=" * 78)
    print(f"  python      : {sys.version.split()[0]} ({platform.machine()})")
    print(f"  classes     : {len(class_names)}  critical: {len(critical)}")
    print(f"  candidates  : {', '.join(names)}")
    print(f"  weighting   : {'on' if with_weighted else 'off'} (boost x{weight_boost})")
    print(f"  floors      : {floors}")

    # -- 1. data ---------------------------------------------------------------------
    splits = dataset.load_all_splits(manifest_path)
    for name in ("train", "val", "test"):
        if name not in splits:
            raise SystemExit(
                f"manifest {manifest_path} has no {name!r} split; found {sorted(splits)}"
            )

    for split in splits.values():
        info = dataset.describe_split(split)
        imbalance = dataset.class_imbalance_report(split, class_names)
        print(
            f"  {split.name:<6}: {info['n_records']:>5} rows "
            f"({info['n_originals']} orig / {info['n_augmented']} aug), "
            f"{info['n_classes_present']}/{len(class_names)} classes"
        )
        if imbalance["absent_classes"]:
            print(
                f"          WARNING absent classes: {imbalance['absent_classes']} -- "
                "macro-F1 will be depressed by their zero recall, correctly."
            )

    train = dataset.training_records(splits["train"])
    val = dataset.training_records(splits["val"])
    # Augmented copies are dropped from the test set: scoring a model on a variant of its own
    # training clip is not a test, and the SRS comparison report needs unseen recordings.
    test = dataset.training_records(splits["test"], exclude_augmented=True)

    # -- 2. features -----------------------------------------------------------------
    from feature_extraction.features import FeatureExtractor

    extractor = FeatureExtractor()
    columns = list(extractor.columns)
    print(f"\n[features] {len(columns)} columns, version {extractor.feature_version}")

    cache_path = REPO_ROOT / FEATURE_CACHE.format(version=extractor.feature_version)
    if tag:
        cache_path = cache_path.with_name(cache_path.stem + f"_{tag}" + cache_path.suffix)

    started = time.perf_counter()
    unusable: list[dict[str, Any]] = []
    print(f"[features] train ({len(train)} rows)...")
    X_train, y_train, train_ids = tuning.build_feature_matrix(
        train, extractor, cache_path=cache_path, progress=True, unusable=unusable
    )
    print(f"[features] val ({len(val)} rows)...")
    X_val, y_val, val_ids = tuning.build_feature_matrix(
        val, extractor, cache_path=cache_path, progress=True, unusable=unusable
    )
    print(f"[features] test ({len(test)} rows)...")
    X_test, y_test, test_ids = tuning.build_feature_matrix(
        test, extractor, cache_path=cache_path, progress=True, unusable=unusable
    )
    print(f"[features] done in {time.perf_counter() - started:.1f}s")
    if unusable:
        by_split: dict[str, int] = {}
        for row in unusable:
            by_split[row["split"]] = by_split.get(row["split"], 0) + 1
        print(
            f"[features] WARNING: {len(unusable)} recording(s) failed the audio-quality "
            f"gate and are excluded from the matrix ({by_split}); "
            f"reasons: {sorted({r['reason'][:60] for r in unusable})[:3]}"
        )

    # -- 3. protocol -----------------------------------------------------------------
    protocol = tuning.TuningProtocol(
        class_names=class_names,
        critical_classes=critical,
        fit=_fit_callable,
        predict_proba_of=_predict_proba_callable,
        selection_metric="macro_f1",
        critical_recall_floor=floors.get("critical_recall", 0.85),
        seeds=seeds,
        log_path=REPO_ROOT / METRICS_DIR / "tuning_trials_deep.jsonl" if not tag else None,
    ).attach_data(
        X_train=X_train,
        y_train=y_train,
        train_ids=train_ids,
        X_val=X_val,
        y_val=y_val,
        val_ids=val_ids,
    )

    # -- 4. selection on validation --------------------------------------------------
    grid, weight_map = build_candidate_grid(
        names,
        class_names,
        critical,
        weight_boost=weight_boost,
        with_weighted=with_weighted,
        epochs_override=epochs_override,
    )
    print(f"\n[tune] {len(grid)} candidate configurations x {len(seeds)} seed(s)")
    selection = protocol.select(grid, class_weights=None, split_used_for_selection="val")
    selection.rationale = selection.rationale.replace("class weights None", "per-candidate")
    print(f"\n[tune] WINNER: {selection.winner}")
    print(f"       {selection.rationale}")

    # -- 5. CV on the winner ---------------------------------------------------------
    winner_weights = weight_map.get(selection.winner)
    cv = {}
    if cv_folds > 1:
        print(f"\n[cv] {cv_folds}-fold stratified CV on training rows (winner only)")
        cv = tuning.cross_validate_selected(
            protocol, selection, folds=cv_folds, class_weights=winner_weights
        )
        print(
            f"[cv] macro-F1 {cv['macro_f1_mean']:.4f} +/- {cv['macro_f1_std']:.4f} "
            f"across {cv['folds']} folds"
        )

    # -- 6. refit on train+val, score test ONCE --------------------------------------
    print("\n[final] refitting winner on train+val")
    winner_spec = deep.get_candidate(_base_name(selection.winner))
    # ``_candidate`` must survive into the refit -- it is how the zoo knows which family to
    # rebuild. Only the other underscore keys (internal bookkeeping) are dropped.
    refit_params = {
        k: v for k, v in selection.params.items() if k == "_candidate" or not k.startswith("_")
    }
    final_estimator = _fit_callable(
        np.vstack([X_train, X_val]),
        list(y_train) + list(y_val),
        refit_params,
        selection.seed,
        winner_weights,
    )
    final = tuning.finalize(
        protocol,
        selection,
        final_estimator,
        X_test=X_test,
        y_test=y_test,
        test_ids=test_ids,
        test_records=splits["test"].records,
        model_version=MODEL_VERSION,
        fitted_on="train+val",
    )
    print("\n[final] TEST SPLIT (scored once, never used for selection)")
    print_result(final.result, floors)

    # -- 7. save ---------------------------------------------------------------------
    metrics_dir = REPO_ROOT / METRICS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    # Feature importances are {} for deep models by design -- see _per_candidate_metrics.
    best_dir = REPO_ROOT / BEST_DIR if not tag else REPO_ROOT / BEST_DIR / tag
    out_dir = deep_save(
        final_estimator,
        best_dir,
        class_names=class_names,
        feature_version=extractor.feature_version,
        feature_columns=list(columns),
        model_name=f"deep_{selection.winner}",
        model_version=MODEL_VERSION,
        metrics=final.result.to_dict(),
        metadata={
            "algorithm": selection.winner,
            "spec_notes": winner_spec.notes,
            "class_weights": winner_weights,
            "fitted_on": final.fitted_on,
            "n_features": len(columns),
            "function": "python_models.deep",
            "feature_importances": {},
            "cv": cv,
        },
    )
    print(f"\n[save] best model -> {_rel(out_dir)}")

    per_candidate = _per_candidate_metrics(protocol, selection, class_names, critical)
    artifact = {
        "model_family": "deep",
        "model_version": MODEL_VERSION,
        "python": sys.version.split()[0],
        "classes": class_names,
        "critical_classes": critical,
        "n_features": len(columns),
        "feature_version": extractor.feature_version,
        "candidates_described": deep.describe_zoo(),
        "floors": floors,
        "manifest": str(manifest_path),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "matrix_rows": {
            "train": int(X_train.shape[0]),
            "val": int(X_val.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "split_integrity": {
            "train_val_overlap": len(set(train_ids) & set(val_ids)),
            "train_test_overlap": len(set(train_ids) & set(test_ids)),
            "val_test_overlap": len(set(val_ids) & set(test_ids)),
        },
        # Recordings present in the frozen split but rejected by the audio-quality gate.
        # Reported, never hidden: the comparison table's denominators come from here.
        "unusable_recordings": {
            "n": len(unusable),
            "by_split": _count_by(unusable, "split"),
            "by_class": _count_by(unusable, "class_label"),
            "rows": unusable,
        },
        "selection": selection.to_dict(),
        "cross_validation": cv,
        "winner": selection.winner,
        "winner_params": selection.params,
        "winner_class_weights": winner_weights,
        "test": final.result.to_dict(),
        "test_predictions": final.rows,
        "per_candidate": per_candidate,
        "inference_latency_ms": _measure_latency(final_estimator, X_test),
        "meets_floors": final.result.meets_floors(floors)[0],
    }
    metrics_path = metrics_dir / f"deep_metrics{suffix}.json"
    metrics_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"[save] metrics    -> {_rel(metrics_path)}")

    comparison_path = metrics_dir / f"deep_comparison{suffix}.csv"
    _write_comparison_csv(comparison_path, per_candidate, floors)

    _plot_confusion(
        artifact["test"]["confusion_matrix"],
        class_names,
        metrics_dir / f"confusion_matrix_deep{suffix}.png",
        f"{selection.winner} (test split, n={artifact['test']['n_records']})",
    )

    # -- 8. verdict ------------------------------------------------------------------
    ok, failures = final.result.meets_floors(floors)
    print("\n" + "=" * 78)
    if ok:
        print("RESULT: all SRS floors met on the untouched test split.")
    else:
        print("RESULT: SRS floors NOT met --")
        for failure in failures:
            print(f"  - {failure}")
    print("=" * 78)
    return artifact


# --------------------------------------------------------------------------------------
# Smoke manifest: the same two-step path the real corpus uses, at a tenth of the size
# --------------------------------------------------------------------------------------


def make_smoke_manifest(out_root: Path, per_class: int, seed: int) -> Path:
    """Generate a tiny corpus and freeze its split via the real corpus path.

    Reuses ``train_classical.make_smoke_manifest`` rather than duplicating it: the smoke run
    must exercise the same ``generate_corpus.py`` -> ``build_split.py`` path the real corpus
    uses, and a second copy of that logic would drift.  It deliberately produces metrics with
    no meaning -- its only job is to prove extraction, tuning, saving, loading and reporting
    all work before the real run spends the budget.
    """
    from python_models.train_classical import make_smoke_manifest as _make

    return _make(out_root, per_class, seed)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and compare the deep models.")
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="manifest CSV with a dataset_split column (default: the frozen one)",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=None,
        help="candidate subset (default: all of " + ", ".join(deep.DEEP_CANDIDATES) + ")",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="seeds to average over")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="stratified CV folds on the winner (0 to skip)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override the epoch count of every candidate (scales run time linearly)",
    )
    parser.add_argument(
        "--no-weighted",
        action="store_true",
        help="skip the critical-class-weighted variants",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny synthetic corpus -- proves the pipeline, metrics carry NO meaning",
    )
    parser.add_argument(
        "--smoke-per-class",
        type=int,
        default=6,
        help="originals per class in smoke mode (default 6)",
    )
    parser.add_argument("--tag", default="", help="suffix for output dirs/files")
    args = parser.parse_args(argv)

    if args.smoke:
        manifest = make_smoke_manifest(
            REPO_ROOT / "audio_dataset" / "smoke_deep", args.smoke_per_class, seed=17
        )
        manifest_path = str(manifest)
    else:
        manifest_path = str(args.manifest)

    run_training(
        manifest_path,
        candidate_names=args.candidates,
        seeds=tuple(args.seeds),
        with_weighted=not args.no_weighted,
        cv_folds=max(0, args.cv_folds),
        epochs_override=args.epochs,
        tag=args.tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
