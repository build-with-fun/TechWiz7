"""Train and compare the classical models -- SRS Step 7, FR xxiii-xxvi, NFR floor.

Owner: bilal.

WHAT IT DOES, IN ORDER
----------------------
1. Reads the frozen split (``audio_dataset/manifest_with_split.csv``). Never re-splits.
2. Extracts the locked 254-column feature matrix for train / val / test, cached on disk.
3. Tunes every classical candidate plus its critical-class-weighted variant on VALIDATION.
4. Runs 5-fold stratified CV on the winner over training rows only, for a variance estimate.
5. Refits the winner on train+val, then scores the untouched test split exactly once.
6. Saves the winner through ``src.inference.predictor.save_bundle`` so the web app loads it
   by the same code path as any other model, plus a metrics artefact and a confusion matrix
   PNG per candidate under ``python_models/metrics/``.

USAGE
-----
    # full run (needs the real dataset on disk)
    .venv/bin/python python_models/train_classical.py

    # fast smoke run on a tiny generated corpus -- proves the whole path in ~2 minutes
    .venv/bin/python python_models/train_classical.py --smoke

    # one candidate only, for a quick iteration while developing
    .venv/bin/python python_models/train_classical.py --candidates xgboost

THE RULE THIS SCRIPT WILL NOT BREAK
-----------------------------------
The test split is read once, at the end, after selection is final. ``select`` asserts this
itself (``assert_no_test_peeking``), and ``finalize`` re-checks the test rows against the
frozen split through lorena's ``assert_reportable_split``. There is no flag to relax either:
an honest number that took longer beats a better-looking number that an evaluator can
disprove with two questions.

A CHEAP MODEL THAT MEETS THE FLOOR IS A LEGITIMATE WINNER
---------------------------------------------------------
The selection criterion is validation macro-F1 subject to the critical-recall floor, but the
report records fit time and inference latency next to every score. If the SVM matches
XGBoost's F1 at a tenth of the inference cost, that is the model the app should serve, and
the comparison table will show why.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from python_models import classical, dataset, tuning  # noqa: E402

DEFAULT_MANIFEST = "audio_dataset/manifest_with_split.csv"
METRICS_DIR = "python_models/metrics"
BEST_DIR = "python_models/best_model"
FEATURE_CACHE = "python_models/.cache/features_{version}.json"
MODEL_VERSION = "1.0.0"


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def load_classes_config() -> tuple[list[str], list[str]]:
    """``(class_names, critical_classes)`` from config/classes.json -- never hard-coded.

    The SRS says evaluators may demand a changed category live, so the class list is read
    from the same file the app reads. Duplicating it here would mean a new class added to
    the config silently trains a model that cannot predict it.
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
    """The SRS performance floors, from config so they are auditable next to the results."""
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
# Money path: candidates -> selection -> test
# --------------------------------------------------------------------------------------


def build_candidate_grid(
    names: list[str],
    class_names: list[str],
    critical: list[str],
    *,
    weight_boost: float = 2.0,
    with_weighted: bool = True,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, dict[str, float]]]:
    """Every ``(candidate_name, params)`` to try, and the class weights each one uses.

    Each model appears twice: unweighted, and with the critical classes boosted. That makes
    "did weighting help?" a measured question with both answers in the comparison table,
    rather than an assumption baked into every model.
    """
    weights = classical.critical_class_weights(class_names, critical, boost=weight_boost)
    grid: list[tuple[str, dict[str, Any]]] = []
    weight_map: dict[str, dict[str, float]] = {}

    for name in names:
        spec = classical.get_candidate(name)
        for params in _expand_grid(spec.param_grid):
            base = {"_candidate": name, **params}
            tag = f"{name}{_param_tag(params)}"
            grid.append((tag, base))
            if with_weighted:
                wtag = f"{tag}+cw"
                # The weights travel separately from the params (class_weight for some
                # estimators, sample_weight for others), so they are looked up per candidate
                # at fit time rather than smuggled through the search space.
                grid.append((wtag, {**base, "_weighted": True}))
                weight_map[wtag] = weights
    return grid, weight_map


def _expand_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product of a param grid, as a list of dicts (small grids only)."""
    import itertools

    if not grid:
        return [{}]
    keys = list(grid)
    combos = itertools.product(*(grid[k] for k in keys))
    return [dict(zip(keys, values)) for values in combos]


def _param_tag(params: dict[str, Any]) -> str:
    parts = [
        f"{k}={v}"
        for k, v in sorted(params.items())
        if not k.startswith("_") and v is not None
    ]
    return ("[" + ",".join(parts) + "]") if parts else ""


def _resolve_spec(tag: str) -> classical.CandidateSpec:
    """Map a tagged candidate name back to its spec (``xgboost[max_depth=6]+cw`` -> xgboost)."""
    base = tag.split("[")[0]
    return classical.get_candidate(base)


def run_training(
    manifest_path: str,
    *,
    candidate_names: list[str] | None = None,
    seeds: tuple[int, ...] = (0,),
    weight_boost: float = 2.0,
    with_weighted: bool = True,
    cv_folds: int = 5,
    allow_weighting: bool = False,
    tag: str = "",
) -> dict[str, Any]:
    """Full train/compare/save run. Returns the summary dict it also writes to disk."""
    class_names, critical = load_classes_config()
    floors = load_floors()
    names = candidate_names or list(classical.CLASSICAL_CANDIDATES)

    print("=" * 78)
    print("SonicSentinel AI -- classical model training (SRS Step 7)")
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
    # Test rows go through the same builder as train/val, but augmented copies are dropped:
    # scoring a model on an augmented variant of its own training clip is not a test.
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
        log_path=REPO_ROOT / METRICS_DIR / "tuning_trials.jsonl" if not tag else None,
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
        names, class_names, critical, weight_boost=weight_boost, with_weighted=with_weighted
    )
    print(f"\n[tune] {len(grid)} candidate configurations x {len(seeds)} seed(s)")
    selection = protocol.select(grid, class_weights=None, split_used_for_selection="val")
    selection.rationale = selection.rationale.replace(
        "class weights None", "per-candidate"
    )
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
    winner_spec = _resolve_spec(selection.winner)
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

    all_results = {selection.winner: final.result}
    best_dir = REPO_ROOT / BEST_DIR if not tag else REPO_ROOT / BEST_DIR / tag
    classical.feature_importances(final_estimator, columns)

    out_dir = classical_save(
        final_estimator,
        best_dir,
        class_names=class_names,
        feature_version=extractor.feature_version,
        columns=columns,
        model_name=f"classical_{selection.winner}",
        model_version=MODEL_VERSION,
        metrics=final.result.to_dict(),
        metadata={
            "algorithm": selection.winner,
            "spec_notes": winner_spec.notes,
            "class_weights": winner_weights,
            "fitted_on": final.fitted_on,
            "n_features": len(columns),
            "function": "python_models.classical",
            "feature_importances": classical.feature_importances(final_estimator, columns),
            "cv": cv,
        },
    )
    print(f"\n[save] best model -> {_rel(out_dir)}")

    # Per-candidate metrics artefact + confusion matrix PNG.
    per_candidate = _per_candidate_metrics(
        protocol, selection, class_names, critical
    )
    artifact = {
        "model_family": "classical",
        "model_version": MODEL_VERSION,
        "python": sys.version.split()[0],
        "classes": class_names,
        "critical_classes": critical,
        "n_features": len(columns),
        "feature_version": extractor.feature_version,
        "floors": floors,
        "manifest": str(manifest_path),
        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
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
    metrics_path = metrics_dir / f"classical_metrics{suffix}.json"
    metrics_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"[save] metrics    -> {_rel(metrics_path)}")

    comparison_path = metrics_dir / f"classical_comparison{suffix}.csv"
    _write_comparison_csv(comparison_path, per_candidate, floors)
    print(f"[save] comparison -> {_rel(comparison_path)}")

    _plot_confusion(
        artifact["test"]["confusion_matrix"],
        class_names,
        metrics_dir / f"confusion_matrix_winner{suffix}.png",
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
# Fit / score callables handed to the harness
# --------------------------------------------------------------------------------------


def _fit_callable(X, y, params, seed, class_weights):
    """Wrap the zoo so the harness can treat a sklearn pipeline and a CNN identically.

    The candidate's family name travels inside ``params`` as ``_candidate`` (underscore
    keys never reach the estimator's constructor), so one callable serves every model in
    the grid and the harness sees a plain ``fit(X, y) -> estimator``.
    """
    resolved = {k: v for k, v in params.items() if not k.startswith("_")}
    name = str(params.get("_candidate") or "")
    if not name:
        raise ValueError(
            "params must carry '_candidate' so the zoo knows which estimator to build; "
            "got keys: " + ", ".join(sorted(params))
        )
    spec = classical.get_candidate(name)
    estimator = classical.build_estimator(spec, {**resolved, "_seed": seed}, seed=seed)
    return classical.fit_estimator(estimator, X, y, class_weights=class_weights)


def _base_name(tag: str) -> str:
    return tag.split("[")[0].replace("+cw", "")


def _predict_proba_callable(estimator, X) -> np.ndarray:
    proba = estimator.predict_proba(X)
    return np.asarray(proba)


# --------------------------------------------------------------------------------------
# Saving and reporting helpers
# --------------------------------------------------------------------------------------


def classical_save(estimator, out_dir, **kwargs):
    from src.inference.predictor import save_bundle

    return save_bundle(estimator, out_dir, **kwargs)


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
    """Small tally helper for the artefact's rejection summaries."""
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _per_candidate_metrics(protocol, selection, class_names, critical) -> list[dict[str, Any]]:
    """Validation metrics for every candidate tried, winner first -- the tuning table."""
    rows: list[dict[str, Any]] = []
    for trial in selection.ranked:
        rows.append(
            {
                "candidate": trial.candidate,
                "params": trial.params,
                "val_accuracy": trial.val_accuracy,
                "val_macro_f1": trial.val_macro_f1,
                "val_critical_recall": trial.val_critical_recall,
                "fit_seconds": round(trial.fit_seconds, 2),
                "selected": trial.candidate == selection.winner,
            }
        )
    return rows


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]], floors) -> None:
    import csv

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "candidate",
                "val_accuracy",
                "val_macro_f1",
                "val_critical_recall",
                "fit_seconds",
                "selected",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate": row["candidate"],
                    "val_accuracy": f"{row['val_accuracy']:.4f}",
                    "val_macro_f1": f"{row['val_macro_f1']:.4f}",
                    "val_critical_recall": f"{row['val_critical_recall']:.4f}",
                    "fit_seconds": row["fit_seconds"],
                    "selected": row["selected"],
                }
            )


def _plot_confusion(matrix, class_names, out_path: Path, title: str) -> None:
    """Confusion matrix PNG generated from the real matrix, not drawn by hand."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib unavailable; skipping confusion matrix PNG")
        return

    data = np.asarray(matrix, dtype=np.float64)
    # Row-normalised so classes with different support are comparable by eye.
    row_sums = data.sum(axis=1, keepdims=True)
    normalised = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    image = ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if data[i, j]:
                ax.text(
                    j,
                    i,
                    f"{int(data[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if normalised[i, j] > 0.5 else "black",
                )
    fig.colorbar(image, ax=ax, label="row-normalised share")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[save] confusion    -> {_rel(out_path)}")


def _measure_latency(estimator, X: np.ndarray, repeats: int = 5) -> dict[str, float]:
    """Single-row inference latency on this CPU -- the SRS's 3-second live budget is real."""
    import time as _time

    row = X[:1]
    # Warm up (first call pays lazy imports / thread-pool setup, not real inference).
    estimator.predict_proba(row)
    samples: list[float] = []
    for _ in range(repeats):
        started = _time.perf_counter()
        estimator.predict_proba(row)
        samples.append((_time.perf_counter() - started) * 1000.0)
    arr = np.asarray(samples)
    return {
        "single_row_mean_ms": float(arr.mean()),
        "single_row_max_ms": float(arr.max()),
        "repeats": repeats,
    }


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------
# Smoke mode
# --------------------------------------------------------------------------------------


def make_smoke_manifest(out_root: Path, per_class: int, seed: int) -> Path:
    """Generate a tiny corpus and freeze its split, using the SAME two steps as the real run.

    Smoke mode is not a substitute for the real run and says so: it uses a dozen originals
    per class, so the metrics carry no meaning whatsoever. Its only job is to prove that
    extraction, tuning, saving, loading and reporting all work before anyone spends forty
    minutes on the real dataset -- and it deliberately goes through the same
    ``generate_corpus.py`` then ``build_split.py`` path the real corpus uses, so a
    difference between smoke and real is a data problem, not a code path only smoke takes.

    The corpus is generated *inside* ``audio_dataset/smoke/`` with every manifest filename
    prefixed ``smoke/``, so the ordinary path resolver finds the files with no special case
    in the reader. It is removed again once the smoke run finishes.
    """
    import csv
    import shutil
    import subprocess

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] generating {per_class} originals/class into {out_root}")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "data" / "generate_corpus.py"),
            "--per-class",
            str(per_class),
            "--out-root",
            str(out_root),
            "--seed",
            str(seed),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    generated = out_root / "manifest_generated.csv"
    if not generated.exists():
        raise SystemExit(f"generator did not write {generated}")

    # Prefix the relative filenames so they resolve under audio_dataset/ like the real corpus.
    with generated.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        filename = str(row.get("filename", ""))
        if filename and not filename.startswith(out_root.name + "/"):
            row["filename"] = f"{out_root.name}/{filename}"
    with generated.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Freeze the split with the team's own builder -- the same one the real run uses.
    # It always writes the merged CSV to the canonical repo path, so the real manifest is
    # moved aside for the duration and restored afterwards: a smoke run must never be able
    # to leave a 120-clip manifest where the 3,000-clip one belongs.
    canonical = REPO_ROOT / "audio_dataset" / "manifest_with_split.csv"
    split_manifest = out_root / "manifest_with_split.csv"
    saved_canonical: bytes | None = canonical.read_bytes() if canonical.exists() else None

    try:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "audio_dataset" / "build_split.py"),
                "--manifest",
                str(generated),
                "--out",
                str(out_root / "split.json"),
                "--ids-dir",
                str(out_root / "ids"),
                "--force",
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        if not canonical.exists():
            raise SystemExit(
                f"build_split did not write {canonical}; the smoke run cannot continue"
            )
        split_manifest.write_bytes(canonical.read_bytes())
    finally:
        if saved_canonical is not None:
            canonical.write_bytes(saved_canonical)
        elif canonical.exists():
            canonical.unlink()

    with split_manifest.open(newline="", encoding="utf-8") as fh:
        n_rows = sum(1 for _ in csv.DictReader(fh))
    print(f"[smoke] manifest -> {split_manifest} ({n_rows} rows with frozen splits)\n")
    return split_manifest


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and compare the classical models.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="manifest CSV with a dataset_split column")
    parser.add_argument("--candidates", nargs="*", default=None, help="subset of candidate names")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0], help="random seeds to try")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--weight-boost", type=float, default=2.0)
    parser.add_argument("--no-weighted", action="store_true", help="skip critical-class-weighted variants")
    parser.add_argument("--smoke", action="store_true", help="generate a tiny corpus and run on it")
    parser.add_argument("--smoke-per-class", type=int, default=6)
    parser.add_argument("--tag", default="", help="suffix for output files (keeps runs separate)")
    args = parser.parse_args(argv)

    manifest = args.manifest
    tag = args.tag
    smoke_root: Path | None = None
    if args.smoke:
        smoke_root = REPO_ROOT / "audio_dataset" / "smoke"
        manifest = str(
            make_smoke_manifest(smoke_root, per_class=args.smoke_per_class, seed=1234)
        )
        tag = tag or "smoke"
        # A handful of originals per class cannot support 5 folds across 10 classes.
        args.cv_folds = min(args.cv_folds, 2)
        print(
            f"[smoke] per-class={args.smoke_per_class} => metrics are NOT meaningful; "
            "this proves the path only\n"
        )

    try:
        started = time.perf_counter()
        artifact = run_training(
            manifest,
            candidate_names=args.candidates,
            seeds=tuple(args.seeds),
            weight_boost=args.weight_boost,
            with_weighted=not args.no_weighted,
            cv_folds=args.cv_folds,
            tag=tag,
        )
    finally:
        if smoke_root is not None:
            import shutil

            shutil.rmtree(smoke_root, ignore_errors=True)
            print(f"[smoke] cleaned up {smoke_root}")

    print(f"\n[total] {time.perf_counter() - started:.1f}s")
    print(f"[total] meets_floors={artifact['meets_floors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
