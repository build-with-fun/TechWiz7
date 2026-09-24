#!/usr/bin/env python3
"""
SonicSentinel AI — canonical dataset split builder.

Owner: lorena.  Authority: SRS v1.0 Step 5, FR xvii, FR xviii, Deliverable 3.

PURPOSE
-------
Produce THE ONE train/validation/test assignment that *both* the Python model and the
Google Teachable Machine model are trained and evaluated on. SRS Step 5:

    "The same underlying split must be used for both models."
    "Validation and testing samples must not be used for either model's training."
    "All segments derived from one original recording must remain in the same dataset split."

If two models are judged on different splits, the comparison is worthless and the
competition integrity rules are broken. This file exists so that cannot happen by accident.

DESIGN DECISIONS (and why)
--------------------------
1. Assignment depends ONLY on `audio_id`, never on arrival order, file size or row order.
   The split is frozen *before* the audio is fetched, so a later download cannot bias it.
2. Ordering uses sha256(f"{seed}:{audio_id}"), not `random.shuffle`. Python's Mersenne
   Twister shuffle is stable in practice but is not a documented cross-version guarantee;
   a content hash is deterministic on every machine and every Python ever. An evaluator
   re-running this must get byte-identical output.
3. Stratified per class, so each class contributes 70/15/15. With 300 originals per class
   that is exactly 210/45/45 per class -> 2100/450/450 overall (SRS Hint).
4. Lineage: an augmented clip or a segment inherits its *parent recording's* split.
   Segments are therefore never split across train/test, which would be leakage.
5. Augmented rows are excluded from the original counts entirely (SRS: "Augmented
   recordings must not be counted as unique original clips").

Run:
    .venv/bin/python audio_dataset/build_split.py --manifest audio_dataset/manifest.csv --strict
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

SEED = 20260923
ALGORITHM = "sha256-order-v1"
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ("train", "val", "test")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "audio_dataset" / "manifest.csv"
DEFAULT_SPLIT_PATH = REPO_ROOT / "data" / "splits" / "split.json"
DEFAULT_IDS_DIR = REPO_ROOT / "data" / "splits"

# Manifest columns, in the frozen order documented in audio_dataset/manifest_schema.md
MANIFEST_COLUMNS = [
    "audio_id", "filename", "class_label", "source", "source_url", "licence", "author",
    "date_fetched", "duration_sec", "sampling_rate", "channels", "recording_environment",
    "recording_device", "approximate_distance", "original_or_augmented", "parent_audio_id",
    "segment_start_sec", "segment_end_sec", "sha256", "dataset_split",
]

REQUIRED_COLUMNS = [
    "audio_id", "filename", "class_label", "source", "licence", "author",
    "original_or_augmented", "dataset_split",
]

ORIGINAL = "original"
AUGMENTED = "augmented"


class ManifestError(Exception):
    """Raised when the manifest cannot produce a trustworthy split."""


# --------------------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------------------

def load_classes(classes_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the canonical class list from config/classes.json.

    Class names must never be hard-coded in modelling code: the SRS lists "adding a new
    sound category" as a surprise modification the evaluators may demand.
    """
    path = classes_path or (REPO_ROOT / "config" / "classes.json")
    with path.open(encoding="utf-8") as fh:
        cfg = json.load(fh)
    return {c["name"]: c for c in cfg["classes"]}


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read the manifest CSV and keep only the columns we understand."""
    if not manifest_path.exists():
        raise ManifestError(f"manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ManifestError("manifest is empty (no header row)")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            # A bare "missing columns" list sends people off renaming their whole
            # generator. Name the contract, the rename map, and an escape hatch.
            renames = {
                "sample_rate": "sampling_rate",
                "approx_distance_m": "approximate_distance",
                "distance_m": "approximate_distance",
                "sound_category": "class_label",
                "label": "class_label",
                "path": "filename",
                "augmented": "original_or_augmented",
                "split": "dataset_split",
                "hash": "sha256",
            }
            hints = [f"{k} -> {v}" for k, v in renames.items() if k in reader.fieldnames]
            raise ManifestError(
                "manifest is missing required columns: "
                f"{missing}\n"
                "The frozen contract is audio_dataset/manifest_schema.md "
                "(SRS Step 1 metadata + provenance).\n"
                + (f"Rename map for this file: {', '.join(hints)}\n" if hints else "")
                + "Do NOT hand-write dataset_split: audio_dataset/build_split.py is the only "
                "thing allowed to assign it — two split definitions means one of them leaks "
                "train data into test metrics.\n"
                "Escape hatch: extra columns are preserved in the output, so a generator "
                "only has to emit the required set, not exactly the required set."
            )
        rows = [dict(r) for r in reader]

    if not rows:
        raise ManifestError("manifest has a header but no data rows")
    return rows


def validate_records(records: list[dict[str, str]], classes: dict[str, dict[str, Any]]) -> None:
    """Structural validation. Raises ManifestError with ALL problems listed, not just the first."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    by_id: dict[str, dict[str, str]] = {}

    for i, rec in enumerate(records, start=2):  # line 1 is the header
        audio_id = (rec.get("audio_id") or "").strip()
        if not audio_id:
            problems.append(f"line {i}: empty audio_id")
            continue
        if audio_id in seen_ids:
            problems.append(f"line {i}: duplicate audio_id {audio_id!r}")
            continue
        seen_ids.add(audio_id)
        by_id[audio_id] = rec

        label = (rec.get("class_label") or "").strip()
        if label not in classes:
            problems.append(
                f"line {i} ({audio_id}): class_label {label!r} is not one of the "
                f"{len(classes)} configured classes"
            )

        if not (rec.get("filename") or "").strip():
            problems.append(f"line {i} ({audio_id}): empty filename")

        if not (rec.get("licence") or "").strip():
            problems.append(
                f"line {i} ({audio_id}): empty licence — an unlicensed clip may not "
                f"enter the dataset (SRS ethical-sourcing requirement)"
            )

        status = (rec.get("original_or_augmented") or "").strip().lower()
        if status not in (ORIGINAL, AUGMENTED):
            problems.append(
                f"line {i} ({audio_id}): original_or_augmented must be "
                f"{ORIGINAL!r} or {AUGMENTED!r}, got {status!r}"
            )
        elif status == AUGMENTED and not (rec.get("parent_audio_id") or "").strip():
            problems.append(
                f"line {i} ({audio_id}): augmented row has no parent_audio_id — its "
                f"lineage is unknown, so it cannot be placed in a split honestly"
            )

    # Parent references must resolve, and an augmented clip's parent must be an original.
    for rec in records:
        status = (rec.get("original_or_augmented") or "").strip().lower()
        parent = (rec.get("parent_audio_id") or "").strip()
        if status != AUGMENTED or not parent:
            continue
        if parent not in by_id:
            problems.append(
                f"{rec['audio_id']}: parent_audio_id {parent!r} is not in the manifest"
            )
        elif (by_id[parent].get("original_or_augmented") or "").strip().lower() == AUGMENTED:
            problems.append(
                f"{rec['audio_id']}: parent {parent!r} is itself augmented; chain the "
                f"lineage back to an original recording instead"
            )

    if problems:
        head = "\n  - ".join(problems[:50])
        more = f"\n  ... and {len(problems) - 50} more" if len(problems) > 50 else ""
        raise ManifestError(f"manifest validation failed ({len(problems)} problems):\n  - {head}{more}")


# --------------------------------------------------------------------------------------
# Deterministic ordering
# --------------------------------------------------------------------------------------

def _order_key(seed: int, audio_id: str) -> str:
    """Deterministic, platform- and version-independent ordering key."""
    return hashlib.sha256(f"{seed}:{audio_id}".encode("utf-8")).hexdigest()


def split_counts(n: int, ratios: tuple[float, float, float] = SPLIT_RATIOS) -> tuple[int, int, int]:
    """Split n items into train/val/test counts that always sum to n.

    With n=300 this returns exactly (210, 45, 45), matching the SRS Hint.
    """
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


# --------------------------------------------------------------------------------------
# Split construction
# --------------------------------------------------------------------------------------

def build_split(
    records: list[dict[str, str]],
    classes: dict[str, dict[str, Any]],
    seed: int = SEED,
) -> dict[str, Any]:
    """Assign every record to train/val/test. Pure function — no I/O, no globals mutated."""
    validate_records(records, classes)

    by_id = {r["audio_id"]: r for r in records}
    originals = [r for r in records if r["original_or_augmented"].strip().lower() == ORIGINAL]
    derived = [r for r in records if r["original_or_augmented"].strip().lower() == AUGMENTED]

    assignments: dict[str, dict[str, str]] = {}

    # --- Pass 1: originals, stratified per class -------------------------------------
    per_class_originals: dict[str, list[str]] = defaultdict(list)
    for rec in originals:
        per_class_originals[rec["class_label"].strip()].append(rec["audio_id"])

    counts_by_split: dict[str, Counter[str]] = {s: Counter() for s in SPLIT_NAMES}

    for label in sorted(per_class_originals):
        ids = sorted(per_class_originals[label], key=lambda a: _order_key(seed, a))
        n_train, n_val, _ = split_counts(len(ids))
        buckets = {
            "train": ids[:n_train],
            "val": ids[n_train:n_train + n_val],
            "test": ids[n_train + n_val:],
        }
        for split_name, bucket in buckets.items():
            for audio_id in bucket:
                assignments[audio_id] = {
                    "split": split_name,
                    "class_label": label,
                    "role": ORIGINAL,
                }
                counts_by_split[split_name][label] += 1

    # --- Pass 2: derived rows inherit their parent's split ---------------------------
    # Segments of a recording and augmented copies must never straddle train/test.
    for rec in derived:
        parent = rec["parent_audio_id"].strip()
        parent_split = assignments[parent]["split"]
        label = rec["class_label"].strip()
        if label != assignments[parent]["class_label"]:
            raise ManifestError(
                f"{rec['audio_id']}: class_label {label!r} differs from its parent "
                f"{parent}'s {assignments[parent]['class_label']!r}"
            )
        role = "segment" if (rec.get("segment_start_sec") or "").strip() else AUGMENTED
        assignments[rec["audio_id"]] = {
            "split": parent_split,
            "class_label": label,
            "role": role,
        }

    return {
        "algorithm": ALGORITHM,
        "seed": seed,
        "ratios": {"train": SPLIT_RATIOS[0], "val": SPLIT_RATIOS[1], "test": SPLIT_RATIOS[2]},
        "class_names": sorted(classes),
        "assignments": assignments,
        "counts": {
            "originals_by_split": {s: dict(counts_by_split[s]) for s in SPLIT_NAMES},
            "originals_totals": {s: sum(counts_by_split[s].values()) for s in SPLIT_NAMES},
            "derived_by_split": {
                s: sum(1 for a in assignments.values() if a["split"] == s and a["role"] != ORIGINAL)
                for s in SPLIT_NAMES
            },
        },
    }


def assert_strict(records: list[dict[str, str]], classes: dict[str, dict[str, Any]]) -> None:
    """--strict: refuse to freeze a split that cannot meet the SRS dataset floor."""
    originals = [r for r in records if r["original_or_augmented"].strip().lower() == ORIGINAL]
    per_class = Counter(r["class_label"].strip() for r in originals)

    problems = []
    for label in sorted(classes):
        want = int(classes[label].get("required_originals", 300))
        got = per_class.get(label, 0)
        if got != want:
            problems.append(f"{label}: {got} originals, expected exactly {want}")
    total = len(originals)
    if total != 3000:
        problems.append(f"total originals {total}, expected exactly 3000")

    if problems:
        raise ManifestError(
            "--strict was requested but the dataset does not meet the SRS floor:\n  - "
            + "\n  - ".join(problems)
            + "\n\nFetch the missing audio first, or run without --strict to produce a "
              "provisional split for development. A provisional split is NOT valid "
              "evidence for the final report."
        )


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------

def write_split(split: dict[str, Any], out_path: Path, ids_dir: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(split)
    payload["assignments"] = {k: payload["assignments"][k] for k in sorted(payload["assignments"])}
    payload["generated_by"] = "audio_dataset/build_split.py"

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    for split_name in SPLIT_NAMES:
        ids = sorted(a for a, v in payload["assignments"].items() if v["split"] == split_name)
        with (ids_dir / f"{split_name}_ids.txt").open("w", encoding="utf-8") as fh:
            fh.write("\n".join(ids) + ("\n" if ids else ""))


def write_manifest_with_split(records: list[dict[str, str]], split: dict[str, Any], out_path: Path) -> None:
    """Emit manifest + resolved dataset_split so downstream code never re-derives it."""
    assignments = split["assignments"]
    columns = [c for c in MANIFEST_COLUMNS if c in (records[0].keys() | {"dataset_split"})]
    extra = [c for c in records[0] if c not in columns]
    columns += extra

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["dataset_split"] = assignments[rec["audio_id"]]["split"]
            writer.writerow({c: row.get(c, "") for c in columns})


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the canonical SonicSentinel split.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--ids-dir", type=Path, default=DEFAULT_IDS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--strict", action="store_true",
                        help="require exactly 300 originals per class (3000 total)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        classes = load_classes()
        records = load_manifest(args.manifest)
        if args.strict:
            assert_strict(records, classes)
        split = build_split(records, classes, seed=args.seed)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.out.exists() and not args.force:
        print(
            f"REFUSING to overwrite the frozen split at {args.out}.\n"
            f"Re-run with --force only if the dataset genuinely changed — a moved split "
            f"invalidates every model already trained or evaluated against it.",
            file=sys.stderr,
        )
        return 3

    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    split["manifest_sha256"] = manifest_hash
    split["manifest_path"] = str(args.manifest.relative_to(REPO_ROOT)
                                 if args.manifest.is_relative_to(REPO_ROOT) else args.manifest)

    write_split(split, args.out, args.ids_dir)
    write_manifest_with_split(records, split, REPO_ROOT / "audio_dataset" / "manifest_with_split.csv")

    totals = split["counts"]["originals_totals"]
    print(f"Frozen split written to {args.out}")
    print(f"  algorithm      : {split['algorithm']}  seed={split['seed']}")
    print(f"  manifest sha256: {manifest_hash[:16]}...")
    print(f"  originals      : train={totals['train']} val={totals['val']} test={totals['test']}")
    print(f"  derived        : {split['counts']['derived_by_split']}")
    for label in sorted(classes):
        row = split["counts"]["originals_by_split"]
        print(f"    {label:<24} " + " ".join(f"{s}={row[s].get(label, 0):>4}" for s in SPLIT_NAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
