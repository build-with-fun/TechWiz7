#!/usr/bin/env python3
"""Build the train/validation/test split used by both models (SRS Step 5, FR xvii-xviii).

- Clips from the same source recording always go to the same split.
- Ordering uses sha256(f"{seed}:{key}") so the split is identical on every machine.
- Stratified per class: 300 originals per class gives 210/45/45.
- Augmented clips and segments get their parent's split.

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

# Constants

SEED = 20260923
# v2 (26 Sep 2026) assigns whole source groups. v1 split clip by clip and leaked 125 source
# recordings across partitions (see documentation/devlog.md).
ALGORITHM = "sha256-order-v2-source-groups"
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ("train", "val", "test")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "audio_dataset" / "manifest.csv"
DEFAULT_SPLIT_PATH = REPO_ROOT / "data" / "splits" / "split.json"
DEFAULT_IDS_DIR = REPO_ROOT / "data" / "splits"

# Manifest columns, in the order from audio_dataset/manifest_schema.md
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
    """Raised when the manifest has problems."""


# Loading and validation

def load_classes(classes_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Class list from config/classes.json."""
    path = classes_path or (REPO_ROOT / "config" / "classes.json")
    with path.open(encoding="utf-8") as fh:
        cfg = json.load(fh)
    return {c["name"]: c for c in cfg["classes"]}


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read the manifest CSV, keeping only known columns."""
    if not manifest_path.exists():
        raise ManifestError(f"manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ManifestError("manifest is empty (no header row)")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            # Point to the schema and the rename map.
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
                + "Don't write dataset_split by hand; audio_dataset/build_split.py assigns it.\n"
                "Extra columns are kept, so a generator only needs to include the required ones."
            )
        rows = [dict(r) for r in reader]

    if not rows:
        raise ManifestError("manifest has a header but no data rows")
    return rows


def validate_records(records: list[dict[str, str]], classes: dict[str, dict[str, Any]]) -> None:
    """Check the manifest and raise ManifestError listing every problem."""
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
                f"line {i} ({audio_id}): empty licence (every clip needs one)"
            )

        status = (rec.get("original_or_augmented") or "").strip().lower()
        if status not in (ORIGINAL, AUGMENTED):
            problems.append(
                f"line {i} ({audio_id}): original_or_augmented must be "
                f"{ORIGINAL!r} or {AUGMENTED!r}, got {status!r}"
            )
        elif status == AUGMENTED and not (rec.get("parent_audio_id") or "").strip():
            problems.append(
                f"line {i} ({audio_id}): augmented row has no parent_audio_id, so its "
                f"split can't be decided"
            )

    # Parents must exist, and an augmented clip's parent must be an original.
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


# Ordering

def _order_key(seed: int, audio_id: str) -> str:
    """Hash-based ordering key, the same on every platform."""
    return hashlib.sha256(f"{seed}:{audio_id}".encode("utf-8")).hexdigest()


def source_group(rec: dict[str, str]) -> str:
    """The source recording a clip came from.

    UrbanSound8K, ESC-50 and FSD50K clips group by Freesound ID, synthetic help phrases by
    voice + phrase. Anything else is its own group.
    """
    import re

    source = rec.get("source", "") or ""
    original = rec.get("original_filename", "") or ""
    freesound = (rec.get("freesound_id", "") or "").split(".")[0]
    if source.startswith("UrbanSound8K"):
        match = re.match(r"(\d+)-", original)
        if match:
            return f"freesound:{match.group(1)}"
    if source.startswith("ESC-50"):
        match = re.match(r"\d+-(\d+)-", original)
        if match:
            return f"freesound:{match.group(1)}"
    if freesound and freesound.lower() != "nan":
        return f"freesound:{freesound}"
    voice, phrase = rec.get("voice_name", "") or "", rec.get("phrase", "") or ""
    if voice and phrase:
        words = re.sub(r"[^a-z ]", "", phrase.lower()).split()
        return f"tts:{voice}:{' '.join(words)}"
    return f"clip:{rec['audio_id']}"


def split_counts(n: int, ratios: tuple[float, float, float] = SPLIT_RATIOS) -> tuple[int, int, int]:
    """Train/val/test counts that sum to n, e.g. (210, 45, 45) for 300."""
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


# Split construction

def build_split(
    records: list[dict[str, str]],
    classes: dict[str, dict[str, Any]],
    seed: int = SEED,
) -> dict[str, Any]:
    """Assign every record to train/val/test (no side effects)."""
    validate_records(records, classes)

    by_id = {r["audio_id"]: r for r in records}
    originals = [r for r in records if r["original_or_augmented"].strip().lower() == ORIGINAL]
    derived = [r for r in records if r["original_or_augmented"].strip().lower() == AUGMENTED]

    assignments: dict[str, dict[str, str]] = {}

    # Pass 1: originals, stratified per class, whole source groups at a time
    groups: dict[str, list[str]] = defaultdict(list)
    for rec in originals:
        groups[source_group(rec)].append(rec["audio_id"])
    counts_by_split: dict[str, Counter[str]] = {s: Counter() for s in SPLIT_NAMES}
    per_class_total = Counter(r["class_label"].strip() for r in originals)
    remaining_by_class = {label: dict(zip(SPLIT_NAMES, split_counts(n)))
                          for label, n in per_class_total.items()}

    # A recording with two labels (Freesound 43806) goes to train.
    mixed_groups = []
    per_class_groups: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for key, members in groups.items():
        labels = {by_id[a]["class_label"].strip() for a in members}
        if len(labels) > 1:
            mixed_groups.append(key)
            for audio_id in members:
                label = by_id[audio_id]["class_label"].strip()
                assignments[audio_id] = {"split": "train", "class_label": label,
                                         "role": ORIGINAL, "group": key}
                counts_by_split["train"][label] += 1
                remaining_by_class[label]["train"] -= 1
            continue
        per_class_groups[labels.pop()].append((key, sorted(members)))

    for label in sorted(per_class_groups):
        target = dict(zip(SPLIT_NAMES, split_counts(per_class_total[label])))
        remaining = remaining_by_class[label]
        # Biggest groups first, ties in hash order.
        ordered = sorted(per_class_groups[label],
                         key=lambda g: (-len(g[1]), _order_key(seed, g[0])))
        for key, members in ordered:
            fits = [s for s in SPLIT_NAMES if remaining[s] >= len(members)]
            if not fits:
                raise ManifestError(f"{label}: group {key} ({len(members)} clips) does not fit "
                                    f"the remaining quotas {remaining}")
            # Give the group to the emptiest partition (relative to its target).
            chosen = max(fits, key=lambda s: (remaining[s] / target[s], -SPLIT_NAMES.index(s)))
            remaining[chosen] -= len(members)
            for audio_id in members:
                assignments[audio_id] = {"split": chosen, "class_label": label,
                                         "role": ORIGINAL, "group": key}
                counts_by_split[chosen][label] += 1

    # Pass 2: segments and augmented copies inherit their parent's split.
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
            "source_groups": len(groups),
            "mixed_class_groups": sorted(mixed_groups),
            "multi_clip_groups": sum(1 for m in groups.values() if len(m) > 1),
            "derived_by_split": {
                s: sum(1 for a in assignments.values() if a["split"] == s and a["role"] != ORIGINAL)
                for s in SPLIT_NAMES
            },
        },
    }


def assert_strict(records: list[dict[str, str]], classes: dict[str, dict[str, Any]]) -> None:
    """--strict: fail if the split doesn't meet the SRS dataset minimums."""
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


# Persistence

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
    """Write the manifest with the dataset_split column filled in."""
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


# CLI

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
            f"Not overwriting the frozen split at {args.out}.\n"
            f"Use --force only if the dataset really changed; a new split invalidates every "
            f"model trained or evaluated on the old one.",
            file=sys.stderr,
        )
        return 3

    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    split["manifest_sha256"] = manifest_hash
    split["manifest_path"] = str(args.manifest.relative_to(REPO_ROOT)
                                 if args.manifest.is_relative_to(REPO_ROOT) else args.manifest)

    write_split(split, args.out, args.ids_dir)
    # Rewrite manifest.csv with dataset_split so it matches split.json.
    write_manifest_with_split(records, split, args.manifest)
    # Keep a copy of the manifest before the split.
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
