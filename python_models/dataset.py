"""Dataset access for model training -- the only place that resolves a manifest row to a file.

Owner: bilal.

WHY THIS EXISTS SEPARATELY FROM ``audio_dataset/``
--------------------------------------------------
``audio_dataset/`` owns the corpus, its schema and its split. Training owns *reading* it.
Keeping the reader here means a change to the corpus layout breaks exactly one module with
an explicit error, instead of every training script discovering the problem as a
``FileNotFoundError`` halfway through a 40-minute feature extraction run.

It also means there is one answer to "where is this file", which matters because the
manifest stores a relative path and several plausible roots exist in this repository
(``audio_dataset/``, ``audio_dataset/originals/``, the raw download cache). Guessing wrong
produces a silently empty class, which is the kind of failure that reaches the report.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Roots a manifest ``filename`` may be relative to, tried in order.
_AUDIO_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "audio_dataset",
    REPO_ROOT,
)

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


class DatasetError(RuntimeError):
    """Raised when the corpus cannot be read the way training requires."""


# --------------------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------------------


def resolve_audio_path(record: Mapping[str, Any]) -> Path:
    """Absolute path to a manifest row's audio file.

    Accepts an absolute path in ``filename`` as-is, otherwise tries each known root. Raises
    with the list of locations it tried, because "file not found" without the candidates is
    a twenty-minute debugging session.
    """
    raw = str(record.get("filename") or record.get("filepath") or "").strip()
    if not raw:
        raise DatasetError(f"record {record.get('audio_id')!r} has no filename field")

    candidate = Path(raw)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise DatasetError(f"manifest points at missing absolute path: {candidate}")

    tried: list[Path] = []
    for root in _AUDIO_ROOTS:
        path = root / raw
        tried.append(path)
        if path.exists():
            return path
    raise DatasetError(
        f"audio file for {record.get('audio_id')!r} not found. Tried:\n  "
        + "\n  ".join(str(p) for p in tried)
    )


# --------------------------------------------------------------------------------------
# Manifest loading
# --------------------------------------------------------------------------------------


@dataclass
class SplitData:
    """A split's records, plus the class counts, so a thin class is visible immediately."""

    name: str
    records: list[dict[str, str]]

    @property
    def audio_ids(self) -> list[str]:
        return [str(r["audio_id"]) for r in self.records]

    @property
    def labels(self) -> list[str]:
        return [str(r["class_label"]) for r in self.records]

    def class_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(self.labels).items()))

    def __len__(self) -> int:
        return len(self.records)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """All rows of a manifest CSV, or a clear error explaining what is wrong with it."""
    manifest = Path(path)
    if not manifest.is_absolute():
        manifest = REPO_ROOT / manifest
    if not manifest.exists():
        raise DatasetError(
            f"manifest not found: {manifest}\n"
            "Build it with: .venv/bin/python audio_dataset/build_split.py "
            "--manifest audio_dataset/manifests/manifest.csv"
        )
    with manifest.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        for required in ("audio_id", "class_label"):
            if required not in fieldnames:
                raise DatasetError(
                    f"{manifest} is missing the required column {required!r}; "
                    f"found: {fieldnames}"
                )
        return [dict(row) for row in reader]


def load_all_splits(manifest_path: str | Path) -> dict[str, SplitData]:
    """Every split, keyed ``train``/``val``/``test``.

    Refuses a manifest with no ``dataset_split`` column rather than inventing a split: the
    split is frozen by ``audio_dataset/build_split.py`` and re-deriving it here would give
    the Python model and the GTM model different test sets, which quietly invalidates the
    comparison.
    """
    rows = read_manifest(manifest_path)
    if not rows:
        raise DatasetError(f"manifest {manifest_path} has no data rows")
    if "dataset_split" not in rows[0]:
        raise DatasetError(
            "manifest has no dataset_split column. Run audio_dataset/build_split.py first "
            "-- training must never assign the split itself."
        )

    blank = [r.get("audio_id") for r in rows if not str(r.get("dataset_split", "")).strip()]
    if blank:
        raise DatasetError(
            f"{len(blank)} row(s) have an empty dataset_split, e.g. {blank[:5]}"
        )

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset_split"]).strip(), []).append(row)
    return {name: SplitData(name=name, records=recs) for name, recs in sorted(grouped.items())}


def load_records_for_split(manifest_path: str | Path, split: str) -> SplitData:
    splits = load_all_splits(manifest_path)
    if split not in splits:
        raise DatasetError(
            f"split {split!r} not in manifest; available: {sorted(splits)}"
        )
    return splits[split]


# --------------------------------------------------------------------------------------
# Training-only rows
# --------------------------------------------------------------------------------------


def training_records(
    split: SplitData,
    *,
    exclude_augmented: bool = False,
    require_originals: bool = True,
) -> list[dict[str, str]]:
    """Rows of a split that are legal to train on.

    ``require_originals`` does not filter anything out of training -- originals and their
    augmented copies both legitimately train the model, which is the point of augmentation.
    It exists to record in the metrics artefact how much of the training set was augmented,
    because "trained on 2,100 clips" is a different claim depending on whether 300 of them
    are generated variants of the other 1,800, and the report must be able to say which.

    ``exclude_augmented`` is for the evaluation path, where augmented audio must never be
    scored: a model that memorised an augmented copy of a training clip would score it as
    generalisation. (``src/training/evaluation.py`` enforces this independently; this
    provides the same filter at the training boundary.)
    """
    records = list(split.records)
    if exclude_augmented:
        records = [
            r
            for r in records
            if str(r.get("original_or_augmented", "original")).strip().lower() == "original"
        ]
    if require_originals:
        for record in records:
            if not str(record.get("original_or_augmented", "")).strip():
                raise DatasetError(
                    f"record {record.get('audio_id')!r} has no original_or_augmented value; "
                    "the report cannot distinguish real clips from augmented copies without it"
                )
    return records


def describe_split(split: SplitData) -> dict[str, Any]:
    """Counts the training artefact records, so the report never asserts them from memory."""
    counts = split.class_counts()
    augmented = sum(
        1
        for r in split.records
        if str(r.get("original_or_augmented", "original")).strip().lower() != "original"
    )
    return {
        "split": split.name,
        "n_records": len(split),
        "n_originals": len(split) - augmented,
        "n_augmented": augmented,
        "class_counts": counts,
        "n_classes_present": len(counts),
        "min_class_count": min(counts.values()) if counts else 0,
        "max_class_count": max(counts.values()) if counts else 0,
    }


def class_imbalance_report(split: SplitData, class_names: Sequence[str]) -> dict[str, Any]:
    """The class-imbalance numbers, stated the way a reviewer would check them.

    Returns the counts per class, the ratio of the largest class to the smallest, and the
    count for any configured class that is entirely absent -- an absent class is the failure
    that makes a macro-F1 meaningless while still printing a plausible number, so it is
    called out by name.
    """
    counts = Counter(split.labels)
    present = {name: int(counts.get(name, 0)) for name in class_names}
    absent = [name for name, n in present.items() if n == 0]
    non_zero = [n for n in present.values() if n > 0]
    return {
        "split": split.name,
        "counts": present,
        "absent_classes": absent,
        "imbalance_ratio": (max(non_zero) / min(non_zero)) if non_zero else None,
    }


# --------------------------------------------------------------------------------------
# Integrity helpers used by tests
# --------------------------------------------------------------------------------------


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash, for duplicate detection."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def find_duplicate_content(paths: Iterable[str | Path]) -> dict[str, list[str]]:
    """Audio files sharing byte-identical content, keyed by hash.

    Byte-identical files are the cheapest form of near-duplicate leakage: the same
    recording entered twice lands in train and test, and the test score stops being a
    measurement. Reporting the groups is enough for the split auditor to act on.
    """
    groups: dict[str, list[str]] = {}
    for path in paths:
        groups.setdefault(file_sha256(path), []).append(str(path))
    return {h: sorted(files) for h, files in groups.items() if len(files) > 1}
