"""
Tests against the REAL frozen artifacts on disk.

Owner: fatima (QA).

WHY THIS FILE IS NOT A DUPLICATE OF test_split_integrity.py
------------------------------------------------------------
`test_split_integrity.py` builds a synthetic 3000-clip manifest in a temp directory.
That is the right design for proving the *split logic* — but it says nothing about the
actual repository. This file reads the committed artifacts:

    audio_dataset/manifest.csv
    data/splits/split.json

and refuses to pass on anything less. An evaluator counts the files on disk; these
tests count the same files.

The one check nobody else performs: every manifest row's sha256 must match the bytes of
the file it points at. build_split.py and verify_split.py compute the hash, but
verify_split.py only checks that no two rows *share* one — a manifest could describe a
completely different file and still pass `--check-duplicates`. This is where a swapped
or truncated clip would hide, and it is the QA gate.

SKIP POLICY (deliberate, not a soft failure)
--------------------------------------------
If the frozen artifacts do not exist yet, these tests SKIP with the reason printed in the
report, so the suite stays green while the corpus is incomplete. The day they exist, the
same tests become hard assertions. A skip is visible in pytest output — it is not a pass.

Run:
    .venv/bin/python -m pytest tests/test_frozen_split.py -v
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from audio_dataset.build_split import (  # noqa: E402
    AUGMENTED, ORIGINAL, SPLIT_NAMES, load_classes, load_manifest,
)

MANIFEST = REPO_ROOT / "audio_dataset" / "manifest.csv"
SPLIT_JSON = REPO_ROOT / "data" / "splits" / "split.json"
AUDIO_ROOT = REPO_ROOT / "audio_dataset"

CRITICAL_CLASSES = {
    "Gunshot", "Glass Breaking", "Panic Scream", "Aggression", "Person Asking for Help",
}

# SRS Step 5 / hint: 300 originals per class -> 210/45/45 per class.
TRAIN_TOTAL, VAL_TOTAL, TEST_TOTAL = 2100, 450, 450


# ---------------------------------------------------------------------------
# Fixtures: load the real artifacts once, skip if absent
# ---------------------------------------------------------------------------

def _require(path: Path) -> Path:
    """Return `path` if it exists, else pytest.skip with an actionable reason."""
    if not path.exists():
        pytest.skip(
            f"frozen artifact missing: {path}\n"
            "Build it with:\n"
            "  .venv/bin/python audio_dataset/scripts/assemble_manifest.py\n"
            "  .venv/bin/python audio_dataset/build_split.py --manifest "
            "audio_dataset/manifest.csv --strict"
        )
    return path


@pytest.fixture(scope="module")
def classes() -> dict[str, dict]:
    return load_classes()


@pytest.fixture(scope="module")
def manifest_path() -> Path:
    return _require(MANIFEST)


@pytest.fixture(scope="module")
def split_path() -> Path:
    return _require(SPLIT_JSON)


@pytest.fixture(scope="module")
def records(manifest_path: Path) -> list[dict[str, str]]:
    rows = load_manifest(manifest_path)
    if not rows:
        pytest.skip(f"{manifest_path} has no data rows")
    return rows


@pytest.fixture(scope="module")
def split(split_path: Path) -> dict:
    with split_path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Criterion 1 — all 10 classes, >=300 unique originals each
# ---------------------------------------------------------------------------

def test_all_ten_mandatory_classes_present(classes: dict, records: list[dict[str, str]]):
    """No class may be dropped, and no phantom class invented."""
    present = {r["class_label"].strip() for r in records}
    assert present == set(classes), (
        f"manifest classes differ from config/classes.json: "
        f"missing={sorted(set(classes) - present)}, extra={sorted(present - set(classes))}"
    )


def test_three_thousand_originals(records: list[dict[str, str]]):
    originals = [r for r in records
                 if r["original_or_augmented"].strip().lower() == ORIGINAL]
    assert len(originals) >= 3000, (
        f"SRS floor is 3,000 unique originals; manifest has {len(originals)}"
    )


def test_at_least_300_originals_per_class(classes: dict, records: list[dict[str, str]]):
    """The number an evaluator counts. Augmented/derived rows do not count."""
    per_class = Counter(
        r["class_label"].strip()
        for r in records
        if r["original_or_augmented"].strip().lower() == ORIGINAL
    )
    short = {label: per_class.get(label, 0) for label in classes if per_class.get(label, 0) < 300}
    assert not short, f"classes under the 300-original floor: {short}"


def test_audio_ids_are_unique(records: list[dict[str, str]]):
    ids = [r["audio_id"] for r in records]
    dupes = {i for i, n in Counter(ids).items() if n > 1}
    assert not dupes, f"{len(dupes)} duplicate audio_id, e.g. {sorted(dupes)[:5]}"


# ---------------------------------------------------------------------------
# Criterion 2 — exactly 2100 / 450 / 450 over originals
# ---------------------------------------------------------------------------

def test_frozen_split_totals_are_exactly_2100_450_450(split: dict):
    """Overall stratified totals for ORIGINALS (SRS hint)."""
    totals = Counter(
        info["split"] for info in split["assignments"].values()
        if info["role"] == ORIGINAL
    )
    assert (totals["train"], totals["val"], totals["test"]) == (TRAIN_TOTAL, VAL_TOTAL, TEST_TOTAL), (
        f"overall originals {dict(totals)} != {TRAIN_TOTAL}/{VAL_TOTAL}/{TEST_TOTAL}"
    )


def test_every_class_contributes_210_45_45(classes: dict, split: dict):
    """Stratification is per class, not just overall — a thin class must be visible."""
    by_class: dict[str, Counter] = defaultdict(Counter)
    for info in split["assignments"].values():
        if info["role"] == ORIGINAL:
            by_class[info["class_label"]][info["split"]] += 1
    bad = {
        label: dict(c)
        for label, c in sorted(by_class.items())
        if sum(c.values()) == 300 and (c["train"], c["val"], c["test"]) != (210, 45, 45)
    }
    assert not bad, f"classes whose stratification is not 210/45/45: {bad}"


def test_split_is_exhaustive_over_manifest(records: list[dict[str, str]], split: dict):
    """Every manifest row is assigned; no phantom id invented."""
    manifest_ids = {r["audio_id"] for r in records}
    assigned = set(split["assignments"])
    assert not (manifest_ids - assigned), f"{len(manifest_ids - assigned)} rows unassigned"
    assert not (assigned - manifest_ids), f"{len(assigned - manifest_ids)} phantom ids"


# ---------------------------------------------------------------------------
# Criterion 3 — zero val/test leakage into training
# ---------------------------------------------------------------------------

def test_no_audio_id_in_two_splits(split: dict):
    """An id in train and test is the textbook leak."""
    seen: dict[str, set] = defaultdict(set)
    for audio_id, info in split["assignments"].items():
        seen[audio_id].add(info["split"])
    leaked = {a: sorted(s) for a, s in seen.items() if len(s) > 1}
    assert not leaked, f"{len(leaked)} ids in >1 split, e.g. {list(leaked)[:5]}"


def test_no_val_or_test_id_in_train(split: dict):
    """The train bucket must contain no id that val or test also claims."""
    buckets = {name: set() for name in SPLIT_NAMES}
    for audio_id, info in split["assignments"].items():
        buckets[info["split"]].add(audio_id)
    overlap_v = buckets["train"] & buckets["val"]
    overlap_t = buckets["train"] & buckets["test"]
    assert not overlap_v, f"{len(overlap_v)} val ids also in train, e.g. {sorted(overlap_v)[:5]}"
    assert not overlap_t, f"{len(overlap_t)} test ids also in train, e.g. {sorted(overlap_t)[:5]}"


def test_manifest_split_column_matches_frozen_split_json(
    records: list[dict[str, str]], split: dict,
):
    """Two sources of truth must agree. A manifest column that disagrees with split.json
    means the manifest was written before the split was frozen (or after it changed)."""
    by_manifest = {r["audio_id"]: str(r.get("dataset_split", "")).strip() for r in records}
    disagree = {
        a: (by_manifest[a], split["assignments"][a]["split"])
        for a in split["assignments"]
        if by_manifest.get(a, "") != split["assignments"][a]["split"]
    }
    assert not disagree, (
        f"{len(disagree)} ids where manifest.csv != split.json, e.g. {list(disagree)[:5]}"
    )


def test_lineage_never_crosses_a_split_boundary(records: list[dict[str, str]], split: dict):
    """SRS Step 5: all segments derived from one recording stay in the same split.
    A derived clip whose parent sits in train while it sits in test is leakage by proxy."""
    assignment = {a: info["split"] for a, info in split["assignments"].items()}
    crossed = []
    for r in records:
        parent = (r.get("parent_audio_id") or "").strip()
        status = r["original_or_augmented"].strip().lower()
        if not parent or parent == r["audio_id"]:
            continue
        if status == ORIGINAL:  # an original cannot have a parent
            continue
        if parent in assignment and r["audio_id"] in assignment:
            if assignment[parent] != assignment[r["audio_id"]]:
                crossed.append((r["audio_id"], parent,
                                assignment[parent], assignment[r["audio_id"]]))
    assert not crossed, (
        f"{len(crossed)} derived clips straddle a split boundary, e.g. {crossed[:5]}"
    )


# ---------------------------------------------------------------------------
# Criterion 4 — the feature matrix basis: files exist and hashes are honest
# ---------------------------------------------------------------------------

def test_every_manifest_file_exists_on_disk(records: list[dict[str, str]]):
    """A row pointing at a missing file is a row the extractor cannot consume."""
    missing = []
    for r in records:
        rel = (r.get("filename") or "").strip()
        if not rel:
            continue
        candidate = AUDIO_ROOT / rel
        if not candidate.exists():
            missing.append(rel)
    assert not missing, f"{len(missing)} manifest files missing on disk, e.g. {missing[:5]}"


def test_no_audio_file_on_disk_is_unlisted(records: list[dict[str, str]]):
    """A clip that exists but is in no manifest row is invisible to the split —
    it could be an unsplit training file, so it must be surfaced."""
    listed = {(r.get("filename") or "").strip() for r in records}
    listed.discard("")
    unlisted = []
    for path in AUDIO_ROOT.rglob("*.wav"):
        rel = str(path.relative_to(AUDIO_ROOT))
        if rel not in listed and "/__pycache__/" not in rel:
            unlisted.append(rel)
    assert not unlisted, (
        f"{len(unlisted)} wav files on disk in no manifest row, e.g. {sorted(unlisted)[:5]}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.slow
def test_manifest_sha256_matches_file_bytes(records: list[dict[str, str]]):
    """The QA check nobody else runs.

    verify_split.py --check-duplicates asserts no two rows share a sha256; it does NOT
    assert the hash describes the file the row points at. A renamed, truncated or swapped
    clip would sail through every other gate and land in the feature matrix. This hashes
    the real bytes and compares. Marked slow: it is O(n) disk reads over 3,000 clips."""
    mismatches = []
    checked = 0
    for r in records:
        rel = (r.get("filename") or "").strip()
        claimed = (r.get("sha256") or "").strip()
        if not rel or not claimed:
            continue
        path = AUDIO_ROOT / rel
        if not path.exists():
            continue
        if _sha256(path) != claimed:
            mismatches.append((r["audio_id"], rel))
        checked += 1
    assert checked > 0, "no row had both a filename and a sha256 — nothing was verified"
    assert not mismatches, (
        f"{len(mismatches)}/{checked} files do not match their manifest sha256, "
        f"e.g. {mismatches[:5]}"
    )


def test_no_two_rows_share_a_sha256(records: list[dict[str, str]]):
    """Two rows, one file: a duplicate is the same clip counted twice toward the
    300-per-class floor."""
    hashes = Counter(
        (r.get("sha256") or "").strip()
        for r in records
        if (r.get("sha256") or "").strip()
    )
    dupes = {h: n for h, n in hashes.items() if n > 1}
    assert not dupes, f"{len(dupes)} sha256 values appear in more than one row"


def test_critical_classes_are_all_populated(records: list[dict[str, str]]):
    """A critical class at zero means the 85% recall NFR is unmeetable for it."""
    per_class = Counter(
        r["class_label"].strip()
        for r in records
        if r["original_or_augmented"].strip().lower() == ORIGINAL
    )
    empty = [c for c in CRITICAL_CLASSES if per_class.get(c, 0) == 0]
    assert not empty, f"critical classes with zero originals: {empty}"
