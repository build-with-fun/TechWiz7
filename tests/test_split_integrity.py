"""
Split integrity tests — the leakage guard for the whole project.

Owner: lorena.  SRS Step 5, FR xvii, FR xviii, Deliverable 3, and Integrity rules 7 and 8.

These tests are self-contained: they build a synthetic 3000-clip manifest in a temp
directory, so they prove the split logic is correct without depending on how much audio
`omar` has downloaded. That matters — if these only ran against real data they would be
skipped exactly when the dataset is incomplete, which is when leakage is most likely.

A leak here would not fail loudly in training. It would show up as a suspiciously good
accuracy figure in the report, and an evaluator who counted the files would find it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from audio_dataset.build_split import (
    AUGMENTED,
    ManifestError,
    ORIGINAL,
    SPLIT_NAMES,
    assert_strict,
    build_split,
    load_classes,
    split_counts,
    validate_records,
)

CLASSES = list(load_classes())
N_PER_CLASS = 300


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def make_record(audio_id: str, label: str, *, filename: str | None = None,
                status: str = ORIGINAL, parent: str | None = None,
                licence: str = "CC0-1.0", **extra) -> dict[str, str]:
    return {
        "audio_id": audio_id,
        "filename": filename if filename is not None else f"raw/{audio_id}.wav",
        "class_label": label,
        "source": "synthetic-test",
        "source_url": "",
        "licence": licence,
        "author": "test",
        "date_fetched": "2026-09-23",
        "duration_sec": "3.0",
        "sampling_rate": "22050",
        "channels": "1",
        "recording_environment": "indoor",
        "recording_device": "test",
        "approximate_distance": "near",
        "original_or_augmented": status,
        "parent_audio_id": parent or "",
        "segment_start_sec": extra.pop("segment_start_sec", ""),
        "segment_end_sec": extra.pop("segment_end_sec", ""),
        "sha256": f"{abs(hash(audio_id)):064x}",
        "dataset_split": "",
        **extra,
    }


def make_manifest(n_per_class: int = N_PER_CLASS, classes: list[str] | None = None,
                  augmented_per_class: int = 0) -> list[dict[str, str]]:
    """Build a synthetic manifest: n_per_class originals per class, plus optional augmentations."""
    classes = classes or CLASSES
    records: list[dict[str, str]] = []
    for label in classes:
        code = "".join(w[0] for w in label.split()).upper()[:3]
        originals = []
        for i in range(n_per_class):
            audio_id = f"SS-{code}-{i:04d}"
            originals.append(audio_id)
            records.append(make_record(audio_id, label))
        for j in range(augmented_per_class):
            parent = originals[j % len(originals)]
            records.append(
                make_record(f"{parent}-AUG{j:02d}", label, status=AUGMENTED, parent=parent)
            )
    return records


def classes_config() -> dict:
    return load_classes()


# --------------------------------------------------------------------------------------
# split_counts — boundary behaviour
# --------------------------------------------------------------------------------------

def test_split_counts_300_matches_srs_hint_exactly():
    """300 per class must give 210/45/45, the exact numbers in the SRS Hint."""
    assert split_counts(300) == (210, 45, 45)


def test_split_counts_always_sum_to_total():
    """No item may be lost or duplicated at any size — including awkward ones."""
    for n in range(1, 501):
        n_train, n_val, n_test = split_counts(n)
        assert n_train + n_val + n_test == n, f"n={n} lost items"
        assert min(n_train, n_val, n_test) >= 0, f"n={n} produced a negative bucket"


def test_split_counts_degenerate_sizes():
    """1 and 2 items must not crash or produce negative counts."""
    assert sum(split_counts(1)) == 1
    assert sum(split_counts(2)) == 2
    assert split_counts(0) == (0, 0, 0)


# --------------------------------------------------------------------------------------
# The headline requirement
# --------------------------------------------------------------------------------------

def test_exact_2100_450_450_split():
    """SRS Hint: 70/15/15 over 3000 originals = 2100 train / 450 val / 450 test."""
    split = build_split(make_manifest(), classes_config())
    totals = split["counts"]["originals_totals"]
    assert totals["train"] == 2100
    assert totals["val"] == 450
    assert totals["test"] == 450
    assert sum(totals.values()) == 3000


def test_per_class_stratification_is_exact():
    """Every class contributes 210/45/45 — no class is under-represented in any split."""
    split = build_split(make_manifest(), classes_config())
    for label in CLASSES:
        row = split["counts"]["originals_by_split"]
        assert (row["train"][label], row["val"][label], row["test"][label]) == (210, 45, 45), label


def test_all_ten_mandatory_classes_present():
    """The split must cover all ten mandatory classes, not a convenient subset."""
    split = build_split(make_manifest(), classes_config())
    assert set(split["class_names"]) == set(CLASSES)
    assert len(CLASSES) == 10


# --------------------------------------------------------------------------------------
# Disjointness and leakage — the project-ending defects
# --------------------------------------------------------------------------------------

def test_splits_are_pairwise_disjoint():
    """An audio_id in two splits is leakage. Prove it never happens."""
    split = build_split(make_manifest(), classes_config())
    buckets = {s: set() for s in SPLIT_NAMES}
    for audio_id, info in split["assignments"].items():
        buckets[info["split"]].add(audio_id)

    assert buckets["train"] & buckets["val"] == set()
    assert buckets["train"] & buckets["test"] == set()
    assert buckets["val"] & buckets["test"] == set()
    assert len(buckets["train"] | buckets["val"] | buckets["test"]) == 3000


def test_every_record_is_assigned():
    """An unassigned row would silently drop out of training with nobody noticing."""
    records = make_manifest()
    split = build_split(records, classes_config())
    assert set(split["assignments"]) == {r["audio_id"] for r in records}


def test_augmented_clips_inherit_parent_split():
    """SRS: augmented recordings must stay in the same split as the original."""
    split = build_split(make_manifest(augmented_per_class=4), classes_config())
    for audio_id, info in split["assignments"].items():
        if info["role"] == AUGMENTED:
            parent = audio_id.rsplit("-AUG", 1)[0]
            assert info["split"] == split["assignments"][parent]["split"], audio_id


def test_segments_of_one_recording_stay_together():
    """
    The classic leakage pattern: slice a 30s recording into 6 segments, put 5 in train and
    1 in test, and report a wonderful score. Prove the builder refuses to do that.
    """
    records = []
    for label in CLASSES:
        code = "".join(w[0] for w in label.split()).upper()[:3]
        for i in range(N_PER_CLASS):
            parent = f"SS-{code}-{i:04d}"
            records.append(make_record(parent, label, filename=f"raw/{parent}.wav"))
            # six 5-second segments carved from that recording
            for s in range(6):
                records.append(make_record(
                    f"{parent}-SEG{s}", label,
                    filename=f"segments/{parent}-SEG{s}.wav", status=AUGMENTED, parent=parent,
                    segment_start_sec=str(s * 5.0), segment_end_sec=str(s * 5.0 + 5.0),
                ))

    split = build_split(records, classes_config())
    groups: dict[str, set[str]] = {}
    for audio_id, info in split["assignments"].items():
        root = audio_id.rsplit("-SEG", 1)[0]
        groups.setdefault(root, set()).add(info["split"])

    crossed = {root: s for root, s in groups.items() if len(s) > 1}
    assert not crossed, f"{len(crossed)} recordings straddle a split boundary, e.g. {list(crossed)[:3]}"


def test_augmented_are_not_counted_as_originals():
    """SRS: augmented recordings must not be counted as unique original clips."""
    with_aug = build_split(make_manifest(augmented_per_class=8), classes_config())
    assert with_aug["counts"]["originals_totals"] == {"train": 2100, "val": 450, "test": 450}

    plain = build_split(make_manifest(), classes_config())
    assert plain["counts"]["originals_totals"] == with_aug["counts"]["originals_totals"]
    # ...and the augmentations really were placed somewhere
    assert sum(with_aug["counts"]["derived_by_split"].values()) == 80


# --------------------------------------------------------------------------------------
# Determinism — an evaluator must be able to reproduce the freeze
# --------------------------------------------------------------------------------------

def test_split_is_deterministic_across_runs():
    records = make_manifest()
    first = build_split(records, classes_config())
    second = build_split(list(records), classes_config())
    assert first["assignments"] == second["assignments"]


def test_split_is_independent_of_manifest_row_order():
    """A differently-ordered manifest must produce the identical assignment."""
    records = make_manifest()
    shuffled = list(reversed(records))
    assert (build_split(records, classes_config())["assignments"]
            == build_split(shuffled, classes_config())["assignments"])


def test_different_seed_gives_a_different_split():
    """Sanity: the seed actually does something, so the freeze is a real choice."""
    records = make_manifest()
    a = build_split(records, classes_config(), seed=1)["assignments"]
    b = build_split(records, classes_config(), seed=2)["assignments"]
    assert a != b


# --------------------------------------------------------------------------------------
# Validation — negative tests
# --------------------------------------------------------------------------------------

def test_duplicate_audio_id_is_rejected():
    records = make_manifest()
    records.append(dict(records[0]))
    with pytest.raises(ManifestError, match="duplicate audio_id"):
        validate_records(records, classes_config())


def test_unknown_class_label_is_rejected():
    records = make_manifest()
    records[0]["class_label"] = "Explosion"  # an optional class we did NOT adopt
    with pytest.raises(ManifestError, match="not one of the"):
        validate_records(records, classes_config())


def test_missing_licence_is_rejected():
    """Ethical sourcing: an unlicensed clip may not enter the dataset."""
    records = make_manifest()
    records[0]["licence"] = ""
    with pytest.raises(ManifestError, match="licence"):
        validate_records(records, classes_config())


def test_augmented_without_parent_is_rejected():
    """Without a parent we cannot know its split, so it could leak."""
    records = make_manifest()
    records[0]["original_or_augmented"] = AUGMENTED
    records[0]["parent_audio_id"] = ""
    with pytest.raises(ManifestError, match="parent_audio_id"):
        validate_records(records, classes_config())


def test_parent_must_exist():
    records = make_manifest()
    records[0]["original_or_augmented"] = AUGMENTED
    records[0]["parent_audio_id"] = "SS-DOES-NOT-EXIST"
    with pytest.raises(ManifestError, match="not in the manifest"):
        validate_records(records, classes_config())


def test_augmented_parent_must_be_an_original():
    """A chain of augmentations could otherwise launder a clip across a split."""
    records = make_manifest(augmented_per_class=1)
    aug = next(r for r in records if r["original_or_augmented"] == AUGMENTED)
    records.append(make_record("SS-CHAIN-0001", aug["class_label"], status=AUGMENTED,
                               parent=aug["audio_id"]))
    with pytest.raises(ManifestError, match="itself augmented"):
        validate_records(records, classes_config())


def test_strict_rejects_undersized_dataset():
    """--strict must refuse to freeze a split that cannot meet the >=3000 floor."""
    small = make_manifest(n_per_class=100)
    with pytest.raises(ManifestError, match="strict"):
        assert_strict(small, classes_config())


def test_strict_rejects_unbalanced_dataset():
    """3000 clips with the wrong distribution is still off-spec."""
    records = make_manifest(n_per_class=300)
    # move 10 clips from Machinery Fault to Gunshot: total stays 3000, balance breaks
    moved = 0
    for rec in records:
        if rec["class_label"] == "Machinery Fault" and moved < 10:
            rec["class_label"] = "Gunshot"
            moved += 1
    with pytest.raises(ManifestError, match="strict"):
        assert_strict(records, classes_config())


def test_strict_accepts_a_compliant_dataset():
    assert_strict(make_manifest(), classes_config())


# --------------------------------------------------------------------------------------
# End-to-end artifact test: build -> write -> reload -> re-audit
# --------------------------------------------------------------------------------------

def test_build_write_and_reload_round_trip(tmp_path: Path):
    """The frozen file must survive a write/read cycle and still audit clean."""
    from audio_dataset.build_split import write_split

    records = make_manifest(augmented_per_class=2)
    split = build_split(records, classes_config())

    out = tmp_path / "split.json"
    write_split(split, out, tmp_path)

    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["assignments"] == split["assignments"]
    assert reloaded["counts"]["originals_totals"] == {"train": 2100, "val": 450, "test": 450}
    assert reloaded["seed"] == 20260923

    # id lists written for the training scripts to consume
    for split_name in SPLIT_NAMES:
        ids = (tmp_path / f"{split_name}_ids.txt").read_text(encoding="utf-8").split()
        assert len(ids) == len(set(ids))
        for audio_id in ids:
            assert reloaded["assignments"][audio_id]["split"] == split_name


def test_split_json_records_its_provenance():
    """The artifact must state how it was made, so an evaluator can reproduce it."""
    split = build_split(make_manifest(), classes_config())
    assert split["algorithm"] == "sha256-order-v1"
    assert split["seed"] == 20260923
    assert split["ratios"] == {"train": 0.7, "val": 0.15, "test": 0.15}


def test_manifest_csv_round_trip(tmp_path: Path):
    """The manifest schema must survive a CSV write/read cycle unchanged."""
    records = make_manifest(n_per_class=5)
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    with path.open(newline="", encoding="utf-8") as fh:
        back = list(csv.DictReader(fh))
    assert len(back) == len(records)
    assert {r["audio_id"] for r in back} == {r["audio_id"] for r in records}


# --------------------------------------------------------------------------------------
# Schema interop — the contract must accept other people's generators
# --------------------------------------------------------------------------------------

def test_extra_columns_from_a_generator_are_preserved_not_rejected(tmp_path: Path):
    """A generator may carry more than the contract requires; that must not be an error.

    Guaranteeing interop by forbidding extra columns would make every generator conform
    to exactly the contract, which is how you get two contracts. Extra columns are kept
    so the split output loses nothing the generator knew.
    """
    records = make_manifest(n_per_class=5)
    for record in records:
        record["condition"] = "clean"
        record["perceptual_notes"] = "synthetic"
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    from audio_dataset.build_split import load_manifest

    loaded = load_manifest(path)
    assert len(loaded) == len(records)
    assert loaded[0]["condition"] == "clean"
    assert loaded[0]["perceptual_notes"] == "synthetic"
    # The required ones are still there.
    assert loaded[0]["audio_id"] and loaded[0]["class_label"]


def test_missing_columns_error_names_the_fix_and_the_no_write_rule(tmp_path: Path):
    """The error is read by whoever is mid-debug on a generator at 1am.

    It must say which contract, map the obvious near-misses, and warn against emitting
    dataset_split itself — two split definitions is the one thing that quietly produces
    an 85% score that means nothing.
    """
    path = tmp_path / "manifest.csv"
    path.write_text(
        "audio_id,filename,class_label,sample_rate,label\n"
        "SS-001,a.wav,Gunshot,22050,Gunshot\n",
        encoding="utf-8",
    )

    from audio_dataset.build_split import load_manifest

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(path)
    message = str(excinfo.value)

    assert "manifest_schema.md" in message            # which contract
    assert "sample_rate -> sampling_rate" in message  # the rename map, derived from the file
    assert "dataset_split" in message                 # who owns the split column
    assert "abuild_split" not in message              # no garbled paths
    assert "build_split.py" in message
