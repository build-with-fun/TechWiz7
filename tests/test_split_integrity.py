"""Split logic on a synthetic 3,000-clip manifest (SRS Step 5, FR xvii-xviii).

Doesn't need the real dataset.
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


# Helpers

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
    """Synthetic manifest with n_per_class originals per class, plus optional augmented rows."""
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


# split_counts

def test_split_counts_300_matches_srs_hint_exactly():
    """300 gives 210/45/45."""
    assert split_counts(300) == (210, 45, 45)


def test_split_counts_always_sum_to_total():
    """Counts always add up to the total."""
    for n in range(1, 501):
        n_train, n_val, n_test = split_counts(n)
        assert n_train + n_val + n_test == n, f"n={n} lost items"
        assert min(n_train, n_val, n_test) >= 0, f"n={n} produced a negative bucket"


def test_split_counts_degenerate_sizes():
    """1 and 2 items work."""
    assert sum(split_counts(1)) == 1
    assert sum(split_counts(2)) == 2
    assert split_counts(0) == (0, 0, 0)


# Totals

def test_exact_2100_450_450_split():
    """70/15/15 of 3000 = 2100 / 450 / 450."""
    split = build_split(make_manifest(), classes_config())
    totals = split["counts"]["originals_totals"]
    assert totals["train"] == 2100
    assert totals["val"] == 450
    assert totals["test"] == 450
    assert sum(totals.values()) == 3000


def test_per_class_stratification_is_exact():
    """Every class is split 210/45/45."""
    split = build_split(make_manifest(), classes_config())
    for label in CLASSES:
        row = split["counts"]["originals_by_split"]
        assert (row["train"][label], row["val"][label], row["test"][label]) == (210, 45, 45), label


def test_all_ten_mandatory_classes_present():
    """All ten classes are present."""
    split = build_split(make_manifest(), classes_config())
    assert set(split["class_names"]) == set(CLASSES)
    assert len(CLASSES) == 10


# Leakage

def test_splits_are_pairwise_disjoint():
    """No audio_id is in two splits."""
    split = build_split(make_manifest(), classes_config())
    buckets = {s: set() for s in SPLIT_NAMES}
    for audio_id, info in split["assignments"].items():
        buckets[info["split"]].add(audio_id)

    assert buckets["train"] & buckets["val"] == set()
    assert buckets["train"] & buckets["test"] == set()
    assert buckets["val"] & buckets["test"] == set()
    assert len(buckets["train"] | buckets["val"] | buckets["test"]) == 3000


def test_every_record_is_assigned():
    """Every row gets a split."""
    records = make_manifest()
    split = build_split(records, classes_config())
    assert set(split["assignments"]) == {r["audio_id"] for r in records}


def test_augmented_clips_inherit_parent_split():
    """Augmented clips get their parent's split."""
    split = build_split(make_manifest(augmented_per_class=4), classes_config())
    for audio_id, info in split["assignments"].items():
        if info["role"] == AUGMENTED:
            parent = audio_id.rsplit("-AUG", 1)[0]
            assert info["split"] == split["assignments"][parent]["split"], audio_id


def test_segments_of_one_recording_stay_together():
    """Segments of one recording all go to the same split."""
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
    """Augmented clips don't count as originals."""
    with_aug = build_split(make_manifest(augmented_per_class=8), classes_config())
    assert with_aug["counts"]["originals_totals"] == {"train": 2100, "val": 450, "test": 450}

    plain = build_split(make_manifest(), classes_config())
    assert plain["counts"]["originals_totals"] == with_aug["counts"]["originals_totals"]
    # but they are still assigned
    assert sum(with_aug["counts"]["derived_by_split"].values()) == 80


# Reproducibility

def test_split_is_deterministic_across_runs():
    records = make_manifest()
    first = build_split(records, classes_config())
    second = build_split(list(records), classes_config())
    assert first["assignments"] == second["assignments"]


def test_split_is_independent_of_manifest_row_order():
    """Row order doesn't matter."""
    records = make_manifest()
    shuffled = list(reversed(records))
    assert (build_split(records, classes_config())["assignments"]
            == build_split(shuffled, classes_config())["assignments"])


def test_different_seed_gives_a_different_split():
    """A different seed gives a different split."""
    records = make_manifest()
    a = build_split(records, classes_config(), seed=1)["assignments"]
    b = build_split(records, classes_config(), seed=2)["assignments"]
    assert a != b


# Validation

def test_duplicate_audio_id_is_rejected():
    records = make_manifest()
    records.append(dict(records[0]))
    with pytest.raises(ManifestError, match="duplicate audio_id"):
        validate_records(records, classes_config())


def test_unknown_class_label_is_rejected():
    records = make_manifest()
    records[0]["class_label"] = "Explosion"  # not one of our classes
    with pytest.raises(ManifestError, match="not one of the"):
        validate_records(records, classes_config())


def test_missing_licence_is_rejected():
    """A clip without a licence is rejected."""
    records = make_manifest()
    records[0]["licence"] = ""
    with pytest.raises(ManifestError, match="licence"):
        validate_records(records, classes_config())


def test_augmented_without_parent_is_rejected():
    """An augmented clip needs a parent."""
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
    """An augmented clip's parent must be an original."""
    records = make_manifest(augmented_per_class=1)
    aug = next(r for r in records if r["original_or_augmented"] == AUGMENTED)
    records.append(make_record("SS-CHAIN-0001", aug["class_label"], status=AUGMENTED,
                               parent=aug["audio_id"]))
    with pytest.raises(ManifestError, match="itself augmented"):
        validate_records(records, classes_config())


def test_strict_rejects_undersized_dataset():
    """--strict rejects fewer than 3000 clips."""
    small = make_manifest(n_per_class=100)
    with pytest.raises(ManifestError, match="strict"):
        assert_strict(small, classes_config())


def test_strict_rejects_unbalanced_dataset():
    """--strict rejects unbalanced classes."""
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


# Build, write, reload

def test_build_write_and_reload_round_trip(tmp_path: Path):
    """The split survives writing and reading back."""
    from audio_dataset.build_split import write_split

    records = make_manifest(augmented_per_class=2)
    split = build_split(records, classes_config())

    out = tmp_path / "split.json"
    write_split(split, out, tmp_path)

    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["assignments"] == split["assignments"]
    assert reloaded["counts"]["originals_totals"] == {"train": 2100, "val": 450, "test": 450}
    assert reloaded["seed"] == 20260923

    # id lists for the training scripts
    for split_name in SPLIT_NAMES:
        ids = (tmp_path / f"{split_name}_ids.txt").read_text(encoding="utf-8").split()
        assert len(ids) == len(set(ids))
        for audio_id in ids:
            assert reloaded["assignments"][audio_id]["split"] == split_name


def test_split_json_records_its_provenance():
    """split.json records how it was made."""
    split = build_split(make_manifest(), classes_config())
    from audio_dataset.build_split import ALGORITHM

    assert split["algorithm"] == ALGORITHM
    assert split["seed"] == 20260923
    assert split["ratios"] == {"train": 0.7, "val": 0.15, "test": 0.15}


def test_manifest_csv_round_trip(tmp_path: Path):
    """The manifest survives a CSV round trip."""
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


# Extra columns

def test_extra_columns_from_a_generator_are_preserved_not_rejected(tmp_path: Path):
    """Extra generator columns are kept, not rejected."""
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
    """The missing-columns error points to the schema and suggests renames."""
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

    assert "manifest_schema.md" in message
    assert "sample_rate -> sampling_rate" in message
    assert "dataset_split" in message
    assert "abuild_split" not in message              # no garbled paths
    assert "build_split.py" in message
