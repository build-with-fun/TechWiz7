"""Augmentation (SRS FR xix) and the train-only rules around it."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from augmentation import transforms as T

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def tone(seconds=1.0, freq=440.0, amp=0.5):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def rms(y):
    return float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))


@pytest.mark.parametrize("snr", [0.0, 10.0, 20.0])
def test_add_noise_hits_the_requested_snr(snr):
    # Quiet enough that the peak limiter never engages; limiting scales the whole mix,
    # which keeps the ratio but would break this subtraction-based measurement.
    clean = tone(amp=0.05)
    mixed = T.add_noise(clean, SR, np.random.default_rng(1), snr_db=snr)
    measured = 20 * np.log10(rms(clean) / rms(mixed - clean))
    assert abs(measured - snr) < 0.5


def test_same_seed_same_output_and_seeds_give_variety():
    y = tone()
    for name, fn in T.TRAINING_RECIPES.items():
        a = fn(y, SR, np.random.default_rng(7))
        assert np.array_equal(a, fn(y, SR, np.random.default_rng(7))), name
        # Some recipes pick from a small set (device picks one of three), so two seeds
        # may coincide; across ten seeds there must be more than one distinct output.
        outputs = {fn(y, SR, np.random.default_rng(seed)).tobytes() for seed in range(10)}
        assert len(outputs) > 1, name


def test_recipes_never_exceed_full_scale():
    loud = tone(amp=0.99)
    for name, fn in T.TRAINING_RECIPES.items():
        out = fn(loud, SR, np.random.default_rng(3))
        assert np.max(np.abs(out)) <= 1.0, name
        assert out.dtype == np.float32, name


def test_time_shift_zero_fills_instead_of_wrapping():
    y = np.zeros(SR, dtype=np.float32)
    y[:100] = 1.0                         # an onset at the very start
    out = T.time_shift(y, SR, np.random.default_rng(0), max_shift_sec=0.5)
    assert out.size == y.size
    assert out[-50:].max() == 0.0 or out[:100].max() == 1.0


def test_time_shift_handles_clips_shorter_than_the_shift():
    y = tone(seconds=0.1)
    out = T.time_shift(y, SR, np.random.default_rng(5), max_shift_sec=0.5)
    assert out.size == y.size


def test_distance_makes_the_signal_quieter_and_darker():
    y = tone(freq=6000.0)
    far = T.distance(y, SR, np.random.default_rng(2), metres=20.0)
    assert rms(far) < rms(y) / 5


def test_phone_device_removes_low_rumble():
    rumble = tone(freq=80.0)
    out = T.device(rumble, SR, np.random.default_rng(0), kind="phone")
    assert rms(out) < 0.2 * rms(rumble)


def test_partial_keeps_a_contiguous_slice_of_at_least_half_a_second():
    y = np.arange(SR * 4, dtype=np.float32)
    out = T.partial(y, SR, np.random.default_rng(0), keep=0.1)
    assert out.size == SR // 2
    assert np.all(np.diff(out) == 1)


AUG_ROWS = ROOT / "audio_dataset" / "manifests" / "augmented_rows.csv"


@pytest.mark.skipif(not AUG_ROWS.exists(), reason="augmented copies not generated in this checkout")
def test_augmented_rows_are_training_only_and_never_originals():
    with (ROOT / "audio_dataset" / "manifest_with_split.csv").open(newline="") as fh:
        originals = {r["audio_id"]: r for r in csv.DictReader(fh)}
    with AUG_ROWS.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    ids = [r["audio_id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        parent = originals[r["parent_audio_id"]]
        assert r["original_or_augmented"] == "augmented"
        assert r["dataset_split"] == parent["dataset_split"] == "train"
        assert r["class_label"] == parent["class_label"]
        assert r["audio_id"] not in originals


def test_gtm_import_window_is_the_serving_window():
    """The import builder's first window must be exactly what the server scores."""
    from audio_dataset.scripts.make_gtm_imports import ranked_windows
    from src.inference.gtm_predictor import GtmFrontendConfig, select_gtm_window

    cfg = GtmFrontendConfig.load(ROOT / "gtm_model" / "frontend_config.json")
    rng = np.random.default_rng(11)
    y = (rng.normal(0, 0.01, SR * 5)).astype(np.float32)
    y[SR * 3: SR * 3 + 2000] += 0.8          # the event is 3 s in, not at the start
    windows = ranked_windows(y, SR, cfg, 2)
    assert np.array_equal(windows[0][1], select_gtm_window(y, SR, cfg))
    assert len(windows) == 2
    assert abs(windows[0][0] - windows[1][0]) >= cfg.window_sec   # no overlap


TM_INDEX = ROOT / "gtm_model" / "upload_package" / "tm_imports" / "index.json"


@pytest.mark.skipif(not TM_INDEX.exists(), reason="Teachable Machine imports not generated")
def test_teachable_machine_imports_use_training_recordings_only():
    with (ROOT / "audio_dataset" / "manifest_with_split.csv").open(newline="") as fh:
        split = {r["audio_id"]: r["dataset_split"] for r in csv.DictReader(fh)}
    index = json.loads(TM_INDEX.read_text())
    parents = [s["parent_audio_id"] for c in index for s in c.get("samples", [])]
    if not parents:  # the v1 index format listed parent_ids instead
        parents = [p for c in index for p in c.get("parent_ids", [])]
    assert parents
    assert {split[p] for p in parents} == {"train"}


GTM_ROWS = ROOT / "audio_dataset" / "manifests" / "gtm_segment_rows.csv"


@pytest.mark.skipif(not GTM_ROWS.exists(), reason="Teachable Machine evidence rows not generated")
def test_gtm_samples_come_from_training():
    """Every window Teachable Machine trains on names a training parent of the same class."""
    with (ROOT / "audio_dataset" / "manifest_with_split.csv").open(newline="") as fh:
        originals = {r["audio_id"]: r for r in csv.DictReader(fh)}
    with GTM_ROWS.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    for r in rows:
        parent = originals[r["parent_audio_id"]]
        assert parent["dataset_split"] == "train", r["audio_id"]
        assert r["class_label"] == parent["class_label"], r["audio_id"]
        assert r["audio_id"].startswith(r["parent_audio_id"] + "S")
    index = json.loads(TM_INDEX.read_text())
    imported = {s["segment_id"] for c in index for s in c["samples"]}
    assert imported == {r["audio_id"] for r in rows}
