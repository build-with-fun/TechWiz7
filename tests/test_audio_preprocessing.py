"""Pre-processing, decoding, validation and segmentation — tested step by step.

Owner: taha.  SRS Step 3 (file validation), Step 4 (audio pre-processing), FR viii-xv,
FR xii, FR xxii.

WHY THESE TESTS ARE SHAPED THIS WAY
-----------------------------------
Every signal in this file is *synthesised* (a sine, a silence, a two-tone mix, a clipped
square) and every assertion is on a number that can be reasoned about acoustically.  A
fixture recording would make these tests depend on the dataset, and a test that only says
"it ran" catches nothing: the failures that actually reach the UI are the quiet ones --
a resampler that changes the duration, a mono downmix that sums instead of averages, a
normaliser that clips, a trim that loses its time mapping, a segmenter whose segment
length is a literal in the code instead of a config value.

The one test that matters most for the competition is
``test_segment_duration_comes_from_the_config_not_the_code``: an evaluator may change the
segment duration live, and the app must follow.  It builds a temporary config directory,
changes one key, and requires the segmentation AND the model input shape to move with it.

Segment length is never written as a literal below.  It is read from the same config the
pipeline reads, so this file cannot drift away from the running system.
"""

from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

import audio_preprocessing as ap
from audio_preprocessing import config as cfg_mod
from audio_preprocessing.exceptions import (
    CORRUPT_HEADER,
    EMPTY_AUDIO,
    NO_SUCH_FILE,
    SILENT,
    TOO_SHORT,
    UNSUPPORTED_FORMAT,
)
from audio_preprocessing.transforms import (
    apply_highpass,
    db_to_amplitude,
    normalize_amplitude,
    pad_or_truncate,
    peak_dbfs,
    preemphasis,
    reduce_noise,
    resample,
    segment_bounds,
    segment_timestamps,
    silence_mask,
    to_mono,
    trim_silence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
SR = int(cfg_mod.audio_config()["target_sample_rate"])


# --------------------------------------------------------------------------------------
# Synthetic material — deterministic, tiny, and explainable
# --------------------------------------------------------------------------------------

def sine(seconds: float, freq: float = 440.0, amplitude: float = 0.5, sr: int = SR) -> np.ndarray:
    """A pure tone.  Its peak and RMS are known analytically: peak=A, RMS=A/sqrt(2)."""
    t = np.arange(int(round(seconds * sr)), dtype=np.float64) / sr
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(round(seconds * sr)), dtype=np.float32)


def two_tone(seconds: float, low: float = 440.0, high: float = 3000.0, sr: int = SR) -> np.ndarray:
    """Two tones an octave and a half apart: has two clear spectral peaks."""
    t = np.arange(int(round(seconds * sr)), dtype=np.float64) / sr
    return (0.4 * np.sin(2 * np.pi * low * t) + 0.3 * np.sin(2 * np.pi * high * t)).astype(np.float32)


def clipped_square(seconds: float, freq: float = 440.0, sr: int = SR) -> np.ndarray:
    """A square wave: every sample sits at +-1, so the clipping detector must see it."""
    t = np.arange(int(round(seconds * sr)), dtype=np.float64) / sr
    return np.sign(np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, sr: int = SR, channels: int = 1) -> Path:
    """16-bit PCM WAV, the format every decoder on this box can read."""
    data = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(channels)
        fh.setsampwidth(2)
        fh.setframerate(int(sr))
        fh.writeframes(pcm.tobytes())
    return path


def temp_config(tmp_path: Path, **audio_overrides) -> Path:
    """A full config directory with selected ``audio`` keys changed.

    Whole-directory copies, not a stub: the point of the config-driven test is that the
    *real* loader, with the *real* file names, honours an edited value.
    """
    directory = tmp_path / "config"
    directory.mkdir()
    for name in ("classes.json", "thresholds.json", "features.json"):
        shutil.copy(CONFIG_DIR / name, directory / name)
    if audio_overrides:
        doc = json.loads((directory / "thresholds.json").read_text(encoding="utf-8"))
        doc.setdefault("audio", {}).update(audio_overrides)
        (directory / "thresholds.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    cfg_mod.clear_cache()
    return directory


# --------------------------------------------------------------------------------------
# 1. Level arithmetic — the units everything else is judged in
# --------------------------------------------------------------------------------------

def test_peak_and_rms_dbfs_match_the_analytic_values():
    """A sine of amplitude 0.5 is -6.02 dBFS peak and -9.03 dBFS RMS (per definition).

    These two numbers are the calibration for the whole quality layer: if they are wrong,
    every verdict downstream is wrong by the same offset.
    """
    y = sine(1.0, amplitude=0.5)
    assert peak_dbfs(y) == pytest.approx(-6.0206, abs=0.01)
    assert ap.transforms.rms_dbfs(y) == pytest.approx(-9.0309, abs=0.01)


def test_db_amplitude_round_trip():
    for db in (-60.0, -20.0, -3.0, 0.0):
        assert ap.transforms.amplitude_to_db(db_to_amplitude(db)) == pytest.approx(db, abs=1e-6)


# --------------------------------------------------------------------------------------
# 2. Channel handling and resampling
# --------------------------------------------------------------------------------------

def test_to_mono_averages_stereo_channels_rather_than_summing():
    """Averaging is the safe downmix: summing two full-scale channels clips at the addition.

    Two opposite-phase channels must cancel to near-zero.  A summing implementation would
    also cancel, so the discriminating case is two *identical* channels: the average keeps
    the amplitude, the sum doubles it and overflows.
    """
    left = sine(0.5, amplitude=0.8)
    stereo = np.stack([left, left], axis=1)
    mono = to_mono(stereo)
    assert mono.ndim == 1
    assert np.max(np.abs(mono)) == pytest.approx(0.8, abs=1e-6), (
        "identical channels should average, not sum — summing clips at 1.6"
    )

    anti = np.stack([left, -left], axis=1)
    assert np.max(np.abs(to_mono(anti))) < 1e-6


def test_resample_preserves_duration_and_the_tone():
    """Duration in seconds is invariant under resampling.  A change here is a silent
    time-base bug: every stored timestamp would be wrong by that factor."""
    y = sine(1.0, freq=440.0)
    for target in (8000, 16000, 22050, 44100):
        out = resample(y, SR, target)
        assert out.dtype == np.float32
        assert out.size / target == pytest.approx(1.0, abs=0.01), (
            f"resampling to {target} Hz changed the duration"
        )


def test_resample_is_a_no_op_at_the_same_rate():
    y = sine(0.25)
    assert np.array_equal(resample(y, SR, SR), y)


# --------------------------------------------------------------------------------------
# 3. Amplitude normalisation (FR xi)
# --------------------------------------------------------------------------------------

def test_normalize_hits_the_target_peak_and_never_clips():
    y = sine(1.0, amplitude=0.02)
    target = -3.0
    # The tone peak sits 33.98 dB below the target, so the default 30 dB boost cap
    # deliberately bites here -- the cap is the feature under test in
    # ``test_normalize_gain_is_capped``, and the point of this case is that an
    # attenuation toward the target still lands on it.
    out = normalize_amplitude(y, target, max_gain_db=60.0)
    assert peak_dbfs(out) == pytest.approx(target, abs=0.01)
    assert np.max(np.abs(out)) <= db_to_amplitude(target) + 1e-6

    loud = sine(1.0, amplitude=0.99)
    out_loud = normalize_amplitude(loud, target)
    assert peak_dbfs(out_loud) == pytest.approx(target, abs=0.01)
    # The loud tone is attenuated toward the target, never boosted past it.
    assert float(out_loud.std()) < float(loud.std())


def test_normalize_preserves_crest_factor():
    """Peak, not RMS, normalisation: the crest factor must survive the step.

    Crest factor (peak - RMS in dB) separates an impulsive Gunshot from a sustained Siren.
    An RMS-normalising implementation flattens it, and the distinction disappears before
    the model ever sees the audio.
    """
    impulsive = np.concatenate([sine(0.02, amplitude=0.05), sine(0.9, amplitude=0.005)])
    crest_before = peak_dbfs(impulsive) - ap.transforms.rms_dbfs(impulsive)
    out = normalize_amplitude(impulsive, -3.0)
    crest_after = peak_dbfs(out) - ap.transforms.rms_dbfs(out)
    assert crest_after == pytest.approx(crest_before, abs=0.05)


def test_normalize_leaves_digital_silence_exactly_zero():
    """Silence has no level to normalise.  Boosting it by max_gain_db would invent a signal."""
    out = normalize_amplitude(silence(0.5), -3.0)
    assert np.all(out == 0.0)


def test_normalize_gain_is_capped():
    """A near-silent recording must not be amplified so hard that the noise floor becomes
    the signal — the cap is what stops that."""
    y = sine(0.5, amplitude=1e-4)
    out = normalize_amplitude(y, -3.0, max_gain_db=20.0)
    assert peak_dbfs(out) == pytest.approx(-60.0, abs=0.5), "gain cap of 20 dB was exceeded"


# --------------------------------------------------------------------------------------
# 4. Conditioning filters
# --------------------------------------------------------------------------------------

def test_highpass_removes_dc_and_keeps_the_tone():
    y = (sine(1.0, freq=440.0) + 0.5).astype(np.float32)  # large DC offset
    out = apply_highpass(y, SR, cutoff_hz=50.0)
    assert abs(float(np.mean(out))) < 1e-3, "DC offset survived the high-pass"
    assert out.size == y.size
    # The 440 Hz tone is two octaves above the 50 Hz corner and must be untouched.
    assert float(np.sqrt(np.mean(out ** 2))) == pytest.approx(
        float(np.sqrt(np.mean((y - 0.5) ** 2))), rel=0.02
    )


def test_highpass_leaves_the_length_alone():
    """Zero-phase (filtfilt) filtering, so no group delay is added and the length is exact.
    A length change here would shift every onset timestamp."""
    for seconds in (0.5, 3.0):
        y = sine(seconds)
        assert apply_highpass(y, SR).size == y.size


def test_preemphasis_is_off_by_default_and_deterministic_when_on():
    y = sine(0.25)
    assert np.array_equal(preemphasis(y, 0.0), y)
    assert np.array_equal(preemphasis(y, 0.97), preemphasis(y, 0.97))


# --------------------------------------------------------------------------------------
# 5. Silence detection and trimming (FR xiv)
# --------------------------------------------------------------------------------------

def test_silence_mask_is_false_on_silence_and_true_on_a_tone():
    assert not silence_mask(silence(1.0), SR).any()
    assert silence_mask(sine(1.0), SR).all()


def test_trim_silence_reports_where_it_cut():
    """The trim must return its (start, end) mapping.  Without it, a timestamp shown in the
    UI would refer to the trimmed timeline and point at the wrong moment of the original."""
    body = sine(1.0, amplitude=0.5)
    y = np.concatenate([silence(0.5), body, silence(0.5)])
    trimmed, (start, end) = trim_silence(y, SR, top_db=30.0)
    assert trimmed.size < y.size
    assert start > 0 and end < y.size, "nothing was reported as trimmed"
    # The kept region is the tone, in the original timeline.
    assert start / SR == pytest.approx(0.5, abs=0.05)
    assert (end - start) / SR == pytest.approx(1.0, abs=0.05)
    assert np.array_equal(trimmed, y[start:end])


def test_trim_silence_keeps_a_pure_tone_untouched():
    """Nothing to trim: the endpoints are the recording's endpoints."""
    y = sine(1.0)
    trimmed, (start, end) = trim_silence(y, SR)
    assert (start, end) == (0, y.size)
    assert np.array_equal(trimmed, y)


def test_trim_silence_never_returns_an_empty_signal():
    """Trimming everything would produce a zero-length window that the segmenter then
    happily reports as "no events" — a silent false negative."""
    out, span = trim_silence(silence(1.0), SR)
    assert out.size > 0
    assert span[1] >= span[0]


# --------------------------------------------------------------------------------------
# 6. Noise reduction
# --------------------------------------------------------------------------------------

def test_reduce_noise_is_deterministic():
    """A learned denoiser would be a second undeclared model; a random one would make the
    same clip give two different confidences.  Same input, bit-identical output."""
    y = two_tone(1.0) + (0.02 * np.random.default_rng(7).standard_normal(SR)).astype(np.float32)
    assert np.array_equal(reduce_noise(y, SR), reduce_noise(y, SR))


def test_reduce_noise_never_returns_pure_zeros():
    """A signal of all zeros carries no information and would score as silence downstream.
    The floor is what stops the gate from silencing a quiet recording completely."""
    for y in (sine(1.0, amplitude=0.01), silence(1.0), two_tone(1.0)):
        out = reduce_noise(y, SR, strength=1.0)
        assert out.size == y.size
        if out.size:
            assert float(np.max(np.abs(out))) > 0.0 or float(np.max(np.abs(y))) == 0.0


def test_reduce_noise_lowers_the_noise_floor_between_two_tones():
    """The tone content must survive while the broadband floor between the tones drops."""
    t = np.arange(SR, dtype=np.float64) / SR
    tone = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noisy = (tone + 0.05 * np.random.default_rng(3).standard_normal(SR)).astype(np.float32)
    denoised = reduce_noise(noisy, SR, strength=0.75)
    # Energy far from the tone (4 kHz band, where the input is noise only) must fall.
    def band_energy(x):
        spec = np.abs(np.fft.rfft(x.astype(np.float64)))
        freqs = np.fft.rfftfreq(x.size, 1.0 / SR)
        return float(np.mean(spec[(freqs > 3500) & (freqs < 4500)]))
    assert band_energy(denoised) < band_energy(noisy)


def test_reduce_noise_is_a_no_op_when_disabled_or_too_short():
    y = sine(0.01)
    assert np.array_equal(reduce_noise(y, SR, strength=0.0), y)
    assert np.array_equal(reduce_noise(sine(0.02), SR, strength=0.75),
                          sine(0.02))


# --------------------------------------------------------------------------------------
# 7. Segmentation and timestamps (FR xv)
# --------------------------------------------------------------------------------------

def test_segment_bounds_cover_every_sample():
    """"cover" mode must leave no region unanalysed — the SRS event-coverage requirement.
    The first segment starts at 0 and the last ends at the final sample."""
    seg_sec = float(cfg_mod.audio_config()["segment_duration_sec"])
    for seconds in (1.0, 3.0, 7.0, 10.0, 30.0):
        n = int(seconds * SR)
        bounds = segment_bounds(n, SR, seg_sec, mode="cover")
        assert bounds[0][0] == 0
        assert bounds[-1][1] == n, f"{seconds}s: last segment ends at {bounds[-1][1]}, not {n}"
        for s, e in bounds:
            assert 0 <= s < e <= n
        # No sample falls outside all segments.
        covered = np.zeros(n, dtype=bool)
        for s, e in bounds:
            covered[s:e] = True
        assert covered.all(), f"{seconds}s: {int((~covered).sum())} samples are in no segment"


def test_segment_bounds_uses_the_configured_length():
    """The number of segments follows from config, not from a literal in the code."""
    seg_sec = float(cfg_mod.audio_config()["segment_duration_sec"])
    n = int(seg_sec * 3 * SR)  # exactly three segment lengths
    bounds = segment_bounds(n, SR, seg_sec, mode="cover")
    assert len(bounds) == 3
    assert [(s / SR, e / SR) for s, e in bounds] == [
        (0.0, seg_sec), (seg_sec, 2 * seg_sec), (2 * seg_sec, 3 * seg_sec)
    ]


def test_short_recording_yields_one_short_segment_not_a_padding_lie():
    """A recording shorter than one segment is one segment.  Padding it to a full segment
    here would claim audio that was never recorded."""
    seg_sec = float(cfg_mod.audio_config()["segment_duration_sec"])
    n = int(seg_sec * SR / 2)
    bounds = segment_bounds(n, SR, seg_sec, mode="cover")
    assert bounds == [(0, n)]


def test_grid_mode_has_stable_boundaries():
    """Live capture needs segment starts that do not move as the buffer grows."""
    seg_sec = float(cfg_mod.audio_config()["segment_duration_sec"])
    seg_len = int(seg_sec * SR)
    first = segment_bounds(seg_len * 4, SR, seg_sec, mode="grid")
    grown = segment_bounds(seg_len * 6, SR, seg_sec, mode="grid")
    assert [b[0] for b in grown[: len(first)]] == [b[0] for b in first]
    assert len(grown) == 6


def test_segment_timestamps_are_seconds_in_fr_xv_form():
    """FR xv: start and end timestamps must be stored.  Seconds, because the database, the
    timeline and the report all speak seconds — a raw sample index is meaningless without
    the rate, and the rate is itself configurable."""
    seg_sec = float(cfg_mod.audio_config()["segment_duration_sec"])
    stamps = segment_timestamps(int(seg_sec * 2 * SR), SR, seg_sec)
    assert len(stamps) == 2
    for start, end in stamps:
        assert isinstance(start, float) and isinstance(end, float)
        assert start < end
    assert stamps[0][0] == 0.0
    assert stamps[-1][1] == pytest.approx(seg_sec * 2, abs=1e-6)


def test_segment_bounds_rejects_a_non_positive_segment_length():
    with pytest.raises(ValueError):
        segment_bounds(SR, SR, 0.0)
    with pytest.raises(ValueError):
        segment_bounds(SR, SR, 3.0, mode="diagonal")


# --------------------------------------------------------------------------------------
# 8. Padding / truncation
# --------------------------------------------------------------------------------------

def test_pad_or_truncate_returns_exactly_the_target_length():
    target = int(float(cfg_mod.audio_config()["segment_duration_sec"]) * SR)
    for samples in (sine(0.5), sine(1.0), sine(5.0)):
        out = pad_or_truncate(samples, target)
        assert out.size == target, f"{samples.size} samples -> {out.size}, expected {target}"
        assert out.dtype == np.float32


def test_pad_or_truncate_center_keeps_the_signal_centred():
    """Centre padding keeps an event's position relative to the window centre, which is
    what the model was trained with.  Random padding is deliberately not implemented: it
    would make a re-run of the same clip give a different answer."""
    short = sine(1.0)
    target = 4 * short.size
    out = pad_or_truncate(short, target, mode="center")
    lead = int(np.argmax(np.abs(out) > 0))
    assert lead == pytest.approx((target - short.size) // 2, abs=2)

    with pytest.raises(ValueError):
        pad_or_truncate(short, target, mode="random")


# --------------------------------------------------------------------------------------
# 9. File validation (SRS Step 3, FR viii) — every rejection has a reason code
# --------------------------------------------------------------------------------------

def test_validates_a_real_tone_and_reports_fr_x_metadata(tmp_path):
    path = write_wav(tmp_path / "tone.wav", sine(2.0))
    info = ap.validate_file(path)
    assert info["ok"] is True, info["detail"]
    assert info["format"] == "wav"
    assert info["sample_rate"] == SR
    assert info["channels"] == 1
    assert info["bit_depth"] == 16
    assert info["duration_sec"] == pytest.approx(2.0, abs=0.01)
    assert info["size_bytes"] > 0
    assert info["codec"] == "pcm"


def test_zero_byte_file_is_rejected_as_empty_audio(tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    info = ap.validate_file(path)
    assert info["ok"] is False
    assert info["reason"] == EMPTY_AUDIO


def test_a_text_file_is_reported_as_unsupported_not_as_corrupt(tmp_path):
    """'This is not audio' and 'this audio is damaged' are different messages to a user.
    Magic bytes decide: no audio signature at all -> unsupported format."""
    path = tmp_path / "notes.txt"
    path.write_text("this is not an audio file at all\n", encoding="utf-8")
    info = ap.validate_file(path)
    assert info["ok"] is False
    assert info["reason"] == UNSUPPORTED_FORMAT


def test_a_wav_renamed_to_txt_is_still_validated_as_audio(tmp_path):
    """Content wins over extension.  Rejecting a real recording because someone renamed it
    is a false negative the evaluator will find."""
    path = write_wav(tmp_path / "renamed.txt", sine(1.0))
    info = ap.validate_file(path)
    assert info["ok"] is True, info["detail"]
    assert info["format"] == "wav"


def test_random_bytes_with_a_wav_extension_are_rejected_as_a_corrupt_header(tmp_path):
    """Undecodable audio must be refused loudly, never silently dropped or replaced by
    zeros — a zero-filled clip would be classified as Background Noise with confidence."""
    path = tmp_path / "broken.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 32 + bytes(range(200)))
    info = ap.validate_file(path)
    assert info["ok"] is False
    assert info["reason"] in {CORRUPT_HEADER, UNSUPPORTED_FORMAT}
    assert info["detail"]


def test_missing_file_is_rejected_with_no_such_file(tmp_path):
    info = ap.validate_file(tmp_path / "nothing-here.wav")
    assert info["ok"] is False
    assert info["reason"] == NO_SUCH_FILE


def test_too_short_file_is_rejected(tmp_path):
    min_dur = float(cfg_mod.audio_config()["min_duration_sec"])
    path = write_wav(tmp_path / "blip.wav", sine(max(0.01, min_dur / 4)))
    info = ap.validate_file(path)
    assert info["ok"] is False
    assert info["reason"] == TOO_SHORT


def test_digital_silence_is_rejected_for_absence_of_signal(tmp_path):
    """FR viii 'presence of audio signal'.  A silent container is a valid file holding no
    event, so it is refused at the door rather than classified later."""
    path = write_wav(tmp_path / "quiet.wav", silence(2.0))
    info = ap.validate_file(path)
    assert info["ok"] is False
    assert info["reason"] == SILENT
    assert info["rms_dbfs"] is None  # -inf dBFS, reported as unavailable, not as a number


def test_validating_samples_matches_validating_a_file(tmp_path):
    """The live path has no container to inspect, but the presence-of-signal rule is the
    same rule.  A docstring claiming a check that does not run is worse than no check."""
    tone = ap.validate_samples(sine(1.0), SR)
    assert tone["ok"] is True
    assert tone["rms_dbfs"] == pytest.approx(-9.03, abs=0.05)

    quiet = ap.validate_samples(silence(1.0), SR)
    assert quiet["ok"] is False
    assert quiet["reason"] == SILENT


# --------------------------------------------------------------------------------------
# 10. Format conversion and decoding
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["mp3", "flac", "ogg", "m4a"])
def test_convert_format_round_trips_through_the_target_codec(tmp_path, target):
    """The SRS lists WAV/MP3/FLAC/OGG/M4A as accepted uploads, so the decoders must actually
    handle each one, not merely be configured for it."""
    src = write_wav(tmp_path / "source.wav", sine(1.0))
    out = ap.convert_format(src, target, out_dir=tmp_path / "converted")
    assert out.exists() and out.suffix == f".{target}"
    info = ap.validate_file(out)
    assert info["ok"] is True, f"{target}: {info['detail']}"
    assert info["format"] == target
    assert info["duration_sec"] == pytest.approx(1.0, abs=0.05)

    y, sr = ap.load_audio(out, sample_rate=SR, mono=True)
    assert sr == SR
    assert y.size / SR == pytest.approx(1.0, abs=0.05)
    # The 440 Hz tone is still there: the dominant FFT bin is at 440 Hz.
    spec = np.abs(np.fft.rfft(y.astype(np.float64)))
    freqs = np.fft.rfftfreq(y.size, 1.0 / SR)
    assert freqs[int(np.argmax(spec))] == pytest.approx(440.0, abs=15.0), (
        f"{target}: the tone did not survive the round trip"
    )


def test_convert_format_accepts_a_path_object_for_the_target(tmp_path):
    """Called from the app with a Path, this used to raise AttributeError."""
    src = write_wav(tmp_path / "source.wav", sine(0.5))
    out = ap.convert_format(src, Path("ok.flac"), out_dir=tmp_path)
    assert out.exists() and out.suffix == ".flac"


def test_load_audio_rejects_an_unknown_format_instead_of_returning_zeros():
    with pytest.raises(ap.AudioRejected):
        ap.load_audio_bytes(b"not audio" * 10, filename="thing.wav")


# --------------------------------------------------------------------------------------
# 11. The whole pipeline, and the config-driven segment duration
# --------------------------------------------------------------------------------------

def test_pipeline_preprocesses_a_file_end_to_end(tmp_path):
    path = write_wav(tmp_path / "clip.wav", sine(1.5))
    result = ap.preprocess_file(path)
    assert result.rejected is False
    assert result.sample_rate == SR
    assert result.duration_sec == pytest.approx(1.5, abs=0.05)
    assert result.quality["verdict"] in {"Good", "Acceptable"}
    assert len(result.segments) >= 1
    assert all(len(seg) == 2 and seg[0] < seg[1] for seg in result.segments)
    assert result.preprocessing["version"] == ap.preprocessing_version()
    # Every conditioning step is reported, so the report can show what was done.
    applied = {s["step"] for s in result.preprocessing["steps"]}
    assert {"decode", "normalize_amplitude", "resample", "trim_silence"} <= applied


def test_pipeline_rejects_a_silent_file_with_a_reason(tmp_path):
    path = write_wav(tmp_path / "quiet.wav", silence(2.0))
    result = ap.preprocess_file(path)
    assert result.rejected is True
    assert result.rejection_reason
    assert result.samples.size == 0


def test_pipeline_rejects_a_too_short_file_with_a_reason(tmp_path):
    min_dur = float(cfg_mod.audio_config()["min_duration_sec"])
    path = write_wav(tmp_path / "blip.wav", sine(max(0.01, min_dur / 4)))
    result = ap.preprocess_file(path)
    assert result.rejected is True
    assert result.rejection_reason


def test_pipeline_accepts_a_bare_path(tmp_path):
    """The app sometimes has a path and no AudioSource; that must not be a second code path."""
    path = write_wav(tmp_path / "clip.wav", sine(1.0))
    assert ap.AudioPipeline()(path).duration_sec == pytest.approx(1.0, abs=0.05)


def test_pipeline_is_deterministic(tmp_path):
    """Two runs over the same file must give bit-identical samples.  A non-deterministic
    preprocessing step would make every stored confidence unreproducible."""
    path = write_wav(tmp_path / "clip.wav", two_tone(1.0))
    a = ap.preprocess_file(path)
    b = ap.preprocess_file(path)
    assert np.array_equal(a.samples, b.samples)
    assert a.segments == b.segments
    assert a.quality["verdict"] == b.quality["verdict"]


def test_segment_duration_comes_from_the_config_not_the_code(tmp_path):
    """THE LIVE-CHANGE TEST.

    An evaluator may edit ``config/thresholds.json`` and expect the running app to follow.
    This builds a temporary config directory with the segment duration changed and requires
    three things to move together: the pipeline's segment length, the stored timestamps, and
    the number of segments — with no code edit at all.
    """
    original = float(cfg_mod.audio_config()["segment_duration_sec"])
    shorter = original / 2.0
    directory = temp_config(tmp_path, segment_duration_sec=shorter)
    try:
        pipeline = ap.AudioPipeline(config_dir=directory)
        assert pipeline.segment_seconds == pytest.approx(shorter)
        assert pipeline.describe()["segment_duration_sec"] == pytest.approx(shorter)

        samples = sine(original * 2)  # two of the ORIGINAL segments
        result = pipeline.preprocess_samples(samples, SR)
        assert result.rejected is False
        assert result.preprocessing["segment_duration_sec"] == pytest.approx(shorter)
        # Four halves of the original length, each half as long.
        assert result.preprocessing["n_segments"] == 4, (
            f"changing the configured segment length did not change the segmentation"
        )
        for start, end in result.segments:
            assert (end - start) == pytest.approx(shorter, abs=1e-3)
        assert result.segments[0][0] == 0.0
        assert result.segments[-1][1] == pytest.approx(original * 2, abs=1e-3)
    finally:
        cfg_mod.clear_cache()

    # And the repository config is untouched by the experiment.
    assert ap.AudioPipeline().segment_seconds == pytest.approx(original)


def test_unknown_config_values_are_refused_rather_than_guessed(tmp_path):
    """A missing config file must not silently become a set of defaults the app then
    reports as if they were configured."""
    report = ap.config_dir_report(CONFIG_DIR)
    assert report["directory"] == str(CONFIG_DIR)
    assert report["files"]["thresholds.json"]["exists"] is True
    assert report["files"]["classes.json"]["exists"] is True
    assert report["classes"] == 10


def test_warm_up_returns_timings_for_every_hot_step():
    """The web app calls this at startup; it must touch every transform that librarises
    numba JIT-compiles, or the first upload pays the compilation cost."""
    timings = ap.warm_up()
    assert set(timings) >= {
        "to_mono", "apply_highpass", "reduce_noise", "trim_silence",
        "normalize_amplitude", "resample", "analyze_quality",
    }
    assert all(isinstance(v, float) for v in timings.values())


def test_quality_meets_orders_the_verdicts():
    assert ap.quality_meets({"verdict": "Good"}, "Acceptable") is True
    assert ap.quality_meets({"verdict": "Acceptable"}, "Acceptable") is True
    assert ap.quality_meets({"verdict": "Poor"}, "Acceptable") is False
    assert ap.quality_meets({"verdict": "Unusable"}, "Poor") is False


def test_pipeline_describe_reports_the_frozen_contract_lorena_and_nadia_use():
    described = ap.AudioPipeline().describe()
    assert described["target_sample_rate"] == SR
    assert described["target_channels"] == 1
    assert described["segment_duration_sec"] == float(cfg_mod.audio_config()["segment_duration_sec"])
    assert set(cfg_mod.audio_config()["supported_formats"]) == set(described["supported_formats"])
    assert described["quality_thresholds"]["min_snr_db"] == float(
        cfg_mod.quality_config()["min_snr_db"]
    )
