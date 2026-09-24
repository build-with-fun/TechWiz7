"""The signal transforms of SRS Step 4, each one small and separately testable.

Owner: taha.  SRS Step 4, FR xi-xv, lxxii.

Every function here is **deterministic** (no random seeds, no iteration-order dependence)
and **length-explicit**: a transform that can change the sample count says so in its
docstring, because the single most common silent bug in an audio pipeline is a rate or
length mismatch that nothing downstream notices.  Each one has a unit test on a synthetic
array in ``tests/test_audio_preprocessing.py``.
"""

from __future__ import annotations

import numpy as np

from .exceptions import AudioRejected, EMPTY_AUDIO

EPS = 1e-12


# --------------------------------------------------------------------------------------
# Level measurement -- the units everything else is judged in
# --------------------------------------------------------------------------------------

def peak_dbfs(y: np.ndarray) -> float:
    """Peak level in dBFS.  ``0.0`` is full scale; digital silence returns ``-inf``."""
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("-inf")
    peak = float(np.max(np.abs(arr)))
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(peak))


def rms_dbfs(y: np.ndarray) -> float:
    """RMS level in dBFS."""
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("-inf")
    power = float(np.mean(np.square(arr)))
    if power <= 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(power))


def db_to_amplitude(db: float) -> float:
    """dBFS to linear gain."""
    return float(10.0 ** (float(db) / 20.0))


def amplitude_to_db(amplitude: float) -> float:
    """Linear gain to dBFS (``0`` amplitude -> ``-inf``)."""
    if amplitude <= 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(amplitude))


# --------------------------------------------------------------------------------------
# Rate and channel layout
# --------------------------------------------------------------------------------------

def to_mono(y: np.ndarray) -> np.ndarray:
    """Downmix to a contiguous 1-D float32 array by averaging channels (SRS Step 4).

    Averaging rather than taking channel 0 because a stereo recording with the event in
    one channel only would otherwise halve the captured energy.
    """
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample, preserving duration to within one output sample.

    Uses ``soxr`` through librosa (high quality, and the same resampler the training
    pipeline uses, so train and inference cannot drift).
    """
    arr = to_mono(y)
    if int(orig_sr) <= 0 or int(target_sr) <= 0:
        raise AudioRejected(EMPTY_AUDIO, f"invalid sample rate pair {orig_sr} -> {target_sr}")
    if int(orig_sr) == int(target_sr) or arr.size == 0:
        return arr
    import librosa

    out = librosa.resample(arr, orig_sr=int(orig_sr), target_sr=int(target_sr), res_type="soxr_hq")
    return np.ascontiguousarray(out, dtype=np.float32)


def ensure_min_amplitude(y: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Guard against an all-zero array reaching a transform that divides by its level."""
    arr = to_mono(y)
    if arr.size and float(np.max(np.abs(arr))) < floor:
        return np.zeros_like(arr)
    return arr


# --------------------------------------------------------------------------------------
# Amplitude normalisation
# --------------------------------------------------------------------------------------

def normalize_amplitude(
    y: np.ndarray,
    target_peak_dbfs: float = -3.0,
    *,
    max_gain_db: float = 30.0,
    allow_boost: bool = True,
) -> np.ndarray:
    """Peak-normalise to ``target_peak_dbfs`` (SRS Step 4, FR xi).

    Deliberately **peak** normalisation, not RMS: peak normalisation preserves the crest
    factor (the peak-to-RMS ratio), which is itself a discriminator between a gunshot and
    a siren.  An RMS-normalising step would flatten exactly the information the model needs.

    Boosting is capped by ``max_gain_db`` so that amplifying a near-silent recording does
    not raise the noise floor into a false signal; pass ``allow_boost=False`` to attenuate
    only.  Never clips: the output peak is ``<= target_peak_dbfs``.
    """
    arr = to_mono(y)
    if arr.size == 0:
        return arr
    peak = float(np.max(np.abs(arr)))
    if peak <= EPS:
        return arr  # silence has no level to normalise; leave it exactly zero
    target = db_to_amplitude(target_peak_dbfs)
    gain = target / peak
    if not allow_boost:
        gain = min(1.0, gain)
    gain = min(gain, db_to_amplitude(max_gain_db))
    out = arr * float(gain)
    # Guarantee the stated peak even in the face of float32 rounding.
    out_peak = float(np.max(np.abs(out))) if out.size else 0.0
    if out_peak > target and out_peak > EPS:
        out = out * float(target / out_peak)
    return np.ascontiguousarray(out, dtype=np.float32)


def apply_highpass(y: np.ndarray, sample_rate: int, cutoff_hz: float = 50.0, order: int = 4) -> np.ndarray:
    """Remove DC offset and sub-audible rumble (a common artefact of microphone capture).

    Below ~50 Hz there is no information in any of the ten classes, and the energy there
    dominates the Mel bands and corrupts the spectral centroid.
    """
    arr = to_mono(y)
    nyquist = 0.5 * float(sample_rate)
    if arr.size == 0 or cutoff_hz <= 0 or cutoff_hz >= nyquist:
        return arr
    from scipy.signal import butter, sosfiltfilt

    sos = butter(order, cutoff_hz / nyquist, btype="highpass", output="sos")
    # filtfilt doubles the order but has zero phase shift, so onset timing is preserved.
    padlen = min(3 * (2 * len(sos) + 1), max(0, arr.size - 1))
    out = sosfiltfilt(sos, arr.astype(np.float64), padlen=padlen) if padlen > 0 else arr.astype(np.float64)
    return np.ascontiguousarray(out, dtype=np.float32)


def preemphasis(y: np.ndarray, coef: float = 0.0) -> np.ndarray:
    """First-order pre-emphasis.  Off by default (``coef=0``): MFCC of a log-Mel spectrogram
    already applies the right perceptual weighting, and doubling it costs accuracy."""
    arr = to_mono(y)
    if coef <= 0.0 or arr.size < 2:
        return arr
    out = np.empty_like(arr)
    out[0] = arr[0]
    out[1:] = arr[1:] - np.float32(coef) * arr[:-1]
    return np.ascontiguousarray(out, dtype=np.float32)


# --------------------------------------------------------------------------------------
# Silence trimming
# --------------------------------------------------------------------------------------

def silence_mask(
    y: np.ndarray,
    sample_rate: int,
    *,
    frame_length: int = 2048,
    hop_length: int | None = None,
    top_db: float = 30.0,
) -> np.ndarray:
    """Frame-level boolean mask: True where the frame carries signal above ``top_db``
    below the peak frame.  Exposed separately so tests and the UI can show *what* was trimmed.
    """
    import librosa

    arr = to_mono(y)
    hop = hop_length or frame_length // 4
    if arr.size < frame_length:
        return np.ones(1, dtype=bool) if arr.size and float(np.max(np.abs(arr))) > EPS else np.zeros(1, dtype=bool)
    rms = librosa.feature.rms(y=arr, frame_length=frame_length, hop_length=hop, center=True)[0]
    ref = float(np.max(rms)) if rms.size else 0.0
    if ref <= EPS:
        return np.zeros_like(rms, dtype=bool)
    threshold = ref * db_to_amplitude(-abs(top_db))
    return (rms >= threshold).astype(bool)


def trim_silence(
    y: np.ndarray,
    sample_rate: int,
    *,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int | None = None,
    min_keep_sec: float = 0.1,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Trim leading and trailing near-silence (SRS Step 4).

    Only the *ends* are trimmed.  Interior silence is kept because a pause between two
    events is itself a feature, and an interior gap removed here would move the onset
    positions the event timeline is built from.

    Returns ``(trimmed, (start_sample, end_sample))`` so the caller can map trimmed time
    back to original time -- required for honest timestamps in the UI (FR xv).
    """
    arr = to_mono(y)
    if arr.size == 0:
        return arr, (0, 0)
    mask = silence_mask(arr, sample_rate, frame_length=frame_length, hop_length=hop_length, top_db=top_db)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return arr, (0, arr.size)

    hop = hop_length or frame_length // 4
    # ``librosa.feature.rms(center=True)`` pads the signal so frame ``i`` covers samples
    # ``i*hop - frame//2`` .. ``i*hop + frame//2``.  Frame granularity is 2048 samples, so a
    # span anchored to a frame edge is up to a whole frame (128 ms at 16 kHz) off the real
    # onset -- and every UI timestamp derived from it would be off by the same amount.
    first, last = int(idx[0]), int(idx[-1])
    start = max(0, min(arr.size, first * hop - (frame_length // 2)))
    end = min(arr.size, max(start, last * hop + (frame_length // 2)))
    # Refine both edges to the sample: the frame mask already decided *which* region carries
    # signal, this only tightens the edges to the first and last sample that clear the same
    # top_db margin in amplitude, so the span can never widen past what the mask allowed.
    # An edge already flush with the array boundary is left alone -- nothing is padded there,
    # so there is no slop to remove, and a tone that starts on a zero crossing must keep its
    # first sample rather than be shaved back by one.
    peak_sample = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak_sample > EPS and end > start:
        amp_threshold = peak_sample * db_to_amplitude(-abs(top_db))
        mag = np.abs(arr[start:end])
        rising = np.flatnonzero(mag >= amp_threshold)
        if rising.size and start > 0:
            start = start + int(rising[0])
        falling = np.flatnonzero(np.abs(arr[start:end]) >= amp_threshold)
        if falling.size and end < arr.size:
            end = start + int(falling[-1]) + 1
    keep_floor = int(round(min_keep_sec * sample_rate))
    if end - start < keep_floor:
        # Everything was near-silent or the signal is too short to trim safely.
        centre = (start + end) // 2
        start = max(0, min(arr.size - keep_floor, centre - keep_floor // 2))
        end = min(arr.size, start + keep_floor)
    trimmed = arr[start:end]
    return np.ascontiguousarray(trimmed, dtype=np.float32), (start, end)


# --------------------------------------------------------------------------------------
# Noise reduction
# --------------------------------------------------------------------------------------

def reduce_noise(
    y: np.ndarray,
    sample_rate: int,
    *,
    strength: float = 0.75,
    n_fft: int = 1024,
    hop_length: int = 512,
    noise_percentile: float = 20.0,
    prop_decrease: float = 1.0,
) -> np.ndarray:
    """Deterministic spectral-gating noise reduction (SRS Step 4).

    The noise floor is estimated from the quietest ``noise_percentile`` per cent of frames
    by RMS -- a percentile, not a random sample, so the same input always gives the same
    output and the result is reproducible in front of an evaluator.

    ``strength`` in ``[0, 1]`` is how far the gate is pushed: 0 leaves the signal alone,
    1 subtracts the estimated noise floor completely.  The residual is floored at a small
    fraction of the original magnitude so the result is never pure digital silence (which
    would turn a noisy-but-usable clip into a "silent" rejection).

    Deliberately **not** a machine-learning denoiser: a learned denoiser trained on our own
    corpus would be a second model we would have to declare, document and validate.
    """
    arr = to_mono(y)
    strength = float(np.clip(strength, 0.0, 1.0))
    if arr.size < n_fft or strength <= 0.0:
        return arr

    import librosa

    spec = librosa.stft(arr, n_fft=n_fft, hop_length=hop_length, window="hann", center=True)
    mag, phase = np.abs(spec), np.angle(spec)

    frame_energy = np.mean(np.square(mag), axis=0)
    if frame_energy.size == 0 or float(np.max(frame_energy)) <= EPS:
        return arr
    cutoff = float(np.percentile(frame_energy, noise_percentile))
    noise_frames = frame_energy <= max(cutoff, EPS)
    if not np.any(noise_frames):
        noise_frames = frame_energy <= float(np.min(frame_energy))
    noise_profile = np.mean(mag[:, noise_frames], axis=1, keepdims=True)

    # Soft mask: proportional subtraction, floored so no bin is gated to exactly zero.
    floor = 0.05
    ratio = (mag - strength * prop_decrease * noise_profile) / np.maximum(mag, EPS)
    mask = np.clip(ratio, floor, 1.0)

    out = librosa.istft(
        mag * mask * np.exp(1j * phase),
        hop_length=hop_length,
        n_fft=n_fft,
        window="hann",
        center=True,
        length=arr.size,
    )
    if out.size < arr.size:
        out = np.pad(out, (0, arr.size - out.size))
    out = out[: arr.size]
    if not np.isfinite(out).all():
        return arr
    return np.ascontiguousarray(out, dtype=np.float32)


# --------------------------------------------------------------------------------------
# Fixed-duration segmentation (FR xv: start and end timestamps must be stored)
# --------------------------------------------------------------------------------------

def segment_bounds(
    n_samples: int,
    sample_rate: int,
    segment_seconds: float,
    *,
    hop_seconds: float | None = None,
    mode: str = "cover",
) -> list[tuple[int, int]]:
    """Sample-index bounds of every fixed-duration segment.

    Two modes, and the difference matters:

    ``"cover"`` (default, uploads)
        Exactly ``ceil(n / segment)`` segments, evenly spaced from sample 0 to the end,
        so every sample of the recording falls inside at least one segment and the final
        segment ends exactly at the last sample.  This is the minimum number of segments
        that guarantees no region is missed -- which is what the SRS event coverage
        requirement needs -- and it keeps the 30 s clip analysis inside its 8 s budget.

    ``"grid"`` (dataset building, live capture)
        Fixed hop from sample 0.  Boundaries are stable as the buffer grows, which is what
        live incremental processing requires, and what makes a dataset reproducible.
    """
    if n_samples <= 0:
        return []
    seg_len = int(round(float(segment_seconds) * sample_rate))
    if seg_len <= 0:
        raise ValueError(f"segment_seconds must be positive, got {segment_seconds!r}")

    if mode == "grid":
        hop = int(round(float(hop_seconds) * sample_rate)) if hop_seconds else seg_len
        hop = max(1, hop)
        starts = list(range(0, max(1, n_samples - seg_len + 1), hop))
        if not starts:
            starts = [0]
        return [(s, min(n_samples, s + seg_len)) for s in starts]

    if mode != "cover":
        raise ValueError(f"unknown segmentation mode {mode!r}")

    if n_samples <= seg_len:
        return [(0, n_samples)]
    k = int(np.ceil(n_samples / seg_len))
    span = n_samples - seg_len
    hops = np.round(np.linspace(0.0, float(span), num=k)).astype(int)
    bounds: list[tuple[int, int]] = []
    for start in hops:
        s = int(start)
        e = min(n_samples, s + seg_len)
        if bounds and s == bounds[-1][0]:
            continue
        bounds.append((s, e))
    if bounds[-1][1] < n_samples:
        bounds.append((max(0, n_samples - seg_len), n_samples))
    return bounds


def segment_timestamps(
    n_samples: int,
    sample_rate: int,
    segment_seconds: float,
    *,
    hop_seconds: float | None = None,
    mode: str = "cover",
) -> list[tuple[float, float]]:
    """The same segmentation as seconds -- the form FR xv requires to be persisted.

    Seconds, not sample indices, because the timestamps are stored in the database,
    shown in the event timeline and used to build the report; a sample index is only
    meaningful to whoever knows the rate, and the rate is a config value that can change.
    """
    return [
        (round(s / sample_rate, 6), round(e / sample_rate, 6))
        for s, e in segment_bounds(
            n_samples, sample_rate, segment_seconds, hop_seconds=hop_seconds, mode=mode
        )
    ]


def iter_segments(y: np.ndarray, sample_rate: int, segment_seconds: float, **kwargs):
    """Yield ``(index, start_sec, end_sec, samples)`` per segment, for streaming use."""
    arr = to_mono(y)
    bounds = segment_bounds(
        arr.size, sample_rate, segment_seconds,
        hop_seconds=kwargs.get("hop_seconds"), mode=kwargs.get("mode", "cover"),
    )
    for i, (s, e) in enumerate(bounds):
        yield i, round(s / sample_rate, 6), round(e / sample_rate, 6), arr[s:e]


# --------------------------------------------------------------------------------------
# Padding / truncation
# --------------------------------------------------------------------------------------

def pad_or_truncate(
    y: np.ndarray,
    target_length: int,
    *,
    mode: str = "center",
) -> np.ndarray:
    """Force an exact length for a fixed-input model (SRS Step 4 padding/truncation).

    ``mode`` is how padding is placed, and it is a real decision, not cosmetics:

    ``"center"``  pad symmetrically.  Default: an event near a segment edge keeps its
                  time-relation to the window centre, which is what the model was trained
                  with and what keeps onset timing meaningful.
    ``"random"``  not implemented -- deliberately.  Random padding would make inference
                  non-deterministic, and a re-run of the same clip would give a different
                  answer, which is indefensible under evaluator scrutiny.
    """
    arr = to_mono(y)
    if arr.size == target_length:
        return arr
    if arr.size > target_length:
        return np.ascontiguousarray(arr[:target_length], dtype=np.float32)
    missing = target_length - arr.size
    if mode == "center":
        left = missing // 2
    elif mode == "start":
        left = 0
    elif mode == "end":
        left = missing
    else:
        raise ValueError(f"unknown pad mode {mode!r}; only deterministic modes are supported")
    right = missing - left
    return np.ascontiguousarray(np.pad(arr, (left, right), mode="constant"), dtype=np.float32)


def crop_or_pad_seconds(y: np.ndarray, sample_rate: int, seconds: float, **kwargs) -> np.ndarray:
    """``pad_or_truncate`` expressed in seconds, which is how the spec states it."""
    target = int(round(float(seconds) * sample_rate))
    if target <= 0:
        raise ValueError(f"seconds must be positive, got {seconds!r}")
    return pad_or_truncate(y, target, **kwargs)
