#!/usr/bin/env python3
"""
SonicSentinel AI -- "Person Asking for Help" original-clip generator (TTS).

Authority: SRS v1.0 Step 1 (self-created dataset, >=300 unique originals per
class) and the frozen manifest contract audio_dataset/manifest_schema.md.
Class phrase constraint: config/classes.json allowed_help_phrases -- the class
is restricted to the phrases "Help me", "Somebody help", "Please help",
"Call for help", "Emergency" (SRS 1.4 / Step 1).

WHAT THIS PRODUCES
------------------
300 *distinct* original clips of a human voice asking for help, synthesised
with pyttsx3 (offline TTS on this machine, no external generative-AI API --
integrity rule 1.10).  Distinctness is layered:

  phrase       the 5 allowed phrases x punctuation/intensity wording variants
               (>= 20 distinct spoken texts across the corpus)
  voice        pyttsx3 system voice id (deterministically cycled)
  rate/pitch   per-clip values drawn from a seeded RNG
  rendering    pyttsx3 -> WAV via espeak backend, then a seeded post-chain
               (telephone band-pass, slap-back echo, room reverb, background
               bed mixed at a seeded SNR, distance attenuation, mic colouring)

Every clip is written to audio_dataset/synthetic/person_asking_for_help/ with
an SS-HAL-<nnnn> id continuing after the real recordings' numbering, and emits
frozen-schema manifest rows with dataset_split LEFT EMPTY (build_split.py is
the only writer of that column).

Deterministic: same seed -> same bytes on every machine.  Re-running skips
ids already on disk and on the manifest.

Usage:
    python audio_dataset/scripts/generate_help_clips.py --check
    python audio_dataset/scripts/generate_help_clips.py                # 300
    python audio_dataset/scripts/generate_help_clips.py --limit 5     # smoke
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import subprocess
import sys
import tempfile
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DATASET = REPO_ROOT / "audio_dataset"
CLASSES_CONFIG = REPO_ROOT / "config" / "classes.json"

CLASS_LABEL = "Person Asking for Help"
CLASS_CODE = "HAL"
SLUG = "person_asking_for_help"

#: ids continue after the acquired (real) range that starts at 0501
ID_START = 501
TARGET = 300
MASTER_SEED = 20260923

LICENCE = "CC0-1.0"                       # own generated work -> public domain
SOURCE_TAG = "procedural_synthesis:audio_dataset/scripts/generate_help_clips.py"
AUTHOR_TAG = "sonicsentinel procedural synthesis (pyttsx3 TTS, generate_help_clips.py)"
DATE_FETCHED = "2026-09-24"

OUT_DIR = AUDIO_DATASET / "synthetic" / SLUG
ROWS_CSV = AUDIO_DATASET / "manifests" / "help_tts_rows.csv"

# ---------------------------------------------------------------------------
# Phrase inventory: the 5 allowed phrases x intensity/punctuation variants.
# The *spoken text* varies (>= 20 distinct texts) but every variant is one of
# the five allowed phrases in wording, satisfying config/classes.json.
# ---------------------------------------------------------------------------
PHRASE_VARIANTS: list[tuple[str, str]] = [
    # (spoken text, intensity tag)
    ("Help me.", "calm"),
    ("Help me!", "urgent"),
    ("Help me, please.", "pleading"),
    ("Help me. Please.", "pleading"),
    ("HELP ME!", "desperate"),
    ("Somebody help.", "calm"),
    ("Somebody help!", "urgent"),
    ("Somebody help me.", "pleading"),
    ("Somebody, please, help me!", "desperate"),
    ("Somebody help me, please.", "pleading"),
    ("Please help.", "calm"),
    ("Please help me.", "pleading"),
    ("Please, help me!", "urgent"),
    ("Please help me. Hurry.", "desperate"),
    ("Please help, somebody.", "pleading"),
    ("Call for help.", "calm"),
    ("Call for help!", "urgent"),
    ("Please call for help.", "pleading"),
    ("Somebody call for help, please.", "urgent"),
    ("Call for help. Quickly!", "desperate"),
    ("Emergency.", "calm"),
    ("Emergency!", "urgent"),
    ("Emergency! Help me!", "desperate"),
    ("Emergency. Please help me.", "pleading"),
]
# 24 distinct spoken texts >= the 20-variant requirement.

FROZEN_COLUMNS = [
    "audio_id", "filename", "class_label", "source", "source_url", "licence", "author",
    "date_fetched", "duration_sec", "sampling_rate", "channels", "recording_environment",
    "recording_device", "approximate_distance", "original_or_augmented", "parent_audio_id",
    "segment_start_sec", "segment_end_sec", "sha256", "dataset_split",
]

# extra telemetry columns (preserved by the assembler / split builder)
EXTRA_COLUMNS = [
    "seed", "audio_provenance", "sim_environment", "sim_distance_m", "sim_intensity",
    "sim_interference", "voice_name", "speech_rate_wpm", "pitch_hz", "post_chain",
    "phrase", "notes",
]

ENVIRONMENT_MAP = {
    "indoor_room": "indoor",
    "indoor_hall": "indoor",
    "outdoor_open": "outdoor",
    "outdoor_street": "outdoor",
    "car_interior": "vehicle",
}

DEVICES = {
    "phone_builtin": (90.0, 8000.0, 0.12, -58.0),
    "tablet_builtin": (70.0, 15000.0, 0.08, -62.0),
    "lavalier_mic": (40.0, 18000.0, 0.06, -66.0),
    "cctv_mic": (150.0, 5500.0, 0.02, -52.0),
    "usb_condenser_mic": (25.0, 20000.0, 0.10, -68.0),
    "wristband_mic": (120.0, 7000.0, 0.05, -56.0),
}
DEVICE_NAMES = list(DEVICES)

DISTANCE_BANDS = {
    "near_0.5-2m": "near",
    "mid_3-15m": "medium",
    "far_20-60m": "far",
}
#: numeric bounds for simulation (manifest gets the enum value above)
DISTANCE_RANGE = {
    "near": (0.5, 2.0),
    "medium": (3.0, 15.0),
    "far": (20.0, 60.0),
}
INTENSITIES = {"calm": (0.25, 0.45), "urgent": (0.5, 0.75),
               "pleading": (0.35, 0.6), "desperate": (0.7, 0.95)}
RECORDER_GAIN = (0.7, 1.5)


def seed_for(audio_id: str, master_seed: int = MASTER_SEED) -> int:
    digest = hashlib.sha256(f"{master_seed}:{audio_id}".encode()).hexdigest()
    return int(digest[:8], 16)


# ---------------------------------------------------------------------------
# DSP helpers (same primitives/policy as data/generate_corpus.py)
# ---------------------------------------------------------------------------
def _norm(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x / peak if peak > 1e-12 else x


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    from scipy import signal as sps
    nyq = sr / 2.0
    lo, hi = max(float(lo), 1.0), min(float(hi), nyq * 0.98)
    if lo >= hi:
        return x
    sos = sps.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sps.sosfilt(sos, x)


def lowpass(x: np.ndarray, sr: int, fc: float, order: int = 4) -> np.ndarray:
    from scipy import signal as sps
    nyq = sr / 2.0
    fc = min(float(fc), nyq * 0.98)
    if fc <= 0:
        return x
    sos = sps.butter(order, fc / nyq, btype="low", output="sos")
    return sps.sosfilt(sos, x)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = freqs[1] if freqs.size > 1 else 1.0
    spec = spec / np.sqrt(freqs)
    spec[0] = 0.0
    return _norm(np.fft.irfft(spec, n=n))


def reverb(x: np.ndarray, sr: int, rt60: float, wet: float, rng: np.random.Generator) -> np.ndarray:
    """Schroeder reverb (comb + allpass), RMS-preserving (see generate_corpus.py)."""
    from scipy import signal as sps
    out = np.zeros_like(x)
    for ct in (0.0297, 0.0371, 0.0411, 0.0437):
        d = max(int(ct * sr * rng.uniform(0.7, 1.3)), 1)
        g = float(10 ** (-3.0 * d / sr / max(rt60, 0.02)))
        buf = x.copy()
        delayed = np.zeros_like(x)
        delayed[d:] = g * x[:-d]
        buf = x + delayed
        for i in range(d, len(buf)):
            buf[i] += g * buf[i - d]
        out += buf / 4.0
    for at in (0.005, 0.0017):
        d = max(int(at * sr), 1)
        g = 0.5
        y = out.copy()
        for i in range(d, len(y)):
            y[i] = -g * out[i] + out[i - d] + g * y[i - d]
        out = y
    r_in, r_tail = _rms(x), _rms(out)
    if r_tail > 1e-12:
        out = out * (r_in / r_tail)
    return (1.0 - wet) * x + wet * out


def post_chain(y: np.ndarray, sr: int, rng: np.random.Generator, chain: str) -> np.ndarray:
    """Per-clip rendering chain chosen by the seeded RNG."""
    if chain == "clean":
        return y
    if chain == "telephone":
        y = bandpass(y, sr, 300, 3400, order=6)
        y = y + 0.05 * bandpass(pink_noise(len(y), rng), sr, 300, 3400)  # line hiss
        return y
    if chain == "slap_echo":
        d = int(rng.uniform(0.08, 0.22) * sr)
        out = y.copy()
        out[d:] += y[:-d] * rng.uniform(0.25, 0.5)
        return out
    if chain == "small_room":
        return reverb(y, sr, rng.uniform(0.15, 0.35), rng.uniform(0.15, 0.3), rng)
    if chain == "hall":
        return reverb(y, sr, rng.uniform(0.7, 1.6), rng.uniform(0.3, 0.5), rng)
    if chain == "street_bed":
        bed = 0.4 * lowpass(pink_noise(len(y), rng), sr, 900) \
            + 0.15 * bandpass(pink_noise(len(y), rng), sr, 1200, 3000)
        return y + bed * _rms(y) / max(_rms(bed), 1e-12) * rng.uniform(0.15, 0.4)
    if chain == "crowd_bed":
        # murmur-like bed: band-limited noise with slow amplitude wander
        t = np.arange(len(y)) / sr
        wander = 0.6 + 0.4 * np.sin(2 * math.pi * rng.uniform(0.2, 0.8) * t + rng.uniform(0, 6))
        bed = bandpass(pink_noise(len(y), rng), sr, 200, 2500) * wander
        return y + bed * _rms(y) / max(_rms(bed), 1e-12) * rng.uniform(0.15, 0.4)
    return y


# ---------------------------------------------------------------------------
# TTS rendering
# ---------------------------------------------------------------------------
def render_tts(text: str, voice_uri: str, rate: int, pitch_hz: int, out_path: Path,
               sr_target: int = 22050) -> float:
    """Render one clip with pyttsx3 -> WAV, resampled to sr_target. Returns duration."""
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("voice", voice_uri)
    engine.setProperty("rate", rate)
    # pyttsx3/espeak 'pitch' is 0..200 (100 = default); map from an Hz-ish hint
    engine.setProperty("pitch", int(max(0, min(200, pitch_hz))))
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()
    engine.stop()

    if not out_path.exists() or out_path.stat().st_size < 44:
        raise RuntimeError(f"TTS produced no audio for {text!r}")

    data, sr = sf.read(str(out_path), dtype="float64", always_2d=True)
    mono = data.mean(axis=1)
    if sr != sr_target:
        from scipy import signal as sps
        n_out = int(round(len(mono) * sr_target / sr))
        mono = sps.resample(mono, n_out)
    return mono, sr_target


# ---------------------------------------------------------------------------
# Clip plan
# ---------------------------------------------------------------------------
def clip_plan(rng: np.random.Generator, n: int) -> list[dict]:
    """Deterministic per-clip plan: phrase, voice, rate, pitch, post chain."""
    import pyttsx3
    voices = [v for v in pyttsx3.init().getProperty("voices")
              if "english" in (v.name or "").lower() or "en" in (v.id or "").lower()]
    voices = voices[:12] or pyttsx3.init().getProperty("voices")[:12]
    plans = []
    for i in range(n):
        text, intensity = PHRASE_VARIANTS[i % len(PHRASE_VARIANTS)]
        voice = voices[i % len(voices)]
        env = str(rng.choice(list(ENVIRONMENT_MAP), p=[0.3, 0.1, 0.28, 0.2, 0.12]))
        dist = str(rng.choice(list(DISTANCE_RANGE.keys()), p=[0.42, 0.36, 0.22]))
        plans.append({
            "idx": i,
            "text": text,
            "voice_uri": voice.id,
            "voice_name": voice.name,
            "rate": int(rng.integers(110, 190)),          # words per minute
            "pitch": int(rng.integers(30, 170)),           # espeak 0..200 scale
            "env": env,
            "dist_band": dist,              # already the frozen enum value
            "dist_m": float(rng.uniform(*DISTANCE_RANGE[dist])),
            "device": str(rng.choice(list(DEVICES))),
            "intensity": str(rng.choice(list(INTENSITIES), p=[0.25, 0.35, 0.25, 0.15])),
            "chain": str(rng.choice(
                ["clean", "telephone", "slap_echo", "small_room", "hall",
                 "street_bed", "crowd_bed"],
                p=[0.12, 0.16, 0.14, 0.18, 0.12, 0.14, 0.14])),
            "snr_db": float(rng.uniform(14, 30)),
        })
    return plans


def build_clip(plan: dict, rng: np.random.Generator, wav_out: Path) -> tuple[float, int, int]:
    """Render TTS, apply post chain + channel simulation; write WAV. Returns (dur, sr, ch)."""
    tmp = Path(tempfile.mkdtemp(prefix="ss_help_")) / "raw.wav"
    mono, sr = render_tts(plan["text"], plan["voice_uri"], plan["rate"],
                          plan["pitch"], tmp, sr_target=22050)

    # trim leading/trailing silence (espeak pads), keep >= 0.6 s
    env = np.abs(mono)
    thr = 0.02 * env.max()
    nz = np.where(env > thr)[0]
    if nz.size:
        pad = int(0.05 * sr)
        mono = mono[max(0, nz[0] - pad): min(len(mono), nz[-1] + pad)]
    if len(mono) < int(0.6 * sr):
        pad = np.zeros(int(0.6 * sr) - len(mono))
        mono = np.concatenate([mono, pad])

    # intensity + post chain + mic + distance (order mirrors simulate_channel)
    lo, hi = INTENSITIES[plan["intensity"]]
    y = _norm(mono) * float(rng.uniform(lo, hi))
    y = post_chain(y, sr, rng, plan["chain"])
    l, hi_f, presence, floor_db = DEVICES[plan["device"]]
    from scipy import signal as sps
    nyq = sr / 2.0
    sos_hp = sps.butter(2, max(l, 1.0) / nyq, btype="high", output="sos")
    sos_lp = sps.butter(4, min(hi_f, nyq * 0.98) / nyq, btype="low", output="sos")
    y = sps.sosfilt(sos_lp, sps.sosfilt(sos_hp, y))
    if presence > 0:
        y = y + presence * bandpass(y, sr, 2500, 5000, order=2)
    d = max(plan["dist_m"], 0.4)
    y *= (1.0 / d) ** 0.85
    air_cut = float(np.clip(19000.0 / (1.0 + d / 6.0), 1800.0, 19000.0))
    y = lowpass(y, sr, air_cut, order=2)
    # background bed (interference) at the drawn SNR
    sig_rms = _rms(y) + 1e-9
    bed = 0.5 * lowpass(pink_noise(len(y), rng), sr, 800) \
        + 0.3 * bandpass(pink_noise(len(y), rng), sr, 1000, 4000)
    y += bed * sig_rms / (10 ** (plan["snr_db"] / 20.0)) / max(_rms(bed), 1e-12) * sig_rms
    # mic self-noise then gain staging
    y = y + (10 ** (floor_db / 20.0)) * rng.standard_normal(len(y))
    y = y * float(rng.uniform(*RECORDER_GAIN))
    # Level floor: the distance attenuation above can push far clips below the
    # corpus silence gate (RMS -50 dBFS, audio_preprocessing/io.py), which would
    # waste the clip.  Renormalise the *signal* to a floor that keeps every clip
    # clearly audible while preserving the relative loudness spread between clips.
    # Applied after the whole chain so the bed/noise stay proportional (SNR is
    # preserved -- bed was mixed relative to sig_rms).
    rms_after = _rms(y)
    target_floor = 0.02  # -34 dBFS, comfortably above the -50 dBFS gate
    if rms_after < target_floor:
        y = y * (target_floor / max(rms_after, 1e-9))
    if float(np.max(np.abs(y))) > 0.95:
        y = np.tanh(y) * 0.97
    y = np.clip(y, -1.0, 1.0)

    # resample to 16 kHz mono (the corpus's dominant format)
    y16 = signal_resample(y, sr, 16000)
    wav_out = OUT_DIR / f"{plan['audio_id']}.wav"
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_out), y16.astype(np.float64), 16000, subtype="PCM_16")
    dur = len(y16) / 16000
    try:
        tmp.unlink(missing_ok=True)
        tmp.parent.rmdir()
    except OSError:
        pass
    return dur, 16000, 1


def signal_resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    from scipy import signal as sps
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    return sps.resample(x, n_out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def existing_ids() -> set[int]:
    out: set[int] = set()
    if OUT_DIR.exists():
        for p in OUT_DIR.glob("SS-HAL-*.wav"):
            try:
                out.add(int(p.stem.rsplit("-", 1)[-1]))
            except ValueError:
                continue
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=TARGET)
    ap.add_argument("--check", action="store_true", help="print plan, write nothing")
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    args = ap.parse_args(argv)

    have = existing_ids()
    todo = [n for n in range(ID_START, ID_START + TARGET) if n not in have][: args.limit]
    print(f"help-clip generator: {len(have)} on disk, {len(todo)} to render "
          f"(target {TARGET}, ids SS-HAL-{ID_START:04d}+)")
    if not todo:
        print("nothing to do")
    if args.check or not todo:
        return 0

    plans = clip_plan(np.random.default_rng(args.seed), len(todo))
    rows: list[dict] = []
    for plan, num in zip(plans, todo):
        audio_id = f"SS-{CLASS_CODE}-{num:04d}"
        plan["audio_id"] = audio_id
        # per-clip RNG seeded by id -> re-runs regenerate the same condition set
        rng = np.random.default_rng(seed_for(audio_id, args.seed))
        dur, sr, ch = build_clip(plan, rng, None)
        rel = f"synthetic/{SLUG}/{audio_id}.wav"
        digest = hashlib.sha256((AUDIO_DATASET / rel).read_bytes()).hexdigest()
        rows.append({
            "audio_id": audio_id,
            "filename": rel,
            "class_label": CLASS_LABEL,
            "source": SOURCE_TAG,
            "source_url": "",
            "licence": LICENCE,
            "author": AUTHOR_TAG,
            "date_fetched": DATE_FETCHED,
            "duration_sec": round(dur, 3),
            "sampling_rate": sr,
            "channels": ch,
            "recording_environment": ENVIRONMENT_MAP[plan["env"]],
            "recording_device": plan["device"],
            "approximate_distance": plan["dist_band"],
            "original_or_augmented": "original",
            "parent_audio_id": "",
            "segment_start_sec": "",
            "segment_end_sec": "",
            "sha256": digest,
            "dataset_split": "",  # build_split.py is the only writer
            # extras
            "seed": seed_for(audio_id, args.seed),
            "audio_provenance": "synthetic",
            "sim_environment": plan["env"],
            "sim_intensity": plan["intensity"],
            "voice_name": plan["voice_name"],
            "speech_rate_wpm": plan["rate"],
            "pitch_hz": plan["pitch"],
            "post_chain": plan["chain"],
            "phrase": plan["text"],
            "notes": f"tts=pyttsx3;phrase_variant={plan['text']!r};chain={plan['chain']};"
                     f"synthetic=true",
        })
        if len(rows) % 25 == 0:
            print(f"  rendered {len(rows)}/{len(todo)}")

    # write rows (merge with any existing rows CSV)
    existing_rows: list[dict] = []
    if ROWS_CSV.exists():
        with ROWS_CSV.open(newline="", encoding="utf-8") as fh:
            existing_rows = [r for r in csv.DictReader(fh)]
    have_ids = {r["audio_id"] for r in existing_rows}
    rows = [r for r in rows if r["audio_id"] not in have_ids]
    all_rows = existing_rows + rows
    all_rows.sort(key=lambda r: r["audio_id"])
    fields = FROZEN_COLUMNS + EXTRA_COLUMNS
    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    print(f"wrote {len(rows)} rows ({len(all_rows)} total) -> {ROWS_CSV}")
    texts = Counter(r["phrase"] for r in all_rows)
    print(f"distinct phrases: {len(texts)}; voices used: "
          f"{len({r['voice_name'] for r in all_rows})}; "
          f"post-chains: {len({r['post_chain'] for r in all_rows})}")
    short = TARGET - len(all_rows)
    if short > 0:
        print(f"SHORTFALL: {short} clips below target {TARGET}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())