#!/usr/bin/env python3
"""Perceptual audit of the procedural corpus (`data/generate_corpus.py`).

WHY THIS EXISTS
---------------
The dataset is procedurally synthesised, which is honest and reproducible, but
it means every acoustic cue in the training data was written by *our own code*.
If the synthesis does not carry the cue a class is defined by, the model learns
the artefact instead of the phenomenon, and the whole comparison report is
measuring a DSP bug.

This script does not read the generator's intent; it measures the audio that
comes out of it and asks the questions an auditory physiologist would ask:

  1. TIMING      -- does generating 3,000 clips fit in the build budget?
                    (the `reverb` function has a per-sample Python loop)
  2. DISTANCE    -- is `approx_distance_m` recoverable from the waveform, or
                    is it metadata with no acoustic correlate?  And is the
                    separately-drawn intensity a PHYSICALLY CONTRADICTORY
                    combination (a 60 m source at clipping level)?
  3. TRANSIENCE  -- do the four transient classes actually have the crest
                    factor that separates them from the six sustained ones?
  4. P03         -- is the siren-vs-horn modulation-spectrum discriminator
                    (0.5-4 Hz) actually present in the synthesised signals?
  5. P01/P10     -- are the two pairs my map calls irreducible/priority
                    genuinely overlapping in short-time spectrum?

Run:  .venv/bin/python documentation/perception/audit_corpus_realism.py
Exit code 0 always; it is an audit, not a gate.  Findings go in the report.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("gen", ROOT / "data" / "generate_corpus.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

SR = 16000
SEED = 20260923


def profile_for(**over) -> dict:
    """A nominal profile, overridden per test."""
    p = {
        "environment": "outdoor_street",
        "distance_band": "mid_3-15m",
        "distance_m": 8.0,
        "device": "smartphone_builtin",
        "intensity": "normal",
        "interference": "clean",
        "overlap": False,
        "overlap_class": "",
        "duration_s": 3.0,
        "overlap_gain_db": -12.0,
    }
    p.update(over)
    return p


def make(label: str, seed: int, **over) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prof = profile_for(**over)
    n = int(prof["duration_s"] * SR)
    y = gen.GENERATORS[label](rng, SR, n, prof)
    return gen.simulate_channel(y, SR, rng, prof)


def crest_db(y: np.ndarray) -> float:
    """Peak-to-RMS in dB -- the impulsive/sustained axis."""
    rms = float(np.sqrt(np.mean(y ** 2)))
    return 20 * np.log10(float(np.max(np.abs(y))) / max(rms, 1e-12))


def hf_ratio(y: np.ndarray, split_hz: float = 4000.0) -> float:
    """Fraction of energy above `split_hz` -- the distance/air-absorption cue."""
    spec = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    total = float(spec.sum()) + 1e-20
    return float(spec[f >= split_hz].sum() / total)


def centroid_hz(y: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    return float((spec * f).sum() / (spec.sum() + 1e-20))


def mod_spectrum_peak(y: np.ndarray, lo: float = 0.5, hi: float = 4.0) -> float:
    """Peak of the amplitude-envelope modulation spectrum in [lo,hi] Hz.

    This is the siren-vs-horn discriminator from class_confusability.json P03:
    a siren sweeps/warbles at 0.5-4 Hz, a horn is a steady tone.
    """
    env = np.abs(y)
    # smooth to ~50 Hz envelope bandwidth before taking the modulation spectrum
    win = max(int(SR / 50), 1)
    env = np.convolve(env, np.ones(win) / win, mode="same")
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env))
    f = np.fft.rfftfreq(len(env), 1.0 / SR)
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return 0.0
    return float(f[band][np.argmax(spec[band])])


def short_time_spectrum_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of mean log-Mel-ish spectra (0..1).

    A crude stand-in for 'how alike are these two sounds to an auditory
    front end', used to sanity-check the irreducible pairs.
    """
    n = min(len(a), len(b), SR * 3)
    A = np.abs(np.fft.rfft(a[:n] * np.hanning(n)))
    B = np.abs(np.fft.rfft(b[:n] * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    edges = np.array([50, 150, 300, 500, 800, 1200, 1800, 2600, 3600, 5000, 7000, 8000])
    va, vb = [], []
    for i in range(len(edges) - 1):
        m = (f >= edges[i]) & (f < edges[i + 1])
        if m.any():
            va.append(np.log(A[m].mean() + 1e-9))
            vb.append(np.log(B[m].mean() + 1e-9))
    va, vb = np.array(va), np.array(vb)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-20))


def banner(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def assert_mode() -> int:
    """Exit non-zero when a measured perceptual property fails.

    These are properties the corpus must have for the classes to be learnable
    for the right reason. They are not stylistic: each one, if false, means a
    feature the team is building rests on nothing.
    """
    failures: list[str] = []
    n = SR * 3

    print("Perceptual property assertions on the generated corpus\n")

    # (a) every class produces finite, in-range, correctly-sized audio
    for label in gen.CLASSES:
        y = make(label, 11)
        ok = np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0 + 1e-9 and len(y) == n
        print(f"  {'PASS' if ok else 'FAIL'}  finite/in-range/length: {label}")
        if not ok:
            failures.append(f"{label}: non-finite, out of range, or wrong length")

    # (b) HF energy must fall with distance -- otherwise the corpus has no
    #     distance cue at all and every 'distant source' claim in the report
    #     is unsupportable.
    hf = {}
    for band, d in (("near_0.5-2m", 1.0), ("mid_3-15m", 8.0), ("far_20-60m", 40.0)):
        vals = [hf_ratio(make("Glass Breaking", 3000 + i, environment="outdoor_open",
                              distance_band=band, distance_m=d, intensity="normal"))
                for i in range(6)]
        hf[band] = float(np.mean(vals))
    ok = hf["near_0.5-2m"] > hf["mid_3-15m"] > hf["far_20-60m"]
    print(f"  {'PASS' if ok else 'FAIL'}  HF energy falls with distance: "
          f"{hf['near_0.5-2m']:.3f} > {hf['mid_3-15m']:.3f} > {hf['far_20-60m']:.3f}")
    if not ok:
        failures.append("HF energy does not fall monotonically with distance")

    # (c) absolute peak must NOT encode distance: intensity is drawn
    #     independently and clips are re-normalised. This documents the
    #     deliberately level-free corpus so no feature may use absolute level.
    peaks = []
    for band, d in (("near_0.5-2m", 1.0), ("far_20-60m", 40.0)):
        vals = [20 * np.log10(np.max(np.abs(make("Glass Breaking", 4000 + i,
                environment="outdoor_open", distance_band=band, distance_m=d,
                intensity="normal"))) + 1e-12) for i in range(4)]
        peaks.append(float(np.mean(vals)))
    ok = abs(peaks[0] - peaks[1]) < 1.0
    print(f"  {'PASS' if ok else 'FAIL'}  absolute peak does NOT encode distance "
          f"(spread {abs(peaks[0] - peaks[1]):.2f} dB)")
    if not ok:
        failures.append("absolute peak varies with distance: corpus is not level-free")

    # (d) the transient/sustained axis must exist and be large for the two
    #     transient critical classes.
    crest = {}
    for label in ("Gunshot", "Glass Breaking", "Alarm or Siren", "Vehicle Horn",
                  "Person Asking for Help"):
        crest[label] = float(np.mean([crest_db(make(label, 5000 + i)) for i in range(8)]))
    ok = crest["Gunshot"] - crest["Alarm or Siren"] > 6.0
    print(f"  {'PASS' if ok else 'FAIL'}  Gunshot transient vs sustained siren: "
          f"{crest['Gunshot'] - crest['Alarm or Siren']:+.2f} dB (>6 required)")
    if not ok:
        failures.append("crest gap Gunshot vs Alarm or Siren too small")
    ok = crest["Gunshot"] == max(crest.values())
    print(f"  {'PASS' if ok else 'FAIL'}  Gunshot has the highest crest factor of the "
          f"classes measured")
    if not ok:
        failures.append("Gunshot is not the most impulsive class")

    # (e) Person Asking for Help is a SUSTAINED critical class -- a threshold
    #     or feature design that assumes 'critical implies impulsive' breaks it.
    ok = crest["Person Asking for Help"] < crest["Gunshot"]
    print(f"  {'PASS' if ok else 'FAIL'}  Person Asking for Help is less impulsive than "
          f"Gunshot ({crest['Person Asking for Help']:.2f} < {crest['Gunshot']:.2f} dB)")
    if not ok:
        failures.append("Person Asking for Help is unexpectedly impulsive")

    # (f) P03: the corpus must actually carry a spectral cue separating siren
    #     from horn, since the amplitude-modulation cue the map first named is
    #     confounded by the honk gate.
    sh = [hf_ratio(make("Alarm or Siren", 6000 + i)) for i in range(8)]
    hh = [hf_ratio(make("Vehicle Horn", 7000 + i)) for i in range(8)]
    ratio = float(np.mean(sh)) / max(float(np.mean(hh)), 1e-9)
    ok = ratio > 3.0
    print(f"  {'PASS' if ok else 'FAIL'}  siren/horn >4kHz energy ratio = {ratio:.1f}x "
          f"(>3 required) -- the real P03 discriminator is present")
    if not ok:
        failures.append("no spectral cue separates Alarm or Siren from Vehicle Horn")

    print()
    if failures:
        print(f"CORPUS PERCEPTUAL AUDIT: FAIL ({len(failures)} propert{'y' if len(failures)==1 else 'ies'})")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("CORPUS PERCEPTUAL AUDIT: PASS (all measured properties hold)")
    return 0


def generator_revision() -> str:
    """The corpus generator is being edited while we measure it.

    Every number in this audit is only true of ONE revision of
    data/generate_corpus.py, so the revision is printed with the report and
    must be quoted alongside any measured claim in the deliverable.
    """
    import hashlib
    p = ROOT / "data" / "generate_corpus.py"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def main() -> int:
    print("Perceptual audit of the procedural corpus")
    print(f"generator: {ROOT / 'data' / 'generate_corpus.py'}")
    print(f"generator revision (sha256/12): {generator_revision()}  (at {time.strftime('%H:%M:%S')})")

    # ---------------------------------------------------------------- 1. timing
    banner("1. TIMING -- does 3,000 clips fit the build budget?")
    t0 = time.time()
    N = 12
    for i in range(N):
        make("Gunshot", 1000 + i, environment="indoor_hall", intensity="normal")
    dt = (time.time() - t0) / N
    print(f"  mean wall time per clip (indoor_hall, 3.0 s, reverb path): {dt:.3f} s")
    t0 = time.time()
    for i in range(N):
        make("Gunshot", 2000 + i, environment="outdoor_open", intensity="normal")
    dt_dry = (time.time() - t0) / N
    print(f"  mean wall time per clip (outdoor_open, reverb path skipped): {dt_dry:.3f} s")
    est_h = dt * 3000 / 3600.0
    print(f"  => extrapolated for 3,000 clips at the reverb-path cost: {est_h:.1f} h")
    print(f"  => extrapolated at the dry cost:                          {dt_dry * 3000 / 3600.0:.1f} h")

    # -------------------------------------------------------------- 2. distance
    banner("2. DISTANCE -- is approx_distance_m recoverable from the audio?")
    print("  profile: outdoor_open, smartphone_builtin, intensity='normal'")
    print(f"  {'band':<14}{'d(m)':>7}{'peak dBFS':>11}{'>4kHz %':>10}{'centroid Hz':>13}")
    rows = {}
    for band, d in (("near_0.5-2m", 1.0), ("mid_3-15m", 8.0), ("far_20-60m", 40.0)):
        ys = [make("Glass Breaking", 3000 + i, environment="outdoor_open",
                   distance_band=band, distance_m=d, intensity="normal") for i in range(6)]
        peak = np.mean([20 * np.log10(np.max(np.abs(y)) + 1e-12) for y in ys])
        hf = np.mean([hf_ratio(y) for y in ys]) * 100
        cen = np.mean([centroid_hz(y) for y in ys])
        rows[band] = hf
        print(f"  {band:<14}{d:>7.1f}{peak:>11.1f}{hf:>10.2f}{cen:>13.0f}")
    if rows["near_0.5-2m"] > rows["far_20-60m"]:
        print("  => HF energy DOES fall with distance (a real cue survives).")
    else:
        print("  => !! HF energy does NOT fall with distance -- no cue survives.")

    print("\n  intensity is drawn INDEPENDENTLY of distance. Contradictory combos:")
    print(f"  {'band':<14}{'intensity':<10}{'peak dBFS':>11}")
    for band, d in (("far_20-60m", 40.0), ("near_0.5-2m", 1.0)):
        for inten in ("quiet", "loud", "clipped"):
            ys = [make("Glass Breaking", 4000 + i, environment="outdoor_open",
                       distance_band=band, distance_m=d, intensity=inten) for i in range(4)]
            peak = np.mean([20 * np.log10(np.max(np.abs(y)) + 1e-12) for y in ys])
            print(f"  {band:<14}{inten:<10}{peak:>11.1f}")

    # ------------------------------------------------------------ 3. transience
    banner("3. TRANSIENCE -- crest factor separates transient from sustained?")
    print("  profile: outdoor_street, mid_3-15m, intensity='normal'")
    print(f"  {'class':<26}{'crest dB (mean)':>17}{'sd':>7}")
    measured = {}
    for label in gen.CLASSES:
        vals = [crest_db(make(label, 5000 + i)) for i in range(8)]
        measured[label] = float(np.mean(vals))
        print(f"  {label:<26}{np.mean(vals):>17.2f}{np.std(vals):>7.2f}")
    print("\n  SRS-critical transients vs the sustained bed:")
    for a, b in (("Gunshot", "Alarm or Siren"), ("Glass Breaking", "Vehicle Horn"),
                 ("Gunshot", "Background Noise")):
        gap = measured[a] - measured[b]
        print(f"    crest({a}) - crest({b}) = {gap:+.2f} dB"
              + ("   <- separable" if gap > 3 else "   <- !! TOO CLOSE"))

    # ------------------------------------------------------------------ 4. P03
    banner("4. P03 Siren vs Horn -- is the 0.5-4 Hz modulation cue present?")
    print(f"  {'class':<18}{'mod peak Hz':>13}{'crest dB':>10}{'>4kHz %':>9}")
    for label in ("Alarm or Siren", "Vehicle Horn"):
        peaks, crests, hfs = [], [], []
        for i in range(10):
            y = make(label, 6000 + i)
            peaks.append(mod_spectrum_peak(y))
            crests.append(crest_db(y))
            hfs.append(hf_ratio(y) * 100)
        print(f"  {label:<18}{np.mean(peaks):>13.2f}{np.mean(crests):>10.2f}{np.mean(hfs):>9.2f}")
    stab = [mod_spectrum_peak(make("Vehicle Horn", 7000 + i)) for i in range(10)]
    print(f"  horn mod-peak spread: {min(stab):.2f}-{max(stab):.2f} Hz "
          f"(a steady horn should NOT show a low-rate peak)")

    # ------------------------------------------------------------ 5. confusables
    banner("5. SHORT-TIME SPECTRAL OVERLAP of the pairs the map calls hard")
    pairs = [
        ("P01 Gunshot vs Vehicle Horn (map: irreducible-ish)", "Gunshot", "Vehicle Horn"),
        ("P02 Glass vs Machinery Fault", "Glass Breaking", "Machinery Fault"),
        ("P04 Panic Scream vs Aggression", "Panic Scream", "Aggression"),
        ("P10 Panic Scream vs Person Asking for Help", "Panic Scream", "Person Asking for Help"),
    ]
    print(f"  {'pair':<48}{'cos sim':>9}")
    for name, a, b in pairs:
        sims = []
        for i in range(5):
            sims.append(short_time_spectrum_corr(make(a, 8000 + i), make(b, 9000 + i)))
        print(f"  {name:<48}{np.mean(sims):>9.3f}")

    banner("audit complete")
    print(f"measured against generator revision {generator_revision()}")
    return 0


if __name__ == "__main__":
    if "--assert" in sys.argv:
        raise SystemExit(assert_mode())
    raise SystemExit(main())
