# Feature rationale: what each acoustic feature can and cannot detect

Author: `amina` (auditory neuroscience / psychoacoustics)
Date: 2026-09-23
Status: review of SRS Step 6 / FR xx feature list. Machine-readable parameter decisions are in
`recommended_perceptual_params.json`; per-pair analysis is in `class_similarity_and_confusability.md`.

This document answers one question per feature: **does the feature carry the information the class
needs, or is it a plausible-looking number that the classifier will learn a shortcut from?** A
feature is justified here only if a normal-hearing listener demonstrably uses the same cue. Where a
feature is in the SRS list but is weak for this task, that is said plainly rather than silently
implemented.

The governing principle, taken from the auditory system itself: the cochlea performs a
non-uniform frequency analysis (narrow bandwidths at low frequencies, wide at high), it is phase-
insensitive above roughly 4-5 kHz, it integrates energy over a few hundred milliseconds for
loudness, and it strips fine temporal structure into a small number of envelope and modulation
channels before the signal reaches cortex. Everything below is a consequence of that.

---

## 1. Why the Mel scale and MFCCs are not arbitrary

The Mel scale is a perceptual, not physical, axis: listeners judge equal *musical* intervals as
equal only when the frequency axis is warped so that equal distances represent equal critical-band
spacing (Stevens, Volkmann & Newman 1937, *JASA* 8(3):185-190). It approximates the cochlea's
frequency-place map, which is roughly logarithmic at low frequency and increasingly linear above
1 kHz. The Bark scale is the psychoacoustic version of the same idea (Zwicker 1961; Traunmüller
1990, *JASA* 88(1):97-100, gives the closest empirical fit).

Consequences that matter to this project:

- **Mel-spaced bins are the right resolution for the spectral features.** Events whose identity is
  in low-frequency structure (machinery harmonics, speech formants) need fine resolution below
  1 kHz; broadband transients (gunshot, glass) are differentiated mostly by their high-frequency
  distribution, which the Mel scale deliberately coarsens. That is a fair trade for a 10-class
  classifier but it is why **broadband transients must be carried by a separate, unsmoothed feature
  family** (crest factor, high-frequency energy ratio, spectral flux) rather than by MFCCs alone.
- **MFCCs are a decorrelating transform on a smoothed log-Mel spectrum.** The DCT compresses each
  frame to a handful of coefficients that behave roughly like a compact spectral-envelope
  description. They are excellent for harmonic and speech-like sources and notoriously insensitive
  to exactly the fine broadband detail that separates gunshot from a door slam - hence the
  "glass vs metal impact" failure mode in `class_confusability.json` (P06) when only MFCC
  statistics are used.
- **Coefficient count is a real decision, not a default.** Too few cepstral coefficients and the
  envelope is over-smoothed (harmonics lost); too many and the high coefficients encode
  frame-specific detail that does not generalise and is the first thing a model overfits to.
  Recommendation: n_mfcc in the 20-40 range *including the deltas*, and always include delta and
  delta-delta, because the deltas are what encode motion through the spectrum (the modulation
  information the class pairs depend on).

**Verdict: keep MFCCs and Mel spectrogram, but never as the sole feature family for the four
impulsive classes.**

---

## 2. Onsets, transients and the classes that live there

Two of the ten classes are defined almost entirely by their impulse: **Gunshot** and **Glass
Breaking**, and a third (Machinery Fault) is defined by *repeated* impulses.

An impulsive event's identity is concentrated in the first tens of milliseconds:

- rise time (a gunshot's muzzle blast rises in about 0.5-2 ms; a glass crack in 5-20 ms; a door
  slam in 20-50 ms),
- peak-to-RMS ratio (crest factor: gunshot and glass sit above 15-20 dB; a siren or a hum sits at
  3-8 dB),
- and what happens *after* the onset, which is where the same-onset pairs separate:
  glass shatters into many short non-harmonic modes; metal rings on a few long, high-Q,
  harmonically related modes; a gunshot's tail is environment reverberation of a low-frequency
  blast.

This is why **onset strength alone cannot separate the impulsive classes**, and why the recommended
feature set adds a decay-phase family (decay-envelope slope, onset density 150-800 ms after the
first onset, high-frequency energy ratio, spectral roll-off measured on the decay only).

- **Spectral flux** (frame-to-frame spectral change) is the right companion for onset strength: it
  marks where the spectrum *changes*, which is how a listener hears a new event begin (Bregman
  1990, *Auditory Scene Analysis*, MIT Press).
- **Zero-crossing rate** is cheap and genuinely useful, but only in its proper role: it rises with
  noisiness/high-frequency content and is therefore a good rough discriminator between tonal
  (alarm, horn, hum) and noisy (glass shower, crowd) content. It is *not* a pitch or content
  feature and should not be relied on for the speech classes.

---

## 3. Temporal integration: why 1-3 s windows are defensible, and why 2 s is my recommendation

The SRS (Step 12, FR xv, NFR 1) says live windows such as 1-3 s and a prediction within 3 s.

The perceptual literature does not give 1-3 s as a *loudness* integration window - loudness
integrates over roughly 100-200 ms and saturates well before a second (Zwicker & Fastl,
*Psychoacoustics*, 3rd ed., 2013). What 1-3 s does correspond to is the scale of an **auditory
event**: listeners segment a continuous acoustic stream into discrete events on a timescale of
roughly 1.5-2 s (Sridharan, Levitin, Chafe, Berger & Menon 2007, *Neuron* 55(3):521-532), and it
also comfortably contains the full duration of the fastest critical classes rather than only their
onsets (gunshot 0.2-0.8 s; glass shatter 0.3-1.5 s; scream 0.5-2 s).

**Recommendation: 2.0 s window with a 1.0 s hop (50% overlap), 3.0 s as the hard ceiling.** The
full argument, including the latency arithmetic and the hop-size trap, is in
`recommended_perceptual_params.json` under `live_window`. The short version:

- **2.0 s not 3.0 s**, because the NFR's "within three seconds" is a latency promise. A 3.0 s
  window consumes the entire budget before inference starts; 2.0 s + hop leaves a real inference
  budget while still satisfying the NFR under any reading.
- **1.0 s hop, not hop == duration**, because with a non-overlapping 2 s/2 s scheme a 0.3-0.6 s
  event can fall inside exactly one window, making SRS Step 15's consecutive-detection requirement
  structurally unsatisfiable - a Gunshot alert would never fire. This is the single most important
  number in the live pipeline and it is not obvious from the spec.

---

## 4. Amplitude: normalisation versus distance (the trap in Step 4)

The SRS requires amplitude normalisation. The SRS also requires the dataset to vary recording
distance. These two pull in opposite directions, and the naive framing ("normalise, therefore
distance cues are lost") is only half true.

- Absolute sound pressure level is **not recoverable** from an uncalibrated file: capture gain,
  microphone sensitivity and ADC gain are unknown and mixed into one number. So some normalisation
  is mandatory for cross-device consistency. Correct.
- But **peak normalisation destroys the crest factor**, and crest factor is one of the strongest
  non-content cues in the whole system (impulsive vs sustained). RMS normalisation destroys it
  even more thoroughly.

**The correct resolution - measure, then normalise:**

1. Compute and persist level features *before* any normalisation: `rms_dbfs`, `peak_dbfs`,
   `crest_factor_db`, `noise_floor_dbfs`, and a short-term loudness spread.
2. Normalise the waveform that goes into the model (fixed target RMS with a peak limiter) so the
   classifier is device-independent.
3. Pass the *pre-normalisation* level features to the classifier as their own inputs.

Distance itself is better approached through the cues a listener actually uses, all of which survive
an uncalibrated capture: **direct-to-reverberant ratio**, **high-frequency attenuation above ~4 kHz**
(air absorption increases with range), and **estimate of the reverberation decay**. A listener can
tell a distant event from a close one with exactly these cues and no SPL reference; so can a model
(Blauert, *Spatial Hearing*, MIT Press; Zahorik 2002, *JASA* 111(4):1832-1846). Whether this
project extracts them is optional; claiming distance estimation without them would not be.

---

## 5. Modulation and pitch: the features that decide the alarm/horn and scream/aggression pairs

Two of the eight SRS Step-14 pairs are only separable in the *slow* modulation domain, at a
timescale of one second - which is to say, at a timescale no single analysis frame can see.

- **Alarm or Siren vs Vehicle Horn** (P03): both are sustained tonal harmonic stacks in the same
  few-hundred-Hz region, so their frame spectra overlap heavily. A siren sweeps or pulses at
  roughly 0.5-4 Hz; a horn holds two steady tones. A frame-level classifier is blind to this by
  construction; a modulation spectrum over the whole 2 s window sees it immediately. This is the
  highest-value single feature addition available to the project.
- **Panic Scream vs Aggression/shouting** (P04): both are loud human vocalisation by the same
  speaker. The difference is phonation mode. When subglottal pressure is pushed past the point of
  stable oscillation the voice breaks into subharmonics and biphonation, producing fast amplitude
  modulation in the tens-of-Hz roughness range. Arnal, Flinker, Kleinschmidt, Giraud & Poeppel
  (2015, *Current Biology* 25(15):2051-2056) showed screams carry a distinctive modulation
  component in this range that makes them stand out against other loud sounds, including
  perceptually. That is a measurable, non-obvious, decisive feature: compute the amplitude
  modulation spectrum around 30-150 Hz, plus jitter/shimmer and harmonic-to-noise ratio.

**Verdict: add a modulation-spectrum feature family. Without it, P03 and P04 are guesswork.**

`tempo`, listed in Step 6, is only meaningful for the rhythmic classes - the pulse rate of a
siren's T3 pattern or the repetition rate of a machinery defect. It is meaningless for a gunshot.
Use it as an onset-envelope autocorrelation rather than as a music tempo estimate, and only feed it
where the envelope shows periodicity.

---

## 6. Feature-by-feature verdict against the classes

| Feature (SRS Step 6 / FR xx) | Perceptual basis | Strong for | Weak / misleading for |
|---|---|---|---|
| Mel spectrogram | Cochlear frequency-place warping (Stevens 1937; Traunmüller 1990) | all classes as a common substrate; speech, tonal classes | fine broadband transient detail (coarse above 1 kHz) |
| MFCC (+ Δ, ΔΔ) | Decorrelated log-Mel envelope; Δ encodes spectral motion | speech classes, machinery harmonics, animal calls | impulsive classes when used without decay features |
| Chroma | Pitch-class folding; robust to octave | Alarm, Vehicle Horn, tonal animal calls | gunshot, glass, noise; anything inharmonic |
| Zero-crossing rate | Noisiness / high-frequency content proxy | tonal vs noisy gross separation; glass | pitch, content; noise sensitivity at low level |
| RMS energy | Envelope / loudness proxy | segmentation, background-noise estimation, onset support | leaks capture gain; **must never be the only level feature** |
| Spectral centroid | Brightness - the perceptual "sharpness" axis | glass, scream, animal, horn | noisy sources where it is dominated by the masker |
| Spectral bandwidth | Spectral spread; distinguishes tonal from broadband | siren vs hum; machinery vs noise | alone, ambiguous between many pairs |
| Spectral roll-off | Fraction of energy below a frequency; transient brightness | glass, gunshot, screams | slowly-varying sounds |
| Onset strength | Event-boundary detection (the basis of auditory stream segmentation) | all impulsive and repeated-event classes | **cannot separate same-onset pairs; needs a decay companion** |
| Tempo / onset-envelope periodicity | Rhythmic rate of a repeated source | Machinery Fault, pulsed alarms | everything aperiodic (most of the ten classes) |
| **Crest factor (dB)** *(added)* | Peak-to-RMS; the impulsive/sustained axis listeners hear as "sharp" vs "smooth" | Gunshot, Glass, vs all sustained classes | sensitive to a single outlier sample - use with the clipped-run check |
| **Decay-envelope slope & onset density after onset** *(added)* | The ring-down carries the source identity (shards vs modes vs reverberation) | Glass vs Metal/Machinery (P06), Glass vs Gunshot (P11) | heavily reverberant recordings, where the room dominates the decay |
| **Modulation spectrum 0.5-4 Hz** *(added)* | Sweep/pulse rate is the siren's designed signature | Alarm vs Horn (P03) | stationary sources |
| **Modulation spectrum 30-150 Hz, jitter, HNR** *(added)* | Voice-quality roughness, the scream's privileged cue | Panic Scream vs Aggression (P04) | clean speech classes unaffected |
| **Audible-margin (per-band SNR)** *(added)* | Energetic masking - is the event perceptually present at all? | gates every critical alert (P12) | requires a noise-floor estimate; over-estimates audibility in non-stationary noise |
| **DRR, HF attenuation, RT60** *(added, optional)* | Distance cues a listener can actually use | relative near/far context, robustness | not absolute distance; needs careful estimation |

---

## 7. What I will not sign off on

1. **Reporting macro-F1 or accuracy as the headline for the critical classes.** The critical
   classes are the ones that carry the NFR recall floor and the ones whose false positives destroy
   operator trust. Report per-class precision *and* recall at a named threshold.
2. **A confidence threshold whose only justification is that it makes the numbers look good.**
   Thresholds must be chosen on validation against the false-alarm budget in
   `recommended_perceptual_params.json`, and the operating point must be stated.
3. **Reading ">=85% accuracy" as permission to tune on the test split.** An augmented or
   re-tuned test set is not the unseen test set the SRS compares the two models on.
4. **Any critical-class confusion called "fixed" without the named pair (P01-P12) being checked in
   the confusion matrix.** Perceptual analysis says which pairs those are; the matrix says whether
   they were actually addressed.
