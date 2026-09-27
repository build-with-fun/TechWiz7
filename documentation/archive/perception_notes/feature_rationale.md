# Feature rationale: what each acoustic feature can and can't detect

> Archived design notes from 23 Sep, written while the corpus was still procedurally
> generated. The reasoning about which classes get confused still holds; measured numbers
> refer to that earlier corpus.

Date: 2026-09-23. A review of the SRS Step 6 / FR xx feature list. Parameter choices are in
`recommended_perceptual_params.json`; the class pairs are discussed in
`class_similarity_and_confusability.md`.

For each feature the question is whether it carries information the class needs, or just
gives the classifier something to overfit. We prefer features that match cues people
actually use. The ear analyses frequency non-uniformly (fine at low frequencies, coarse at
high ones), ignores phase above about 4-5 kHz, integrates loudness over a few hundred
milliseconds, and reduces fine timing to a few envelope and modulation channels.

---

## 1. Mel scale and MFCCs

The Mel scale warps frequency so equal distances sound like equal pitch steps (Stevens,
Volkmann & Newman 1937, *JASA* 8(3):185-190). It roughly follows the cochlea: close to
logarithmic at low frequencies, closer to linear above 1 kHz. The Bark scale is the
psychoacoustic version (Zwicker 1961; Traunmüller 1990, *JASA* 88(1):97-100).

- **Mel bins suit this task.** Machinery harmonics and speech formants need detail below
  1 kHz. Broadband transients (gunshot, glass) differ mostly at high frequencies, which Mel
  smooths out, so they need separate unsmoothed features (crest factor, high-frequency
  energy ratio, spectral flux), not just MFCCs.
- **MFCCs** compress each frame's log-mel spectrum into a few coefficients describing the
  spectral envelope. They work well for harmonic and speech-like sounds but miss the fine
  broadband detail that separates a gunshot from a door slam (the glass vs metal problem,
  P06 in `class_confusability.json`).
- **The number of coefficients matters.** Too few smooth away harmonics; too many encode
  frame-specific detail that overfits. Use 20-40 including deltas, and always include delta
  and delta-delta, since they capture how the spectrum moves.

**Keep MFCCs and the mel spectrogram, but don't rely on them alone for the impulsive classes.**

---

## 2. Onsets and transients

Gunshot and Glass Breaking are defined mostly by their impulse, and Machinery Fault by
repeated impulses. What identifies an impulse is in the first tens of milliseconds:

- rise time (gunshot about 0.5-2 ms, glass crack 5-20 ms, door slam 20-50 ms);
- crest factor (gunshot and glass above 15-20 dB, a siren or hum 3-8 dB);
- what happens after the onset: glass breaks into many short, non-harmonic modes; metal rings
  on a few long harmonic modes; a gunshot's tail is room reverb of a low-frequency blast.

So onset strength alone can't separate the impulsive classes. We suggest adding decay
features: decay slope, onset density 150-800 ms after the first onset, high-frequency energy
ratio, and roll-off measured on the decay only.

- **Spectral flux** (change from frame to frame) goes well with onset strength; it marks
  where a new sound starts (Bregman 1990, *Auditory Scene Analysis*, MIT Press).
- **Zero-crossing rate** is cheap and useful for telling tonal sounds (alarm, horn, hum)
  from noisy ones (glass, crowd). It isn't a pitch or content feature.

---

## 3. Window length: why 2 s

The SRS mentions live windows of 1-3 s and a result within 3 s.

Loudness integrates over about 100-200 ms (Zwicker & Fastl, *Psychoacoustics*, 3rd ed.,
2013), so 1-3 s isn't about loudness. It matches how people split continuous sound into
events, roughly every 1.5-2 s (Sridharan, Levitin, Chafe, Berger & Menon 2007, *Neuron*
55(3):521-532), and it fits the whole of the short critical sounds (gunshot 0.2-0.8 s, glass
0.3-1.5 s, scream 0.5-2 s).

**Suggestion: 2.0 s windows with a 1.0 s hop, never more than 3.0 s.** Details are under
`live_window` in `recommended_perceptual_params.json`.

- 2.0 s rather than 3.0 s, because "within three seconds" is a latency target. A 3 s window
  uses the whole budget before inference even starts.
- A 1.0 s hop rather than no overlap, because with 2 s windows and a 2 s hop a short event
  can land in just one window, so the consecutive-detection rule (Step 15) could never be met
  and Gunshot would never alert.

---

## 4. Normalisation vs distance

The SRS wants amplitude normalisation and also recordings at different distances, which
pull in opposite directions.

- Absolute sound level can't be recovered from an uncalibrated file (gain and microphone
  sensitivity are unknown), so some normalisation is needed.
- But peak normalisation changes the crest factor, one of the most useful cues (impulsive vs
  sustained), and RMS normalisation changes it even more.

**So measure first, then normalise:**

1. Before normalising, compute and keep `rms_dbfs`, `peak_dbfs`, `crest_factor_db`,
   `noise_floor_dbfs` and short-term loudness spread.
2. Normalise the waveform the model sees (target RMS with a peak limiter).
3. Give the classifier the pre-normalisation level features as extra inputs.

For distance, use the cues a listener uses, which work without calibration:
direct-to-reverberant ratio, high-frequency loss above about 4 kHz, and reverb decay
(Blauert, *Spatial Hearing*, MIT Press; Zahorik 2002, *JASA* 111(4):1832-1846). These are
optional, but don't claim distance estimation without them.

---

## 5. Modulation and pitch

Two of the SRS Step 14 pairs can only be separated by slow modulation over about a second,
which one frame can't see.

- **Alarm or Siren vs Vehicle Horn** (P03): both are steady tonal harmonics in the same
  range. A siren sweeps or pulses at about 0.5-4 Hz; a horn holds two steady tones. A
  modulation spectrum over the 2 s window shows this straight away. This is the most useful
  single feature to add.
- **Panic Scream vs Aggression** (P04): both are loud voices. When a voice is pushed hard it
  breaks into subharmonics and gets fast amplitude modulation in the 30-150 Hz "roughness"
  range. Arnal, Flinker, Kleinschmidt, Giraud & Poeppel (2015, *Current Biology*
  25(15):2051-2056) showed screams stand out because of this. Measure the modulation spectrum
  around 30-150 Hz, plus jitter, shimmer and harmonic-to-noise ratio.

**Add a modulation-spectrum feature family; without it P03 and P04 are guesswork.**

`tempo` from Step 6 only makes sense for rhythmic sounds (a siren's pulse pattern, a
repeating machine fault). Compute it as onset-envelope autocorrelation, not a music tempo,
and only where the envelope is periodic.

---

## 6. Feature summary

| Feature (SRS Step 6 / FR xx) | Why it works | Good for | Weak for |
|---|---|---|---|
| Mel spectrogram | Follows the cochlea's frequency map (Stevens 1937; Traunmüller 1990) | all classes; speech and tonal sounds | fine detail in broadband transients |
| MFCC (+ Δ, ΔΔ) | Compact log-mel envelope; deltas capture movement | speech, machinery harmonics, animal calls | impulsive classes without decay features |
| Chroma | Pitch class, ignoring octave | Alarm, Vehicle Horn, tonal animal calls | gunshot, glass, noise, anything inharmonic |
| Zero-crossing rate | Rough measure of noisiness | tonal vs noisy; glass | pitch, content; unreliable at low levels |
| RMS energy | Envelope and loudness | segmentation, noise estimate, onsets | depends on recording gain; don't use as the only level feature |
| Spectral centroid | Brightness | glass, scream, animal, horn | noisy recordings |
| Spectral bandwidth | Tonal vs broadband | siren vs hum; machinery vs noise | ambiguous on its own |
| Spectral roll-off | Where the energy sits | glass, gunshot, screams | slowly changing sounds |
| Onset strength | Where events start | impulsive and repeated-event classes | same-onset pairs without decay features |
| Tempo / periodicity | Rhythm of repeated sounds | Machinery Fault, pulsed alarms | most non-rhythmic classes |
| **Crest factor (dB)** *(added)* | Peak vs RMS: sharp vs smooth | Gunshot, Glass vs sustained classes | one outlier sample; pair with the clipping check |
| **Decay slope and onset density after onset** *(added)* | The ring-down identifies the source | Glass vs metal (P06), Glass vs Gunshot (P11) | very reverberant rooms |
| **Modulation spectrum 0.5-4 Hz** *(added)* | A siren's sweep rate | Alarm vs Horn (P03) | steady sounds |
| **Modulation spectrum 30-150 Hz, jitter, HNR** *(added)* | Rough voice quality in screams | Panic Scream vs Aggression (P04) | normal speech |
| **Per-band SNR (audible margin)** *(added)* | Whether the event is audible at all | gating critical alerts (P12) | needs a noise-floor estimate; less reliable in changing noise |
| **DRR, HF loss, RT60** *(added, optional)* | Distance cues | near vs far, robustness | not absolute distance |

---

## 7. Things to avoid

1. Using macro-F1 or accuracy as the headline for the critical classes. Report per-class
   precision and recall at a stated threshold.
2. Picking a confidence threshold only because it makes the numbers look good. Choose it on
   validation against the false-alarm budget in `recommended_perceptual_params.json`, and
   state the operating point.
3. Tuning on the test split to reach 85%.
4. Calling a confusion "fixed" without checking that pair (P01-P12) in the confusion matrix.
