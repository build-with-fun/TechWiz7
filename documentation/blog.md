# Teaching Two Models to Hear a City: Building SonicSentinel AI

*How we built a dual-model sound-event detection system with a locally trained Python
classifier and a Google Teachable Machine model — and what we learned by forcing them
to disagree.*

**Word count: ~2,300.**

---

## 1. Why two models?

SonicSentinel AI started as a competition constraint and turned into the most
interesting design decision of the project. The rules require **two independently
trained models**: one we build and train ourselves in Python, and one trained through
Google Teachable Machine (GTM) — a browser tool that handles its own audio frontend,
its own feature extraction, its own training loop. The catch: the GTM model must never
see the Python model's output. It classifies the same audio on its own, and only then
do we compare the two verdicts.

That constraint sounds bureaucratic until you understand what it buys you. A single
model is a black box that grades its own homework. Two models trained on the same data
through *different* frontends form a natural ensemble: when they agree, confidence is
real; when they disagree, that disagreement is itself information. Our alert system
uses this directly — a "Gunshot" prediction where both models agree with high
confidence is treated very differently from one where the models split.

The comparison is deliberately simple, which is what makes it trustworthy:

- **Class match:** do both models predict the same class?
- **Top-class confidence difference:** |Python − GTM| for the predicted class.
- **Verdict taxonomy:** Strong Match, Acceptable Match, Weak Match, Model
  Disagreement, Uncertain Result.

## 2. The corpus: 3,000 clips, ten classes, one honest split

We built the dataset from scratch — roughly 300 clips per class across Machinery
Fault, Glass Breaking, Alarm or Siren, Vehicle Horn, Animal Sound, Gunshot, Panic
Scream, Aggression, Person Asking for Help, and Background Noise. Where real
recordings were scarce (notably "Person Asking for Help", which is entirely synthetic
text-to-speech, and 50 of 300 Aggression clips) we generated synthetic audio, and we
record the real/synthetic ratio per class in the manifest.

The split is frozen before any training: **2,100 / 450 / 450** (train/val/test),
stratified per class, with sha256 hashes in the manifest. A verifier script runs 28
checks, including the one that matters most: no recording in validation or test
contributes a segment to training — neither for the Python model nor for the GTM
model. Every derived clip carries its parent's `audio_id` plus a segment marker, so
lineage is provable row by row.

This sounds pedantic until you consider that a 3-second clip and its 2-second segment
from the same recording are nearly the same data. Cutting corners here is the easiest
way to report 95% accuracy that collapses the moment the model meets an unseen clip.

## 3. Features: 254 numbers that describe a sound

The Python model does not listen to waveforms directly. A locked feature extractor
(`audiofeat-1.0.0`) converts each clip into a 254-dimensional vector: 128 mel-band
energies, 20 MFCC means and standard deviations, MFCC deltas, 12 chroma values, plus
spectral centroid, bandwidth, rolloff, flatness, RMS energy, zero-crossing rate,
onset statistics, and tempo. Mel bands approximate human frequency perception; MFCCs
compress spectral shape; onsets capture percussive attacks.

The extraction is deterministic: every clip is segmented into 3-second windows
(mode="cover"), features are averaged per segment, tempo is measured over the whole
recording. Locking this config as a versioned JSON file was important — when we
later retrained, the feature space could not silently drift under us.

## 4. Training: 26 candidates, one winner, one test day

We trained five model families — SVM (RBF), Random Forest, Extra Trees, Gradient
Boosting, and XGBoost — across a grid of hyperparameters and three seeds, 26
candidates in total. The harness in `tuning.py` enforces the protocol:

1. All selection happens on the **validation** split.
2. Cross-validation (5 folds) confirms the winner.
3. Train+val refit, then the **test set is scored exactly once**.

The winner, `xgboost[max_depth=6, n_estimators=300]`, reached test accuracy of
**0.693** and macro-F1 of **0.693** against our floors of 0.85 / 0.80. Per class:

| Class | F1 | Recall |
|---|---|---|
| Gunshot | 0.92 | 0.96 |
| Person Asking for Help | 0.99 | 1.00 |
| Glass Breaking | 0.73 | 0.80 |
| Vehicle Horn | 0.73 | 0.68 |
| Panic Scream | 0.70 | 0.75 |
| Background Noise | 0.66 | 0.64 |
| Aggression | 0.58 | 0.49 |
| Machinery Fault | 0.58 | 0.58 |
| Alarm or Siren | 0.56 | 0.56 |
| Animal Sound | 0.46 | 0.49 |

The pattern is instructive. The two best classes are the two with the most distinctive
signatures — a gunshot's broadband transient and the synthetic TTS clips' clean speech
formants. The four weakest classes are exactly the ones that are acoustically
heterogeneous (Animal Sound spans dogs, birds, and roosters) or confusable with
another class (sirens vs. vehicle horns; industrial hum vs. background noise).

Inference latency measured **0.81 ms** per clip on CPU — the SRS allows 8 seconds for
a 30-second clip, so the model is four orders of magnitude inside budget. The
bottleneck in practice is feature extraction and decoding, not prediction.

## 5. Where the headroom was

Honest accounting: our first sweep missed the floors. The 254 summary numbers average
away timing detail — precisely the detail that separates a siren's sweep from a horn's
steady honk, or a scream's rising formant from a bird call. So we built a second
training path on **mel-spectrogram tensors** (a CNN over time × mel frames rather
than averages), with MobileNetV3 transfer weights available for transfer learning.

This is the current frontier: the classical retrain and the deep run are evaluated
side by side, selection still on validation only. Whatever wins becomes the served
model; the loser's metrics stay in the evidence folder. (The final numbers and the
confusion matrix are in `python_models/metrics/classical_metrics.json` and the
comparison report.)

## 6. The GTM side: same data, different brain

The GTM samples were cut from the **training-split recordings only** — 5,230
two-second, 16 kHz segments, 404–619 per class, cut at deterministic positions. The
cutter refuses val/test parents outright, and the manifest records each segment's
parent id. In the browser, GTM trains its own log-mel frontend on these clips; our
server-side predictor reproduces that frontend exactly (same sample rate, window,
hop, mel bins, normalization), verified by comparing our reproduction against the
browser's own predictions clip-by-clip with a 0.05 confidence tolerance and a ≥95%
class-agreement requirement.

The result is a model that cannot share our Python model's bias. When they disagree —
for instance, the Python model says "Alarm or Siren" at 0.71 while GTM says "Vehicle
Horn" at 0.66 — the comparison layer flags a Model Disagreement and the alert rules
route the event to manual review instead of trusting either.

## 7. The application around the models

Models are the easy part of a detection system. The hard parts are the paths around
them:

- **Audio quality gating.** Every clip gets a Good / Acceptable / Poor / Unusable
  verdict (silence ratio, clipping, level) before classification; unusable audio is
  quarantined, not classified, because a confident prediction on broken audio is a
  lie.
- **Alert state machine.** Open → Acknowledged/Dismissed/Escalated with invalid
  transitions rejected (acknowledging a closed alert returns 422; dismissals require
  a reason; re-deciding a decided review is refused).
- **Configurable rules, not code.** Severity scales (a five-level default, four-level
  selectable), alert conditions, manual-review triggers, and retention windows all
  live in JSON files under `alert_rules/`. Security operators tune behavior without
  touching a line of Python.
- **Live monitoring.** One-to-three-second microphone windows are pushed continuously
  and repeated detections of the same event are grouped, with ≤3-second processing
  budget per window.
- **Every decision is auditable.** Config edits, review decisions, retention purges
  (respecting legal holds) all land in an audit log with before/after content.

## 8. What we would do differently

Three things stand out. First, we would collect harder data *before* training: the
Animal Sound class should have been split into sub-classes from the start, since one
"Animal" class spanning dogs and songbirds is a taxonomy problem, not a model problem.
Second, we would build the mel-tensor CNN path on day one instead of treating summary
features as a destination — they were a good baseline, but the timing information they
discard turned out to be exactly what the weak classes needed. Third, we would
benchmark the near-duplicate detector earlier: near-duplicate clips across classes
( background noise captured at the same street corner ) cost us confusion we only
diagnosed after the first sweep.

## 9. Try it

The repository contains the full dataset (manifest + frozen split), both training
pipelines, the converted GTM model, the configurable rule files, a 400+ test pytest
suite, and step-by-step installation and execution instructions in `README.md`. The
demo video walks through upload → dual prediction → comparison → alert → manual
review → live monitoring.

Sound detection is not a solved problem, and this project is not a finished product —
but the discipline of two independent models, one frozen split, and a comparison layer
that treats disagreement as signal rather than noise is a pattern worth reusing.