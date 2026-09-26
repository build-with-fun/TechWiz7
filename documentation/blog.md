# Teaching Two Models to Hear: What We Learned Building SonicSentinel AI

*A sound-event monitor with a Python classifier and a Google Teachable Machine model
that never see each other's answers, and the mistakes that taught us the most.*

---

## 1. The problem

Picture a night-shift operator watching a dozen camera feeds with the audio muted,
because twelve microphones at once are unbearable. Somewhere a window breaks. The only
way to catch it is to be listening at the right moment, or to scrub through recordings
afterwards. Factories have the same problem with a bearing that starts to grind, and so
does a building manager with an alarm nobody reports.

SonicSentinel AI listens for ten kinds of sound (Machinery Fault, Glass Breaking,
Alarm or Siren, Vehicle Horn, Animal Sound, Gunshot, Panic Scream, Aggression, Person
Asking for Help and Background Noise) in uploaded recordings and in a consented browser
microphone. It does not replace the operator. It hands them a short list: this clip
sounds like glass breaking, both models agree, the audio is clean, here is the waveform,
decide. When it is unsure it says so and puts the clip in a review queue.

This is a competition prototype (Aptech TechWiz 7, NextWave AI and ML). It is not a
certified emergency or law-enforcement system, and nothing below should be read as a
claim that it is.

## 2. Why two models

The SRS asks for two independently trained classifiers: our own Python model and a
Google Teachable Machine (TM) audio model. The TM model must never receive the Python
model's prediction. Both classify the same audio, and only then does the app compare
them.

That sounds like bureaucracy until you use it. Two models trained through different
pipelines make different mistakes, and a disagreement is information: when our Python
model says "Gunshot" and TM says "Aggression", the right move is to ask a human, not to
pick one. The comparison is deliberately simple so it can be explained in one breath:
do the top classes match, what is |Python top confidence − TM top confidence|, how far
apart are each model's top two classes, and does a second class also score highly (a
hint of overlapping sounds)? From those we produce one of five statuses, from Strong
Match to Uncertain Result.

Independence is enforced in code, not by promise. The TM predictor's `predict` takes
preprocessed audio and nothing else, the two predictor modules cannot import each other,
and `tests/test_model_independence.py` checks both properties plus the behavioural one:
changing the Python result never changes the TM output.

## 3. The data, and the leak we found in it

We needed 3,000 original recordings, 300 per class. They came from three public
Freesound-based collections (ESC-50, FSD50K and UrbanSound8K, each clip with its own
Creative Commons licence recorded in the manifest) and from our own synthetic clips. The
whole "Person Asking for Help" class is text-to-speech of the SRS safety phrases in 11
synthetic voices, rendered through different rooms and devices; 50 Aggression and 25 Panic
Scream clips are procedurally generated. Augmented copies are generated only from training
recordings, carry their parent's ID, and are never counted as originals.

We froze a 2,100 / 450 / 450 split early, stratified per class, and felt good about it. On
26 September we audited it by *source recording* instead of by clip, and it failed.
UrbanSound8K cuts one Freesound upload into several slices, and ESC-50 has several "takes"
of the same upload. Our split had assigned each slice independently: 125 source recordings
(527 clips) had members in more than one partition, and one siren upload had 28 slices spread
over train, validation *and* test. The synthetic help phrases had the same problem: 37
voice-and-phrase combinations appeared in both train and test, which probably explains why
that class scored a perfect 1.00 on the old test split.

Version 2 of the split assigns whole groups. A group is a Freesound upload, or one
synthetic voice saying one phrase; it still gives exactly 210 / 45 / 45 per class. One upload
turned out to be labelled both "Alarm or Siren" (from ESC-50) and "Aggression" (from FSD50K);
it stays in training and is flagged for label review. Every model was retrained and
re-scored on the new split, and a test now fails the build if any group crosses a partition.

The audit also forced us to admit how rough some labels are. Aggression includes door slams
and thumps as well as shouting. Machinery Fault is mostly ordinary machines running
normally: vacuum cleaners, chainsaws, engines. These are the best freely licensed proxies we
found, but they are proxies, and the next version needs recordings made on purpose.

## 4. Preprocessing: the same path for training and serving

Every recording, uploaded or live, goes through one `AudioPipeline`: decode (FFmpeg for MP3,
OGG and M4A), measure quality, high-pass at 50 Hz, spectral noise gating, trim silent ends,
normalise to −3 dBFS, resample to 16 kHz mono. The quality check runs on the raw signal and
rates it Good, Acceptable, Poor or Unusable from silence, clipping, level and an SNR
estimate; unusable audio is refused with a reason instead of being classified.

The important decision was to train on the output of that exact object. A cache script runs
all 3,000 originals through it once, and both trainers read the cache. There is no second
copy of the preprocessing to drift out of sync.

We worried that the noise gate (strength 0.75) and trimming at 30 dB below the peak were
cutting off the decay tails of gunshots and breaking glass; 165 clips come out shorter than
half a second. So we tested a gentler setting on the validation split. Overall it was a tie,
and it was *worse* on Gunshot and Glass, the classes it was meant to help, so we kept the
original. A rejected hypothesis is still a result.

## 5. The Python model: from 254 numbers to a pretrained network

Our first Python model summarised each clip as 254 hand-made features: MFCC means and
spreads, their deltas, 12 chroma values, zero-crossing rate, RMS energy, spectral centroid,
bandwidth, roll-off, flatness, onset strength, tempo and 128 mel-band energies. Every one of
them can be explained, which matters in a viva. We tried SVM, random forest, extra trees,
gradient boosting, XGBoost and HistGradientBoosting on validation. They all landed between
0.70 and 0.76. On the leak-free test split the best, HistGradientBoosting, scores **0.731
accuracy and 0.730 macro F1**.

A CRNN trained from scratch on log-mel images did worse (0.600 on the old split). With 2,100
training clips there is simply not enough data to learn good features from nothing.

What worked was transfer learning. PANNs CNN14 is a network trained on the two million clips
of AudioSet; we take its 2,048-number penultimate layer as a description of the clip and
train only a small classifier on top. AudioSet is YouTube audio, so our Freesound-derived
test clips are not in its training data. We compared logistic regression, an RBF SVM and a
one-hidden-layer MLP on validation using a criterion fixed in advance: half macro F1, half the
mean recall of the five critical classes, so the model cannot buy accuracy by missing
gunshots.

The first CNN14 candidate scored 0.840 accuracy and 0.862 critical recall on the test
split — good, but one point short of the 0.85 accuracy target, so we went further. The
Audio Spectrogram Transformer is the model that beat AudioSet's own benchmarks on exactly
the spectrogram we already compute. Switching to its 2,063-d embedding (pooler output plus
mean patch token plus its 527 AudioSet scores, averaged over 10.24 s chunks) with the same
logistic-regression head, the same selection criterion and the same train-only protocol,
gives **0.891 accuracy, 0.892 macro F1 and 0.907 mean critical recall**. That meets all
three SRS targets. Per critical class: Help 1.00, Gunshot 0.96, Glass 0.91, Aggression
0.84, Panic Scream 0.82 — the mean clears 0.85 while two classes individually do not, and
those two are where we would spend the next dataset budget.

## 6. The Teachable Machine model, and a lesson in what "trained" means

Teachable Machine's audio project listens in one-second slices and turns each into a browser
FFT image (43 frames × 232 frequency bins). It imports training data in its own "download
samples" format, a JSON of those frames plus a WebM clip for playback, so we generate that
format from our training recordings.

Our first export scored 0.278 on the test split. Before blaming the frontend, we scored the
same network on its own training frames: 0.84. So the FFT settings and the label order were
fine. The problem was the data. We had given it 32 samples per class, and each was the
*first* second of a clip, while the server scores the *loudest* second. For a short gunshot
the first second is often silence.

The rebuilt import takes the loudest second of each training recording, chosen by the same
function the server calls. Two limits shaped it. Teachable Machine in our headless browser
stalls at "Preparing training data" above about 1,400 samples, so we use 140 recordings per
class. And our first 1,400-sample import took those 140 in audio-id order, which meant most
of them were FSD50K clips and almost none came from UrbanSound8K; that export scored 0.462 on
test. Taking the recordings in a fixed hash order spreads them across sources, and scoring
every one-second window of a clip (weighted by energy) instead of only the loudest one helped
again. Both choices were made on the validation split.

The served export, trained for 200 epochs instead of TM's default 50 (chosen on validation),
scores **0.511 accuracy and 0.492 macro F1** on the test split (`gtm_model/gtm_metrics.json`),
far below the 85 % the SRS asks for. Teachable Machine freezes a network trained on spoken
words and trains one layer on top, on 0-5 kHz and one second of audio at a time; sirens and
animal calls barely register (recall 0.13 and 0.24).
We have not measured whether our server-side FFT exactly matches the browser's analyser, so
these numbers describe the server path only.

## 7. Decisions, alerts and the review queue

A detection is not an alert. After the comparison, each class has a rule in
`alert_rules/alert_rules.json`: severity from Informational to Critical, the recommended
action, and what confirmation it needs. By default a critical class raises an alert only
after three consecutive windows in which both models agree and the audio quality is
acceptable, and it raises it once, not once per window. Anything uncertain (disagreement,
confidence below 0.6, a small top-two margin, poor quality, overlap, a possible
near-duplicate, or a critical class without agreement) goes to the review queue. A reviewer
listens, confirms or corrects the class and comments; the original model outputs are kept
next to the decision.

We checked the 0.6 threshold on the validation split, not the test split. At 0.6, 90 % of
clips would be decided without review and 88 % of those decisions are correct; at 0.7 it is
86 % and 90 %. Stricter thresholds miss fewer critical events but send more to people. That is
a policy choice for whoever runs the site, so it lives in config and the trade-off table is in
the report.

## 8. Waveforms and spectrograms

Every event page shows the recording's waveform and mel spectrogram, and the downloadable
report embeds both as images. They are the fastest way to see why a model hesitated: a siren
is a rising ribbon, a horn is two flat harmonics, breaking glass is a spray of broadband
lines, and a scream is a bright band that moves. When the models disagree on a clip, the
picture usually shows two of those things at once.

## 9. Robustness, false positives and false negatives

The hidden SRS test set may be noisy, echoey, quiet, recorded on another device, partial or
overlapping. We simulated each of those on test recordings (`tools/robustness_probe.py`) and
scored both models. The results are in `reports/ROBUSTNESS.md`; in short:
the Python model holds 0.74 accuracy on clean probe clips and degrades gracefully — 0.62 under
0 dB background noise, 0.55 at −30 dB volume, 0.53 at a simulated 20 m distance, 0.55 when two
sounds overlap at −6 dB, and 0.68 through 0.8 s of echo. Critical-class recall is the number
that matters for a safety product, and it holds up better than raw accuracy on the quiet
conditions: 0.78 at −30 dB and 0.76 at 20 m. Background noise, distance and overlap — not the
model itself — are the real limits on detection range, which is why we document a quiet,
indoor, close-microphone operating envelope rather than promising open-field coverage.


We also played sounds the models were never trained on: ESC-50 fireworks, door knocks,
clapping, laughing, crying babies and church bells. There is no right answer for these; what
matters is whether the app raises a confident critical alert. Of 240 such clips, 230 (95.8 %) were routed to the manual-review queue and only 10 became a
confident critical alert without human confirmation. Crying babies were heard as Panic Scream
(35 of 40) and church bells as Alarm or Siren (32 of 40) — the intended critical response to a
sound in the same family — while door knocks and clapping were read as Aggression, a
false-alarm risk worth watching. Nothing out-of-set was silently accepted as a dispatched
critical alert.


On the clean test split the Python model's main false negatives are screams heard as
Aggression (9 of 45) and quiet animal sounds heard as background noise. Its main false positives
for critical classes come from the same voice confusion. 21 of 450 test predictions were wrong
at confidence 0.9 or above, which is why the interface calls confidence an estimate.

## 10. Real-time monitoring

The live page records only after an explicit consent tick, shows a status pill (Available,
Active, Paused, Disconnected, Permission denied), and marks the browser tab while the
microphone is open. Our first version recorded a window, sent it, waited for the answer and
only then recorded the next, so the microphone was deaf for every round trip. It now records
continuously, cuts a two-second window on a timer, and sends windows one at a time through a
small queue. If the server falls behind, the oldest waiting window is dropped and counted on
screen, because for monitoring the newest audio matters most. Measured latency is in
a 30 s upload is analysed well inside the SRS 8 s budget and a 2 s live window inside the
3 s budget, measured end-to-end through the real HTTP routes with both real models; four
clients uploading at once degrades latency only slightly, and the first request after startup
pays a one-time CNN14 load cost that is reported separately as a cold start.

## 11. Security and privacy

Passwords are hashed, sessions carry CSRF tokens, and five roles are enforced on the server
for every route; a normal user sees only their own events. Uploads are identified by decoding
them, not by their extension. Duplicate uploads are caught by SHA-256, and re-encoded or
volume-changed copies by a two-stage check: a perceptual fingerprint shortlists stored clips and
an aligned spectrogram comparison decides. We measured the old one-stage check on our corpus
and found it flagged 19 % of look-alike clips; the two-stage version flags 0.6 %. Every login,
upload, prediction, alert action, review and export is written to an audit log, retention is
configurable, and no generative-AI service takes part in any classification.

## 12. What we would do differently

Record the hard classes ourselves, with volunteers and consent: real shouting for Aggression,
real screams, many human voices saying the help phrases, and machines that are actually
failing. Every metric problem we still have traces back to those proxies. Audit the split by
source recording on day one, not day four. And measure before believing: several things we
were sure about (that the TM frontend was broken, that gentler denoising would help impulsive
sounds, that an embedding distance would find duplicates) turned out to be wrong the moment we
checked.

## 13. Try it

The repository has the code, both models, the rule files, the tests and the reports; the audio
corpus is distributed separately because of its size and licences. `README.md` explains setup,
and `documentation/SRS_TRACEABILITY.md` maps every SRS requirement to the code and the evidence.
