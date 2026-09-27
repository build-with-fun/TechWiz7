# SonicSentinel AI: project report

Aptech TechWiz 7 · NextWave AI and ML · SRS v1.0. Team: Ammar Ahmer, Aimon and Khizr;
task allotment in §9.

Every number in this report comes from a file in the repository, named next to it. Where a
target is not met, we say so.

## 1. Problem, background and necessity

Factories, transport hubs and public buildings are full of sounds that matter: a machine
starting to fail, breaking glass, a gunshot, a scream, a call for help. Right now someone
has to be listening at the right moment, or go through recordings afterwards, which is slow
and easy to get wrong in a noisy place. SonicSentinel AI analyses uploaded recordings and
live microphone audio (with consent), sorts them into ten sound classes, and helps an
operator decide what needs attention.

## 2. Proposed solution, purpose and scope

A Flask web app. Every clip, or every two-second live window, is validated, rated for
quality, preprocessed, and scored separately by two models: a Python classifier we trained
and a Google Teachable Machine (TM) audio model. The app compares the two, applies
configurable alert rules (with confirmation over repeated detections), raises alerts for
critical classes, sends uncertain results to a review queue, and keeps everything (audio,
both score lists, model versions, decisions, audit trail) for search, analytics and
reports. There are five roles: normal user, audio reviewer, security operator,
maintenance operator and administrator.

**In scope:** everything in SRS §1.6. **Out of scope:** acting as a certified emergency or
law-enforcement system, automatic dispatch, and any generative-AI decision.

**Assumptions:** an operator is supervising; recordings are made with consent; the app runs
as one server process (the repeated-detection state lives in that process); FFmpeg is
installed.

**Constraints (SRS §1.5):** audio quality, noise, distance, device and overlapping events
vary; accuracy depends on the dataset; the two models may disagree; privacy and consent;
bias across speakers and devices. Our own constraints: a laptop CPU (4 cores, 15 GB RAM, no
usable GPU for training), freely licensed audio only, five competition days.

## 3. Requirements

Each functional (FR i–lxxx) and non-functional (NFR 1–5) requirement is listed with the
file that implements it, the test or measurement, and its status in
[documentation/SRS_TRACEABILITY.md](documentation/SRS_TRACEABILITY.md).

## 4. Architecture and modules

![Data flow](diagrams/dfd.png)

| Module | Responsibility |
|---|---|
| `src/app.py`, `src/auth.py`, `src/api/` | Flask app, sessions, CSRF, role checks, HTTP routes |
| `audio_preprocessing/` | decoding (FFmpeg), validation, quality verdict, high-pass, noise gate, trimming, normalisation, 16 kHz mono, segmentation |
| `feature_extraction/` | 254 hand-made features (baseline), CNN14 embeddings, and AST embeddings (served model) |
| `python_models/` | training scripts, the served bundle `best/`, metrics |
| `gtm_model/`, `src/inference/gtm_predictor.py` | the TM export and its server-side frontend |
| `src/inference/consistency.py` | comparison of the two models |
| `src/services/pipeline.py` | the decision path for one clip or window |
| `alert_rules/`, `config/` | thresholds, per-class rules, retention, monitoring limits |
| `src/services/persistence.py`, `src/models.py` | database writes and schema |
| `src/services/monitoring.py` | FR lxxviii anomaly checks |
| `templates/`, `static/` | pages, live monitor, event player and visuals |

More detail: [ARCHITECTURE.md](ARCHITECTURE.md), [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

### Diagrams

| Diagram | File |
|---|---|
| Data flow (level 1) | `diagrams/dfd.png` |
| Use case | `diagrams/use_case.png` |
| Activity | `diagrams/activity.png` |
| Sequence (upload) | `diagrams/sequence_upload.png` |
| Decision flow | `diagrams/decision_flow.png` |

All five are drawn by `tools/render_uml.py`, which also checks that no boxes overlap and no
arrow crosses a box.

## 5. Database design and data dictionary

SQLite through SQLAlchemy, with ten tables: users, audio_files, model_versions, events,
confidence_scores, alerts, reviews, live_sessions, live_windows and audit_records. Every
event links to its audio file and to the Python and TM model versions that produced it.
Reviews keep the original model outputs next to the reviewer's decision. Schema, keys and
indexes are in [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md), and the dataset's data dictionary
in [audio_dataset/DATA_DICTIONARY.md](audio_dataset/DATA_DICTIONARY.md).

## 6. Audio pipeline and decision rules

![Decision flow](diagrams/decision_flow.png)

**Validation** (FR viii): format by decoding, not extension; size ≤ 50 MB; 0.5 s–5 min;
integrity; presence of sound. **Quality** (FR xii–xiv, xxxvii): silence, clipping, low
signal and an SNR estimate give Good, Acceptable, Poor or Unusable; Unusable is refused.
**Preprocessing** (FR xi): high-pass 50 Hz, spectral noise gate (strength 0.75), trim ends
at 30 dB below peak, normalise to −3 dBFS, resample to 16 kHz mono, and cut 3 s segments with
timestamps. Training uses the output of this same code (`python_models/preprocess_cache.py`).

**Comparison** (FR xxxi–xxxix): whether the classes match, |Python top − TM top|, each
model's top-two margin, and any second class above 0.25 (a possible overlap). The result is
Strong Match, Acceptable Match, Weak Match, Model Disagreement or Uncertain Result.

**Rules** (FR xl–liii): each class has a severity, recommended action and escalation in
`alert_rules/alert_rules.json`. A critical alert needs an alertable class, both models
agreeing, quality of at least Acceptable and, by default, three windows in a row within 8 s.
It is raised once per confirmation. **Review** (FR lvii) is triggered by disagreement,
confidence below 0.6, a top-two margin below 0.15, poor quality, overlap, a possible
near-duplicate, or a critical class without agreement.

## 7. Dataset

3,000 original recordings, 300 per class: ESC-50 (696), FSD50K (about 700), UrbanSound8K
(387), our synthetic help phrases (300) and generated Aggression/Panic clips (75). Every row
in `audio_dataset/manifest.csv` has an audio ID, filename, class, source, licence, author,
duration, sample rate, channels, environment, device, distance where known,
original/augmented status, parent ID and SHA-256. Licences are CC-BY-4.0, CC0 and CC-BY-3.0,
plus 36 Sampling+ clips that need checking before redistribution
([DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md)).

**Split:** 2,100 / 450 / 450, exactly 210 / 45 / 45 per class, assigned by *source
recording*, so slices of one Freesound upload (or one synthetic voice saying one phrase)
stay together. Our first split didn't do this and leaked: 125 uploads and 37 voice+phrase
pairs ended up in more than one partition. We replaced it on 26 Sep and re-scored every
model.

**How it was collected, and its limits:** public, licensed collections mapped to our
classes, plus text-to-speech for the five SRS help phrases in 11 voices. Weak spots:
Aggression includes door slams and thumps, Machinery Fault is just normal machinery, Help is
synthetic only, and 2,123 rows have no environment recorded.

**Quality:** every clip was run through the app's own preprocessing: 1,775 came out Good,
1,169 Acceptable, 56 Poor and none Unusable. Glass Breaking and Panic Scream have the most
Poor clips (15 each). Per class and per split: [reports/DATASET_QUALITY.md](reports/DATASET_QUALITY.md).

**Augmentation** (FR xix): `augmentation/` has noise (real background recordings or
coloured noise), time shift, pitch shift, time stretch, volume, reverb, and distance and
device simulation. Copies are only made from training recordings, keep their parent's ID and
split, and never count as originals. The robustness probes use the same functions.

## 8. Models

### 8.1 Python model

| Family | Test acc. | Macro F1 | Critical recall |
|---|---:|---:|---:|
| HistGradientBoosting, 254 features | 0.731 | 0.730 | 0.822 |
| CRNN on log-mel (old split) | 0.600 | 0.596 | 0.667 |
| CNN14 embeddings + MLP | 0.840 | 0.840 | 0.862 |
| CNN14 embeddings + logreg, +4,200 augmented train copies | 0.838 | 0.837 | 0.849 |
| **AST embeddings + logistic regression (served)** | **0.891** | **0.892** | **0.907** |

The served model uses the Audio Spectrogram Transformer (pretrained on AudioSet, model card
`MIT/ast-finetuned-audioset-10-10-0.4593`, pinned revision) with logistic regression on its
2,063-value embedding. Validation picked logreg with `C 0.01` (selection score 0.885). Every
family was trained on the training split only and scored once on test. The served model
scores 0.997 accuracy on the training split, 0.873 on validation and 0.891 on test
(`python_models/metrics/served_model_split_results.json`, from `tools/split_results.py`), so
it fits the training clips much more closely than new ones.

The design, validation grid, per-class precision/recall/F1, confusion matrix and error
analysis are in [documentation/MODEL_EVALUATION.md](documentation/MODEL_EVALUATION.md). The
features are described in `feature_extraction/features.py`. The served bundle records its
training-set hash, seed, selection criterion and pretrained checksum
(`python_models/best/model_meta.json`).

![Python confusion matrix](python_models/metrics/confusion_matrix_transfer_test.png)

### 8.2 Teachable Machine model

A TM audio project with the ten SRS class names, trained on 1,400 one-second samples: 140
training recordings per class, picked in a fixed hash order so every source is represented,
one sample each (the loudest second after preprocessing, the same rule the server uses).
1,400 was the most Teachable Machine would train in our browser without stalling. It was
exported as TensorFlow.js and converted to Keras for the server, which averages the scores
of every one-second window of a clip, weighted by energy (chosen on validation). Training
used TM's defaults except Epochs = 200 (picked on validation over 50 and 100, on 27 Sep).
Evidence: `screenshots/gtm/`, `gtm_model/metadata.json`,
`gtm_model/upload_package/tm_imports/index.json`, `audio_dataset/gtm_samples/`.

Project link: <https://teachablemachine.withgoogle.com/train/audio/17pC3F6eg_sY_HHF8fY8aI2M73_B87UQ_> (a Google sign-in is needed to open it in Teachable Machine; the
project file itself is at <https://drive.google.com/file/d/17pC3F6eg_sY_HHF8fY8aI2M73_B87UQ_/view>). Hosted model: <https://teachablemachine.withgoogle.com/models/56AmxJNhY/>. This linked project was trained
on 27 Sep with the same 1,400 samples and settings. Each TM run comes out a little
different, and this one scored slightly lower on validation (0.531 accuracy against 0.536),
so the app keeps serving the earlier export; the linked model is in
`gtm_model/candidates/tm_linked_e200/`.

Test result: **0.511 accuracy, 0.492 macro F1, 0.600 mean critical-class recall**
(`gtm_model/gtm_metrics.json`). Earlier attempts: the first export (32 samples per class,
first second of each clip) scored 0.278; 1,400 samples in audio-id order (mostly FSD50K)
scored 0.462; the same samples in hash order with the default 50 epochs scored 0.493. Runs
with 1,750 and 2,100 samples stalled inside TM on both GPUs. The model is still well below
the SRS targets; `documentation/MODEL_EVALUATION.md` explains why and what we tried.

### 8.3 Prediction and confidence comparison

`reports/MODEL_COMPARISON.md` and `reports/model_comparison.csv` cover all 450 test recordings
with every SRS column: both predictions, all ten confidences from each model, class match,
the top-class confidence difference, top-two margins, quality, severity, alert status, review
status, final decision, correctness and an explanation of each disagreement.

**Summary over the 450 test recordings** (Python AST + logreg, Teachable Machine 200-epoch
export, 27 Sep). The models picked the same class for 223 of 450 (49.6%): 36 Strong Matches
(8.0%), 26 Acceptable Matches (5.8%) and 46 Weak Matches (10.2%). They disagreed outright on
47 (10.4%), and 295 (65.6%) came out as Uncertain Result, mostly because the Teachable
Machine model is much less confident and accurate than the Python model. When both models
agree, they are right 97.3% of the time. In practice, 388 of 450 recordings (86.2%) went to
the manual-review queue with a reason, and 61 of the 62 decided automatically were correct
(98.4%), compared with 0.511 for the Teachable Machine model alone. So a disagreement isn't
averaged away; it is treated as a reason for a person to look.

## 9. Team and task allotment

Team name: _to be filled in_. Roll numbers: _to be filled in_. Module-by-module ownership
for the viva is in `documentation/VIVA_PACK.md` §1, and each member's own account of their
work is in `documentation/TEAM_CONTRIBUTION_RECORD.md`.

| Member | Roll no. | Work | Evidence |
|---|---|---|---|
| Ammar Ahmer | | Backend (Flask app, REST API, database, authentication and roles), model training and selection for the Python model and the Teachable Machine model, dataset pipeline, documentation, deployment | Git history (every commit is under his identity), `src/`, `database/`, `python_models/`, `gtm_model/`, `audio_dataset/`, `documentation/`, README deployment section |
| Aimon | | Frontend (pages, styling, browser scripts), testing, other support | `templates/`, `static/` |
| Khizr | | Suggestions, audio recordings for testing, testing, help with implementation | to be linked by Khizr |

All commits in the repository use one author identity (Ammar's). Aimon's and Khizr's work is
recorded here and in the contribution record, not in the Git history.

## 10. Testing

`pytest` covers functional, integration (real Flask app and database), boundary, negative,
security (CSRF, roles, lockout), database (including 20,000 events), audio format, silence,
clipping, noise, preprocessing, feature extraction, both models, comparison, alert rules,
duplicates and near-duplicates, low confidence and overlap. The latest run and the full test
plan are in [TEST_PLAN.md](TEST_PLAN.md). Other evidence:

| Evidence | File |
|---|---|
| Both models on 450 unseen recordings | `reports/model_comparison.csv` |
| Noise, echo, low volume, device, distance, partial, overlap, re-encoding, confusables | `reports/ROBUSTNESS.md` |
| Latency against NFR 1 | `reports/performance.json` |
| Near-duplicate detection | `reports/near_duplicates.json` |
| Confidence threshold on validation | `reports/threshold_calibration.json` |
| UI at four widths | `reports/ui_review.json` |

### Noise robustness, false positives and false negatives

**How the models degrade.** These probes (`reports/ROBUSTNESS.md`, 26 Sep) were run with
the previous Python model, CNN14 + MLP, and haven't been re-run for the AST model yet.
Python accuracy dropped from 0.74 on the clean probe clips to
0.62 with 0 dB background noise, 0.55 at −30 dB volume, 0.53 at a simulated 20 m distance
and 0.55 with a second sound overlapping at −6 dB. Critical-class recall held up better in
the quiet conditions (0.78 at −30 dB, 0.76 at 20 m), since a faint critical sound is still
more likely than the alternatives. The Teachable Machine model started much lower (0.37
clean) and dropped by about the same amounts. Noise, distance and overlap are the main
limits, so the app is best used indoors, in fairly quiet places, with the microphone close.

**Similar and unknown sounds.** We played 240 clips of sounds the models never trained on
(ESC-50 fireworks, door knocks, clapping, laughing, crying babies, church bells). 230 of
them (95.8%) went to the manual-review queue, and only 10 ended up as a confident critical
alert with no human check. Crying babies were heard as Panic Scream (35 of 40) and church
bells as Alarm or Siren (32 of 40), which is reasonable for sounds that close. Door knocks
and clapping came out as Aggression, which could cause false alarms. This is why the
comparison and the review queue matter: they stop an unfamiliar sound from turning into a
critical alert by itself.

False negatives on the clean test split: 4 of 45 Panic Screams were called Aggression and 3
Animal Sound, and some quiet animal sounds were called Background Noise. False positives for
critical classes mostly come from the same voice confusion (4 Aggression clips called Panic
Scream, 4 Machinery Fault clips called Aggression). 10 of 450 predictions were wrong at
≥ 0.9 confidence.

### Performance

Measured over HTTP with both real models (`tools/benchmark_latency.py`); full timings are
in `reports/performance.json`. A 30 s upload (decoding, both models, the alert rules and
the database write) takes a median **7.15 s** (p95 7.47 s) against the SRS budget of 8 s,
and a 2 s live window a median **2.06 s** (p95 2.30 s) against 3 s. Both targets are met.
With four clients uploading at once the median goes up to 25.7 s, which is about the limit
of one CPU-only process. The very first request after start-up includes loading the models
(11.6 s) and is reported separately; the app now warms both models at start-up
(`AnalysisPipeline.warm()`), so users don't pay that cost.

Scale (NFR 2, `tools/benchmark_scale.py`, `reports/scale.json`): 20,000 synthetic events,
gunicorn with one worker and 8 threads, real HTTP requests from signed-in users. The p95
response time over all page and API reads was 0.065 s for one user, 0.37 s for 10 and
0.60 s for 20, with no errors and no endpoint above 0.76 s. The largest export allowed
(9,334 rows) took 1.6 s. On the same server a 30 s upload took 4.8 s (median), and four at
once took 17.5 s, which is what this laptop's CPU can do. We use one worker on purpose,
because the live page's "N windows in a row" counter is kept in the worker's memory.

## 11. Security and privacy

Passwords are hashed and an account locks after 5 failed logins. Every route checks roles
on the server, forms use CSRF tokens, and responses carry security headers with a strict
Content-Security-Policy. Uploads are validated by actually decoding them. Stored audio is
only served through routes that check permissions, and paths can't leave the storage
folder. Exact duplicates are caught by SHA-256, and CSV exports are escaped against
spreadsheet formula injection. The audit log covers logins, uploads, sessions,
predictions, alerts, reviews, overrides, exports and config changes, and there are anomaly
checks for failed uploads, model failures, low-confidence spikes, bursts of alerts,
duplicates and failed logins.

For privacy, the microphone only starts after the user ticks the consent box, the page
shows when it is recording, retention is configurable (with a preview before deleting),
and no audio is sent anywhere else. No secrets are committed; settings go in `.env`
(template: `.env.example`). The demo passwords are published for the evaluators and must
be changed before any public deployment.

## 12. Limitations and future work

- SRS-4: the served Python model reaches **0.891** test accuracy (target 0.85, met) and
  0.892 macro F1 (target 0.80, met). For the 85% recall per critical class, Help, Gunshot
  and Glass pass but Aggression (0.84) and Panic Scream (0.82) don't, although the five
  average 0.907. Retraining with augmented copies on Day 5 didn't beat the served model on
  validation. The TM model misses all three targets (0.511 / 0.492 / 0.600), see §8.2.
- Aggression and Machinery Fault use stand-in labels, and the help phrases are synthetic only.
- There is no "Unknown" class, so a sound outside the ten gets the closest class and relies
  on review.
- We haven't measured whether the TM spectrogram on the server matches the browser's.
- The repeated-detection counter lives in one process, so the app runs as a single
  gunicorn worker with threads; four 30 s uploads at once take 17.5 s. The app is served
  from the team's laptop through ngrok, and uptime over the evaluation period still has to
  be recorded with `tools/uptime_probe.py`. A Hugging Face Space kit
  (`deploy/huggingface/`) is ready if we get an account.
- Trimmed re-uploads are only recognised as near-duplicates about 22% of the time.

Next steps: record our own clips for the weak classes; add an "Unknown" class trained on
out-of-set sounds; fine-tune on a GPU; move the repeated-detection counter into the
database so several workers can run; and record uptime through the evaluation period.
