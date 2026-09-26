# Model evaluation

How the Python model was chosen, how it and the Teachable Machine (TM) model perform on
unseen recordings, and where they fail. Every number is copied from a file named next to
it; re-running the named script regenerates the file.

## Data and protocol

- 3,000 original recordings, 300 per class, split 2,100 / 450 / 450 by source recording
  (`data/splits/split.json`, algorithm `sha256-order-v2-source-groups`). Slices of one
  Freesound upload, and one synthetic voice saying one phrase, never cross partitions.
- **Train** fits the model. **Validation** chooses the family, hyper-parameters and
  checks the confidence threshold. **Test** is scored once per family after the choice
  is written to `python_models/metrics/transfer_selection_current.json`.
- Selection criterion, fixed in advance: `0.5 x macro-F1 + 0.5 x mean recall of the five
  critical classes` on validation; within 0.005, the faster model wins.
- Critical classes (SRS NFR 4): Gunshot, Glass Breaking, Panic Scream, Aggression,
  Person Asking for Help.

## Three model families

| Family | Input | Test acc. | Macro F1 | Critical recall | Evidence |
|---|---|---:|---:|---:|---|
| HistGradientBoosting | 254 hand-made features (MFCC, deltas, chroma, ZCR, RMS, centroid, bandwidth, roll-off, flatness, onset, tempo, 128 mel bands) | 0.731 | 0.730 | 0.822 | `classical_metrics_hgb_split_v2.json` |
| CRNN (PyTorch) | 128 x 94 log-mel | 0.600 | 0.596 | 0.667 | `deep_metrics_deep.json` (**v1 split**, not re-run) |
| CNN14 embedding + MLP | 2,048-d PANNs CNN14 embedding (AudioSet-pretrained) | 0.840 | 0.840 | 0.862 | `transfer_test_current.json` |
| CNN14 embedding + logreg, +4,200 augmented train copies | 2,048-d PANNs CNN14 embedding | 0.838 | 0.837 | 0.849 | `transfer_test_current_aug.json` |
| **AST embedding + logreg (served)** | 2,063-d Audio Spectrogram Transformer embedding (pooler + mean patch token + 527 AudioSet scores) | **0.891** | **0.892** | **0.907** | `transfer_test_ast_current.json` |

Notes. The HGB run refits on train + validation before its single test score, which is
how `train_classical.py` was written; the CNN14 models are fitted on train only. The CRNN
was trained from scratch on 2,100 clips and was the weakest by far; it was not re-run on
the v2 split because each run takes over an hour on this laptop's CPU.

### Why transfer learning won

The hand-made features plateaued at 0.70–0.76 on validation whatever the classifier
(SVM, random forest, extra trees, gradient boosting, XGBoost, HGB; see
`classical_comparison_*.csv`). The errors were between sounds with a similar spectral
envelope: door slams against gunshots, screams against shouting. CNN14 was trained on
two million AudioSet clips and its penultimate layer already separates those textures, so
a small classifier on top of it needs far less of our data. AudioSet is YouTube audio; our
corpus is Freesound-derived plus our own synthetic clips, so the test recordings are not in
CNN14's training data.

### Validation grid (CNN14 embeddings, 2,100 train / 450 validation)

| Classifier | Parameters | Val acc. | Val macro F1 | Val critical recall | Score | Fit (s) | ms / clip |
|---|---|---:|---:|---:|---:|---:|---:|
| **MLP** | 512 hidden, alpha 0.1 | 0.840 | 0.840 | 0.853 | **0.847** | 3.7 | 0.03 |
| MLP | 512 hidden, alpha 0.001 | 0.836 | 0.835 | 0.849 | 0.842 | 4.0 | 0.03 |
| Logistic regression | C 0.01 | 0.840 | 0.839 | 0.831 | 0.835 | 1.5 | 0.03 |
| Logistic regression | C 0.1 | 0.836 | 0.835 | 0.831 | 0.833 | 1.5 | 0.02 |
| Logistic regression | C 1.0 | 0.824 | 0.824 | 0.818 | 0.821 | 1.0 | 0.02 |
| SVM (RBF) | C 3 | 0.820 | 0.823 | 0.796 | 0.809 | 11.5 | 2.1 |
| SVM (RBF) | C 1 | 0.793 | 0.795 | 0.773 | 0.784 | 10.9 | 2.2 |

The classifier's own time is negligible; the CNN14 forward pass dominates (110–620 ms
per clip on this CPU, most under 300 ms).

### Served model: AST embedding + logistic regression

The served model is the Audio Spectrogram Transformer (AST) backbone with a logistic
regression head. Its embeddings are AudioSet-pretrained; the model card is
`MIT/ast-finetuned-audioset-10-10-0.4593` pinned to revision
`f826b80d28226b62986cc218e5cec390b1096902`. Selection chose logreg `C 0.01`
(selection score 0.885 on validation, `transfer_selection_ast_current.json`).

### Test results per class (AST + logreg, 450 recordings)

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Machinery Fault | 0.76 | 0.91 | 0.83 |
| Glass Breaking ★ | 0.93 | 0.91 | 0.92 |
| Alarm or Siren | 0.97 | 0.84 | 0.90 |
| Vehicle Horn | 0.95 | 0.87 | 0.91 |
| Animal Sound | 0.85 | 0.87 | 0.86 |
| Gunshot ★ | 0.96 | 0.96 | 0.96 |
| Panic Scream ★ | 0.88 | 0.82 | 0.85 |
| Aggression ★ | 0.83 | 0.84 | 0.84 |
| Person Asking for Help ★ | 1.00 | 1.00 | 1.00 |
| Background Noise | 0.83 | 0.89 | 0.86 |

★ critical class. Top-2 accuracy 0.964. Confusion matrix:
`python_models/metrics/confusion_matrix_transfer_test.png`.

Against the SRS targets: accuracy 0.891 ≥ 0.85 (**met**); macro F1 0.892 ≥ 0.80
(**met**); critical recall 0.907 ≥ 0.85 (**met**) — Gunshot 0.956, Glass 0.911,
Panic Scream 0.822, Aggression 0.844, Person Asking for Help 1.000.

The previous CNN14 + MLP model (accuracy 0.840, critical recall 0.862) met the
macro-F1 and critical-recall targets but fell one point short on accuracy, which is why
the AST backbone was adopted as the served model. Both were fitted on train only and
scored once on test per family, exactly as the protocol above requires.

### Test results per class (CNN14 + MLP, 450 recordings)

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Machinery Fault | 0.88 | 0.93 | 0.90 |
| Glass Breaking ★ | 0.80 | 0.82 | 0.81 |
| Alarm or Siren | 0.81 | 0.84 | 0.83 |
| Vehicle Horn | 0.97 | 0.87 | 0.92 |
| Animal Sound | 0.83 | 0.67 | 0.74 |
| Gunshot ★ | 0.91 | 0.93 | 0.92 |
| Panic Scream ★ | 0.81 | 0.76 | 0.78 |
| Aggression ★ | 0.67 | 0.84 | 0.75 |
| Person Asking for Help ★ | 0.98 | 0.96 | 0.97 |
| Background Noise | 0.80 | 0.78 | 0.79 |

★ critical class. Top-2 accuracy 0.927. Confusion matrix:
`python_models/metrics/confusion_matrix_transfer_test.png`.

Against the SRS targets: macro F1 0.840 ≥ 0.80 (**met**); accuracy 0.840 < 0.85 (missed by
one point); critical recall ≥ 0.85 for Gunshot and Help (**met**), not for Aggression
(0.84), Glass (0.82) or Panic Scream (0.76). This is the *previous* served model; the AST
model above replaced it precisely because of this miss.

## Where the Python model goes wrong

- **Panic Scream → Aggression: 9 of 45.** The Aggression class contains shouting, and the
  screams that fail are mostly short or distant. The reverse happens too.
- **Animal Sound → Aggression (5), → Background Noise (4).** Growling dogs appear in both
  Animal and Aggression training data; quiet birdsong sits close to ambient noise.
- **Vehicle Horn → Alarm (4), Glass ↔ Alarm (3 each way).** Tonal, sustained sounds.
- **21 of 450 predictions were wrong at confidence ≥ 0.9.** Confidence must not be read as
  proof.
- **Confidence is informative on average:** at ≥ 0.6 the test accuracy is 89 % (398
  recordings); below 0.6 it is 44 % (52 recordings).

### Confidence threshold, checked on validation

`reports/threshold_calibration.json` (`tools/calibrate_thresholds.py`):

| min_confidence | Decided without review | Accuracy of those | Critical clips wrongly auto-decided |
|---:|---:|---:|---:|
| 0.5 | 94 % | 86.1 % | 27 |
| **0.6 (configured)** | 90 % | 87.9 % | 23 |
| 0.7 | 86 % | 90.2 % | 19 |
| 0.8 | 82 % | 91.6 % | 15 |

In the app confidence is only one gate: model disagreement, a small top-two margin, poor
quality or overlap also send a clip to review, so the real auto-decided share is lower
(see `reports/MODEL_COMPARISON.md`). Raising the threshold to 0.7 is a reasonable choice
for a site that prefers more reviews over missed critical events; it is a team decision.

## Teachable Machine model

Trained in Google Teachable Machine's own audio project (default settings, run in a headless
Chrome by `tools/train_gtm_browser.py`) from training-split recordings only. Each sample is
the loudest one-second window of a recording after the app's preprocessing, the same rule
the server uses. Import lineage: `gtm_model/upload_package/tm_imports/index.json`; the
windows themselves: `audio_dataset/gtm_samples/` and `audio_dataset/manifests/gtm_segment_rows.csv`.

TM's model is the speech-commands network with its layers frozen and one new softmax layer
trained on top, on 0-5 kHz spectrograms of one second. In this browser it stops at
"Preparing training data" above roughly 1,400 samples (a 2,100-sample run was still there
after ten minutes), so 140 recordings per class is the practical maximum here.

| Export | Samples | Selection of recordings | Scoring | Val accuracy | Test accuracy | Test macro-F1 | Test critical recall |
|---|---|---|---|---|---|---|---|
| 25 Sep | 320 | first 32 per class, first second | loudest window | — | 0.278 | — | — |
| 26 Sep a (`candidates/tm140`) | 1,400 | first 140 per class by audio id | loudest window | 0.438 | 0.462 | 0.442 | 0.569 |
| 26 Sep b (`candidates/tm_v3_140`, served) | 1,400 | 140 per class in hash order | energy-weighted windows | 0.504 | **0.493** | **0.473** | **0.591** |

Taking recordings in audio-id order meant 764 of the 1,400 samples were FSD50K clips and only
66 came from UrbanSound8K; hash order spreads them across sources. On the same new export,
averaging the scores of every one-second window weighted by its energy beat scoring only the
loudest second (0.504 against 0.469 on validation), so the server now does that
(`window_aggregation` in `gtm_model/frontend_config.json`). Both choices were made on the
validation split; the test split was scored once, for the served configuration
(`gtm_model/gtm_metrics.json`).

Test recall per class: Person Asking for Help 0.98, Panic Scream 0.62, Background Noise,
Machinery Fault and Vehicle Horn 0.58, Glass Breaking 0.51, Gunshot 0.49, Aggression 0.36,
Alarm or Siren 0.13, Animal Sound 0.11.

The TM model is far below the SRS targets (85 % accuracy, 0.80 macro-F1, 85 % critical
recall). With a frozen speech-command network, a single trained layer and a ~1,400-sample
ceiling, we did not find a way to raise it further inside Teachable Machine. The app treats
its opinion accordingly: a disagreement or a low TM confidence sends the clip to manual
review rather than being averaged away.

## Leakage history

The first version of these numbers (26 Sep morning) used the v1 split. After the
source-group audit, every model was retrained and re-scored on v2. For the record, the
CNN14 + logistic regression model scored 0.824 on the v1 test split and 0.831 on v2; the
v1 artefacts are kept in `python_models/metrics/archive_split_v1/`.
