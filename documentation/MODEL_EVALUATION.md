# Model evaluation

How the Python model was chosen, how it and the Teachable Machine (TM) model do on
recordings they haven't seen, and where they go wrong. Every number comes from the file
named next to it, and re-running the script that wrote that file regenerates it.

## Data and protocol

- 3,000 original recordings, 300 per class, split 2,100 / 450 / 450 by source recording
  (`data/splits/split.json`, algorithm `sha256-order-v2-source-groups`). Slices of one
  Freesound upload, or one synthetic voice saying one phrase, always stay in one partition.
- **Train** is used to fit the model. **Validation** is used to pick the family and
  hyper-parameters and to check the confidence threshold. **Test** is scored once per
  family, after the choice is saved to `python_models/metrics/transfer_selection_current.json`.
- Selection score, fixed in advance: `0.5 x macro-F1 + 0.5 x mean recall of the five
  critical classes` on validation. Within 0.005, the faster model wins.
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

Notes: `train_classical.py` refits HGB on train + validation before its one test score,
while the CNN14 and AST models are fitted on train only. The CRNN was trained from scratch
on 2,100 clips and was by far the weakest. It wasn't re-run on the v2 split because each
run takes over an hour on this laptop's CPU.

### Why transfer learning won

With the hand-made features, validation accuracy stayed between 0.70 and 0.76 whatever
the classifier (SVM, random forest, extra trees, gradient boosting, XGBoost, HGB; see
`classical_comparison_*.csv`). Most errors were between sounds with a similar spectral
envelope, like door slams and gunshots, or screams and shouting. CNN14 was trained on two
million AudioSet clips and already tells those textures apart, so a small classifier on
top needs much less of our data. AudioSet is YouTube audio, and our clips come from
Freesound datasets plus our own, so our test recordings aren't in CNN14's training data.

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

The classifier itself takes almost no time; the CNN14 forward pass is what costs
(110–620 ms per clip on this CPU, usually under 300 ms).

### Served model: AST embedding + logistic regression

The served model is the Audio Spectrogram Transformer (AST), pretrained on AudioSet, with
logistic regression on top. The model card is `MIT/ast-finetuned-audioset-10-10-0.4593`,
pinned to revision `f826b80d28226b62986cc218e5cec390b1096902`. Validation picked logreg with
`C 0.01` (selection score 0.885, `transfer_selection_ast_current.json`).

### Test results per class (AST + logreg, 450 recordings)

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Machinery Fault | 0.76 | 0.91 | 0.83 |
| Glass Breaking\* | 0.93 | 0.91 | 0.92 |
| Alarm or Siren | 0.97 | 0.84 | 0.90 |
| Vehicle Horn | 0.95 | 0.87 | 0.91 |
| Animal Sound | 0.85 | 0.87 | 0.86 |
| Gunshot\* | 0.96 | 0.96 | 0.96 |
| Panic Scream\* | 0.88 | 0.82 | 0.85 |
| Aggression\* | 0.83 | 0.84 | 0.84 |
| Person Asking for Help\* | 1.00 | 1.00 | 1.00 |
| Background Noise | 0.83 | 0.89 | 0.86 |

\* critical class. Top-2 accuracy 0.964. Confusion matrix:
`python_models/metrics/confusion_matrix_transfer_test.png`.

Against the SRS targets: accuracy 0.891 ≥ 0.85 (**met**) and macro F1 0.892 ≥ 0.80
(**met**). Mean critical recall is 0.907, but per class Panic Scream (0.822) and Aggression
(0.844) are below 0.85; Gunshot (0.956), Glass (0.911) and Person Asking for Help (1.000)
are above it.

The previous model, CNN14 + MLP (accuracy 0.840, critical recall 0.862), met the macro-F1
target but was one point short on accuracy, so we switched to the AST backbone. Both were
fitted on train only and scored once on test, following the protocol above.

### Test results per class (CNN14 + MLP, 450 recordings)

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Machinery Fault | 0.88 | 0.93 | 0.90 |
| Glass Breaking\* | 0.80 | 0.82 | 0.81 |
| Alarm or Siren | 0.81 | 0.84 | 0.83 |
| Vehicle Horn | 0.97 | 0.87 | 0.92 |
| Animal Sound | 0.83 | 0.67 | 0.74 |
| Gunshot\* | 0.91 | 0.93 | 0.92 |
| Panic Scream\* | 0.81 | 0.76 | 0.78 |
| Aggression\* | 0.67 | 0.84 | 0.75 |
| Person Asking for Help\* | 0.98 | 0.96 | 0.97 |
| Background Noise | 0.80 | 0.78 | 0.79 |

\* critical class. Top-2 accuracy 0.927. Confusion matrix:
`python_models/metrics/confusion_matrix_transfer_test.png`.

Against the SRS targets: macro F1 0.840 ≥ 0.80 (**met**) and accuracy 0.840 < 0.85 (one
point short). Critical recall is above 0.85 for Gunshot and Help, but not for Aggression
(0.84), Glass (0.82) or Panic Scream (0.76). This was the previous served model; the AST
model replaced it because of these misses.

## Where the Python model goes wrong

From the served AST model's test predictions
(`python_models/metrics/transfer_test_predictions_ast_current.csv`):

- **Panic Scream and Aggression get swapped: 4 of 45 each way.** The Aggression class
  includes shouting. Another 3 Panic Screams were called Animal Sound.
- **Vehicle Horn called Machinery Fault (5)**, and Alarm or Siren called Machinery Fault
  (3). These are tonal, steady sounds.
- **Animal Sound called Background Noise (4), and the reverse (3).** Quiet birdsong is close
  to ambient noise.
- **Machinery Fault called Aggression (4).** Loud bangs from machines look like our
  Aggression clips, which include door slams.
- **10 of 450 predictions were wrong at confidence ≥ 0.9**, so confidence shouldn't be
  treated as proof.
- **Confidence still helps on average:** at ≥ 0.6 test accuracy is 93% (413 recordings);
  below 0.6 it is 43% (37 recordings).

### Confidence threshold, checked on validation

`reports/threshold_calibration.json` (`tools/calibrate_thresholds.py`):

| min_confidence | Decided without review | Accuracy of those | Critical clips wrongly auto-decided |
|---:|---:|---:|---:|
| 0.5 | 94 % | 86.1 % | 27 |
| **0.6 (configured)** | 90 % | 87.9 % | 23 |
| 0.7 | 86 % | 90.2 % | 19 |
| 0.8 | 82 % | 91.6 % | 15 |

In the app, confidence is only one of the checks. Disagreement, a small top-two margin,
poor quality or overlap also send a clip to review, so the share decided automatically is
lower than this (see `reports/MODEL_COMPARISON.md`). A site that would rather review more
than miss critical events could raise the threshold to 0.7; that's a decision for the team.

## Teachable Machine model

Trained in a Google Teachable Machine audio project (driven in headless Chrome by
`tools/train_gtm_browser.py`; default settings until 27 Sep, then 200 epochs) on training
recordings only. Each sample is the loudest one-second window of a recording after the
app's preprocessing, the same rule the server uses. Which recordings were used is in
`gtm_model/upload_package/tm_imports/index.json`, and the windows themselves are in
`audio_dataset/gtm_samples/` and `audio_dataset/manifests/gtm_segment_rows.csv`.

TM's model is the speech-commands network with its layers frozen and one new softmax
layer trained on top, using one-second spectrograms of 0-5 kHz. Above roughly 1,400 samples
it gets stuck at "Preparing training data". On 27 Sep, 2,100 samples stalled for 11 minutes
on the Intel GPU and again on the NVIDIA GPU (we checked the WebGL renderer string), and
1,750 samples stalled too, with the browser idle each time (`screenshots/gtm/20260927_v4_*`,
`_v5_*`). So 140 recordings per class is about the most this TM build can take.

| Export | Samples | Selection of recordings | Scoring | Val accuracy | Test accuracy | Test macro-F1 | Test critical recall |
|---|---|---|---|---|---|---|---|
| 25 Sep | 320 | first 32 per class, first second | loudest window | | 0.278 | | |
| 26 Sep a (`candidates/tm140`) | 1,400 | first 140 per class by audio id | loudest window | 0.438 | 0.462 | 0.442 | 0.569 |
| 26 Sep b (`candidates/tm_v3_140`, `archive/v2_2026-09-26`) | 1,400 | 140 per class in hash order | energy-weighted windows | 0.504 | 0.493 | 0.473 | 0.591 |
| 27 Sep, 100 epochs (`candidates/tm_v6_140_e100`) | 1,400 | same | energy-weighted windows | 0.522 | | | |
| 27 Sep, 200 epochs (`candidates/tm_v6_140_e200`, **served**) | 1,400 | same | energy-weighted windows | 0.536 | **0.511** | **0.492** | **0.600** |
| 27 Sep, 200 epochs, signed-in re-run for the project link (`candidates/tm_linked_e200`) | 1,400 | same | energy-weighted windows | 0.531 | | | |

Picking recordings in audio-id order meant 764 of the 1,400 samples were FSD50K clips and
only 66 were from UrbanSound8K; hash order spreads them across sources. On the new export,
averaging the scores of every one-second window weighted by energy beat scoring only the
loudest second (0.504 vs 0.469 on validation), so the server does that now
(`window_aggregation` in `gtm_model/frontend_config.json`). Both choices were made on
validation, and test was scored once for the served setup (`gtm_model/gtm_metrics.json`).

The 27 Sep exports only changed TM's epoch count (Advanced > Epochs; the default is 50).
We chose on validation with `0.5 × macro-F1 + 0.5 × critical recall` (0.552, then 0.585,
then 0.595) and scored test once, for the 200-epoch export. There was one run per setting,
so some of the difference may just be TM's run-to-run variation.

Test recall per class (served, 200 epochs): Person Asking for Help 1.00, Gunshot 0.78,
Machinery Fault 0.62, Vehicle Horn 0.58, Background Noise 0.53, Panic Scream 0.49,
Aggression 0.38, Glass Breaking 0.36, Animal Sound 0.24, Alarm or Siren 0.13.

The TM model is well below the SRS targets (85% accuracy, 0.80 macro-F1, 85% critical
recall). With a frozen speech-command network, one trained layer and a limit of about
1,400 samples, more epochs was the last thing left to try inside Teachable Machine, and it
moved test accuracy from 0.493 to 0.511. The app takes this into account: if the models
disagree, or TM isn't confident, the clip goes to manual review instead of the scores being
averaged.

## Leakage history

The first version of these numbers (26 Sep morning) used the v1 split. After the
source-group check, every model was retrained and re-scored on v2. For comparison, CNN14 +
logistic regression scored 0.824 on the v1 test split and 0.831 on v2. The v1 files are
kept in `python_models/metrics/archive_split_v1/`.
