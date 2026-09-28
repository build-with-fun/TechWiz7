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

## Three model families, and the ensemble that is served

| Family | Input | Test acc. | Macro F1 | Critical recall | Evidence |
|---|---|---:|---:|---:|---|
| HistGradientBoosting | 254 hand-made features (MFCC, deltas, chroma, ZCR, RMS, centroid, bandwidth, roll-off, flatness, onset, tempo, 128 mel bands) | 0.731 | 0.730 | 0.822 | `classical_metrics_hgb_split_v2.json` |
| CRNN (PyTorch) | 128 x 94 log-mel | 0.600 | 0.596 | 0.667 | `deep_metrics_deep.json` (**v1 split**, not re-run) |
| CNN14 embedding + MLP | 2,048-d PANNs CNN14 embedding (AudioSet-pretrained) | 0.840 | 0.840 | 0.862 | `transfer_test_current.json` |
| CNN14 embedding + logreg, +4,200 augmented train copies | 2,048-d PANNs CNN14 embedding | 0.838 | 0.837 | 0.849 | `transfer_test_current_aug.json` |
| AST embedding + logreg (served until 28 Sep) | 2,063-d Audio Spectrogram Transformer embedding (pooler + mean patch token + 527 AudioSet scores) | 0.891 | 0.892 | 0.907 | `transfer_test_ast_current.json` |
| CLAP embedding + logreg | 1,024-d CLAP audio embedding (LAION, trained on audio-text pairs) | 0.922 | 0.922 | 0.933 | `ensemble_test_ensemble_current.json` |
| **Ensemble: AST + CLAP + CNN14, each + logreg (served)** | the three embeddings, one logistic regression each, probabilities averaged | **0.933** | **0.933** | **0.942** | `ensemble_test_ensemble_current.json` |

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

### Served model: ensemble of three embeddings (28 Sep)

AST, CLAP and CNN14 each turn the clip into an embedding, each embedding gets its own
logistic regression, and the model returns the average of the three score lists
(`src/inference/ensemble.py`, `python_models/train_ensemble.py`). CLAP
(`laion/larger_clap_general`) learned from audio paired with text, so it gets different
clips wrong than the two AudioSet networks do.

How it was chosen:

- **Logistic regression for every member.** A plain average only works if every member's
  scores mean the same thing, and logistic regression gives well-calibrated ones. (On its
  own grid CLAP's best was an SVM at 0.909 against 0.904 for logistic regression; the
  difference is small, and SVM scores are not calibrated.)
- **C from the training recordings only**, by 5-fold cross-validation grouped by source
  (AST 0.003, CLAP 0.01, CNN14 0.003). On those folds the three together scored 0.840,
  CLAP alone 0.832 and AST alone 0.809.
- **Members picked on validation**, equal weights, smaller set on a tie:

| Members | Val acc. | Val macro F1 | Val critical recall | Score |
|---|---:|---:|---:|---:|
| **AST + CLAP + CNN14** | **0.911** | **0.910** | **0.916** | **0.913** |
| AST + CLAP | 0.902 | 0.902 | 0.916 | 0.909 |
| CLAP + CNN14 | 0.907 | 0.906 | 0.911 | 0.908 |
| CLAP | 0.904 | 0.904 | 0.911 | 0.908 |
| AST + CNN14 | 0.882 | 0.880 | 0.880 | 0.880 |
| AST | 0.867 | 0.866 | 0.884 | 0.875 |
| CNN14 | 0.844 | 0.843 | 0.836 | 0.840 |

(`python_models/metrics/ensemble_selection_ensemble_current.json`.) Test was scored once
afterwards.

**Speed.** The three networks run at the same time. Over HTTP, with the app's database and
images included, a 30-second upload took 5.4 s (median, worst 5.7 s) and a 2-second live
window 1.5 s, against the SRS limits of 8 s and 3 s (`reports/performance.json`). CLAP
alone would be about three times faster and scored 0.005 lower on validation.

**A bug found on the way.** The first ensemble (earlier on 28 Sep) was trained on CLAP
embeddings made without passing the sample rate, so the 16 kHz cache was read as 48 kHz.
The app resamples properly, so what it saw didn't match training; the comparison report
caught it (0.898 through the app against 0.916 offline, where AST alone had matched
exactly). The trainer now passes the rate (`tests/test_transfer_model.py`), CLAP was
re-embedded and everything below was redone.

### Test results per class (served ensemble, 450 recordings)

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Machinery Fault | 0.88 | 0.96 | 0.91 |
| Glass Breaking\* | 0.96 | 0.96 | 0.96 |
| Alarm or Siren | 0.95 | 0.93 | 0.94 |
| Vehicle Horn | 0.98 | 0.93 | 0.95 |
| Animal Sound | 0.89 | 0.89 | 0.89 |
| Gunshot\* | 0.96 | 0.98 | 0.97 |
| Panic Scream\* | 0.95 | 0.89 | 0.92 |
| Aggression\* | 0.91 | 0.89 | 0.90 |
| Person Asking for Help\* | 1.00 | 1.00 | 1.00 |
| Background Noise | 0.87 | 0.91 | 0.89 |

\* critical class. Top-2 accuracy 0.991. Confusion matrix:
`python_models/metrics/confusion_matrix_ensemble_test.png`.

Against the SRS targets, all three are **met**: accuracy 0.933, macro F1 0.933, and every
critical class at 0.85 recall or more (Gunshot 0.98, Glass 0.96, Panic Scream 0.89,
Aggression 0.89, Help 1.00).

Its confidence is also easier to trust: of 215 test clips scored 0.9 or higher, one was
wrong (AST alone had 10 wrong at that level), and clips at 0.7 or higher were 98.5% right.

### Served until 28 Sep: AST embedding + logistic regression

The Audio Spectrogram Transformer (AST), pretrained on AudioSet, with logistic regression on
top. The model card is `MIT/ast-finetuned-audioset-10-10-0.4593`, pinned to revision
`f826b80d28226b62986cc218e5cec390b1096902`. Validation picked logreg with `C 0.01`
(selection score 0.885, `transfer_selection_ast_current.json`). The bundle is kept in
`python_models/archive/ast_logreg_2026-09-26/`.

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

Accuracy 0.891 and macro F1 0.892 met the targets; Panic Scream (0.822) and Aggression
(0.844) were below 0.85. The CNN14 + MLP model before it (accuracy 0.840) was one point short
on accuracy, which is why the AST backbone was tried.

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

From the served ensemble's test predictions
(`python_models/metrics/ensemble_test_predictions_ensemble_current.csv`), 30 of 450 wrong:

- **Animal Sound and Background Noise: 4 and 3.** Quiet birdsong is close to ambient noise.
- **Panic Scream called Aggression (3), and Aggression spread over five other classes.**
  The Aggression class mixes shouting with door slams.
- **Alarm or Siren called Glass Breaking (2), Vehicle Horn called Machinery Fault (2).**

## Decision thresholds, chosen on validation

`tools/calibrate_thresholds.py` runs both served models on the 450 validation clips, then
the app's own consistency and review code for every combination of four thresholds. A clip
is accepted when no review condition fires. The rule was set before looking: accept as
many clips as possible while at least 97% of the accepted ones are right (overall and for
critical classes). Result (`reports/threshold_calibration.json`):

| Setting | Before (27 Sep values) | Chosen |
|---|---:|---:|
| `confidence.min_confidence` (floor) | 0.60 | 0.40 |
| `confidence.low_confidence_band` (either model below it: review) | 0.75 | 0.40 |
| `confidence.top_two_margin_min` | 0.15 | 0.20 |
| `consistency.confident_agreement_min` (new) | none | 0.50 |
| Validation clips accepted automatically | 21.1% | 42.4% |
| Accepted clips that were correct | 100% (95) | 97.4% (186 of 191) |
| Accepted critical-class clips that were correct | 100% | 98.2% (109 of 111) |
| Validation clips labelled Uncertain Result | 59 | 4 |

Both columns use the served models and the new status rules; the left one keeps the
27 Sep values.

A floor of 0.40 sounds low, but no clip is accepted on one model's word. Both models must
pick the same class, the top class must lead the next by 0.20, no second class may reach
0.25, the audio can't be Poor, and a critical class needs a Strong or Acceptable Match.
Clips that pass all of that were 97.4% right. For fewer automatic decisions, pick a
stricter row of the same table: floor 0.60, band 0.60, confident agreement 0.70 accepts
28.4% with 2 wrong in 128 (98.4%).

Two rule changes came with this (28 Sep):

- **Two confident models that agree make an Acceptable Match**, even if their scores differ
  a lot. They are trained separately and score on different scales (SRS Step 11). Before,
  Python 0.99 and TM 0.80 on the same class was a Weak Match.
- **Clearer statuses.** Different classes: Model Disagreement (it used to show as Uncertain
  when TM was also unsure). Same class, Python unsure: Uncertain Result. Same class, TM
  unsure: Weak Match. "Possible false alarm" no longer fires on the score gap alone.

### Confidence of the Python model alone

On the validation split, accuracy of the ensemble at or above each confidence (no TM check):

| Confidence ≥ | Clips | Accuracy of those |
|---:|---:|---:|
| 0.4 | 96% (433) | 92.6% |
| 0.5 | 92% (413) | 94.2% |
| 0.6 | 85% (381) | 95.0% |
| 0.7 | 76% (343) | 97.1% |
| 0.8 | 65% (294) | 98.6% |
| 0.9 | 46% (209) | 99.0% |

Alone, the Python model needs about 0.7 to be 97% right. Asking TM to agree gets there at
a lower floor, which is why both models are checked.

## Teachable Machine model

Trained in a Google Teachable Machine audio project (driven in headless Chrome by
`tools/train_gtm_browser.py`; default settings until 27 Sep, then 200 epochs) on training
recordings only. Each sample is the loudest one-second window of a recording after the
app's preprocessing, the same rule the server uses. Which recordings were used is in
`gtm_model/upload_package/tm_imports/index.json`, and the windows themselves are in
`audio_dataset/gtm_samples/` and `audio_dataset/manifests/gtm_segment_rows.csv`.

TM's model is the speech-commands network with its layers frozen and one new softmax
layer trained on top, using one-second spectrograms of 0-5 kHz.

**The 1,400-sample limit was a JavaScript stack overflow.** Above about 1,400 samples TM
used to sit at "Preparing training data" with the browser idle (`screenshots/gtm/20260927_v4_*`,
`_v5_*`). The trainer only logged console messages, so the error was never seen. Once it
also logged page errors, TM showed `Maximum call stack size exceeded`: it loads every sample
in one call, and that call outgrows Chrome's default stack. Starting Chrome with a bigger
stack (`train_gtm_browser.py --js-stack-kb 4000`) let it train on all 2,100 training
recordings (`screenshots/gtm/20260928_v7_2100_e200_*`, 38 minutes for 200 epochs).

| Export | Samples | Selection of recordings | Scoring | Val accuracy | Test accuracy | Test macro-F1 | Test critical recall |
|---|---|---|---|---|---|---|---|
| 25 Sep | 320 | first 32 per class, first second | loudest window | | 0.278 | | |
| 26 Sep a (`candidates/tm140`) | 1,400 | first 140 per class by audio id | loudest window | 0.438 | 0.462 | 0.442 | 0.569 |
| 26 Sep b (`candidates/tm_v3_140`, `archive/v2_2026-09-26`) | 1,400 | 140 per class in hash order | energy-weighted windows | 0.504 | 0.493 | 0.473 | 0.591 |
| 27 Sep, 100 epochs (`candidates/tm_v6_140_e100`) | 1,400 | same | energy-weighted windows | 0.522 | | | |
| 27 Sep, 200 epochs (`candidates/tm_v6_140_e200`, served until 28 Sep) | 1,400 | same | energy-weighted windows | 0.536 | 0.511 | 0.492 | 0.600 |
| 27 Sep, 200 epochs, signed-in re-run for the project link (`candidates/tm_linked_e200`) | 1,400 | same | energy-weighted windows | 0.531 | | | |
| 28 Sep, 200 epochs (`candidates/tm_v7_2100_e200`, **served**) | 2,100 | every training recording | energy-weighted windows | 0.564 | **0.573** | **0.568** | **0.640** |

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

The 28 Sep export beat the 27 Sep one on validation (0.564 vs 0.536 accuracy) and was then
scored once on test. Averaging window scores as a geometric mean (`energy_weighted_log`)
tied on validation, so the served setting stayed.

Test recall per class (served 28 Sep export): Person Asking for Help 1.00, Gunshot 0.71,
Vehicle Horn 0.71, Glass Breaking 0.62, Machinery Fault 0.62, Background Noise 0.53,
Panic Scream 0.49, Aggression 0.38, Animal Sound 0.38, Alarm or Siren 0.29.

### How far Teachable Machine can go here

TM only trains its last layer, so its training can be copied offline: run the frozen network
on every one-second window, then fit a softmax layer on the result. This copy gives the same
validation accuracy as the app (0.536 for the 27 Sep head), so we used it to try ideas before
spending a browser run (validation only):

- More epochs alone didn't help; the copy overfits after about 50.
- More recordings did: all 2,100 instead of 1,400 added about 5 points, which TM then
  confirmed.
- Three windows per recording were no better than one.
- Speeding audio up 1.6x or 2x, so TM's 0-5 kHz view covers more of the spectrum, made it
  worse by 2.5 and 5 points.

With TM's normal input, validation accuracy stays around 0.50 to 0.56 whatever we feed it:
the frozen network was trained on spoken words, not on everyday sounds. So TM stays well
below the SRS targets. The app allows for that: if the models disagree or either is below
the floor, the clip goes to manual review, and the two scores are never averaged.

## Leakage history

The first version of these numbers (26 Sep morning) used the v1 split. After the
source-group check, every model was retrained and re-scored on v2. For comparison, CNN14 +
logistic regression scored 0.824 on the v1 test split and 0.831 on v2. The v1 files are
kept in `python_models/metrics/archive_split_v1/`.
