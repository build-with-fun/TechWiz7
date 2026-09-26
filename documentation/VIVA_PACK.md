# Viva pack

Short answers to what a jury is likely to ask, each pointing at the code. Every number here
is from a file in the repository, with the file named. **Each member should re-run the
commands for their own modules and put their name in the ownership table before the viva.**

## 1. Who owns what

Fill this in with real names. The `Owner:` lines in old code comments do not match the Git
history and must not be used as evidence.

| Module | Files | Owner | Can demo |
|---|---|---|---|
| Web app, auth, roles | `src/app.py`, `src/auth.py`, `src/api/pages.py`, `templates/` | | login, roles, dashboard |
| Upload and live capture | `src/api/audio_api.py`, `src/api/live_api.py`, `static/js/live.js`, `upload.js` | | upload, live monitor |
| Audio preprocessing and quality | `audio_preprocessing/` | | silence/clipping rejection |
| Features and embeddings | `feature_extraction/features.py`, `embeddings.py` | | what one feature means |
| Python models | `python_models/train_transfer.py`, `train_classical.py`, `train_deep.py` | | re-run selection |
| Teachable Machine | `audio_dataset/scripts/make_gtm_imports.py`, `tools/train_gtm_browser.py`, `src/inference/gtm_predictor.py` | | retrain in TM |
| Comparison, rules, review | `src/inference/consistency.py`, `src/services/pipeline.py`, `alert_rules/` | | change a rule live |
| Data, split, augmentation | `audio_dataset/`, `augmentation/` | | leakage audit |
| Reports, analytics, monitoring | `src/api/reports_api.py`, `src/services/monitoring.py` | | report download |

## 2. The request path (if you forget the code, remember this)

```text
browser upload / 2 s mic window
 -> POST /api/audio/upload  or  /api/live/sessions/<id>/windows      (src/api/)
 -> AudioPipeline: decode, validate, quality verdict, high-pass, denoise, trim, normalise,
    resample to 16 kHz                                                (audio_preprocessing/)
 -> Python model: CNN14 embedding -> MLP -> 10 scores                  (python_models/best/)
 -> TM model: loudest 1 s -> browser-style FFT 43x232 -> CNN -> 10 scores (gtm_model/)
    (each model receives only the preprocessed audio)
 -> compare: agree?, |top1 - top1|, top-two margin, overlap            (consistency.py)
 -> rules: severity, repeated-detection tracker, alert, review routing (pipeline.py)
 -> persist: audio, both score lists, versions, alert, review, audit   (persistence.py)
 -> dashboard, event page, report                                      (templates/)
```

## 3. Why three model families, and why this one won

| Family | Test acc. | Macro F1 | Critical recall | Why it was tried |
|---|---:|---:|---:|---|
| HistGradientBoosting on 254 hand-made features | 0.731 | 0.730 | 0.822 | Classical baseline; every feature is explainable (MFCC, chroma, ZCR, RMS, centroid...) |
| CRNN on log-mel (PyTorch) | 0.600 | 0.596 | 0.667 | Learns its own features; *old v1 split*; too little data to train from scratch |
| CNN14 embeddings + MLP | 0.840 | 0.840 | 0.862 | Transfer learning: a network pretrained on AudioSet already separates textures such as slam vs shot |
| CNN14 embeddings + logreg, +4,200 augmented copies | 0.838 | 0.837 | 0.849 | Whether augmentation closed the accuracy gap; it did not generalise to test |
| **AST embeddings + logistic regression (served)** | **0.891** | **0.892** | **0.907** | A transformer over the same spectrogram, stronger than CNN14 on AudioSet; cleared all three targets |

Sources: `python_models/metrics/transfer_test_ast_current.json`, `transfer_test_current.json`,
`transfer_test_current_aug.json`, `classical_metrics_hgb_split_v2.json`, `deep_metrics_deep.json`.

Selection rule, fixed before looking at test numbers: highest
`0.5 x validation macro-F1 + 0.5 x validation critical recall`, ties to the faster model.
Accuracy alone would trade a missed gunshot for a correct horn.

## 4. The split, and the leak we found

- 3,000 originals, exactly 210 / 45 / 45 per class (`data/splits/split.json`).
- **v1 split clip by clip.** On 26 Sep we grouped clips by the recording they came from
  and found 125 Freesound uploads (527 clips) and 37 synthetic voice+phrase pairs (116
  clips) spread across partitions. Example: 28 slices of one siren in train, val *and*
  test. That inflates test scores.
- **v2** (`audio_dataset/build_split.py`, `source_group()`) assigns whole groups. The test
  `test_no_source_recording_or_tts_phrase_spans_two_splits` now guards it.
- Augmented copies and TM samples are made only from training recordings, and a test checks
  the parent IDs.

## 5. Proving the two models are independent

- `GtmModelPredictor.predict_from_preprocessed(preprocessed)` takes audio and nothing
  else; there is no parameter through which a Python score could arrive.
- `tests/test_model_independence.py` checks the signatures, checks the modules cannot import
  each other, and checks the TM output does not change when the Python result does.
- The TM model was trained in Teachable Machine from its own sample format (browser FFT
  frames), not from our features. Only the comparison layer sees both results.
- Live demo: `python tools/predict.py <clip>` prints both score lists side by side; they
  often disagree, which a copied result never would.

## 6. Five known failure cases (say these before you are asked)

1. **Panic Scream heard as Aggression** (9 of 45 test screams). Both are raised human
   voices, and our Aggression class contains shouting.
2. **Aggression is partly a proxy class**: many of its clips are door slams and thumps, so
   a real argument may not look like our training data.
3. **Confident mistakes**: 21 of 450 test predictions were wrong at confidence ≥ 0.9.
   Confidence is the model's estimate, not proof; the review queue exists for this.
4. **Sounds outside the ten classes** (fireworks, knocking, clapping) are forced into the
   nearest class. See `reports/ROBUSTNESS.md` part B for how often that reaches review.
5. **Trimmed re-uploads** are recognised as near-duplicates only about 22 % of the time
   (`reports/near_duplicates.json`); re-encodes and volume changes are ~95 %.

## 7. Surprise modifications: where to change what

| Task | Change | Then |
|---|---|---|
| Confidence threshold | `config/thresholds.json` → `confidence.min_confidence` (or Admin → Configuration) | takes effect on the next request |
| Top-two margin | same file → `confidence.top_two_margin_min` | same |
| Repeated-detection requirement | `config/thresholds.json` → `repeat_detection.required_consecutive_detections`, or per class in `alert_rules/alert_rules.json` | same |
| Add an alert rule / change severity | `alert_rules/alert_rules.json`, the class's entry (`severity`, `recommended_action`, `escalation`) | validated on save; `pytest tests/test_alert_rules_config.py` |
| Segment / live window length | `config/thresholds.json` → `audio.segment_duration_sec`, `audio.live_window_sec` | live window is read by `live.js` from the page config |
| Support a new audio format | add the suffix to `_ACCEPTED_SUFFIXES` and `_MIME_BY_SUFFIX` in `src/api/audio_api.py`, `AUDIO_MIME_TYPES` in `src/models.py`, and `audio.supported_formats` in `config/thresholds.json`; FFmpeg decodes it | `pytest tests/test_audio_download_and_history.py` |
| Add a dashboard/search filter | `filter_fields()` and the query in `src/services/search.py`; the form renders from that list | `pytest tests/test_audio_api.py -k filter` |
| Add a sound category | add it to `config/classes.json` and `alert_rules/alert_rules.json`, add recordings to the manifest, rebuild the split, retrain both models (`train_transfer`, TM) | both models must name the same classes or the app refuses to start |

## 8. Numbers to remember

- Dataset: 3,000 originals, 300 per class, 2,100 / 450 / 450, 2,353 source groups.
- Python model on the test split: accuracy 0.891, macro F1 0.892, critical recall 0.907
  (AST embeddings + logistic regression, `python_models/metrics/transfer_test_ast_current.json`).
- Confidence ≥ 0.6 → 93 % correct on test; below 0.6 → 43 %.
- Old model before 26 Sep: 0.698 accuracy.
- TM model: see `gtm_model/gtm_metrics.json` (fill in after the v2 retrain).
