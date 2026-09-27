# Viva pack

Short answers to what the jury is likely to ask, each pointing at the code. Every number
here comes from a file in the repository, named next to it. **Before the viva, each member
should re-run the commands for their own modules and check their name in the table.**

## 1. Who owns what

Owners as stated by the team on 27 September 2026 (details in
`documentation/TEAM_CONTRIBUTION_RECORD.md`).

| Module | Files | Owner | Can demo |
|---|---|---|---|
| Web app, auth, roles | `src/app.py`, `src/auth.py`, `src/api/pages.py`, `templates/` | Ammar Ahmer (backend), Aimon (pages) | login, roles, dashboard |
| Upload and live capture | `src/api/audio_api.py`, `src/api/live_api.py`, `static/js/live.js`, `upload.js` | Ammar Ahmer (API), Aimon (browser scripts) | upload, live monitor |
| Audio preprocessing and quality | `audio_preprocessing/` | Ammar Ahmer | silence/clipping rejection |
| Features and embeddings | `feature_extraction/features.py`, `embeddings.py`, `ast_embeddings.py` | Ammar Ahmer | what one feature means |
| Python models | `python_models/train_transfer.py`, `train_classical.py`, `train_deep.py` | Ammar Ahmer | re-run selection |
| Teachable Machine | `audio_dataset/scripts/make_gtm_imports.py`, `tools/train_gtm_browser.py`, `src/inference/gtm_predictor.py` | Ammar Ahmer | retrain in TM |
| Comparison, rules, review | `src/inference/consistency.py`, `src/services/pipeline.py`, `alert_rules/` | Ammar Ahmer | change a rule live |
| Data, split, augmentation | `audio_dataset/`, `augmentation/` | Ammar Ahmer | leakage audit |
| Reports, analytics, monitoring | `src/api/reports_api.py`, `src/services/monitoring.py` | Ammar Ahmer | report download |
| Testing and test recordings | `tests/`, `TEST_PLAN.md`, `tools/browser_acceptance.py` | Khizr, Aimon | run the suite, a live-microphone test |

## 2. The request path

```text
browser upload / 2 s mic window
 -> POST /api/audio/upload  or  /api/live/sessions/<id>/windows      (src/api/)
 -> AudioPipeline: decode, validate, quality verdict, high-pass, denoise, trim, normalise,
    resample to 16 kHz                                                (audio_preprocessing/)
 -> Python model: AST embedding -> logistic regression -> 10 scores    (python_models/best/)
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
| **AST embeddings + logistic regression (served)** | **0.891** | **0.892** | **0.907** | A transformer over the same spectrogram, stronger than CNN14 on AudioSet; meets the accuracy and macro-F1 targets |

Sources: `python_models/metrics/transfer_test_ast_current.json`, `transfer_test_current.json`,
`transfer_test_current_aug.json`, `classical_metrics_hgb_split_v2.json`, `deep_metrics_deep.json`.

Selection rule, fixed before looking at any test numbers: the highest
`0.5 x validation macro-F1 + 0.5 x validation critical recall`, with ties going to the
faster model. Going by accuracy alone could trade a missed gunshot for a correct horn.

## 4. The split, and the leak we found

- 3,000 originals, exactly 210 / 45 / 45 per class (`data/splits/split.json`).
- **v1 split clip by clip.** On 26 Sep we grouped clips by the recording they came from
  and found 125 Freesound uploads (527 clips) and 37 synthetic voice+phrase pairs (116
  clips) spread across partitions. For example, 28 slices of one siren were in train, val
  *and* test, which makes test scores look better than they are.
- **v2** (`audio_dataset/build_split.py`, `source_group()`) assigns whole groups. The test
  `test_no_source_recording_or_tts_phrase_spans_two_splits` now guards it.
- Augmented copies and TM samples are made only from training recordings, and a test checks
  the parent IDs.

## 5. Proving the two models are independent

- `GtmModelPredictor.predict_from_preprocessed(preprocessed)` takes audio and nothing
  else; there is no parameter through which a Python score could arrive.
- `tests/test_model_independence.py` checks the signatures, checks the modules cannot import
  each other, and checks the TM output does not change when the Python result does.
- The TM model was trained in Teachable Machine on its own sample format (browser FFT
  frames), not on our features. Only the comparison code sees both results.
- To show it live: `python tools/predict.py <clip>` prints both score lists side by side.
  They often disagree, which they couldn't if one copied the other.

## 6. Five known failure cases (mention these before you're asked)

1. **Panic Scream and Aggression get mixed up** (4 of 45 test screams were called
   Aggression, and 4 of 45 Aggression clips Panic Scream). Both are raised human voices,
   and our Aggression class includes shouting.
2. **Aggression is partly a stand-in class**: many of its clips are door slams and thumps,
   so a real argument may not sound like our training data.
3. **Confident mistakes**: 10 of 450 test predictions were wrong at confidence ≥ 0.9.
   Confidence is only the model's own estimate; that's what the review queue is for.
4. **Sounds outside the ten classes** (fireworks, knocking, clapping) still get the nearest
   class. `reports/ROBUSTNESS.md` part B shows how often they end up in review.
5. **Trimmed re-uploads** are only recognised as near-duplicates about 22% of the time
   (`reports/near_duplicates.json`); re-encodes and volume changes are caught about 95%.

## 7. Surprise changes: where to edit what

| Task | Change | Then |
|---|---|---|
| Confidence threshold | `config/thresholds.json`, `confidence.min_confidence` (or Admin > Configuration) | applies from the next request |
| Top-two margin | same file, `confidence.top_two_margin_min` | same |
| Repeated-detection requirement | `config/thresholds.json`, `repeat_detection.required_consecutive_detections`, or per class in `alert_rules/alert_rules.json` | same |
| Add an alert rule / change severity | `alert_rules/alert_rules.json`, the class's entry (`severity`, `recommended_action`, `escalation`) | validated on save; `pytest tests/test_alert_rules_config.py` |
| Segment / live window length | `config/thresholds.json`, `audio.segment_duration_sec` and `audio.live_window_sec` | `live.js` reads the live window from the page config |
| Support a new audio format | add the suffix to `_ACCEPTED_SUFFIXES` and `_MIME_BY_SUFFIX` in `src/api/audio_api.py`, `AUDIO_MIME_TYPES` in `src/models.py`, and `audio.supported_formats` in `config/thresholds.json`; FFmpeg decodes it | `pytest tests/test_audio_download_and_history.py` |
| Add a dashboard/search filter | `filter_fields()` and the query in `src/services/search.py`; the form renders from that list | `pytest tests/test_audio_api.py -k filter` |
| Add a sound category | add it to `config/classes.json` and `alert_rules/alert_rules.json`, add recordings to the manifest, rebuild the split, retrain both models (`train_transfer`, TM) | both models must name the same classes or the app refuses to start |

## 8. Numbers to remember

- Dataset: 3,000 originals, 300 per class, 2,100 / 450 / 450, 2,353 source groups.
- Python model on the test split: accuracy 0.891, macro F1 0.892, critical recall 0.907
  (AST embeddings + logistic regression, `python_models/metrics/transfer_test_ast_current.json`).
- Confidence ≥ 0.6: 93% correct on test; below 0.6: 43%.
- Old model before 26 Sep: 0.698 accuracy.
- TM model on the test split: accuracy 0.511, macro F1 0.492, critical recall 0.600
  (`gtm_model/gtm_metrics.json`).
