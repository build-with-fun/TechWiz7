# Development Log — SonicSentinel AI

Format per SRS §1.8 rule 3: work completed, problems encountered, dataset changes,
model failures, code changes, tests performed. Newest first.

---

## Day 4, evening — 2026-09-26

### Work completed
- **Teachable Machine rebuilt.** Found that the served TM export had been trained on the first
  140 recordings per class by audio id: 764 of its 1,400 samples were FSD50K clips and only 66
  came from UrbanSound8K. `make_gtm_imports.py` now takes recordings in a fixed hash order,
  writes the exact one-second windows as WAV evidence (`audio_dataset/gtm_samples/`,
  `manifests/gtm_segment_rows.csv`), and the server can average every window weighted by
  energy. Validation 0.438 → 0.504; test (scored once) 0.462 → 0.493 accuracy.
- **Withdrew the 25 Sep TM segment cut.** Under split v2 about 30 % of its parents were
  validation or test recordings. No model used it; the lists are archived with a note in
  `audio_dataset/manifests/archive_split_v1/`.
- **Second pretrained backbone for the Python model**: AST (Audio Spectrogram Transformer,
  AudioSet), `feature_extraction/ast_embeddings.py`, selectable with
  `train_transfer.py --backbone ast`. CNN14 plateaued near 0.84 on validation with every
  classifier we tried, and seed-to-seed spread of the MLP alone was 0.81-0.84. AST's 2,063-d
  embedding (pooler + mean patch token + 527 AudioSet scores, averaged over 10.24 s chunks)
  with the same logistic-regression head and the same selection rule reached 0.891 / 0.892 /
  0.907 on test and replaced CNN14 in `python_models/best/`.
- **Interface redesign**: one dark and one light theme, Inter font (self-hosted, OFL), a
  dashboard built around SRS Step 18 (latest detection with both models' top 3, agreement,
  quality, severity, alert status, waveform, spectrogram and playback; review queue;
  critical timeline; category, severity and trend charts drawn as server-side SVG), a result
  card after each upload, and a readable spectrogram colour map. Removed the 3D decoration,
  gradient hero and four-palette theme picker. `tools/check_ui.py`: no layout failures at 320,
  390, 768 or 1440 px.
- **Clean-up**: comment and docstring lines 9,434 → 5,651 (-40 %); pyflakes clean apart from
  one intentional import; the unused `GridRouter` removed from `tools/render_uml.py`; the old
  audit, API proposal and perception notes moved to `documentation/archive/`.

### Problems encountered and fixed
- Upload results never showed the two models' predictions: `upload.js` read `event.python`
  while the API returns `models.python`.
- The configuration page could never show its integrity hashes (read `hashes`, the snapshot
  calls them `content_hashes`).
- The live microphone status lost its styling after the first update (`live.js` replaced the
  class list); on the event page the second model card was squeezed into the gap column.
- On phones the page was 1,525 px wide: a visually hidden span in the nav escaped its
  scroll container.
- Teachable Machine stalls at "Preparing training data" with more than about 1,400 samples
  in this browser; 2,100 was still stuck after ten minutes.

### Model failures
- TM remains far below the SRS targets (test 0.493 accuracy, 0.473 macro-F1, 0.591 critical
  recall): Alarm or Siren 0.13 and Animal Sound 0.11 recall.

---

## Day 4 — 2026-09-26

### Work completed
- **Baseline audit** (`documentation/BASELINE_AUDIT.md`): 431 non-deep tests passing,
  served Python model 0.698 test accuracy, Teachable Machine (TM) model 0.278.
- **TM diagnosis.** The exported TM network scores 0.84 on its own training frames, so
  the FFT frontend and label order are fine. It was trained on 32 one-second samples per
  class, each the *first* second of a clip, while the server scores the *loudest*
  second. Rebuilt the imports (`make_gtm_imports.py`) from every training recording with
  the same `select_gtm_window` rule the server uses.
- **Preprocessing cache** (`python_models/preprocess_cache.py`): all 3,000 originals
  through the serving `AudioPipeline`, 0 rejected, 175 s.
- **Transfer-learning Python model** (`feature_extraction/embeddings.py`,
  `python_models/train_transfer.py`): PANNs CNN14 (16 kHz, AudioSet) embeddings plus a
  small classifier chosen on validation. Checkpoint verified against Zenodo's MD5.
- **Preprocessing experiment.** Compared the serving settings with a gentler variant
  (noise gate 0.4 instead of 0.75, trim at 45 dB instead of 30 dB) on validation only.
  Overall score 0.8809 vs 0.8825, i.e. a tie. The gentle variant was *worse* on Gunshot
  (-0.05) and Glass (-0.03), the classes it was meant to help, so the hypothesis was
  rejected and serving preprocessing was left unchanged.
- **Split leakage found and fixed.** An audit by source recording found 125 Freesound
  uploads (527 clips: UrbanSound8K slices, ESC-50 takes) and 37 synthetic voice+phrase
  pairs (116 clips) with members in more than one partition. `build_split.py` v2 assigns
  whole source groups; still exactly 210/45/45 per class. One Freesound upload (43806) was
  labelled both Alarm or Siren and Aggression; it is kept in train and flagged for label
  review. The v1 split is archived in `data/splits/archive_v1/`. Every model was retrained
  and re-scored on v2.
- **Augmentation** (`augmentation/`): noise, shift, pitch, stretch, volume, reverb,
  distance and device recipes; train-only copies with lineage and seeds.
- **Near-duplicates (FR lxxiv).** Measured the upload fingerprint on the real corpus: 19 %
  of hard negatives passed the 0.92 threshold and trimmed copies were found only 15 % of
  the time. Added a second stage (aligned log-mel correlation). With the fingerprint used
  only to shortlist 5 candidates: 74 % recall on true copies, 1 of 180 hard negatives
  flagged (`reports/near_duplicates.json`).
- **Live monitor rewritten** (`static/js/live.js`): continuous capture with a bounded
  queue (the old loop was deaf during every server round trip), the five SRS microphone
  states, pause/resume, disconnect handling, abortable requests, top-3 per model and an
  alert banner.
- **Product gaps closed:** anomaly checks for all six FR lxxviii categories, SRS dashboard
  items (recent uploads, quality warnings, critical timeline, admin metrics and trend),
  false-positive/false-negative and alert-response analytics, a full FR lxix event report
  with waveform and spectrogram, bit depth and the source sample rate stored per upload,
  and a replay button.

### Problems encountered
| Problem | Cause | Fix |
|---|---|---|
| Live windows stored with an empty consistency status | `live_api.py` read `comparison["status"]`; the key is `consistency_status` | Corrected; regression test added |
| Class names came back as `np.str_('Gunshot')` | sklearn `classes_` are numpy strings | Converted to `str` when the bundle loads |
| Zenodo download at ~100 KB/s | per-connection throttling | 12 parallel byte ranges, then MD5 check (`tools/fetch_pretrained.py`) |
| TM import timed out after 120 s | 400 samples take longer than 32 to decode | Wait scales with class size |
| Background jobs killed | ran embeddings, headless Chrome and tests at once on 15 GB RAM with swap full | One heavy job at a time |
| CNN14 embeddings unusable for duplicate search | post-ReLU vectors are all similar (cosine 0.99 across classes) | Dropped that approach before shipping it |
| Near-duplicate stage 2 missed an identical sound (0.86) | compared a preprocessed upload with a raw stored file | Both sides now go through the same preprocessor |

### Dataset changes
- Split algorithm `sha256-order-v1` → `sha256-order-v2-source-groups` (same seed 20260923).
- 4,200 augmented copies were generated for the v1 train split and then deleted after the
  re-split, because their parents' partitions changed; they are regenerated from v2.

### Model results on the v2 split (test, 450 recordings, scored once per family)
| Model | Accuracy | Macro F1 | Critical recall (mean) |
|---|---:|---:|---:|
| **AST + logistic regression (served)** | **0.891** | **0.892** | **0.907** |
| CNN14 + MLP (previously served) | 0.840 | 0.840 | 0.862 |
| CNN14 + logistic regression | 0.831 | 0.831 | 0.849 |
| CNN14 + logreg, +4,200 augmented copies | 0.838 | 0.837 | 0.849 |
| CNN14 + SVM | 0.818 | 0.823 | 0.796 |
| HistGradientBoosting, 254 hand-made features (refit on train+val) | 0.731 | 0.730 | 0.822 |

### Tests performed
- New: `test_augmentation.py`, `test_transfer_model.py`,
  `test_dashboard_reports_monitoring.py`, and a source-group leakage test in
  `test_frozen_split.py`.
- Split tests: 45 passed after the v2 split. Dashboard/report/monitoring/near-duplicate
  and HTTP integration tests: 27 passed.

---

## Day 3 — 2026-09-25

### Work completed
- **All API write paths smoke-tested end-to-end** (curl, admin cookie jar):
  - Alerts: acknowledge (JSON 200 + HTML-form 303 + flash), dismiss (422 without
    reason → 200 with reason → 409 once closed), escalate (200, 422 on unknown
    severity), invalid-state transitions now flash a readable error on form posts.
  - Reviews: confirm (adopts original prediction), override with a real class name
    (`Panic Scream`, not `Scream` — 422 otherwise), re-decide → 409, form confirm → 303.
  - Admin: config PUT accepts only the *nested* schema; history records `config_edited`
    with before/after; users create/409/PATCH; retention preview + purge (legal hold
    respected); monitoring anomalies.
  - Live: session create requires consent (422 → 201 with UUID), window push without a
    GTM export returns a clean 503 `pipeline_unavailable` (correct pre-export behavior),
    stop → 200, push-after-stop → 409.
  - Reports: event HTML/JSON, period HTML, CSV/XLSX exports (xlsx verified as a real
    Excel 2007 file).
- **GTM sample set cut**: 5,230 two-second 16 kHz mono segments from **train-split
  originals only** (deterministic positions, refuses val/test parents), lineage ids
  `SS-<CODE>-<NNNN>S<n>`, rows in `audio_dataset/manifests/gtm_segment_rows.csv`.
- **TM upload package**: ten per-class zips + `index.csv` + training README under
  `gtm_model/upload_package/`.
- **tensorflowjs 4.20.0** installed for the GTM export conversion (see problems below).
- Full pytest: **403 passed** (90.7 s). Committed `2f39fa0` (API slices) and `1977fe9`
  (GTM sample set).

### Problems encountered and fixed
| Problem | Root cause | Fix |
|---|---|---|
| 500 on form POSTs to `/api/alerts/*` | `_back_href` was missing its endpoint argument | Signature `(default_endpoint, **params)`; all call sites updated |
| 405 when following the 303 | `curl -L` re-POSTs to a GET-only route | Test method: POST, read redirect URL, GET separately with the same cookie jar |
| 500 POST `/session` | `audit_login` not imported in `pages.py`; raw `User` row has no `.can` | Import added; `_home_for` uses `has_capability(role, …)` |
| `BuildError: auth.session` on `/login?next=` | Template used wrong endpoint name | `url_for('auth.session_login')` |
| "dismissd" typo | Naive `{action}d` formatting | `cannot be {action}ed` |
| 422 "Unknown role" on user create | Roles are the five from `ROLES`, not `operator` | Use `audio_reviewer`/`security_operator`/… |
| 500 `/api/admin/retention/preview` | Artifact keys were `events`/`live_audio`; legal-hold flag queried on `Event` | Keys are `event_records`/`audio`/`microphone_session_audio`; flag lives on `AudioFile` (join added) |
| 500 POST `/api/live/sessions` | `live_sessions.id` is a String(36) UUID PK with no default | `id=str(uuid.uuid4())` |
| 500 on window push | `pipeline_unavailable` missing from the error registry | Added to `ERROR_CODES`/`DEFAULT_STATUS` (503)/`DEFAULT_MESSAGES` |
| 500 `/api/reports/event/1` | `_dtm.timezone` attribute error | `import datetime as _dt`; `_dt.datetime.now(_dt.timezone.utc)` |
| **Config PUT destroyed `config/thresholds.json`** | Flat body `{"top_confidence_min": 0.60}` replaced the whole nested file; app refused to boot ("no such key min_confidence") | Restored from git; **lesson: PUTs must send the nested structure** (`alert_rules.json` references `$thresholds.confidence.min_confidence`) |
| `tensorflowjs` hard-crash at import | `tensorflow-decision-forests` CHECK-fails against protobuf 6.31.1 | Uninstalled tdf; pinned tfjs 4.20.0 `--no-deps`; wrapped the tdf import in the converter in try/except; protobuf 6.31.1 |

### Dataset changes
- Cut GTM segments (train-split parents only) → `audio_dataset/gtm_samples/`.
  The first cut, done 25 Sep against split v1, stamped all 5,230 segments
  `train` while the parents were spread over the old v1 groups; once split v2
  froze source groups by recording, 319 val + 328 test parents were inside that
  set (1,599 of 5,230 segments, 30.6%) — contaminated. Re-cut on 26 Sep against
  the frozen v2 `assignments`, filtering to `split == train` and leaving
  `dataset_split` blank for `build_split.py` to inherit. The shipped model was
  unaffected: the 1400 ids in `tm_imports/index.json` were always train-only.
- `assemble_manifest.py` now accepts segment ids (`S<n>` suffix stripped before the
  `SS-<CODE>-<NNNN>` check); `verify_dataset.py` gained a `SEGMENT_ID_RE`, a
  split-via-parent rule, and passes **28/28** checks on the combined 8,230-row
  manifest (`manifest_with_split_full.csv`): 2100/450/450 originals, 300/class,
  zero leaks into val/test, parent-agreement for all derived rows.

### Model failures
- First retrain run died silently when its parent shell closed (no process, log
  stopped at 09:59). Restarted under `setsid nohup` with `--tag repaired` so artifacts
  don't clobber the registered best model while running.
- Current best (xgboost depth=6): test accuracy 0.693, macro-F1 0.693, critical recall
  0.799 — **floors (0.85 / 0.80 / 0.85) not yet met**. Weakest: Animal Sound F1 0.46,
  Alarm or Siren 0.56, Machinery Fault 0.58, Aggression recall 0.49. Classical
  retrain (`--tag repaired`) running; **deep mel-CNN path (`train_deep --tag deep`)
  launched in parallel** — it trains on mel tensors rather than 254 summary columns,
  which is where the headroom is.

### Tests performed
- `pytest -q` → 403 passed. Manual curl matrix over every write endpoint including
  negative cases (422/409/503) and both form/JSON surfaces.

---

## Day 2 — 2026-09-24

### Work completed
- Dataset repair: 106 collided blobs re-hashed and re-cut (`repair_id_collision.py`),
  manifest reassembled to a clean 3,000 rows, frozen split rebuilt and verified
  2100/450/450 with 300 originals per class (commit `24b34f8`).
- Registered the sweep winner (`xgboost[depth=6]`) as model bundle
  `python_models/best/` + `ModelVersion` row (commit `46bb635`).
- Fixed `predict_proba` column resolution via `estimator.classes_` (commit `c36c6ba`).
- Built the eight missing console pages (commit `bb09dc3`).

### Problems encountered
- Manifest assembly rejected ids after collision repair until the validator matched the
  re-cut numbering; fixed in `repair_id_collision.py`.
- Class-code mapping (`config/classes.json`) drove the audio_id codes — kept as the
  single source of truth.

### Model failures
- First classical sweep showed Person Asking for Help at 0.99 (synthetic TTS is too
  easy) while Animal Sound sat at 0.46 — flagged for Day-3 retraining with feature
  weighting and the deep path.

---

## Day 1 — 2026-09-23

### Work completed
- Read both SRS PDFs; extracted mandatory patterns (dual-model, comparison taxonomy,
  floors, integrity rules) into `documentation/`.
- Corpus acquisition pipeline (`acquire_corpus.py`, `fetch_fsd50k.py`), synthetic clip
  generation for under-represented classes (`generate_help_clips.py`,
  `synthetic_targets.json`), trim to 300/class (`trim_to_300.py`).
- Feature extractor locked at 254 columns (`feature_extraction/features.py`,
  `config/features.json` audiofeat-1.0.0).
- App skeleton: Flask app factory, DB models, audio preprocessing package.

### Problems encountered
- FSD50K licensing: only CC-licensed clips retained; attribution written per file
  (`build_attribution.py`).
- Spec inconsistency noted: Step 16 lists five severity levels, FR lii lists four.
  Implemented **all five** with a configurable `active_scale` in
  `alert_rules/severity_levels.json`.