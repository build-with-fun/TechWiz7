# Development Log — SonicSentinel AI

Format per SRS §1.8 rule 3: work completed, problems encountered, dataset changes,
model failures, code changes, tests performed. Newest first.

---

## Day 5 — 2026-09-27

### Commit history, stated as it is
SRS §1.8 asks for meaningful commits on all five competition days. The Git history has
commits on 24, 25 and 27 September only. Day 1 (23 Sep) work — reading the SRS, acquiring the
corpus, the first feature extractor and app skeleton, described below — was done before the
repository's first commit (24 Sep 14:08). Day 4 (26 Sep) work was committed at 01:30 on
27 Sep as `2174c4e`. No commit has been backdated and none will be; this entry and the
per-day sections below are the record for the two days without commits.

### Selection rules, fixed before any new model was scored
Written down before the runs below started, so that validation decides and test only reports.

- **Python model.** Two candidates on the AST embeddings: the served configuration (fit on
  the 2,100 training recordings) and the same grid fit on training recordings plus their
  4,200 augmented copies (train split only; copies inherit their parent's split). Chosen on
  the 450 validation recordings: among candidates whose lowest critical-class validation
  recall is at least 0.85, the highest `0.5 × macro-F1 + 0.5 × critical recall`; if none
  reaches 0.85, the highest score overall. A difference under 0.005 keeps the served model.
  The chosen candidate is scored on test once and reported as it comes out. A combined
  AST + CNN14 embedding was considered and dropped without being run: it would add a second
  backbone to every request, and the 30-second upload already takes 7.15 s of the 8 s budget.
- **Teachable Machine.** Every retrained export is saved under `gtm_model/candidates/` and
  scored on validation with the served window aggregation. The same score
  (`0.5 × macro-F1 + 0.5 × critical recall`) picks the winner; the served export is the
  baseline (validation 0.504 accuracy, 0.490 macro-F1, 0.613 critical recall). Only the
  winner is scored on test, once.

### Work completed
- **Install fix.** `requirements.txt` still described the CNN14 model and did not list
  `transformers`, so a fresh install that followed the README could not load the served AST
  model. Added the transformers stack at the versions in use and changed the README install
  step to `tools/fetch_pretrained.py --ast`.
- **FR li wording.** Uncertain events now read "Manual Review Required" on the event page and
  in the upload result, the SRS's own phrase (they said "Routed to manual review").
- **Deployment kit (NFR 5).** Render's free 512 MB plan cannot hold the AST model.
  `deploy/huggingface/` builds a Docker image for a free Hugging Face CPU Space (16 GB);
  `scripts/deploy_hf_space.sh` publishes the committed tree with the owner's token;
  `tools/uptime_probe.py` records `/api/health/ready` and reports uptime against 99 %.
- **Screenshot tool.** `tools/capture_screenshots.py` deleted the whole `screenshots/`
  folder before capturing, which removed the Teachable Machine training evidence twice.
  It now clears only its own `screenshots/ui/`.
- **Teachable Machine with more samples.** Built the import package from all 210 training
  recordings per class (2,100 samples). TM imported every class and then stayed at
  "Preparing training data" with the browser idle: 11 minutes on the Intel GPU that headless
  Chrome picks by default, and again after `--nvidia-offload` moved WebGL to the NVIDIA card
  (confirmed through the WebGL renderer string). Screenshots
  `screenshots/gtm/20260927_v4_2100*` show both stalls. The limit is in TM, not the graphics.
  1,750 samples (175 per class) stalled the same way (`20260927_v5_1750*`), so about 1,400
  is the most this TM build trains here.
- **Python candidates, decided on validation.** The augmented copies were embedded with AST
  (4,088 of 4,200; the rest are refused by the preprocessor as the app would refuse them).

  | Candidate (validation, 450 recordings) | Chosen head | Accuracy | Macro-F1 | Critical recall | Lowest critical class | Score |
  |---|---|---:|---:|---:|---|---:|
  | AST, training recordings (served) | logreg C=0.01 | 0.873 | 0.872 | 0.898 | Aggression 0.733 | 0.8851 |
  | AST, + augmented copies | MLP α=0.1 | 0.873 | 0.872 | 0.898 | Aggression 0.689 | 0.8847 |

  Neither reaches 0.85 on every critical class, and the scores differ by 0.0004, under the
  0.005 margin, so the served model stays. No new test scoring was done. Aggression is the
  weakest class on validation as well as on test; its training data mixes real aggression
  with door slams (`BASELINE_AUDIT.md` §4), which more classifier tuning does not fix.
  `transfer_selection_ast_current.json` was regenerated to add the per-class numbers; its
  choice is unchanged.
- **Teachable Machine: more epochs, same 1,400 samples.** `train_gtm_browser.py --epochs`
  sets TM's Advanced → Epochs (the tour card covering the Training panel is hidden first).
  Same import package as the served model (rebuilt byte-for-byte; only the fetch date in
  the evidence rows differed, so the committed rows were kept).

  | TM export (validation, 450 recordings) | Epochs | Accuracy | Macro-F1 | Critical recall | Score |
  |---|---:|---:|---:|---:|---:|
  | served until today (`archive/v2_2026-09-26`) | 50 | 0.504 | 0.490 | 0.613 | 0.552 |
  | `candidates/tm_v6_140_e100` | 100 | 0.522 | 0.504 | 0.667 | 0.585 |
  | `candidates/tm_v6_140_e200` | 200 | 0.536 | 0.522 | 0.667 | 0.595 |

  The 200-epoch export won on validation and now serves (`tm-20260926T2241-ew`). Scored
  once on test: **0.511 accuracy, 0.492 macro-F1, 0.600 critical recall** (was 0.493 /
  0.473 / 0.591). Part of a gap this size can be run-to-run variation (TM picks its own
  random start and internal split); one run per setting was all the time allowed. The
  search stopped at 200 epochs because the gain had shrunk to 0.01 and no setting comes
  near the 0.85 target. `init_db.py` registered the new version as active.
- **Browser acceptance (FR ii, iv, v, vi, vii, ix, li, liv, lxiv, lxxix).** New
  `tools/browser_acceptance.py` starts its own copy of the app on a scratch database and drives
  system Chrome, with a fake microphone playing `sample_audio/person_asking_for_help.wav` for
  the live page: all six accounts through the login form against the permission matrix, one
  upload per format, a mixed batch, the player by keyboard, the "Manual Review Required" label,
  consent, all five microphone states and the live alert after three windows.
  `reports/browser_acceptance.json`, `screenshots/acceptance/`. It found one wrong message: an
  unsupported file was told to send "WebM" audio, which the SRS does not list, and not M4A,
  which it does; fixed.
- **NFR 2, measured** (`tools/benchmark_scale.py` → `reports/scale.json`): 20,000 synthetic
  events in a scratch SQLite database, served by gunicorn with **one worker and 8 threads**,
  real HTTP from signed-in users. Pass line written in the script before the first run: p95
  ≤ 1 s for every page and API read at 10 users, no errors, largest allowed export ≤ 10 s.

  | Users at once | p95, all reads | Slowest endpoint (p95) | Errors |
  |---:|---:|---|---:|
  | 1 | 0.065 s | events filtered by class 0.189 s | 0 |
  | 10 | 0.372 s | analytics page 0.674 s | 0 |
  | 20 | 0.597 s | analytics page 0.756 s | 0 |

  Export: all 20,000 rows is refused in 0.02 s (the app caps exports at 10,000 rows and says
  so); the largest allowed export, 9,334 rows, took 1.61 s. 30-second uploads: median 4.79 s
  one at a time; four at once, median 17.5 s and p95 20.5 s — CPU-bound on this laptop, so
  simultaneous long uploads remain the real limit (NFR 1 holds for one at a time).

  How it got there, in order:
  1. The first runs used 2 workers × 4 threads, the Render kit's setting (10 users: free-text
     search p95 0.67 s, once 1.04 s). Then the live page's streak counter (`RepeatTracker`)
     turned out to live in worker memory: with two workers, one session's windows can land on
     different processes and "3 in a row" may never be seen. The README, the Render kit and
     the benchmark now use one worker with threads (the Space image already did).
  2. One worker with 8 PyTorch threads failed the line (free-text search 1.35 s at 10 users)
     and slowed a 30 s upload to 8.15 s: 8 threads oversubscribe the 4 physical cores.
     `SST_TORCH_THREADS` is now the physical core count.
  3. Free-text search used ILIKE, which SQLite runs as `lower(x) LIKE lower(?)` on six columns
     of every row. SQLite's LIKE is already case-insensitive for ASCII and its `lower()` folds
     nothing else, so plain LIKE returns the same rows: 18.5 → 10.3 ms per scan of 20,000
     events. A test now checks that search ignores case.

  Two harness faults were also fixed: it signed in again for every load level and tripped the
  app's own login rate limit (20 per minute per address), and it expected an all-rows export
  to succeed.
- **Interface redesign.** A public product page now sits at `/` for signed-out visitors
  (signed-in users still go to their home page): a sky-blue hero with tilted product cards, a
  stats band, the console on desktop and phone, how it works, the ten classes with a waveform
  each, the two models' held-out results against the SRS targets (read from the metrics files at
  request time), and who uses it. The console itself moved to the Geist typeface (self-hosted,
  OFL, because the CSP allows no font CDN), a light default theme with an ink sidebar and a
  rebuilt dark theme, a drawer menu below 1025 px in place of the sideways-scrolling strip, and
  tables that turn into stacked cards on phones (column names copied onto the cells by
  `core.js`). Chips show a colour dot next to the word instead of glyph strings, and no longer
  truncate class names. Sign-in and sign-up are a full-screen split: the same sky on the left,
  the form on the right. Every `id` and `data-*` hook the scripts and tests use is unchanged:
  `pytest` 497 passed, `tools/check_ui.py` passed at 320/390/768/1440 px with no overflow,
  `tools/browser_acceptance.py` 35/35.

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