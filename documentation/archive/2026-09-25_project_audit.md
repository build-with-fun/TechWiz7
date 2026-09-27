> Archived: the first audit, kept as a record of the starting point. It is not the current state.

# SonicSentinel AI: project audit

Audited on 2026-09-25 against the repository at `06cf07d` and the supplied **SonicSentinel AI SRS v1.0**. This is a discovery record and implementation plan. No application code or data was changed during this phase.

The statuses below describe that **pre-overhaul baseline**, not the final working tree. See `documentation/BASELINE_AUDIT.md`, `documentation/devlog.md` and `TEST_PLAN.md` for what was fixed afterwards and the current evidence.

## Product and users

SonicSentinel is intended to help a person monitoring a site recognize ten classes of sound in an uploaded recording or a consented microphone stream. A local Python classifier and a separately trained Google Teachable Machine (GTM) audio classifier should each produce a class distribution. The application should compare them, assess audio quality, apply configurable alert rules, keep the evidence, and route uncertain cases to a reviewer. It is a competition prototype, not a certified emergency response service.

The intended users are a normal user submitting recordings, an audio reviewer deciding uncertain cases, a security operator handling alerts, a maintenance operator changing technical configuration, and an administrator managing accounts and retention. The principal journey is **sign in, capture or upload, validate and classify with both models, inspect the evidence, acknowledge an alert or review an uncertain result, then search or export history**. The current checkout cannot complete that journey: the GTM export is absent, so the analysis pipeline does not load; the upload and live integration paths also have independent defects detailed below.

## What is actually in the repository

The application is Python 3.12, Flask 3.1, Jinja, plain CSS/JavaScript, SQLAlchemy with SQLite, librosa/SoundFile/FFmpeg, scikit-learn/XGBoost/PyTorch, and pytest. `src/app.py` assembles the application. `src/api/` contains page and JSON routes. `audio_preprocessing/`, `feature_extraction/`, `src/inference/`, and `src/services/pipeline.py` form the analysis path. `src/models.py` and `src/services/persistence.py` define and write the relational data. JSON in `config/` and `alert_rules/` drives thresholds and decisions. `python_models/` contains training code, saved model bundles, and metrics. `audio_dataset/` contains manifest and split tooling. `templates/`, `static/`, and `screenshots/` form the web interface and its current visual evidence.

The dependency path to preserve while repairing the app is:

```text
browser upload / microphone
  → Flask route + session/role check
  → audio validation and preprocessing → quality + timestamped segments
  → 254-feature extraction → saved Python model ┐
  → GTM-specific audio frontend → GTM model      ├→ comparison → rules/review
                                                 ┘→ persistence → event/alert/review
                                                   → dashboards, reports and playback
```

This is a sensible separation in principle. Its two most important seams, route-to-pipeline persistence and browser-to-live-window encoding, are currently broken.

The local manifest contains **3,000 rows, 300 original clips per class, with a 2,100/450/450 train/validation/test split**. All referenced original files exist in this workspace. Of these, 375 are marked synthetic in `audio_provenance`; the others have a blank value there and carry source and licence fields elsewhere. The verifier's quick strict run passed 26/26 checks, but intentionally skipped SHA-256 integrity and content uniqueness. The audio binaries, sample WAV files, GTM training ZIPs, and GTM segments are ignored by Git, so a clean clone does **not** contain the working local corpus or the advertised sample audio. `audio_dataset/manifests/corpus_statistics.json` currently reports 5,230 augmented rows and zero originals while naming `manifest.csv`, which has 3,000 originals; that statistics file cannot support the claims made for it.

There are saved Python bundles. The default `python_models/best/` bundle is XGBoost; a newer HGB trial is under `python_models/best/hgb/`, and a CRNN trial is under `python_models/best/deep/`. The recorded held-out results are below the SRS targets:

| Evidence file | Test accuracy | Macro F1 | Critical-class recall |
|---|---:|---:|---:|
| `python_models/metrics/classical_metrics.json` (default bundle) | 0.6934 | 0.6932 | 0.7988 |
| `python_models/metrics/classical_metrics_hgb.json` | 0.6978 | 0.6968 | 0.8000 |
| `python_models/metrics/deep_metrics_deep.json` | 0.6000 | 0.5964 | 0.6667 |

The GTM training package and adapter exist, but `gtm_model/` has no exported `metadata.json`, network model, converted Keras model, verified frontend configuration, or GTM test metrics. The app's own loader reports this and disables analysis. A truthful dual-model demonstration depends on training/exporting that separate model and validating the exact audio frontend; a substitute that copies the Python prediction would violate the SRS.

## SRS functional requirement trace

**Status** means the current checkout as a complete product: *complete* has an implemented, inspectable path; *partial* has code or evidence but an incomplete user journey or contract; *missing* has no working implementation. A configured rule alone is not counted as a working detection. Roman numerals follow SRS §1.6 exactly.

| FR | Requirement | Status | Evidence and required work |
|---|---|---|---|
| i | Registration and authentication | Partial | `src/auth.py`, `src/api/auth_api.py`, `src/api/pages.py` implement login; no self-registration route/page. Add safe registration or document an SRS-compliant role assignment flow. |
| ii | Five-role access control | Partial | Capability matrix in `src/auth.py` and route guards exist; exports grant reviewer/operator access although FR lxx says administrator only. Reconcile every route with the SRS. |
| iii | User profile management and unique ID | Partial | `User.id` exists in `src/models.py`; there is no self-service profile update page or endpoint. |
| iv | WAV/MP3/FLAC/OGG/M4A upload | Partial | `src/api/audio_api.py` and `audio_preprocessing/io.py` accept these types; template form/API contract mismatch and missing GTM prevent a real completed upload. |
| v | Authorized batch upload | Partial | `static/js/upload.js` loops dropped files, but the file input lacks `multiple`; there is no reliable batch summary or isolated server-side completion. |
| vi | Start/stop microphone monitoring | Partial | `src/api/live_api.py` has sessions; `templates/live.html` hides the consent panel and `static/js/live.js` corrupts encoded windows. GTM is absent. |
| vii | Microphone status display | Partial | The JS shows some states but does not cover paused and disconnected consistently. Exercise permission and device-loss cases. |
| viii | File validation | Partial | Format, size, duration, decoding and quality checks exist in `src/api/audio_api.py` and `audio_preprocessing/`; wire them through the real upload journey and test each supported format. |
| ix | Audio preview controls | Partial | `templates/event_detail.html` and `static/js/event.js` have playback controls; new events cannot be reached through a working upload. |
| x | Audio metadata | Partial | `AudioFile` stores most fields; bit depth is extracted for WAV in `audio_preprocessing/io.py` but is not stored/displayed consistently. |
| xi | Resample, mono, normalize, trim, reduce noise, segment, pad/truncate | Complete | Implemented in `audio_preprocessing/` and covered by `tests/test_audio_preprocessing.py`; verify the final app uses the same configuration. |
| xii | Silence detection | Complete | `audio_preprocessing/quality.py` rejects unusable silence; unit coverage exists. |
| xiii | Clipping detection | Complete | `audio_preprocessing/quality.py` measures clipping and returns a quality warning. |
| xiv | Background noise estimation | Complete | Quality module estimates a noise measure; accuracy of that estimate on field audio still needs study. |
| xv | Timestamped fixed segments | Partial | Preprocessor produces segment spans; persisted event/live evidence does not reliably retain and expose them. |
| xvi | Common ten-class dataset | Partial | Local 3,000-original manifest is balanced; source audio is ignored by Git and licence attribution output is missing. Package a permitted, reproducible submission. |
| xvii | Dataset metadata | Partial | The manifest has IDs, labels, source, duration and split; generated statistics contradict it. Rebuild and validate statistics/attribution. |
| xviii | Balance checking | Complete | `audio_dataset/scripts/verify_dataset.py` checks counts and split distribution. |
| xix | Controlled augmentation | Partial | `data/generate_corpus.py` and `audio_dataset/scripts/generate_help_clips.py` create synthetic variation; a distinct train-only augmentation workflow/evidence for the listed transforms is not complete. |
| xx | Required acoustic features | Complete | `feature_extraction/features.py` implements the listed families with a locked 254-column configuration. |
| xxi | Waveform | Partial | `src/services/visuals.py` and event canvas exist; integrate with actually persisted upload/live events and verify rendering. |
| xxii | Spectrogram | Partial | Same visual path exists; the module claims it shows GTM input although it computes a separate visualization frontend. Correct that claim and verify rendering. |
| xxiii | Compare at least three Python models | Complete | Classical/deep training code and comparison metric files document multiple candidates. |
| xxiv | Hyperparameter tuning | Complete | `python_models/tuning.py` and metric selection records show validation-based trials. |
| xxv | Python ten-class classification | Partial | Saved model and `src/inference/predictor.py` exist; app analysis is disabled by the absent GTM export. |
| xxvi | Python scores for every class | Partial | Predictor contract supports them; no real dual-model app result is reachable. |
| xxvii | Separately trained GTM audio model | Missing | Training samples and browser instructions exist; no trained/exported GTM model or training evidence is present. |
| xxviii | GTM integration | Missing | `src/inference/gtm_predictor.py` is an adapter awaiting the export and verified frontend. |
| xxix | Independent GTM classification of the same segment | Missing | Interface design and tests check independence structurally; no runtime GTM prediction can occur. |
| xxx | GTM scores for all classes | Missing | No model or metrics from which to produce actual scores. |
| xxxi | Compare predicted categories | Partial | `src/inference/consistency.py` implements comparison for injected predictions; real GTM path is absent. |
| xxxii | Absolute top-confidence difference | Partial | Comparison code exists; requires real paired outputs. |
| xxxiii | Consistency status taxonomy | Partial | Five-status taxonomy includes the four SRS statuses; runtime proof awaits GTM. |
| xxxiv | Top-three predictions per model | Partial | Scores are available in the response shape and detail template; verify visible top-three display with real results. |
| xxxv | Admin confidence threshold | Partial | `config/thresholds.json` and config API are editable; verify a live edit actually changes an analysis after pipeline repair. |
| xxxvi | Admin top-two margin threshold | Partial | Same config path; validate a changed threshold end to end. |
| xxxvii | Four audio-quality verdicts | Complete | `audio_preprocessing/quality.py` returns Good/Acceptable/Poor/Unusable. |
| xxxviii | Mark uncertainty from all specified causes | Partial | `src/inference/consistency.py` and review conditions cover several causes; exercise each cause through the final event path. |
| xxxix | Possible overlap detection | Partial | Comparison heuristic exists; needs real mixed-sound cases and visible evidence. |
| xl | Consecutive confirmation | Partial | `RepeatTracker` implements a global streak; per-class rule counts are not used and the live path is broken. |
| xli | Machinery fault and maintenance action | Partial | Rule exists in `alert_rules/alert_rules.json`; the rule engine does not apply all declared escalation behavior and no live alert can be raised. |
| xlii | Glass-breaking high-severity alert | Partial | Rule exists; end-to-end alert path unavailable. |
| xliii | Alarm/siren attention alert | Partial | Rule exists; end-to-end alert path unavailable. |
| xliv | Vehicle-horn environmental event | Partial | Class/rule exist; current pipeline may raise an alert for any eligible class after a streak. |
| xlv | Animal sound as noncritical/contextual | Partial | Class/rule exist; alert eligibility needs an explicit class gate. |
| xlvi | Gunshot confirmation and confidence | Partial | Gunshot thresholds exist, but `RepeatTracker` uses its global count rather than the class rule. |
| xlvii | Panic-scream critical alert | Partial | Rule exists; end-to-end verification missing. |
| xlviii | Aggression severity based on confidence and repetition | Partial | Escalation is declared in JSON but `severity_block()` reads only the base severity. |
| xlix | Defined help phrases and critical alert | Partial | Synthetic help samples and rule exist; speech diversity and live alert behavior need verification. |
| l | Background noise limit | Partial | A threshold is declared in JSON; pipeline alert decision does not evaluate its `suppress_when`/noise escalation fields. |
| li | Unknown/manual review outcome | Partial | Review conditions and uncertain status exist; no actual full-path ambiguous upload. |
| lii | Required severity levels | Partial | Config supports a five-level and mapped four-level scale; exercise the SRS four-level presentation and rule output together. |
| liii | Configurable alert rules | Partial | JSON loader validates base rules; several declared escalation/suppression fields are unused by the decision path. |
| liv | Real-time dashboard alert | Partial | Alert models/pages exist; GTM, live encoding and persistence gaps prevent a genuine new alert. |
| lv | Acknowledge/dismiss/escalate | Partial | `src/api/alerts_api.py` has guarded transitions and forms; no integrated new alert flow to validate them. |
| lvi | Alert history | Partial | Alert and audit tables/routes exist; depends on actual alert persistence. |
| lvii | Manual review queue | Partial | Review table/route/page exist; cannot receive genuine pipeline results yet. |
| lviii | Reviewer audio playback | Partial | Event detail playback exists; linked live/upload audio storage needs repair. |
| lix | Reviewer class decision | Partial | `src/api/reviews_api.py` handles decisions; complete journey unverified. |
| lx | Reviewer comments/actions | Partial | Review form/API have comments/action fields; verify meaningful validation and display. |
| lxi | Preserve model outputs on override | Partial | Schema and service support snapshots; test with actual persisted dual-model event. |
| lxii | Event lifecycle statuses | Partial | Schema declares statuses; creation/update path is broken at upload persistence. |
| lxiii | User dashboard | Partial | `templates/dashboard.html` and summary API exist; no reliable new event data and layout classes lack CSS. |
| lxiv | Live dashboard | Partial | Page/JS exist, but consent, encoding, history and model readiness prevent the journey. |
| lxv | Administrator metrics dashboard | Partial | Aggregate API exists; some requested values/trends are absent from the visible dashboard. |
| lxvi | Chronological event timeline | Partial | Timeline API and event list exist; verify visual chronology and filters with seeded events. |
| lxvii | Full search/filter set | Partial | `src/services/search.py` implements many fields; verify all nine SRS filters on a populated database. |
| lxviii | Analytics including false negatives/response | Partial | Several aggregate endpoints exist; false negatives need adjudicated ground truth, and alert response analysis is incomplete. |
| lxix | Downloadable full analysis report | Partial | `src/api/reports_api.py` emits JSON/HTML but omits waveform/spectrogram and some required details. |
| lxx | Admin CSV/Excel export | Partial | Both routes exist, but `export_data` is granted to non-admin roles and filtered exports appear limited by search pagination. |
| lxxi | Access-controlled audio storage | Partial | `StorageLayout` resolves relative paths and download routes are guarded; real route-to-persistence wiring is broken. |
| lxxii | Required database records | Partial | Ten SQLAlchemy tables cover the listed entities; genuine uploads/live analyses do not reliably write them. |
| lxxiii | Exact duplicate hash | Partial | SHA-256 and a unique database constraint exist; API duplicate detection depends on uploads being persisted. |
| lxxiv | Near-duplicate attempt | Partial | Fingerprinting function exists, but no route supplies `near_duplicate_candidates`; actual uploads compare zero candidates. |
| lxxv | Model version tracking | Partial | Version table and saved metadata exist; documented model register/activate API routes are absent. |
| lxxvi | Audit trail | Partial | Many administrative actions are logged; predictions/uploads/live alerts cannot be fully audited until persistence works. |
| lxxvii | Understandable errors | Partial | Shared error envelope exists; UI displays internal model paths and several documents promise non-existent behavior. |
| lxxviii | Monitoring/anomaly alerts | Partial | Health/anomaly APIs exist; requested anomaly categories and operational alerting are not demonstrated. |
| lxxix | Microphone privacy/consent | Partial | Backend requires `consent_ack`; hidden consent UI and broken capture prevent an accessible, dependable flow. |
| lxxx | Configurable retention | Partial | Retention JSON and preview route exist; `retention_purge()` reports a purge while `_retention()` deletes nothing. |

## Nonfunctional and submission requirements

| SRS item | Current evidence and gap |
|---|---|
| ≤8 s for a ≤30 s clip; ≤3 s per live window | Budgets are configured and an acceptance script exists, but `reports/` has no successful real dual-model run. The GTM dependency blocks measurement. |
| 20,000 event records/concurrency | SQLite WAL and indexes are present; no 20,000-record load/concurrency benchmark is present. A global in-memory repeat tracker would also need scrutiny with multiple workers. |
| Accessible desktop/tablet/mobile UI | Some focus and contrast tokens exist; many template class names have no CSS rule, and screenshots show unstyled panels and broken branding. No browser/device or accessibility result is present. |
| Accuracy target: ≥0.85 accuracy, ≥0.80 macro F1, ≥0.85 critical recall for both models | Recorded Python results miss all three targets; no GTM test metrics exist. Do not present the targets as achieved. |
| 99% uptime | Health endpoints exist, but no deployment or observation period supports an uptime figure. |
| Integrity and evidence | `AI_USAGE.md` and a dev log exist. Git commits in this checkout have dates only on 2026-09-24 and 2026-09-25; the SRS asks for meaningful commits across five competition days and from all members. Do not fabricate earlier commits or attribution. |
| Required artifacts | `PROJECT_REPORT.md`, a finished dual-model comparison report, GTM evidence/export, `LICENSE`, an attribution file, deployment URL, and demo video are absent. `reports/` is empty. The dedicated `augmentation/` folder is absent. |

## Critical defects and risks

1. **No operational classification:** `AnalysisPipeline.load()` requires a real GTM export. The checked app boot returned `analysis_ready=false` because `gtm_model/metadata.json` is missing. This blocks every model-dependent SRS journey.
2. **Upload does not call the real persistence callback:** `src/api/audio_api.py` calls `pipeline.analyse_bytes(..., meta={...}, persist=persist)`, but the real `AnalysisPipeline.analyse_bytes()` accepts those through `**meta` and only calls `self.persist`. The callback is therefore metadata, and no event/audio/alert/review row is written. `tests/test_audio_api.py` uses a stub with a different signature and hides this mismatch. Fix the contract and add a real pipeline integration test.
3. **Live browser sends empty/invalid audio:** `static/js/live.js` makes `encodeWav()` return a base64 string, then `toBase64()` wraps that string in `Uint8Array` before encoding again. The server receives no valid WAV. The consent panel starts `hidden`, and no initialization code reveals it. Live sessions can be created through the API but the browser journey is blocked.
4. **Alert rules are not fully enforced:** `src/services/pipeline.py` can set `alert.raised` after a global streak for any eligible class; it does not require a critical/alertable class, and `severity_block()` ignores JSON escalation/suppression clauses. This could turn ordinary horn/noise detections into alerts and miss declared severity changes.
5. **Retention purge is a no-op:** `src/api/admin_api.py::_retention()` computes an event count, reports `purge executed`, and deletes nothing. This contradicts the control shown to administrators.
6. **Near-duplicate detection is disconnected:** `find_near_duplicate()` needs database candidates; no upload or live route passes any, so a nonempty corpus is never searched.
7. **Security policy is incomplete:** There is no CSRF verification for cookie-authenticated state-changing forms/API calls. `config/auth.json` sets `cookie_secure=false` by default, and `src/app.py` falls back to a predictable key if none is configured. Published demo passwords must never be used on a public deployment. `src/auth.py` grants CSV/XLSX export more broadly than the SRS permits.
8. **Presentation/documentation drift:** `templates/upload.html` posts `name="audio"` to a conditional `#` action, whereas the API expects `file`; the button is `type="button"`, so the claimed no-JS fallback cannot submit. It says uploads are stored in `audio_dataset/originals/`, whereas runtime storage is `data/storage/`. `templates/base.html` requests nonexistent `static/img/favicon.svg`. The report route's date parsing uses `_dtm.timezone.utc` even though `_dtm` is the `datetime` class. `documentation/api_contract.md` lists model administration routes absent from the app. `scripts/deploy_render.sh` calls pytest with an uninstalled `--timeout` option and advertises a deployment profile that lacks the GTM model.
9. **Submission reproducibility:** A clean clone lacks the audio files and GTM training ZIPs due to `.gitignore`; the corpus statistics file is wrong for its named manifest; the reported local test accuracy is far below target. These facts need a truthful packaging and evidence plan.

## UI review

The palette and sidebar are a useful starting point, but the current screenshots do not represent a polished operations console. `templates/dashboard.html`, `upload.html`, and `live.html` use classes such as `panel`, `page-title`, `grid--2col`, `duo`, `queue`, `scale-list`, and `tape` that `static/css/app.css` does not define. The result is a partly styled sidebar around default-flow content, poor information hierarchy, crowded copy, and unstyled data sections. The logo icon is broken in the supplied screenshots. On the live page, the UI describes 1–3-second windows while the configured window is 2 seconds, and the consent state is inaccessible. Empty states often explain implementation details instead of guiding the user. The dashboard has no clear first action for a new judge and no confident mobile layout can be inferred from the current CSS. A design pass should start by making the actual journeys work, then implement a small set of shared page, card, metric, chart, form, table and state components, and verify real browser screenshots at mobile/tablet/desktop widths.

## Verification performed for this audit

- `git status --short` was clean before writing this file. The current branch is `master` at `06cf07d`.
- `.venv/bin/python -m pytest -q`: **422 passed, 2 failed, 13 errors**. All failures/errors are in `tests/test_deep_models.py`; its Keras transfer candidate tries to download MobileNet weights from `storage.googleapis.com`, which was unavailable in this environment. This suite does not cover the real browser upload/live path. The failed run was not treated as a pass.
- `.venv/bin/python audio_dataset/scripts/verify_dataset.py --strict --quick`: **26/26 checks passed**, with SHA-256 and content uniqueness deliberately skipped by `--quick`.
- `.venv/bin/python -m compileall -q src audio_preprocessing feature_extraction python_models audio_dataset`: passed.
- A Flask test-client boot with an in-memory database returned HTTP 200 for `/login`, `/api/health`, and `/api/health/ready`; it logged the missing GTM export. `/static/img/favicon.svg` returned HTTP 404. These are smoke observations, not an end-to-end product test.
- No lint, formatting, type-check, browser-test, or package-build configuration is present. No deployment, uptime, throughput, accessibility, or real dual-model latency result was verified.

## Prioritized implementation plan

1. **Restore a truthful runnable analysis path.** Obtain/train/export the separate GTM model from train-only recordings; capture and verify its frontend against browser predictions; record GTM test metrics on the frozen test split. If that export cannot be produced, leave dual-model inference explicitly unavailable and do not fabricate it.
2. **Repair the vertical journey before redesign.** Align `audio_api` with the real pipeline/persistence signature, add a real HTTP upload integration test, repair live WAV encoding and consent visibility, persist live windows/events/alerts, and make the upload form usable without JavaScript. Check the whole path: sample WAV, event row, detail and playback, alert or review, dashboard and report.
3. **Fix decision and privacy boundaries.** Apply per-class alert eligibility, suppression, escalation and repeat counts; supply near-duplicate candidates; implement or accurately remove the retention purge action; add CSRF protection and deployment-safe session settings; limit exports to administrators.
4. **Complete SRS journeys.** Add registration/profile editing, actual batch selection/results, full microphone status, model version administration, full report evidence, missing analytics, and complete search/filter behavior. Use realistic stored events for each role and every failure state.
5. **Polish the product UI.** Consolidate the mismatched template/CSS vocabulary; establish typography, spacing and severity hierarchy; build responsive dashboard/upload/live/detail/review views; fix imagery and asset paths; test keyboard, contrast, reduced motion and three viewport sizes.
6. **Make the submission reproducible and honest.** Rebuild corpus statistics and attribution, choose a licensed audio distribution method, package permitted demo clips, correct README/API/deployment instructions, add report/schema/design/test/demo/interview documents, and keep `AI_USAGE.md` current. Preserve actual model scores and unresolved competition evidence gaps.
7. **Gate the final hand-in.** Run formatting/lint/type checks once configured, unit and integration tests without network downloads, full browser journeys, dataset integrity verification, dual-model comparison on at least 100 held-out clips with ten per class, performance/accessibility checks, and a clean-clone install. Record actual results in `TEST_PLAN.md`.

Major refactoring and data migration should begin only after this audit has been reviewed. The GTM model/export and any evidence of team contributions or competition-day activity require real source material; software changes cannot manufacture them.
