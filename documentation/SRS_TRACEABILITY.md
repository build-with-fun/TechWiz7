# SRS traceability matrix

One row per requirement in SonicSentinel AI SRS v1.0, §1.6 (FR i–lxxx), §1.7 (NFR 1–5),
§1.8 (integrity) and §1.10 (deliverables). Last checked **26 Sep 2026** against the
working tree; update a row whenever its evidence changes.

**Status**: *verified* = an automated test or a recorded measurement shows it working;
*partial* = implemented but some part is missing or unmeasured; *missing* = not built;
*blocked* = needs something only a person can supply (recordings, accounts, a decision).
**Owner** is left for the team to fill in. Code comments name people who do not match the
Git history, so ownership must be confirmed by the team, not copied from those comments.
**Priority**: P0 = core flow a judge tries first, P1 = explicitly required, P2 = polish.

Test files are under `tests/`; `pytest -q` runs them all.

## Functional requirements (§1.6)

| FR | Requirement | Where | Evidence | Status | Owner | Pri | Next action |
|---|---|---|---|---|---|---|---|
| i | Registration and secure login | `src/api/pages.py`, `src/auth.py`, `templates/auth/` | `test_account_pages.py::test_register_and_update_own_profile`, `test_csrf.py`, lockout tests in `test_database.py` | verified | | P0 | Self-registration creates a normal user; privileged roles are granted by an administrator (deliberate: nobody can register as admin). |
| ii | Five roles, server-enforced | `ROLE_CAPABILITIES` in `src/auth.py` | `test_database.py::test_every_srs_role_is_accepted`, export 403 in `test_pipeline_http_integration.py` | verified | | P0 | Browser walk of every role still to record on video. |
| iii | Profile and unique User ID | `pages.py` profile routes, `User.id` | `test_register_and_update_own_profile` | verified | | P1 | |
| iv | Upload WAV/MP3/FLAC/OGG/M4A | `src/api/audio_api.py`, `audio_preprocessing/io.py` (FFmpeg) | `test_audio_download_and_history.py::test_every_accepted_audio_format_has_a_mime_type`, conversion round-trip in `test_audio_preprocessing.py`, MP3 path in `tools/robustness_probe.py` | partial | | P0 | Record an upload of each format through the UI. |
| v | Batch upload | `templates/upload.html` (multiple), `static/js/upload.js` | per-file server calls; no automated browser test | partial | | P1 | Browser test with a mixed good/bad batch. |
| vi | Start/stop live monitoring after permission | `static/js/live.js`, `src/api/live_api.py` | `test_dashboard_reports_monitoring.py::test_live_window_payload_carries_consistency_and_top3`, consent refusal tests | partial | | P0 | Real microphone run on the demo laptop; the capture loop was rewritten on 26 Sep. |
| vii | Mic states: Available, Active, Paused, Disconnected, Permission denied | `live.js` `setMic()`, `templates/live.html` | code review; no browser test yet | partial | | P1 | Test with Chrome's fake-device flags and by unplugging a USB mic. |
| viii | Validate format, size, duration, rate, channels, integrity, sound | `audio_preprocessing/io.py`, `quality.py`, `audio_api.py` | `test_audio_preprocessing.py` (silence, unknown format, short), `test_failed_upload_is_audited_for_anomaly_detection` | verified | | P0 | |
| ix | Play, pause, replay, seek, volume | `templates/event_detail.html`, `static/js/event.js` | replay added 26 Sep; manual check only | partial | | P1 | Keyboard walk of the player. |
| x | Metadata incl. bit depth, size, upload time | `AudioFile` (`bit_depth` added 26 Sep), `persistence.py` | `test_upload_stores_source_rate_and_bit_depth` | verified | | P1 | |
| xi | Resample, mono, normalise, trim, denoise, segment, pad, truncate | `audio_preprocessing/pipeline.py`, `transforms.py` | `test_audio_preprocessing.py` (40+ tests) | verified | | P0 | |
| xii | Silence detection | `quality.py` | `test_pipeline_rejects_a_silent_file_with_a_reason` | verified | | P0 | |
| xiii | Clipping warning | `quality.py` | quality tests; `sample_audio/clipped_loud_tone.wav` | verified | | P1 | |
| xiv | Background-noise estimate | `quality.py` (SNR estimate) | quality tests | verified | | P1 | Estimate not validated against labelled SNR recordings. |
| xv | Fixed segments with timestamps | `transforms.py` segment bounds | `test_segment_timestamps_are_seconds_in_fr_xv_form` | verified | | P1 | |
| xvi | Common ten-class dataset | `audio_dataset/manifest.csv` | `test_frozen_split.py::test_three_thousand_originals` | verified | | P0 | Label quality issues in `BASELINE_AUDIT.md` §4. |
| xvii | Dataset metadata fields | manifest (42 columns), `DATA_DICTIONARY.md` | `verify_dataset.py --strict` | verified | | P1 | 2,123 rows have `recording_environment=unspecified`. |
| xviii | Balance / insufficiency check | `verify_dataset.py`, `build_split.py --strict` | `test_every_class_contributes_210_45_45` | verified | | P1 | |
| xix | Controlled augmentation | `augmentation/transforms.py`, `augment_dataset.py` | `test_augmentation.py` (SNR accuracy, determinism, train-only lineage) | verified | | P1 | Decide from validation whether augmented copies improve the served model. |
| xx | MFCC, mel, chroma, ZCR, RMS, centroid, bandwidth, roll-off | `feature_extraction/features.py` (254 columns) | `test_feature_extraction.py` | verified | | P0 | The served model uses CNN14 embeddings; the hand-crafted set feeds the HGB baseline. |
| xxi | Waveform display | `src/services/visuals.py`, `event.js`, report PNG | `test_event_report_contains_every_fr_lxix_item` | verified | | P1 | |
| xxii | Spectrogram display | same | same | verified | | P1 | |
| xxiii | Train and compare ≥3 models | `train_transfer.py`, `train_classical.py`, `train_deep.py` | `python_models/metrics/transfer_test_current.json`, `classical_metrics_hgb_split_v2.json`, `documentation/MODEL_EVALUATION.md` | verified | | P0 | CRNN numbers are from the superseded v1 split. |
| xxiv | Systematic tuning | validation grids in `train_transfer.py`, `tuning.py` | `transfer_selection_current.json`, `test_tuning_protocol_labels.py` | verified | | P1 | |
| xxv | Python model predicts one of ten classes | `src/inference/predictor.py`, `python_models/best/` | `test_transfer_model.py::test_predictor_returns_plain_strings_and_all_ten_scores` | verified | | P0 | |
| xxvi | Python scores for all classes | same | same | verified | | P0 | |
| xxvii | Separately trained TM audio model | `gtm_model/`, `tools/train_gtm_browser.py` | TM export + `screenshots/gtm/` | partial | | P0 | Team must repeat the training in their own browser and keep the project link and class screenshots (deliverable 5). |
| xxviii | TM integration by a supported method | TF.js export converted to Keras; `src/inference/gtm_predictor.py` | `test_pipeline_http_integration.py` (real export), `gtm_model/gtm_metrics.json` | verified | | P0 | Browser/server frontend parity not measured (`frontend_verified=false`). |
| xxix | TM classifies the same segment independently | `gtm_predictor.py` | `test_model_independence.py` (9 tests) | verified | | P0 | |
| xxx | TM scores for all classes | same | comparison CSV | verified | | P0 | |
| xxxi | Compare predicted classes | `src/inference/consistency.py` | `test_consistency_taxonomy.py` | verified | | P0 | |
| xxxii | Absolute top-confidence difference | same | same; `reports/model_comparison.csv` | verified | | P0 | |
| xxxiii | Acceptable/Weak Match, Disagreement, Uncertain | same (plus Strong Match) | `test_consistency_taxonomy.py` | verified | | P0 | |
| xxxiv | Top-3 per model | event detail, live cards (26 Sep), API | `test_live_window_payload_carries_consistency_and_top3` | verified | | P1 | |
| xxxv | Admin-configurable confidence threshold | `config/thresholds.json`, admin config API | `test_taxonomy_responds_to_changed_thresholds` | verified | | P1 | |
| xxxvi | Admin-configurable top-two margin | same | `test_top_two_margin_reported_for_both_models` | verified | | P1 | |
| xxxvii | Quality: Good/Acceptable/Poor/Unusable | `quality.py` | `test_all_four_quality_verdicts_are_accepted` | verified | | P1 | |
| xxxviii | Uncertain on low confidence, similar tops, poor quality, disagreement, overlap | `consistency.py`, `evaluate_review` | taxonomy and review tests | verified | | P0 | |
| xxxix | Overlap detection | `consistency.py` secondary detection | `test_overlap_detected_when_runner_up_is_strong` | verified | | P1 | Heuristic; overlap probe in `reports/ROBUSTNESS.md`. |
| xl | Consecutive-window confirmation | `RepeatTracker` in `pipeline.py` | rule tests in `test_alert_rules_config.py` | verified | | P0 | |
| xli–xlix | Per-class behaviour (machinery, glass, alarm, horn, animal, gunshot, scream, aggression, help) | `alert_rules/alert_rules.json`, `severity_block` | `test_alert_rules_config.py` | verified | | P0 | Detection quality per class: see NFR 4. |
| l | Background noise non-critical unless loud | `alert_rules.json` ambient limit | `test_background_noise_carries_a_configurable_ambient_limit` | verified | | P1 | |
| li | Unknown / manual review | review conditions | `reports/ROBUSTNESS.md` part B (sounds outside the ten classes) | partial | | P1 | No explicit "Unknown" class; out-of-set sounds rely on the review route. |
| lii | Severity levels | `config` severity scale | severity tests | verified | | P1 | |
| liii | Configurable alert rules | `alert_rules/`, admin config API with validation | `test_alert_rules_config.py` (12 tests) | verified | | P0 | |
| liv | Visible real-time alerts | dashboard, alerts page, live banner (26 Sep) | alert page tests | partial | | P0 | Live banner needs a recorded browser run. |
| lv | Acknowledge, dismiss, escalate | `src/api/alerts_api.py` | `test_alert_review_pages.py`, `test_an_acknowledged_alert_records_who_and_when` | verified | | P0 | |
| lvi | Alert history | alerts + audit tables | same | verified | | P1 | |
| lvii | Manual-review queue | `reviews_api.py`, `templates/reviews.html` | review tests | verified | | P0 | |
| lviii | Reviewer playback | event detail player | download/playback tests | verified | | P1 | |
| lix–lxi | Confirm/correct, comments, override keeping model output | `reviews_api.py`, `Review.original_*` | `test_an_override_preserves_the_original_model_output` | verified | | P0 | |
| lxii | Event statuses | `Event.status` | database tests | verified | | P1 | |
| lxiii | User dashboard | `dashboard()` in `pages.py` (26 Sep) | `test_normal_user_dashboard_shows_only_their_own_uploads` | verified | | P0 | |
| lxiv | Live dashboard items | `templates/live.html` | live payload test | partial | | P0 | Browser run with a microphone. |
| lxv | Administrator dashboard | admin block in dashboard | `test_admin_dashboard_has_srs_metrics_and_anomalies` | verified | | P1 | |
| lxvi | Chronological timeline | critical-event timeline (26 Sep), `/api/dashboard/timeline` | dashboard test | verified | | P1 | |
| lxvii | Nine search filters | `src/services/search.py` | `test_severity_filter_narrows_the_list`, search tests | verified | | P1 | |
| lxviii | Analytics incl. FP, FN, alert response | `analytics()` (26 Sep) | `test_analytics_page_reports_false_positive_and_alert_response_sections` | verified | | P1 | FP/FN come from reviewer decisions only. |
| lxix | Downloadable report with every listed item | `reports_api.event_report` (26 Sep) | `test_event_report_contains_every_fr_lxix_item` | verified | | P1 | |
| lxx | Admin CSV/Excel export | `reports_api.py` | export tests (403 for non-admin) | verified | | P1 | |
| lxxi | Secure audio storage | `StorageLayout`, guarded download routes | download tests | verified | | P1 | Files are not encrypted at rest. |
| lxxii | Database contents | `src/models.py` (10 tables) | `test_database.py` | verified | | P0 | |
| lxxiii | Exact duplicate by hash | SHA-256 unique column, 409 | `test_allow_duplicate_stores_a_second_event` | verified | | P1 | |
| lxxiv | Near-duplicate attempt | two-stage check (26 Sep) | `test_two_stage_near_duplicate_flags_a_quieter_copy_but_not_a_different_sound`, `reports/near_duplicates.json` | verified | | P1 | Trimmed copies found only ~22 % of the time. |
| lxxv | Model version per prediction | `ModelVersion`, event foreign keys | `test_init_db_register_model_then_activate` | verified | | P1 | Re-seed after changing a bundle. |
| lxxvi | Audit trail | `AuditRecord`, `record_audit` | audit tests | verified | | P1 | |
| lxxvii | Understandable errors | `src/errors.py` | error-envelope tests | verified | | P1 | |
| lxxviii | Six anomaly alerts | `src/services/monitoring.py` (26 Sep), dashboard banner | `test_admin_dashboard_has_srs_metrics_and_anomalies` | verified | | P1 | In-app only; no email or SMS. |
| lxxix | Privacy: visible mic state, no secret recording | consent gate, Active pill and tab dot (26 Sep) | consent tests | partial | | P0 | Browser check. |
| lxxx | Configurable retention | retention config, preview and purge | `test_retention_api.py` | verified | | P1 | |
| — | Responsive UI | `static/css/`, `tools/check_ui.py` | `reports/ui_review.json` (52 route/width checks) | verified | | P0 | Firefox, Edge, Opera not tested. |

## Non-functional requirements (§1.7)

| NFR | Target | Measured | Status | Next action |
|---|---|---|---|---|
| 1 Performance | ≤8 s for a 30 s upload; ≤3 s per live window | `reports/performance.json`: 30 s upload median 7.15 s (p95 7.47), 2 s live window median 2.06 s (p95 2.30), both against budget | **met** | Concurrency is single-process only: 4 clients degrade to 25.7 s median, the honest scaling limit. |
| 2 Scale | 20,000 events, several users | `test_twenty_thousand_events_insert_and_the_search_filter_stays_fast` | partial | Concurrency not measured. |
| 3 Usability | intuitive UI for five roles | UI checks at four widths | partial | Record a first-time user walkthrough. |
| 4 Accuracy | ≥85 % accuracy, ≥0.80 macro F1, ≥85 % recall per critical class, both models | Python (AST+logreg, v2 test): **0.891 / 0.892**; critical recall Help 1.00, Gunshot 0.96, Glass 0.91, Aggression 0.84, Panic 0.82; mean critical 0.907. TM: 0.493 accuracy — see `gtm_model/gtm_metrics.json` and §8.2 of `PROJECT_REPORT.md` | partial | Python **met** all three targets. TM is far below 0.85; it can only be retrained in a member's own browser session, and the app routes disagreements to human review as designed. Aggression 0.84 and Panic 0.82 are individually below 0.85 even though the mean clears it. |
| 5 Availability | 99 % uptime | no deployment yet | blocked | Needs a hosted deployment and a monitoring period. |

## Integrity (§1.8) and deliverables (§1.10)

| Item | Status | Evidence / blocker |
|---|---|---|
| Members explain their modules | blocked | Team must fill `TEAM_CONTRIBUTION_RECORD.md` and `documentation/VIVA_PACK.md`. |
| Meaningful commits on all five days, by all members | blocked | History has one author on 24–25 Sep. Must not be backdated or faked. |
| Development log | verified | `documentation/devlog.md` |
| Surprise modification readiness | verified | quick edits in `documentation/VIVA_PACK.md` |
| No hard-coded predictions or hidden APIs | verified | `test_model_independence.py`; no network calls at inference |
| Tested on unseen recordings | verified | frozen v2 test split, scored once per model family |
| AI_USAGE.md | partial | updated 26 Sep; team must add their own verification names |
| Project report with diagrams | partial | `PROJECT_REPORT.md`, `diagrams/` |
| Dataset deliverable | blocked | audio is not in Git; the team must choose a licensed distribution (36 clips need permission review) |
| Python model evidence | verified | `python_models/metrics/`, `documentation/MODEL_EVALUATION.md` |
| TM evidence (project link, class screenshots) | blocked | automated run screenshots exist; the project link must come from a team member's own TM session |
| Comparison report (≥100 unseen, ≥10/class) | verified | `reports/model_comparison.csv` (all 450 test recordings) |
| Alert rule files | verified | `alert_rules/alert_rules.json` |
| Test cases incl. the 21 listed kinds | verified | `tests/`, `TEST_PLAN.md` |
| Installation/execution instructions | verified | `README.md` |
| Public GitHub, deployment URL, MP4 video, blog link | blocked | need the team's accounts and recording |
| LICENSE | blocked | team decision |
