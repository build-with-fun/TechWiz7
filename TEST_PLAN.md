# Test plan and results

`pytest -q` runs everything below except the browser checks. Each result has the date it
was run; anything not run yet says so.

## Latest runs

| Date | Command | Result |
|---|---|---|
| 27 Sep | `pytest -q` (full suite, deep-model tests included) | **497 passed**, 0 failed, 196 s |
| 27 Sep | `tools/browser_acceptance.py` (Chrome, fake microphone, isolated database) | **35/35 checks passed** (`reports/browser_acceptance.json`, `screenshots/acceptance/`) |
| 27 Sep | `tools/benchmark_scale.py` (20,000 events, gunicorn 1 × 8 threads) | pass line met: p95 of reads 0.37 s at 10 users, 0.60 s at 20, no errors (`reports/scale.json`) |
| 27 Sep | `tools/evaluate_gtm.py --split test` (200-epoch TM export) | 0.511 accuracy, 0.492 macro-F1, 0.600 critical recall |
| 27 Sep | `tools/build_comparison_report.py` (both served models, 450 test recordings) | 450 scored, 0 failed (`reports/MODEL_COMPARISON.md`) |
| 26 Sep | `pytest -q -k "not deep_models"` (before the final fixes) | 456 passed, 1 skipped, 27 deselected, 79 s |
| 26 Sep | `tools/render_uml.py --check` | PASS (5 diagrams, no overlaps, no edge through a node) |
| 25 Sep | `pytest -q` including deep-model tests (needs network for MobileNet weights) | 449 passed |
| 25 Sep | `tools/check_ui.py` (Chrome, 13 routes × 4 widths) | 52 layout checks passed (`reports/ui_review.json`) |

The deep-model tests download MobileNet weights on first run, so they fail offline.

## SRS deliverable 8: test kinds and where they live

| SRS test kind | Tests / evidence |
|---|---|
| Functional | `test_account_pages.py`, `test_alert_review_pages.py`, `test_event_detail_page.py`, `test_dashboard_reports_monitoring.py` |
| Integration | `test_pipeline_http_integration.py` (real Flask app, database, both exported models), `test_dashboard_reports_monitoring.py` |
| Boundary | `test_audio_preprocessing.py` (0.5 s minimum, short clips, segment edges), `test_augmentation.py` (clips shorter than the shift) |
| Negative | invalid file, unsupported format, silent audio (422, audited), dismissed alert refusing further actions (409) |
| Security | `test_csrf.py`, role checks (export 403 for non-admins), lockout and audit tests in `test_database.py` |
| Database | `test_database.py` (every role, overrides keep model output, 20,000-event search speed), `test_persistence.py` |
| Audio format | `test_audio_download_and_history.py` (every accepted format), MP3 round trip in `tools/robustness_probe.py` |
| Microphone | live-window payload test; `tools/browser_acceptance.py` with Chrome's fake microphone: consent, Available/Active/Paused/Disconnected/Permission denied, windows from both models, live alert |
| Silence / clipping / noise | `test_audio_preprocessing.py`, quality tests, `test_augmentation.py::test_add_noise_hits_the_requested_snr` |
| Preprocessing | `test_audio_preprocessing.py` |
| Feature extraction | `test_feature_extraction.py`, `test_transfer_model.py` |
| Python model | `test_predictor_contract.py`, `test_transfer_model.py`, `test_classical_zoo.py` |
| GTM model | `test_model_independence.py`, real export in `test_pipeline_http_integration.py`, `gtm_model/gtm_metrics.json` |
| Comparison | `test_consistency_taxonomy.py`, `test_comparison_report.py`, `reports/model_comparison.csv` |
| Alert rules | `test_alert_rules_config.py`, `test_an_acknowledged_alert_can_still_be_escalated_then_dismissed` |
| Duplicate audio | `test_allow_duplicate_stores_a_second_event`, `test_two_stage_near_duplicate_flags_a_quieter_copy_but_not_a_different_sound`, `reports/near_duplicates.json` |
| Low confidence | `test_consistency_taxonomy.py`, `reports/threshold_calibration.json` |
| Overlapping sound | `test_overlap_detected_when_runner_up_is_strong`, overlap probe in `reports/ROBUSTNESS.md` |
| Data leakage | `test_frozen_split.py` (incl. source groups), `test_split_integrity.py`, TM imports train-only test |
| Hidden-test readiness | `reports/ROBUSTNESS.md`: noise, echo, low volume, device, distance, partial, overlap, re-encoding, out-of-set sounds |

## Hidden-test readiness checklist (SRS §1.8 rule 9)

| Condition | Covered by | Result |
|---|---|---|
| Background noise | probe at 10 dB and 0 dB SNR with real background recordings | see `reports/ROBUSTNESS.md` |
| Echo | probe with RT60 0.8 s | same |
| Low volume | probe at −30 dB | same |
| Different devices | phone-band simulation | same |
| Partial events | half of the clip | same |
| Overlapping sounds | second event at −6 dB | same |
| Similar categories | ESC-50 fireworks, knocking, clapping, laughing, crying, bells | same |
| Re-encoded files | 64 kbit/s MP3 | same |
| Distant sources | 20 m simulation | same |

## Manual checks still to do before the video

Done (automated, 27 Sep):

- Upload each format (WAV, MP3, FLAC, OGG, M4A) through the browser.
- Batch upload with bad files mixed in with good ones.
- Live monitor states and the three-window alert, using Chrome's fake microphone.

Still to do:

- The same live run with a real microphone on the demo laptop (unplug it for Disconnected).
- A keyboard-only pass through login, upload, the event page and review.
- Firefox and Edge (only Chrome has been checked).
