# Test plan and current results

The acceptance focus is a fresh judge journey, not only isolated functions. A result is marked **not run** until it is observed. The tests use deterministic predictors where indicated; those tests verify transport, persistence and access control, not acoustic accuracy.

| Case | Expected result | Actual result |
|---|---|---|
| Register a normal-user account and update its profile | New account cannot choose a privileged role; email/name persist | Account-page integration test passes |
| Sign in with seeded evaluator/reviewer/operator accounts; wrong password and lockout | Correct role session or uniform error | Existing auth tests pass; full browser role walk not run |
| Upload a valid WAV and supported MP3/FLAC/OGG/M4A | Stored audio, event, both scores, evidence link | WAV HTTP upload passed with both exported models and persisted an event/audio row; all-format browser pass not run |
| Submit invalid, empty, oversized, silent, clipped audio | Useful validation/quality response, no false event | Unit/API tests cover validation/quality; full browser pass not run |
| Upload multiple files, including one failure | Per-file status, successful files remain accessible | File input and JS support multiple; mixed batch browser pass not run |
| Start live without consent, then start/stop with consent | Refusal without consent; status/history and persisted windows with consent | HTTP live window persistence integration passes with deterministic predictors; hardware microphone browser pass not run |
| Compare both independent models on a real clip | Ten scores per model, top three, consistency, versions | Real dual-model HTTP upload passed; GTM held-out accuracy 0.2778 on 450 clips; browser/server frontend parity unverified |
| Repeat critical detections and change rule threshold | Per-class confirmation and alert once per run | Rule unit tests pass; real acoustic trigger not run |
| Reviewer changes class/severity with comments | Original scores remain, final decision/audit recorded | Existing review tests pass; browser journey not run |
| Operator acknowledges, dismisses and escalates | Valid guarded transitions; history records result | Existing alert tests pass; browser journey not run |
| Search by date/class/status/confidence and export | Role-scoped results; only admin gets CSV/XLSX | Export/period HTTP integration passes; complete filter matrix not run |
| Preview then purge expired unheld data | Same counts; bytes and eligible rows removed; held items preserved | Preview/purge and expired-audio response tests pass; held-item test not run |
| CSRF, role and ownership boundaries | Mutations reject missing token; unauthorized records concealed | CSRF and export-role tests pass; full role matrix not run |
| Liveness and readiness | Liveness stays 200; readiness is 503 without both models or database | Missing-model and loaded-model HTTP readiness tests pass |
| Responsive desktop/tablet/mobile, keyboard, reduced motion | No clipped controls, visible focus, readable states | Desktop login/dashboard/upload/live and 390px login/registration/upload/live reviewed in headless Chrome; tablet and hands-on keyboard/reduced-motion checks not run |
| Full real dual-model latency and 450 held-out clips | SRS time and metric targets recorded with evidence | Full 450-clip GTM server-path evaluation recorded in `gtm_model/gtm_metrics.json`; 13 local sample clips and live slices passed a process-level smoke in `reports/e2e_acceptance.json` (maximum 1.526 s clip, 0.260 s window); 30-second and load targets not validated |

## Commands run during overhaul

- `pytest -q tests/test_pipeline_http_integration.py tests/test_csrf.py tests/test_alert_rules_config.py`: **35 passed**.
- `pytest -q tests/test_retention_api.py`: **2 passed**.
- `pytest -q tests/test_account_pages.py`: **1 passed**.
- `pytest -q -k 'not deep_models'`: **418 passed, 27 deselected** after the account, retention and frontend changes. This includes the focused cases above.
- `pytest -q tests/test_pipeline_http_integration.py`: **2 passed**, including an HTTP upload through the actual Python and GTM exports.
- Audit baseline `pytest -q`: **422 passed, 2 failed, 13 errors**; failures/errors were in `tests/test_deep_models.py` because an uncached MobileNet download was unavailable.
- Final full suite with permitted access to pretrained MobileNet weights: **449 passed, 213 warnings** (`pytest -q --tb=short`). An earlier ordinary sandboxed run passed 431 tests but had the same 2 failures/13 errors when DNS was blocked; this is an environment limitation of the deep experiment tests, not a passing sandbox run.
- `audio_dataset/scripts/verify_dataset.py --strict --quick`: **26/26 checks passed** before implementation; hash and content-uniqueness checks skipped by `--quick`.
- `audio_dataset/scripts/verify_dataset.py --strict`: **28/28 checks passed**, including file hashes and duplicate-content checks.
- `tools/evaluate_gtm.py`: **450 frozen-test clips** (45 per class), **0.2778 accuracy, 0.2755 macro F1, 0.3200 critical macro recall** through the server path. Browser/server frontend parity is not verified.
- `tools/check_e2e_upload.py`: **passed on 13 local sample WAVs** with both real models; maximum observed process-level clip/window times were 1.526/0.260 seconds. These are not 30-second recordings or a concurrency benchmark.

The remaining acceptance work includes a clean install, an offline path for deep-model tests, 768px browser review, all roles, and browser/server GTM frontend parity. Mark any remaining gap in `HUMANIZATION_REPORT.md` and `PROJECT_REPORT.md` rather than calling an unrun check successful.
