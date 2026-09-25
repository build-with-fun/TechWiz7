# Engineering and writing overhaul report

## Audit findings

`PROJECT_AUDIT.md` contains the 80-requirement SRS trace and original baseline. The most consequential issues were a missing independent GTM export, ignored upload persistence callback, corrupt live WAV payload, hidden microphone consent, incomplete alert-rule enforcement, a retention purge that deleted nothing, export access wider than the SRS, CSRF absence, and documentation describing routes or model training steps that did not work. Several templates used layout classes without CSS definitions. The original README overstated the GTM state and contained broken installation/test commands.

## Work completed

The upload and live routes now call the real analysis/persistence boundary and return stored event IDs. The browser encodes live WAV bytes once, exposes consent and history, and the upload form supports multiple files. Per-class repeat counts, alert eligibility/suppression and escalation are applied. Reports use valid date parsing, administrator-only exports, complete filtered pagination and spreadsheet-safe CSV cells. Unsafe session requests use CSRF protection; production mode requires a configured secret and secure cookies. Config edits validate the proposed JSON before replacing a file. Retention preview and purge now use the same eligibility logic and delete eligible bytes/records while retaining held evidence.

A shared dark operations-console style was added in `static/css/product.css` and the login, dashboard, upload and live views were revised. Current desktop screenshots are in `screenshots/13_login_refreshed.png` and `screenshots/14_dashboard_refreshed.png`; 390px registration/login captures are `screenshots/15_register_mobile.png` and `screenshots/16_login_mobile.png`. The Teachable Machine import generator creates browser-compatible sample archives from 32 distinct training parents per class and records lineage. The separate model was trained in the real Teachable Machine browser app, exported as TensorFlow.js, and converted to a server-loadable Keras artifact. The GTM adapter gained a browser-FFT path and now resamples before choosing the scoring window. A real dual-model HTTP upload persists both score distributions. Browser/server feature parity has not been measured.

Documentation was rewritten around the delivered code: `README.md`, `ARCHITECTURE.md`, `DATABASE_SCHEMA.md`, `API_DOCUMENTATION.md`, `DESIGN_SYSTEM.md`, `PROJECT_REPORT.md`, `TEST_PLAN.md`, `DEMO_GUIDE.md`, and `INTERVIEW_PREP.md`. The older `documentation/api_contract.md` should be treated as a proposal; it lists unimplemented model administration routes. No fictional team member, performance figure or competition history was added.

A fresh seed now registers the exact Python bundle served from `python_models/best/` rather than an experiment with the same version string. It records the GTM frontend version and measured summary metrics. The upload page uses readable format names and one accessible file picker; mobile navigation stays on one scrollable row. The manifest-based `DATA_ATTRIBUTION.md` flags recordings needing an explicit redistribution check.

The public readiness endpoint now checks model availability as well as the database, so a demo check cannot return "ready" while classification is disabled. Public health responses use a generic unavailable reason; detailed loader failures remain in server logs.

The base dependency pins were aligned with TensorFlow 2.19: NumPy is 2.1.3 and protobuf is 5.29.6. The optional PyTorch dependency used by deep-model tests is in `requirements-test.txt`. A local `pip install --dry-run --no-index -r requirements.txt` resolves with the installed packages. A fresh-environment download/install was not run.

## Verification

- Focused pipeline/HTTP/CSRF/rules tests: **35 passed**.
- Retention preview/purge and expired-audio tests: **2 passed**.
- Current non-deep suite before the final real-model regression test: **418 passed, 27 deselected** (`pytest -q -k 'not deep_models'`).
- Real-model HTTP integration: **2 passed**, including the exported Python and GTM models.
- Deep-model suite with permitted pretrained-weight download: **27 passed**.
- Full suite with permitted network access for pretrained weights: **449 passed, 213 warnings**. An earlier ordinary sandboxed full run had 431 passes, 2 failures and 13 errors when the MobileNet weights host was unreachable.
- Baseline full suite before modifications: **422 passed, 2 failed, 13 errors**; deep-model tests attempted an unavailable MobileNet download. The same tests now pass with network access.
- Dataset quick strict verifier: **26/26 checks passed**; hashes and content uniqueness skipped.
- Dataset full strict verifier: **28/28 checks passed**, including hashes and content uniqueness.
- Desktop login/dashboard/upload/live and 390px login/registration/upload/live visual smoke reviewed in headless Chrome.
- GTM server-path evaluation on all 450 frozen test clips: **0.2778 accuracy, 0.2755 macro F1, 0.3200 critical-class recall**. The exact sample IDs and confusion matrix are in `gtm_model/gtm_metrics.json`.
- A real HTTP upload with both models returned 201 and stored one event and one audio row. A fresh database seed created six accounts and registered both models with matching version identifiers. JavaScript syntax checks, Python compilation and `git diff --check` passed.
- The corrected real-model acceptance script passed on 13 local sample WAVs; its process-level clip/window maximums were 1.526/0.260 seconds. `tools/predict.py` independently displayed both predictions and the manual-review decision for `sample_audio/gunshot.wav`.

The deterministic HTTP tests verify transport and storage, while the separate real-model test verifies that both exported artifacts execute. Only `gtm_model/gtm_metrics.json` and the recorded Python evaluation contain acoustic performance measurements.

## Remaining limitations and compatibility

Both independent models now run, but **neither meets the SRS accuracy, macro F1 and critical-recall targets**. The GTM server frontend has not been compared numerically with the browser analyser. A clean clone lacks the ignored audio corpus and generated import ZIPs. Browser/device accessibility, deployment, concurrency and latency targets have not been validated. Public deployments must replace demo credentials and provide HTTPS. The retention purge is destructive when explicitly invoked with `dry_run=0`; preview it first and back up evidence if policy requires it.

The repository has no project-code `LICENSE`. Choosing one requires the project owners' decision; the source-audio licence metadata in `DATA_ATTRIBUTION.md` is a separate matter. `documentation/TEAM_CONTRIBUTION_RECORD.md` is deliberately an evidence template because the Git history cannot establish individual contributions.

The working `.venv` contains an older optional `tensorflow-cpu 2.21.0` install alongside the required `tensorflow 2.19.0`, and optional `tensorflowjs` is missing its declared `tensorflow-decision-forests` dependency. `pip check` therefore fails in this environment despite the base requirements dry run and real model tests passing. A clean virtual environment should be used for deployment validation; optional converter dependencies need a separate reproducibility check.

The API contracts for persisted upload/live results now include event IDs. CSV/XLSX exports are administrator-only, which tightens access compared with the old role grants. CSRF tokens are required for cookie-authenticated mutations in normal operation; external clients must first obtain a session and token. These changes may require updates to any scripts that called the prior routes without a token.
