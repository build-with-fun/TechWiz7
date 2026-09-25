# Seven-minute live demo

Run `database/init_db.py` to seed the demo accounts, start the app with both models loaded, and check `/api/health/ready` before opening the room. Use the published evaluator demo account `evaluator` / `Eval#Sonic2026` from `database/seed_credentials.json`. Keep one permitted clip from `sample_audio/` ready and verify it is actually present on the demo machine; source clips are ignored by Git and may be missing from a clean clone. Do not claim an alert will fire for a particular clip until that clip has been run against the final models and configured thresholds. Keep the measured GTM accuracy (**0.2778 on all 450 held-out clips through the server**) and Python result visible in `PROJECT_REPORT.md`; describe this as a supervised review prototype.

| Time | Screen and action | Value to explain |
|---|---|---|
| 0:00–0:45 | Login and dashboard | What the console is for, who uses it, and honest model readiness/status |
| 0:45–2:15 | Upload one prepared audio file | Validation, quality verdict, persisted event ID and immediate feedback |
| 2:15–3:30 | Open event detail | Show independent Python/GTM top-three scores, agreement, waveform, spectrogram and source playback |
| 3:30–4:20 | Alerts or review queue | Explain that critical/uncertain results are routed to a human; use a previously verified stored case if the new clip does not trigger one |
| 4:20–5:10 | Live microphone page | Read the consent statement, start/stop only with permission, and show a persisted session window/history |
| 5:10–6:00 | Search, period report and CSV/XLSX export | Traceability and role-restricted reporting; use evaluator/admin privileges |
| 6:00–7:00 | Architecture and limitations | Explain two-model independence, SQLite/audio storage, actual test scores and next validation work |

If the room has no microphone permission, show the consent UI and an already recorded session; do not simulate live results. If Teachable Machine artifacts are missing or the readiness endpoint is false, show the health reason and the train-only import/export flow, then demonstrate non-model pages and unit/integration evidence. Never present deterministic test predictors or static screenshots as a live inference result. For an alert/review demo, prepare a real stored event in advance and say when it was recorded.

With the current bundled models, `sample_audio/gunshot.wav` is a useful prepared case: the Python model predicts Gunshot, GTM predicts Aggression, and the pipeline requests manual review without raising an unconfirmed alert. This was observed with `tools/predict.py`; rehearse it again if either artifact changes. The sample file is local and ignored by Git, so make sure it is on the demo machine.

Before presenting, run the tests listed in `TEST_PLAN.md`, sign in as each role once, check that the prepared recording plays, confirm that the database is writable and storage has space, and keep a local copy of this guide. Any team contribution slide must be filled from the actual contributors' work; this repository cannot establish that attribution on its own.
