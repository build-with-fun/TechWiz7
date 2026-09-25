# SonicSentinel AI

SonicSentinel is a sound-event review console for uploaded recordings and consented microphone sessions. It classifies ten types of sound, compares a locally trained Python model with a separately trained Google Teachable Machine (GTM) audio model, and keeps the evidence behind alerts and manual reviews. It is a competition prototype for supervised operators, not an emergency dispatch system.

The ten categories are Machinery Fault, Glass Breaking, Alarm or Siren, Vehicle Horn, Animal Sound, Gunshot, Panic Scream, Aggression, Person Asking for Help, and Background Noise. The [SRS audit](PROJECT_AUDIT.md) maps every functional requirement to the implementation and remaining gaps.

## What you can do

- Upload WAV, MP3, FLAC, OGG, or M4A audio; inspect quality, both model scores, agreement, waveform, and an event record.
- Start and stop a browser microphone session with explicit consent. Each window is associated with a stored event and session history.
- Work through alerts and uncertain results using the security-operator and reviewer roles.
- Search event history, inspect the audit trail, and generate event/period reports. Administrators can export CSV/XLSX and preview or run retention cleanup.
- Edit validated alert and threshold configuration through the administrator interface.

Analysis requires **both** model artifacts. The app starts without them, but upload and live classification return a clear unavailable response. This prevents a single-model result from being presented as a comparison. Check `/api/health/ready` before a demo. The current measured Python-model results and any GTM verification result are recorded in [PROJECT_REPORT.md](PROJECT_REPORT.md); do not infer accuracy from the interface.

**Model limitation:** the included GTM export is operational but scored only **27.78% accuracy on all 450 held-out clips** through the server path. The served Python model's recorded full-test accuracy is **69.78%**. Neither meets the SRS acceptance target. Use this as a review prototype, not as an unattended safety detector.

## Run locally

Use Python 3.12, FFmpeg, and a modern browser. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python database/init_db.py
SST_SECRET_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_hex(32))')" \
  .venv/bin/python -c 'from src.app import create_app; create_app().run(host="127.0.0.1", port=5055)'
```

Open `http://127.0.0.1:5055/login`. The explicit `database/init_db.py` step creates the SQLite schema and demo accounts from `database/seed_credentials.json`; app startup creates only the schema if needed. The evaluator account is `evaluator` / `Eval#Sonic2026`; the reviewer is `reviewer` / `Review#Sonic2026`; the security operator is `operator` / `Operate#Sonic2026`. These are deliberately published demo credentials. Replace the seed file or set `SST_SEED_CREDENTIALS` before any public deployment, and set `SST_PRODUCTION=1` plus a strong `SST_SECRET_KEY` behind HTTPS so session cookies are secure. The default database is `database/sonicsentinel.db`, and audio evidence is under `data/storage/`; set `SST_DB_PATH` and `SST_STORAGE_DIR` to change those paths.

The Python bundle lives in `python_models/best/`. The separate GTM TensorFlow.js export, converted server model and frontend configuration live in `gtm_model/`. To reproduce training, generate train-only imports with `audio_dataset/scripts/make_gtm_imports.py`, then follow [the GTM handoff](gtm_model/upload_package/README.md). The training browser run uses Playwright as an optional tool dependency; install it with `.venv/bin/python -m pip install -r requirements-training.txt` and run `tools/train_gtm_browser.py`. The source audio and generated ZIPs are ignored by Git, so a clean clone needs the authorized corpus supplied separately.

## Verify

```bash
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q
.venv/bin/python audio_dataset/scripts/verify_dataset.py --strict
.venv/bin/python tools/evaluate_gtm.py
.venv/bin/python tools/check_e2e_upload.py
.venv/bin/python -m compileall -q src audio_preprocessing feature_extraction python_models audio_dataset
```

`--quick` skips the expensive hash/content-uniqueness checks. The full test suite needs the optional PyTorch test dependency and its deep-model experiments may download pretrained MobileNet weights; see [TEST_PLAN.md](TEST_PLAN.md) for the latest actual result and any environment limitation. For a quick API check, visit `/api/health` and `/api/health/ready`.

To inspect a single clip without writing to the database, run `.venv/bin/python tools/predict.py sample_audio/gunshot.wav` when the local sample corpus is present. The two models currently disagree on this sample and route it to manual review; the output changes if the models are retrained.

## Where to look

| Area | Files |
|---|---|
| Flask routes and role checks | `src/api/`, `src/auth.py`, `src/app.py` |
| Audio and model pipeline | `audio_preprocessing/`, `feature_extraction/`, `src/inference/`, `src/services/pipeline.py` |
| Storage and relational records | `src/models.py`, `src/services/persistence.py`, `src/db.py` |
| Interface | `templates/`, `static/css/`, `static/js/` |
| Configuration | `config/`, `alert_rules/` |
| Training and evidence | `audio_dataset/`, `python_models/`, `gtm_model/`, `reports/` |

The [architecture](ARCHITECTURE.md), [API reference](API_DOCUMENTATION.md), [database schema](DATABASE_SCHEMA.md), [design system](DESIGN_SYSTEM.md), [data attribution](DATA_ATTRIBUTION.md), and [demo guide](DEMO_GUIDE.md) cover the details needed to develop or present the project. [AI_USAGE.md](AI_USAGE.md) records assistance disclosure; no team contribution or competition history is implied by generated files.

Visual review captures: desktop [sign-in](screenshots/13_login_refreshed.png), [dashboard](screenshots/14_dashboard_refreshed.png) and [live monitoring](screenshots/17_live_1440.png), plus 390px [registration](screenshots/15_register_mobile.png), [sign-in](screenshots/16_login_mobile.png), [upload](screenshots/17_upload_390.png) and [live monitoring](screenshots/17_live_390.png). They are layout evidence; model behavior is verified separately.
