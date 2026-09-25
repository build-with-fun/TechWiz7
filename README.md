# SonicSentinel AI — Real-Time Sound-Event Detection

Aptech **NextWave AI and ML** competition project (AcousticX Intelligence).
SonicSentinel AI classifies uploaded clips and live microphone audio into ten sound
classes with a locally trained **Python model**, independently classifies the same
audio with a separately trained **Google Teachable Machine** model, compares the two,
evaluates audio quality, assigns severity, and raises/escalates alerts for critical
events.

> © Aptech Limited — SRS v1.0. This repository is the competition submission.

## The ten classes

Machinery Fault · Glass Breaking · Alarm or Siren · Vehicle Horn · Animal Sound ·
Gunshot · Panic Scream · Aggression · Person Asking for Help · Background Noise

Critical classes (higher recall floor): Gunshot, Glass Breaking, Panic Scream,
Aggression, Person Asking for Help.

## Architecture at a glance

- **Web app** — Flask 3 (blueprints under `src/api/`, Jinja templates, session auth).
- **Audio pipeline** — `audio_preprocessing/` (decode → resample → quality verdict),
  `feature_extraction/` (locked 254-column vector, `audiofeat-1.0.0`).
- **Python model** — xgboost (selected over SVM/RF/ET/GB in the sweep), trained by
  `python_models/train_classical.py` under the harness in `python_models/tuning.py`
  (selection on validation only; test scored exactly once).
- **GTM model** — trained in the browser at teachablemachine.withgoogle.com on the same
  training recordings (2-second segments, `audio_dataset/gtm_samples/`), exported as
  TF.js, converted to Keras and served by `src/inference/gtm_predictor.py`. The Python
  model's prediction is structurally unable to reach it (`tests/test_model_independence.py`).
- **Comparison** — `src/inference/consistency.py`: class match + |Python − GTM| top-class
  difference → Strong / Acceptable / Weak Match, Model Disagreement, Uncertain Result.
- **Alerts** — configurable JSON rules in `alert_rules/` (category, minimum confidence,
  top-two margin, consecutive detections, model agreement, audio quality, severity,
  escalation, manual review).

## 1. Installation

### Prerequisites

| Software | Version | Notes |
|---|---|---|
| OS | Linux (tested on Ubuntu 24.04) / macOS / Windows+WSL | |
| Python | **3.12** | torch/librosa wheels; 3.13+ not supported |
| FFmpeg | 6.x | `sudo apt install ffmpeg` — OGG/M4A transcode |
| Git | any | |

### Steps

```bash
git clone <repository-url> sonicsentinel-ai
cd sonicsentinel-ai

# virtual environment (Python 3.12)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
# torch CPU wheel (already pinned in requirements.txt):
.venv/bin/pip install torch==2.14.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# convert the exported Teachable Machine model (gtm_model/model.json → gtm_model.h5)
.venv/bin/pip install tensorflowjs==4.20.0 tensorflow==2.19.0
```

### Database configuration and initialization

SQLite by default — no server needed. The first app boot creates and seeds
`database/sonicsentinel.db`. To force initialization:

```bash
.venv/bin/python -c "
from src.db import create_engine_for, default_db_path, init_db
init_db(create_engine_for(default_db_path()))
"
```

### Model placement

- **Python model:** `python_models/best/` (`model.joblib`, `label_encoder.json`,
  `feature_config.json`, `model_meta.json`). Retrain:
  `.venv/bin/python -m python_models.train_classical`
- **GTM model:** `gtm_model/` — place the TM export (`metadata.json`, `model.json`,
  `weights.bin`) here, then convert:
  `.venv/bin/python -c "import tensorflowjs as tfjs; ..."` → `gtm_model.h5` (see
  `gtm_model/upload_package/README.md`), and verify with
  `tools/capture_gtm_frontend.py verify --recordings gtm_model/browser_recordings.json`.
  Without an export the app still boots; live windows answer 503 `pipeline_unavailable`.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SONICSENTINEL_DB` | `database/sonicsentinel.db` | SQLite path |
| `SECRET_KEY` | dev fallback | **Set in production** — session signing |
| `FLASK_ENV` | `production` | |

## 2. Execution

```bash
# development
.venv/bin/python -c "from src.app import create_app; app = create_app(); app.run(port=5055)"

# production
.venv/bin/gunicorn -w 4 -b 0.0.0.0:8000 "src.app:create_app()"

# health
curl http://127.0.0.1:5055/api/health
```

Then open <http://127.0.0.1:5055/>.

### Tests

```bash
.venv/bin/python -m pytest -q          # full suite
.venv/bin/python -m pytest -q tests/test_alerts_api.py
```

## 3. Using the application

1. **Register and log in** — `/register`, `/login`. Roles: normal user, audio reviewer,
   security operator, maintenance operator, administrator.
2. **Upload audio** — Audio page → choose a WAV/MP3/FLAC/OGG/M4A clip (≤ 60 s). The app
   decodes it, shows metadata, waveform and spectrogram, then runs both models.
3. **Python prediction** — predicted class + confidence for **every** class.
4. **GTM prediction** — the same audio is preprocessed and classified by the exported
   Teachable Machine model independently.
5. **Model comparison** — class-match status, top-class confidence difference
   |Python − GTM|, top-two margin, and the consistency verdict (Strong/Acceptable/
   Weak Match, Model Disagreement, Uncertain Result).
6. **Audio quality** — Good / Acceptable / Poor / Unusable verdict with reasons.
7. **Alerts** — critical detections raise alerts; acknowledge / dismiss (with reason) /
   escalate from the Alerts page.
8. **Manual review** — reviewers confirm or override (final class/severity must be
   justified in comments; re-deciding a decided review is rejected).
9. **Dashboard** — event history with filters, severity distribution, model agreement.
10. **Live monitoring** — Live page grants microphone consent; 1–3 s windows are pushed
    continuously, repeated detections are grouped, critical windows alert immediately.
11. **Reports & exports** — per-event and period reports; CSV/XLSX export (admin).

### Administrator / evaluator credentials

| Role | Username | Password |
|---|---|---|
| Administrator | `admin` | `Admin#Sonic2026` |
| Audio reviewer | `reviewer` | `Review#Sonic2026` |
| Security operator | `operator` | `Operate#Sonic2026` |

*(Change these before any public deployment.)*

## 4. Sample audio

`sample_audio/` holds one permitted clip per class for quick evaluation, plus a silent
clip, an invalid file, and a low-quality clip to exercise the failure paths.

## 5. Assumptions and limitations

- **Assumptions.** Single-node deployment; SQLite is sufficient for the expected load;
  the microphone is available on the user's browser; evaluators test via the UI and the
  REST API; clips longer than 60 s are truncated rather than streamed.
- **Limitations.** Classical + deep models are trained on 300 originals/class — rare
  real-world acoustic variation (far-field gunshots, multi-source overlap) is
  under-represented; synthetic TTS clips make Person Asking for Help easier in-domain
  than outdoors; the GTM model runs a 2-second frontend window, so long events are
  scored on their loudest segment; CPU-only inference (no GPU assumed).
- **Noise robustness, false positives/negatives** are analysed in
  `documentation/perception/` and the project report.

## 6. Repository map

```
src/                 Flask app: api/, services/, inference/, auth, db, errors
templates/ static/   Jinja pages + assets
audio_dataset/       manifest, frozen split, verify scripts, GTM samples
python_models/       training scripts, harness, saved model (best/)
gtm_model/           TM export + converted model + upload package
feature_extraction/  locked 254-column extractor
audio_preprocessing/ decode/resample/quality pipeline
alert_rules/         configurable JSON rules (severity, review, retention)
database/            SQLite database + schema scripts
tests/               pytest suite (functional/integration/negative/security)
documentation/       dev log, API contract, perception studies
tools/               smoke_pipeline.py, capture_gtm_frontend.py
reports/ notebooks/ screenshots/ sample_audio/  evidence folders
```

## 7. Integrity statement

See `AI_USAGE.md` (AI tool declaration) and `documentation/devlog.md` (development
log). The final sound classification is generated **only** by the Python model and the
GTM model — never by an external generative-AI API.