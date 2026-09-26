# SonicSentinel AI

SonicSentinel listens to uploaded recordings and to a consented browser microphone, and
sorts what it hears into ten sound classes: Machinery Fault, Glass Breaking, Alarm or
Siren, Vehicle Horn, Animal Sound, Gunshot, Panic Scream, Aggression, Person Asking for
Help and Background Noise. Two models trained separately judge every clip: our own Python
model and a Google Teachable Machine (TM) audio model. The app compares them, rates the
audio quality, applies alert rules, and sends anything uncertain to a human reviewer.

It is an Aptech TechWiz 7 (NextWave AI and ML) competition prototype for supervised
operators. **It is not a certified emergency-response or law-enforcement system.**

## Results on unseen recordings

Frozen test split: 450 original recordings (45 per class) that no model trained on or was
tuned on. The split keeps every source recording in one partition (see
[the leakage fix](documentation/devlog.md)).

| Model | Accuracy | Macro F1 | Critical-class recall (mean) | Evidence |
|---|---:|---:|---:|---|
| Python: AST embeddings + logistic regression (served) | **0.891** | **0.892** | **0.907** | `python_models/metrics/transfer_test_ast_current.json` |
| Python: CNN14 embeddings + MLP (previous) | 0.840 | 0.840 | 0.862 | `python_models/metrics/transfer_test_current.json` |
| Python baseline: HistGradientBoosting, 254 hand-made features | 0.731 | 0.730 | 0.822 | `python_models/metrics/classical_metrics_hgb_split_v2.json` |
| Teachable Machine audio model | see `gtm_model/gtm_metrics.json` | | | `tools/evaluate_gtm.py` |
| SRS target (both models) | 0.85 | 0.80 | 0.85 per class | |

The served Python model **meets all three SRS targets**: accuracy 0.891 ≥ 0.85, macro F1
0.892 ≥ 0.80, critical recall 0.907 ≥ 0.85. Per critical class it reaches Help 1.00,
Gunshot 0.96, Glass 0.91, Aggression 0.84 and Panic Scream 0.82; two of five critical
classes are below 0.85 individually even though their mean clears it, so those two remain
the honest weak point. Where it goes wrong is described in
[MODEL_EVALUATION.md](documentation/MODEL_EVALUATION.md) and
[ROBUSTNESS.md](reports/ROBUSTNESS.md). A confidence score is the model's own estimate,
not proof: 21 of 450 test predictions were wrong at 0.9 or higher.

## Install (Ubuntu 22.04+ or Windows 10/11 with WSL; Python 3.12)

Prerequisites: Python 3.12, FFmpeg (decodes MP3/OGG/M4A), Git, Google Chrome for the
browser tools, ~2 GB free disk.

```bash
sudo apt install ffmpeg                       # Windows: winget install ffmpeg
python3.12 -m venv .venv
.venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python tools/fetch_pretrained.py --ast   # AST weights for the served model, ~350 MB
cp .env.example .env                          # then set SST_SECRET_KEY
.venv/bin/python database/init_db.py          # SQLite schema + demo accounts + model versions
```

The served Python model's AST weights are fetched automatically from Hugging Face on
first analysis (model card `MIT/ast-finetuned-audioset-10-10-0.4593`, ~340 MB, cached in
`~/.cache/huggingface`). Set `HF_HUB_OFFLINE=1` after the first run to go cache-only.

Models: the Python model is `python_models/best/` (in Git, ~25 MB). The TM export is
`gtm_model/` (`model.json`, `weights.bin`, `metadata.json`, converted `gtm_model.h5`,
`frontend_config.json`). Both must be present or analysis stays disabled with a clear
message. Database: SQLite at `database/sonicsentinel.db` (`SST_DB_PATH`). Stored audio:
`data/storage/` (`SST_STORAGE_DIR`). Settings are listed in `.env.example`.

## Run

```bash
set -a; . ./.env; set +a
.venv/bin/python -c 'from src.app import create_app; create_app().run(host="127.0.0.1", port=5055)'
```

Open <http://127.0.0.1:5055/login>. Check <http://127.0.0.1:5055/api/health/ready>
before a demo; it reports 503 unless the database and both models are loaded.

Evaluator accounts (demo only; replace `database/seed_credentials.json` before any public
deployment):

| Role | Username | Password |
|---|---|---|
| Administrator | `admin` | `Admin#Sonic2026` |
| Evaluator (administrator rights) | `evaluator` | `Eval#Sonic2026` |
| Audio reviewer | `reviewer` | `Review#Sonic2026` |
| Security operator | `operator` | `Operate#Sonic2026` |
| Maintenance operator | `maintenance` | `Maint#Sonic2026` |
| Normal user | `user` | `User#Sonic2026` |

Anyone can register at `/register` as a normal user; an administrator assigns other roles
under Admin → Users.

**Behind a reverse proxy** (nginx, Render, Cloudflare): set `SST_TRUSTED_PROXY_HOPS` to
the number of proxies in front of the app — `1` for a single nginx, `2` if a CDN sits in
front of it. The app reads client IPs from `X-Forwarded-For` only when this is non-zero;
with the default `0` the socket peer address is used, so a spoofed forwarded header can
never impersonate another client. Set it correctly or audit logs and rate limits will
attribute every request to your proxy.

## Using it

- **Upload**: Upload page, choose one or several WAV/MP3/FLAC/OGG/M4A files (≤50 MB,
  0.5 s–5 min). Unreadable, silent or unusable files are refused with the reason.
- **Preview and metadata**: the event page plays the recording (play, pause, replay,
  seek, volume) and lists filename, format, duration, sample rate, channels, bit depth,
  size and upload time.
- **Waveform and spectrogram**: drawn on the event page and embedded in the report.
- **Both predictions**: each model's class, confidence and top three; the full ten-class
  scores are on the event page and in the report.
- **Reading the comparison**: *Strong/Acceptable Match* = same class; *Weak Match* = same
  class but low confidence; *Model Disagreement* = different classes; *Uncertain Result*
  = low confidence or a near tie. Δ is |Python top confidence − TM top confidence|.
- **Audio quality**: Good, Acceptable, Poor (analysed but sent to review), Unusable
  (refused).
- **Live monitoring**: Live page, tick the consent box, allow the microphone. The pill
  shows Available / Active / Paused / Disconnected / Permission denied. Every 2 s window
  goes through both models; by default a critical class needs 3 consecutive agreeing windows before it alerts (`config/thresholds.json`).
- **Alerts**: Alerts page (operator, admin) to acknowledge, dismiss with a reason, or
  escalate.
- **Manual review**: Reviews page (reviewer, admin). Listen, confirm or correct the
  class, comment. The models' original output is kept.
- **Dashboard, search, reports**: dashboard (recent uploads, critical timeline, admin
  metrics and anomalies), Events (nine filters), Analytics (confidence distribution,
  false positives/negatives from reviews, alert response), event report download, and
  CSV/XLSX export (administrators).

## Verify

```bash
.venv/bin/python -m pytest -q                                 # full suite
.venv/bin/python audio_dataset/scripts/verify_dataset.py --strict
.venv/bin/python tools/build_comparison_report.py             # both models, all 450 test clips
.venv/bin/python tools/robustness_probe.py                    # noise, echo, device, confusables
.venv/bin/python tools/evaluate_near_duplicates.py
.venv/bin/python tools/predict.py sample_audio/gunshot.wav    # one clip, no database
```

## Reproduce the models

```bash
.venv/bin/python -m python_models.preprocess_cache                     # 3,000 clips, ~3 min
.venv/bin/python -m python_models.train_transfer embed --splits train val test
.venv/bin/python -m python_models.train_transfer select                # validation only
.venv/bin/python -m python_models.train_transfer final --out python_models/best
.venv/bin/python audio_dataset/scripts/make_gtm_imports.py --max-per-class 140   # train-only TM samples
.venv/bin/python tools/train_gtm_browser.py                            # trains in teachablemachine.withgoogle.com
.venv/bin/tensorflowjs_converter --input_format tfjs_layers_model --output_format keras \
    gtm_model/model.json gtm_model/gtm_model.h5
.venv/bin/python tools/evaluate_gtm.py
```

The audio itself is not in Git (licences and size). A clean clone needs the corpus from
the team; `DATA_ATTRIBUTION.md` lists every source and licence.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/api/health/ready` is 503 | a model is missing: run `tools/fetch_pretrained.py --ast`; check `gtm_model/gtm_model.h5` exists |
| MP3/M4A upload refused as unreadable | FFmpeg is not on `PATH` |
| First analysis takes ~5 s | CNN14 loads on first use; later clips take well under a second |
| "Permission denied" on the live page | allow the microphone in the browser's site settings; use `localhost` or HTTPS |
| Login lockout | 5 failures lock an account for 15 min (`config/auth.json`) |

## Limitations

- Three of five critical classes are below 0.85 recall; Panic Scream and Aggression are
  the weakest and are often confused with each other.
- Several labels are proxies: Aggression includes door slams, Machinery Fault is normal
  machinery, Person Asking for Help is text-to-speech only. See
  [BASELINE_AUDIT.md](documentation/BASELINE_AUDIT.md).
- Sounds outside the ten classes are always forced into one of them; the review queue
  is the safety net.
- Not load-tested for many concurrent users; the repeated-detection state is per process.
- Stored audio is access-controlled but not encrypted at rest.

## Where to look

| Area | Files |
|---|---|
| Web app and roles | `src/app.py`, `src/auth.py`, `src/api/`, `templates/`, `static/` |
| Audio pipeline | `audio_preprocessing/`, `augmentation/`, `feature_extraction/` |
| Models | `python_models/`, `gtm_model/`, `src/inference/` |
| Decision logic | `src/inference/consistency.py`, `src/services/pipeline.py`, `alert_rules/`, `config/` |
| Data | `audio_dataset/`, `data/splits/` |
| Evidence | `reports/`, `python_models/metrics/`, `screenshots/`, `documentation/` |

Documents: [SRS traceability](documentation/SRS_TRACEABILITY.md),
[baseline audit](documentation/BASELINE_AUDIT.md), [model evaluation](documentation/MODEL_EVALUATION.md),
[viva pack](documentation/VIVA_PACK.md), [development log](documentation/devlog.md),
[architecture](ARCHITECTURE.md), [API](API_DOCUMENTATION.md), [database](DATABASE_SCHEMA.md),
[test plan](TEST_PLAN.md), [demo guide](DEMO_GUIDE.md), [AI usage](AI_USAGE.md),
[blog](documentation/blog.md).
