# SonicSentinel AI

SonicSentinel listens to uploaded recordings and to a consented browser microphone, and
sorts what it hears into ten sound classes: Machinery Fault, Glass Breaking, Alarm or
Siren, Vehicle Horn, Animal Sound, Gunshot, Panic Scream, Aggression, Person Asking for
Help and Background Noise. Two models trained separately judge every clip: our own Python
model and a Google Teachable Machine (TM) audio model. The app compares them, rates the
audio quality, applies alert rules, and sends anything uncertain to a human reviewer.

It is an Aptech TechWiz 7 (NextWave AI and ML) competition prototype for supervised
operators. **It is not a certified emergency-response or law-enforcement system.**

**Live application:** <https://shelf-starlight-subfloor.ngrok-free.dev/login>. Evaluator
login: `evaluator` / `Eval#Sonic2026` (all accounts are listed under [Run](#run)). It runs
on the team's machine through an ngrok tunnel during the evaluation period. The first visit
shows ngrok's notice page; click **Visit Site** once. If the link is down, the local
instructions below run the same app.

## Links

| Deliverable | Link |
|---|---|
| Live application | <https://shelf-starlight-subfloor.ngrok-free.dev/login> |
| Source code (this repository) | <https://github.com/build-with-fun/TechWiz7> |
| Project report | [PROJECT_REPORT.md](PROJECT_REPORT.md) |
| Technical blog (2,200+ words) | <https://dev.to/buildwithfun/ai-voice-analysis-13oh> |
| Demonstration video (.mp4) | to be added |
| Teachable Machine project (opens after any Google sign-in) | <https://teachablemachine.withgoogle.com/train/audio/17pC3F6eg_sY_HHF8fY8aI2M73_B87UQ_> (project file on Drive: <https://drive.google.com/file/d/17pC3F6eg_sY_HHF8fY8aI2M73_B87UQ_/view>) |
| Teachable Machine model (hosted by TM) | <https://teachablemachine.withgoogle.com/models/56AmxJNhY/> |
| Dataset (training, validation, test audio) | <https://drive.google.com/drive/folders/19dNe0p0zEIV5f4IOaD0WQHPgDrCQHL1B> (see [Get the dataset](#get-the-dataset)) |

## Results on unseen recordings

Frozen test split: 450 original recordings (45 per class) that no model trained on or was
tuned on. The split keeps every source recording in one partition (see
[the leakage fix](documentation/devlog.md)).

| Model | Accuracy | Macro F1 | Critical-class recall (mean) | Evidence |
|---|---:|---:|---:|---|
| Python: AST embeddings + logistic regression (served) | **0.891** | **0.892** | **0.907** | `python_models/metrics/transfer_test_ast_current.json` |
| Python: CNN14 embeddings + MLP (previous) | 0.840 | 0.840 | 0.862 | `python_models/metrics/transfer_test_current.json` |
| Python baseline: HistGradientBoosting, 254 hand-made features | 0.731 | 0.730 | 0.822 | `python_models/metrics/classical_metrics_hgb_split_v2.json` |
| Teachable Machine audio model (served, 200 epochs) | 0.511 | 0.492 | 0.600 | `gtm_model/gtm_metrics.json` |
| SRS target (both models) | 0.85 | 0.80 | 0.85 per class | |

The served Python model meets the accuracy target (0.891 ≥ 0.85) and the macro-F1 target
(0.892 ≥ 0.80). The SRS also asks for 85% recall **per critical class**. Help (1.00),
Gunshot (0.96) and Glass (0.91) meet it, but Aggression (0.84, 38/45) and Panic Scream
(0.82, 37/45) don't, even though the five average 0.907. Those two get confused with each
other, but 43/45 Aggression and 41/45 Panic Scream clips are still labelled as *some*
critical class, so the alert still fires. Retraining with augmented copies on Day 5 didn't
help on validation ([devlog](documentation/devlog.md)). The mistakes are described in
[MODEL_EVALUATION.md](documentation/MODEL_EVALUATION.md) and
[ROBUSTNESS.md](reports/ROBUSTNESS.md). Keep in mind that a confidence score is only the
model's own estimate: 10 of the 450 test predictions were wrong at 0.9 or higher.

The Teachable Machine model is well below all three targets. Teachable Machine only trains
one layer on top of a frozen speech-command network, and in our runs it stalled above about
1,400 training samples. Training for 200 epochs instead of the default 50 was the last thing
left to try (test accuracy went from 0.493 to 0.511). The app doesn't average the two models:
if they disagree, or TM isn't confident, the clip goes to manual review. More in
[MODEL_EVALUATION.md](documentation/MODEL_EVALUATION.md).

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

Models: the Python model is in `python_models/best/` (in Git, ~25 MB) and the TM export in
`gtm_model/` (`model.json`, `weights.bin`, `metadata.json`, the converted `gtm_model.h5` and
`frontend_config.json`). If either is missing, analysis is disabled and the app says why.
The database is SQLite at `database/sonicsentinel.db` (`SST_DB_PATH`) and uploaded audio
goes to `data/storage/` (`SST_STORAGE_DIR`). All settings are listed in `.env.example`.

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

Anyone can register at `/register` as a normal user. An administrator can give other roles
under Admin > Users.

**Several users at once**: use gunicorn instead of the development server, with **one
worker** and several threads. The live page's "N windows in a row" counter is kept in the
worker's memory, so with two workers a session's windows could be split between them and
the streak would never be seen. One worker also means only one copy of the models in
memory. Set `SST_TORCH_THREADS` to the number of physical cores.

```bash
set -a; . ./.env; set +a
SST_TORCH_THREADS=4 .venv/bin/gunicorn -w 1 --threads 8 --timeout 300 -b 127.0.0.1:5055 "src.app:create_app()"
```

`tools/benchmark_scale.py` tests this setup with 20,000 events and up to 20 users
(`reports/scale.json`).

**Public URL from your own machine** (this is how the live link above works): run gunicorn
as above in production mode (`SST_PRODUCTION=1`, `SST_SECRET_KEY` set,
`SST_TRUSTED_PROXY_HOPS=1`), then `ngrok http 5055` (a free account is enough; it prints the
address). Keep both running and stop the machine from sleeping, for example with
`systemd-inhibit --what=sleep:idle ngrok http 5055`.

**Hosted deployment** (Hugging Face Docker Space; since July 2026 this needs a paid PRO
account): create a Docker Space, add the secret `SST_SECRET_KEY`, then run
`HF_TOKEN=hf_... scripts/deploy_hf_space.sh <owner>/<space>`. Details are in
`deploy/huggingface/SPACE.md`. To measure availability (NFR 5), keep
`tools/uptime_probe.py https://<owner>-<space>.hf.space` running during the evaluation and
then read `reports/uptime_summary.json`.

**Behind a reverse proxy** (nginx, Render, Cloudflare): set `SST_TRUSTED_PROXY_HOPS` to the
number of proxies in front of the app, e.g. `1` for a single nginx or `2` with a CDN in
front of it. The app only reads client IPs from `X-Forwarded-For` when this is above zero.
With the default `0` it uses the socket address, so nobody can fake their IP with that
header. If it's set wrong, the audit log and rate limits will see every request as coming
from your proxy.

## Using it

- **Upload**: on the Upload page, pick one or more WAV/MP3/FLAC/OGG/M4A files (up to
  50 MB, 0.5 s to 5 min). Unreadable, silent or unusable files are refused with a reason.
- **Preview and metadata**: the event page plays the recording (play, pause, replay,
  seek, volume) and lists filename, format, duration, sample rate, channels, bit depth,
  size and upload time.
- **Waveform and spectrogram**: drawn on the event page and embedded in the report.
- **Both predictions**: each model's class, confidence and top three; the full ten-class
  scores are on the event page and in the report.
- **Reading the comparison**: *Strong/Acceptable Match* means the same class; *Weak
  Match* means the same class but the confidences are far apart; *Model Disagreement*
  means different classes; *Uncertain Result* means low confidence or a near tie. Δ is
  |Python top confidence − TM top confidence|.
- **Audio quality**: Good, Acceptable, Poor (analysed but sent to review), Unusable
  (refused).
- **Live monitoring**: on the Live page, tick the consent box and allow the microphone.
  The status shows Available, Active, Paused, Disconnected or Permission denied. Every 2 s
  window goes through both models. By default a critical class needs 3 agreeing windows in
  a row before it alerts (`config/thresholds.json`).
- **Alerts**: on the Alerts page (operator, admin) you can acknowledge, dismiss with a
  reason, or escalate.
- **Manual review**: on the Reviews page (reviewer, admin), listen, confirm or correct the
  class, and add a comment. The models' original output is kept.
- **Dashboard, search, reports**: the dashboard (recent uploads, critical timeline, admin
  metrics and anomalies), Events (nine filters), Analytics (confidence distribution, false
  positives/negatives from reviews, alert response times), the event report download, and
  CSV/XLSX export for administrators.

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
.venv/bin/python -m python_models.train_transfer embed --backbone ast --splits train val test
.venv/bin/python -m python_models.train_transfer select --backbone ast # validation only
.venv/bin/python -m python_models.train_transfer final --backbone ast --out python_models/best
.venv/bin/python audio_dataset/scripts/make_gtm_imports.py --max-per-class 140   # train-only TM samples
.venv/bin/python tools/train_gtm_browser.py --epochs 200               # trains in teachablemachine.withgoogle.com
.venv/bin/tensorflowjs_converter --input_format tfjs_layers_model --output_format keras \
    gtm_model/model.json gtm_model/gtm_model.h5
.venv/bin/python tools/evaluate_gtm.py
```

## Get the dataset

The audio is not in Git (size and licences). It is shared on Google Drive:
<https://drive.google.com/drive/folders/19dNe0p0zEIV5f4IOaD0WQHPgDrCQHL1B>

| On Drive | Contents |
|---|---|
| `originals/` | 2,625 real recordings (ESC-50, UrbanSound8K, FSD50K), one folder per class |
| `synthetic/` | 375 generated clips: 300 Person Asking for Help (offline TTS), 50 Aggression, 25 Panic Scream |
| `gtm_samples/` | the 1,400 one-second training windows the Teachable Machine model learned from |
| `manifest.csv`, `manifest_with_split.csv`, `manifest_schema.md`, `manifests/` | per-clip metadata: audio ID, class, source, licence, original or augmented, split, sha256 |
| `split.json`, `train_ids.txt`, `val_ids.txt`, `test_ids.txt` | the frozen split: 2,100 train, 450 validation, 450 test (300 per class overall) |
| `DATA_DICTIONARY.md`, `DATA_ATTRIBUTION.md`, `licences/` | column definitions, sources and per-author credits |

To use it with a clone, download the folder and copy `originals/`, `synthetic/` and
`gtm_samples/` into `audio_dataset/`. The metadata files are already in this repository and
identical to the Drive copies. Then check every file against the manifest:

```bash
.venv/bin/python audio_dataset/scripts/verify_dataset.py --strict
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/api/health/ready` is 503 | a model is missing: run `tools/fetch_pretrained.py --ast`; check `gtm_model/gtm_model.h5` exists |
| MP3/M4A upload refused as unreadable | FFmpeg is not on `PATH` |
| First analysis is slow | the AST model loads on first use; later clips are much faster |
| "Permission denied" on the live page | allow the microphone in the browser's site settings; use `localhost` or HTTPS |
| Login lockout | 5 failures lock an account for 15 min (`config/auth.json`) |

## Limitations

- Two of the five critical classes, Aggression (0.84) and Panic Scream (0.82), are below
  0.85 recall, and they are often confused with each other.
- Several labels are proxies: Aggression includes door slams, Machinery Fault is normal
  machinery, Person Asking for Help is text-to-speech only. See
  [BASELINE_AUDIT.md](documentation/BASELINE_AUDIT.md).
- A sound that isn't one of the ten classes still gets one of the ten labels; the review
  queue is there to catch that.
- Tested with up to 20 users at once on one machine. The repeated-detection counter lives
  in one process, so the app runs as a single worker.
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
