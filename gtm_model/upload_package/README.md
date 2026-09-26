# Teachable Machine training package

Teachable Machine's audio uploader only accepts the ZIP format its own **Download Samples**
button produces: `samples.json` with browser-FFT frequency frames (43 frames × 232 bins, one
second) plus a WebM recording per sample. A ZIP of WAV files does not import.

## Build the imports (training recordings only)

```bash
.venv/bin/python audio_dataset/scripts/make_gtm_imports.py --max-per-class 140
```

For training recordings of split v2 (140 of the 210 per class with this cap), the script takes the app's own
preprocessed audio, resamples it to 44.1 kHz and cuts the loudest one-second window (the same
rule the server uses). It writes:

- `tm_imports/<class>.zip`: what you upload to Teachable Machine;
- `tm_imports/index.json`: every sample's parent recording, window rank and offset;
- `audio_dataset/gtm_samples/<class>/<parent>S1.wav`: the same windows as WAV, to listen to;
- `audio_dataset/manifests/gtm_segment_rows.csv`: one row per window (parent, class, offset,
  licence, sha256).

Every parent is checked against the manifest split before anything is written, and
`tests/test_augmentation.py` re-checks that each sample names a training parent of the same
class. Validation and test recordings are never read. `--max-per-class N` caps a class; the
recordings are then taken in a fixed hash order so the cap does not favour one source.

The previous imports (v2, 140 per class in audio-id order) used 764 FSD50K clips but only 66
of the 253 UrbanSound8K training clips, which is one reason that model generalised poorly.
The older 25 Sep WAV cut is withdrawn; see `audio_dataset/manifests/archive_split_v1/README.md`.

## Current export

The served model (26 Sep) was trained from `--max-per-class 140`: 1,400 samples, the most this
browser's Teachable Machine would train without stalling at "Preparing training data". On the
test split it scores 0.493 accuracy, 0.473 macro-F1 and 0.591 critical-class recall
(`gtm_model/gtm_metrics.json`), well below the SRS targets; `documentation/MODEL_EVALUATION.md`
has the full history.

## Train and export

```bash
.venv/bin/python -m pip install -r requirements-training.txt   # Playwright
.venv/bin/python tools/train_gtm_browser.py --out gtm_model/candidates/<name>
.venv/bin/tensorflowjs_converter --input_format tfjs_layers_model --output_format keras \
  gtm_model/candidates/<name>/model.json gtm_model/candidates/<name>/gtm_model.h5
cp gtm_model/frontend_config.json gtm_model/candidates/<name>/
.venv/bin/python tools/evaluate_gtm.py --split val --gtm-dir gtm_model/candidates/<name>
```

The automation opens `https://teachablemachine.withgoogle.com/train/audio` without signing in,
creates the ten classes (Background Noise first, names exactly as in `config/classes.json`),
imports each ZIP, trains with Teachable Machine's defaults and downloads the TensorFlow.js
export. Screenshots of each step go to `screenshots/gtm/`. To do it by hand, follow the same
order in the browser and use each class's Upload control with its ZIP.

Candidates are compared on the validation split; only the chosen one is copied into
`gtm_model/` and scored once on test (`tools/evaluate_gtm.py --split test`).

## How the server runs the model

`gtm_model/frontend_config.json` records the input shape, z-score normalisation and
`window_aggregation`. With `energy_weighted`, the server scores every one-second window of a
clip at half-second hops and averages the scores weighted by each window's energy; with
`loudest`, it scores only the loudest second. The frames are computed in Python to match the
browser's analyser (Blackman window, 2048-point FFT, 1024 hop, dB). Same-clip agreement with
the browser has not been measured, so `frontend_verified` stays false.

Never substitute Python-model scores for a Teachable Machine result.
