# Teachable Machine audio training handoff

Google Teachable Machine's audio uploader accepts ZIP archives made by its **Download Samples** feature: `samples.json` containing browser-FFT frequency frames plus the matching WebM recordings. A plain ZIP of WAV files does not import. The older WAV archives in this folder remain source material, not files to feed directly into Teachable Machine.

`audio_dataset/scripts/make_gtm_imports.py` builds ten compatible archives in `tm_imports/`, using 32 distinct **train-split parent recordings per class**. It checks every parent ID against `audio_dataset/manifest.csv` and writes `tm_imports/index.json` with the lineage. Validation and test parents are excluded. The generated ZIPs and local source audio are ignored by Git; a clean clone requires the authorized corpus separately. The FFT extraction is an engineering approximation of the browser analyser and must be validated against browser predictions before treating server inference as equivalent.

To reproduce the browser model:

```bash
.venv/bin/python audio_dataset/scripts/make_gtm_imports.py
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python tools/train_gtm_browser.py
```

The automation opens `https://teachablemachine.withgoogle.com/train/audio` without account credentials, names the ten classes, imports each archive, trains the transfer model, and downloads the TensorFlow.js export to `gtm_model/`. It may take several minutes on CPU and requires Chrome plus network access. It does **not** use validation or test clips for training. If reproducing manually, start with Background Noise, create the remaining class names exactly as in `config/classes.json`, use each class's Upload control to select its matching `tm_imports/*.zip`, then train and export the TensorFlow.js model.

The included export was converted for server inference with:

```bash
.venv/bin/tensorflowjs_converter --input_format tfjs_layers_model --output_format keras \
  gtm_model/model.json gtm_model/gtm_model.h5
```

`frontend_config.json` records the exported `[43,232,1]` browser FFT input and z-score normalization. `tools/evaluate_gtm.py` scored the server path on all 450 held-out clips; the result in `gtm_model/gtm_metrics.json` is 0.2778 accuracy and 0.3200 critical-class recall. The browser/server feature comparison remains unverified, so `frontend_verification.json` does not exist. Write it only after measuring same-clip agreement. Never substitute Python-model scores for a GTM result or report the SRS thresholds as achieved.
