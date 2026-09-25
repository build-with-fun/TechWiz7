# Teachable Machine upload package — SonicSentinel AI

These ten zips contain the **GTM training samples**: 5,230 two-second, 16 kHz mono
segments cut ONLY from the frozen **train**-split recordings of the Python dataset
(SRS Step 9: "GTM training samples must originate from the same underlying training
recordings used by the Python model"). No validation or test recording contributed a
sample, so neither model has ever seen a val/test clip — proven by
`audio_dataset/manifests/gtm_segment_rows.csv` (parent lineage) and
`audio_dataset/scripts/verify_dataset.py --strict` (26/28 → 28/28 checks incl. leakage).

## How to train (browser)
1. Open https://teachablemachine.withgoogle.com/train/audio → New Audio Project.
2. Create the ten classes with EXACTLY these names (same class names as the Python model,
   SRS Step 9):
   Machinery Fault, Glass Breaking, Alarm or Siren, Vehicle Horn, Animal Sound,
   Gunshot, Panic Scream, Aggression, Person Asking for Help, Background Noise
3. For each class: Upload Audio Samples → choose the matching zip's contents
   (zip is just a container; extract and multi-select the wavs).
4. Train (default epochs are fine; TM audio uses a 20 ms-hop log-mel frontend).
5. Export Model → Tensorflow.js → Download. The export zip must contain:
   `metadata.json`, `model.json`, `weights.bin`.
6. Place the three files in this repo's `gtm_model/` directory, then run:
   `.venv/bin/python scripts/capture_gtm_frontend_config.py`  (writes frontend_config.json)
   and the frontend verification script to produce `frontend_verification.json`.

Backlight: TM caps per-class samples; if an upload is refused, use the stratified
sub-list in `index.csv` order (first N of each class) and record the reduced count in
`gtm_metrics.json`.
