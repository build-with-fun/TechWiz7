# AI Tool Usage Declaration — SonicSentinel AI

**Competition:** Aptech NextWave AI and ML — SonicSentinel AI (AcousticX Intelligence)
**SRS reference:** v1.0, deliverable 15, integrity rules §1.8
**Declaration date:** 2026-09-26

This file declares every use of artificial-intelligence tooling on the project, what it was
**not** used for, and our understanding of the competition's integrity rules. Each member
adds their own rows and signs at the bottom.

## 1. What was used

AI assistance was limited to **document summarization, research, and the writing of code
comments and documentation**. No AI tool produced a model, a dataset, a prediction, or any
part of the runtime decision logic.
The Ai is used in docs , summarizations , comments and image generation only.

## 2. What was NOT used, and never will be

- **No external generative-AI API participates in any runtime decision.** The final sound
  classification comes exclusively from (a) the locally trained Python model
  (`python_models/best/`, loaded by `src/services/pipeline.py`) and (b) the Google Teachable
  Machine model trained in the browser and exported to `gtm_model/`.
  `tests/test_model_independence.py` proves by inspection that `GtmModelPredictor.predict`
  structurally cannot receive the Python model's prediction or confidence.
- **No AI generated the models or the dataset.** The Python model was trained here on the
  team's own frozen 3,000-clip split (`data/splits/split.json`); the Teachable Machine model
  was trained in the browser from the same corpus. Accuracy and macro-F1 figures in
  `python_models/metrics/` and `gtm_model/gtm_metrics.json` are measured on the held-out
  test split, not written by hand or by an AI.
- **No AI generated the diagrams or screenshots.** The five UML PNGs in `diagrams/` are drawn
  by `tools/render_uml.py` (pure matplotlib, verified by its `--check` overlap test); the
  waveforms and spectrograms in the event reports come from `src/services/visuals.py`;
  `screenshots/` are captures of the running application.
- No hard-coded predictions, invented confidences, hidden decision endpoints, or
  hard-coded test answers anywhere in `src/`, `python_models/`, or `gtm_model/`.
- No test answers embedded in source: every test in `tests/` exercises real code paths
  through the public Flask/HTTP or module surface.

## 3. Declaration of understanding

Every AI-assisted file above was read line-by-line, modified, and exercised before
commit. The team can explain any function, class, preprocessing step, feature, or model
component on demand (SRS §1.8 rule 4). Where an AI suggestion was wrong — e.g. the
flat-JSON overwrite of `config/thresholds.json`, the missing `_back_href` endpoint
argument, the `Event.flagged_for_investigation` attribute that lives on `AudioFile` —
the error and its fix are recorded in `documentation/devlog.md`.

We understand that the **final decision in every case must not come from an external
generative-AI API** (SRS §1.8), and that evaluators may demand an explanation of any
function, inject a defect for us to fix, or run hidden tests against this repository.

## 4. Signatures

| Member | Roll no. | Modules owned | Date | Signature |
|---|---|---|---|---|
| 👤 Ammar | — | full pipeline, models, web console, dataset, documentation | 2026-09-26 | pending |
| 👤 | | | | |
