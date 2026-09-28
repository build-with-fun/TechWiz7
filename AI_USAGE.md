# AI tool usage declaration: SonicSentinel AI

**Competition:** Aptech NextWave AI and ML, SonicSentinel AI
**SRS reference:** v1.0, deliverable 15, integrity rules §1.8
**Declaration date:** 2026-09-26

This file declares every use of artificial-intelligence tooling on the project, what it was
**not** used for, and our understanding of the competition's integrity rules. Each member
adds their own rows and signs at the bottom.

## 1. What was used

Until 27 Sep, AI assistance was limited to **document summarization, some testing, research,
and the writing of code comments and documentation**. On 28 Sep an AI coding assistant was
also used for model and decision-rule work, listed in section 5. No AI tool produced a
dataset or a prediction: every prediction comes from the trained Python and Teachable
Machine models, never from a generative-AI API.

We have probably taken research, code samples and parts of code from AI and from other
open-source projects as well, because we had a short time to build this.

## 2. Declaration of understanding

Every AI-assisted file above was read line by line, modified, and exercised before commit.

We understand that the **final decision in every case must not come from an external
generative AI API**, and that evaluators may demand an explanation of any function, inject a
defect for us to fix, or run hidden tests against this repository.

## 3. Signatures

| Member | Roll no. | Modules owned | Date | Signature |
|---|---|---|---|---|
| Ammar Ahmer | | Backend, model training and model making, documentation, deployment | 2026-09-26 | pending |
| Aimon | | Frontend, testing, other support | | |
| Khizr | | Suggestions, recordings, testing, implementation help | | |

## 4. Tools

| Tool | Used for |
|---|---|
| **xeno** (custom-built AI agents) | Research, docs, SRS summarization, testing, GitHub references |
| **Claude Code** | Code comments and testing; on 28 Sep, model and decision-rule work (section 5) |
| **ChatGPT, Cloudflare AI models** | Image generation, help and suggestions |

## 5. 28 Sep: model accuracy and decision rules

| Field | Entry |
|---|---|
| Tool | Claude Code |
| Purpose | Improve both models' accuracy and cut down "Uncertain Result" outcomes without accepting wrong results |
| Assistance requested | Find why most clips went to review, try new embeddings and TM settings, find why TM stalled at 1,400 samples, retrain, pick thresholds on validation, update tests and docs; also a show/hide button on the password fields |
| Files affected | `feature_extraction/clap_embeddings.py`, `feature_extraction/ensemble_embeddings.py`, `src/inference/ensemble.py`, `src/inference/consistency.py`, `src/inference/gtm_predictor.py`, `src/services/pipeline.py`, `python_models/train_ensemble.py`, `python_models/train_transfer.py`, `tools/calibrate_thresholds.py`, `tools/train_gtm_browser.py`, `tools/evaluate_gtm.py`, `tools/fetch_pretrained.py`, `tools/split_results.py`, `audio_dataset/scripts/make_gtm_imports.py`, `config/thresholds.json`, `alert_rules/`, `templates/_macros.html`, `templates/auth/`, `static/js/core.js`, `static/css/design.css`, the model files in `python_models/best/` and `gtm_model/`, the tests that cover them, and the docs with the new results |
| How the models were produced | The Python model was trained by the project's own scripts on the 2,100 training recordings; the TM model was trained inside Teachable Machine on the same recordings. Choices were made on validation and test was scored once |
| Student modifications | to be completed by the team |
| Testing completed | `pytest` suite, validation and test scoring of both models, `tools/build_comparison_report.py` on all 450 test recordings, latency check (`documentation/devlog.md`, 28 Sep) |
| Verifying team members | to be completed by the team |
