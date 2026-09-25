# AI Tool Usage Declaration — SonicSentinel AI

**Competition:** Aptech NextWave AI and ML — SonicSentinel AI (AcousticX Intelligence)
**SRS reference:** v1.0, deliverable 15, integrity rules 1.8
**Declaration date:** 2026-09-25 (updated as work proceeds)

## 1. What was used

| Tool | Purpose | Assistance requested | Files affected | Student modifications | Testing completed | Verified by |
|---|---|---|---|---|---|---|
| Anthropic Claude (chat) | Scaffolding Flask blueprints, SQLAlchemy models, pytest suites | "Write an alerts API slice with acknowledge/dismiss/escalate and tests" | `src/api/alerts_api.py`, `tests/test_alerts_api.py` | Renamed helpers, fixed `_back_href` endpoint bug, re-worded flash messages, added flash round-trip tests | `pytest -q` (403 passed at commit `2f39fa0`), manual curl smoke of every write path | Ammar |
| Google Gemini (chat) | Dataset repair scripts (blob collision repair, manifest assembly) | "Validate and reassemble a 3,000-row audio manifest with sha256 and split inheritance" | `audio_dataset/scripts/assemble_manifest.py`, `audio_dataset/scripts/verify_dataset.py` | Added GTM segment-id handling (`S<n>` suffix), split-via-parent check, report wording | `verify_dataset.py --strict` 28/28 checks on both the 3,000-row and 8,230-row manifests | Ammar |
| Xeno agent sessions | Boilerplate generation, smoke-test scripting, log tailing while training runs | "Smoke-test every API write path"; "cut 5,230 GTM segments from train split only" | `src/api/*`, `audio_dataset/scripts/cut_gtm_samples.py`, `gtm_model/upload_package/*` | Every file reviewed, edited where behavior was wrong (405s, NameErrors, PK defaults, config schemas) | Full pytest suite; curl-level verification of each endpoint including negative cases | Ammar |
| OpenAI Whisper (local, offline) | Transcribing TTS scripts for synthetic "Person Asking for Help" clips | None — the clips themselves are the model's input, not Whisper output | `audio_dataset/synthetic/` (scripts only) | Synthetic clips were generated, then reviewed by ear and by waveform | Spot-checked via quality-verdict pipeline runs | Ammar |

## 2. What was NOT used, and never will be

- **No external generative-AI API participates in any runtime decision.** The final
  sound classification comes exclusively from (a) the locally trained Python model
  (`python_models/best/`, loaded by `src/services/pipeline.py`) and (b) the Google
  Teachable Machine model trained in the browser and exported to `gtm_model/`.
  `tests/test_model_independence.py` proves by inspection that `GtmModelPredictor.predict`
  structurally cannot receive the Python model's prediction or confidence.
- No hard-coded predictions, invented confidences, hidden decision endpoints, or
  hard-coded test answers anywhere in `src/`, `python_models/`, or `gtm_model/`.
- No test answers embedded in source: every test in `tests/` exercises real code paths
  through the public Flask/HTTP or module surface.

## 3. Declaration of understanding

Every AI-assisted file above was read line-by-line, modified, and exercised before
commit. The team can explain any function, class, preprocessing step, feature, or model
component on demand (SRS 1.8 rule 4). Where an AI suggestion was wrong — e.g. the
flat-JSON overwrite of `config/thresholds.json`, the missing `_back_href` endpoint
argument, the `Event.flagged_for_investigation` attribute that lives on `AudioFile` —
the error and its fix are recorded in `documentation/devlog.md`.