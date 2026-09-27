# Submission checklist

Deadline for NextWave AI and ML: **29 September 2026**, 16:00 Pakistan time (check your
centre's row in the official table). Aim to be finished by **29 Sep 04:00**, twelve hours early.

Status key: ✔ done in the repository · ☐ still to do · 👤 needs a team member (cannot be
done by code).

## SRS §1.10 item 16

| Item | Status | Where / what is missing |
|---|---|---|
| Project report | ✔ draft | `PROJECT_REPORT.md`; 👤 add team names, roll numbers, task allotment (§9) |
| Public GitHub URL | ✔ | <https://github.com/build-with-fun/TechWiz7> (public, pushed 27 Sep), linked in the README |
| Complete source code | ✔ | this repository |
| Training / validation / testing datasets | 👤 | audio is not in Git (size, licences). Decide how to share it (e.g. a Drive/Kaggle archive with `DATA_ATTRIBUTION.md`); review the 36 Sampling+ clips first |
| Dataset metadata | ✔ | `audio_dataset/manifest.csv`, `data/splits/split.json`, `DATA_DICTIONARY.md` |
| Python model | ✔ | `python_models/best/` (+ `tools/fetch_pretrained.py` for CNN14) |
| GTM model | ✔ export · 👤 evidence | `gtm_model/`; 👤 a team member trains the project in their own TM session, saves the project link and screenshots of every class |
| Preprocessing and feature scripts | ✔ | `audio_preprocessing/`, `feature_extraction/`, `augmentation/` |
| Alert rules | ✔ | `alert_rules/` |
| Model comparison report | ✔ | `reports/MODEL_COMPARISON.md`, `reports/model_comparison.csv` |
| Installation / execution instructions | ✔ | `README.md` |
| Deployment URL | ✔ | <https://shelf-starlight-subfloor.ngrok-free.dev/login> (laptop + ngrok, 27 Sep). 👤 keep the laptop on, both terminals open, and `tools/uptime_probe.py` running until judging ends; then `--summary` for NFR 5 |
| Demonstration video (MP4) | 👤 | record using `DEMO_GUIDE.md` |
| Technical blog (≥ 2,000 words) | ✔ | published: <https://dev.to/buildwithfun/ai-voice-analysis-13oh> (2,226 words), linked in the README |
| AI_USAGE.md | ✔ draft | 👤 each member adds what they reviewed and changed, and signs |
| Team contribution record | 👤 | `documentation/TEAM_CONTRIBUTION_RECORD.md` |

## Integrity items (SRS §1.8)

- 👤 Every member can explain their modules (`documentation/VIVA_PACK.md`).
- 👤 Commits from all members. Do not backdate or fake history; if the history shows one
  author, say so honestly in the contribution record.
- ✔ Development log (`documentation/devlog.md`); 👤 members add their own entries.
- ✔ No hard-coded predictions, no generative-AI API at inference (`tests/test_model_independence.py`).

## Before you zip / push

- [x] `pytest -q` green; paste the result into `TEST_PLAN.md` — 497 passed, 0 failed (27 Sep)
- [x] `python tools/render_uml.py --check` PASS
- [x] `grep -rn "{{" README.md PROJECT_REPORT.md TEST_PLAN.md documentation/` finds no unfilled placeholders
- [x] No secrets: `.env` is not committed; `git grep -n "SECRET_KEY="` shows only `.env.example` and a `...` placeholder in the Dockerfile comment (27 Sep)
- [ ] Fresh clone: follow `README.md` from scratch in a new folder and upload one sample clip (27 Sep: `requirements.txt` was missing `transformers`, now fixed; a clean install is still untested)
- [ ] Record the video; export 1080p MP4 named per the portal's rule
- [ ] Share links (report, blog, video, dataset) set to "anyone with the link can view"

## Timeline (Pakistan time)

| When | Task |
|---|---|
| 27 Sep | Team review of code and docs; AI_USAGE sign-off; contribution record; GTM re-training in a member's own account with screenshots |
| 28 Sep morning | Deployment or local-only decision; dataset archive uploaded; fresh-clone test |
| 28 Sep afternoon | Video recording and editing |
| 28 Sep evening | Blog published; README links filled; final `pytest` |
| 29 Sep 04:00 | Package complete; submit through the centre's channel |
| 29 Sep 16:00 | Official deadline |
