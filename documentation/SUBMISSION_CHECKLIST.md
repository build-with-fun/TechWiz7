# Submission checklist

Deadline for NextWave AI and ML: **29 September 2026**, 16:00 Pakistan time (check your
centre's row in the official table). Aim to finish by **29 Sep 04:00**, twelve hours early.

"Done" means it is in the repository. "Team" marks things only a team member can do.

## SRS §1.10 item 16

| Item | Status | Where / what is missing |
|---|---|---|
| Project report | Done (draft) | `PROJECT_REPORT.md`; team and task allotment filled in (§9). Team: add the team name and roll numbers |
| Public GitHub URL | Done | <https://github.com/build-with-fun/TechWiz7> (public, pushed 27 Sep), linked in the README |
| Complete source code | Done | this repository |
| Training / validation / testing datasets | Done | Google Drive: <https://drive.google.com/drive/folders/19dNe0p0zEIV5f4IOaD0WQHPgDrCQHL1B> (all 3,000 originals, the split, metadata and licences; README "Get the dataset"). Team: review the 36 Sampling+ clips before making it public |
| Dataset metadata | Done | `audio_dataset/manifest.csv`, `data/splits/split.json`, `DATA_DICTIONARY.md` |
| Python model | Done | `python_models/best/` (AST weights via `tools/fetch_pretrained.py --ast`) |
| GTM model | Done | `gtm_model/`. Project link: <https://teachablemachine.withgoogle.com/train/audio/17pC3F6eg_sY_HHF8fY8aI2M73_B87UQ_> (Drive file <https://drive.google.com/file/d/17pC3F6eg_sY_HHF8fY8aI2M73_B87UQ_/view>, anyone with the link can view); hosted model: <https://teachablemachine.withgoogle.com/models/56AmxJNhY/>; screenshots `screenshots/gtm/20260927_signed_in_*` |
| Preprocessing and feature scripts | Done | `audio_preprocessing/`, `feature_extraction/`, `augmentation/` |
| Alert rules | Done | `alert_rules/` |
| Model comparison report | Done | `reports/MODEL_COMPARISON.md`, `reports/model_comparison.csv` |
| Installation / execution instructions | Done | `README.md` |
| Deployment URL | Done | <https://shelf-starlight-subfloor.ngrok-free.dev/login> (laptop + ngrok, 27 Sep). Team: keep the laptop on, both terminals open and `tools/uptime_probe.py` running until judging ends, then run `--summary` for NFR 5 |
| Demonstration video (MP4) | Team | record it using `DEMO_GUIDE.md` |
| Technical blog (≥ 2,000 words) | Done | published: <https://dev.to/buildwithfun/ai-voice-analysis-13oh> (2,226 words), linked in the README |
| AI_USAGE.md | Done (draft) | Team: each member adds what they reviewed and changed, and signs |
| Team contribution record | Done (draft) | `documentation/TEAM_CONTRIBUTION_RECORD.md`. Team: Aimon and Khizr confirm their rows and add evidence |

## Integrity items (SRS §1.8)

- Team: every member can explain their own modules (`documentation/VIVA_PACK.md`).
- Team: commits from all members. Don't backdate or fake history; if the history shows one
  author, say so in the contribution record.
- Done: development log (`documentation/devlog.md`). Team: members add their own entries.
- Done: no hard-coded predictions and no generative-AI API at inference
  (`tests/test_model_independence.py`).

## Before you zip / push

- [x] `pytest -q` passes and the result is in `TEST_PLAN.md` (505 passed, 0 failed, 28 Sep)
- [x] `python tools/render_uml.py --check` PASS
- [x] `grep -rn "{{" README.md PROJECT_REPORT.md TEST_PLAN.md documentation/` finds no unfilled placeholders
- [x] No secrets: `.env` is not committed; `git grep -n "SECRET_KEY="` shows only `.env.example` and a `...` placeholder in the Dockerfile comment (27 Sep)
- [ ] Fresh clone: follow `README.md` from scratch in a new folder and upload one sample clip (28 Sep: `requirements.txt` was missing `transformers`, now fixed; a clean install is still untested — `requirements.txt` now also lists the CLAP weights the served ensemble needs)
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
