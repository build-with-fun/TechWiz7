# Baseline audit, 26 September 2026

This is the state of the repository on the morning of 26 Sep 2026, before the model and
data work recorded in `documentation/devlog.md` under that date. It supersedes nothing:
`documentation/archive/2026-09-25_project_audit.md` describes the older 25 Sep baseline, and most of its defects were
fixed later that day (see `documentation/devlog.md`). Everything below was measured on
this laptop unless it is marked as an assumption.

Machine used for every number here: 4-core / 8-thread Intel laptop CPU, 15 GB RAM,
Ubuntu (kernel 7.0), Python 3.12.14, no usable GPU (TensorFlow cannot load CUDA; PyTorch
is the CPU build).

## How to reproduce the baseline

```bash
git checkout 1673b00                      # last commit before this audit
.venv/bin/python -m pytest -q -k "not deep_models" -p no:cacheprovider
.venv/bin/python audio_dataset/scripts/verify_dataset.py --strict
.venv/bin/python tools/evaluate_gtm.py    # GTM on all 450 test clips
```

## Measured facts

| Area | Result | Evidence |
|---|---|---|
| Test suite (without the deep-model tests that download MobileNet) | 431 passed, 27 deselected, 79 s | run on 26 Sep, output in devlog |
| Dataset | 3,000 originals, 300 per class, split 2,100 / 450 / 450, no augmented rows | `audio_dataset/manifest_with_split.csv` |
| Preprocessing of the whole corpus | 3,000 / 3,000 accepted, 0 rejected, 175 s total (0.06 s per clip) | `python -m python_models.preprocess_cache` |
| Served Python model (HistGradientBoosting on 254 hand-made features) | test accuracy 0.6978, macro F1 0.6968, critical recall 0.8000 (Aggression 0.49) | `python_models/metrics/classical_metrics_hgb.json` |
| Teachable Machine model v1 | test accuracy 0.2778, macro F1 0.2755, critical recall 0.3200 | `gtm_model/gtm_metrics.json` |
| Same TM network scored on its own training frames | 0.84 (269 / 320) | diagnostic in devlog, 26 Sep |
| TM training set size | 32 one-second samples per class, 320 in total | `gtm_model/archive/v1_2026-09-25/tm_imports_index.json` |
| Git history | 34 commits, all by one identity, dated 24 Sep (7) and 25 Sep (27) | `git log --format=%ad --date=short` |

## What was wrong, in order of impact

1. **Both models are far below the SRS targets** (0.85 accuracy, 0.80 macro F1, 0.85
   critical recall). The Python model has plateaued: every hand-crafted-feature
   candidate lands between 0.70 and 0.76 on validation.

2. **The Teachable Machine model was starved of data, and trained on the wrong slice.**
   It is not a frontend bug: the exported network gets 0.84 on its own training frames,
   so FFT settings and label order match. But it saw 32 samples per class, and each one
   was the *first* second of a clip, while the server scores the *loudest* second. For
   short impulsive clips the first second is often the quiet lead-in.

3. **Preprocessing throws away the tails of impulsive sounds.** Spectral noise gating
   (strength 0.75) runs before end-trimming at 30 dB below peak. After preprocessing,
   165 clips are shorter than 0.5 s and 592 are shorter than 1 s. Glass Breaking keeps a
   median of 60% of its duration. The decay after a gunshot or a glass break is part of
   what makes it recognisable, so this may be costing accuracy. (Hypothesis, tested on
   the validation split, see the devlog.)

4. **Several labels are proxies, not the sound the SRS names.**
   - *Aggression*: at least 69 of 300 clips are door slams or thumps from FSD50K and
     50 are procedurally synthesised; only about 30 are real shouting. The class is
     really "impact or raised voice".
   - *Machinery Fault*: the clips are ordinary machines running normally (vacuum
     cleaner, chainsaw, helicopter, engine, washing machine). None is a verified fault.
   - *Person Asking for Help*: all 300 are text-to-speech from 11 synthetic voices.
     Some phrases go beyond the five SRS phrases ("Please help me. Hurry.").
   - 2,123 of 3,000 rows have `recording_environment = unspecified`, and 1,086 have an
     unspecified device.

   No amount of modelling fixes this; it needs recordings made or licensed by the team.

5. **No augmentation code exists**, although SRS FR xix and the source-code list require
   an `augmentation/` folder with noise, shift, pitch, stretch, volume and reverb.

6. **No comparison report exists.** SRS deliverable 6 asks for at least 100 unseen
   recordings with both models' full score lists, quality, severity, alert, review and
   final decision per row. `reports/` holds only the UI check and a 13-clip smoke test.

7. **Integrity gaps that code cannot close.** There is no `LICENSE` (a team decision).
   `documentation/TEAM_CONTRIBUTION_RECORD.md` is an empty template. Source files carry
   `Owner:` names (sara, taha, lorena, omar, bilal, nadia, fatima, junaid, kamran)
   that do not match any Git author and number more than the 4 to 6 team members the
   rules allow. They must be confirmed or removed by the team.

## Strengths worth keeping

- Clean separation: preprocessing, features, the two predictors, the comparison, the
  rule engine and persistence are separate modules with tests.
- The Python and TM predictors cannot see each other's output by construction, and
  `tests/test_model_independence.py` checks this.
- Label-order and feature-order drift are checked when a model bundle loads.
- The split is built at recording level and verified: `verify_dataset.py --strict`
  passes 28 / 28, including hashes and duplicate content.
- Alert rules, thresholds and retention are JSON, validated before they are saved,
  and editable by an administrator.

## Repair plan (in the order it was done)

1. Cache the serving preprocessor's output for every recording, so training and serving
   share one code path (`python_models/preprocess_cache.py`).
2. Add a transfer-learning Python model: CNN14 embeddings (PANNs, pretrained on AudioSet)
   with a small classifier chosen on validation (`python_models/train_transfer.py`).
3. Compare the current preprocessing with a gentler variant on validation only; adopt
   the better one for both models.
4. Rebuild the TM imports from every training recording using the same loudest-second
   rule as the server, retrain in Teachable Machine, re-evaluate on the 450 test clips.
5. Add `augmentation/` (training copies with lineage, plus robustness probes).
6. Generate the comparison report and a robustness report from the real pipeline.
7. Write the SRS traceability matrix from the final state, not from intentions.
8. Hand the team the decisions only they can make: licence, owners, new recordings,
   GTM project link and screenshots, deployment URL, video.
