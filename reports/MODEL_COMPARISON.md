# Model comparison on unseen test recordings

Generated 2026-09-28T11:13:34 UTC by `tools/build_comparison_report.py` from 449 frozen test recordings (1 failed to analyse). Full rows: `reports/model_comparison.csv`.

- Python model: `ensemble(ast+clap+cnn14)` v3.1.0-ensemble_current
- Teachable Machine model: vtm-20260928T0918-ew

| Measure | Value |
|---|---:|
| Python top-1 accuracy | 93.3% |
| GTM top-1 accuracy | 57.2% |
| Models agree on the class | 59.7% |
| Accuracy when they agree | 95.5% |
| Sent to manual review | 270 |
| Decided automatically | 179 |
| Accuracy of automatic decisions | 97.2% |
| Alerts raised (tracker reset per clip) | 197 |
| Median end-to-end time per clip | 1776.816 ms |

## Per class

| Class | Clips | Python correct | GTM correct | Agree | Sent to review |
|---|---:|---:|---:|---:|---:|
| Aggression | 44 | 39 | 16 | 19 | 37 |
| Alarm or Siren | 45 | 42 | 13 | 16 | 37 |
| Animal Sound | 45 | 40 | 17 | 19 | 36 |
| Background Noise | 45 | 41 | 24 | 25 | 27 |
| Glass Breaking | 45 | 43 | 28 | 28 | 32 |
| Gunshot | 45 | 44 | 32 | 32 | 24 |
| Machinery Fault | 45 | 43 | 28 | 28 | 23 |
| Panic Scream | 45 | 40 | 22 | 24 | 29 |
| Person Asking for Help | 45 | 45 | 45 | 45 | 4 |
| Vehicle Horn | 45 | 42 | 32 | 32 | 21 |

## Consistency status counts

- Acceptable Match: 132
- Model Disagreement: 170
- Strong Match: 43
- Uncertain Result: 18
- Weak Match: 86

## Largest disagreements

Sorted by the absolute top-class confidence difference.

| Audio ID | True class | Python | GTM | Difference | Explanation |
|---|---|---|---|---:|---|
| SS-ALM-0047 | Alarm or Siren | Alarm or Siren 0.1790 | Glass Breaking 0.9561 | 0.7771 | Python Alarm or Siren (0.18) vs GTM Glass Breaking (0.96); Python correct. True class Alarm or Siren ranked #1 by Python and #7 by GTM. |
| SS-BGN-0080 | Background Noise | Animal Sound 0.9750 | Machinery Fault 0.2244 | 0.7507 | Python Animal Sound (0.98) vs GTM Machinery Fault (0.22); neither model correct. True class Background Noise ranked #2 by Python and #6 by GTM. |
| SS-GUN-0047 | Gunshot | Gunshot 0.9822 | Background Noise 0.2728 | 0.7093 | Python Gunshot (0.98) vs GTM Background Noise (0.27); Python correct. True class Gunshot ranked #1 by Python and #2 by GTM. |
| SS-ALM-0665 | Alarm or Siren | Alarm or Siren 0.9856 | Background Noise 0.2791 | 0.7065 | Python Alarm or Siren (0.99) vs GTM Background Noise (0.28); Python correct. True class Alarm or Siren ranked #1 by Python and #5 by GTM. |
| SS-AGG-0651 | Aggression | Aggression 0.9800 | Background Noise 0.3032 | 0.6768 | Python Aggression (0.98) vs GTM Background Noise (0.30); Python correct. True class Aggression ranked #1 by Python and #8 by GTM. |
| SS-ANI-0643 | Animal Sound | Animal Sound 0.9413 | Aggression 0.2748 | 0.6665 | Python Animal Sound (0.94) vs GTM Aggression (0.27); Python correct. True class Animal Sound ranked #1 by Python and #2 by GTM. |
| SS-AGG-0538 | Aggression | Aggression 0.9231 | Animal Sound 0.2595 | 0.6636 | Python Aggression (0.92) vs GTM Animal Sound (0.26); Python correct. True class Aggression ranked #1 by Python and #2 by GTM. |
| SS-ALM-0675 | Alarm or Siren | Alarm or Siren 0.9838 | Aggression 0.3212 | 0.6626 | Python Alarm or Siren (0.98) vs GTM Aggression (0.32); Python correct. True class Alarm or Siren ranked #1 by Python and #5 by GTM. |
| SS-ALM-0037 | Alarm or Siren | Alarm or Siren 0.9678 | Animal Sound 0.3145 | 0.6533 | Python Alarm or Siren (0.97) vs GTM Animal Sound (0.31); Python correct. True class Alarm or Siren ranked #1 by Python and #3 by GTM. |
| SS-BGN-0065 | Background Noise | Background Noise 0.9804 | Machinery Fault 0.3285 | 0.6519 | Python Background Noise (0.98) vs GTM Machinery Fault (0.33); Python correct. True class Background Noise ranked #1 by Python and #2 by GTM. |
| SS-GLA-0097 | Glass Breaking | Glass Breaking 0.9325 | Machinery Fault 0.2892 | 0.6433 | Python Glass Breaking (0.93) vs GTM Machinery Fault (0.29); Python correct. True class Glass Breaking ranked #1 by Python and #4 by GTM. |
| SS-AGG-0053 | Aggression | Aggression 0.8584 | Animal Sound 0.2180 | 0.6405 | Python Aggression (0.86) vs GTM Animal Sound (0.22); Python correct. True class Aggression ranked #1 by Python and #5 by GTM. |
| SS-BGN-0542 | Background Noise | Background Noise 0.9818 | Gunshot 0.3491 | 0.6327 | Python Background Noise (0.98) vs GTM Gunshot (0.35); Python correct. True class Background Noise ranked #1 by Python and #2 by GTM. |
| SS-MAC-0571 | Machinery Fault | Machinery Fault 0.9375 | Aggression 0.3066 | 0.6309 | Python Machinery Fault (0.94) vs GTM Aggression (0.31); Python correct. True class Machinery Fault ranked #1 by Python and #2 by GTM. |
| SS-BGN-0680 | Background Noise | Background Noise 0.9045 | Glass Breaking 0.2744 | 0.6301 | Python Background Noise (0.90) vs GTM Glass Breaking (0.27); Python correct. True class Background Noise ranked #1 by Python and #3 by GTM. |
