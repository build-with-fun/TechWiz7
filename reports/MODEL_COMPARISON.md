# Model comparison on unseen test recordings

Generated 2026-09-26T22:57:41 UTC by `tools/build_comparison_report.py` from 450 frozen test recordings (0 failed to analyse). Full rows: `reports/model_comparison.csv`.

- Python model: `ast_embedding+logreg` v2.0.0-ast_current
- Teachable Machine model: vtm-20260926T2241-ew

| Measure | Value |
|---|---:|
| Python top-1 accuracy | 89.1% |
| GTM top-1 accuracy | 51.1% |
| Models agree on the class | 49.6% |
| Accuracy when they agree | 97.3% |
| Sent to manual review | 388 |
| Decided automatically | 62 |
| Accuracy of automatic decisions | 98.4% |
| Alerts raised (tracker reset per clip) | 176 |
| Median end-to-end time per clip | 1428.327 ms |

## Per class

| Class | Clips | Python correct | GTM correct | Agree | Sent to review |
|---|---:|---:|---:|---:|---:|
| Aggression | 45 | 38 | 17 | 18 | 44 |
| Alarm or Siren | 45 | 38 | 6 | 5 | 44 |
| Animal Sound | 45 | 39 | 11 | 13 | 44 |
| Background Noise | 45 | 40 | 24 | 23 | 42 |
| Glass Breaking | 45 | 41 | 16 | 15 | 43 |
| Gunshot | 45 | 43 | 35 | 33 | 43 |
| Machinery Fault | 45 | 41 | 28 | 28 | 42 |
| Panic Scream | 45 | 37 | 22 | 21 | 36 |
| Person Asking for Help | 45 | 45 | 45 | 45 | 16 |
| Vehicle Horn | 45 | 39 | 26 | 22 | 34 |

## Consistency status counts

- Acceptable Match: 26
- Model Disagreement: 47
- Strong Match: 36
- Uncertain Result: 295
- Weak Match: 46

## Largest disagreements

Sorted by the absolute top-class confidence difference.

| Audio ID | True class | Python | GTM | Difference | Explanation |
|---|---|---|---|---:|---|
| SS-AGG-0651 | Aggression | Aggression 0.9971 | Animal Sound 0.2191 | 0.7780 | Python Aggression (1.00) vs GTM Animal Sound (0.22); Python correct. True class Aggression ranked #1 by Python and #5 by GTM. |
| SS-BGN-0080 | Background Noise | Animal Sound 0.9948 | Machinery Fault 0.2462 | 0.7486 | Python Animal Sound (0.99) vs GTM Machinery Fault (0.25); neither model correct. True class Background Noise ranked #2 by Python and #8 by GTM. |
| SS-HOR-0103 | Vehicle Horn | Vehicle Horn 0.9944 | Panic Scream 0.2484 | 0.7460 | Python Vehicle Horn (0.99) vs GTM Panic Scream (0.25); Python correct. True class Vehicle Horn ranked #1 by Python and #2 by GTM. |
| SS-GLA-0534 | Glass Breaking | Glass Breaking 0.9935 | Vehicle Horn 0.2502 | 0.7432 | Python Glass Breaking (0.99) vs GTM Vehicle Horn (0.25); Python correct. True class Glass Breaking ranked #1 by Python and #2 by GTM. |
| SS-GLA-0521 | Glass Breaking | Glass Breaking 0.9977 | Gunshot 0.2562 | 0.7416 | Python Glass Breaking (1.00) vs GTM Gunshot (0.26); Python correct. True class Glass Breaking ranked #1 by Python and #2 by GTM. |
| SS-ANI-0101 | Animal Sound | Animal Sound 0.9707 | Background Noise 0.2310 | 0.7397 | Python Animal Sound (0.97) vs GTM Background Noise (0.23); Python correct. True class Animal Sound ranked #1 by Python and #4 by GTM. |
| SS-GLA-0014 | Glass Breaking | Glass Breaking 0.9651 | Alarm or Siren 0.2261 | 0.7390 | Python Glass Breaking (0.97) vs GTM Alarm or Siren (0.23); Python correct. True class Glass Breaking ranked #1 by Python and #4 by GTM. |
| SS-HOR-0022 | Vehicle Horn | Vehicle Horn 0.9995 | Gunshot 0.2654 | 0.7341 | Python Vehicle Horn (1.00) vs GTM Gunshot (0.27); Python correct. True class Vehicle Horn ranked #1 by Python and #2 by GTM. |
| SS-BGN-0647 | Background Noise | Background Noise 0.9876 | Gunshot 0.2536 | 0.7340 | Python Background Noise (0.99) vs GTM Gunshot (0.25); Python correct. True class Background Noise ranked #1 by Python and #2 by GTM. |
| SS-HOR-0576 | Vehicle Horn | Vehicle Horn 0.9756 | Gunshot 0.2438 | 0.7318 | Python Vehicle Horn (0.98) vs GTM Gunshot (0.24); Python correct. True class Vehicle Horn ranked #1 by Python and #3 by GTM. |
| SS-HOR-0552 | Vehicle Horn | Vehicle Horn 0.9927 | Gunshot 0.2621 | 0.7306 | Python Vehicle Horn (0.99) vs GTM Gunshot (0.26); Python correct. True class Vehicle Horn ranked #1 by Python and #4 by GTM. |
| SS-ANI-0021 | Animal Sound | Animal Sound 0.9430 | Gunshot 0.2153 | 0.7278 | Python Animal Sound (0.94) vs GTM Gunshot (0.22); Python correct. True class Animal Sound ranked #1 by Python and #9 by GTM. |
| SS-ANI-0558 | Animal Sound | Animal Sound 0.9910 | Panic Scream 0.2760 | 0.7150 | Python Animal Sound (0.99) vs GTM Panic Scream (0.28); Python correct. True class Animal Sound ranked #1 by Python and #2 by GTM. |
| SS-BGN-0634 | Background Noise | Background Noise 0.9250 | Person Asking for Help 0.2160 | 0.7090 | Python Background Noise (0.93) vs GTM Person Asking for Help (0.22); Python correct. True class Background Noise ranked #1 by Python and #8 by GTM. |
| SS-HOR-0657 | Vehicle Horn | Vehicle Horn 0.9984 | Aggression 0.2918 | 0.7066 | Python Vehicle Horn (1.00) vs GTM Aggression (0.29); Python correct. True class Vehicle Horn ranked #1 by Python and #2 by GTM. |
