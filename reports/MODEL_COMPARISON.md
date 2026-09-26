# Model comparison on unseen test recordings

Generated 2026-09-26T19:32:15 UTC by `tools/build_comparison_report.py` from 450 frozen test recordings (0 failed to analyse). Full rows: `reports/model_comparison.csv`.

- Python model: `ast_embedding+logreg` v2.0.0-ast_current
- Teachable Machine model: vtm-20260926T1246-ew

| Measure | Value |
|---|---:|
| Python top-1 accuracy | 89.1% |
| GTM top-1 accuracy | 49.3% |
| Models agree on the class | 49.3% |
| Accuracy when they agree | 94.6% |
| Sent to manual review | 403 |
| Decided automatically | 47 |
| Accuracy of automatic decisions | 100.0% |
| Alerts raised (tracker reset per clip) | 179 |
| Median end-to-end time per clip | 1847.035 ms |

## Per class

| Class | Clips | Python correct | GTM correct | Agree | Sent to review |
|---|---:|---:|---:|---:|---:|
| Aggression | 45 | 38 | 16 | 17 | 45 |
| Alarm or Siren | 45 | 38 | 6 | 8 | 44 |
| Animal Sound | 45 | 39 | 5 | 7 | 45 |
| Background Noise | 45 | 40 | 26 | 26 | 42 |
| Glass Breaking | 45 | 41 | 23 | 23 | 44 |
| Gunshot | 45 | 43 | 22 | 21 | 45 |
| Machinery Fault | 45 | 41 | 26 | 26 | 45 |
| Panic Scream | 45 | 37 | 28 | 27 | 37 |
| Person Asking for Help | 45 | 45 | 44 | 44 | 22 |
| Vehicle Horn | 45 | 39 | 26 | 23 | 34 |

## Consistency status counts

- Acceptable Match: 22
- Model Disagreement: 39
- Strong Match: 26
- Uncertain Result: 321
- Weak Match: 42

## Largest disagreements

Sorted by the absolute top-class confidence difference.

| Audio ID | True class | Python | GTM | Difference | Explanation |
|---|---|---|---|---:|---|
| SS-HOR-0103 | Vehicle Horn | Vehicle Horn 0.9944 | Background Noise 0.1972 | 0.7972 | Python Vehicle Horn (0.99) vs GTM Background Noise (0.20); Python correct. True class Vehicle Horn ranked #1 by Python and #2 by GTM. |
| SS-BGN-0080 | Background Noise | Animal Sound 0.9948 | Machinery Fault 0.1977 | 0.7971 | Python Animal Sound (0.99) vs GTM Machinery Fault (0.20); neither model correct. True class Background Noise ranked #2 by Python and #5 by GTM. |
| SS-HOR-0001 | Vehicle Horn | Vehicle Horn 0.9915 | Animal Sound 0.1968 | 0.7947 | Python Vehicle Horn (0.99) vs GTM Animal Sound (0.20); Python correct. True class Vehicle Horn ranked #1 by Python and #4 by GTM. |
| SS-ANI-0645 | Animal Sound | Animal Sound 0.9764 | Aggression 0.1921 | 0.7843 | Python Animal Sound (0.98) vs GTM Aggression (0.19); Python correct. True class Animal Sound ranked #1 by Python and #2 by GTM. |
| SS-MAC-0555 | Machinery Fault | Machinery Fault 0.9906 | Animal Sound 0.2083 | 0.7822 | Python Machinery Fault (0.99) vs GTM Animal Sound (0.21); Python correct. True class Machinery Fault ranked #1 by Python and #2 by GTM. |
| SS-MAC-0544 | Machinery Fault | Machinery Fault 0.9851 | Animal Sound 0.2082 | 0.7769 | Python Machinery Fault (0.99) vs GTM Animal Sound (0.21); Python correct. True class Machinery Fault ranked #1 by Python and #7 by GTM. |
| SS-HOR-0657 | Vehicle Horn | Vehicle Horn 0.9984 | Aggression 0.2411 | 0.7573 | Python Vehicle Horn (1.00) vs GTM Aggression (0.24); Python correct. True class Vehicle Horn ranked #1 by Python and #2 by GTM. |
| SS-SCR-0714 | Panic Scream | Vehicle Horn 0.9589 | Alarm or Siren 0.2065 | 0.7524 | Python Vehicle Horn (0.96) vs GTM Alarm or Siren (0.21); neither model correct. True class Panic Scream ranked #3 by Python and #2 by GTM. |
| SS-HOR-0576 | Vehicle Horn | Vehicle Horn 0.9756 | Glass Breaking 0.2276 | 0.7480 | Python Vehicle Horn (0.98) vs GTM Glass Breaking (0.23); Python correct. True class Vehicle Horn ranked #1 by Python and #2 by GTM. |
| SS-ANI-0558 | Animal Sound | Animal Sound 0.9910 | Panic Scream 0.2433 | 0.7477 | Python Animal Sound (0.99) vs GTM Panic Scream (0.24); Python correct. True class Animal Sound ranked #1 by Python and #2 by GTM. |
| SS-ANI-0598 | Animal Sound | Animal Sound 0.9841 | Machinery Fault 0.2454 | 0.7387 | Python Animal Sound (0.98) vs GTM Machinery Fault (0.25); Python correct. True class Animal Sound ranked #1 by Python and #3 by GTM. |
| SS-GUN-0061 | Gunshot | Gunshot 0.9872 | Glass Breaking 0.2502 | 0.7370 | Python Gunshot (0.99) vs GTM Glass Breaking (0.25); Python correct. True class Gunshot ranked #1 by Python and #2 by GTM. |
| SS-ANI-0101 | Animal Sound | Animal Sound 0.9707 | Background Noise 0.2377 | 0.7330 | Python Animal Sound (0.97) vs GTM Background Noise (0.24); Python correct. True class Animal Sound ranked #1 by Python and #2 by GTM. |
| SS-ALM-0669 | Alarm or Siren | Alarm or Siren 0.9975 | Person Asking for Help 0.2705 | 0.7270 | Python Alarm or Siren (1.00) vs GTM Person Asking for Help (0.27); Python correct. True class Alarm or Siren ranked #1 by Python and #5 by GTM. |
| SS-HOR-0552 | Vehicle Horn | Vehicle Horn 0.9927 | Glass Breaking 0.2689 | 0.7238 | Python Vehicle Horn (0.99) vs GTM Glass Breaking (0.27); Python correct. True class Vehicle Horn ranked #1 by Python and #3 by GTM. |
