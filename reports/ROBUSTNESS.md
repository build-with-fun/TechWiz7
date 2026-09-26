# Robustness and confusable-sound probes

Generated 2026-09-26T08:57:37 UTC by `tools/robustness_probe.py`. Python model `cnn14_embedding+mlp v2.0.0-current`, Teachable Machine `vgtm-browser-fft-2048-43x232-zscore-v1`. These are probes on degraded or out-of-set audio, **not** test accuracy.

## A. Degraded test recordings (15 per class)

| Condition | Python acc. | GTM acc. | Python critical recall | Rejected | Sent to review |
|---|---:|---:|---:|---:|---:|
| clean | 0.74 | 0.37 | 0.81 | 0 | 135 |
| background_noise_10dB | 0.67 | 0.37 | 0.69 | 0 | 133 |
| background_noise_0dB | 0.62 | 0.29 | 0.60 | 0 | 139 |
| low_volume_-30dB | 0.55 | 0.27 | 0.78 | 40 | 103 |
| echo_rt60_0.8s | 0.68 | 0.42 | 0.76 | 0 | 129 |
| phone_device | 0.71 | 0.37 | 0.75 | 1 | 136 |
| distant_20m | 0.53 | 0.30 | 0.76 | 41 | 98 |
| partial_50pct | 0.63 | 0.32 | 0.68 | 4 | 137 |
| overlap_-6dB | 0.55 | 0.33 | 0.59 | 0 | 140 |
| mp3_64kbps | 0.73 | 0.38 | 0.80 | 0 | 131 |

## B. Sounds the models never saw (ESC-50 categories outside our classes)

| Category | Risk | Clips | Python's choices | Sent to review | Critical class without review |
|---|---|---:|---|---:|---:|
| fireworks | Gunshot | 40 | Gunshot 27, Panic Scream 11, Background Noise 2 | 38 | 2 |
| door_wood_knock | Gunshot | 40 | Aggression 32, Machinery Fault 4, Glass Breaking 3 | 40 | 0 |
| clapping | Gunshot | 40 | Aggression 30, Background Noise 6, Glass Breaking 2 | 37 | 2 |
| laughing | Panic Scream | 40 | Aggression 27, Panic Scream 10, Glass Breaking 2 | 38 | 2 |
| crying_baby | Panic Scream | 40 | Panic Scream 35, Aggression 4, Animal Sound 1 | 38 | 2 |
| church_bells | Alarm or Siren | 40 | Alarm or Siren 32, Vehicle Horn 8 | 40 | 0 |
