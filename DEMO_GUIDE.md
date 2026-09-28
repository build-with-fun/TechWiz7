# Demo guide and video script

The SRS (§1.10 item 13) asks for an MP4 showing 29 specific things. This shot list covers
them in about nine minutes, plus the checks to do before recording. Record at 1080p.
Explain what is happening, and say so when a result is wrong or uncertain; the jury will
test with their own audio anyway.

## Before recording

- [ ] `pytest -q` passes; `/api/health/ready` returns 200 (both models loaded).
- [ ] `database/init_db.py` run on a fresh database; sign in once as each role.
- [ ] Fresh browser profile, zoom 100 %, notifications off, microphone allowed for
      `localhost`, a real microphone plugged in and a phone ready to play sounds.
- [ ] Clips ready: the ten files in `sample_audio/` (test-split recordings; sources in
      `sample_audio/SOURCES.csv`), `silence.wav`, `low_quality_quiet_tone.wav`,
      `clipped_loud_tone.wav`, `invalid.notaudio`, and the prepared cases below.
- [ ] Run every prepared clip once through `python tools/predict.py <clip>` with the final
      models and note what it does. **Only show a clip as "the model disagreement case"
      etc. if you observed that behaviour on the final models.**

### Prepared cases (pick from real outputs)

`reports/model_comparison.csv` has every test recording with both predictions. Filter it:

| Case needed | Filter | Clip to use |
|---|---|---|
| Model disagreement | `class_match == mismatch` | |
| Low-confidence result | `python_confidence < 0.4` (the floor in `config/thresholds.json`), or `consistency_status == Uncertain Result` | |
| Overlapping sound | `overlap_flag == yes`, or mix two sample clips with `augmentation.transforms.overlay` | |
| Noisy case | `tools/robustness_probe.py` writes nothing to disk, so mix a clip with noise: `add_noise(y, 16000, rng, snr_db=5)` and save it | |
| High-severity / critical alert | a Gunshot or Glass clip both models agree on, uploaded 3 times in the live window, or played live 3 windows running | |
| Every class | the ten `sample_audio/<class>.wav` files (batch upload) | |

## Shot list

| Time | Screen | Show | SRS items covered |
|---|---|---|---|
| 0:00–0:30 | Title slide | Project, team, "a prototype for supervised operators, not an emergency system" | |
| 0:30–1:00 | `/login`, then dashboard (evaluator) | Sign in; dashboard tiles, recent detections, critical timeline, anomalies panel | Login, Dashboard |
| 1:00–1:40 | Upload | Upload `invalid.notaudio` (refused), `silence.wav` (refused: silent), then `clipped_loud_tone.wav` (clipping warning) | File validation, Audio quality |
| 1:40–2:30 | Upload, batch | Select all ten `sample_audio/<class>.wav` at once; per-file results | Audio upload, Every sound class |
| 2:30–3:40 | Event page for one clip | Metadata (format, rate, channels, bit depth, size), preprocessing steps, play/pause/replay/seek/volume, waveform, spectrogram | Metadata, Preprocessing, Waveform, Spectrogram |
| 3:40–4:30 | Same event | Python class + confidence + top-3; TM class + confidence + top-3; consistency status; Δ = \|Python − TM\| | Python prediction/confidence, GTM prediction/confidence, Model comparison, Confidence difference |
| 4:30–5:00 | Disagreement and low-confidence events | Both route to manual review; say why (status, reasons) | Model-disagreement case, Low-confidence case |
| 5:00–5:30 | Noisy and overlap events | Quality verdict, overlap flag, review reason | Noisy case, Overlapping-sound case |
| 5:30–6:45 | Live monitor | Tick consent, allow mic: pill shows Active. Pause (pill: Paused), resume. Play a gunshot/glass clip from the phone three windows running: consecutive counter, then the alert banner. Stop (device released) | Live microphone monitoring, Repeated detection, Real-time critical detection, Critical alert |
| 6:45–7:20 | Alerts (operator) | Acknowledge the alert; escalate another; alert history | Alert acknowledgement |
| 7:20–8:00 | Reviews (reviewer) | Listen, correct the class, comment; show the original model output is still recorded | Manual review, Reviewer override |
| 8:00–8:30 | Events + Analytics | Filter by class/date/confidence/severity; analytics FP/FN and alert response | Event history |
| 8:30–9:00 | Report | Download the event report (metadata, both models' scores, waveform, spectrogram, review) and a CSV export | Report generation |
| 9:00–9:30 | Close | Results table from `README.md`, what misses the target, next steps | |

## If something goes wrong on the day

- **Models not ready** (`/api/health/ready` 503): show the reason, run `tools/fetch_pretrained.py`
  or restore `gtm_model/gtm_model.h5`; do not demo analysis with a fallback.
- **Microphone blocked**: the pill shows "Permission denied"; show that state, then use an
  uploaded clip. Never replay an old session as if it were live.
- **A clip behaves differently from rehearsal**: say so and open its event page; the
  comparison and review routing are still worth explaining.
- **Slow first request**: the AST model loads on first use, so upload one warm-up clip
  before recording.
