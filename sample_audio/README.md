# Sample audio

One permitted clip per class (all from the **test** split — never seen in training),
plus failure-path clips:

| File | Purpose |
|---|---|
| `<class>.wav` × 10 | clean evaluation clip per class |
| `silence.wav` | no-event path |
| `low_quality_quiet_tone.wav` | should get a Poor/Unusable quality verdict |
| `clipped_loud_tone.wav` | clipping detection |
| `invalid.notaudio` | upload validation error |

Sources and licences are recorded per original clip in
`audio_dataset/licences/` (attribution rows).
