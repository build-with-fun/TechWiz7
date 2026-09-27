# Sample audio

One clip per class, all from the **test** split (so never seen in training), plus some clips
that should fail:

| File | Purpose |
|---|---|
| `<class>.wav` × 10 | a clean clip for each class |
| `silence.wav` | no-event path |
| `low_quality_quiet_tone.wav` | should get a Poor/Unusable quality verdict |
| `clipped_loud_tone.wav` | clipping detection |
| `invalid.notaudio` | upload validation error |

The source and licence of each clip are in `sample_audio/SOURCES.csv` and
`audio_dataset/licences/`.
