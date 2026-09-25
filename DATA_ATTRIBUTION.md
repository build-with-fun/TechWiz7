# Audio source and attribution record

The local `audio_dataset/manifest.csv` is the per-clip attribution record. Each of its
3,000 rows includes an audio ID, source, author, licence identifier, split, and SHA-256
hash. The 2,625 externally sourced rows also have a source URL and licence URL; the
375 procedurally generated rows identify their generating script in `source` and have
no external URL. This document summarizes the manifest; it does not replace its
per-record credits.

| Licence identifier in manifest | Clip count |
|---|---:|
| `CC-BY-4.0` | 1,539 |
| `CC0-1.0` | 893 |
| `CC-BY-3.0` | 532 |
| `Sampling+-1.0` | 36 |

The manifest names recordings from ESC-50, UrbanSound8K, FSD50K/Freesound and local
procedural synthesis. The `author`, `source_url`, `licence` and `licence_url` fields
must travel with any audio redistributed for judging. In particular, the 36 rows
marked `Sampling+-1.0` need a clip-by-clip permission review before the corpus or a
sample bundle is published. No legal clearance or permission from original recorders
is established by this repository.

The source audio and generated Teachable Machine import ZIPs are intentionally
ignored by Git. The exported model weights are present, but their presence does not
grant permission to redistribute the training clips. If the competition requires
the audio corpus, first prepare an authorized bundle with matching attribution and
check its hashes against `audio_dataset/manifest.csv`.
