# Audio sources and attribution

`audio_dataset/manifest.csv` is the per-clip attribution record. Each of its 3,000 rows
has an audio ID, source, author, licence, split and SHA-256 hash. The 2,625 clips from
outside sources also have a source URL and licence URL. The 375 generated clips name the
script that made them in `source` and have no URL. This page is a summary; the manifest
holds the actual credits.

| Licence in manifest | Clips |
|---|---:|
| `CC-BY-4.0` | 1,539 |
| `CC0-1.0` | 893 |
| `CC-BY-3.0` | 532 |
| `Sampling+-1.0` | 36 |

The recordings come from ESC-50, UrbanSound8K, FSD50K/Freesound and our own generated
audio. If any audio is shared for judging, its `author`, `source_url`, `licence` and
`licence_url` must go with it. The 36 `Sampling+-1.0` clips need to be checked one by one
before the dataset or a sample bundle is published. This repository doesn't establish any
legal clearance or permission from the original recordists.

The audio and the Teachable Machine import ZIPs are not in Git. The exported model weights
are, but that doesn't give permission to redistribute the training clips. The dataset is
shared separately on Google Drive (see the README); check the files against
`audio_dataset/manifest.csv` with `audio_dataset/scripts/verify_dataset.py`.
