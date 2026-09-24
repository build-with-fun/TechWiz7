# SonicSentinel — Dataset Manifest Schema (v1.0.0)

**Status: FROZEN.** `owner: lorena`. Consumers: `omar` (writes it), `taha`, `nadia`, `bilal`
(read it), `imran` (checks it), `raheem` (audits it).

The single machine-readable record of every audio file in the project. One row per file
on disk. Source of truth for provenance (SRS Step 1, FR xvii) and the input to the
split freezer (`audio_dataset/build_split.py`).

## Files

| Path | Role |
|---|---|
| `audio_dataset/manifest.csv` | **The manifest.** One row per audio file. Written by `omar`. |
| `data/splits/split.json` | **The frozen split.** Written *only* by `audio_dataset/build_split.py`. Authoritative train/val/test assignment. |
| `audio_dataset/manifest_with_split.csv` | Manifest + resolved `dataset_split` column, emitted by the builder. Derived — never edit by hand. |

## Columns (exact names and order — do not rename, do not reorder)

| # | Column | Required | Type / allowed values | Notes |
|---|---|---|---|---|
| 1 | `audio_id` | yes | `SS-<CLASS3>-<NNNN>` e.g. `SS-GUN-0007` | **Primary key. Unique across the whole dataset.** Never reused, never renumbered. |
| 2 | `filename` | yes | relative path under `audio_dataset/` | Must exist on disk. Verified by `verify_split.py --check-files`. |
| 3 | `class_label` | yes | one of the 10 exact class names below | Exact string match. No synonyms. |
| 4 | `source` | yes | free text | Where it came from: corpus name, recording session id, or generator. |
| 5 | `source_url` | no | URL | Empty only for our own recordings; then `source` must say so. |
| 6 | `licence` | yes | e.g. `CC0-1.0`, `CC-BY-4.0`, `self-recorded`, `synthetic-generated` | **Never blank.** A clip without a licence is not eligible for the dataset. |
| 7 | `author` | yes | free text | Attribution. `self` for our own, generator name for synthetic. |
| 8 | `date_fetched` | yes | `YYYY-MM-DD` | |
| 9 | `duration_sec` | yes | float seconds | Measured, not assumed. |
| 10 | `sampling_rate` | yes | int Hz | As-stored, before resampling. |
| 11 | `channels` | yes | int | 1 or 2 at source. |
| 12 | `recording_environment` | yes | `indoor`\|`outdoor`\|`vehicle`\|`studio`\|`synthetic`\|`unspecified` | SRS Step 1 + §1.5 variation requirement. `unspecified` is the honest value for a clip whose source records no environment metadata (618 of the FSD50K reals); it is never coerced to a guess. |
| 13 | `recording_device` | yes | free text | e.g. `Pixel 6a`, `Zoom H1n`, `generator`. |
| 14 | `approximate_distance` | yes | `near`\|`medium`\|`far`\|`n/a` | SRS Step 1 "approximate source distance". |
| 15 | `original_or_augmented` | yes | `original`\|`augmented` | **Only `original` counts toward the >=3,000 floor.** |
| 16 | `parent_audio_id` | cond. | `audio_id` of the source recording | Required iff `original_or_augmented == augmented`. Blank for originals. |
| 17 | `segment_start_sec` | cond. | float | Set iff this file is a fixed-duration segment of a longer recording. |
| 18 | `segment_end_sec` | cond. | float | Set together with #17. |
| 19 | `sha256` | yes | hex | Content hash, for duplicate/near-duplicate detection (FR lxxiii). |
| 20 | `dataset_split` | derived | `train`\|`val`\|`test` | **Leave blank in `manifest.csv`.** The builder fills it. |

## The 10 mandatory class names (exact, identical in both models)

```
Machinery Fault
Glass Breaking
Alarm or Siren
Vehicle Horn
Animal Sound
Gunshot
Panic Scream
Aggression
Person Asking for Help
Background Noise
```

Critical classes (recall >= 85% requirement):
`Gunshot`, `Glass Breaking`, `Panic Scream`, `Aggression`, `Person Asking for Help`.

## Lineage rules — enforced by the builder, not by trust

1. **One recording, one split.** Every segment of a recording, and every augmented copy
   of it, is forced into the split of its *original* (`parent_audio_id`).
   Rationale: a 30-second recording cut into six 5-second segments must not put five
   segments in train and one in test — that is textbook leakage.
2. **Augmentation is not data.** An `augmented` row never counts toward the 3,000 unique
   originals and never appears in the train/val/test originals count.
3. **Val and test are never trained on** by either model — Python or Teachable Machine.
4. **A missing parent is an error**, not a warning. An augmented clip with no resolvable
   original halts the build.
5. **The split is frozen before sourcing.** Assignment is decided from `audio_id` alone,
   so later arrival order of files cannot bias it.

## Freezing procedure

```bash
# 1. validate the manifest
.venv/bin/python audio_dataset/verify_split.py --check-manifest

# 2. freeze the split (writes data/splits/split.json, refuses to overwrite without --force)
.venv/bin/python audio_dataset/build_split.py --manifest audio_dataset/manifest.csv --strict

# 3. audit it independently
.venv/bin/python audio_dataset/verify_split.py --check-split --check-files
```

`--strict` requires exactly 300 originals per class (=> 2100/450/450). It fails loudly
rather than silently producing an off-spec split.
