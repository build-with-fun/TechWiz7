# SonicSentinel dataset manifest schema (v1.0.0)

**Status: frozen.** Changing columns means updating every script that reads the manifest.

The manifest lists every audio file in the project, one row per file on disk. It records
where each file came from (SRS Step 1, FR xvii) and is the input to
`audio_dataset/build_split.py`.

## Files

| Path | Role |
|---|---|
| `audio_dataset/manifest.csv` | **The manifest.** One row per audio file, built by `audio_dataset/scripts/assemble_manifest.py`. |
| `data/splits/split.json` | **The frozen split.** Only `audio_dataset/build_split.py` writes it. This is the train/val/test assignment everything uses. |
| `audio_dataset/manifest_with_split.csv` | The manifest with `dataset_split` filled in, written by the builder. Don't edit it by hand. |

## Columns (exact names and order; don't rename or reorder)

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
| 12 | `recording_environment` | yes | `indoor`\|`outdoor`\|`vehicle`\|`studio`\|`synthetic`\|`unspecified` | SRS Step 1 and the §1.5 variety requirement. Use `unspecified` when the source doesn't say; don't guess. |
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

## Lineage rules (checked by the builder)

1. **One recording, one split.** Every segment and augmented copy of a recording goes to
   the split of its original (`parent_audio_id`). Otherwise a 30-second recording cut into
   six segments could end up with five in train and one in test, which is leakage.
2. **Augmented rows aren't originals.** They never count toward the 3,000 originals or the
   per-split totals.
3. **Neither model is trained on val or test**, Python or Teachable Machine.
4. **A missing parent is an error.** An augmented clip whose original can't be found stops
   the build.
5. **The order files arrive in doesn't matter.** Assignment depends only on the IDs (and
   source groups), not on when a file was added.

## Freezing procedure

```bash
# 1. validate the manifest
.venv/bin/python audio_dataset/verify_split.py --check-manifest

# 2. freeze the split (writes data/splits/split.json, refuses to overwrite without --force)
.venv/bin/python audio_dataset/build_split.py --manifest audio_dataset/manifest.csv --strict

# 3. audit it independently
.venv/bin/python audio_dataset/verify_split.py --check-split --check-files
```

`--strict` requires exactly 300 originals per class (giving 2100/450/450) and fails
otherwise.
