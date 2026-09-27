# SonicSentinel AI data dictionary

Version 1.0.0. The column list and order are fixed by `audio_dataset/manifest_schema.md`.

This describes **every column of `audio_dataset/manifest.csv`**, the index of every audio
file in the project, one row per file on disk. With it you can take a claim from the report
("300 originals per class", "no test clip was used in training", "this clip is CC-BY") and
trace it to a file, a licence and a hash.

The manifest isn't edited by hand. `audio_dataset/scripts/assemble_manifest.py` builds it
from per-source CSVs, and only `audio_dataset/build_split.py` assigns the split. Both give
the same output every time they are run.

---

## 1. Schema columns (in order)

| # | Column | Type | Required by | Description |
|---|--------|------|-------------|-------------|
| 1 | `audio_id` | string `SS-<CODE>-<NNNN>` | SRS FR xvii | Primary key. `CODE` is the class's 3-letter code from `config/classes.json` (MAC, GLA, ALM, HOR, ANI, GUN, SCR, AGG, HAL, BGN). Unique across the corpus. A derived sample adds `S<n>` to its parent's ID (e.g. `SS-GUN-0007S2`), so you can see where it came from. |
| 2 | `filename` | relative path | SRS Step 1 | Path **relative to `audio_dataset/`**, e.g. `originals/gunshot/SS-GUN-0007.wav`, so the repo works wherever it is cloned. |
| 3 | `class_label` | one of 10 values | SRS Step 1 | One of the ten classes, spelled exactly as in the SRS. A source clip with several labels goes to exactly one class (see §3). |
| 4 | `source` | string | SRS Step 1 | Where the audio came from, e.g. `FSD50K (Freesound clip <id>)`, `ESC-50 ...`, `UrbanSound8K ...`, or `procedural_synthesis:<script>` for audio we generated. |
| 5 | `source_url` | URL | SRS FR xvii | Link to the original (the Freesound page), or the generator script for our own audio. |
| 6 | `licence` | string | SRS integrity rules | The licence of this particular file, from the source's own per-clip metadata. The dataset has `CC-BY-4.0`, `CC0-1.0`, `CC-BY-3.0` and `Sampling+-1.0`; our generated audio is `CC0-1.0`. **There are no non-commercial (NC) licences.** |
| 7 | `author` | string | SRS FR xvii | Who recorded or uploaded the file (the Freesound username for Freesound clips). Used for `audio_dataset/licences/ATTRIBUTION.md`. |
| 8 | `date_fetched` | ISO date | SRS FR xvii | When the file was added. |
| 9 | `duration_sec` | float | SRS Step 1 | Duration in seconds, **measured from the file** with ffprobe. |
| 10 | `sampling_rate` | int Hz | SRS Step 1 | Sample rate, measured from the file. |
| 11 | `channels` | int | SRS Step 1 | Number of channels, measured from the file. |
| 12 | `recording_environment` | `indoor` \| `outdoor` \| `vehicle` \| `studio` \| `synthetic` \| `unspecified` | SRS Step 1 | The environment of the recording, for the SRS 1.5 variety requirement. `unspecified` (2,123 rows) means the source didn't say, and we didn't guess. For generated audio it is the room the generator simulated. |
| 13 | `recording_device` | string | SRS Step 1 | Recording device. Freesound clips usually say `unspecified` because the uploader didn't give one. Generated rows name the generator. |
| 14 | `approximate_distance` | `near` \| `medium` \| `far` \| `n/a` | SRS Step 1 | Rough distance from source to microphone. Real clips are `n/a` (not stated); generated rows use the band that was simulated. |
| 15 | `original_or_augmented` | `original` \| `augmented` | SRS Step 1 | `original` is a separate recording (real or generated). `augmented` is a segment or an augmented copy. **Only `original` rows count toward the 3,000 minimum**, as the SRS requires. |
| 16 | `parent_audio_id` | string or empty | SRS Step 5 | For an `augmented` row, the `audio_id` of the original it came from. Empty for originals. |
| 17 | `segment_start_sec` | float or empty | SRS Step 5 | Start of the cut in the parent, in seconds. If this is set, `build_split.py` treats the row as a segment that takes its parent's split. |
| 18 | `segment_end_sec` | float or empty | SRS Step 5 | End of the cut in the parent, in seconds. |
| 19 | `sha256` | hex string | integrity | SHA-256 of the file. Proves a file is the one that was checked, and catches two "different" originals that are really the same audio. |
| 20 | `dataset_split` | `train` \| `val` \| `test` \| empty | SRS Step 1 | **Always empty in the source rows.** Only `audio_dataset/build_split.py` sets it (and writes `data/splits/split.json`). If two things assigned splits, test data could leak into training, so the verifier fails the whole dataset when a source row already has one. |

## 2. Extra columns

These aren't in the SRS field list, but they make it possible to re-check a row later.
`build_split.py` keeps them in its output. Some only apply to certain sources.

| Column | Description |
|--------|-------------|
| `freesound_id` | The clip's Freesound ID, for real rows. |
| `fsd50k_split` | Which FSD50K release split (`dev`/`eval`) the clip came from. Just for information; our split is decided by `build_split.py`. |
| `fsd50k_labels` | The clip's labels in the source dataset, so our class choice can be checked. |
| `licence_url` | The licence URL from the source metadata, before it was normalised into `licence`. |
| `environment_basis` | How `recording_environment` was decided: `stated` (in the uploader's description), `inferred_from_tags` or `unmatched`. Shows which environment values are stated and which are guesses from tags. |
| `fetch_batch` | Which download run the row came from (`fsd50k_real_v1`, `esc50_v1`, `urbansound8k_v1`, `fsd50k_dev_topup_v1`, `fsd50k_eval_recovered_v1`); empty for generated audio. |
| `audio_provenance` | `synthetic` for generated audio, empty for real recordings. |
| `parent_class_label` | Class of the parent, on derived rows (must match `class_label`; the split builder rejects a mismatch). |

Generated rows also carry the settings used to make them (`seed`, `sim_*`, `voice_name`,
`phrase` and so on).

## 3. Assignment and counting rules

- **One clip, one class.** A clip with several labels goes to the first class in
  `audio_dataset/scripts/class_label_map.json`'s `assignment_priority` that matches.
  Otherwise one recording could count as an original in two classes.
- **Unique originals.** Rows with `original_or_augmented == original`, counted per class.
  Segments and augmented copies never count. No two originals may share a `sha256`.
- **Real vs generated.** Rows whose `source` starts with `procedural_synthesis` (375: the
  300 help phrases and 75 Aggression/Panic Scream clips) were generated by our own code.
  The other 2,625 are real recordings from ESC-50, UrbanSound8K and FSD50K. The two counts
  are always reported separately.
- **Split.** 70/15/15 per class over originals, by `build_split.py`, keeping clips from the
  same source recording together and using a fixed seed so it can be reproduced. Derived
  rows get their parent's split.

## 4. Where the report's numbers come from

| Report claim | File to check |
|---|---|
| rows, originals, per-class counts | `audio_dataset/manifests/corpus_statistics.json` |
| licence mix, no NC licences | `corpus_statistics.json`, cross-checked by `verify_dataset.py` |
| real vs generated per class | `corpus_statistics.json` (`originals_real_field_recording` vs `originals_synthetic`) |
| how many clips the source had per class | `audio_dataset/manifests/fsd50k_fetch_stats.json` |
| the split | `data/splits/split.json` |
| no leakage, hashes, stratification | output of `verify_dataset.py --strict` and `verify_split.py --all` |
