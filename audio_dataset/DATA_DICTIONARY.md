# SonicSentinel AI — Data Dictionary

Version 1.0.0 · owner: omar (data sourcing) · schema frozen by `audio_dataset/manifest_schema.md` (lorena)

This is the dictionary for **every column of `audio_dataset/manifest.csv`** — the single,
machine-readable index of every audio file in the project. One row = one file that exists on
disk. It is the artefact that lets an evaluator pick any claim in the report ("300 originals per
class", "the test set never trained the model", "this clip is CC-BY") and trace it to a file, a
licence and a byte count.

The manifest is *not* hand-maintained. It is assembled by
`audio_dataset/scripts/assemble_manifest.py` from per-source CSVs, and the split it feeds is
assigned only by `audio_dataset/build_split.py`. Both are deterministic: re-running them
reproduces byte-identical output.

---

## 1. Frozen columns (in schema order)

| # | Column | Type | Required by | Description |
|---|--------|------|-------------|-------------|
| 1 | `audio_id` | string `SS-<CODE>-<NNNN>` | SRS FR xvii | Primary key. `CODE` is the class's 3-letter code from `config/classes.json` (MAC, GLA, ALM, HOR, ANI, GUN, SCR, AGG, HAL, BGN). Unique across the whole corpus. A derived sample appends `S<n>` to its parent's id (e.g. `SS-GUN-0007S2`), so any file's lineage is visible in its own name. |
| 2 | `filename` | relative path | SRS Step 1 | Path **relative to `audio_dataset/`**, e.g. `originals/gunshot/SS-GUN-0007.wav`. Relative so the repo can be cloned anywhere and still verify. |
| 3 | `class_label` | enum (10 values) | SRS Step 1 | One of the ten mandatory classes, spelled exactly as in the SRS. A multi-label source clip is assigned to exactly ONE class (see §3). |
| 4 | `source` | string | SRS Step 1 (provenance) | Where the audio came from. Either `FSD50K (Freesound clip <id>)` for a real field recording, or `procedural_synthesis:<script>` for a generated file, or `derived:segment of <audio_id>` for a segment. |
| 5 | `source_url` | URL | SRS FR xvii | Directly resolvable link to the original record (Freesound page, or the generator script path). |
| 6 | `licence` | string | SRS integrity rules | The licence of THIS file, taken from the source's own per-clip metadata — not from the corpus as a whole. Only redistribution-and-reuse licences are admitted: `CC0-1.0`, `CC-BY-3.0`, `Sampling+-1.0`, or `project-generated (synthetic)` for files this project created. **No non-commercial (NC) licence appears anywhere in this dataset.** |
| 7 | `author` | string | SRS FR xvii | The recordist/uploader credited for the file (the Freesound username for FSD50K clips). Basis for `audio_dataset/licences/ATTRIBUTION.md`. |
| 8 | `date_fetched` | ISO date | SRS FR xvii | When the file entered the corpus. |
| 9 | `duration_sec` | float | SRS Step 1 | Duration in seconds, **measured from the file** with ffprobe at assembly time (never copied from a catalogue). |
| 10 | `sampling_rate` | int Hz | SRS Step 1 | Sample rate, measured from the file. |
| 11 | `channels` | int | SRS Step 1 | Channel count, measured from the file. |
| 12 | `recording_environment` | `indoor` \| `outdoor` \| `vehicle` \| `studio` \| `synthetic` | SRS Step 1 | The environment the recording represents. This column carries the SRS 1.5 variation requirement: the corpus must not be all-indoor or all-clean. For synthetic rows it is the room the generator modelled. |
| 13 | `recording_device` | string | SRS Step 1 | Capture device. Real FSD50K clips carry `unspecified` because the uploader did not state one — recorded honestly rather than guessed. Synthetic rows name the generator. |
| 14 | `approximate_distance` | `near` \| `medium` \| `far` \| `n/a` | SRS Step 1 | Approximate source-to-microphone distance band. Real clips are `n/a` (not stated by the uploader); synthetic rows set the band the generator simulated. |
| 15 | `original_or_augmented` | `original` \| `augmented` | SRS Step 1 | `original` = a distinct real capture (or a distinct generated recording). `augmented` = a derived segment or an augmented copy. **Only `original` rows count towards the ≥3,000 unique-original floor**; the SRS forbids counting augmented copies as unique. |
| 16 | `parent_audio_id` | string \| empty | SRS Step 5 (lineage) | For an `augmented` row, the `audio_id` of the original it came from. Empty for originals. |
| 17 | `segment_start_sec` | float \| empty | SRS Step 5 | Start of the cut, in seconds into the parent. Presence of this field is what makes `build_split.py` treat a row as a *segment* (which must inherit its parent's split). |
| 18 | `segment_end_sec` | float \| empty | SRS Step 5 | End of the cut, in seconds into the parent. |
| 19 | `sha256` | hex string | integrity | SHA-256 of the file's bytes. Lets anyone prove a file is the one that was verified, and lets the verifier detect two "different" originals that are actually the same audio. |
| 20 | `dataset_split` | `train` \| `val` \| `test` \| empty | SRS Step 1 | **Empty in the source rows by rule.** Assigned exclusively by `audio_dataset/build_split.py`, which writes `data/splits/split.json`. Two things assigning a split is how test data leaks into training; the verifier fails the whole dataset if a source row pre-sets this column. |

## 2. Extra columns (preserved by the split builder)

These are not in the SRS field list but are kept because they are what makes a row
re-verifiable months later. `build_split.py` preserves extra columns in its output.

| Column | Description |
|--------|-------------|
| `freesound_id` | The source clip's Freesound ID, for real rows. |
| `fsd50k_split` | Which FSD50K release split (`dev`/`eval`) the clip came from. Informational only — our split is independent and is decided by `build_split.py`. |
| `fsd50k_labels` | The clip's own labels in the source corpus, so our class assignment can be audited (this is what shows a label was *not* invented). |
| `licence_url` | The canonical licence URL as it appears in the source metadata, before it was normalised into `licence`. |
| `environment_basis` | How `recording_environment` was determined: `stated` (the uploader's own description says so) or `inferred_from_tags` or `unmatched`. This is the honesty label on the weakest column in the table — an evaluator can see at a glance which environment values are claims and which are inferences. |
| `fetch_batch` | Which fetch run produced the row (`fsd50k_real_v1`, `gtm_train_v1`, …). |
| `audio_provenance` | `real` or `synthetic`, for the generated corpus. |
| `parent_class_label` | Class of the parent, on derived rows (must equal `class_label`; the split builder rejects a mismatch). |

## 3. Assignment and counting rules (the definitions that make the numbers honest)

- **One clip, one class.** A multi-label source clip is assigned to exactly one class: the first
  class in `audio_dataset/scripts/class_label_map.json`'s `assignment_priority` whose label list it
  matches. Without this, one recording could be counted as an "original" in two classes.
- **Unique originals.** `original_or_augmented == original` rows, counted per class. Segments and
  augmented copies are never counted. Two originals must not share a `sha256`.
- **Real vs synthetic.** `fetch_batch == fsd50k_real_v1` rows are real field recordings from
  FSD50K. Everything else marked `original` is generated by this project's own code and is stated
  as synthetic in `source`. Both counts are reported separately; synthetic audio is never presented
  as a real recording.
- **Split.** Stratified 70/15/15 computed **per class over originals** by `build_split.py`, using a
  fixed seed so the result is reproducible. Derived rows inherit their parent's split exactly.

## 4. Where the numbers in the report come from

| Report claim | File that proves it |
|---|---|
| rows / originals / per-class counts | `audio_dataset/manifests/corpus_statistics.json` |
| licence mix, NC exclusion | `corpus_statistics.json`, cross-checked by `verify_dataset.py` |
| real vs synthetic per class | `corpus_statistics.json` (`originals_real_field_recording` vs `originals_synthetic`) |
| class populations available in the source | `audio_dataset/manifests/fsd50k_fetch_stats.json` |
| the split, and who is in it | `data/splits/split.json` |
| no leakage, hashes, stratification | `verify_dataset.py --strict` and `verify_split.py --all` output |
