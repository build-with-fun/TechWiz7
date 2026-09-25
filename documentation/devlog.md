# Development Log — SonicSentinel AI

Format per SRS §1.8 rule 3: work completed, problems encountered, dataset changes,
model failures, code changes, tests performed. Newest first.

---

## Day 3 — 2026-09-25

### Work completed
- **All API write paths smoke-tested end-to-end** (curl, admin cookie jar):
  - Alerts: acknowledge (JSON 200 + HTML-form 303 + flash), dismiss (422 without
    reason → 200 with reason → 409 once closed), escalate (200, 422 on unknown
    severity), invalid-state transitions now flash a readable error on form posts.
  - Reviews: confirm (adopts original prediction), override with a real class name
    (`Panic Scream`, not `Scream` — 422 otherwise), re-decide → 409, form confirm → 303.
  - Admin: config PUT accepts only the *nested* schema; history records `config_edited`
    with before/after; users create/409/PATCH; retention preview + purge (legal hold
    respected); monitoring anomalies.
  - Live: session create requires consent (422 → 201 with UUID), window push without a
    GTM export returns a clean 503 `pipeline_unavailable` (correct pre-export behavior),
    stop → 200, push-after-stop → 409.
  - Reports: event HTML/JSON, period HTML, CSV/XLSX exports (xlsx verified as a real
    Excel 2007 file).
- **GTM sample set cut**: 5,230 two-second 16 kHz mono segments from **train-split
  originals only** (deterministic positions, refuses val/test parents), lineage ids
  `SS-<CODE>-<NNNN>S<n>`, rows in `audio_dataset/manifests/gtm_segment_rows.csv`.
- **TM upload package**: ten per-class zips + `index.csv` + training README under
  `gtm_model/upload_package/`.
- **tensorflowjs 4.20.0** installed for the GTM export conversion (see problems below).
- Full pytest: **403 passed** (90.7 s). Committed `2f39fa0` (API slices) and `1977fe9`
  (GTM sample set).

### Problems encountered and fixed
| Problem | Root cause | Fix |
|---|---|---|
| 500 on form POSTs to `/api/alerts/*` | `_back_href` was missing its endpoint argument | Signature `(default_endpoint, **params)`; all call sites updated |
| 405 when following the 303 | `curl -L` re-POSTs to a GET-only route | Test method: POST, read redirect URL, GET separately with the same cookie jar |
| 500 POST `/session` | `audit_login` not imported in `pages.py`; raw `User` row has no `.can` | Import added; `_home_for` uses `has_capability(role, …)` |
| `BuildError: auth.session` on `/login?next=` | Template used wrong endpoint name | `url_for('auth.session_login')` |
| "dismissd" typo | Naive `{action}d` formatting | `cannot be {action}ed` |
| 422 "Unknown role" on user create | Roles are the five from `ROLES`, not `operator` | Use `audio_reviewer`/`security_operator`/… |
| 500 `/api/admin/retention/preview` | Artifact keys were `events`/`live_audio`; legal-hold flag queried on `Event` | Keys are `event_records`/`audio`/`microphone_session_audio`; flag lives on `AudioFile` (join added) |
| 500 POST `/api/live/sessions` | `live_sessions.id` is a String(36) UUID PK with no default | `id=str(uuid.uuid4())` |
| 500 on window push | `pipeline_unavailable` missing from the error registry | Added to `ERROR_CODES`/`DEFAULT_STATUS` (503)/`DEFAULT_MESSAGES` |
| 500 `/api/reports/event/1` | `_dtm.timezone` attribute error | `import datetime as _dt`; `_dt.datetime.now(_dt.timezone.utc)` |
| **Config PUT destroyed `config/thresholds.json`** | Flat body `{"top_confidence_min": 0.60}` replaced the whole nested file; app refused to boot ("no such key min_confidence") | Restored from git; **lesson: PUTs must send the nested structure** (`alert_rules.json` references `$thresholds.confidence.min_confidence`) |
| `tensorflowjs` hard-crash at import | `tensorflow-decision-forests` CHECK-fails against protobuf 6.31.1 | Uninstalled tdf; pinned tfjs 4.20.0 `--no-deps`; wrapped the tdf import in the converter in try/except; protobuf 6.31.1 |

### Dataset changes
- Cut 5,230 GTM segments (train-split parents only) → `audio_dataset/gtm_samples/`.
- `assemble_manifest.py` now accepts segment ids (`S<n>` suffix stripped before the
  `SS-<CODE>-<NNNN>` check); `verify_dataset.py` gained a `SEGMENT_ID_RE`, a
  split-via-parent rule, and passes **28/28** checks on the combined 8,230-row
  manifest (`manifest_with_split_full.csv`): 2100/450/450 originals, 300/class,
  zero leaks into val/test, parent-agreement for all derived rows.

### Model failures
- First retrain run died silently when its parent shell closed (no process, log
  stopped at 09:59). Restarted under `setsid nohup` with `--tag repaired` so artifacts
  don't clobber the registered best model while running.
- Current best (xgboost depth=6): test accuracy 0.693, macro-F1 0.693, critical recall
  0.799 — **floors (0.85 / 0.80 / 0.85) not yet met**. Weakest: Animal Sound F1 0.46,
  Alarm or Siren 0.56, Machinery Fault 0.58, Aggression recall 0.49. Classical
  retrain (`--tag repaired`) running; **deep mel-CNN path (`train_deep --tag deep`)
  launched in parallel** — it trains on mel tensors rather than 254 summary columns,
  which is where the headroom is.

### Tests performed
- `pytest -q` → 403 passed. Manual curl matrix over every write endpoint including
  negative cases (422/409/503) and both form/JSON surfaces.

---

## Day 2 — 2026-09-24

### Work completed
- Dataset repair: 106 collided blobs re-hashed and re-cut (`repair_id_collision.py`),
  manifest reassembled to a clean 3,000 rows, frozen split rebuilt and verified
  2100/450/450 with 300 originals per class (commit `24b34f8`).
- Registered the sweep winner (`xgboost[depth=6]`) as model bundle
  `python_models/best/` + `ModelVersion` row (commit `46bb635`).
- Fixed `predict_proba` column resolution via `estimator.classes_` (commit `c36c6ba`).
- Built the eight missing console pages (commit `bb09dc3`).

### Problems encountered
- Manifest assembly rejected ids after collision repair until the validator matched the
  re-cut numbering; fixed in `repair_id_collision.py`.
- Class-code mapping (`config/classes.json`) drove the audio_id codes — kept as the
  single source of truth.

### Model failures
- First classical sweep showed Person Asking for Help at 0.99 (synthetic TTS is too
  easy) while Animal Sound sat at 0.46 — flagged for Day-3 retraining with feature
  weighting and the deep path.

---

## Day 1 — 2026-09-23

### Work completed
- Read both SRS PDFs; extracted mandatory patterns (dual-model, comparison taxonomy,
  floors, integrity rules) into `documentation/`.
- Corpus acquisition pipeline (`acquire_corpus.py`, `fetch_fsd50k.py`), synthetic clip
  generation for under-represented classes (`generate_help_clips.py`,
  `synthetic_targets.json`), trim to 300/class (`trim_to_300.py`).
- Feature extractor locked at 254 columns (`feature_extraction/features.py`,
  `config/features.json` audiofeat-1.0.0).
- App skeleton: Flask app factory, DB models, audio preprocessing package.

### Problems encountered
- FSD50K licensing: only CC-licensed clips retained; attribution written per file
  (`build_attribution.py`).
- Spec inconsistency noted: Step 16 lists five severity levels, FR lii lists four.
  Implemented **all five** with a configurable `active_scale` in
  `alert_rules/severity_levels.json`.