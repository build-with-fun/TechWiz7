# `database/`

This is where SonicSentinel AI keeps its data: accounts and roles, audio metadata and
hashes, both models' predictions and confidences, alerts, manual reviews, model versions
and the audit trail. The folder has the schema, the setup script, and the demo accounts
(which are public on purpose).

SRS coverage: **FR ii** (roles), **FR lxxi** (secure audio storage), **FR lxxii** (what the
database stores), **FR lxxiii–lxxiv** (duplicates and near-duplicates), **FR lxxv** (model
versions), **FR lxxvi** (audit trail), **FR lxxx** (retention), **§1.10 item 11** (folder
list and demo credentials).

---

## Files

| File | What it is |
| --- | --- |
| `schema.sql` | The DDL, generated from the SQLAlchemy models. Don't edit it by hand. |
| `init_db.py` | Creates the database, adds the demo accounts, registers model versions, prints stats. |
| `seed_credentials.json` | The demo accounts (public on purpose, see below). |
| `sonicsentinel.db` | The SQLite database (created on first run, not in Git). |

## Quick start

From the repository root, using the project virtualenv:

```bash
# 1. Create (or update) the database and add the demo accounts.
.venv/bin/python database/init_db.py --seed

# 2. See what's in it.
.venv/bin/python database/init_db.py --stats
```

That's all the storage setup there is. Running `--seed` again adds nothing and tells you so.

### Using a different location

Both paths can be changed with a flag or an environment variable, so tests and throwaway
demos don't touch the real database:

```bash
.venv/bin/python database/init_db.py --db /tmp/demo.db --storage /tmp/demo-storage --seed
```

| Flag | Environment variable | Default |
| --- | --- | --- |
| `--db` | `SST_DB_PATH` | `database/sonicsentinel.db` |
| `--storage` | `SST_STORAGE_DIR` | `data/storage` |

## Command-line options

| Option | What it does |
| --- | --- |
| *(none)* | Creates any missing tables, adds the demo accounts and registers any trained models found on disk. Doesn't delete anything. |
| `--seed` | Same as the default; seeding happens unless `--schema-only` is given. |
| `--credentials FILE` | Seeds from a different credentials file. |
| `--schema-only` | Creates the tables and seeds nothing. |
| `--emit-schema` | Rewrites `database/schema.sql` from the models and exits. |
| `--stats` | Prints row counts per table, the accounts and the registered model versions. |
| `--register-model {python,gtm}` | Registers one model version: `--version V --artifact PATH`, plus optional `--feature-version`, `--algorithm`, `--model-label`, `--activate`. |
| `--reset` | **Deletes everything** and rebuilds the tables. |
| `--yes` | Skips the `--reset` confirmation (for scripts; without it and without a terminal, `--reset` refuses). |

Examples:

```bash
# Register a Python model version and make it active (FR lxxv).
.venv/bin/python database/init_db.py --register-model python \
    --version 3.1.0 --artifact python_models/svm_mfcc/model.joblib \
    --feature-version mfcc_v1 --algorithm SVC --activate

# Start over, then add the demo accounts back.
.venv/bin/python database/init_db.py --reset --yes --seed
```

## Tables

| Table | Holds | Requirement |
| --- | --- | --- |
| `users` | Accounts, password hashes, one of the five roles, lockout state | FR i, FR ii |
| `audio_files` | Original filename, stored path, SHA-256, perceptual fingerprint, duration, sample rate, source, consent, retention date | FR lxxi, lxxiii, lxxiv, lxxix, lxxx |
| `model_versions` | One row per trained version of each model, with its metrics and whether it's active | FR lxxv |
| `events` | One row per analysed clip or live window: predicted and final class, severity, consistency status, confidence difference, quality, review flag, and a snapshot of the config used | FR xxxi–xli, lxii, lxix |
| `confidence_scores` | One row per model and class, with rank and top flag; never updated | FR xxxi, xxxii, xxxiv |
| `alerts` | Severity, status, a copy of the rule that fired, recommended action, acknowledgement | FR liii–lvi |
| `reviews` | The queue entry and the reviewer's decision, with the original predictions kept | FR lvii–lxi |
| `audit_records` | Append-only log of logins, uploads, mic sessions, predictions, alerts, reviews, overrides, exports, model updates and refusals | FR lxxvi |
| `live_sessions` | Microphone session, consent, status, counts, consecutive-detection state | FR xxxvi, xl, lxxix |
| `live_windows` | One row per 1–3 s window: both models' answers, confirmation count, latency | FR xxxvi, xl |

## Design choices

**Audit rows survive when a user is deleted.** `audit_records.actor_id` is
`ON DELETE SET NULL`, and each row also stores the actor's username and role. So after an
account is removed, a row still reads something like "o.operator, security_operator
acknowledged alert 7 at 20:14". Tested in
`tests/test_database.py::test_record_audit_copies_the_actor_so_history_survives_deletion`.

**A new model version doesn't change old results** (FR lxxv). Each event stores the IDs of
the two model versions that produced it, and each confidence row stores its version.
Activating a newer version doesn't rewrite anything. The test checks that the whole event
(class, severity, confidence difference and all six scores) is identical before and
after. Old version rows can't be deleted either, because events reference them.

**A reviewer's override keeps the models' answer.** `events.predicted_class` is what the
models said and is never overwritten; `events.final_class` holds the reviewer's decision,
and `reviews` keeps the original Python and GTM class, confidence and version next to it.
`Event.effective_class` returns `final_class or predicted_class`. That way the comparison
report and accuracy analytics are still right after an override (FR lxi).

**Alerts keep a copy of their rule.** Alert rules can be edited while the app runs
(FR liii), so each alert stores a `rule_snapshot`, and you can still see why it fired after
`alert_rules/alert_rules.json` changes. Events do the same with `config_snapshot`, which
records the thresholds used, so re-classifying with new thresholds doesn't change old
verdicts.

**Exact duplicates are blocked by the database.** `audio_files.sha256` is `UNIQUE`, so two
uploads of the same bytes at the same moment can't both get in (FR lxxiii). Near-duplicates
are recorded with `perceptual_fingerprint` and `near_duplicate_of_id` but not merged
(FR lxxiv): the re-upload gets its own row and event, and a person decides.

**Stored paths are relative.** `stored_path` is relative to the storage root and goes
through `StorageLayout.resolve()`, which rejects any path that points outside it.

**Audio IDs come from the highest existing number, not a row count.** Otherwise deleting a
row (retention, FR lxxx) could make the next upload reuse an ID that the search and audit
trail still refer to.

**Foreign keys are on.** SQLite ignores them by default, so every connection sets
`PRAGMA foreign_keys=ON`, plus `journal_mode=WAL` and `synchronous=NORMAL` so reads don't
wait on writes during live monitoring.

## Generated DDL

`init_db.py --emit-schema` writes `schema.sql` from the models in `src/models.py`.
`tests/test_database.py` regenerates it and fails if the committed file is different. After
changing a model:

```bash
.venv/bin/python database/init_db.py --emit-schema      # rewrite it
.venv/bin/python -m pytest tests/test_database.py -q    # check they match
```

## Demo accounts (public on purpose)

`seed_credentials.json` has the evaluator logins the SRS asks us to hand over (§1.10 item
11). They are demo credentials, not secrets: they're in the repository, the README and the
report so evaluators can log in without asking. In the database they are hashed with
Werkzeug's `pbkdf2:sha256` like any other account.

One account per role, so every permission in the role matrix can be shown:

| Username | Role | Covers |
| --- | --- | --- |
| `admin` | administrator | Everything |
| `evaluator` | administrator | The login we hand to the evaluators |
| `reviewer` | audio reviewer | Review queue, playback, decisions, overrides |
| `operator` | security operator | Alerts, acknowledgement, escalation, live monitoring |
| `maintenance` | maintenance operator | Machinery Fault events and their alerts |
| `user` | normal user | Uploads, and sees only their own events |

Change the passwords for any real deployment. `tests/test_database.py` checks that all six
are stored as hashes and that each of the five roles has an account.

## Backup and reset

The database is one file plus the storage folder, so a backup is a copy of both:

```bash
cp database/sonicsentinel.db /tmp/backup.db
cp -r data/storage /tmp/backup-storage
```

To start over completely (**this deletes every event, alert, review and audit record**):

```bash
.venv/bin/python database/init_db.py --reset --yes --seed
```

Without `--yes` it asks first, and if there's no terminal to ask on it refuses.

## Related code

- `src/models.py`: the ORM models the tables come from.
- `src/db.py`: engine, sessions, storage layout, `record_audit()`.
- `tests/test_database.py`: tests for this layer, including the FR lxxv test and a
  20,000-event search test.
- `API_DOCUMENTATION.md`: the HTTP API that reads and writes these tables.
