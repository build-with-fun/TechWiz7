# `database/` — the store of record

Everything SonicSentinel AI knows lives here: accounts and roles, audio metadata and
hashes, both models' predictions and confidences, alerts, manual reviews, model versions
and the audit trail. This folder holds the schema, the initialiser, and the deliberately
published demo credentials.

SRS coverage: **FR ii** (roles), **FR lxxi** (secure audio storage), **FR lxxii** (what the
database stores), **FR lxxiii–lxxiv** (duplicate / near-duplicate detection), **FR lxxv**
(model version tracking), **FR lxxvi** (audit trail), **FR lxxx** (retention),
**§1.10 item 11** (folder list and demo credentials).

---

## Files

| File | What it is |
| --- | --- |
| `schema.sql` | The DDL, **generated** from the SQLAlchemy models. Never hand-edit. |
| `init_db.py` | Creates the database, seeds accounts, registers model versions, prints stats. |
| `seed_credentials.json` | The demo accounts. Published on purpose — see below. |
| `sonicsentinel.db` | The SQLite database itself (created on first run, git-ignored). |

## Quick start

Run from the repository root, with the project virtualenv:

```bash
# 1. Create (or bring up to date) the database and seed the demo accounts.
.venv/bin/python database/init_db.py --seed

# 2. Look at what you got.
.venv/bin/python database/init_db.py --stats
```

That is the whole install step for the storage layer. `--seed` is idempotent: running it
twice adds nothing and says so.

### Pointing it somewhere else

Both paths are overridable, by flag or by environment variable, so the test suite and a
disposable demo run never touch the real store:

```bash
.venv/bin/python database/init_db.py --db /tmp/demo.db --storage /tmp/demo-storage --seed
```

| Flag | Environment variable | Default |
| --- | --- | --- |
| `--db` | `SST_DB_PATH` | `database/sonicsentinel.db` |
| `--storage` | `SST_STORAGE_DIR` | `data/storage` |

## All command-line options

| Option | Does |
| --- | --- |
| *(no options)* | Create the schema if it is missing, then exit. Non-destructive. |
| `--seed` | Also insert the demo accounts and register any model artifacts found on disk. |
| `--credentials FILE` | Seed from a different credential file. |
| `--schema-only` | Create the tables and seed nothing. |
| `--emit-schema` | Rewrite `database/schema.sql` from the models and exit. |
| `--stats` | Print per-table row counts, the accounts and the registered model versions. |
| `--register-model {python,gtm}` | Register one model version: `--version V --artifact PATH` and optionally `--feature-version`, `--algorithm`, `--model-label`, `--activate`. |
| `--reset` | **Destructive.** Drop every table and rebuild. |
| `--yes` | Skip the `--reset` confirmation prompt (for scripts; refuses without a TTY otherwise). |

Examples:

```bash
# Register the Python model and make it the active version (FR lxxv).
.venv/bin/python database/init_db.py --register-model python \
    --version 3.1.0 --artifact python_models/svm_mfcc/model.joblib \
    --feature-version mfcc_v1 --algorithm SVC --activate

# Start over, keeping the seeded accounts.
.venv/bin/python database/init_db.py --reset --yes --seed
```

## Tables, and which requirement each one answers

| Table | Holds | Requirement |
| --- | --- | --- |
| `users` | Accounts, hashed passwords, one of the five roles, lockout state | FR i, FR ii |
| `audio_files` | Original filename, stored path, SHA-256, perceptual fingerprint, duration, sample rate, source, consent, retention expiry | FR lxxi, lxxiii, lxxiv, lxxix, lxxx |
| `model_versions` | One row per trained version of each model, with metrics and active flag | FR lxxv |
| `events` | One row per analysed clip or live-window sequence: predicted and final class, severity, consistency status, confidence difference, quality verdict, review flag, **config snapshot** | FR xxxi–xli, lxii, lxix |
| `confidence_scores` | Append-only, one row per model per class, with rank and top flag | FR xxxi, xxxii, xxxiv |
| `alerts` | Severity, status, the **rule snapshot that fired**, recommended action, acknowledgement | FR liii–lvi |
| `reviews` | The queue entry *and* the reviewer's decision, with the original prediction preserved | FR lvii–lxi |
| `audit_records` | Append-only trail of logins, uploads, mic sessions, predictions, alerts, reviews, overrides, exports, model updates, denials | FR lxxvi |
| `live_sessions` | Microphone session, consent, status, counts, consecutive-detection state | FR xxxvi, xl, lxxix |
| `live_windows` | One row per 1–3 s window: both models' answers, confirmation count, latency | FR xxxvi, xl |

## Design decisions an evaluator is likely to probe

**The audit trail survives the deletion of a user.** `audit_records.actor_id` is
`ON DELETE SET NULL`, and the actor's username and role are copied onto every row. Deleting
an account (FR lxxx retention, or an administrator removing one) therefore blanks the
foreign key but leaves "o.operator, security_operator acknowledged alert 7 at 20:14"
readable. Tested in `tests/test_database.py::test_record_audit_copies_the_actor_so_history_survives_deletion`.

**Activating a new model version cannot change a past result.** FR lxxv says so verbatim,
and it is the first thing a sceptical evaluator will try. Every event stores the *ids* of
the two model versions that produced it, and every confidence row stores the version it came
from; nothing on an existing event is rewritten when a newer version is activated. The test
asserts the whole event — class, severity, confidence difference, and all six confidence
scores — is byte-identical before and after. The old version row cannot be deleted either:
its events hold a foreign key to it.

**A reviewer's override does not erase the models' answer.** `events.predicted_class` is
what the two models said and is never overwritten; `events.final_class` is the effective
answer, and `reviews` keeps the original Python and GTM class, confidence and version
alongside the decision. `Event.effective_class` returns `final_class or predicted_class`.
That is what makes the comparison report and the per-class accuracy analytics still correct
after an override (FR lxi).

**An alert keeps the rule that fired.** Alert rules are editable live (FR liii), so each
alert stores a `rule_snapshot`. "Why did this alert?" keeps answering correctly after an
administrator changes `alert_rules/alert_rules.json` mid-demo. The same reasoning gives
`events.config_snapshot`, which records the thresholds in force when the event was classified.

**An event keeps the thresholds it was judged by.** Same mechanism: change
`config/thresholds.json` and re-classify, and old events keep their original verdict.

**The duplicate check is a constraint, not a query.** `audio_files.sha256` is `UNIQUE`, so
two simultaneous uploads of the same bytes cannot both win a check-then-insert race
(FR lxxiii). Near-duplicates are *recorded* — `perceptual_fingerprint` plus
`near_duplicate_of_id` — never silently merged (FR lxxiv): the re-upload keeps its own row
and its own event, and a human decides.

**Stored paths are relative.** `stored_path` is relative to the storage root and resolved
through `StorageLayout.resolve()`, which refuses any path that escapes it. A path that came
out of the database is data, not an instruction.

**Event identifiers are derived from the highest existing suffix, not from a row count.**
Deleting a row (retention, FR lxxx) must not make the next upload reuse a live identifier —
the search filter and the audit trail both address an audio file by this id.

**Foreign keys are enforced.** SQLite ignores them unless asked, so the engine sets
`PRAGMA foreign_keys=ON` on every connection, and `journal_mode=WAL` with
`synchronous=NORMAL` for the concurrency of a live-monitoring demo.

## Generated DDL

`schema.sql` is written by `init_db.py --emit-schema` from the models in `src/models.py`,
so the two cannot drift silently — `tests/test_database.py` re-generates it and fails if the
committed file differs. After changing a model:

```bash
.venv/bin/python database/init_db.py --emit-schema      # rewrite it
.venv/bin/python -m pytest tests/test_database.py -q    # prove they agree
```

## Demo accounts — published on purpose

`seed_credentials.json` holds the evaluator credentials the SRS requires to be handed over
(§1.10 item 11). They are **demo credentials, not secrets**: they are in the repository, in
`README.md` and in the report, so that an evaluator can log in without asking anyone. They
are still hashed with Werkzeug's `pbkdf2:sha256` in the database, exactly like any other
account — the file gives the passwords, the database never stores them.

One account per role, so every permission boundary in the five-role matrix is demonstrable:

| Username | Role | Covers |
| --- | --- | --- |
| `admin` | administrator | Everything |
| `evaluator` | administrator | The login the SRS asks us to hand over |
| `reviewer` | audio reviewer | The manual-review queue, playback, decision, override |
| `operator` | security operator | Alerts, acknowledgement, escalation, live monitoring |
| `maintenance` | maintenance operator | Machinery Fault events and their alerts |
| `user` | normal user | Upload and see only their own events |

Passwords are in the file, and the admin account is instructed to rotate them for a real
deployment. `tests/test_database.py` checks that all six are stored as hashes and that
every one of the five SRS roles has an account.

## Backing up and resetting

The database is a single file plus the storage tree, so a backup is a copy of both:

```bash
cp database/sonicsentinel.db /tmp/backup.db
cp -r data/storage /tmp/backup-storage
```

To start over completely — **this deletes every event, alert, review and audit record**:

```bash
.venv/bin/python database/init_db.py --reset --yes --seed
```

Without `--yes` it asks for confirmation, and refuses outright when there is no terminal to
ask on, so a stray `--reset` in a script cannot silently empty the demo database.

## Where the rest of it lives

- `src/models.py` — the ORM models these tables are generated from.
- `src/db.py` — engine, session factory, storage layout, `record_audit()`.
- `tests/test_database.py` — the contract on this layer, including the FR lxxv immutability
  test and a 20,000-event test that asserts the search filter uses the index.
- `documentation/api_contract.md` — the HTTP surface that reads and writes these tables.
