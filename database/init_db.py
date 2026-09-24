#!/usr/bin/env python
"""Create, seed and inspect the SonicSentinel AI database.

Owner: sara.  SRS FR lxxi-lxxii (storage and contents), FR lxxv (model versions),
FR lxxx (retention), §1.10 item 11 (ship with credentials).

Typical use::

    .venv/bin/python database/init_db.py                     # create + seed, create missing dirs
    .venv/bin/python database/init_db.py --stats             # what is in there now
    .venv/bin/python database/init_db.py --reset --yes       # wipe and rebuild (destructive)
    .venv/bin/python database/init_db.py --emit-schema       # rewrite database/schema.sql
    .venv/bin/python database/init_db.py --register-model python \\
        --version 3.1.0 --artifact python_models/svm_mfcc_v3/model.joblib --activate

Idempotent by default: running it twice changes nothing and says so. ``--reset`` is the only
destructive path and refuses to run without ``--yes`` (or an interactive confirmation when a
terminal is attached), because the database is the evidence for the whole submission.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import func, inspect, select  # noqa: E402
from sqlalchemy.schema import CreateIndex, CreateTable  # noqa: E402
from werkzeug.security import generate_password_hash  # noqa: E402

from src.db import (  # noqa: E402
    StorageLayout,
    create_engine_for,
    create_schema,
    default_db_path,
    make_session_factory,
    record_audit,
    session_scope,
)
from src.models import (  # noqa: E402
    ALERT_STATUSES,
    CONSISTENCY_STATUSES,
    EVENT_STATUSES,
    MODEL_NAMES,
    QUALITY_VERDICTS,
    ROLES,
    AuditRecord,
    Base,
    ModelVersion,
    User,
    utcnow,
)

DEFAULT_CREDENTIALS = Path(__file__).with_name("seed_credentials.json")
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


# --------------------------------------------------------------------------------------
# schema.sql
# --------------------------------------------------------------------------------------

HEADER = """-- SonicSentinel AI -- canonical database schema.
--
-- GENERATED from src/models.py. Do not hand-edit this file: change the models and run
--   .venv/bin/python database/init_db.py --emit-schema
-- tests/test_database.py fails if this file and the ORM models disagree, so the two
-- definitions can never drift apart.
--
-- Dialect: SQLite 3. Datetimes are UTC, stored naive (no offset). See database/README.md.
-- Owner: sara.  SRS FR lxxi-lxxii, FR lxxv, FR lxxvi, FR lxxx.
"""


def emit_schema(target: Path = SCHEMA_PATH) -> str:
    """Render ``schema.sql`` deterministically: stable table order, stable index order."""
    dialect = create_engine_for(":memory:").dialect
    lines = [HEADER]
    for table in Base.metadata.sorted_tables:
        lines.append(str(CreateTable(table).compile(dialect=dialect)).strip() + ";")
        for index in sorted(table.indexes, key=lambda i: i.name or ""):
            lines.append(str(CreateIndex(index).compile(dialect=dialect)).strip() + ";")
        lines.append("")
    lines.append("-- Vocabulary enforced by CHECK constraints on this schema:")
    lines.append("--   users.role              : " + ", ".join(ROLES))
    lines.append("--   events.status           : " + ", ".join(EVENT_STATUSES))
    lines.append("--   events.quality_verdict  : " + ", ".join(QUALITY_VERDICTS))
    lines.append("--   events.consistency_status: " + ", ".join(CONSISTENCY_STATUSES))
    lines.append("--   alerts.status           : " + ", ".join(ALERT_STATUSES))
    lines.append("--   model_versions.model_name: " + ", ".join(MODEL_NAMES))
    lines.append("")
    text = "\n".join(lines)
    target.write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------------------


def load_credentials(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"credential seed file not found: {path}\n"
            "Set SST_SEED_CREDENTIALS to an alternative, or restore database/seed_credentials.json."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"credential seed file is not valid JSON: {path}: {exc}") from exc
    users = payload.get("users") or []
    if not users:
        raise SystemExit(f"credential seed file declares no users: {path}")
    seen: set[str] = set()
    for entry in users:
        for key in ("username", "password", "role"):
            if not entry.get(key):
                raise SystemExit(f"seed user is missing '{key}': {entry!r}")
        if entry["role"] not in ROLES:
            raise SystemExit(
                f"seed user '{entry['username']}' has unknown role '{entry['role']}'; "
                f"valid roles are {list(ROLES)}"
            )
        if entry["username"] in seen:
            raise SystemExit(f"duplicate seed username: {entry['username']}")
        seen.add(entry["username"])
    return payload


def seed_users(session, credentials: dict, *, verbose: bool = True) -> tuple[int, int]:
    """Create any seed user that does not exist yet. Returns (created, skipped)."""
    created = skipped = 0
    for entry in credentials["users"]:
        existing = session.execute(
            select(User).where(User.username == entry["username"])
        ).scalar_one_or_none()
        if existing is not None:
            skipped += 1
            if verbose:
                print(f"  = user '{entry['username']}' already exists "
                      f"({existing.role}) -- left untouched, password unchanged")
            continue
        user = User(
            username=entry["username"],
            email=entry.get("email"),
            display_name=entry.get("display_name"),
            password_hash=generate_password_hash(entry["password"]),
            role=entry["role"],
            is_active=True,
            must_change_password=bool(credentials.get("force_password_change", False)),
            password_changed_at=utcnow(),
        )
        session.add(user)
        session.flush()
        record_audit(
            session,
            action="user_create",
            actor=None,
            target_type="user",
            target_id=user.id,
            detail=f"seeded {entry['role']} account '{entry['username']}' by database/init_db.py",
            after={"username": user.username, "role": user.role},
        )
        created += 1
        if verbose:
            print(f"  + user '{entry['username']}' ({entry['role']})")
    return created, skipped


def discover_model_artifacts() -> list[dict]:
    """Register any trained model that is actually on disk. Never invents a version.

    ``python_models/**/model_meta.json`` and ``gtm_model/metadata.json`` are the artifacts the
    ML owners produce; if they are absent, the app reports the model as unavailable from
    ``/api/health`` rather than pretending it works.
    """
    found: list[dict] = []

    for meta in sorted((REPO_ROOT / "python_models").glob("**/model_meta.json")):
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        found.append(
            {
                "model_name": "python",
                "version": str(payload.get("model_version") or payload.get("version") or "0.0.0"),
                "label": payload.get("model_name"),
                "artifact_path": str(meta.parent.relative_to(REPO_ROOT)),
                "feature_version": payload.get("feature_version"),
                "algorithm": payload.get("algorithm") or payload.get("estimator"),
                "metrics": payload.get("metrics"),
                "trained_at": _parse_dt(payload.get("trained_at")),
                "dataset_manifest_hash": payload.get("dataset_manifest_hash")
                or payload.get("manifest_sha256"),
            }
        )

    gtm_dir = REPO_ROOT / "gtm_model"
    for meta in sorted(gtm_dir.glob("**/metadata.json")):
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        labels = payload.get("wordLabels") or []
        if not labels:
            continue
        found.append(
            {
                "model_name": "gtm",
                "version": str(payload.get("model_version") or payload.get("version") or "0.0.0"),
                "label": payload.get("model_name") or "Teachable Machine audio model",
                "artifact_path": str(meta.parent.relative_to(REPO_ROOT)),
                "feature_version": payload.get("feature_version"),
                "algorithm": "Google Teachable Machine (audio)",
                "metrics": payload.get("metrics"),
                "trained_at": _parse_dt(payload.get("trained_at")),
                "dataset_manifest_hash": payload.get("dataset_manifest_hash"),
            }
        )
    return found


def _parse_dt(value):
    if not value:
        return None
    import datetime as _dt

    try:
        return _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def register_models(session, artifacts: list[dict], *, verbose: bool = True) -> tuple[int, int]:
    created = skipped = 0
    active_seen: set[str] = set()
    for spec in artifacts:
        exists = session.execute(
            select(ModelVersion).where(
                ModelVersion.model_name == spec["model_name"],
                ModelVersion.version == spec["version"],
            )
        ).scalar_one_or_none()
        if exists is not None:
            skipped += 1
            continue
        activate = spec["model_name"] not in active_seen
        if activate:
            active_seen.add(spec["model_name"])
        session.add(
            ModelVersion(
                model_name=spec["model_name"],
                version=spec["version"],
                label=spec.get("label"),
                artifact_path=spec.get("artifact_path"),
                feature_version=spec.get("feature_version"),
                algorithm=spec.get("algorithm"),
                metrics=spec.get("metrics"),
                trained_at=spec.get("trained_at"),
                is_active=activate,
                dataset_manifest_hash=spec.get("dataset_manifest_hash"),
                notes="discovered by database/init_db.py",
            )
        )
        created += 1
        if verbose:
            print(f"  + model {spec['model_name']} v{spec['version']}"
                  f"{' (active)' if activate else ''}")
    return created, skipped


def register_one_model(session, *, model_name, version, artifact, feature_version=None,
                       algorithm=None, activate=False, metrics=None, label=None) -> ModelVersion:
    if model_name not in MODEL_NAMES:
        raise SystemExit(f"--register-model must be one of {list(MODEL_NAMES)}")
    existing = session.execute(
        select(ModelVersion).where(
            ModelVersion.model_name == model_name, ModelVersion.version == version
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise SystemExit(f"model {model_name} v{version} is already registered (id={existing.id})")
    if activate:
        for row in session.execute(
            select(ModelVersion).where(ModelVersion.model_name == model_name)
        ).scalars():
            row.is_active = False
    record = ModelVersion(
        model_name=model_name,
        version=version,
        label=label,
        artifact_path=artifact,
        feature_version=feature_version,
        algorithm=algorithm,
        metrics=metrics,
        is_active=activate,
        notes="registered via database/init_db.py --register-model",
    )
    session.add(record)
    session.flush()
    record_audit(
        session,
        action="model_version_register",
        target_type="model_version",
        target_id=record.id,
        detail=f"registered {model_name} v{version}",
        after={"model_name": model_name, "version": version, "is_active": activate},
    )
    if activate:
        record_audit(
            session,
            action="model_activate",
            target_type="model_version",
            target_id=record.id,
            detail=f"{model_name} active version is now {version}",
            after={"model_name": model_name, "version": version},
        )
    return record


# --------------------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------------------


def print_stats(engine) -> int:
    inspector = inspect(engine)
    tables = sorted(inspector.get_table_names())
    print(f"database : {engine.url.database}")
    print(f"tables   : {len(tables)}")
    from sqlalchemy.orm import Session as _Session

    with _Session(engine) as session:
        print("\nrows per table:")
        for name in tables:
            count = session.execute(
                select(func.count()).select_from(Base.metadata.tables[name])
            ).scalar_one()
            print(f"  {name:<22} {count:>8,}")

        print("\nusers:")
        for user in session.execute(select(User).order_by(User.role, User.username)).scalars():
            state = "active" if user.is_active else "disabled"
            print(f"  {user.username:<14} {user.role:<22} {state}")

        print("\nmodel versions:")
        rows = session.execute(
            select(ModelVersion).order_by(ModelVersion.model_name, ModelVersion.version)
        ).scalars().all()
        if not rows:
            print("  (none registered yet -- the ML owners have not published artifacts)")
        for row in rows:
            print(f"  {row.model_name:<8} v{row.version:<10} "
                  f"{'ACTIVE' if row.is_active else '      '}  {row.label or ''}")

        print("\ncatalogue counts:")
        for label, count in (
            ("audit records", session.execute(select(func.count()).select_from(AuditRecord)).scalar_one()),
        ):
            print(f"  {label:<22} {count:>8,}")
    return 0


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def confirm_destructive(db_path: Path) -> bool:
    prompt = (
        f"\nThis will DELETE every user, event, alert, review and audit record in\n"
        f"  {db_path}\n"
        "and cannot be undone. Type 'yes' to continue: "
    )
    if not sys.stdin.isatty():
        print(
            "refusing to reset without confirmation: no terminal attached.\n"
            "Re-run with --reset --yes if you are certain.",
            file=sys.stderr,
        )
        return False
    return input(prompt).strip().lower() == "yes"


def reset_database(engine, db_path: Path) -> None:
    """Drop everything and rebuild. FKs are disabled for the drop because SQLAlchemy emits
    tables in dependency order and SQLite would otherwise refuse a parent drop."""
    from sqlalchemy import text

    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys=OFF"))
        Base.metadata.drop_all(connection)
        connection.execute(text("PRAGMA foreign_keys=ON"))
    print("dropped every table")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create, seed and inspect the SonicSentinel AI database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default=None,
                        help="database file (default: database/sonicsentinel.db, or $SST_DB_PATH)")
    parser.add_argument("--storage", default=None,
                        help="storage root for audio/exports (default: data/storage, or $SST_STORAGE_DIR)")
    parser.add_argument("--reset", action="store_true", help="drop every table first (destructive)")
    parser.add_argument("--yes", action="store_true", help="skip the --reset confirmation prompt")
    parser.add_argument("--seed", action="store_true",
                        help="create the seed accounts (implied unless --schema-only)")
    parser.add_argument("--credentials", default=None, help="alternative seed credential file")
    parser.add_argument("--schema-only", action="store_true", help="create tables, seed nothing")
    parser.add_argument("--emit-schema", action="store_true", help="rewrite database/schema.sql and exit")
    parser.add_argument("--stats", action="store_true", help="print table counts and accounts, then exit")
    parser.add_argument("--register-model", choices=list(MODEL_NAMES), default=None)
    parser.add_argument("--version", default=None, help="version for --register-model")
    parser.add_argument("--artifact", default=None, help="artifact path for --register-model")
    parser.add_argument("--feature-version", default=None)
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--activate", action="store_true",
                        help="make this the active version (affects new events only)")
    args = parser.parse_args(argv)

    if args.emit_schema:
        emit_schema()
        print(f"wrote {SCHEMA_PATH.relative_to(REPO_ROOT)} "
              f"({len(Base.metadata.tables)} tables)")
        return 0

    db_path = Path(args.db).expanduser().resolve() if args.db else default_db_path()
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine_for(str(db_path))

    if args.stats:
        return print_stats(engine)

    print(f"database : {db_path}")
    print(f"storage  : {args.storage or os.environ.get('SST_STORAGE_DIR') or 'data/storage'}")

    if args.reset:
        if not args.yes and not confirm_destructive(db_path):
            print("aborted; nothing was changed")
            return 1
        reset_database(engine, db_path)

    create_schema(engine)
    print(f"schema   : {len(Base.metadata.tables)} tables ready")

    layout = StorageLayout(args.storage).ensure()
    print(f"storage  : ready at {layout.root}")

    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        if args.register_model:
            if not args.version:
                raise SystemExit("--register-model requires --version")
            record = register_one_model(
                session,
                model_name=args.register_model,
                version=args.version,
                artifact=args.artifact,
                feature_version=args.feature_version,
                algorithm=args.algorithm,
                label=args.model_label,
                activate=args.activate,
            )
            print(f"registered {record.model_name} v{record.version}"
                  f"{' and activated' if args.activate else ''}")
            return 0

        if not args.schema_only:
            creds_path = Path(args.credentials).expanduser() if args.credentials else \
                Path(os.environ.get("SST_SEED_CREDENTIALS", DEFAULT_CREDENTIALS))
            credentials = load_credentials(creds_path)
            print(f"\nseeding accounts from {creds_path}")
            created, skipped = seed_users(session, credentials)
            print(f"  accounts: {created} created, {skipped} already present")

            artifacts = discover_model_artifacts()
            if artifacts:
                print("\nregistering trained models found on disk")
                made, _ = register_models(session, artifacts)
                if not made:
                    print("  = all discovered versions already registered")
            else:
                print("\nregistering trained models: none found yet.\n"
                      "  looking for python_models/**/model_meta.json and\n"
                      "  gtm_model/**/metadata.json -- run again once the ML owners publish.\n"
                      "  the app boots without them and reports them unavailable in /api/health.")

            record_audit(
                session,
                action="user_create",
                target_type="database",
                target_id=str(db_path),
                detail="database initialised by database/init_db.py",
                after={"tables": len(Base.metadata.tables), "db": str(db_path)},
            )

    print(f"\nready. rerun with --stats to inspect:")
    print(f"  .venv/bin/python database/init_db.py --db {db_path} --stats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
