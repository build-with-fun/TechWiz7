"""Engine, sessions and audio storage (SRS FR lxxi-lxxii, lxxvi, lxxx).

Separate from src.models so database/init_db.py and the tests can build a database without a
Flask app. Storage layout (root overridable by SST_STORAGE_DIR)::

    <root>/audio/<yyyy>/<mm>/<audio_id>.<ext>   uploaded and retained audio
    <root>/live/<session_id>/<seq>.wav          short-lived microphone windows
    <root>/exports/<yyyy>/<mm>/<name>           generated exports
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import os
import re
import secrets
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.models import AuditRecord, Base, utcnow

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DB_FILENAME = "sonicsentinel.db"


def repo_root() -> Path:
    return REPO_ROOT


def default_db_path() -> Path:
    """``SST_DB_PATH`` if set (tests use a temp file), else ``database/sonicsentinel.db``."""
    override = os.environ.get("SST_DB_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "database" / DEFAULT_DB_FILENAME


def default_storage_dir() -> Path:
    override = os.environ.get("SST_STORAGE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "data" / "storage"


def database_url(path: str | os.PathLike[str] | None = None) -> str:
    """SQLite URL for a path, or ``:memory:`` when the caller passes it explicitly."""
    if path is None:
        path = default_db_path()
    if str(path) == ":memory:":
        return "sqlite+pysqlite:///:memory:"
    return f"sqlite+pysqlite:///{Path(path).resolve()}"


def create_engine_for(target: str | os.PathLike[str] | None = None, *, echo: bool = False) -> Engine:
    """SQLite engine with foreign keys on (SQLite ignores them by default) and WAL journaling,
    so searches can read while a classification is being written.
    """
    is_memory = str(target) == ":memory:"
    url = "sqlite+pysqlite:///:memory:" if is_memory else database_url(target)
    kwargs: dict = {"echo": echo, "future": True}
    if is_memory:
        # A single shared connection, otherwise each session gets its own empty database.
        from sqlalchemy.pool import StaticPool

        kwargs["poolclass"] = StaticPool
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, **kwargs)

    @event.listens_for(engine, "connect")
    def _configure(dbapi_connection, _record):  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        if not is_memory:
            cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine


# Columns added after the first release (create_all never alters a table): (table, column, SQL
# type).
_ADDED_COLUMNS = (
    ("audio_files", "bit_depth", "INTEGER"),
)


def create_schema(engine: Engine) -> None:
    """Create every table and index. Idempotent: safe to re-run on an existing database."""
    Base.metadata.create_all(engine)
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, column, sql_type in _ADDED_COLUMNS:
            if table in inspector.get_table_names() and column not in {
                c["name"] for c in inspector.get_columns(table)
            }:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextlib.contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    """Commit on success, roll back on any exception, always close.

    Without a factory it uses the running app's factory, so background callers (the persistence
    callback) never open a second engine against the app's WAL database.
    """
    owns_factory = factory is None
    if owns_factory:
        factory = app_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def app_session_factory() -> sessionmaker[Session]:
    """The running app's session factory, for code outside a request; raises when no app is running.
    """
    from flask import current_app

    if not current_app:
        raise RuntimeError(
            "no Flask application is running; pass a session factory or call create_app first"
        )
    factory = current_app.config.get("SST_SESSION_FACTORY")
    if factory is None:
        raise RuntimeError(
            "the running application has no session factory; create_app did not initialise it"
        )
    return factory


# Identifiers

_AUDIO_ID_RE = re.compile(r"^SST-(\d{4})-(\d{2})-(\d{2})-(\d{6})$")


def format_audio_id(sequence: int, when: _dt.datetime | None = None) -> str:
    """``SST-YYYY-MM-DD-NNNNNN``. Sortable, human-readable, bounded length."""
    when = when or utcnow()
    return f"SST-{when:%Y-%m-%d}-{sequence:06d}"


def parse_audio_id(audio_id: str) -> _dt.date | None:
    match = _AUDIO_ID_RE.match((audio_id or "").strip())
    if not match:
        return None
    year, month, day, _seq = match.groups()
    return _dt.date(int(year), int(month), int(day))


def next_audio_id(session: Session, when: _dt.datetime | None = None) -> str:
    """Next free id for the day, from the highest existing suffix (a row count would repeat after a
    delete).
    """
    from src.models import AudioFile

    when = when or utcnow()
    prefix = f"SST-{when:%Y-%m-%d}-"
    rows = session.execute(
        select(AudioFile.audio_id).where(AudioFile.audio_id.like(prefix + "%"))
    ).scalars().all()
    highest = 0
    for value in rows:
        try:
            highest = max(highest, int(value.rsplit("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return format_audio_id(highest + 1, when)


def new_session_id() -> str:
    """UUID4-ish, but generated here so the format is one decision, not many."""
    return secrets.token_hex(16)


# Storage paths


class StorageLayout:
    """Resolves and creates the on-disk locations. Paths are stored *relative* to the root
    so the database survives the project being moved (FR lxxi)."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root).expanduser().resolve() if root else default_storage_dir()

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def live_dir(self) -> Path:
        return self.root / "live"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    def ensure(self) -> "StorageLayout":
        for path in (self.audio_dir, self.live_dir, self.exports_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def audio_path_for(
        self, audio_id: str, extension: str, when: _dt.datetime | None = None
    ) -> Path:
        when = when or utcnow()
        extension = extension if extension.startswith(".") else f".{extension}"
        return self.audio_dir / f"{when:%Y}" / f"{when:%m}" / f"{audio_id}{extension}"

    def live_path_for(self, session_id: str, seq: int, extension: str = ".wav") -> Path:
        return self.live_dir / session_id / f"{seq:06d}{extension}"

    def export_path_for(self, name: str, when: _dt.datetime | None = None) -> Path:
        when = when or utcnow()
        return self.exports_dir / f"{when:%Y}" / f"{when:%m}" / name

    def resolve(self, stored_path: str) -> Path:
        """Turn a stored relative path back into an absolute one, refusing escapes."""
        candidate = (self.root / stored_path).resolve()
        root = self.root.resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError(f"stored path escapes the storage root: {stored_path}")
        return candidate


# Audit helper


def record_audit(
    session: Session,
    *,
    action: str,
    actor=None,
    target_type: str | None = None,
    target_id: str | int | None = None,
    outcome: str = "success",
    detail: str | None = None,
    before: dict | None = None,
    after: dict | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
) -> AuditRecord:
    """Append an audit row (FR lxxvi). ``actor`` may be None (e.g. a failed login for an unknown
    user); username and role are copied so the row stays readable after the user is deleted.
    """
    record = AuditRecord(
        action=action,
        actor_id=getattr(actor, "id", None),
        actor_username=getattr(actor, "username", None),
        actor_role=getattr(actor, "role", None),
        target_type=target_type,
        target_id=None if target_id is None else str(target_id),
        outcome=outcome,
        detail=detail,
        before=before,
        after=after,
        ip_address=ip_address,
        user_agent=(user_agent or "")[:255] or None,
        request_id=request_id,
        sha256=(after or before or {}).get("sha256") if isinstance(after or before, dict) else None,
    )
    session.add(record)
    return record
