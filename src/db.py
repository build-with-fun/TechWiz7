"""Engine, session and storage plumbing for the SonicSentinel database.

Owner: sara.  SRS FR lxxi (secure audio storage), FR lxxii, FR lxxvi, FR lxxx.

Kept separate from :mod:`src.models` so that the schema can be imported — and the database
created — without a Flask application object. Evaluators run ``database/init_db.py``
directly, and the test suite builds throwaway databases; neither should need a running app.

Storage layout, with the root overridable by ``SST_STORAGE_DIR``::

    <root>/
      audio/<yyyy>/<mm>/<audio_id>.<ext>     uploaded and retained event audio
      live/<session_id>/<seq>.wav            short-lived microphone windows
      exports/<yyyy>/<mm>/<name>             generated CSV/Excel/report artifacts
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
    """Where the application database lives unless told otherwise.

    ``SST_DB_PATH`` wins, then ``database/sonicsentinel.db``. Tests set the env var to a
    temp file so a test run can never touch the development database.
    """
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
    """An engine configured for correctness, not for defaults.

    Two pragmas are set explicitly because both default to the wrong thing:

    * ``foreign_keys=ON`` — SQLite ignores FK constraints unless asked, which would let a
      confidence score outlive its event and quietly corrupt the analytics.
    * ``journal_mode=WAL`` — lets the search endpoint read while a classification is being
      written, which is what keeps the dashboard responsive during a live demo.
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


def create_schema(engine: Engine) -> None:
    """Create every table and index. Idempotent: safe to re-run on an existing database."""
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextlib.contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    """Commit on success, roll back on any exception, always close.

    Always commits on the way out, whether the factory was supplied or looked up. Callers
    that pass one rely on that commit to finish their unit of work -- the seed and admin
    paths write rows inside this block and expect them to survive it. Callers that need to
    hold a transaction open across several statements call ``commit`` themselves and the
    trailing one is then a harmless empty transaction.

    Without a factory this uses the application's own session factory, which is what a
    background caller reaches for -- the pipeline's ``persist`` callback runs with no
    request context of its own and must not open a second engine against a database the
    app already has open in WAL mode.
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
    """The session factory the running application is bound to.

    ``create_app`` stores its engine and factory on ``current_app.config`` because that is
    the one place every request shares; this is the reader for code that is not inside a
    request and still needs the same database (the pipeline's persistence callback, the
    background live monitor). It raises ``RuntimeError`` when no app is running rather than
    silently opening a second engine against a database the app already has open.
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


# --------------------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------------------

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
    """Next free identifier.

    Derived from the highest existing suffix for the same day rather than the row count:
    a row count would hand out a duplicate the moment anything is ever deleted, and
    ``audio_id`` is unique-constrained, so the bug would surface as a 500 on upload.
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


# --------------------------------------------------------------------------------------
# Storage paths
# --------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------
# Audit helper
# --------------------------------------------------------------------------------------


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
    """Append an audit row (FR lxxvi). Callers never construct the model directly.

    ``actor`` may be a :class:`~src.models.User`, or ``None`` for an anonymous action such
    as a failed login for a username that does not exist. The username and role are copied
    onto the row so the record stays readable after the user is deleted.
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
