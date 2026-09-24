"""Live-editable configuration for the SonicSentinel backend.

Why this module exists
----------------------
The SRS is explicit that thresholds and alert rules are **administrator-editable
at runtime** (FR xxxv, FR xxxvi, FR liii, FR lxxx) and the integrity rules state
that a change must take effect without editing code (SRS 1.8, rule 5). Evaluators
are expected to ask for a threshold to be changed *during* the demonstration and
then watch the behaviour change.

So nothing in the backend may read a threshold from a literal. Every module reads
it from here, and this module:

* loads ``config/*.json`` and ``alert_rules/*.json`` from disk;
* caches on the file's modification time, re-reading the moment it changes, so a
  save is enough -- no restart, no reload endpoint required (one exists anyway,
  for convenience);
* resolves ``"$thresholds.<dotted.path>"`` references so a number is stated once,
  in ``config/thresholds.json``, and the rule files only record deviations;
* validates on load: an unknown class name, a missing class rule, an unknown
  severity name or a bad reference raises :class:`ConfigError` instead of quietly
  disabling an alert. A configuration mistake must be loud at boot, never silent
  at 3am.

Usage
-----
::

    from src.services.config import ConfigStore
    store = ConfigStore()                 # repo config/ + alert_rules/
    store.thresholds()["confidence"]["min_confidence"]
    store.rule_for_class("Gunshot")["severity"]
    store.severity_of("Glass Breaking")   # -> "High"

Tests point the store at a temporary directory (constructor argument or
``SST_CONFIG_DIR`` / ``SST_ALERT_RULES_DIR``) to prove that editing a JSON file
changes behaviour.

SRS references: FR xxxv, FR xxxvi, FR liii, FR lxxx, SRS 1.8 rule 5,
SRS 1.10 deliverable 7.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_ALERT_RULES_DIR = REPO_ROOT / "alert_rules"

REFERENCE_PREFIX = "$thresholds."

#: Keys every effective alert rule must carry after the ``defaults`` block is
#: merged into it. A missing key is a load-time error, not a runtime surprise.
REQUIRED_RULE_KEYS = (
    "class",
    "severity",
    "recommended_action",
    "min_confidence",
    "min_top_two_margin",
    "required_consecutive_detections",
    "window_seconds",
    "requires_model_agreement",
    "min_audio_quality",
)


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing, malformed or contradictory.

    Deliberately not a subclass of ``ValueError``: the app factory catches this
    to fail startup with a readable message, and tests assert on it.
    """


def _read_json(path: Path) -> dict:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file is missing: {path}") from exc
    except OSError as exc:  # pragma: no cover - permission/IO problems
        raise ConfigError(f"configuration file is unreadable: {path}: {exc}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"configuration file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"configuration file must contain a JSON object: {path}")
    return data


def _file_token(path: Path) -> tuple[float, int]:
    """Cache key: modification time plus size, so a same-second edit still lands."""
    try:
        st = path.stat()
    except OSError as exc:
        raise ConfigError(f"configuration file is missing: {path}") from exc
    return (st.st_mtime, st.st_size)


def _dig(data: Mapping[str, Any], dotted: str, *, origin: str) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            raise ConfigError(
                f"unresolved configuration reference '{origin}': "
                f"no such key '{part}' while walking '{dotted}'"
            )
    return node


def resolve_references(value: Any, thresholds: Mapping[str, Any], *, origin: str) -> Any:
    """Replace ``"$thresholds.a.b"`` strings with the value from thresholds.json.

    Walks lists and dicts. Any other string is returned untouched, so a severity
    name or a path is never mistaken for a reference.
    """
    if isinstance(value, str):
        if value.startswith(REFERENCE_PREFIX):
            dotted = value[len(REFERENCE_PREFIX):]
            return _dig(thresholds, dotted, origin=origin)
        return value
    if isinstance(value, list):
        return [resolve_references(v, thresholds, origin=origin) for v in value]
    if isinstance(value, dict):
        return {k: resolve_references(v, thresholds, origin=origin) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class ConfigSnapshot:
    """What judged one event.

    Stored alongside every prediction so a result can always be explained in the
    terms that were in force when it was produced -- the same reasoning as FR lxxv
    (model version tracking), applied to the rules that consumed the model output.
    """

    thresholds_version: str
    classes_version: str
    alert_rules_version: str
    manual_review_version: str
    severity_scale: str
    content_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds_version": self.thresholds_version,
            "classes_version": self.classes_version,
            "alert_rules_version": self.alert_rules_version,
            "manual_review_version": self.manual_review_version,
            "severity_scale": self.severity_scale,
            "content_hashes": dict(self.content_hashes),
        }


class ConfigStore:
    """mtime-cached, validated reader for ``config/`` and ``alert_rules/``."""

    def __init__(
        self,
        config_dir: str | os.PathLike[str] | None = None,
        alert_rules_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.config_dir = Path(
            config_dir or os.environ.get("SST_CONFIG_DIR") or DEFAULT_CONFIG_DIR
        ).resolve()
        self.alert_rules_dir = Path(
            alert_rules_dir or os.environ.get("SST_ALERT_RULES_DIR") or DEFAULT_ALERT_RULES_DIR
        ).resolve()
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[tuple[float, int], dict]] = {}
        self._hashes: dict[str, tuple[float, str]] = {}

    # ------------------------------------------------------------------ paths
    def _config_path(self, name: str) -> Path:
        return self.config_dir / f"{name}.json"

    def _rules_path(self, name: str) -> Path:
        return self.alert_rules_dir / f"{name}.json"

    # ----------------------------------------------------------------- caching
    def _load(self, path: Path) -> dict:
        token = _file_token(path)
        key = str(path)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and hit[0] == token:
                return hit[1]
        data = _read_json(path)  # read outside the lock; cheap and avoids blocking
        with self._lock:
            self._cache[key] = (token, data)
        return data

    def _content_hash(self, path: Path) -> str:
        token = _file_token(path)
        key = str(path)
        with self._lock:
            hit = self._hashes.get(key)
            if hit is not None and hit[0] == token[0]:
                return hit[1]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        with self._lock:
            self._hashes[key] = (token[0], digest)
        return digest

    # ------------------------------------------------------------- raw access
    def thresholds(self) -> dict:
        """``config/thresholds.json`` -- owned by lorena, read by everyone."""
        return self._load(self._config_path("thresholds"))

    def classes_config(self) -> dict:
        """``config/classes.json`` -- the single source of truth for class names."""
        return self._load(self._config_path("classes"))

    def auth_config(self) -> dict:
        """``config/auth.json`` -- sessions, login lockout, rate limits, headers.

        Owned by sara and read by ``src/app.py``/``src/auth.py``. Kept in a config file
        because the lockout window and the session lifetime are things an administrator
        is expected to change without editing code (SRS §1.8 integrity rule 5).
        """
        return self._load(self._config_path("auth"))

    def auth_setting(self, dotted: str, default: Any = None) -> Any:
        """One value out of ``config/auth.json``, by dotted path, with a default.

        Returns ``default`` for a missing key rather than raising: an app setting that is
        absent should fall back to the documented default, not stop the console booting.
        """
        node: Any = self.auth_config()
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def class_names(self) -> list[str]:
        data = self.classes_config()
        names = [c["name"] for c in data.get("classes", [])]
        if not names:
            raise ConfigError("config/classes.json declares no classes")
        if len(set(names)) != len(names):
            raise ConfigError("config/classes.json contains duplicate class names")
        return names

    def class_codes(self) -> dict[str, str]:
        return {c["name"]: c.get("code", "") for c in self.classes_config().get("classes", [])}

    def critical_classes(self) -> list[str]:
        """Critical categories, with the FR liii administrator override applied.

        An EMPTY ``critical_categories.override`` means "inherit classes.json".
        A non-empty override REPLACES the inherited list, and is validated against
        the known class names so a typo cannot silently drop a critical alert.
        """
        inherited = list(self.classes_config().get("critical_classes", []))
        rules = self.alert_rules()
        override = list(rules.get("critical_categories", {}).get("override", []) or [])
        known = set(self.class_names())
        unknown = [c for c in override if c not in known]
        if unknown:
            raise ConfigError(
                "alert_rules/alert_rules.json: critical_categories.override names "
                f"unknown classes {unknown}; known classes are {sorted(known)}"
            )
        chosen = override if override else inherited
        return [c for c in self.class_names() if c in set(chosen)]

    def is_critical(self, class_name: str) -> bool:
        return class_name in set(self.critical_classes())

    # ------------------------------------------------------------ alert rules
    def alert_rules(self) -> dict:
        """``alert_rules/alert_rules.json`` with threshold references resolved."""
        raw = self._load(self._rules_path("alert_rules"))
        thresholds = self.thresholds()
        return resolve_references(raw, thresholds, origin=str(self._rules_path("alert_rules")))

    def effective_rules(self) -> dict[str, dict]:
        """Per-class rule with ``defaults`` merged in. Keyed by class name."""
        data = self.alert_rules()
        defaults = dict(data.get("defaults", {}))
        out: dict[str, dict] = {}
        for rule in data.get("rules", []):
            name = rule.get("class")
            if not name:
                raise ConfigError("alert_rules.json: a rule has no 'class' key")
            merged = dict(defaults)
            merged.update(rule)
            merged["critical"] = bool(merged.get("critical", False))
            out[name] = merged
        return out

    def rule_for_class(self, class_name: str) -> dict:
        rules = self.effective_rules()
        if class_name not in rules:
            raise ConfigError(
                f"alert_rules/alert_rules.json has no rule for class '{class_name}'"
            )
        return rules[class_name]

    def severity_of(self, class_name: str) -> str:
        return self.rule_for_class(class_name)["severity"]

    def recommended_action_of(self, class_name: str) -> str:
        return self.rule_for_class(class_name)["recommended_action"]

    def severity_scale(self) -> list[str]:
        levels = self.severity_levels()
        name = levels.get("active_scale")
        scales = levels.get("scales", {})
        if name not in scales:
            raise ConfigError(
                f"alert_rules/severity_levels.json: active_scale '{name}' is not in "
                f"'scales' ({sorted(scales)})"
            )
        return list(scales[name])

    def severity_levels(self) -> dict:
        return self._load(self._rules_path("severity_levels"))

    def display_severity(self, recorded: str) -> str:
        """Map a recorded severity onto the active scale for display and filtering.

        The stored value is never rewritten; only the presented name changes when
        an administrator switches to the four-level scale of FR lii.
        """
        if recorded in self.severity_scale():
            return recorded
        levels = self.severity_levels()
        mapping = (
            levels.get("mapping_between_scales", {}).get("five_level_to_four_level", {})
        )
        if recorded in mapping and mapping[recorded] in self.severity_scale():
            return mapping[recorded]
        raise ConfigError(
            f"severity '{recorded}' is on no known scale and no mapping covers it"
        )

    def severity_rank(self, recorded: str) -> int:
        """Rank on the *five-level* scale, used to order critical events."""
        for level in self.severity_levels().get("levels", []):
            if level.get("name") == recorded:
                return int(level.get("rank", 0))
        return -1

    def quality_ordering(self) -> list[str]:
        return list(self.alert_rules().get("quality_ordering", []))

    def quality_at_least(self, quality: str, minimum: str) -> bool:
        """True when ``quality`` is at least ``minimum`` on the configured ordering."""
        order = self.quality_ordering()
        if quality not in order or minimum not in order:
            raise ConfigError(
                f"unknown audio quality '{quality}' or minimum '{minimum}'; "
                f"expected one of {order}"
            )
        return order.index(quality) >= order.index(minimum)

    # --------------------------------------------------------- manual review
    def manual_review(self) -> dict:
        raw = self._load(self._rules_path("manual_review_conditions"))
        thresholds = self.thresholds()
        return resolve_references(
            raw, thresholds, origin=str(self._rules_path("manual_review_conditions"))
        )

    def review_conditions(self) -> list[dict]:
        return [c for c in self.manual_review().get("conditions", []) if c.get("enabled", True)]

    def review_condition_ids(self) -> list[str]:
        return [c["id"] for c in self.review_conditions()]

    # ------------------------------------------------------------- retention
    def retention(self) -> dict:
        return self._load(self._rules_path("retention"))

    def retention_days(self, artifact: str) -> int | None:
        """Days an artifact class is kept; ``None`` means indefinitely."""
        defaults = self.retention().get("defaults", {})
        key = f"{artifact}_days"
        if key not in defaults:
            raise ConfigError(f"alert_rules/retention.json has no default for '{key}'")
        return defaults[key]

    # -------------------------------------------------------------- snapshot
    def snapshot(self) -> ConfigSnapshot:
        """A record of exactly which configuration judged an event."""
        files = {
            "thresholds": self._config_path("thresholds"),
            "classes": self._config_path("classes"),
            "alert_rules": self._rules_path("alert_rules"),
            "manual_review": self._rules_path("manual_review_conditions"),
            "severity_levels": self._rules_path("severity_levels"),
            "retention": self._rules_path("retention"),
        }
        hashes = {k: self._content_hash(p) for k, p in files.items()}
        return ConfigSnapshot(
            thresholds_version=str(self.thresholds().get("version", "unknown")),
            classes_version=str(self.classes_config().get("version", "unknown")),
            alert_rules_version=str(self.alert_rules().get("version", "unknown")),
            manual_review_version=str(self.manual_review().get("version", "unknown")),
            severity_scale=str(self.severity_levels().get("active_scale", "unknown")),
            content_hashes=hashes,
        )

    # ------------------------------------------------------------ validation
    def validate(self) -> list[str]:
        """Check every cross-file invariant. Returns the list of problems found.

        Called by the app factory (which raises) and by the test suite (which
        asserts the list is empty). Split out so ``--check`` tooling can print it.
        """
        problems: list[str] = []

        try:
            names = self.class_names()
        except ConfigError as exc:
            return [str(exc)]

        # config/auth.json -- the web layer's own settings. A malformed one would only be
        # discovered at the first login, which is the worst possible moment to find it.
        try:
            auth = self.auth_config()
        except ConfigError as exc:
            problems.append(str(exc))
        else:
            login = auth.get("login")
            if not isinstance(login, Mapping):
                problems.append("config/auth.json has no 'login' settings block")
            else:
                max_attempts = login.get("max_failed_attempts")
                if not isinstance(max_attempts, int) or max_attempts < 1:
                    problems.append(
                        "config/auth.json login.max_failed_attempts must be a positive "
                        "integer, or an attacker gets unlimited guesses"
                    )
                lockout = login.get("lockout_minutes")
                if not isinstance(lockout, (int, float)) or lockout <= 0:
                    problems.append(
                        "config/auth.json login.lockout_minutes must be greater than zero"
                    )
                min_length = login.get("min_password_length")
                if not isinstance(min_length, int) or min_length < 8:
                    problems.append(
                        "config/auth.json login.min_password_length must be at least 8"
                    )
            session = auth.get("session")
            if not isinstance(session, Mapping):
                problems.append("config/auth.json has no 'session' settings block")
            elif not isinstance(session.get("lifetime_minutes"), (int, float)):
                problems.append(
                    "config/auth.json session.lifetime_minutes must be a number"
                )

        try:
            rules = self.effective_rules()
        except ConfigError as exc:
            return [str(exc)]

        missing = [n for n in names if n not in rules]
        if missing:
            problems.append(
                "alert_rules.json declares no rule for: " + ", ".join(missing)
            )
        extra = [n for n in rules if n not in names]
        if extra:
            problems.append(
                "alert_rules.json declares rules for unknown classes: " + ", ".join(extra)
            )

        try:
            scale = set(self.severity_scale())
        except ConfigError as exc:
            problems.append(str(exc))
            scale = set()

        vocabulary = set(self.alert_rules().get("escalation_condition_vocabulary", {}))
        quality_order = set(self.quality_ordering())

        for name, rule in rules.items():
            for key in REQUIRED_RULE_KEYS:
                if key not in rule or rule[key] is None:
                    problems.append(f"rule '{name}' is missing required key '{key}'")
            severity = rule.get("severity")
            if scale and severity not in scale:
                problems.append(
                    f"rule '{name}' has severity '{severity}' which is not on the "
                    f"active scale {sorted(scale)}"
                )
            min_q = rule.get("min_audio_quality")
            if quality_order and min_q not in quality_order:
                problems.append(
                    f"rule '{name}' has min_audio_quality '{min_q}' which is not a "
                    f"known quality {sorted(quality_order)}"
                )
            margin = rule.get("min_top_two_margin")
            if isinstance(margin, (int, float)) and not 0.0 <= float(margin) <= 1.0:
                problems.append(f"rule '{name}' has min_top_two_margin {margin} outside 0..1")
            conf = rule.get("min_confidence")
            if isinstance(conf, (int, float)) and not 0.0 <= float(conf) <= 1.0:
                problems.append(f"rule '{name}' has min_confidence {conf} outside 0..1")
            consecutive = rule.get("required_consecutive_detections")
            if isinstance(consecutive, int) and consecutive < 1:
                problems.append(
                    f"rule '{name}' requires {consecutive} consecutive detections; "
                    "at least 1 is required"
                )
            for esc in rule.get("escalation", []) or []:
                when = esc.get("when", {})
                if not when:
                    problems.append(f"rule '{name}' escalation '{esc.get('id')}' has no 'when'")
                for key in self._condition_keys(when):
                    if vocabulary and key not in vocabulary:
                        problems.append(
                            f"rule '{name}' escalation '{esc.get('id')}' uses unknown "
                            f"condition '{key}'"
                        )
                target = esc.get("to_severity")
                if target and scale and self._severity_or_mapping_missing(target, scale):
                    problems.append(
                        f"rule '{name}' escalation '{esc.get('id')}' escalates to "
                        f"unknown severity '{target}'"
                    )

        # A critical class must actually be able to raise an alert.
        for name in self.critical_classes():
            rule = rules.get(name)
            if rule and not rule.get("enabled", True):
                problems.append(
                    f"critical class '{name}' has its alert rule disabled, so it can "
                    "never raise an alert"
                )

        try:
            conditions = self.review_conditions()
        except ConfigError as exc:
            problems.append(str(exc))
            conditions = []
        if not conditions:
            problems.append("manual_review_conditions.json enables no conditions")
        for cond in conditions:
            if not cond.get("when"):
                problems.append(f"manual-review condition '{cond.get('id')}' has no 'when'")
            if not cond.get("reason_template"):
                problems.append(
                    f"manual-review condition '{cond.get('id')}' has no reason_template, "
                    "so a reviewer would be told only that the item needs review"
                )

        try:
            self.severity_scale()
        except ConfigError as exc:
            problems.append(str(exc))

        try:
            retention = self.retention().get("defaults", {})
            for key in ("audio_days", "event_records_days", "audit_days"):
                if key not in retention:
                    problems.append(f"retention.json has no default for '{key}'")
        except ConfigError as exc:
            problems.append(str(exc))

        return problems

    def _severity_or_mapping_missing(self, severity: str, scale: set[str]) -> bool:
        if severity in scale:
            return False
        mapping = (
            self.severity_levels().get("mapping_between_scales", {}).get("five_level_to_four_level", {})
        )
        mapped = mapping.get(severity)
        return not (mapped and mapped in scale)

    @staticmethod
    def _condition_keys(when: Any) -> Iterable[str]:
        """Every condition key inside a ``when`` block, including ``all``/``any`` nests."""
        if not isinstance(when, Mapping):
            return []
        found: list[str] = []
        for key, value in when.items():
            if key in ("all", "any") and isinstance(value, list):
                for sub in value:
                    found.extend(ConfigStore._condition_keys(sub))
            else:
                found.append(key)
        return found

    def validate_or_raise(self) -> None:
        problems = self.validate()
        if problems:
            raise ConfigError(
                "configuration is inconsistent:\n  - " + "\n  - ".join(problems)
            )


_default_store: ConfigStore | None = None
_default_lock = threading.Lock()


def get_store() -> ConfigStore:
    """Process-wide store. Cached per directory pair, cheap to call anywhere."""
    global _default_store
    if _default_store is None:
        with _default_lock:
            if _default_store is None:
                _default_store = ConfigStore()
    return _default_store


def set_store(store: ConfigStore | None) -> None:
    """Override the process-wide store; used by tests and the app factory."""
    global _default_store
    with _default_lock:
        _default_store = store
