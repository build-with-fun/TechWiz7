"""Print the configuration the loader actually resolved, references substituted.

    .venv/bin/python -m tools.show_alert_rules            # everything
    .venv/bin/python -m tools.show_alert_rules Gunshot    # one class
    .venv/bin/python -m tools.show_alert_rules --check    # validate, exit 1 on problems

Written for the demonstration: it is the fastest honest answer to "show me the rule that
just fired", and the quickest proof that a threshold edited on disk took effect.
"""

from __future__ import annotations

import json
import sys

from src.services.config import ConfigError, ConfigStore


def _dump(label: str, payload) -> None:
    print(f"\n=== {label} " + "=" * max(0, 60 - len(label)))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(argv: list[str]) -> int:
    store = ConfigStore()

    if "--check" in argv:
        problems = store.validate()
        if problems:
            print("CONFIGURATION PROBLEMS:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print("configuration OK: %d class rules, %d review conditions, scale %s"
              % (len(store.effective_rules()), len(store.review_conditions()),
                 store.severity_scale()))
        return 0

    wanted = [a for a in argv if not a.startswith("-")]

    try:
        snap = store.snapshot()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    print("config dir       :", store.config_dir)
    print("alert rules dir  :", store.alert_rules_dir)
    print("versions         : thresholds %s | classes %s | alert_rules %s | manual_review %s"
          % (snap.thresholds_version, snap.classes_version,
             snap.alert_rules_version, snap.manual_review_version))
    print("severity scale   :", " -> ".join(store.severity_scale()),
          f"(active_scale={snap.severity_scale})")
    print("critical classes :", ", ".join(store.critical_classes()))
    print("content hashes   :", json.dumps(snap.content_hashes))

    if wanted:
        for name in wanted:
            try:
                _dump(f"rule: {name}", store.rule_for_class(name))
            except ConfigError as exc:
                print(f"unknown class '{name}': {exc}", file=sys.stderr)
                return 1
        return 0

    _dump("global defaults (references resolved)", store.alert_rules()["defaults"])
    _dump("per-class effective rules",
          {n: store.rule_for_class(n) for n in store.class_names()})
    _dump("manual-review conditions", store.review_conditions())
    _dump("retention", store.retention())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(1)
