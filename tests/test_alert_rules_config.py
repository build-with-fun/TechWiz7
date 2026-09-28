"""Alert rule files and the config loader (SRS FR xxxv, xxxvi, liii, lxxx; 1.8 rule 5).

1. The shipped files are complete: every class has a rule, every severity is valid, every
   condition is known.
2. Editing a file on disk changes what the loader returns on the next call, no restart.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.services.config import ConfigError, ConfigStore, REPO_ROOT

CONFIG_DIR = REPO_ROOT / "config"
RULES_DIR = REPO_ROOT / "alert_rules"


@pytest.fixture()
def store() -> ConfigStore:
    return ConfigStore()


@pytest.fixture()
def sandbox(tmp_path: Path) -> ConfigStore:
    """Copy of the real config that tests can edit."""
    for src in (CONFIG_DIR, RULES_DIR):
        shutil.copytree(src, tmp_path / src.name)
    return ConfigStore(tmp_path / "config", tmp_path / "alert_rules")


# The shipped files

def test_all_six_configuration_files_exist():
    for name in ("classes.json", "thresholds.json"):
        assert (CONFIG_DIR / name).is_file(), f"config/{name} is missing"
    for name in ("alert_rules.json", "manual_review_conditions.json", "severity_levels.json", "retention.json"):
        assert (RULES_DIR / name).is_file(), f"alert_rules/{name} is missing (SRS deliverable 7)"


def test_shipped_configuration_is_consistent(store: ConfigStore):
    problems = store.validate()
    assert problems == [], "configuration problems:\n  - " + "\n  - ".join(problems)


def test_every_class_has_exactly_one_rule(store: ConfigStore):
    names = store.class_names()
    rules = store.effective_rules()
    assert sorted(rules) == sorted(names)
    assert len(names) == 10, "the SRS mandates exactly ten sound classes"


def test_critical_categories_match_classes_json_by_default(store: ConfigStore):
    """Without an override, critical classes come from classes.json (FR liii)."""
    assert store.alert_rules()["critical_categories"]["override"] == []
    assert sorted(store.critical_classes()) == sorted(store.classes_config()["critical_classes"])
    assert len(store.critical_classes()) == 5


def test_severity_per_class_matches_the_srs_step_16_examples(store: ConfigStore):
    """Severities match the SRS Step 16 examples."""
    expected = {
        "Background Noise": "Informational",
        "Vehicle Horn": "Low",          # Step 16 allows "Low or Medium"
        "Animal Sound": "Low",
        "Machinery Fault": "High",
        "Glass Breaking": "High",
        "Alarm or Siren": "High",
        "Aggression": "High",
        "Panic Scream": "Critical",
        "Person Asking for Help": "Critical",
        "Gunshot": "Critical",
    }
    actual = {name: store.severity_of(name) for name in store.class_names()}
    assert actual == expected


def test_five_level_scale_is_active_and_ordered(store: ConfigStore):
    """The five-level scale is active (the SRS examples use 'High')."""
    assert store.severity_scale() == ["Informational", "Low", "Medium", "High", "Critical"]
    assert store.severity_rank("Critical") > store.severity_rank("Informational")


def test_four_level_fallback_covers_every_five_level_severity(store: ConfigStore):
    """Every five-level severity maps onto the four-level scale."""
    levels = store.severity_levels()
    mapping = levels["mapping_between_scales"]["five_level_to_four_level"]
    four = set(levels["scales"]["four_level_fr_lii"])
    for name in levels["scales"]["five_level"]:
        assert name in mapping, f"{name} has no mapping onto the four-level scale"
        assert mapping[name] in four
    # High must not become a routine level.
    assert mapping["High"] == "Critical"


def test_recommended_action_is_present_and_specific(store: ConfigStore):
    for name in store.class_names():
        action = store.recommended_action_of(name)
        assert isinstance(action, str) and len(action) > 40, (
            f"rule for '{name}' has a placeholder recommended_action"
        )


def test_every_srs_manual_review_trigger_is_covered(store: ConfigStore):
    """All eight SRS Step 17 triggers have an enabled condition."""
    ids = set(store.review_condition_ids())
    for required in (
        "model_disagreement",
        "low_confidence",
        "poor_audio_quality",
        "similar_top_classes",
        "overlapping_sounds",
        "unsupported_sound_pattern",
        "critical_without_agreement",
        "possible_false_alarm",
    ):
        assert required in ids, f"manual-review condition '{required}' is missing or disabled"


def test_review_condition_reason_templates_name_their_variables(store: ConfigStore):
    """reason_template only uses known fields."""
    allowed = {
        "python_class", "python_confidence", "gtm_class", "gtm_confidence",
        "confidence_difference", "quality", "quality_detail", "margin",
        "margin_min", "secondary_detection", "consistency_status",
        "predicted_class", "low_confidence_band",
    }
    import re

    for cond in store.review_conditions():
        for field in re.findall(r"\{(\w+)", cond["reason_template"]):
            assert field in allowed, (
                f"condition '{cond['id']}' template uses unknown field '{field}'"
            )


def test_repeat_detection_defaults_inherit_the_thresholds_file(store: ConfigStore):
    """Only these per-class overrides of thresholds.json exist."""
    thresholds = store.thresholds()
    expected_deviations = {
        # class -> fields it overrides
        "Gunshot": {"min_confidence"},                         # FR xlvi
        "Glass Breaking": {"min_top_two_margin"},              # FR xlii
    }
    inheritable = ("min_confidence", "min_top_two_margin",
                   "required_consecutive_detections", "window_seconds")

    for name in store.class_names():
        rule = store.rule_for_class(name)
        allowed = expected_deviations.get(name, set())
        for field in inheritable:
            default = thresholds[{
                "min_confidence": ("confidence", "min_confidence"),
                "min_top_two_margin": ("confidence", "top_two_margin_min"),
                "required_consecutive_detections": ("repeat_detection", "required_consecutive_detections"),
                "window_seconds": ("repeat_detection", "window_seconds"),
            }[field][0]][{
                "min_confidence": "min_confidence",
                "min_top_two_margin": "top_two_margin_min",
                "required_consecutive_detections": "required_consecutive_detections",
                "window_seconds": "window_seconds",
            }[field]]
            if field in allowed:
                assert rule[field] != default, (
                    f"'{name}' is listed as deviating on {field} but has not"
                )
            else:
                assert rule[field] == default, (
                    f"'{name}' deviates on {field} without being declared here: "
                    "state the number once, in config/thresholds.json"
                )


def test_editing_the_shared_threshold_file_moves_every_inheriting_rule(sandbox: ConfigStore):
    """One edit to thresholds.json changes every rule that inherits it."""
    shared = sandbox.thresholds()["confidence"]["min_confidence"]
    assert sandbox.rule_for_class("Glass Breaking")["min_confidence"] == shared

    path = sandbox.config_dir / "thresholds.json"
    doc = json.loads(path.read_text())
    doc["confidence"]["min_confidence"] = 0.88
    path.write_text(json.dumps(doc, indent=2))

    assert sandbox.rule_for_class("Glass Breaking")["min_confidence"] == 0.88
    assert sandbox.rule_for_class("Alarm or Siren")["min_confidence"] == 0.88
    # The explicit override stays as it is.
    assert sandbox.rule_for_class("Gunshot")["min_confidence"] == 0.75


def test_gunshot_is_stricter_than_the_global_default(store: ConfigStore):
    """FR xlvi: Gunshot has stricter thresholds."""
    rule = store.rule_for_class("Gunshot")
    assert rule["min_confidence"] > store.thresholds()["confidence"]["min_confidence"]
    assert rule["requires_model_agreement"] is True
    assert store.quality_at_least(rule["min_audio_quality"], "Acceptable")


def test_help_phrases_class_accepts_poor_quality_with_a_recorded_reason(store: ConfigStore):
    """The help class accepts Poor quality and says why."""
    rule = store.rule_for_class("Person Asking for Help")
    assert rule["min_audio_quality"] == "Poor"
    assert "rationale_for_looser_quality" in rule


def test_background_noise_carries_a_configurable_ambient_limit(store: ConfigStore):
    """FR l: background noise escalates above a configured level."""
    rule = store.rule_for_class("Background Noise")
    assert rule["severity"] == "Informational"
    assert "noise_level_dbfs_limit" in rule["thresholds"]
    escalation = rule["escalation"]
    assert any(e["id"] == "noise_level_exceeded" for e in escalation)


def test_retention_defaults_cover_the_required_artifact_classes(store: ConfigStore):
    for artifact in ("audio", "event_records", "alert", "review", "audit", "export"):
        assert store.retention_days(artifact) is not None
    assert store.retention()["enforce_on_purge"] is True
    assert store.retention()["purge_dry_run_default"] is True


def test_retention_never_truncates_a_critical_event(store: ConfigStore):
    """Severity overrides only extend retention."""
    retention = store.retention()
    base = retention["defaults"]["event_records_days"]
    assert retention["overrides_by_severity"]["Critical"]["event_records_days"] >= base


# Live editing

def test_changing_min_confidence_on_disk_changes_what_the_loader_returns(sandbox: ConfigStore):
    before = sandbox.thresholds()["confidence"]["min_confidence"]
    assert before != 0.93

    path = sandbox.config_dir / "thresholds.json"
    doc = json.loads(path.read_text())
    doc["confidence"]["min_confidence"] = 0.93
    path.write_text(json.dumps(doc, indent=2))

    assert sandbox.thresholds()["confidence"]["min_confidence"] == 0.93


def test_changing_a_rule_severity_on_disk_changes_the_verdict(sandbox: ConfigStore):
    assert sandbox.severity_of("Animal Sound") == "Low"

    path = sandbox.alert_rules_dir / "alert_rules.json"
    doc = json.loads(path.read_text())
    for rule in doc["rules"]:
        if rule["class"] == "Animal Sound":
            rule["severity"] = "Medium"
    path.write_text(json.dumps(doc, indent=2))

    assert sandbox.severity_of("Animal Sound") == "Medium"


def test_changing_the_severity_scale_on_disk_changes_the_display(sandbox: ConfigStore):
    """Switching to the four-level scale (FR lii) changes the display."""
    assert sandbox.display_severity("High") == "High"

    path = sandbox.alert_rules_dir / "severity_levels.json"
    doc = json.loads(path.read_text())
    doc["active_scale"] = "four_level_fr_lii"
    path.write_text(json.dumps(doc, indent=2))

    assert sandbox.severity_scale() == ["Informational", "Low", "Medium", "Critical"]
    assert sandbox.display_severity("High") == "Critical"
    assert sandbox.display_severity("Low") == "Low"


def test_a_renamed_class_is_caught_by_validation_not_ignored(sandbox: ConfigStore):
    """A misspelled class name fails validation."""
    path = sandbox.alert_rules_dir / "alert_rules.json"
    doc = json.loads(path.read_text())
    for rule in doc["rules"]:
        if rule["class"] == "Gunshot":
            rule["class"] = "Gun Shot"
    path.write_text(json.dumps(doc, indent=2))

    problems = sandbox.validate()
    assert any("Gun Shot" in p for p in problems), problems
    assert any("Gunshot" in p for p in problems), problems
    with pytest.raises(ConfigError):
        sandbox.validate_or_raise()


def test_a_missing_class_rule_is_caught(sandbox: ConfigStore):
    path = sandbox.alert_rules_dir / "alert_rules.json"
    doc = json.loads(path.read_text())
    doc["rules"] = [r for r in doc["rules"] if r["class"] != "Panic Scream"]
    path.write_text(json.dumps(doc, indent=2))
    assert any("Panic Scream" in p for p in sandbox.validate())


def test_an_unknown_severity_is_caught(sandbox: ConfigStore):
    path = sandbox.alert_rules_dir / "alert_rules.json"
    doc = json.loads(path.read_text())
    doc["rules"][0]["severity"] = "Catastrophic"
    path.write_text(json.dumps(doc, indent=2))
    assert any("Catastrophic" in p for p in sandbox.validate())


def test_an_unresolvable_threshold_reference_is_a_loud_error(sandbox: ConfigStore):
    """A bad $thresholds reference is an error."""
    path = sandbox.alert_rules_dir / "alert_rules.json"
    doc = json.loads(path.read_text())
    doc["defaults"]["min_confidence"] = "$thresholds.confidence.min_confidnce"
    path.write_text(json.dumps(doc, indent=2))

    with pytest.raises(ConfigError, match="min_confidnce"):
        sandbox.alert_rules()


def test_an_unknown_escalation_condition_is_caught(sandbox: ConfigStore):
    path = sandbox.alert_rules_dir / "alert_rules.json"
    doc = json.loads(path.read_text())
    doc["rules"][0]["escalation"] = [
        {"id": "typo", "when": {"confidence_greater_than_x": 0.9}, "to_severity": "Critical"}
    ]
    path.write_text(json.dumps(doc, indent=2))
    assert any("confidence_greater_than_x" in p for p in sandbox.validate())


def test_disabling_a_critical_class_rule_is_caught(sandbox: ConfigStore):
    """Disabling a critical class's rule fails validation."""
    path = sandbox.alert_rules_dir / "alert_rules.json"
    doc = json.loads(path.read_text())
    for rule in doc["rules"]:
        if rule["class"] == "Gunshot":
            rule["enabled"] = False
    path.write_text(json.dumps(doc, indent=2))
    assert any("Gunshot" in p and "disabled" in p for p in sandbox.validate())


def test_a_malformed_config_file_is_a_loud_error(sandbox: ConfigStore):
    (sandbox.config_dir / "thresholds.json").write_text("{ this is not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        sandbox.thresholds()


def test_the_critical_category_override_replaces_the_inherited_list(sandbox: ConfigStore):
    assert "Alarm or Siren" not in sandbox.critical_classes()
    path = sandbox.alert_rules_dir / "alert_rules.json"
    doc = json.loads(path.read_text())
    doc["critical_categories"]["override"] = ["Alarm or Siren"]
    path.write_text(json.dumps(doc, indent=2))
    assert sandbox.critical_classes() == ["Alarm or Siren"]
    assert sandbox.is_critical("Alarm or Siren") is True
    assert sandbox.is_critical("Gunshot") is False


def test_a_typo_in_the_critical_override_is_refused(sandbox: ConfigStore):
    path = sandbox.alert_rules_dir / "alert_rules.json"
    doc = json.loads(path.read_text())
    doc["critical_categories"]["override"] = ["Gun Shot"]
    path.write_text(json.dumps(doc, indent=2))
    with pytest.raises(ConfigError, match="Gun Shot"):
        sandbox.critical_classes()


# Config snapshot

def test_snapshot_records_a_hash_per_file_and_survives_json(store: ConfigStore):
    snap = store.snapshot()
    assert set(snap.content_hashes) == {
        "thresholds", "classes", "alert_rules", "manual_review", "severity_levels", "retention",
    }
    assert all(len(h) == 16 for h in snap.content_hashes.values())
    payload = snap.to_dict()
    assert json.loads(json.dumps(payload)) == payload


def test_snapshot_hash_changes_when_a_file_changes(sandbox: ConfigStore):
    before = sandbox.snapshot().content_hashes["alert_rules"]
    path = sandbox.alert_rules_dir / "alert_rules.json"
    doc = json.loads(path.read_text())
    doc["rules"][0]["recommended_action"] += " "
    path.write_text(json.dumps(doc, indent=2))
    assert sandbox.snapshot().content_hashes["alert_rules"] != before


def test_quality_ordering_is_used_for_comparison(store: ConfigStore):
    assert store.quality_at_least("Good", "Acceptable") is True
    assert store.quality_at_least("Acceptable", "Acceptable") is True
    assert store.quality_at_least("Poor", "Acceptable") is False
    assert store.quality_at_least("Unusable", "Poor") is False
    with pytest.raises(ConfigError):
        store.quality_at_least("Excellent", "Good")


def test_a_missing_alert_rules_directory_fails_loudly(tmp_path: Path):
    empty = ConfigStore(tmp_path / "config", tmp_path / "alert_rules")
    (tmp_path / "config").mkdir()
    shutil.copy(CONFIG_DIR / "thresholds.json", tmp_path / "config" / "thresholds.json")
    with pytest.raises(ConfigError, match="missing"):
        empty.alert_rules()
