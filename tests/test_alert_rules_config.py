"""The alert-rule files and the config loader are the contract the whole backend reads.

Owner: sara.  SRS FR xxxv, xxxvi, liii, lxxx; SRS 1.10 deliverable 7; SRS 1.8 rule 5.

Two things are under test, and they are the two things an evaluator will poke at:

1. **The shipped files are complete and consistent.** Every one of the ten classes has a
   rule, every severity is on the active scale, every escalation condition is in the
   documented vocabulary -- so a demonstration cannot fail because a rule was missing.
2. **The loader actually reads the files.** A threshold edited on disk changes what the
   loader returns on the next call, with no restart and no cache-clearing call. If the
   loader cached for the lifetime of the process, the SRS "change a threshold live"
   requirement would fail in front of the evaluator, which is the worst possible place.
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
    """A copy of the real config, so a test may edit it without touching the repo."""
    for src in (CONFIG_DIR, RULES_DIR):
        shutil.copytree(src, tmp_path / src.name)
    return ConfigStore(tmp_path / "config", tmp_path / "alert_rules")


# --------------------------------------------------------------------------------------
# The shipped files
# --------------------------------------------------------------------------------------

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
    """FR liii lets an administrator override; with no override we must inherit."""
    assert store.alert_rules()["critical_categories"]["override"] == []
    assert sorted(store.critical_classes()) == sorted(store.classes_config()["critical_classes"])
    assert len(store.critical_classes()) == 5


def test_severity_per_class_matches_the_srs_step_16_examples(store: ConfigStore):
    """Step 16 works through all ten classes; the file must reproduce it exactly."""
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
    """Step 16 lists five levels; FR lii lists four. Five is active because the SRS's
    own worked examples use 'High'."""
    assert store.severity_scale() == ["Informational", "Low", "Medium", "High", "Critical"]
    assert store.severity_rank("Critical") > store.severity_rank("Informational")


def test_four_level_fallback_covers_every_five_level_severity(store: ConfigStore):
    """Switching to the FR lii scale must not orphan a stored severity."""
    levels = store.severity_levels()
    mapping = levels["mapping_between_scales"]["five_level_to_four_level"]
    four = set(levels["scales"]["four_level_fr_lii"])
    for name in levels["scales"]["five_level"]:
        assert name in mapping, f"{name} has no mapping onto the four-level scale"
        assert mapping[name] in four
    # The mapping must not silently downgrade a security alert into a routine one.
    assert mapping["High"] == "Critical"


def test_recommended_action_is_present_and_specific(store: ConfigStore):
    for name in store.class_names():
        action = store.recommended_action_of(name)
        assert isinstance(action, str) and len(action) > 40, (
            f"rule for '{name}' has a placeholder recommended_action"
        )


def test_every_srs_manual_review_trigger_is_covered(store: ConfigStore):
    """SRS Step 17 lists eight triggers; each must map to an enabled condition."""
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
    """A reason_template that references an unknown field would raise at queue time."""
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
    """Numbers must be stated once. The rule file must not restate them.

    The rule file is allowed to deviate per class, but every deviation must be a
    *deliberate, documented* judgement -- not nine copies of the same number waiting to
    drift out of step. This test pins the exact set of deviations, so adding a tenth
    requires editing this list and saying why.
    """
    thresholds = store.thresholds()
    expected_deviations = {
        # class -> which fields it is allowed to state itself, and the SRS reason
        "Gunshot": {"min_confidence", "min_top_two_margin"},   # FR xlvi: confirm before a critical alert
        "Glass Breaking": {"min_top_two_margin"},              # FR xlii: high-severity security alert
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
    """The strongest form of the requirement: one edit, nine rules change with it."""
    assert sandbox.rule_for_class("Glass Breaking")["min_confidence"] == 0.60

    path = sandbox.config_dir / "thresholds.json"
    doc = json.loads(path.read_text())
    doc["confidence"]["min_confidence"] = 0.88
    path.write_text(json.dumps(doc, indent=2))

    assert sandbox.rule_for_class("Glass Breaking")["min_confidence"] == 0.88
    assert sandbox.rule_for_class("Alarm or Siren")["min_confidence"] == 0.88
    # The explicit per-class override is untouched -- it is a stated deviation, not an
    # inherited value, which is exactly why it was written as a literal.
    assert sandbox.rule_for_class("Gunshot")["min_confidence"] == 0.75


def test_gunshot_is_stricter_than_the_global_default(store: ConfigStore):
    """FR xlvi: a critical alert needs confirmation, so Gunshot deliberately deviates."""
    rule = store.rule_for_class("Gunshot")
    assert rule["min_confidence"] > store.thresholds()["confidence"]["min_confidence"]
    assert rule["requires_model_agreement"] is True
    assert store.quality_at_least(rule["min_audio_quality"], "Acceptable")


def test_help_phrases_class_accepts_poor_quality_with_a_recorded_reason(store: ConfigStore):
    """A deliberate trade-off is fine; an undocumented one is not."""
    rule = store.rule_for_class("Person Asking for Help")
    assert rule["min_audio_quality"] == "Poor"
    assert "rationale_for_looser_quality" in rule


def test_background_noise_carries_a_configurable_ambient_limit(store: ConfigStore):
    """FR l: non-critical UNLESS the noise level exceeds a configured limit."""
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
    """A severity override may extend retention, never shorten it."""
    retention = store.retention()
    base = retention["defaults"]["event_records_days"]
    assert retention["overrides_by_severity"]["Critical"]["event_records_days"] >= base


# --------------------------------------------------------------------------------------
# Live editing -- the SRS "change a threshold in front of the evaluator" requirement
# --------------------------------------------------------------------------------------

def test_changing_min_confidence_on_disk_changes_what_the_loader_returns(sandbox: ConfigStore):
    before = sandbox.thresholds()["confidence"]["min_confidence"]
    assert before == 0.60

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
    """Satisfies an evaluator who quotes FR lii's four-level list."""
    assert sandbox.display_severity("High") == "High"

    path = sandbox.alert_rules_dir / "severity_levels.json"
    doc = json.loads(path.read_text())
    doc["active_scale"] = "four_level_fr_lii"
    path.write_text(json.dumps(doc, indent=2))

    assert sandbox.severity_scale() == ["Informational", "Low", "Medium", "Critical"]
    assert sandbox.display_severity("High") == "Critical"
    assert sandbox.display_severity("Low") == "Low"


def test_a_renamed_class_is_caught_by_validation_not_ignored(sandbox: ConfigStore):
    """Renaming a class in the rule file must fail loudly, not drop its alert."""
    path = sandbox.alert_rules_dir / "alert_rules.json"
    doc = json.loads(path.read_text())
    for rule in doc["rules"]:
        if rule["class"] == "Gunshot":
            rule["class"] = "Gun Shot"          # a plausible paraphrase, a real bug
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
    """A typo in a reference must not silently become a string in a float comparison."""
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
    """A critical class that can never alert is a safety hole; validation must find it."""
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


# --------------------------------------------------------------------------------------
# The snapshot the audit trail stores
# --------------------------------------------------------------------------------------

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
