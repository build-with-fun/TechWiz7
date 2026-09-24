#!/usr/bin/env python3
"""Validate the perception deliverables.

Reusable by imran's submission readiness checker. Fails loudly on the mistakes that
actually matter: a class name that does not match the SRS ten, a pair naming a class
that does not exist, a live window that breaks the latency NFR, or a hop that makes
the SRS consecutive-detection rule unsatisfiable.

Run:  python3 documentation/perception/validate_perception.py
Exit: 0 = all checks pass, 1 = at least one failure (all failures printed).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The ten mandatory classes, exact names as mandated by the SRS.
SRS_TEN = [
    "Machinery Fault",
    "Glass Breaking",
    "Alarm or Siren",
    "Vehicle Horn",
    "Animal Sound",
    "Gunshot",
    "Panic Scream",
    "Aggression",
    "Person Asking for Help",
    "Background Noise",
]

REQUIRED_DOCS = [
    "feature_rationale.md",
    "class_similarity_and_confusability.md",
    "alert_human_factors.md",
    "class_confusability.json",
    "recommended_perceptual_params.json",
]

VALID_REASON_CLASSES = {
    "irreducible",
    "reducible_with_dedicated_features",
    "reducible_with_augmentation",
    "label_ambiguity",
}

VALID_SEVERITIES = {"Informational", "Low", "Medium", "High", "Critical"}

failures = []
checks = 0


def check(cond, msg):
    global checks
    checks += 1
    if not cond:
        failures.append(msg)


def main():
    # --- required files exist and are not empty ---
    for name in REQUIRED_DOCS:
        p = os.path.join(HERE, name)
        check(os.path.isfile(p), "missing file: %s" % name)
        if os.path.isfile(p):
            check(os.path.getsize(p) > 400, "file suspiciously small: %s" % name)

    cpath = os.path.join(HERE, "class_confusability.json")
    ppath = os.path.join(HERE, "recommended_perceptual_params.json")
    if not (os.path.isfile(cpath) and os.path.isfile(ppath)):
        report()
        return

    with open(cpath) as f:
        conf = json.load(f)
    with open(ppath) as f:
        par = json.load(f)

    # --- the ten classes, exactly, no more no less ---
    classes = [c["class"] for c in conf["classes"]]
    check(len(classes) == 10, "expected 10 classes, found %d" % len(classes))
    check(len(set(classes)) == 10, "duplicate class names in confusability map")
    check(
        sorted(classes) == sorted(SRS_TEN),
        "class names do not match the SRS ten exactly.\n    missing: %s\n    unexpected: %s"
        % (
            sorted(set(SRS_TEN) - set(classes)),
            sorted(set(classes) - set(SRS_TEN)),
        ),
    )
    check(
        conf.get("classes_are_exactly_the_srs_ten") is True,
        "classes_are_exactly_the_srs_ten must be true",
    )

    # severity must NOT be a single static level per class: the SRS gives three
    # classes a two-valued severity (Vehicle Horn Low or Medium; Aggression high or
    # critical depending on confidence and repeated detection; Background Noise
    # escalated above a configured limit). If everything is a singleton, the map has
    # been flattened back into a hard-coded table.
    multi = [c["class"] for c in conf["classes"] if len(c["severity"]["allowed"]) > 1]
    for expected in ("Vehicle Horn", "Aggression", "Background Noise"):
        check(
            expected in multi,
            "%r must allow more than one severity level (SRS makes it context-dependent), "
            "found %s" % (expected, conf["classes"][[c["class"] for c in conf["classes"]].index(expected)]["severity"]["allowed"]),
        )

    # --- every class carries the fields the consumers rely on ---
    for c in conf["classes"]:
        for field in ("class", "perceptual_signature", "human_cue", "key_features", "severity"):
            check(field in c, "class %r missing field %r" % (c.get("class"), field))
        sev = c.get("severity", {})
        for field in ("default", "allowed", "srs_step16_verbatim", "srs_fr_verbatim"):
            check(field in sev, "class %r severity missing field %r" % (c.get("class"), field))
        check(
            sev.get("default") in VALID_SEVERITIES,
            "class %r has invalid default severity %r" % (c.get("class"), sev.get("default")),
        )
        allowed = sev.get("allowed", [])
        check(
            isinstance(allowed, list) and len(allowed) > 0,
            "class %r has an empty allowed severity list" % c.get("class"),
        )
        check(
            sev.get("default") in allowed,
            "class %r default severity %r not in its allowed list %s"
            % (c.get("class"), sev.get("default"), allowed),
        )
        for lvl in allowed:
            check(
                lvl in VALID_SEVERITIES,
                "class %r allows unknown severity level %r" % (c.get("class"), lvl),
            )
        # FR xliv / xlviii / l make severity context-dependent: a class with more than
        # one allowed level MUST state what escalates it.
        if len(allowed) > 1:
            check(
                isinstance(sev.get("escalation_trigger"), str) and sev.get("escalation_trigger").strip(),
                "class %r allows multiple severity levels but does not say what escalates them"
                % c.get("class"),
            )
        check(
            isinstance(c.get("key_features"), list) and len(c.get("key_features", [])) > 0,
            "class %r has no key_features" % c.get("class"),
        )
        check(
            isinstance(c.get("critical_for_recall_nfr"), bool),
            "class %r missing critical_for_recall_nfr flag" % c.get("class"),
        )

    # the five NFR-critical classes must be flagged exactly
    critical = sorted(c["class"] for c in conf["classes"] if c["critical_for_recall_nfr"])
    srs_critical = sorted(["Gunshot", "Glass Breaking", "Panic Scream", "Aggression", "Person Asking for Help"])
    check(
        critical == srs_critical,
        "NFR-critical class set is wrong.\n    got:      %s\n    expected: %s" % (critical, srs_critical),
    )

    # --- pairs: SRS names at least eight; each names real classes and a valid mechanism ---
    known = set(SRS_TEN) | set(conf.get("reference_classes_outside_the_ten", []))
    pairs = conf["pairs"]
    check(len(pairs) >= 8, "expected at least 8 confusable pairs, found %d" % len(pairs))
    ids = set()
    for p in pairs:
        for field in ("id", "a", "b", "reason_class", "mitigation", "perceptual_mechanism", "human_cue"):
            check(field in p, "pair %r missing field %r" % (p.get("id"), field))
        check(p.get("id") not in ids, "duplicate pair id %r" % p.get("id"))
        ids.add(p.get("id"))
        check(p.get("a") in known, "pair %r names unknown class %r" % (p.get("id"), p.get("a")))
        check(p.get("b") in known, "pair %r names unknown class %r" % (p.get("id"), p.get("b")))
        check(p.get("a") != p.get("b"), "pair %r compares a class with itself" % p.get("id"))
        check(
            p.get("reason_class") in VALID_REASON_CLASSES,
            "pair %r has invalid reason_class %r" % (p.get("id"), p.get("reason_class")),
        )
        check(
            isinstance(p.get("augmentation_plan"), str) and p.get("augmentation_plan").strip(),
            "pair %r has no augmentation_plan" % p.get("id"),
        )

    # the four SRS Step-14 named pairs that matter most must be present
    def has_pair(x, y):
        return any({p["a"], p["b"]} == {x, y} for p in pairs)

    for x, y in [("Gunshot", "Vehicle Backfire"), ("Alarm or Siren", "Vehicle Horn"),
                 ("Panic Scream", "Aggression"), ("Glass Breaking", "Machinery Fault")]:
        check(has_pair(x, y), "SRS Step-14 pair not covered: %s vs %s" % (x, y))

    # --- live window vs the NFR ---
    lw = par["live_window"]
    dur = lw["duration_s"]
    hop = lw["hop_s"]
    budget = lw["inference_budget_s"]
    check(isinstance(dur, (int, float)) and 0 < dur <= 3.0, "live window duration must be in (0, 3.0]")
    check(isinstance(hop, (int, float)) and hop > 0, "live window hop must be positive")
    check(hop <= dur, "hop (%s) must not exceed window duration (%s)" % (hop, dur))
    check(
        hop < dur,
        "hop must be strictly smaller than the window, or a short critical event "
        "(gunshot 0.2-0.8 s) can fall inside one window and the SRS consecutive-detection "
        "rule is unsatisfiable",
    )
    check(
        dur + budget <= 3.0,
        "window (%ss) + inference budget (%ss) exceeds the NFR 3 s latency promise" % (dur, budget),
    )
    check(dur <= 3.0, "live window exceeds the SRS hard maximum of 3 s")

    # --- severity set must cover both readings of the SRS, configurable ---
    sev = par["alerting_human_factors"]
    check("srs_conflict_note" in sev, "the Step-16 vs FR-lii severity contradiction must be documented")
    weights = sev["review_queue_priority"]["severity_weight"]
    check(
        set(weights) == VALID_SEVERITIES,
        "severity_weight keys must cover all five levels, got %s" % sorted(weights),
    )
    check("Critical" in weights and weights["Critical"] == max(weights.values()),
          "Critical must carry the highest severity weight")

    # --- quality gate before critical alerts ---
    check(
        "quality_bands" in par["audio_quality_thresholds"],
        "quality bands missing",
    )
    for band in ("Good", "Acceptable", "Poor", "Unusable"):
        check(band in par["audio_quality_thresholds"]["quality_bands"], "quality band %r undefined" % band)

    # --- measured corpus evidence must be present and pinned to a revision ---
    ev = conf.get("measured_corpus_evidence")
    check(ev is not None, "class_confusability.json must carry measured_corpus_evidence")
    if ev:
        rev = ev.get("generator_revision_measured", "")
        check(
            len(rev) >= 64 and all(c in "0123456789abcdef" for c in rev[:64]),
            "corpus findings must be pinned to a generator sha256, got %r" % rev[:40],
        )
        check(len(ev.get("findings", [])) >= 5, "at least 5 measured corpus findings expected")
        check(
            len(ev.get("assert_mode_guarantees", [])) >= 5,
            "assert_mode_guarantees must list what the audit actually guarantees",
        )
        check(
            "does_not_establish" in " ".join(ev.keys()),
            "corpus evidence must state what it does NOT establish",
        )

    # --- optional live corpus audit; --with-corpus-audit turns it into a gate ---
    if "--with-corpus-audit" in sys.argv:
        import subprocess

        audit = os.path.join(HERE, "audit_corpus_realism.py")
        check(os.path.exists(audit), "audit_corpus_realism.py must exist for --with-corpus-audit")
        if os.path.exists(audit):
            py = os.path.join(os.path.dirname(os.path.dirname(HERE)), ".venv", "bin", "python")
            if not os.path.exists(py):
                py = sys.executable
            r = subprocess.run([py, audit, "--assert"], capture_output=True, text=True, timeout=900)
            check(
                r.returncode == 0,
                "corpus perceptual audit failed:\n" + (r.stdout or "")[-1500:] + (r.stderr or "")[-500:],
            )

    report()


def report():
    if failures:
        print("PERCEPTION DELIVERABLE VALIDATION: FAIL (%d checks, %d failures)" % (checks, len(failures)))
        for f in failures:
            print("  - %s" % f)
        sys.exit(1)
    print("PERCEPTION DELIVERABLE VALIDATION: PASS (%d checks)" % checks)
    sys.exit(0)


if __name__ == "__main__":
    main()
