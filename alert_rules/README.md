# `alert_rules/` — the configurable rule files

**Owner: `sara`.** SRS §1.10 deliverable 7 ("configurable rule files"), FR xxxv, FR xxxvi,
FR liii, FR lxxx, and Step 15 / Step 16 / Step 17.

These four files are what an administrator edits. **No threshold, severity, recommended
action or retention window is a literal in the Python code.** The SRS says evaluators may
demand a changed threshold during the demonstration, so an edit here is the *only* action
needed: the loader keys its cache on each file's modification time and re-reads on the next
call. There is no restart step and no code change.

## The files

| File | What it controls | SRS |
|---|---|---|
| `alert_rules.json` | Per-class alert rule: critical flag, severity, recommended action, min confidence, top-two margin, required consecutive detections, model-agreement requirement, min audio quality, escalation conditions. | FR liii, xli–l, Step 15, Step 16 |
| `manual_review_conditions.json` | The eight Step 17 triggers that route an event to the manual-review queue, each with the priority and the sentence the reviewer is shown. | Step 17, FR lvii |
| `severity_levels.json` | The severity scale(s), their order, display colour tokens, and the mapping used if an evaluator prefers FR lii's four-level list. | FR lii, Step 16 |
| `retention.json` | How long audio, events, alerts, reviews, audits and exports are kept. | FR lxxx |

## How numbers are shared (do not duplicate them)

`config/thresholds.json` (owner `lorena`) is the single source of truth for numerics.
A rule file writes a **reference** instead of a copy:

```json
"min_confidence": "$thresholds.confidence.min_confidence"
```

The loader replaces the reference with the value from `thresholds.json` at load time. So:

- editing `config/thresholds.json` moves **all ten** rules at once;
- writing a literal in a rule file **overrides** it for that class only.

That second behaviour is used twice, deliberately, and both are pinned by a test:

| Class | Deviates | Why |
|---|---|---|
| `Gunshot` | `min_confidence` 0.75, `min_top_two_margin` 0.20 | FR xlvi: a critical alert must satisfy configured confirmation *and* confidence rules. |
| `Glass Breaking` | `min_top_two_margin` 0.10 | FR xlii: a high-severity security alert should not be suppressed by a near-tie between two breakage-like classes. |

`test_alert_rules_config.py::test_repeat_detection_defaults_inherit_the_thresholds_file`
holds this list. Adding a third deviation means editing that list and saying why.

## Editing it live, in front of an evaluator

```bash
# 1. see exactly what the loader resolved (references already substituted)
.venv/bin/python -m tools.show_alert_rules

# 2. change a number, e.g. min confidence 0.60 -> 0.93
$EDITOR config/thresholds.json

# 3. run it again -- 0.93 is now the floor for every rule that inherits it
.venv/bin/python -m tools.show_alert_rules

# 4. prove it changed behaviour, not just config
.venv/bin/python -m pytest tests/test_alert_rules_config.py -q
```

The same edit is available through the admin UI (`PUT /api/admin/config/thresholds`), which
writes the file through the same validation, so the UI and the text editor cannot produce
different results.

## What a bad edit does

Every file is validated at boot and on every write. An unknown class name, a class with no
rule, a severity that is on no scale, an unresolvable `$thresholds.` reference, an unknown
escalation condition, or a **disabled rule for a critical class** fails loudly
(`ConfigError`) with the file and the offending value named. A configuration mistake must
never silently remove an alert — that is the failure mode that gets someone hurt.

```bash
.venv/bin/python -m pytest tests/test_alert_rules_config.py -q   # 33 tests
```

## Severity: the SRS contradicts itself

Step 16 lists **five** levels (Informational, Low, Medium, High, Critical). FR lii lists
**four** (Informational, Low, Medium, Critical). Step 16 is the step that actually assigns a
severity to each of the ten classes, and its own examples use *High*, so **five levels is
active** and the file's `levels` list is the source of the ordering, colours and
notification behaviour.

If an evaluator quotes FR lii, switch it live — no data migration, because stored
severities are names and the display order comes from the file:

```json
"active_scale": "four_level_fr_lii"
```

`display_severity()` then maps `High → Critical` for display and filtering. The mapping
deliberately does **not** map `High → Medium`: FR xlii and xliii promise Glass Breaking and
Alarm or Siren a *high-severity* alert, and downgrading them to Medium would break those FRs
at the moment the scale was switched. Recorded severities are never rewritten.

## Escalation conditions

`alert_rules.json` `escalation_condition_vocabulary` is the closed set of condition keys a
rule may use. Validation rejects anything else, so a typo cannot produce a rule that never
fires:

`confidence_gte`, `consecutive_gte`, `top_two_margin_gte`, `repeats_within_window_gte`,
`sustained_seconds_gte`, `noise_level_dbfs_gte`, `quality_in`.

Conditions nest with `all` and `any`.
