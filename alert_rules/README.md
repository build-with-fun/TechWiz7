# `alert_rules/`: configurable rule files

SRS §1.10 deliverable 7 ("configurable rule files"), FR xxxv, FR xxxvi,
FR liii, FR lxxx, and Step 15 / Step 16 / Step 17.

An administrator edits these four files. No threshold, severity, recommended action or
retention period is hard-coded in Python. The SRS says evaluators may ask for a threshold
change during the demo, and editing the file is all it takes: the loader notices the new
modification time and re-reads the file on the next call, with no restart.

## The files

| File | What it controls | SRS |
|---|---|---|
| `alert_rules.json` | Per-class alert rule: critical flag, severity, recommended action, min confidence, top-two margin, required consecutive detections, model-agreement requirement, min audio quality, escalation conditions. | FR liii, xli–l, Step 15, Step 16 |
| `manual_review_conditions.json` | The eight Step 17 triggers that route an event to the manual-review queue, each with the priority and the sentence the reviewer is shown. | Step 17, FR lvii |
| `severity_levels.json` | The severity scale(s), their order, display colour tokens, and the mapping used if an evaluator prefers FR lii's four-level list. | FR lii, Step 16 |
| `retention.json` | How long audio, events, alerts, reviews, audits and exports are kept. | FR lxxx |

## Shared numbers

The numbers live in `config/thresholds.json`, and a rule file refers to them instead of
copying them:

```json
"min_confidence": "$thresholds.confidence.min_confidence"
```

The loader swaps in the value from `thresholds.json` when it loads. So:

- editing `config/thresholds.json` changes **all ten** rules at once;
- writing a number directly in a rule file **overrides** it for that class only.

Two classes override on purpose, and a test checks the list:

| Class | Deviates | Why |
|---|---|---|
| `Gunshot` | `min_confidence` 0.75, `min_top_two_margin` 0.20 | FR xlvi: a critical alert must satisfy configured confirmation *and* confidence rules. |
| `Glass Breaking` | `min_top_two_margin` 0.10 | FR xlii: a high-severity security alert shouldn't be blocked by a near-tie between two similar classes. |

The list is in `test_alert_rules_config.py::test_repeat_detection_defaults_inherit_the_thresholds_file`.
To add another override, add it there too, with the reason.

## Editing during a demo

```bash
# 1. see the rules as the loader sees them (references filled in)
.venv/bin/python -m tools.show_alert_rules

# 2. change a number, e.g. min confidence 0.60 to 0.93
$EDITOR config/thresholds.json

# 3. run it again: 0.93 is now the minimum for every rule that uses it
.venv/bin/python -m tools.show_alert_rules

# 4. check that behaviour changed too
.venv/bin/python -m pytest tests/test_alert_rules_config.py -q
```

You can make the same edit in the admin UI (`PUT /api/admin/config/thresholds`). It runs
the same validation before writing the file, so both ways give the same result.

## What a bad edit does

Every file is checked at start-up and on every write. An unknown class name, a class
without a rule, a severity that isn't on any scale, a `$thresholds.` reference that doesn't
exist, an unknown escalation condition, or a **disabled rule for a critical class** all
raise a `ConfigError` naming the file and the bad value. A config mistake should never
quietly switch off an alert.

```bash
.venv/bin/python -m pytest tests/test_alert_rules_config.py -q   # 33 tests
```

## Severity: four levels or five

Step 16 of the SRS lists **five** levels (Informational, Low, Medium, High, Critical), but
FR lii lists **four** (Informational, Low, Medium, Critical). Step 16 is the one that gives
each class a severity, and its examples use *High*, so the **five-level scale is active**.
The file's `levels` list sets the order, colours and notification behaviour.

If an evaluator prefers FR lii, you can switch while the app runs. Nothing needs migrating,
because stored severities are names and the display order comes from the file:

```json
"active_scale": "four_level_fr_lii"
```

`display_severity()` then shows `High` as `Critical` for display and filtering. It doesn't
map `High` to `Medium`, because FR xlii and xliii promise Glass Breaking and Alarm or Siren a
*high-severity* alert. The stored severities are never rewritten.

## Escalation conditions

`escalation_condition_vocabulary` in `alert_rules.json` lists the condition keys a rule can
use. Anything else fails validation, so a typo can't create a rule that never fires:

`confidence_gte`, `consecutive_gte`, `top_two_margin_gte`, `repeats_within_window_gte`,
`sustained_seconds_gte`, `noise_level_dbfs_gte`, `quality_in`.

Conditions nest with `all` and `any`.
