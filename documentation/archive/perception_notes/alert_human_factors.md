# Alerts and manual review: human factors

> Archived design notes from 23 Sep, written while the corpus was still procedurally
> generated. The reasoning still holds; measured numbers refer to that earlier corpus.

Date: 2026-09-23. The numbers suggested here, with reasons, are in
`recommended_perceptual_params.json` under `alerting_human_factors`.

These notes explain why SRS Step 15 (repeated-detection confirmation), Step 13 (audio
quality) and Step 17 (manual review) matter, and what "severity" should mean.

---

## 1. Severity is not confidence

Severity says how fast someone has to act and who else needs to know. It doesn't say how
sure the model is. A 0.99-confidence Vehicle Horn is a sure thing that doesn't matter much;
a 0.72-confidence Gunshot is an unsure thing that might be an emergency. If the UI shows
them with the same weight, the operator's attention follows model confidence, which is the
wrong thing to follow.

So severity comes from the class and the rules, and is shown separately from confidence.
Confidence gets its own display (a number, a bar, a threshold marker), never the alert
colour.

### Four levels or five

- SRS Step 16 lists five levels: Informational, Low, Medium, High, Critical.
- FR lii lists four: Informational, Low, Medium, Critical (no High).
- Step 16's own examples use High (Machinery Fault, Glass Breaking, Alarm or Siren,
  Aggression).

Suggestion: use the five-level scale and make the level set configurable, so it can be
switched to four without code changes. Mention the conflict and the choice in the report.
Alarm or Siren should be High, not treated like Gunshot, even though Step 14 lists it among
"critical acoustic events".

### Severity depends on context for three classes

| Class | SRS wording | What it means |
|---|---|---|
| Vehicle Horn | Step 16: "Low or Medium"; FR xliv: stored as a traffic event | Base Low; Medium when confirmed over consecutive windows at Good quality. |
| Aggression | Step 16: "High"; FR xlviii: "high or critical alert depending on confidence and repeated detection" | Base High; Critical on high confidence plus repeated detection. |
| Background Noise | Step 16: "Informational"; FR l: "non-critical unless the estimated noise level exceeds a configured limit" | Base Informational; escalates above a configured noise level. |

So the rule file needs a base level plus escalation conditions per class, not one fixed
level (FR liii lists these settings as administrator-configurable).

---

## 2. Why repeated detection

The startle reflex kicks in within about 30-40 ms, but it also fades quickly: its size drops
a lot within 5-10 repeats of the same stimulus (Blumenthal et al. 2005, *Psychophysiology*
42(1):1-15). One isolated detection is exactly the kind of thing people learn to ignore.
Requiring the same event in consecutive windows, plus a confidence minimum, turns "something
twitched" into "this is still happening".

The hop size matters too (`live_window` in `recommended_perceptual_params.json`). With
non-overlapping 2 s windows, a 0.3-0.6 s gunshot fits inside one window, so "three in a row"
could never happen and Gunshot would never alert. The windows have to overlap.

---

## 3. False alarms

- In clinical monitoring, 72-99% of alarms need no action (Cvach 2012, *Biomed Instrum
  Technol* 46(4):268-277; Sendelbach & Funk 2013, *AACN Adv Crit Care* 24(4):378-386), and
  alarm fatigue is a known safety hazard (The Joint Commission, Sentinel Event Alert 50, 2013).
- Operators who see false alarms first distrust and then ignore the system (Bliss & Dunn
  2000, *Ergonomics* 43(9):1393-1407; Parasuraman & Riley 1997, *Human Factors*
  39(2):230-253).

A monitor with too many false alarms makes people slower to react to the real ones. That's
why FR lxxviii lists "excessive critical alerts" as an anomaly.

A suggested budget: **at most one false critical alert per 8-hour shift** (0.125/h). Pick
critical-class thresholds on the validation split as the lowest that meet it, never on the
test split. With about 45 validation clips per class, one false positive is already a big
share of a class, so this means fairly high thresholds at this data size. Only report the
budget if it was actually measured.

---

## 4. Quality gate

Step 15 requires acceptable audio quality before a critical event is confirmed. Below about
10 dB in-band SNR a person can't reliably hear the event, and below 0 dB it's effectively
gone. A classifier still outputs a distribution for such a window and can be confidently
wrong.

Clipping is worse: heavy clipping adds broadband distortion with a sharp envelope, which
looks like a gunshot or glass breaking. A clipped recording of a loud harmless sound can
produce a confident critical false alarm, so clipping should block critical alerts, not just
change a quality badge.

---

## 5. Acknowledgement and alert storms

- **Acknowledge or escalate within 60 s.** Otherwise "unacknowledged" looks the same as
  "nobody is watching".
- **300 s cooldown per event class.** Don't re-notify for the same event within the
  cooldown, or people end up muting the whole system.
- **Different events still alert.** The cooldown must not hide another class.
- **Make higher severities stand out more**, or the tenth Critical of the shift gets the
  attention of an Informational.
- **Never show severity by colour alone** (WCAG 1.4.1; about 8% of men have some red-green
  colour blindness). Always add a label and an icon.

---

## 6. Review queue order

Sorting by arrival time makes reviewers spend effort on whatever came first, not on what
matters most. Sort by the expected cost of getting it wrong:

```
priority = severity_weight * (1 - effective_confidence) * quality_penalty + critical_miss_bonus
```

Weights are in `recommended_perceptual_params.json`.

- `severity_weight`: how bad a mistake is (Gunshot 10, Background Noise 1).
- `1 - effective_confidence`: how likely the models are wrong, using both models together.
- `quality_penalty`: raises anything with poor audio.
- `critical_miss_bonus`: added when a critical class is a strong runner-up, or the models
  disagree about a critical class. A missed gunshot should outrank uncertainty about
  Background Noise, even if the latter is "more uncertain".

Also:

- Playback should be one click and show the window's start and end times (FR lviii). For
  hard pairs like gunshot vs backfire, listening helps more than any confidence number.
- The original model outputs should stay visible after an override (FR lxi). Reviewer
  overrides are also a useful record of where the models fail.

---

## 7. Things to watch for in review

1. Severity colours that mean different things on different screens, or colour-only severity.
2. A Critical alert raised on a Poor or Unusable window.
3. Critical thresholds picked for a good headline number, or picked on the test split.
4. A review queue sorted by time, or where a non-critical doubt outranks a critical near-miss.
5. No escalation after acknowledgement, or a cooldown that hides different events.
6. Slow redraws between detection and display on the live screen.
