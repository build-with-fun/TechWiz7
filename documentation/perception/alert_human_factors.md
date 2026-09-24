# Alert and manual-review human factors

Author: `amina` (auditory neuroscience / psychoacoustics)
Date: 2026-09-23
Consumers: `sara` (config values, FR liii/lv/lvi/lvii), `junaid` (live loop and alert display),
`zainab` + `hina` (severity presentation), `zainab` (severity is never colour alone),
`raheem` (report: "Critical-event rule design" and "Security/Privacy" chapters).
Numbers, with their justification, are in `recommended_perceptual_params.json` under
`alerting_human_factors`.

This is the part of the project where an acoustic model meets a tired human at 3 a.m. The
neuroscience here is not decorative: it is the reason SRS Step 15 (repeated-detection
confirmation), SRS Step 13 (audio quality) and SRS Step 17 (manual review) exist at all, and it
determines what "severity" is allowed to mean.

---

## 1. What severity must mean (and must not mean)

**Severity answers: how fast must a human act, and who else must know?**
It does **not** answer: how sure is the model?

These are different axes and conflating them is the most common failure of alerting consoles. A
0.99-confidence Vehicle Horn is a certainty about something unimportant; a 0.72-confidence Gunshot
is uncertainty about something that may be a life-safety emergency. If the interface renders them
with the same weight, the operator's attention is allocated by model confidence, which is the wrong
variable.

Concrete rule for the design system: **severity is derived from the class and the rules, and is
displayed independently of the confidence number.** Confidence gets its own visual channel
(a numeric value, a bar, a threshold marker), never the alert colour.

### The five-level question (a real SRS contradiction the team must answer)

- **SRS Step 16** lists five levels: Informational, Low, Medium, High, Critical.
- **SRS FR lii** lists four: Informational, Low, Medium, Critical - **High is missing**.
- Step 16's own worked examples require High: Machinery Fault: High, Glass Breaking: High,
  Alarm or Siren: High, Aggression: High.

So the examples depend on a level the FR list omits. **Recommendation: implement the five-level
scale, with High as a real and distinct level, and make the level set itself config-driven** so an
evaluator can collapse it to four without a code change (SRS 1.8 item 5 - "changing the confidence
threshold" and similar surprises are explicitly promised). Document the contradiction and the
choice in the report; an evaluator may well probe exactly this, and "we noticed, chose the
five-level scale because Step 16's examples depend on it, and made the level set configurable" is a
complete and defensible answer. Note also that my reading of the classes agrees with Step 16's
examples: Alarm or Siren is alert-worthy but is not a life-safety event, and Step 14 listing it
among "critical acoustic events" is not a severity assignment - Step 16 and FR xliii both grade it
**High**. It must **not** carry the same escalation path as Gunshot.

### Correction to a common assumption: severity is NOT a static class -> level table

I read the FR list directly rather than trusting the Step-16 examples, and the SRS makes severity
**context-dependent for three classes**. This matters more than the missing level, because a static
`class -> severity` map in code would silently violate three requirements at once:

| Class | SRS wording | Consequence |
|---|---|---|
| Vehicle Horn | Step 16: "Low or Medium"; FR xliv: stored as a traffic/environmental event | Two allowed levels. Base **Low**; escalate to **Medium** when confirmed across consecutive windows at Good quality. |
| Aggression | Step 16: "High"; **FR xlviii: "high or critical alert depending on confidence and repeated detection"** | Two allowed levels, and the escalation condition is named by the SRS itself. Base **High**; escalate to **Critical** on confidence + repeated detection. |
| Background Noise | Step 16: "Informational"; **FR l: "normally non-critical unless the estimated noise level exceeds a configured limit"** | Escalates above Informational on a **configured** noise-level limit. Base **Informational**. |

FR xlviii is the clearest statement in the whole document that severity is *derived*, not looked up:
"depending on confidence and repeated detection". FR l is the reason the noise-floor estimator
feeds the alert path and not just the quality badge.

**Consequence for `sara`'s config (FR liii** lists exactly these as administrator-configurable:
critical categories, minimum confidence, repeated-detection requirement, audio-quality requirement,
model-agreement requirement, alert severity, recommended action **)**: the rule file must express,
per class, a **base level plus named escalation conditions**, not a single level. A one-level-per-class
schema is a hard-coded answer wearing a config file's clothes.

---

## 2. Why repeated-detection confirmation exists (startle, habituation)

The acoustic startle reflex has an onset latency of roughly 30-40 ms - it is one of the fastest
auditory-motor responses a human has, faster than any voluntary reaction. A critical alert is
therefore asking for a response on a reflex-like timescale, which is why the app's three-second
promise matters and why a full-page reload or a chart redraw between detection and display is not a
cosmetic cost.

But the startle response also **habituates**, and it does so quickly: the magnitude of the blink
component falls substantially within 5-10 presentations of the same stimulus (Blumenthal et al.
2005, *Psychophysiology* 42(1):1-15). Operationally this is the whole argument for SRS Step 15. A
single isolated detection - especially an unstable one - is exactly the kind of event a nervous
system, and an operator, learns to stop responding to. Requiring the same event in **consecutive
windows** plus a confidence floor changes the stimulus from "something twitched" into "the same
event persists in time", which is a far more habituation-resistant trigger.

This is also the reason the **hop size matters** (`recommended_perceptual_params.json`,
`live_window`): with a non-overlapping 2 s window scheme, a 0.3-0.6 s gunshot can be entirely inside
one window, so the consecutive-detection requirement can never be satisfied and the Gunshot alert
can never fire. The confirmation rule the SRS demands is only implementable if the windows overlap.

---

## 3. Why false alarms are a physiological problem, not a UX preference

I will fight any threshold choice that ignores this, so here is the evidence.

- In clinical monitoring, **72-99% of alarms are non-actionable** (Cvach 2012, *Biomed Instrum
  Technol* 46(4):268-277; Sendelbach & Funk 2013, *AACN Adv Crit Care* 24(4):378-386), and alarm
  fatigue is a recognised patient-safety hazard (The Joint Commission Sentinel Event Alert 50,
  2013).
- In automation, operators who experience false alarms first distrust and then **ignore** the
  system - the "cry wolf" / alarm-mistrust effect, measured experimentally (Bliss & Dunn 2000,
  *Ergonomics* 43(9):1393-1407; Parasuraman & Riley 1997, *Human Factors* 39(2):230-253).

The mechanism matters for the design: the failure is not that a human is mildly annoyed. It is that
the response to the *real* event is **attenuated** by prior exposure to false ones, and the
attenuation happens at the reflex level before conscious evaluation. A critical-event monitor that
cries wolf does not merely have bad numbers; it has removed its own ability to summon a response.
That is why SRS FR lxxviii lists "excessive critical alerts" as an anomaly an administrator must be
alerted about - the SRS itself treats alert volume as a failure mode.

**The operational consequence, stated as a number:** a false-alarm budget of **1 false critical
alert per 8-hour shift** (0.125/h). Thresholds for critical classes should be chosen on the
VALIDATION split as the lowest thresholds meeting that budget, never chosen on the test split, and
reported together with the operating point. On this dataset's validation split (450 clips, ~22-23
per class under the SRS's 2100/450/450 split) a single false positive is already about 4% of a
class - so the budget implies a high threshold on critical classes at this data size. Report what
is actually measurable; do not claim a budget that was not evaluated.

---

## 4. The quality gate is a masking argument, not a nicety

SRS Step 15 requires "acceptable audio quality" as a condition for confirming a critical event, and
Step 13/FR xxxvii requires the Good/Acceptable/Poor/Unusable verdict. The perceptual reason is
energetic masking: below roughly 10 dB in-band SNR the event is not reliably audible to a human at
all, and below 0 dB it is not perceptually present. A classifier given such a window will still
output a distribution - softmax always does - and can be confidently wrong.

And there is a nastier case: **clipping**. Heavy clipping generates broadband harmonic and
intermodulation distortion with an impulsive envelope - which is the acoustic signature of the two
most alarming classes. A heavily clipped recording of a loud benign sound can therefore produce a
high-confidence **critical false positive**. So clipping must gate critical alerts, not just
decorate a quality badge. This is the strongest single argument for the SRS's own quality
requirement and it should appear in the report's critical-event rule design.

---

## 5. Acknowledgement, escalation and the alarm storm

- **Acknowledge or escalate within 60 s.** An unacknowledged alert must escalate; otherwise
  "unacknowledged" and "nobody is watching" are indistinguishable. 60 s is long enough not to
  punish a human who is mid-task and short enough to matter in an emergency.
- **Re-notification cooldown of 300 s per event class.** Suppressing repeats of the *same* event
  within a cooldown is what prevents an alarm storm. Alarm storms are dangerous not because they
  are noisy but because the standard human response - silencing the whole system - removes
  protection against every *other* event too.
- **Distinct events still alert.** A cooldown must never suppress a different class.
- **Escalate the treatment with severity.** If Informational and Critical look and sound identical,
  the tenth Critical of the shift gets the attention of an Informational. Salience must be spent
  where it matters.
- **Severity never by colour alone** (WCAG 1.4.1, and roughly 8% of men have some red-green colour
  vision deficiency). Every severity carries an explicit label and icon. This is `zainab`'s call on
  the tokens but it is not optional.

---

## 6. Review-queue priority: designing for a tired reviewer at 3 a.m.

A queue ordered by arrival time is unusable under load, because it asks the reviewer to spend
attention in proportion to how *early* something was detected rather than how *costly* a wrong call
would be. Order by expected cost of being wrong:

```
priority = severity_weight * (1 - effective_confidence) * quality_penalty + critical_miss_bonus
```

with the weights in `recommended_perceptual_params.json`. The reasoning:

- **severity_weight** carries the consequence. Gunshot 10 vs Background Noise 1. A wrong call on a
  Critical event is the only class of error that is effectively unrecoverable.
- **(1 - effective_confidence)** carries the probability the model is wrong, using the combined
  Python/GTM evidence rather than either model alone.
- **quality_penalty** raises anything the pipeline already distrusts on physical grounds.
- **critical_miss_bonus** is the important term and the one a naive design omits: it fires when a
  critical class is a substantial *runner-up* but not the top prediction, or when the two models
  disagree on a critical class. **A missed gunshot is the failure the whole system exists to
  prevent**, so it must outrank a cosmetic uncertainty about Background Noise even though the latter
  is "more uncertain" by every probability measure. This is an asymmetric-cost decision, not a
  probability-ranking decision.

Two more queue requirements that follow from the human factors:

- **Playback must be one click and must show the window's start/end timestamps** (FR lviii). The
  reviewer's fastest route to a correct call on P01 (Gunshot vs Backfire) is context, not
  confidence - a human hearing the clip has information the model does not.
- **The original model outputs must remain visible after an override** (FR lxi, SRS lxi). This is
  not just an audit requirement; it is how the team learns which pairs the models actually fail on
  in the field, since the reviewer's overrides are a labelled error stream.

---

## 7. What I will challenge in review

1. A severity colour that changes meaning between screens, or severity conveyed only by hue.
2. A "Critical" alert raised on a Poor or Unusable quality window without an audible-margin gate.
3. Critical-class thresholds chosen for a headline number rather than against the false-alarm
   budget, or chosen on the test split.
4. A review queue sorted by time, or one where a non-critical uncertainty outranks a critical
   near-miss.
5. An acknowledgement path with no escalation, or a cooldown that suppresses distinct events.
6. Any UI that blocks or redraws heavily between detection and display on the live screen. The
   startle latency argument means every hundred milliseconds there is spent out of a reflex window.
