# Which classes get confused, and why

> Archived design notes from 23 Sep, written while the corpus was still procedurally
> generated. The reasoning about which classes get confused still holds; measured numbers
> refer to that earlier corpus.

Date: 2026-09-23

The aim was to sort confusions between the ten classes into ones we can fix (by better
features or data) and ones we can't (the information just isn't in the audio), so effort
goes where it can help and we don't claim to have solved a pair that can't be solved.

The reference point is a human listener. If an attentive person with a clean recording
can't tell two sounds apart, our features won't either, and the right answer is a review
route and an honestly reported error rate, not a lower threshold.

---

## Four kinds of confusion

1. **Same physics.** Two sources make almost identical sounds (gunshot vs backfire). Can't be
   fixed; send to review and report the error rate.
2. **Same short-term spectrum, different behaviour over a second** (siren vs horn; glass vs
   metal impact). Fixable with the right features, because the information is there.
3. **Same voice, used differently** (scream vs shout; scream vs help phrase). Fixable with
   voice-quality features.
4. **Overlapping label definitions** (aggression vs normal conversation; help phrase vs any
   speech). Only fixable by writing down the definition and sticking to it when labelling.

Effort should go into 2 and 3. Number 4 is about labelling discipline, and number 1 should be
reported and routed to review.

---

## Pair by pair

Details, features, augmentation plans and expected ceilings are in
`class_confusability.json` (pairs `P01`-`P12`). Summary:

| ID | Pair | Kind | Verdict |
|---|---|---|---|
| P01 | Gunshot vs vehicle backfire | 1 | **Can't be separated reliably.** Both are impulsive combustion. Ceiling about 0.6-0.75. |
| P02 | Gunshot vs fireworks | 2 | Fixable: a full firework has a crackling multi-pop tail; a clip of just the bang doesn't. |
| P03 | Alarm or Siren vs Vehicle Horn | 2 | **The most useful fix**: modulation spectrum at 0.5-4 Hz. About 0.6-0.75 without it, 0.90-0.96 with it. |
| P04 | Panic Scream vs Aggression/shouting | 3 | Fixable: roughness (30-150 Hz AM), jitter, HNR. |
| P05 | Aggression vs normal conversation | 4 | **Partly unfixable.** Needs a written labelling rule, not a better model. |
| P06 | Glass Breaking vs metal/machinery impact | 2 | Fixable with decay features, not onset features. |
| P07 | Machinery Fault vs healthy machinery | 1 (same source, different context) | Only with training data at a range of SNRs, plus an audibility check. |
| P08 | Person Asking for Help vs normal speech | 4 | The five fixed SRS phrases are the only reliable separator; the number of different speakers is what limits it. |
| P09 | Animal Sound vs Alarm or Siren | 2 | Fixable with how regular the tonal track is. (Not in the SRS list; found by listening.) |
| P10 | Panic Scream vs Person Asking for Help | 3 | **Both are critical and both count toward the recall target.** Check this pair first. |
| P11 | Glass Breaking vs Gunshot | 2, plus both impulsive | Fixable with high-frequency energy ratio and decay features. |
| P12 | Alarm or Siren vs Background Noise | masking | Not a model problem: below about 0 dB in-band SNR the siren simply isn't audible. |

### Gunshot vs fireworks and backfire (P01, P02)

A gunshot, a backfire and a firework are all sudden releases of gas: a fast rise, broadband
energy dominated by a low-frequency blast, and a decay shaped by the surroundings. From the
transient alone it's genuinely ambiguous, which is why gunshot-detection systems use several
microphones, timing and context. Fireworks are easier because a whole firework has a long
crackling tail that a gunshot doesn't, but only if the clip keeps that tail. Backfire is the
hard one; better to report the error rate than chase a number.

### Panic scream vs shouting (P04)

This one can be separated, for a specific reason. Arnal et al. (2015, *Current Biology*
25(15):2051-2056) found that screams have a fast amplitude modulation in the tens-of-Hz
"roughness" range that other loud voices don't, which is what makes them grab attention.
Shouting keeps a steady (if raised) pitch and clear words; a scream breaks into subharmonics.
Measure the modulation spectrum around 30-150 Hz plus jitter, shimmer and harmonic-to-noise
ratio. Both classes are critical, so it's worth the effort.

---

## The critical-vs-critical voice pairs matter most

Three of the five critical classes are human voices (Panic Scream, Aggression, Person Asking
for Help). Mixing them up either raises the wrong critical alert or misses one, which hurts
both critical recall and operator trust.

**Every error analysis should start with P10 (scream vs help), then P04 (scream vs
aggression).** If those are clean, the recall target is mostly a question of the threshold;
if they aren't, no threshold will fix them.

---

## Masking (P12) and the quality requirement

The SRS requires quality analysis (Step 13, FR xxxvii) and acceptable quality before a
critical event is confirmed (Step 15). That second rule is necessary, not just cautious.

If a sound isn't about 10 dB above the background in its own frequency band, a listener
won't hear it, however hard they try. A classifier given the same window still outputs
probabilities and can be confidently wrong. That's behind the "excessive critical alerts"
anomaly in FR lxxviii and many low-SNR false alarms.

So the pipeline needs an audibility check (per-band SNR in the event's own bands) before any
critical alert. Below the margin, the right output is "Poor or Unusable quality, manual
review", not an alert and not a low-confidence guess.

---

## What this means for each part of the work

**Data and augmentation.** Person Asking for Help and Panic Scream will decide the headline
numbers. They need many different speakers and voice qualities, not more of the same: one
speaker recorded 300 times teaches the model that speaker. The augmentation priorities are in
`class_confusability.json` (`augmentation_priority`); the top one, mixing every class with
noise at +15/+10/+5/0 dB SNR, follows from SRS 1.8 item 9 (hidden tests include low volume,
echo, distance and overlap). Judge the dataset against that plan, not just the 300-per-class
count.

**Models.** Don't add model capacity to attack P01 and P05; the information isn't there and
the model will just fit label noise. Do add modulation and decay features (P03, P04, P06,
P11), which is where gains are possible. In the error analysis, name the pairs that are still
confused, at a stated threshold.

**Model selection.** A model picked on overall accuracy might be great at Background Noise and
Vehicle Horn and poor at the five critical classes. Select on macro-F1 plus per-class critical
recall at the operating point, and say which model wins on which measure.

**App.** P01, P05 and P12 are why the review queue exists. Send disagreements on a critical
class to review with high priority, and give the reviewer the audio (FR lviii); someone
listening can use context the model doesn't have.

**Report.** The reasons for the review queue, the quality gate and repeated-detection
confirmation are all here and in `recommended_perceptual_params.json`. If asked why a
low-quality recording doesn't raise a critical alert, the answer is masking.

---

## Limits of these notes

- These are predictions from how hearing works, not measurements on this dataset. They only
  become evidence once checked against real confusion matrices.
- The "expected ceilings" are estimates, there to stop us promising numbers we can't reach,
  not results to quote.
- The cited papers should be checked against the originals before anything from here goes
  into the final report.
- This was written before the dataset existed. Measuring each class's variety (sources,
  speakers, environments) will predict performance better than any of the above.
