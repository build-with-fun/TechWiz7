# Class-similarity and confusability analysis

Author: `amina` (auditory neuroscience / psychoacoustics)
Date: 2026-09-23
Companion: `class_confusability.json` (machine-readable, for `omar`, `bilal`, `nadia`, `lorena`, `sara`),
`recommended_perceptual_params.json`, `feature_rationale.md`.

The purpose of this document is to tell the team **where confusion between the ten classes is a
pipeline failure and where it is a limit of the information available**, so that modelling effort
and dataset effort go to the places where they can change the result - and so that nobody claims to
have solved a pair that is not solvable and gets caught doing it in front of an evaluator.

The reference point throughout is the human listener. Where a person with full attention and a
clean recording cannot separate two sounds either, no feature set this project can build will
separate them, and the honest response is a manual-review route and a reported residual rate - not
a lower threshold.

---

## The organising idea: four mechanisms of confusion

Every confusable pair in this task reduces to one of four mechanisms, and which mechanism applies
determines the fix:

1. **Shared generating physics.** Two sources produce genuinely near-identical waveforms (gunshot
   vs backfire). *Not fixable. Review route + reported residual.*
2. **Shared short-time spectrum, different long-time behaviour.** Frame-level features see the same
   thing, but the two sounds evolve differently over a second (siren vs horn; glass vs metal
   impact). *Fixable with the right feature family - the information exists.*
3. **Same organ, different mode.** Human voice doing two things (scream vs shout; scream vs help
   phrase). *Fixable with voice-quality features.*
4. **Label definition, not acoustics.** The classes overlap because of how they were defined
   (aggression vs ordinary conversation; help phrase vs any speech). *Fixable only by writing the
   definition down and applying it consistently - never by modelling.*

Mechanisms 2 and 3 are where the team should spend effort. Mechanism 4 is a dataset-discipline
problem. Mechanism 1 should be reported and routed, not fought.

---

## Per-pair findings

Full detail, features, augmentation plans and expected ceilings are in `class_confusability.json`
as pairs `P01`-`P12`. Summary:

| ID | Pair | Mechanism | Verdict |
|---|---|---|---|
| P01 | Gunshot vs Vehicle Backfire | 1 - shared physics | **Irreducible.** Both are impulsive combustion events. Ceiling ~0.6-0.75. |
| P02 | Gunshot vs Fireworks | 2 - long-time behaviour | Fixable: whole firework events have a crackling multi-pop tail; a burst-only clip is not. |
| P03 | Alarm or Siren vs Vehicle Horn | 2 - long-time behaviour | **Highest-value fix in the project**: modulation spectrum 0.5-4 Hz. 0.6-0.75 without it, 0.90-0.96 with. |
| P04 | Panic Scream vs Aggression / shouting | 3 - phonation mode | Fixable: roughness (30-150 Hz AM), jitter, HNR. |
| P05 | Aggression vs ordinary conversation | 4 - label definition | **Partially irreducible.** Needs a written labelling rule, not a model. |
| P06 | Glass Breaking vs Metal/Machinery impact | 2 - long-time behaviour | Fixable with decay-phase features, not onset features. |
| P07 | Machinery Fault vs healthy machinery noise | 1 - shared source, different context | Fixable only with SNR-stratified training data; needs an audible-margin gate. |
| P08 | Person Asking for Help vs ordinary speech | 4 - label definition | The five-phrase SRS restriction is the only reliable separator. Speaker diversity is the binding constraint. |
| P09 | Animal Sound vs Alarm or Siren | 2 - spectral overlap | Fixable with tonal-track regularity. (Found by inspection; not in the SRS list.) |
| P10 | Panic Scream vs Person Asking for Help | 3 - same organ | **Both are Critical, both count toward the recall floor.** Check first in every error analysis. |
| P11 | Glass Breaking vs Gunshot | 2 + shared impulsive character | Fixable with high-frequency energy ratio + decay features. |
| P12 | Alarm or Siren vs Background Noise | 3' - energetic masking | Not a model problem at all. Below ~0 dB in-band SNR the signal is not perceptually present. |

### On the two pairs the SRS names that I want to be explicit about

**Gunshot vs firework / backfire (P01, P02).** A muzzle blast, a backfire and a firework burst are
all impulsive release of gas: fast rise, broadband energy with a low-frequency-dominated blast, and
an exponential decay whose shape is set by the environment. Given only the isolated transient, the
inference is not merely difficult, it is ambiguous - which is why ballistic-acoustic systems use
sensor arrays, timing and context rather than a single microphone. Fireworks are the easier half of
the problem because a whole firework event has an extended crackling tail (many small stochastic
onsets) that a gunshot does not; if the dataset stores only the burst, the team has thrown away the
one feature that separates them. Backfire is the hard half, and I would rather the team reported an
honest residual than chased a number.

**Panic scream vs normal shouting (P04).** This is separable, and the reason is specific and
non-obvious, so it should be implemented deliberately rather than left to MFCCs. Arnal et al.
(2015, *Current Biology* 25(15):2051-2056) showed that screams carry a distinctive fast amplitude
modulation - tens of Hz, in the roughness range - that distinguishes them from other loud
vocalisations and gives them their attention-capturing property. Shouting preserves a stable though
elevated F0 and intact phonemes; a panic scream breaks into subharmonics and biphonation. Compute
the modulation spectrum around 30-150 Hz and the voice-quality measures (jitter, shimmer,
harmonics-to-noise ratio). This pair is doubly worth the effort because *both* classes are Critical
severity.

---

## The two Critical-vs-Critical pairs are the ones that matter most

Three of the five NFR-critical classes are human voice (Panic Scream, Aggression, Person Asking for
Help) and two of those three are Critical severity. A confusion between them is not a neutral
statistical event: it is a false critical alert (if the wrong one is raised) or a missed critical
event (if a scream is filed as conversation). Both damage the two things the project is scored on -
critical-class recall and operator trust.

**Every error analysis must therefore begin with P10 (Scream vs Help), then P04 (Scream vs
Aggression), before looking at anything else.** If those two pairs are clean, the critical-class
recall floor is largely a matter of the operating point; if they are not, no threshold will fix
them.

---

## The masking problem (P12) - and why it justifies the SRS's own quality requirement

The SRS requires audio-quality analysis (Step 13, FR xxxvii) and requires acceptable quality as one
of the conditions for confirming a critical event (Step 15). Perceptually, that second requirement
is not a stylistic precaution - it is necessary.

When a signal's energy inside its critical band does not exceed the masking sound's by roughly
10 dB, the signal is not perceptually present: a listener does not hear a weak siren buried in
traffic noise, however hard they attend. A classifier given the same masked window will still emit
a probability distribution, and it is entirely capable of emitting a confident wrong answer, because
softmax over a set of classes always produces something. This is the mechanism behind the
"excessive critical alerts" failure mode in SRS FR lxxviii and behind low-SNR false positives.

**Consequence for the pipeline:** an audible-margin gate (per-band SNR in the event's own bands)
must sit in front of critical alerting. Below the configured margin, the honest output is "Unusable
or Poor quality - manual review", not a critical alert and not a low-confidence guess. Below 0 dB
in-band SNR the correct answer is a physical one: the information is gone.

---

## What the team should do with this

**To `omar` (data sourcing and augmentation).** The two classes that will decide the project's
headline numbers are **Person Asking for Help** and **Panic Scream** - not because they are hard to
source in bulk, but because they need *speaker diversity* and *voice-quality variety*, not more of
the same. One speaker recorded 300 times will produce a model that recognises that speaker. The
augmentation priority order in `class_confusability.json` (`augmentation_priority`) is ranked, and
the top item - SNR-stratified mixing at +15/+10/+5/0 dB for every class - is mandated by SRS 1.8
item 9, which states that hidden tests contain low volume, echo, distant sources and overlapping
sounds. Sourcing should be judged against the augmentation plan, not against the raw 300-per-class
count alone.

**To `bilal` and `nadia` (models).** Do not add capacity to attack P01 and P05 - the information is
not in the signal and extra capacity will simply fit the label noise. Do add the modulation-spectrum
and decay-phase features (P03, P04, P06, P11) - those four pairs are where a measurable gain is
available. Report whichever pairs remain confused, by name, in the error analysis, and report them
at a stated threshold. A confusion that is predicted by this document and confirmed by the matrix
is a *validated limitation*; the same confusion appearing without explanation is a defect.

**To `lorena` (selection).** The critical classes are three human-voice classes and two impulsive
classes. A model selected purely on overall accuracy may be a model that has learned Background
Noise and Vehicle Horn very well and the five classes the NFR cares about poorly. Select on
macro-F1 plus per-class critical recall at the operating point, and say which model wins on which
criterion when they differ.

**To `sara` / `junaid` (app).** P01, P05 and P12 are the pairs that justify the manual-review queue
existing at all. Route disagreement on a *critical* class to review with a high priority, and give
the reviewer the audio (FR lviii) - a reviewer hearing the clip can separate Gunshot from Backfire
using context the model does not have, which is precisely why the SRS requires a human in the loop.

**To `raheem` (report and defence).** The justification for the review queue, for the quality gate,
and for repeated-detection confirmation are all in this document and in
`recommended_perceptual_params.json`. If an evaluator asks "why does a low-quality recording not
raise a critical alert?", the answer is energetic masking and it is defensible with literature.

---

## Honest limitations of this analysis

- These are perceptual predictions, not measurements on this dataset. They become evidence only
  when checked against the real confusion matrices - which is a task I intend to take once the
  first models are trained. Until then they are hypotheses with strong prior support.
- The "expected ceilings" are judgements from the mechanisms, not measurements. They are there to
  stop the team promising a number it cannot reach, not to be quoted as results.
- The literature cited is real and is listed with links in `sofia`'s research briefs; I have not
  independently re-verified every figure quoted here in this session. Anything that ends up in the
  final report should be checked against the primary source before publication - I will ask `sofia`
  to confirm the exact page/DOI for each before the report is frozen.
- I have not seen the actual dataset yet. When the manifest exists, the per-class variance (how
  many distinct sources, speakers and environments per class) can be measured directly, and that
  measurement will be a better predictor of per-class performance than any of the above.
