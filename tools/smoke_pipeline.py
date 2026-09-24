#!/usr/bin/env python3
"""Orchestration smoke test for ``src/services/pipeline.py`` -- REAL audio, STUB models.

Owner: junaid.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is **not** the acceptance check. It deliberately substitutes the two trained models
with deterministic stubs so that the *wiring* -- preprocessing, quality verdict, feature
extraction, the frozen comparison, severity, the confirmation counter, the review
conditions, budgets and rejection handling -- can be exercised today, on real audio files,
before ``python_models/best`` and ``gtm_model/`` exist.

The acceptance check is ``tools/check_e2e_upload.py``, and it loads the real bundles. A
stub can prove the pipes connect; only the real model can prove the *result* is right, and
the two claims are kept apart on purpose.

Every clip it analyses is real decoded audio (from ``data/generate_corpus.py``); no
synthetic waveform is invented here, so the DSP path is genuinely exercised.

Usage
-----
    .venv/bin/python tools/smoke_pipeline.py
    .venv/bin/python tools/smoke_pipeline.py --clips /tmp/junaid_smoke

Exit status is 0 only when every structural assertion holds.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hashlib  # noqa: E402
import numpy as np  # noqa: E402

from audio_preprocessing.pipeline import AudioPipeline  # noqa: E402
from feature_extraction.features import FeatureExtractor  # noqa: E402
from src.inference.contract import PredictionResult  # noqa: E402
from src.services.config import ConfigStore  # noqa: E402
from src.services.pipeline import (  # noqa: E402
    AnalysisPipeline,
    PipelineModels,
    RepeatTracker,
    evaluate_review,
    severity_block,
)

# --------------------------------------------------------------------------------------
# Stub models.  Clearly labelled as stubs everywhere they surface, so a stub prediction can
# never be mistaken for a real one in a screenshot or a report.
# --------------------------------------------------------------------------------------

class StubPredictor:
    """Returns a fixed confidence distribution, keyed by a crude acoustic cue.

    The class is chosen from a real measurement (spectral centroid), not at random, so
    different clips genuinely produce different results and the comparison/severity/review
    paths get exercised across their branches.
    """

    def __init__(self, name: str, class_names: list[str], *, bias: dict[str, float] | None = None):
        self.model_name = name
        self.model_version = "stub-0.0.0"
        self.class_names = list(class_names)
        self._bias = dict(bias or {})

    def _distribution(self, preprocessed) -> np.ndarray:
        y = np.asarray(preprocessed.samples, dtype=np.float64)
        sr = int(preprocessed.sample_rate or 16000)
        if y.size == 0:
            raise ValueError("stub got empty audio")
        spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size)))
        freqs = np.fft.rfftfreq(y.size, 1.0 / sr)
        centroid = float((spectrum * freqs).sum() / max(spectrum.sum(), 1e-12))
        # Low centroid -> machinery; high -> glass.  Deterministic, and different per clip.
        t = float(np.clip(centroid / 4000.0, 0.0, 1.0))
        raw = np.full(len(self.class_names), 0.02)
        raw[0] = 0.9 * (1.0 - t)          # Machinery Fault
        raw[1] = 0.9 * t                  # Glass Breaking
        if len(self.class_names) > 2:
            raw[2] = 0.25
        for name, value in self._bias.items():
            if name in self.class_names:
                raw[self.class_names.index(name)] += float(value)
        return raw

    def predict_from_preprocessed(self, preprocessed, *, origin="upload", **kwargs) -> PredictionResult:
        started = time.perf_counter()
        raw = self._distribution(preprocessed)
        probs = raw / raw.sum()
        confidences = {c: float(p) for c, p in zip(self.class_names, probs)}
        predicted = max(confidences.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return PredictionResult(
            model_name=self.model_name,
            model_version=self.model_version,
            predicted_class=predicted,
            confidence=confidences[predicted],
            confidences=confidences,
            latency_sec=time.perf_counter() - started,
            feature_version="stub",
            source_origin=origin,
            extra={"stub": True},
        )

    def describe(self) -> dict:
        return {"model_name": self.model_name, "model_version": self.model_version, "stub": True}


# --------------------------------------------------------------------------------------

class Checking:
    def __init__(self) -> None:
        self.checks = 0
        self.failures: list[str] = []

    def ok(self, condition: bool, label: str, detail: str = "") -> bool:
        self.checks += 1
        if not condition:
            self.failures.append(f"{label}{(' -- ' + detail) if detail else ''}")
            print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))
        return bool(condition)

    def report(self, title: str) -> int:
        print()
        if self.failures:
            print(f"{title}: {len(self.failures)} of {self.checks} checks FAILED")
            for failure in self.failures:
                print(f"  - {failure}")
            return 1
        print(f"{title}: all {self.checks} checks passed")
        return 0


def _sha256_of(path: Path) -> str:
    """The clip's bytes, hashed the same way the pipeline hashes them (FR lxxiii)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pipeline(clips: Path) -> AnalysisPipeline:
    store = ConfigStore()
    problems = store.validate()
    if problems:
        raise SystemExit("configuration is invalid:\n  - " + "\n  - ".join(problems))

    classes = store.class_names()
    models = PipelineModels(
        # The stubs are biased in opposite directions on purpose: the GTM stub agrees with
        # the Python stub on a loud, bright clip and disagrees on a dark one, so both the
        # "agree" and "disagree" branches of the comparison get hit.
        python=StubPredictor("Python (stub)", classes, bias={"Glass Breaking": 0.35}),
        gtm=StubPredictor("Google Teachable Machine (stub)", classes, bias={"Machinery Fault": 0.35}),
        preprocessor=AudioPipeline(),
        feature_extractor=FeatureExtractor(),
    )
    return AnalysisPipeline(models, store=store, tracker=RepeatTracker(store.thresholds(), store=store))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=Path("/tmp/junaid_smoke"),
                        help="root of generated clips (originals/<class>/*.wav)")
    args = parser.parse_args()

    clip_paths = sorted(args.clips.rglob("*.wav"))
    if not clip_paths:
        print(f"no .wav clips under {args.clips}. Generate some with:\n"
              f"  .venv/bin/python data/generate_corpus.py --per-class 1 --out-root {args.clips}")
        return 2

    c = Checking()
    pipeline = build_pipeline(args.clips)

    print(f"smoke: {len(clip_paths)} real clips, {len(pipeline.store.class_names())} classes\n")

    # -- 1. warm-up ---------------------------------------------------------------------
    print("[1] start-up warm-up")
    warm = pipeline.warm()
    print(f"    warm-up: {warm.get('total_ms')} ms total  {warm}")
    c.ok(pipeline.warmed is not None, "warm-up recorded timings")
    c.ok("error" not in warm, "warm-up completed without error", str(warm.get("error", "")))

    # -- 2. a full analysis on each clip ------------------------------------------------
    print("\n[2] analyse every clip end to end")
    results = []
    for path in clip_paths:
        result = pipeline.analyse_file(path, location="smoke")
        results.append((path, result))

    analysed = [(p, r) for p, r in results if r["ok"]]
    c.ok(len(analysed) > 0, "at least one clip analysed",
         f"{len(analysed)}/{len(results)} ok")
    for path, r in results:
        if not r["ok"]:
            print(f"    rejected {path.name}: {r.get('rejection', {}).get('message')}")

    path, record = analysed[0]
    print(f"\n    sample: {path.name}")
    print(f"      python : {record['predictions']['python']['predicted_class']} "
          f"({record['predictions']['python']['confidence']:.3f})")
    print(f"      gtm    : {record['predictions']['gtm']['predicted_class']} "
          f"({record['predictions']['gtm']['confidence']:.3f})")
    print(f"      status : {record['comparison']['consistency_status']}")
    print(f"      quality: {record['quality']['verdict']}")
    print(f"      severity: {record['severity']['severity_display']}")
    print(f"      review : {record['review']['matched']}")
    print(f"      {record['elapsed_ms']} ms (budget {record['budget_sec']} s) "
          f"within_budget={record['within_budget']}")

    # -- 3. the shape the UI and the report depend on -----------------------------------
    print("\n[3] mandatory result shape (FR xxiv, lxix)")
    classes = pipeline.store.class_names()
    for _p, r in analysed:
        tag = Path(r["meta"].get("filename", "?")).name
        py = r["predictions"]["python"]
        gtm = r["predictions"]["gtm"]
        c.ok(set(py["confidences"]) == set(classes), f"{tag}: Python gives all 10 classes")
        c.ok(set(gtm["confidences"]) == set(classes), f"{tag}: GTM gives all 10 classes")
        c.ok(abs(sum(py["confidences"].values()) - 1.0) < 1e-3,
             f"{tag}: Python confidences sum to 1", f"{sum(py['confidences'].values())}")
        c.ok(abs(sum(gtm["confidences"].values()) - 1.0) < 1e-3,
             f"{tag}: GTM confidences sum to 1")
        c.ok(len(py["top3"]) == 3, f"{tag}: Python top-3 present")
        c.ok(len(gtm["top3"]) == 3, f"{tag}: GTM top-3 present")
        c.ok(r["comparison"]["consistency_status"] in {
            "Strong Match", "Acceptable Match", "Weak Match",
            "Model Disagreement", "Uncertain Result",
        }, f"{tag}: consistency status in the SRS vocabulary")
        c.ok(isinstance(r["comparison"]["confidence_difference"], float),
             f"{tag}: confidence difference present")
        # Both sides are independently rounded to 6 dp on the way out, so the comparison
        # itself is made at 5 dp -- a tolerance of 1e-6 would fail on the rounding, not on
        # the arithmetic.
        expected_diff = abs(py["confidence"] - gtm["confidence"])
        c.ok(abs(r["comparison"]["confidence_difference"] - expected_diff) < 2e-6,
             f"{tag}: difference == |python - gtm|",
             f"reported {r['comparison']['confidence_difference']:.6f} vs {expected_diff:.6f}")
        c.ok(r["quality"]["verdict"] in {"Good", "Acceptable", "Poor", "Unusable"},
             f"{tag}: quality verdict in vocabulary")
        c.ok(r["severity"]["severity_display"] in pipeline.store.severity_scale(),
             f"{tag}: severity_display on the active scale")
        c.ok(r["review"]["required"] == bool(r["review"]["matched"]),
             f"{tag}: review decision matches its named conditions")
        c.ok("needed" in r["alert"] and "consecutive" in r["alert"],
             f"{tag}: confirmation counter exposed")
        c.ok(bool(r["config_snapshot"]["content_hashes"]), f"{tag}: config snapshot recorded")

    # -- 4. budgets ---------------------------------------------------------------------
    print("\n[4] performance budgets")
    worst = max(r["elapsed_ms"] for _p, r in analysed)
    print(f"    worst upload analysis: {worst:.0f} ms (budget {analysed[0][1]['budget_sec'] * 1000:.0f} ms)")
    c.ok(all(r["within_budget"] for _p, r in analysed),
         "every upload analysis within the configured budget",
         f"worst {worst:.0f} ms")
    c.ok("warm" not in str(warm), "warm-up did not need to be repeated")

    # -- 5. rejection is a result, not a crash -------------------------------------------
    print("\n[5] rejection states")
    import soundfile as sf

    silent = REPO_ROOT / "data" / "tmp" / "smoke_silence.wav"
    silent.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(silent), np.zeros(16000 * 2, dtype=np.float32), 16000)
    silent_result = pipeline.analyse_file(silent, location="smoke")
    print(f"    2 s of silence -> ok={silent_result['ok']} status={silent_result['status']} "
          f"reason={silent_result.get('rejection', {}).get('code')}")
    c.ok(silent_result["ok"] is False, "silent clip is refused rather than analysed")
    c.ok(bool(silent_result.get("rejection", {}).get("message")),
         "refusal carries a human-readable message")
    c.ok(silent_result["quality"]["verdict"] == "Unusable",
         "silence is reported as Unusable quality")

    tiny = REPO_ROOT / "data" / "tmp" / "smoke_tiny.wav"
    sf.write(str(tiny), np.zeros(1600, dtype=np.float32), 16000)
    tiny_result = pipeline.analyse_file(tiny, location="smoke")
    print(f"    0.1 s clip     -> ok={tiny_result['ok']} reason="
          f"{tiny_result.get('rejection', {}).get('code')}")
    c.ok(tiny_result["ok"] is False, "over-short clip is refused")
    c.ok("timings_ms" in tiny_result, "a refusal still reports timings")

    # -- 6. repeated-detection confirmation ----------------------------------------------
    print("\n[6] critical-event confirmation (FR xl / FR xlvi)")
    tracker = RepeatTracker(pipeline.store.thresholds(), store=pipeline.store)

    def feed(quality, agreed, class_name, n):
        out = []
        for i in range(n):
            out.append(tracker.observe(class_name=class_name, agreed=agreed, quality=quality,
                                       confidence=0.9, at=1000.0 + i * 2.0, source="live"))
        return out

    needs = tracker.needed
    r1 = feed("Good", True, "Gunshot", needs - 1)
    print(f"    {needs - 1} consecutive Gunshot windows -> "
          f"confirmed={r1[-1]['confirmed']} consecutive={r1[-1]['consecutive']}")
    c.ok(not r1[-1]["confirmed"], "a single window never raises a critical alert")
    r2 = feed("Good", True, "Gunshot", 1)
    print(f"    the {needs}th consecutive window        -> confirmed={r2[0]['confirmed']}")
    c.ok(r2[0]["confirmed"], f"{needs} consecutive windows confirm the alert")

    tracker.reset()
    r3 = feed("Good", False, "Gunshot", needs + 1)
    print(f"    {needs + 1} windows, models disagreeing -> confirmed={r3[-1]['confirmed']}")
    c.ok(not r3[-1]["confirmed"], "model disagreement blocks confirmation")

    tracker.reset()
    r4 = feed("Poor", True, "Gunshot", needs + 1)
    print(f"    {needs + 1} windows, Poor quality       -> confirmed={r4[-1]['confirmed']}")
    c.ok(not r4[-1]["confirmed"], "Poor quality blocks confirmation")

    tracker.reset()
    feed("Good", True, "Gunshot", 1)
    mixed = tracker.observe(class_name="Glass Breaking", agreed=True, quality="Good",
                            confidence=0.9, at=1005.0)
    print(f"    Gunshot then Glass Breaking            -> "
          f"consecutive={mixed['consecutive']} (streak reset)")
    c.ok(mixed["consecutive"] == 1, "an intervening class breaks the streak")

    tracker.reset()
    staggered = [tracker.observe(class_name="Gunshot", agreed=True, quality="Good",
                                 confidence=0.9, at=2000.0 + i * (tracker.window_seconds + 1.0))
                 for i in range(needs)]
    print(f"    {needs} windows spread wider than {tracker.window_seconds:.0f} s -> "
          f"confirmed={staggered[-1]['confirmed']} consecutive={staggered[-1]['consecutive']}")
    c.ok(not staggered[-1]["confirmed"],
         "detections spread beyond window_seconds are not 'consecutive'")

    # -- 7. severity / alert gates -------------------------------------------------------
    print("\n[7] severity and the alert gate (FR lii-lv)")
    gun_rule = pipeline.store.rule_for_class("Gunshot")
    blocked = severity_block("Gunshot", store=pipeline.store, confidence=0.61,
                             top_two_margin=0.05, quality="Poor", classes_agree=False)
    print(f"    Gunshot @0.61, margin 0.05, Poor, disagreeing -> "
          f"severity={blocked['severity_display']} eligible={blocked['alert_eligible']}")
    c.ok(blocked["severity_display"] == pipeline.store.display_severity(gun_rule["severity"]),
         "severity is the class's configured value regardless of gates")
    c.ok(blocked["alert_eligible"] is False,
         "a weak, low-quality, disagreed Gunshot does not page anyone")
    c.ok(len(blocked["gates"]) >= 1, "the failing gates are named, not just the verdict")

    strong = severity_block("Gunshot", store=pipeline.store, confidence=0.95,
                            top_two_margin=0.60, quality="Good", classes_agree=True)
    print(f"    Gunshot @0.95, margin 0.60, Good, agreeing   -> eligible={strong['alert_eligible']}")
    c.ok(strong["alert_eligible"] is True, "a strong, clean, agreed Gunshot raises the gate")
    c.ok(strong["critical_class"] is True, "Gunshot is a critical class")

    bg = severity_block("Background Noise", store=pipeline.store, confidence=0.95,
                        top_two_margin=0.60, quality="Good", classes_agree=True)
    print(f"    Background Noise                             -> "
          f"severity={bg['severity_display']} critical={bg['critical_class']}")
    c.ok(bg["critical_class"] is False, "Background Noise is not critical")

    # -- 8. manual review conditions -----------------------------------------------------
    print("\n[8] manual-review conditions (Step 17, FR lvii)")
    disagree = evaluate_review(
        store=pipeline.store, python_class="Gunshot", python_confidence=0.91,
        gtm_class="Vehicle Horn", gtm_confidence=0.88, classes_agree=False,
        confidence_difference=0.03, top_two_margin=0.70,
        consistency_status="Model Disagreement", quality="Good", overlapping=False,
    )
    print(f"    disagreement            -> {disagree['matched']}")
    c.ok("model_disagreement" in disagree["matched"], "disagreement is named as a reason")
    c.ok(disagree["required"] is True, "disagreement queues the event for review")
    c.ok(all(f["reason"] and "{" not in f["reason"] for f in disagree["findings"]),
         "every reason is rendered prose, no unsubstituted placeholders")

    clean = evaluate_review(
        store=pipeline.store, python_class="Background Noise", python_confidence=0.97,
        gtm_class="Background Noise", gtm_confidence=0.96, classes_agree=True,
        confidence_difference=0.01, top_two_margin=0.80,
        consistency_status="Strong Match", quality="Good", overlapping=False,
    )
    print(f"    clean background noise  -> required={clean['required']} "
          f"clean_result={clean['clean_result']} matched={clean['matched']}")
    c.ok(clean["required"] is False, "a clean, agreed, strong result stays out of the queue")

    tight = evaluate_review(
        store=pipeline.store, python_class="Machinery Fault", python_confidence=0.55,
        gtm_class="Machinery Fault", gtm_confidence=0.53, classes_agree=True,
        confidence_difference=0.02, top_two_margin=0.03,
        consistency_status="Strong Match", quality="Good", overlapping=False,
    )
    print(f"    low confidence + tight top-two -> {tight['matched']}")
    c.ok("low_confidence" in tight["matched"], "low confidence is named")
    c.ok("similar_top_classes" in tight["matched"], "a near-tie in the top two is named")

    poor = evaluate_review(
        store=pipeline.store, python_class="Machinery Fault", python_confidence=0.80,
        gtm_class="Machinery Fault", gtm_confidence=0.79, classes_agree=True,
        confidence_difference=0.01, top_two_margin=0.50,
        consistency_status="Strong Match", quality="Poor", overlapping=False,
    )
    c.ok("poor_audio_quality" in poor["matched"], "Poor quality is named as a review reason")

    # -- 9. pipeline rejects unusable audio before it reaches a model --------------------
    print("\n[9] the model path is guarded")
    c.ok(silent_result.get("predictions") is None,
         "no prediction is produced for refused audio")
    c.ok(gun_rule.get("min_confidence") is not None,
         "the Gunshot rule still demands a minimum confidence")

    # -- 10. duplicate audio: exact and near (FR lxxiii, FR lxxiv) ------------------------
    print("\n[10] duplicate and near-duplicate audio (FR lxxiii, FR lxxiv)")
    from src.services.pipeline import (
        DEFAULT_NEAR_DUPLICATE_SIMILARITY,
        audio_fingerprint,
        duplicate_config,
        find_near_duplicate,
        fingerprint_similarity,
    )

    source_clip = path
    y, sr = sf.read(str(source_clip), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Fingerprints are compared on the *preprocessed* signal, because that is what the
    # pipeline stores and therefore what every stored fingerprint was computed from.
    # Comparing a raw 22 050 Hz file against a stored 16 kHz fingerprint would compare two
    # different representations and understate the match.
    def fingerprint_of(samples, sample_rate):
        pre = pipeline.models.preprocessor.preprocess_samples(samples, sample_rate, origin="upload")
        return audio_fingerprint(pre.samples, pre.sample_rate)

    original = fingerprint_of(y, sr)
    c.ok(bool(original), "the clip gets a fingerprint")
    c.ok(pipeline.store.thresholds() is not None, "thresholds are readable")

    exact = pipeline.analyse_file(source_clip, location="smoke", audio_id="A-1")
    c.ok(bool(exact["audio"]["sha256"]), "the exact-duplicate key (sha256) is recorded")
    c.ok(exact["audio"]["sha256"] == _sha256_of(source_clip),
         "sha256 is the hash of the bytes on disk")
    c.ok(exact["audio"]["fingerprint"] == original,
         "the stored fingerprint is computed from preprocessed audio")
    c.ok(exact["duplicate"]["checked"] is True, "near-duplicate checking ran")
    c.ok(exact["duplicate"]["near_duplicate_of"] is None,
         "a clip with nothing to compare against claims no duplicate")

    # -- FR lxxiii: the same file, byte for byte, is the same file ----------------------
    copy_path = REPO_ROOT / "data" / "tmp" / f"copy_{source_clip.name}"
    copy_path.write_bytes(source_clip.read_bytes())
    copy = pipeline.analyse_file(copy_path, location="smoke")
    c.ok(copy["audio"]["sha256"] == exact["audio"]["sha256"],
         "a byte-identical copy has the same sha256, whatever it is called")

    # -- FR lxxiv: re-encoded / trimmed / volume-adjusted copies ------------------------
    reencoded = {}
    for fmt, ext in (("MP3", "mp3"), ("OGG", "ogg"), ("FLAC", "flac")):
        target = REPO_ROOT / "data" / "tmp" / f"reencoded_{source_clip.stem}.{ext}"
        try:
            sf.write(str(target), y, sr, format=fmt)
            back, back_sr = sf.read(str(target), dtype="float32")
        except Exception as exc:  # a codec this build of libsndfile lacks
            print(f"    re-encoded as {ext.upper():4s} -> unavailable in this build ({exc})")
            continue
        if back.ndim > 1:
            back = back.mean(axis=1)
        reencoded[f"re-encoded as {ext.upper()}"] = (back, back_sr)

    variants = {
        "volume-adjusted (30 % level)": (y * 0.3, sr),
        "trimmed (last 10 % cut)": (y[: int(y.size * 0.9)], sr),
        "trailing silence added": (np.concatenate([y, np.zeros(sr, np.float32)]), sr),
        "leading silence added": (np.concatenate([np.zeros(sr // 2, np.float32), y]), sr),
        **reencoded,
    }
    for label, (samples, sample_rate) in variants.items():
        similarity = fingerprint_similarity(original, fingerprint_of(samples, sample_rate))
        print(f"    {label:30s} -> {similarity:.4f}")
        c.ok(similarity is not None and similarity >= DEFAULT_NEAR_DUPLICATE_SIMILARITY,
             f"a {label} copy is still recognised as the same recording",
             f"similarity {similarity}")

    # ... and a genuinely different recording must not be.
    for candidate in clip_paths:
        if candidate == source_clip:
            continue
        cand_y, cand_sr = sf.read(str(candidate), dtype="float32")
        if cand_y.ndim > 1:
            cand_y = cand_y.mean(axis=1)
        similarity = fingerprint_similarity(original, fingerprint_of(cand_y, cand_sr))
        c.ok(similarity is None or similarity < DEFAULT_NEAR_DUPLICATE_SIMILARITY,
             f"a different recording ({candidate.parent.name}) is not called a duplicate",
             f"similarity {similarity}")

    # The caller supplies the comparison set; a match must surface as a review finding, not
    # as a silent merge (FR lxxiv).
    flagged = pipeline.analyse_file(
        source_clip, location="smoke", audio_id="A-2",
        near_duplicate_candidates=[("A-1", fingerprint_of((y * 0.3).astype(np.float32), sr))],
    )
    print(f"    match against a stored 30 %-level copy -> "
          f"{flagged['duplicate']['near_duplicate_of']} "
          f"(similarity {flagged['duplicate']['near_duplicate_similarity']})")
    c.ok(flagged["duplicate"]["near_duplicate_of"] == "A-1",
         "a near-duplicate of a stored clip is reported")
    c.ok("near_duplicate" in flagged["review"]["matched"],
         "a near-duplicate goes to manual review")
    c.ok(flagged["review"]["clean_result"] is False,
         "a near-duplicate is never marked a clean result")
    c.ok(any("A-1" in f["reason"] for f in flagged["review"]["findings"]),
         "the review reason names the recording it matched")

    # A clip must never be compared against itself, or everything looks like a duplicate.
    self_match = pipeline.analyse_file(
        source_clip, location="smoke", audio_id="A-1",
        near_duplicate_candidates=[("A-1", original)],
    )
    c.ok(self_match["duplicate"]["near_duplicate_of"] is None,
         "a clip is not reported as a near-duplicate of itself")

    # FR lxxiv says near-duplicates are *flagged*, never dropped or merged with the original.
    other_clip = next(p for p in clip_paths if p != source_clip)
    other_result = pipeline.analyse_file(other_clip, location="smoke", audio_id="A-3",
                                         near_duplicate_candidates=[("A-1", original)])
    c.ok(other_result["ok"] is True, "a different clip is analysed normally, never dropped")
    c.ok(other_result["duplicate"]["near_duplicate_of"] is None,
         "a different clip is not flagged as a duplicate")

    settings = duplicate_config(pipeline.store.thresholds())
    print(f"    threshold {settings['near_duplicate_similarity']} from {settings['source']}")
    c.ok(isinstance(settings["near_duplicate_similarity"], float),
         "the near-duplicate threshold is configurable")

    return c.report("smoke_pipeline")


if __name__ == "__main__":
    raise SystemExit(main())
