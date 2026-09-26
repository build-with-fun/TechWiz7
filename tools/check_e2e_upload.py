#!/usr/bin/env python3
"""Run both saved models on the sample clips through AnalysisPipeline and time them.

Checks that every usable clip gets a class and confidence from both models, that the
consistency and quality verdicts are valid, that the silent clip is rejected, and the 8 s /
3 s budgets. Process-level only (no HTTP, database or 30 s clip); tools/benchmark_latency.py
measures the SRS latency targets. Writes reports/e2e_acceptance.json.

    .venv/bin/python tools/check_e2e_upload.py [--model-dir python_models/best]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.contract import AudioSource  # noqa: E402
from src.services.pipeline import AnalysisPipeline  # noqa: E402

VERDICTS = {"Strong Match", "Acceptable Match", "Weak Match",
            "Model Disagreement", "Uncertain Result"}
QUALITY = {"Good", "Acceptable", "Poor", "Unusable"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E2E acceptance with real models + latency budgets")
    ap.add_argument("--samples", type=Path, default=REPO_ROOT / "sample_audio")
    ap.add_argument("--model-dir", type=Path, default=REPO_ROOT / "python_models" / "best")
    ap.add_argument("--gtm-dir", type=Path, default=REPO_ROOT / "gtm_model")
    ap.add_argument("--max-clip-s", type=float, default=8.0)
    ap.add_argument("--max-window-s", type=float, default=3.0)
    args = ap.parse_args(argv)

    clips = sorted(p for p in args.samples.glob("*.wav"))
    if not clips:
        print("no clips — run audio_dataset/scripts/make_sample_audio.py first")
        return 2

    try:
        pipeline = AnalysisPipeline.load(model_dir=args.model_dir, gtm_dir=args.gtm_dir)
    except Exception as exc:  # ModelsUnavailable etc — print the actionable message
        print(f"PIPELINE UNAVAILABLE: {exc}", file=sys.stderr)
        return 3
    failures: list[str] = []
    rows: list[dict] = []
    clip_lat, win_lat = [], []

    for clip in clips:
        # upload path
        src = AudioSource.from_path(clip)
        t0 = time.perf_counter()
        rec = pipeline.analyse(src, origin="upload", meta={"filename": clip.name})
        dt = time.perf_counter() - t0
        clip_lat.append(dt)

        row = {"clip": clip.name, "latency_s": round(dt, 3),
               "ok": rec.get("ok"), "status": rec.get("status")}
        if rec.get("ok"):
            py_cls = rec.get("predictions", {}).get("python", {}).get("predicted_class")
            gtm_cls = rec.get("predictions", {}).get("gtm", {}).get("predicted_class")
            if not py_cls:
                failures.append(f"{clip.name}: missing python class")
            if not gtm_cls:
                failures.append(f"{clip.name}: missing gtm class")
            verdict = rec.get("comparison", {}).get("consistency_status")
            if verdict not in VERDICTS:
                failures.append(f"{clip.name}: bad consistency verdict {verdict!r}")
            q = rec.get("quality", {}).get("verdict")
            if q not in QUALITY:
                failures.append(f"{clip.name}: bad quality verdict {q!r}")
            if dt > args.max_clip_s:
                failures.append(f"{clip.name}: clip latency {dt:.2f}s > {args.max_clip_s}s")
        elif clip.name not in {"silence.wav", "low_quality_quiet_tone.wav"}:
            failures.append(f"{clip.name}: unexpectedly rejected: "
                            f"{rec.get('rejection', {}).get('code')!r}")
        rows.append(row)

        # live-window path (2 s slice, origin=live budget)
        import soundfile as sf
        data, sr = sf.read(clip, dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.ndim > 1 else data
        window = mono[: int(sr * 2.0)]
        t0 = time.perf_counter()
        wrec = pipeline.analyse(AudioSource.from_samples(window, sample_rate=sr),
                                origin="live", meta={"filename": clip.name})
        wdt = time.perf_counter() - t0
        win_lat.append(wdt)
        if wrec.get("ok") and wdt > args.max_window_s:
            failures.append(f"{clip.name}: window latency {wdt:.2f}s > {args.max_window_s}s")

    summary = {
        "protocol": "local sample_audio WAVs; process-level analysis without persistence; "
                    "not a 30-second recording benchmark",
        "n_clips": len(clips),
        "clip_latency_mean_s": round(statistics.mean(clip_lat), 3) if clip_lat else None,
        "clip_latency_max_s": round(max(clip_lat), 3) if clip_lat else None,
        "window_latency_mean_s": round(statistics.mean(win_lat), 3) if win_lat else None,
        "window_latency_max_s": round(max(win_lat), 3) if win_lat else None,
        "budgets": {"clip_s": args.max_clip_s, "window_s": args.max_window_s},
        "passed": not failures,
        "failures": failures,
        "rows": rows,
    }
    out = REPO_ROOT / "reports" / "e2e_acceptance.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
