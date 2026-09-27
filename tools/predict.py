#!/usr/bin/env python3
"""Try the models on an audio file from the command line.

Runs a clip (or a folder) through the Python model (python_models/best) and the Teachable
Machine model (gtm_model/) and prints both predictions, how well they agree, and the
final decision.

    .venv/bin/python tools/predict.py sample_audio/gunshot.wav
    .venv/bin/python tools/predict.py sample_audio/                 # every .wav in a folder
    .venv/bin/python tools/predict.py my_clip.wav --top 5

Nothing is saved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _scores(block: dict) -> list[tuple[str, float]]:
    """Per-class confidences (full list if available, else top 3)."""
    conf = block.get("confidences")
    if isinstance(conf, dict) and conf:
        return sorted(conf.items(), key=lambda kv: -kv[1])
    return [(row["class"], row["confidence"]) for row in block.get("top3", [])]


def _bar(p: float, width: int = 22) -> str:
    filled = int(round(min(max(p, 0.0), 1.0) * width))
    return "#" * filled + "-" * (width - filled)


def _show(name: str, block: dict, top: int) -> None:
    print(f"  {name}")
    print(f"    predicted : {block.get('predicted_class')}  "
          f"({block.get('confidence', 0.0):.3f})")
    for cls, p in _scores(block)[:top]:
        mark = "<=" if cls == block.get("predicted_class") else "  "
        print(f"      {p:6.3f} {_bar(p)} {cls} {mark}")


def _one(pipeline, path: Path, top: int) -> None:
    from src.inference.contract import AudioSource

    rec = pipeline.analyse(AudioSource.from_path(path, origin="upload"),
                           origin="upload", meta={"filename": path.name})

    print("=" * 78)
    print(f"FILE  {path.name}")
    print("=" * 78)

    if not rec.get("ok"):
        rej = rec.get("rejection") or {}
        print(f"  REJECTED by the quality gate: {rej.get('code')}")
        print(f"  reason: {rej.get('reason') or rec.get('status')}")
        print()
        return

    cmp_ = rec.get("comparison", {})
    _show("Python model (python_models/best)", cmp_.get("python", {}), top)
    print()
    _show("Google Teachable Machine (gtm_model/)", cmp_.get("gtm", {}), top)
    print()

    py = cmp_.get("python", {}).get("predicted_class")
    gt = cmp_.get("gtm", {}).get("predicted_class")
    agree = "yes" if cmp_.get("classes_agree") else "no"
    print(f"  agree           : {agree}   (python={py!r}, gtm={gt!r})")
    print(f"  consistency     : {cmp_.get('consistency_status')}")
    print(f"  confidence gap  : {cmp_.get('confidence_difference')}")

    dec = rec.get("decision", {})
    q = rec.get("quality", {})
    alert = rec.get("alert", {})
    review = rec.get("review", {})

    print()
    print(f"  final class     : {dec.get('final_class')}")
    print(f"  final decision  : {dec.get('final_decision')}")
    print(f"  severity        : {dec.get('severity_display')}")
    print(f"  audio quality   : {q.get('verdict')}  {q.get('problems') or ''}")
    print(f"  alert raised    : {alert.get('raised')}  ({alert.get('note') or ''})")
    print(f"  manual review   : {review.get('required')}  {review.get('matched') or ''}")
    print(f"  latency         : {rec.get('elapsed_ms')} ms  "
          f"(budget {rec.get('budget_sec')}s, within_budget={rec.get('within_budget')})")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Classify audio with both bundled models")
    ap.add_argument("target", type=Path, help="a .wav file, or a folder of .wav files")
    ap.add_argument("--top", type=int, default=3, help="how many classes to show (default 3)")
    ap.add_argument("--model-dir", type=Path, default=REPO_ROOT / "python_models" / "best")
    ap.add_argument("--gtm-dir", type=Path, default=REPO_ROOT / "gtm_model")
    args = ap.parse_args(argv)

    if not args.target.exists():
        print(f"no such file or folder: {args.target}", file=sys.stderr)
        return 2

    clips = sorted(args.target.glob("*.wav")) if args.target.is_dir() else [args.target]
    if not clips:
        print(f"no .wav files in {args.target}", file=sys.stderr)
        return 2

    # Quiet TensorFlow's startup logging.
    import logging
    logging.disable(logging.INFO)

    from src.services.pipeline import AnalysisPipeline

    print(f"loading models ... ({'folder of ' + str(len(clips)) if len(clips) > 1 else clips[0].name})")
    try:
        pipeline = AnalysisPipeline.load(model_dir=args.model_dir, gtm_dir=args.gtm_dir)
    except Exception as exc:
        print(f"PIPELINE UNAVAILABLE: {exc}", file=sys.stderr)
        return 3

    for clip in clips:
        _one(pipeline, clip, args.top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
