#!/usr/bin/env python3
"""Check the server-side Teachable Machine frontend against the browser's own predictions.

    .venv/bin/python tools/capture_gtm_frontend.py verify \
        --recordings gtm_model/browser_recordings.json [--tolerance 0.05]

``browser_recordings.json`` lists clips with the class and confidences TM's web UI showed for
them. The check passes when the classes match on at least 95% of clips and every confidence is
within the tolerance; it writes gtm_model/frontend_verification.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.inference.gtm_predictor import GtmModelPredictor, compare_frontend_agreement  # noqa: E402
from audio_preprocessing.pipeline import AudioPipeline as _AudioPipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Capture/verify the GTM audio frontend.")
    sub = ap.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify", help="compare server-side GTM predictions with browser recordings")
    v.add_argument("--recordings", type=Path, required=True,
                   help="JSON list of {path, gtm_predicted_class, gtm_confidences}")
    v.add_argument("--gtm-dir", type=Path, default=REPO_ROOT / "gtm_model")
    v.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args(argv)

    if args.command == "verify":
        recorded = json.loads(args.recordings.read_text(encoding="utf-8"))
        predictor = GtmModelPredictor.load(args.gtm_dir)
        preprocessor = _AudioPipeline()
        report = compare_frontend_agreement(recorded, predictor, preprocessor,
                                            tolerance=args.tolerance)
        out = args.gtm_dir / "frontend_verification.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"class agreement : {report['class_agreement']:.2%} over {report['n_clips']} clips")
        print(f"max deviation   : {report['max_confidence_deviation']} (tolerance {args.tolerance})")
        print(f"verdict         : {'PASSED' if report['passed'] else 'FAILED'}")
        print(f"wrote {out}")
        return 0 if report["passed"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
