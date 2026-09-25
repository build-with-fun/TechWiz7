#!/usr/bin/env python3
"""
Capture GTM's audio frontend parameters and verify the server-side reproduction.

Two subcommands:

  verify   Run the exported GTM model server-side on every clip listed in a recording
           file, alongside the class/confidences Teachable Machine's own browser UI
           reported for the same clips, and require agreement:
             - predicted class matches on >= 95% of clips
             - per-class confidence within --tolerance (default 0.05) on every clip
           Writes gtm_model/frontend_verification.json ({"passed": bool, ...rows}).

The frontend parameters themselves live in gtm_model/frontend_config.json — captured by
hand from the export (see the README in gtm_model/upload_package). This script does not
guess them; it measures OUR reproduction against the browser's numbers.

Usage
-----
    .venv/bin/python tools/capture_gtm_frontend.py verify \
        --recordings gtm_model/browser_recordings.json [--tolerance 0.05]

`browser_recordings.json` is produced while testing the model in TM's web UI: for each
clip, {"path": ..., "gtm_predicted_class": ..., "gtm_confidences": {...}}.
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
