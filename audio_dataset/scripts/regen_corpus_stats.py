#!/usr/bin/env python3
"""Regenerate corpus_statistics.json from the *frozen* manifest.

`assemble_manifest.py` merges per-source row CSVs and emits both manifest.csv and
corpus_statistics.json in one pass. Its DEFAULT_SOURCES point at intermediate row
files that are no longer kept in the tree, so re-running the full assembly is not
an option this late: the manifest is frozen and must not change.

This script recomputes the statistics deliverable from the frozen manifest instead.
It reads manifest.csv, runs the same `statistics()` the assembler uses, and writes
corpus_statistics.json with the manifest's real sha256.

Usage:
    .venv/bin/python audio_dataset/scripts/regen_corpus_stats.py [--check]

Exit code 1 if any class falls below its required-originals floor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Import the canonical statistics() so the numbers can never drift from the
# assembler's own definition.
from audio_dataset.scripts.assemble_manifest import (  # noqa: E402
    statistics,
    load_classes,
)

MANIFEST = REPO_ROOT / "audio_dataset" / "manifest.csv"
STATS_OUT = REPO_ROOT / "audio_dataset" / "manifests" / "corpus_statistics.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="validate only; print a summary, write nothing")
    args = ap.parse_args(argv)

    if not MANIFEST.exists():
        print(f"ERROR: frozen manifest not found: {MANIFEST}", file=sys.stderr)
        return 2

    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("ERROR: manifest has no data rows", file=sys.stderr)
        return 2

    classes = load_classes()
    stats = statistics(rows, classes)
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    stats["manifest_sha256"] = digest
    # Recomputed from the frozen manifest, not from the assembler's row sources.
    stats["generated_by"] = "audio_dataset/scripts/regen_corpus_stats.py"

    # Integrity check: the stats must describe exactly the manifest they name.
    totals = stats["totals"]
    print(f"manifest: {MANIFEST.name}  sha256 {digest[:16]}...")
    print(f"rows={totals['rows']} originals={totals['originals']} "
          f"augmented={totals['augmented']} classes={totals['distinct_classes']}")

    below = []
    for name, per in sorted(stats["per_class"].items(),
                            key=lambda kv: (not kv[1]["critical"], kv[0])):
        flag = "ok" if per["meets_floor"] else "BELOW FLOOR"
        if not per["meets_floor"]:
            below.append(name)
        print(f"  {name:<24} {per['originals']:>5} "
              f"(real={per['originals_real_field_recording']:>4} "
              f"synth={per['originals_synthetic']:>3})  {flag}")

    if args.check:
        print("\n--check: nothing written.")
        return 1 if below else 0

    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    STATS_OUT.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {STATS_OUT}")
    if below:
        print(f"WARNING: classes below floor: {below}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
