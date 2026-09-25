#!/usr/bin/env python3
"""Deterministic top-up of the procedural corpus for shortfall classes.

The real, openly-licensed pools on disk (FSD50K dev + eval-recovered, ESC-50,
UrbanSound8K) are exhausted for Aggression / Glass Breaking / Panic Scream
(`audio_dataset/scripts/acquire_corpus.py` reports "room left" 195/99/195 with
every candidate pool drained).  This driver completes those three classes from
the repository's own procedural synthesiser -- the SAME generators, seed
discipline and provenance policy as `data/generate_corpus.py`, whose docstring
explains why procedural synthesis is the honest fallback.  It only produces the
SHORTFALL (never more than `required_originals`), only for classes that are
short, and never touches the real recordings' id range.

Usage:
    python data/topup_synthetic.py                 # fills only the shortfall
    python data/topup_synthetic.py --check        # print the plan, write nothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_corpus import (  # noqa: E402
    CLASSES, CLASS_CODES, build, load_class_config, MANIFEST_FIELDS,
)
import csv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DATASET = REPO_ROOT / "audio_dataset"

#: band name -> the frozen manifest enum value (assemble_manifest.py DIST_ENUM)
DIST_ENUM_MAP = {"near_0.5-2m": "near", "mid_3-15m": "medium", "far_20-60m": "far"}

# id-offset: acquired (real) ids for these classes started at 0501; keep a clear
# gap so no future real ingestion can collide with synthetic numbering.
SYNTH_ID_OFFSET = 699  # synthetic ids start at SS-<CODE>-0700


def per_class_target(label: str) -> int:
    cfg = load_class_config()
    for c in cfg["classes"]:
        if c["name"] == label:
            return int(c.get("required_originals", 300))
    raise KeyError(label)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260923,
                    help="same master seed as data/generate_corpus.py")
    ap.add_argument("--id-offset", type=int, default=SYNTH_ID_OFFSET)
    ap.add_argument("--check", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args(argv)

    cfg = load_class_config()
    classes = {c["name"]: c for c in cfg["classes"]}

    # what is on disk vs the frozen floor
    plan: dict[str, int] = {}
    for c in cfg["classes"]:
        label = c["name"]
        slug = label.lower().replace(" ", "_")
        have = len(list((AUDIO_DATASET / "originals" / slug).glob("*.wav"))) \
            if (AUDIO_DATASET / "originals" / slug).exists() else 0
        # synthetic originals for this class already live in synthetic/<slug>/ and
        # carry manifest rows (help_tts_rows.csv, an earlier synthetic_topup_rows.csv).
        # They count toward the floor the same way the assembler/verifier counts:
        # by manifest rows with original_or_augmented == original.
        for rows_csv in (AUDIO_DATASET / "manifests" / "help_tts_rows.csv",
                         AUDIO_DATASET / "manifests" / "synthetic_topup_rows.csv"):
            if not rows_csv.exists():
                continue
            with rows_csv.open(newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("class_label") == label and \
                            (r.get("original_or_augmented") or "").strip().lower() == "original":
                        have += 1
        want = max(0, per_class_target(label) - have)
        if want:
            plan[label] = want

    print("top-up plan (real recordings on disk under originals/):")
    for label, want in sorted(plan.items()):
        print(f"  {label:<24} +{want}")
    if not plan:
        print("  nothing to do")
        return 0
    if args.check:
        return 0

    # ids must continue past whatever synthetic clips already exist on disk.
    # The previous runs numbered their clips from id_offset+1 CONTIGUOUSLY, but the
    # ids actually on disk may be non-contiguous after defect removal, so the safe
    # continuation point is the highest existing numeric id per class, not a count.
    existing_synth: dict[str, int] = {}
    top_existing: dict[str, int] = {}
    for label in plan:
        slug = label.lower().replace(" ", "_")
        d = AUDIO_DATASET / "synthetic" / slug
        nums = [int(p.stem.split("-")[2]) for p in d.glob("*.wav")] if d.exists() else []
        existing_synth[label] = len(nums)
        top_existing[label] = max(nums) if nums else args.id_offset
        if nums:
            print(f"  note: {label} already has {len(nums)} synthetic clips "
                  f"(ids up to SS-{CLASS_CODES[label]}-{max(nums):04d}); "
                  f"new ids continue above that")

    # build() emits `per_class` for EVERY class and writes the WAV files before the
    # rows are filtered -- generating straight into AUDIO_DATASET would leave orphan
    # wavs for the unplanned classes (verified on disk but unlisted in any manifest,
    # which verify_dataset.py correctly rejects). Build into a temp root and move
    # only the kept clips.
    import tempfile, shutil
    tmp_root = Path(tempfile.mkdtemp(prefix="ss_topup_"))
    rows = build(max(plan.values()), tmp_root, args.seed, None,
                 audio_subdir="synthetic", id_offset=args.id_offset)
    # build() emits `per_class` for EVERY class; keep only the planned classes
    # and only the first `want` ids per class (deterministic: id order).
    keep: list[dict] = []
    used: dict[str, int] = {}
    for r in rows:
        label = r["class_label"]
        if label not in plan:
            continue
        k = used.get(label, 0)
        if k >= plan[label]:
            continue
        num = int(r["audio_id"].rsplit("-", 1)[-1])
        # skip ids that already exist on disk for this class (ids numbered from
        # id_offset+1 by an earlier run of this driver or generate_corpus)
        if num <= top_existing.get(label, args.id_offset):
            continue
        used[label] = k + 1
        keep.append(r)

    # If disk ids are non-contiguous (defect removal punched holes), the first
    # build() window may yield fewer than plan[label] fresh ids; emit a second
    # batch strictly ABOVE the highest existing id. Per-clip RNG is seeded from
    # audio_id, so a new id is a new, deterministic clip.
    shortfall = {label: plan[label] - used.get(label, 0) for label in plan}
    shortfall = {label: n for label, n in shortfall.items() if n > 0}
    if shortfall:
        off2 = max(top_existing.values())
        extra_rows = build(max(shortfall.values()), tmp_root, args.seed, None,
                           audio_subdir="synthetic", id_offset=off2)
        extra_used: dict[str, int] = {}
        for r in extra_rows:
            label = r["class_label"]
            if label not in shortfall:
                continue
            num = int(r["audio_id"].rsplit("-", 1)[-1])
            if num <= off2:
                continue
            k = extra_used.get(label, 0)
            if k >= shortfall[label]:
                continue
            extra_used[label] = k + 1
            keep.append(r)

    manifest = AUDIO_DATASET / "manifests" / "synthetic_topup_rows.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    keep.sort(key=lambda r: (r["class_label"], r["audio_id"]))
    # map the generator's internal distance band to the frozen manifest enum
    for r in keep:
        r["approximate_distance"] = {"near_0.5-2m": "near", "mid_3-15m": "medium",
                                     "far_20-60m": "far"}.get(r["approximate_distance"],
                                                              r["approximate_distance"])
    # move the kept clips out of the temp root before it is deleted
    for r in keep:
        src = tmp_root / r["filename"]
        dst = AUDIO_DATASET / r["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        # the sha256 was computed over the same bytes; bytes are unchanged by move
    shutil.rmtree(tmp_root, ignore_errors=True)
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(keep)

    print(f"wrote {len(keep)} synthetic originals -> {manifest}")
    from collections import Counter
    for label in sorted(plan):
        got = Counter(r["class_label"] for r in keep)[label]
        mark = "" if got >= plan[label] else "  <-- SHORT"
        print(f"  {label:<24} +{got:>4} / {plan[label]}{mark}")
    if any(Counter(r["class_label"] for r in keep)[l] < plan[l] for l in plan):
        print("SHORTFALL: not all classes reached their floor; do not freeze the split.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
