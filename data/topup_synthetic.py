#!/usr/bin/env python3
"""Fill the remaining shortfall in Aggression, Glass Breaking and Panic Scream with the
procedural synthesiser once the real pools were exhausted. Only the shortfall is generated,
and never in the real recordings' id range. (The synthetic glass clips were later dropped by
audio_dataset/scripts/trim_to_300.py; 50 Aggression and 25 Panic Scream remain.)

    python data/topup_synthetic.py --check        # print the plan, write nothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_corpus import (  # noqa: E402
    CLASS_CODES, build, load_class_config, MANIFEST_FIELDS,
)
import csv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DATASET = REPO_ROOT / "audio_dataset"

#: band name -> the frozen manifest enum value (assemble_manifest.py DIST_ENUM)
DIST_ENUM_MAP = {"near_0.5-2m": "near", "mid_3-15m": "medium", "far_20-60m": "far"}

# Real ids start at 0501; synthetic ids sit well clear of them.
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

    # what is on disk vs the frozen floor
    plan: dict[str, int] = {}
    for c in cfg["classes"]:
        label = c["name"]
        slug = label.lower().replace(" ", "_")
        have = len(list((AUDIO_DATASET / "originals" / slug).glob("*.wav"))) \
            if (AUDIO_DATASET / "originals" / slug).exists() else 0
        # Existing synthetic originals count toward the floor, as the verifier counts them.
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

    # Continue past the highest existing id per class (defect removal left holes).
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

    # build() writes every class, so build in a temp root and move only the kept clips.
    import tempfile, shutil
    tmp_root = Path(tempfile.mkdtemp(prefix="ss_topup_"))
    rows = build(max(plan.values()), tmp_root, args.seed, None,
                 audio_subdir="synthetic", id_offset=args.id_offset)
    # Keep only planned classes and the first `want` ids per class.
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
        # skip ids already on disk for this class
        if num <= top_existing.get(label, args.id_offset):
            continue
        used[label] = k + 1
        keep.append(r)

    # Holes can leave the first batch short; emit a second batch above the highest id.
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
