#!/usr/bin/env python3
"""
SonicSentinel AI -- dataset verifier. THIS IS THE GATE for the dataset deliverable.

Owner: omar.  Authority: SRS v1.0 Step 1, Step 5, FR xvii/xviii, deliverables 3 and 5,
integrity rules 1.8, and the frozen contract `audio_dataset/manifest_schema.md`.

It answers one question with a number: **can the claims made about this dataset be
checked against the files on disk?** Every check below fails loudly rather than
warning, because a dataset that "mostly" meets the floor is the one that gets a
report disqualified.

Checks
------
manifest   one row per file; frozen columns present and in order; audio_id unique and
           matching its class code; class names exactly the ten; licence/author present;
           filename on disk; sha256 matches the bytes; no two originals share content
           (uniqueness); augmented rows name a resolvable original parent
split      data/splits/split.json exists; every row assigned; train/val/test disjoint;
           stratified 70/15/15 per class; a segment or augmentation shares its parent's
           split (no train/test leakage)
floor      300 unique originals per class (--strict), counted ONLY from real files

Usage
-----
    .venv/bin/python audio_dataset/scripts/verify_dataset.py --strict
    .venv/bin/python audio_dataset/scripts/verify_dataset.py --quick     # skip hashing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_DATASET = SCRIPT_DIR.parent
REPO_ROOT = AUDIO_DATASET.parent

MANIFEST = AUDIO_DATASET / "manifest.csv"
SPLIT = REPO_ROOT / "data" / "splits" / "split.json"

FROZEN_COLUMNS = [
    "audio_id", "filename", "class_label", "source", "source_url", "licence", "author",
    "date_fetched", "duration_sec", "sampling_rate", "channels", "recording_environment",
    "recording_device", "approximate_distance", "original_or_augmented", "parent_audio_id",
    "segment_start_sec", "segment_end_sec", "sha256", "dataset_split",
]

ID_RE = re.compile(r"^SS-([A-Z]{3})-(\d{4})$")


class Report:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if detail:
            line += f" -- {detail}"
        print(line)
        return ok

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [c for c in self.checks if not c[1]]

    def summary(self) -> None:
        n = len(self.checks)
        bad = len(self.failed)
        print(f"\n{'='*78}\n{n - bad}/{n} checks passed" +
              (f", {bad} FAILED" if bad else " -- ALL CHECKS PASSED"))
        for name, _, detail in self.failed:
            print(f"  FAILED: {name} -- {detail}")


def wav_info(path: Path) -> tuple[int, int, float]:
    """Read sample rate, channels and duration from a PCM/float WAV header.

    Done by hand rather than by spawning ffprobe 3,000 times: same numbers, no
    3,000 process launches inside the gate.
    """
    with path.open("rb") as fh:
        head = fh.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError("not a RIFF/WAVE file")
        fmt = None
        data_size = None
        while True:
            chunk = fh.read(8)
            if len(chunk) < 8:
                break
            cid, size = struct.unpack("<4sI", chunk)
            if cid == b"fmt ":
                body = fh.read(size)
                fmt = struct.unpack("<HHIIHH", body[:16])
            elif cid == b"data":
                data_size = size
                fh.seek(size, 1)
            else:
                fh.seek(size + (size & 1), 1)
        if fmt is None or data_size is None:
            raise ValueError("WAV has no fmt/data chunk")
        _, channels, rate, _, _, bits = fmt
        duration = data_size / (rate * channels * max(bits // 8, 1))
        return rate, channels, duration


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_classes() -> dict[str, dict]:
    cfg = json.loads((REPO_ROOT / "config" / "classes.json").read_text(encoding="utf-8"))
    return {c["name"]: c for c in cfg["classes"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the SonicSentinel dataset against its claims.")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument("--strict", action="store_true",
                    help="require exactly 300 originals per class and a 2100/450/450 split")
    ap.add_argument("--quick", action="store_true", help="skip per-file hashing")
    args = ap.parse_args(argv)

    classes = load_classes()
    code_to_name = {c["code"]: n for n, c in classes.items()}
    rep = Report()

    print("SonicSentinel dataset verification")
    print(f"  manifest: {args.manifest}")
    print(f"  split   : {args.split}\n")
    print("-- manifest ------------------------------------------------------------")

    if not args.manifest.exists():
        rep.check("manifest exists", False, f"not found: {args.manifest}")
        rep.summary()
        return 2
    with args.manifest.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        rows = [dict(r) for r in reader]
    rep.check("manifest exists", True, f"{len(rows)} rows")

    missing_core = [c for c in FROZEN_COLUMNS if c not in fields]
    rep.check("frozen columns present", not missing_core, 
              f"missing {missing_core}" if missing_core else f"{len(fields)} columns")
    order_ok = fields[:len(FROZEN_COLUMNS)] == FROZEN_COLUMNS
    rep.check("frozen columns in frozen order", order_ok,
              "" if order_ok else f"got {fields[:len(FROZEN_COLUMNS)]}")

    # --- audio_id ------------------------------------------------------------------
    ids = [r["audio_id"] for r in rows]
    dup_ids = [i for i, n in Counter(ids).items() if n > 1]
    rep.check("audio_id unique", not dup_ids, f"duplicates: {dup_ids[:5]}")

    bad_fmt = [r["audio_id"] for r in rows if not ID_RE.match(r["audio_id"])]
    rep.check("audio_id format SS-<CODE>-<NNNN>", not bad_fmt, f"bad: {bad_fmt[:5]}")

    mismatched_code = [
        r["audio_id"] for r in rows
        if ID_RE.match(r["audio_id"])
        and code_to_name.get(ID_RE.match(r["audio_id"]).group(1)) != r["class_label"]
    ]
    rep.check("audio_id class code matches class_label", not mismatched_code,
              f"{len(mismatched_code)} mismatched, e.g. {mismatched_code[:3]}")

    # --- classes -------------------------------------------------------------------
    unknown = sorted({r["class_label"] for r in rows} - set(classes))
    rep.check("all class labels are the ten", not unknown, f"unknown: {unknown}")

    # --- provenance ----------------------------------------------------------------
    no_lic = [r["audio_id"] for r in rows if not r["licence"].strip()]
    rep.check("licence present on every row", not no_lic, f"{len(no_lic)} blank, e.g. {no_lic[:3]}")
    no_auth = [r["audio_id"] for r in rows if not r["author"].strip()]
    rep.check("author present on every row", not no_auth, f"{len(no_auth)} blank")
    no_src = [r["audio_id"] for r in rows if not r["source"].strip()]
    rep.check("source present on every row", not no_src, f"{len(no_src)} blank")
    nc = [r["audio_id"] for r in rows if "NC" in r["licence"].upper().replace("-", "")]
    rep.check("no non-commercial licence in the dataset", not nc, f"{len(nc)} NC rows: {nc[:3]}")

    # --- files on disk -------------------------------------------------------------
    missing_files = [r["filename"] for r in rows if not (AUDIO_DATASET / r["filename"]).exists()]
    rep.check("every filename exists on disk", not missing_files,
              f"{len(missing_files)} missing, e.g. {missing_files[:3]}")

    present = [r for r in rows if (AUDIO_DATASET / r["filename"]).exists()]

    # --- originals vs augmented ----------------------------------------------------
    statuses = Counter(r["original_or_augmented"].strip().lower() for r in rows)
    bad_status = [s for s in statuses if s not in ("original", "augmented")]
    rep.check("original_or_augmented values valid", not bad_status, f"bad: {bad_status}")

    by_id = {r["audio_id"]: r for r in rows}
    aug_no_parent = [r["audio_id"] for r in rows
                     if r["original_or_augmented"].strip().lower() == "augmented"
                     and not r["parent_audio_id"].strip()]
    rep.check("every augmented row names a parent", not aug_no_parent,
              f"{len(aug_no_parent)} without parent")
    unresolvable = [r["audio_id"] for r in rows
                    if r["original_or_augmented"].strip().lower() == "augmented"
                    and r["parent_audio_id"].strip() and r["parent_audio_id"].strip() not in by_id]
    rep.check("every parent audio_id resolves", not unresolvable, f"{len(unresolvable)} broken")

    # --- measured audio vs recorded metadata ---------------------------------------
    bad_measure: list[str] = []
    for r in present:
        if not r["sampling_rate"] or not r["channels"] or not r["duration_sec"]:
            bad_measure.append(f"{r['audio_id']}: metadata blank")
            continue
        try:
            rate, ch, dur = wav_info(AUDIO_DATASET / r["filename"])
        except Exception as exc:  # noqa: BLE001
            bad_measure.append(f"{r['audio_id']}: unreadable ({exc})")
            continue
        if rate != int(r["sampling_rate"]) or ch != int(r["channels"]):
            bad_measure.append(f"{r['audio_id']}: sr/ch {rate}/{ch} != manifest "
                               f"{r['sampling_rate']}/{r['channels']}")
        elif abs(dur - float(r["duration_sec"])) > 0.25:
            bad_measure.append(f"{r['audio_id']}: dur {dur:.3f} != manifest {r['duration_sec']}")
    rep.check("recorded sampling_rate/channels/duration match the file", not bad_measure,
              f"{len(bad_measure)} mismatched, e.g. {bad_measure[:3]}")

    # --- hashes and uniqueness -----------------------------------------------------
    if args.quick:
        print("  [SKIP] sha256 integrity (--quick)")
        print("  [SKIP] original content uniqueness (--quick)")
    else:
        bad_hash = []
        digests: dict[str, list[str]] = defaultdict(list)
        for r in present:
            if not r["sha256"]:
                bad_hash.append(f"{r['audio_id']}: empty sha256")
                continue
            d = sha256_of(AUDIO_DATASET / r["filename"])
            if d != r["sha256"]:
                bad_hash.append(f"{r['audio_id']}: sha256 mismatch")
            digests[d].append(r["audio_id"])
        rep.check("sha256 matches file content", not bad_hash,
                  f"{len(bad_hash)} bad, e.g. {bad_hash[:3]}")

        dup_content = {d: a for d, a in digests.items()
                       if len([x for x in a if by_id[x]["original_or_augmented"] == "original"]) > 1}
        rep.check("no two originals share content", not dup_content,
                  f"{len(dup_content)} duplicate groups, e.g. "
                  f"{[v[:2] for v in list(dup_content.values())[:2]]}")

    # --- per-class floor -----------------------------------------------------------
    originals = [r for r in rows if r["original_or_augmented"].strip().lower() == "original"]
    per_class = Counter(r["class_label"] for r in originals)
    print("\n-- floor --------------------------------------------------------------")
    print(f"  {'class':<24} {'originals':>9} {'required':>9}")
    short: list[str] = []
    for name in sorted(classes):
        want = int(classes[name].get("required_originals", 300))
        got = per_class.get(name, 0)
        mark = "" if got >= want else "  <-- SHORT"
        if got < want:
            short.append(f"{name}: {got}/{want}")
        print(f"  {name:<24} {got:>9} {want:>9}{mark}")
    print(f"  {'TOTAL':<24} {len(originals):>9} {sum(int(classes[n].get('required_originals',300)) for n in classes):>9}")
    rep.check(">=300 unique originals per class", not short, f"short: {short}")

    # --- split ---------------------------------------------------------------------
    print("\n-- split --------------------------------------------------------------")
    if not args.split.exists():
        rep.check("frozen split exists", False, f"not found: {args.split} (run build_split.py)")
    else:
        split = json.loads(args.split.read_text(encoding="utf-8"))
        assign = split.get("assignments", {})
        rep.check("frozen split exists", True, f"{len(assign)} assignments")

        unassigned = [r["audio_id"] for r in rows if r["audio_id"] not in assign]
        rep.check("every manifest row is assigned to a split", not unassigned,
                  f"{len(unassigned)} unassigned, e.g. {unassigned[:3]}")

        counts = Counter(a["split"] for a in assign.values())
        exp = int(round(sum(counts.values()) * 0.70))
        print(f"  split sizes: {dict(counts)}")
        rep.check("train/val/test disjoint and exhaustive",
                  sum(counts.values()) == len(assign), "")

        # stratified per class among originals
        per_class_split: dict[str, Counter] = defaultdict(Counter)
        for aid, a in assign.items():
            if a.get("role") == "original":
                per_class_split[a["class_label"]][a["split"]] += 1
        ratios_ok = True
        detail = []
        for name in classes:
            c = per_class_split[name]
            tot = sum(c.values())
            if tot == 0:
                ratios_ok = False
                detail.append(f"{name}: no originals")
                continue
            want_tr, want_va = round(tot * 0.70), round(tot * 0.15)
            if abs(c["train"] - want_tr) > 1 or abs(c["val"] - want_va) > 1:
                ratios_ok = False
                detail.append(f"{name}: {dict(c)} want {want_tr}/{want_va}/{tot-want_tr-want_va}")
        rep.check("split is stratified 70/15/15 per class", ratios_ok, "; ".join(detail[:3]))

        # leakage: derived rows must share their parent's split
        leaks = []
        for aid, a in assign.items():
            row = by_id.get(aid)
            if not row:
                continue
            parent = row["parent_audio_id"].strip()
            if parent and parent in assign and assign[parent]["split"] != a["split"]:
                leaks.append(f"{aid} ({a['split']}) vs parent {parent} ({assign[parent]['split']})")
        rep.check("no derived row straddles its parent's split", not leaks,
                  f"{len(leaks)} leaks, e.g. {leaks[:3]}")

        # dataset_split column agrees with split.json
        mism = [r["audio_id"] for r in rows
                if r["audio_id"] in assign and r["dataset_split"].strip()
                and r["dataset_split"].strip() != assign[r["audio_id"]]["split"]]
        rep.check("manifest dataset_split agrees with the frozen split", not mism,
                  f"{len(mism)} mismatched")

    if args.strict:
        st = Counter(r["original_or_augmented"].strip().lower() for r in rows)
        totals = Counter(a["split"] for a in
                         (json.loads(args.split.read_text(encoding="utf-8")).get("assignments", {})
                          if args.split.exists() else {}).values())
        rep.check("--strict: 2100/450/450 split", 
                  totals.get("train") == 2100 and totals.get("val") == 450
                  and totals.get("test") == 450,
                  f"got {dict(totals)}")
        rep.check("--strict: exactly 300 originals per class",
                  all(per_class.get(n, 0) == int(classes[n].get("required_originals", 300))
                      for n in classes),
                  f"per class: {dict(per_class)}")

    rep.summary()
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
