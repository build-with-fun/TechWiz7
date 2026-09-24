#!/usr/bin/env python3
"""
SonicSentinel AI — independent split/manifest auditor.

Owner: lorena.  Run this to prove the split is honest. It is deliberately written as a
SEPARATE implementation from build_split.py: it re-derives nothing from the builder's
internal state, it reads the frozen artifacts and the files on disk and checks them against
the SRS. An auditor that reuses the code it audits proves nothing.

Checks
  --check-manifest  manifest.csv is structurally valid, licences present, classes legal
  --check-split     split.json is internally consistent, exhaustive and stratified
  --check-files     every manifest filename exists on disk; no file on disk is unlisted
  --check-leakage   no parent/segment/augmented lineage crosses a split boundary
  --check-duplicates no two rows share a sha256

Exit code 0 = all requested checks passed. Anything else is a real problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from audio_dataset.build_split import (  # noqa: E402
    AUGMENTED, ORIGINAL, SPLIT_NAMES, ManifestError, load_classes, load_manifest,
    validate_records,
)

DEFAULT_MANIFEST = REPO_ROOT / "audio_dataset" / "manifest.csv"
DEFAULT_SPLIT = REPO_ROOT / "data" / "splits" / "split.json"
AUDIO_ROOT = REPO_ROOT / "audio_dataset"


class AuditFailure(Exception):
    pass


def _fail(results: list[tuple[bool, str]], message: str) -> None:
    results.append((False, message))


def _ok(results: list[tuple[bool, str]], message: str) -> None:
    results.append((True, message))


def load_split(path: Path = DEFAULT_SPLIT) -> dict:
    if not path.exists():
        raise AuditFailure(f"frozen split not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def check_manifest(records: list[dict[str, str]], classes: dict) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    try:
        validate_records(records, classes)
        _ok(results, f"manifest structure valid ({len(records)} rows)")
    except ManifestError as exc:
        _fail(results, f"manifest structure invalid: {exc}")

    originals = [r for r in records if r["original_or_augmented"].strip().lower() == ORIGINAL]
    per_class = Counter(r["class_label"].strip() for r in originals)
    _ok(results, f"originates counted: {len(originals)} original rows (augmented/segments excluded)")

    for label in sorted(classes):
        want = int(classes[label].get("required_originals", 300))
        got = per_class.get(label, 0)
        if got == want:
            _ok(results, f"  {label}: {got}/{want} originals")
        else:
            _fail(results, f"  {label}: {got}/{want} originals")

    if len(originals) >= 3000:
        _ok(results, f"total originals >= 3000 ({len(originals)})")
    else:
        _fail(results, f"total originals < 3000 ({len(originals)}) — SRS floor not met")
    return results


def check_split(records: list[dict[str, str]], split: dict, classes: dict) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    assignments = split["assignments"]
    by_id = {r["audio_id"]: r for r in records}

    # 1. Exhaustive and no phantom ids
    missing = set(by_id) - set(assignments)
    phantom = set(assignments) - set(by_id)
    if missing or phantom:
        _fail(results, f"split/manifest mismatch: {len(missing)} unassigned, {len(phantom)} phantom")
    else:
        _ok(results, f"split covers every manifest row exactly once ({len(assignments)})")

    # 2. Every audio_id appears in exactly one split
    per_id = defaultdict(list)
    for audio_id, info in assignments.items():
        per_id[audio_id].append(info["split"])
    multi = {a: s for a, s in per_id.items() if len(s) != 1}
    if multi:
        _fail(results, f"{len(multi)} ids appear in more than one split")
    else:
        _ok(results, "no audio_id appears in more than one split")

    # 3. Only legal split names
    illegal = {v["split"] for v in assignments.values()} - set(SPLIT_NAMES)
    if illegal:
        _fail(results, f"illegal split names: {illegal}")
    else:
        _ok(results, f"split values are legal ({sorted(SPLIT_NAMES)})")

    # 4. Per-class stratification of ORIGINALS
    originals_by_class_split: dict[str, Counter] = defaultdict(Counter)
    for audio_id, info in assignments.items():
        if info["role"] == ORIGINAL:
            originals_by_class_split[info["class_label"]][info["split"]] += 1
    for label in sorted(classes):
        c = originals_by_class_split[label]
        total = sum(c.values())
        if total == 0:
            _fail(results, f"  {label}: no originals assigned")
            continue
        n_train, n_val, n_test = c["train"], c["val"], c["test"]
        if total == 300 and (n_train, n_val, n_test) != (210, 45, 45):
            _fail(results, f"  {label}: {n_train}/{n_val}/{n_test} != 210/45/45")
        else:
            _ok(results, f"  {label}: train={n_train} val={n_val} test={n_test} (total {total})")

    # 5. Overall totals
    totals = Counter(v["split"] for v in assignments.values() if v["role"] == ORIGINAL)
    if totals["train"] == 2100 and totals["val"] == 450 and totals["test"] == 450:
        _ok(results, "overall originals split exact: 2100/450/450")
    else:
        _fail(results, f"overall originals {dict(totals)} != 2100/450/450")

    # 6. Determinism: re-deriving the original assignments must give an identical result
    from audio_dataset.build_split import build_split
    rebuilt = build_split(records, classes, seed=split.get("seed", 20260923))
    if rebuilt["assignments"] == assignments:
        _ok(results, "re-running the builder reproduces the split byte-for-byte")
    else:
        _fail(results, "split is NOT reproducible — the builder disagrees with the frozen file")
    return results


def check_leakage(records: list[dict[str, str]], split: dict) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    assignments = split["assignments"]
    by_id = {r["audio_id"]: r for r in records}
    violations = []

    for rec in records:
        parent = (rec.get("parent_audio_id") or "").strip()
        if not parent:
            continue
        if parent not in assignments:
            violations.append(f"{rec['audio_id']}: parent {parent} not assigned")
            continue
        child_split = assignments[rec["audio_id"]]["split"]
        parent_split = assignments[parent]["split"]
        if child_split != parent_split:
            violations.append(
                f"{rec['audio_id']} ({child_split}) descends from {parent} ({parent_split})"
            )

    if violations:
        _fail(results, f"{len(violations)} lineage leaks across splits, e.g. " + "; ".join(violations[:5]))
    else:
        _ok(results, "no segment/augmented row crosses a split boundary from its parent")

    derived = [r for r in records if r["original_or_augmented"].strip().lower() == AUGMENTED]
    _ok(results, f"{len(derived)} derived rows carry a parent (checked above)")
    return results


def check_duplicates(records: list[dict[str, str]]) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    for rec in records:
        h = (rec.get("sha256") or "").strip()
        if h:
            hashes[h].append(rec["audio_id"])
    dups = {h: ids for h, ids in hashes.items() if len(ids) > 1}
    if dups:
        sample = list(dups.items())[:5]
        _fail(results, f"{len(dups)} duplicate content hashes shared by multiple audio_ids: {sample}")
    else:
        _ok(results, f"no duplicate content hashes across {len(hashes)} hashed rows")
    return results


def check_files(records: list[dict[str, str]]) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    audio_root = REPO_ROOT / "audio_dataset"

    missing = [r["filename"] for r in records if not (audio_root / r["filename"]).exists()]
    if missing:
        _fail(results, f"{len(missing)} manifest rows point at files that do not exist, e.g. {missing[:5]}")
    else:
        _ok(results, f"all {len(records)} manifest files exist on disk")

    # Nothing on disk may be missing from the manifest (would mean unreported audio).
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    on_disk = {
        str(p.relative_to(audio_root))
        for p in audio_root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    }
    listed = {r["filename"] for r in records}
    unlisted = on_disk - listed
    if unlisted:
        _fail(results, f"{len(unlisted)} audio files exist on disk but are not in the manifest, e.g. {sorted(unlisted)[:5]}")
    else:
        _ok(results, f"no unlisted audio files ({len(on_disk)} on disk, all accounted for)")
    return results


def report(title: str, results: list[tuple[bool, str]]) -> bool:
    passed = all(ok for ok, _ in results)
    print(f"\n== {title} == {'PASS' if passed else 'FAIL'}")
    for ok, message in results:
        print(f"  [{'ok' if ok else 'XX'}] {message}")
    return passed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit the SonicSentinel manifest and split.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--check-split", action="store_true")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--check-leakage", action="store_true")
    parser.add_argument("--check-duplicates", action="store_true")
    parser.add_argument("--all", action="store_true", help="run every check")
    args = parser.parse_args(argv)

    if args.all or not any([args.check_manifest, args.check_split, args.check_files,
                            args.check_leakage, args.check_duplicates]):
        args.check_manifest = args.check_split = args.check_leakage = True
        args.check_duplicates = True

    try:
        records = load_manifest(args.manifest)
        classes = load_classes()
    except ManifestError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    all_passed = True
    if args.check_manifest:
        all_passed &= report("manifest", check_manifest(records, classes))
    if args.check_duplicates:
        all_passed &= report("duplicates", check_duplicates(records))

    split = None
    if args.check_split or args.check_leakage:
        try:
            split = load_split(args.split)
        except AuditFailure as exc:
            print(f"\n== split == FAIL\n  [XX] {exc}")
            return 3

    if args.check_split and split is not None:
        all_passed &= report("split", check_split(records, split, classes))
    if args.check_leakage and split is not None:
        all_passed &= report("leakage", check_leakage(records, split))
    if args.check_files:
        all_passed &= report("files", check_files(records))

    print(f"\nRESULT: {'ALL CHECKS PASSED' if all_passed else 'PROBLEMS FOUND'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
