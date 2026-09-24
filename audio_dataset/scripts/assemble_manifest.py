#!/usr/bin/env python3
"""
SonicSentinel AI -- manifest assembler.

Owner: omar (data sourcing).  Authority: `audio_dataset/manifest_schema.md` (owner:
lorena), SRS deliverable 3.

WHAT THIS DOES
--------------
Merges every provenance-checked row from every source into the ONE frozen manifest
at `audio_dataset/manifest.csv`:

    audio_dataset/manifests/fsd50k_real_rows.csv   -- real FSD50K field recordings (omar)
    audio_dataset/manifest_generated.csv           -- procedural synthesis (nadia)

and writes `audio_dataset/manifests/corpus_statistics.json` (deliverable 3's
"statistics").

WHAT IT REFUSES TO DO (the point of the file)
---------------------------------------------
* It does NOT assign a dataset split. `audio_dataset/build_split.py` is the only
  thing allowed to do that. If any input row carries a non-empty `dataset_split`
  this script FAILS, because two split definitions is how train data leaks into a
  test metric and that would invalidate every number in the report.
* It does NOT allow a duplicate `audio_id`. An id is a primary key.
* It does NOT invent a missing required column. A source that cannot supply the
  frozen required set is rejected with the offending columns named.

Extra columns are preserved (build_split.py keeps them), so a source may carry as
much detail as it likes alongside the frozen schema.

Usage
-----
    .venv/bin/python audio_dataset/scripts/assemble_manifest.py
    .venv/bin/python audio_dataset/scripts/assemble_manifest.py --check
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_DATASET = SCRIPT_DIR.parent
REPO_ROOT = AUDIO_DATASET.parent

FROZEN_COLUMNS = [
    "audio_id", "filename", "class_label", "source", "source_url", "licence", "author",
    "date_fetched", "duration_sec", "sampling_rate", "channels", "recording_environment",
    "recording_device", "approximate_distance", "original_or_augmented", "parent_audio_id",
    "segment_start_sec", "segment_end_sec", "sha256", "dataset_split",
]
REQUIRED_COLUMNS = [
    "audio_id", "filename", "class_label", "source", "licence", "author",
    "original_or_augmented", "dataset_split",
]

DEFAULT_SOURCES = [
    AUDIO_DATASET / "manifests" / "fsd50k_real_rows.csv",
    AUDIO_DATASET / "manifest_generated.csv",
]
DEFAULT_OUT = AUDIO_DATASET / "manifest.csv"
STATS_OUT = AUDIO_DATASET / "manifests" / "corpus_statistics.json"

# 'unspecified' is permitted: FSD50K clips carry no recording-environment metadata at
# the source, so 618 of the real rows are 'unspecified' truthfully. Guessing a value
# would fabricate provenance, which is worse than admitting we do not know.
ENV_ENUM = {"indoor", "outdoor", "vehicle", "studio", "synthetic", "unspecified"}
DIST_ENUM = {"near", "medium", "far", "n/a"}


class AssemblyError(Exception):
    pass


def read_source(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists():
        raise AssemblyError(f"source manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise AssemblyError(f"{path}: empty (no header)")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise AssemblyError(
                f"{path}: missing required columns {missing}.\n"
                f"  The frozen contract is audio_dataset/manifest_schema.md."
            )
        rows = [dict(r) for r in reader]
    if not rows:
        raise AssemblyError(f"{path}: header but no data rows")
    return reader.fieldnames, rows


def load_classes() -> dict[str, dict]:
    cfg = json.loads((REPO_ROOT / "config" / "classes.json").read_text(encoding="utf-8"))
    return {c["name"]: c for c in cfg["classes"]}


def load_class_codes() -> dict[str, str]:
    """class name -> 3-letter code, from the same frozen config the split builder uses."""
    cfg = json.loads((REPO_ROOT / "config" / "classes.json").read_text(encoding="utf-8"))
    return {c["name"]: c["code"] for c in cfg["classes"]}


def assemble(source_paths: list[Path]) -> tuple[list[str], list[dict], dict]:
    classes = load_classes()
    codes = load_class_codes()
    all_rows: list[dict] = []
    extra_cols: list[str] = []
    provenance: dict[str, int] = {}
    problems: list[str] = []

    for path in source_paths:
        fields, rows = read_source(path)
        try:
            label = str(path.relative_to(REPO_ROOT))
        except ValueError:
            label = str(path)  # out-of-tree source (tests, ad-hoc runs)
        provenance[label] = len(rows)
        for c in fields:
            if c not in FROZEN_COLUMNS and c not in extra_cols:
                extra_cols.append(c)

        for i, r in enumerate(rows, start=2):
            where = f"{path.name}:{i} ({r.get('audio_id','?')})"

            # 1. the hard rule: no source may carry a split
            if (r.get("dataset_split") or "").strip():
                problems.append(
                    f"{where}: carries dataset_split={r['dataset_split']!r}. "
                    "build_split.py is the only thing allowed to assign a split."
                )
                continue

            # 2. class must be one of the ten, exactly
            label = (r.get("class_label") or "").strip()
            if label not in classes:
                problems.append(f"{where}: class_label {label!r} is not one of the ten")
                continue

            # 3. normalise the optional-but-schema'd fields so downstream readers
            #    can rely on their values (build_split.py only *requires* 8 columns)
            row = {c: (r.get(c) or "").strip() for c in FROZEN_COLUMNS}
            for c in extra_cols:
                row[c] = (r.get(c) or "").strip()
            row["class_label"] = label

            # 2a. audio_id is a primary key AND must name its own class: SS-<CODE>-<NNNN>.
            #     A mismatch (e.g. SS-MAC-0001 labelled Gunshot) would make the id lie about
            #     the class and quietly corrupt per-class counts. Codes are shared across models
            #     (SRS Step 9), so they must agree from the start.
            aid = row["audio_id"]
            if not aid:
                problems.append(f"{where}: empty audio_id")
                continue
            parts = aid.split("-")
            if len(parts) != 3 or parts[0] != "SS" or not parts[2].isdigit():
                problems.append(f"{where}: audio_id {aid!r} is not SS-<CODE>-<NNNN>")
                continue
            want_code = codes.get(label, "?")
            if parts[1] != want_code:
                problems.append(
                    f"{where}: audio_id code {parts[1]!r} does not match class {label!r} (code {want_code!r})"
                )

            # 2b. licence is the integrity column: never blank, never non-commercial.
            lic = row["licence"]
            if not lic:
                problems.append(f"{where}: empty licence (the SRS requires a licence for every file)")
            elif "NONCOMMERCIAL" in lic.upper() or "NON-COMMERCIAL" in lic.upper() \
                    or "-NC" in lic.upper() or lic.upper().startswith("CC-BY-NC"):
                problems.append(f"{where}: non-commercial licence {lic!r} is not redistributable")

            env = row["recording_environment"].lower()
            if env and env not in ENV_ENUM:
                problems.append(f"{where}: recording_environment {env!r} not in {sorted(ENV_ENUM)}")
            dist = row["approximate_distance"].lower()
            if dist and dist not in DIST_ENUM:
                problems.append(f"{where}: approximate_distance {dist!r} not in {sorted(DIST_ENUM)}")
            if not row["sha256"]:
                problems.append(f"{where}: empty sha256")
            if not row["date_fetched"]:
                row["date_fetched"] = date.today().isoformat()
            if not row["filename"]:
                problems.append(f"{where}: empty filename")
                continue
            if not Path(AUDIO_DATASET / row["filename"]).exists():
                problems.append(f"{where}: file not on disk: {row['filename']}")
            all_rows.append(row)

    # 4. audio_id is a primary key
    seen: dict[str, dict] = {}
    for r in all_rows:
        aid = r["audio_id"]
        if aid in seen:
            problems.append(f"duplicate audio_id {aid!r}")
        seen[aid] = r

    if problems:
        head = "\n  - ".join(problems[:40])
        more = f"\n  ... and {len(problems) - 40} more" if len(problems) > 40 else ""
        raise AssemblyError(f"assembly refused ({len(problems)} problems):\n  - {head}{more}")

    cols = FROZEN_COLUMNS + extra_cols
    all_rows.sort(key=lambda r: (r["class_label"], r["audio_id"]))
    return cols, all_rows, {"sources": provenance}


def statistics(rows: list[dict], classes: dict[str, dict]) -> dict:
    """Deliverable 3: corpus statistics, split by real vs synthetic and by class."""
    per_class: dict[str, dict] = {}
    for name in classes:
        sub = [r for r in rows if r["class_label"] == name]
        originals = [r for r in sub if r["original_or_augmented"] == "original"]
        real = [r for r in originals if r.get("fetch_batch") == "fsd50k_real_v1"]
        synth = [r for r in originals if r not in real]
        lic = Counter(r["licence"] for r in sub)
        env = Counter(r["recording_environment"] for r in sub)
        dist = Counter(r["approximate_distance"] for r in sub)
        per_class[name] = {
            "rows_total": len(sub),
            "originals": len(originals),
            "augmented": len(sub) - len(originals),
            "originals_real_field_recording": len(real),
            "originals_synthetic": len(synth),
            "required_originals": int(classes[name].get("required_originals", 300)),
            "meets_floor": len(originals) >= int(classes[name].get("required_originals", 300)),
            "licences": dict(lic),
            "recording_environments": dict(env),
            "approximate_distances": dict(dist),
            "critical": bool(classes[name].get("critical")),
        }
    tot_env = Counter(r["recording_environment"] for r in rows)
    tot_dist = Counter(r["approximate_distance"] for r in rows)
    return {
        "schema_version": "1.0.0",
        "generated": date.today().isoformat(),
        "generated_by": "audio_dataset/scripts/assemble_manifest.py",
        "manifest": "audio_dataset/manifest.csv",
        "manifest_sha256": None,  # filled by the caller once written
        "note": ("One row per audio file on disk. 'originals' counts towards the SRS >=3000 "
                 "unique-original floor; augmented rows and derived segments do not. "
                 "'originals_synthetic' are generated by this project's own code and are "
                 "labelled as such in the manifest (source=procedural_synthesis)."),
        "totals": {
            "rows": len(rows),
            "originals": sum(1 for r in rows if r["original_or_augmented"] == "original"),
            "augmented": sum(1 for r in rows if r["original_or_augmented"] == "augmented"),
            "distinct_classes": len({r["class_label"] for r in rows}),
            "recording_environments": dict(tot_env),
            "approximate_distances": dict(tot_dist),
        },
        "per_class": per_class,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Assemble the frozen SonicSentinel manifest.")
    ap.add_argument("--source", type=Path, action="append", default=None,
                    help="source CSV (repeatable). Default: real rows + generated rows")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = ap.parse_args(argv)

    sources = args.source if args.source else DEFAULT_SOURCES
    sources = [p if p.is_absolute() else (REPO_ROOT / p) for p in sources]

    try:
        cols, rows, meta = assemble(sources)
    except AssemblyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    classes = load_classes()
    stats = statistics(rows, classes)

    print(f"assembled {len(rows)} rows from {len(sources)} source(s)")
    for src, n in meta["sources"].items():
        print(f"  {n:>6}  {src}")
    print()
    print(f"  {'class':<24} {'orig':>5} {'real':>5} {'synth':>6} {'floor':>6}")
    for name in sorted(classes, key=lambda c: (not classes[c].get("critical"), c)):
        p = stats["per_class"][name]
        flag = "" if p["meets_floor"] else "  <-- BELOW FLOOR"
        print(f"  {name:<24} {p['originals']:>5} {p['originals_real_field_recording']:>5} "
              f"{p['originals_synthetic']:>6} {p['required_originals']:>6}{flag}")
    t = stats["totals"]
    print(f"  {'TOTAL':<24} {t['originals']:>5} "
          f"{sum(p['originals_real_field_recording'] for p in stats['per_class'].values()):>5} "
          f"{sum(p['originals_synthetic'] for p in stats['per_class'].values()):>6}")
    print(f"\nenvironments: {t['recording_environments']}")
    print(f"distances   : {t['approximate_distances']}")

    if args.check:
        print("\n--check: nothing written.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    import hashlib
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    stats["manifest_sha256"] = digest
    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    STATS_OUT.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {args.out}  ({len(rows)} rows, sha256 {digest[:16]}...)")
    print(f"wrote {STATS_OUT}")
    print("Next: .venv/bin/python audio_dataset/build_split.py --manifest audio_dataset/manifest.csv --strict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
