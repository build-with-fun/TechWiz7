"""One-off fix (already applied): two crashed acquire_corpus.py runs both started at
SS-*-0501 and overwrote each other's files. Each damaged clip is taken again from the raw
datasets, checked against its sha256 and written back.

    .venv/bin/python data/repair_collided_blobs.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AD = REPO / "audio_dataset"
MANIFEST = AD / "manifest.csv"
OUT = REPO / "data" / "repair_collided_report.json"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_collided() -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Run the verifier's two checks; return rows by id and hash groups."""
    rows: dict[str, dict] = {}
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["audio_id"]] = r
    bad: list[str] = []
    digests: dict[str, list[str]] = defaultdict(list)
    for aid, r in rows.items():
        p = AD / r["filename"]
        if not p.exists():
            continue
        d = sha256_of(p)
        digests[d].append(aid)
        if d != r["sha256"]:
            bad.append(aid)
    dup = {d: ids for d, ids in digests.items() if len(ids) > 1}
    return rows, dup


def ensure_eval_labels() -> None:
    """from_fsd_eval reads /tmp/eval_labels.json; rebuild it from the corpus zip."""
    labels_json = Path("/tmp/eval_labels.json")
    if labels_json.exists():
        return
    zpath = AD / "raw_downloads" / "FSD50K.ground_truth.zip"
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith("eval.csv"))
        text = z.read(name).decode("utf-8")
    labels = {}
    for line in text.splitlines()[1:]:
        parts = line.split(",", 2)
        if len(parts) >= 2 and parts[0]:
            labels[parts[0]] = parts[1]
    labels_json.write_text(json.dumps(labels), encoding="utf-8")
    print(f"rebuilt /tmp/eval_labels.json ({len(labels)} entries)")


def harvest_needed(needed: dict[str, set[str]]) -> dict[str, bytes]:
    """Read the raw datasets once and keep only the clips we need."""
    import pandas as pd

    blobs: dict[str, bytes] = {}

    if needed.get("esc50"):
        want = needed["esc50"]
        for pq in sorted((AD / "raw_downloads").glob("esc50-train-*.parquet")):
            df = pd.read_parquet(pq)
            for rec in df.itertuples(index=False):
                key = f"esc50:{rec.filename}"
                if key in want:
                    blobs[key] = rec.audio["bytes"]
                    want.discard(key)
            if not want:
                break
        if want:
            print(f"  WARN esc50 not found in cache: {sorted(want)[:5]}")

    if needed.get("us8k"):
        want = needed["us8k"]
        for pq in sorted((REPO / "data" / "uscraping" / "parquet").glob("train-*.parquet")):
            df = pd.read_parquet(pq)
            for rec in df.itertuples(index=False):
                key = f"us8k:{rec.slice_file_name}"
                if key in want:
                    blobs[key] = rec.audio["bytes"]
                    want.discard(key)
            if not want:
                break
        if want:
            print(f"  WARN us8k not found in cache: {sorted(want)[:5]}")

    if needed.get("fsd50k_eval"):
        want = needed["fsd50k_eval"]
        wav_dir = AD / "raw_downloads" / "fsd50k_eval_recovered"
        for key in sorted(want):
            stem = key.split(":", 1)[1]
            p = wav_dir / f"{stem}.wav"
            if p.exists():
                blobs[key] = p.read_bytes()
            else:
                print(f"  WARN fsd_eval wav missing: {p}")
    return blobs


def main() -> int:
    rows, dup = find_collided()
    bad = [aid for aid, r in rows.items()
           if (AD / r["filename"]).exists() and sha256_of(AD / r["filename"]) != r["sha256"]]
    print(f"collided rows: {len(bad)}  duplicate groups: {len(dup)}")
    if not bad:
        print("nothing to repair")
        return 0

    ensure_eval_labels()

    # Which cache does each bad row come from?
    by_src: dict[str, set[str]] = defaultdict(set)
    for aid in bad:
        cid = rows[aid]["corpus_id"]
        src = cid.split(":", 1)[0]
        if src not in ("esc50", "us8k", "fsd50k_eval"):
            print(f"  UNREPAIRABLE {aid}: unknown source {src} ({cid})")
            continue
        by_src[src].add(cid)

    print("harvesting original blobs from raw caches ...")
    blobs = harvest_needed(by_src)
    print(f"  harvested {len(blobs)} blobs")

    repaired, failed = [], []
    for aid in sorted(bad):
        r = rows[aid]
        cid = r["corpus_id"]
        blob = blobs.get(cid)
        if blob is None:
            failed.append({"audio_id": aid, "reason": "blob not in cache", "corpus_id": cid})
            continue
        digest = hashlib.sha256(blob).hexdigest()
        if digest != r["sha256"]:
            failed.append({"audio_id": aid, "reason": "sha mismatch vs manifest row",
                           "corpus_id": cid, "got": digest, "want": r["sha256"]})
            continue
        (AD / r["filename"]).write_bytes(blob)
        repaired.append(aid)

    print(f"repaired {len(repaired)} / {len(bad)}; failures: {len(failed)}")
    for f in failed:
        print(f"  FAILED {f['audio_id']}: {f['reason']}")
    OUT.write_text(json.dumps({"repaired": repaired, "failed": failed}, indent=2))
    print(f"report -> {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
