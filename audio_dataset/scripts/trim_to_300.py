#!/usr/bin/env python3
"""One-off trim (already applied) to exactly 300 originals per class.

Synthetic surplus goes first, then surplus dev-acquired clips; this also resolved 13
SS-GLA-0700..0712 filename collisions. Dropped files move to data/dataset_overflow/.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AD = REPO / "audio_dataset"
OVERFLOW = REPO / "data" / "dataset_overflow"
OVERFLOW.mkdir(parents=True, exist_ok=True)

MANIFESTS = {
    "real": AD / "manifests" / "fsd50k_real_rows.csv",
    "acq": AD / "manifests" / "acquired_rows.csv",
    "tts": AD / "manifests" / "help_tts_rows.csv",
    "top": AD / "manifests" / "synthetic_topup_rows.csv",
}


def load(name: str) -> list[dict]:
    with MANIFESTS[name].open(newline="") as fh:
        return list(csv.DictReader(fh))


def save(name: str, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with MANIFESTS[name].open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def move_clip(rel_filename: str) -> None:
    src = AD / rel_filename
    if not src.exists():
        return
    dest = OVERFLOW / Path(rel_filename).name
    shutil.move(str(src), str(dest))


def main() -> None:
    rows = {n: load(n) for n in MANIFESTS}

    # 1) Drop all 67 synthetic glass clips (their filenames collide with dev clips).
    keep_top = []
    for r in rows["top"]:
        if r["class_label"] == "Glass Breaking":
            move_clip(r["filename"])
            continue
        keep_top.append(r)

    # 2) Synthetic Aggression 45 -> 40 and Panic 47 -> 24 (highest ids first).
    for label, target in (("Aggression", 40), ("Panic Scream", 24)):
        sel = sorted(
            (r for r in keep_top if r["class_label"] == label),
            key=lambda r: int(r["audio_id"].split("-")[2]),
        )
        for r in sel[target:]:
            move_clip(r["filename"])
            keep_top.remove(r)

    # 3) Trim dev glass by 51, highest ids first; removes the colliding 0700..0712 clips.
    acq_glass_dev = sorted(
        (
            r
            for r in rows["acq"]
            if r["class_label"] == "Glass Breaking"
            and r["fetch_batch"] == "fsd50k_dev_topup_v1"
        ),
        key=lambda r: int(r["audio_id"].split("-")[2]),
        reverse=True,
    )
    drop_ids = {r["audio_id"] for r in acq_glass_dev[:51]}
    keep_acq = []
    for r in rows["acq"]:
        if r["audio_id"] in drop_ids:
            move_clip(r["filename"])
            continue
        keep_acq.append(r)

    save("top", keep_top)
    save("acq", keep_acq)

    # 4) Report final tallies.
    import collections

    totals: collections.Counter = collections.Counter()
    for n in MANIFESTS:
        for r in load(n):
            totals[r["class_label"]] += 1
    print("final per-class original totals:")
    for cl, n in sorted(totals.items()):
        flag = "" if n == 300 else "  <-- NOT 300"
        print(f"  {cl:<24} {n}{flag}")
    print("grand total:", sum(totals.values()))


if __name__ == "__main__":
    main()
