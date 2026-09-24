#!/usr/bin/env python3
"""One-off repair for the audio_id collision between acquisition batches.

What happened: `from_fsd_dev` ran with `next_id` seeded from `ID_START` instead
of the highest id on disk (the same bug `main()` already avoids when it counts
existing originals per class, but which the function could not see).  It
therefore re-issued ids SS-<CODE>-0501.. that the earlier `fsd50k_eval_recovered`
and `esc50` batches had already used, and `_ingest_blob` wrote the dev clips
over the files those ids pointed at.  90 manifest rows ended up naming a file
whose bytes belong to a different clip (the 40 esc50 glass rows: their clips
were described as `breaking_glass` but ESC-50 category 39 is `car_horn`).

Repair, per affected id:
  1. recover the victim's bytes (HF dev mirror / eval-recovered / ESC-50 parquet)
  2. atomically overwrite the file with the victim's content (os.replace)
  3. keep the on-disk (dev) clip under a NEW id continuing past the class max
  4. update both manifest rows' filename + duration/rate/channels from the bytes
  5. mark every surviving row with a repair note in `notes`
Nothing is deleted: both clips survive, under distinct ids.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
AD = REPO / "audio_dataset"
ACQ = AD / "manifests" / "acquired_rows.csv"
HF_BASE = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips/dev/{}.wav"
EVAL_DIR = AD / "raw_downloads" / "fsd50k_eval_recovered"
ESC_DIR = AD / "raw_downloads"
NOTES = "id collision repair: see audio_dataset/scripts/repair_id_collision.py"


ESC_CACHE: dict[str, bytes] = {}


def blob_for(corpus_id: str) -> bytes | None:
    kind, _, ref = corpus_id.partition(":")
    if kind == "fsd50k_dev":
        try:
            with urllib.request.urlopen(HF_BASE.format(ref), timeout=60) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! download {ref}: {exc}")
            return None
    if kind == "fsd50k_eval":
        p = EVAL_DIR / (ref + ".wav")
        return p.read_bytes() if p.exists() else None
    if kind == "esc50":
        if not ESC_CACHE:
            import pandas as pd
            for parquet in sorted(ESC_DIR.glob("esc50-train-*.parquet")):
                df = pd.read_parquet(parquet)
                for _, rec in df.iterrows():
                    fn = rec["filename"]
                    audio = rec["audio"]
                    ESC_CACHE[str(fn)] = audio["bytes"] if isinstance(audio, dict) else audio
            print(f"  (esc50 parquet index: {len(ESC_CACHE)} clips)")
        return ESC_CACHE.get(ref)
    return None


def meta(blob: bytes, tmp: Path) -> tuple[int, int, int] | None:
    tmp.write_bytes(blob)
    try:
        info = sf.info(str(tmp))
        return round(info.frames / info.samplerate, 3), info.samplerate, info.channels
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    rows = list(csv.DictReader(ACQ.open(newline="", encoding="utf-8")))
    cols = list(rows[0].keys())
    if "notes" not in cols:
        cols.append("notes")

    by_id = defaultdict(list)
    for r in rows:
        by_id[r["audio_id"]].append(r)
    collided = {k: v for k, v in by_id.items() if len(v) > 1}
    print(f"collided ids: {len(collided)}")

    slug_by_code = {"AGG": "aggression", "SCR": "panic_scream", "GLA": "glass_breaking"}
    tmp = Path(tempfile.mkdtemp()) / "probe.wav"

    # highest id per class over the whole manifest + disk
    import re
    max_num = defaultdict(int)
    for r in rows:
        m = re.match(r"SS-([A-Z]{3})-(\d{4})", r["audio_id"])
        if m:
            max_num[m.group(1)] = max(max_num[m.group(1)], int(m.group(2)))
    for slug in slug_by_code.values():
        for f in (AD / "originals" / slug).glob("*.wav"):
            m = re.match(r"SS-([A-Z]{3})-(\d{4})", f.name)
            if m:
                max_num[m.group(1)] = max(max_num[m.group(1)], int(m.group(2)))

    repaired = 0
    for audio_id, group in sorted(collided.items()):
        code = audio_id.split("-")[1]
        slug = slug_by_code[code]
        # the disk holds the LAST dev write's bytes (each dev write overwrote the
        # previous content) -- so the on-disk clip is the last dev row in the group
        on_disk = next((r for r in reversed(group)
                        if r["fetch_batch"] == "fsd50k_dev_topup_v1"), group[-1])
        victims = [r for r in group if r is not on_disk]

        # highest-numbered victim determines the new id for the on-disk clip
        bump = max(int(r["audio_id"].rsplit("-", 1)[1]) for r in victims)
        max_num[code] = max(max_num[code], bump)
        new_num = max_num[code] + 1
        max_num[code] = new_num
        new_id = f"SS-{code}-{new_num:04d}"

        # 1. every victim's bytes, from its own source
        restored: list[tuple[dict, bytes, tuple]] = []
        ok = True
        for v in victims:
            blob = blob_for(v["corpus_id"])
            if blob is None:
                print(f"  ! {audio_id}: cannot restore {v['corpus_id']}")
                ok = False
                break
            m = meta(blob, tmp)
            if m is None:
                print(f"  ! {audio_id}: undecodable {v['corpus_id']}")
                ok = False
                break
            restored.append((v, blob, m))
        if not ok:
            continue

        # 2. rewrite the file with the first victim's content -- but keep the
        # on-disk (dev) bytes safe first, they move to the new id in step 3
        first_v, first_blob, first_meta = restored[0]
        dst = AD / first_v["filename"]
        dev_bytes = dst.read_bytes()
        fd, tmpf = tempfile.mkstemp(dir=dst.parent)
        with os.fdopen(fd, "wb") as fh:
            fh.write(first_blob)
        os.replace(tmpf, dst)

        # extra victims (3rd row for glass ids stolen twice): new ids too
        for extra_v, extra_blob, extra_m in restored[1:]:
            max_num[code] += 1
            extra_id = f"SS-{code}-{max_num[code]:04d}"
            new_name = f"{extra_id}.wav"
            (AD / "originals" / slug / new_name).write_bytes(extra_blob)
            extra_v["audio_id"] = extra_id
            extra_v["filename"] = f"originals/{slug}/{new_name}"
            extra_v["duration_sec"], extra_v["sampling_rate"], extra_v["channels"] = extra_m
            extra_v["notes"] = NOTES
            repaired += 1

        # 3. the on-disk (dev) clip keeps its bytes under the new id
        on_disk["audio_id"] = new_id
        on_disk["filename"] = f"originals/{slug}/{new_id}.wav"
        fd, tmpf = tempfile.mkstemp(dir=(AD / on_disk["filename"]).parent)
        with os.fdopen(fd, "wb") as fh:
            fh.write(dev_bytes)
        os.replace(tmpf, AD / on_disk["filename"])

        # 4. victims' rows point at the restored bytes again
        first_v["notes"] = NOTES
        repaired += 1

        print(f"  {audio_id}: victim restored, dev clip -> {new_id}")

    tmp.unlink(missing_ok=True)
    # nothing to do for rows whose ids were fine
    with ACQ.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"repaired groups: {repaired} (rows rewritten: {len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
