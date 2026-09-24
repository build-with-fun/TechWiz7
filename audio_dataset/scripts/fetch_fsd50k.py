#!/usr/bin/env python3
"""
SonicSentinel AI -- real field-recording fetcher (FSD50K).

Owner: omar (data sourcing).  Authority: SRS v1.0 Step 1 ("collect, create, or
ethically source labelled audio recordings"), FR xvii, deliverable 3.

WHAT THIS DOES
--------------
Downloads REAL, openly-licensed field recordings of the target sound classes from
FSD50K and writes one manifest row per file in the frozen schema of
`audio_dataset/manifest_schema.md` (owner: lorena).

Provenance is per FILE, not per corpus. FSD50K ships a per-clip metadata file
(`*_clips_info_FSD50K.json`) giving each clip's own Freesound licence and uploader.
We read it and RECORD the licence of the individual clip, so a reviewer can check
any single file. Clips whose licence is not on the accept-list -- notably every
CC-BY-NC clip -- are never downloaded at all.

DESIGN DECISIONS (and why)
--------------------------
1. **One clip, one class.** A clip is assigned to exactly one class: the first
   class in the configured priority order whose label list it matches. FSD50K
   clips are multi-label, so without this rule the same recording could be counted
   as an "original" in two classes and the 300-per-class floor would be a fiction.
2. **Round-robin across uploaders.** Within a class, candidates are taken one at a
   time from each distinct Freesound uploader before taking a second from any.
   Taking the first N by id would concentrate a class in one recordist's mic,
   room and habits -- a model trained on that learns the recordist, not the sound.
   This is the cheapest way to satisfy SRS 1.5's variation requirement.
3. **Deterministic.** No `random`: sort by (uploader, int(fname)) and take in
   order. Re-running yields the same selection on any machine.
4. **Resumable and verifiable.** A file already on disk with a valid RIFF header
   is not re-fetched. Duration/rate/channels are MEASURED with ffprobe, never
   assumed from the request.
5. **Empty `dataset_split`.** `audio_dataset/build_split.py` (lorena) is the only
   thing that assigns a split. This script must not.

Usage
-----
    .venv/bin/python audio_dataset/scripts/fetch_fsd50k.py --check          # plan only
    .venv/bin/python audio_dataset/scripts/fetch_fsd50k.py --workers 8      # fetch
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import requests

# --------------------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_DATASET = SCRIPT_DIR.parent                 # audio_dataset/
REPO_ROOT = AUDIO_DATASET.parent                  # sonicsentinel-ai/
RAW = AUDIO_DATASET / "raw_downloads"
GT_DIR = RAW / "fsd50k_gt" / "FSD50K.ground_truth"
POOL = RAW / "fsd50k_audio"                       # flat download pool
ORIGINALS = AUDIO_DATASET / "originals"
MANIFESTS = AUDIO_DATASET / "manifests"
CLASS_MAP_PATH = SCRIPT_DIR / "class_label_map.json"
STATS_PATH = MANIFESTS / "fsd50k_fetch_stats.json"

FSD50K_BASE = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips"
FREESOUND_PAGE = "https://freesound.org/s/{fname}/"

# Frozen manifest column order -- audio_dataset/manifest_schema.md
FROZEN_COLUMNS = [
    "audio_id", "filename", "class_label", "source", "source_url", "licence", "author",
    "date_fetched", "duration_sec", "sampling_rate", "channels", "recording_environment",
    "recording_device", "approximate_distance", "original_or_augmented", "parent_audio_id",
    "segment_start_sec", "segment_end_sec", "sha256", "dataset_split",
]
# Extra columns beyond the frozen set are preserved by build_split.py; these carry
# provenance a reviewer needs to re-verify the row without re-downloading anything.
EXTRA_COLUMNS = ["freesound_id", "fsd50k_split", "fsd50k_labels", "licence_url",
                 "environment_basis", "fetch_batch"]

PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with PRINT_LOCK:
        print(msg, flush=True)


# --------------------------------------------------------------------------------------
# Environment inference
# --------------------------------------------------------------------------------------
# FSD50K does not state a recording environment. We infer one from the clip's OWN
# title, description and tags (which the uploader wrote), and we record how we got
# it in `environment_basis` -- `stated` if the word appears in the description,
# `inferred_from_tags` if only from tags, `unmatched` if nothing matched. An
# evaluator can therefore see exactly which rows are a claim and which are a guess.

ENV_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("studio",   ("studio", "recording studio", "booth", "sound design", "sounddesign")),
    ("vehicle",  ("in car", "incar", "car interior", "vehicle", "cockpit", "dashboard",
                  "inside a car", "car cabin", "bus interior", "train interior")),
    ("outdoor",  ("outdoor", "outside", "field recording", "field-recording", "street",
                  "park", "forest", "beach", "sea", "ocean", "garden", "urban", "city",
                  "traffic", "road", "rain", "wind", "thunderstorm", "nature", "wild")),
    ("indoor",   ("indoor", "room", "kitchen", "office", "hallway", "hall", "basement",
                  "garage", "warehouse", "factory", "workshop", "church", "bathroom",
                  "house", "home", "apartment", "classroom", "laboratory", "lab")),
]


def infer_environment(meta: dict) -> tuple[str, str]:
    """Return (environment, basis). Never invents: unmatched stays explicit."""
    tags = " ".join(meta.get("tags") or []).lower()
    title = (meta.get("title") or "").lower()
    desc = (meta.get("description") or "").lower()
    stated_text = f"{title} {desc}"
    for env, words in ENV_RULES:
        if any(w in stated_text for w in words):
            return env, "stated"
    for env, words in ENV_RULES:
        if any(w in tags for w in words):
            return env, "inferred_from_tags"
    return "unspecified", "unmatched"


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def load_allowed_licences(cfg: dict) -> dict[str, str]:
    return dict(cfg.get("licence_policy", {}).get("accept", {}))


def load_corpus_info() -> dict[str, dict]:
    """Per-clip FSD50K metadata (licence, uploader) from both dev and eval."""
    info: dict[str, dict] = {}
    for name in ("dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"):
        path = RAW / name
        if not path.exists():
            raise SystemExit(
                f"missing {path}\nFetch it first:\n"
                f"  curl -L -o {path} https://huggingface.co/datasets/Fhrozen/FSD50k/"
                f"resolve/main/metadata/{name}"
            )
        info.update(json.loads(path.read_text(encoding="utf-8")))
    return info


def load_ground_truth() -> list[dict]:
    """Every (fname, labels, fsd50k_split) row from the dev and eval ground truth."""
    rows: list[dict] = []
    for split in ("dev", "eval"):
        path = GT_DIR / f"{split}.csv"
        if not path.exists():
            raise SystemExit(f"missing ground truth {path}")
        for r in csv.DictReader(path.open(encoding="utf-8")):
            labels = [l for l in r["labels"].split(",") if l]
            rows.append({"fname": r["fname"], "labels": labels, "fsd50k_split": split})
    return rows


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------

def assign_and_select(cfg: dict, gt_rows: list[dict], info: dict[str, dict],
                      allowed: dict[str, str]) -> tuple[dict[str, list[dict]], dict]:
    """Assign each licence-clean clip to one class, then pick targets round-robin by uploader."""
    classes = cfg["classes"]
    priority = cfg["assignment_priority"]
    order = {c: i for i, c in enumerate(priority)}

    stats = {"clips_total": len(gt_rows), "licence_rejected": 0, "licence_missing": 0,
             "assigned": Counter(), "unmatched_clips": 0}

    # candidate pool per class
    pool: dict[str, list[dict]] = defaultdict(list)
    for row in gt_rows:
        fname = row["fname"]
        meta = info.get(fname)
        if meta is None:
            stats["licence_missing"] += 1
            continue
        lic_url = (meta.get("license") or "").strip()
        if lic_url not in allowed:
            stats["licence_rejected"] += 1
            continue
        label_set = set(row["labels"])
        # first class in priority order that this clip matches -> assigned there only
        chosen = None
        for cls in sorted(classes, key=lambda c: order.get(c, 999)):
            if label_set & set(classes[cls]["labels"]):
                chosen = cls
                break
        if chosen is None:
            stats["unmatched_clips"] += 1
            continue
        env, basis = infer_environment(meta)
        pool[chosen].append({
            "fname": fname,
            "uploader": (meta.get("uploader") or "unknown").strip() or "unknown",
            "licence_url": lic_url,
            "licence": allowed[lic_url],
            "fsd50k_labels": ",".join(row["labels"]),
            "fsd50k_split": row["fsd50k_split"],
            "environment": env,
            "environment_basis": basis,
        })
        stats["assigned"][chosen] += 1

    selected: dict[str, list[dict]] = {}
    for cls in classes:
        target = int(classes[cls].get("real_target", 0))
        cands = pool.get(cls, [])
        if target == 0 or not cands:
            selected[cls] = []
            continue
        # bucket by uploader, deterministic order inside each bucket
        by_uploader: dict[str, list[dict]] = defaultdict(list)
        for c in sorted(cands, key=lambda c: (c["uploader"], int(c["fname"]))):
            by_uploader[c["uploader"]].append(c)
        # rotate uploaders (most-prolific first, name as tiebreak) so every
        # recordist is represented before any is used twice
        uploaders = sorted(by_uploader, key=lambda u: (-len(by_uploader[u]), u))
        picked: list[dict] = []
        idx = 0
        while len(picked) < target:
            progressed = False
            for u in uploaders:
                if idx < len(by_uploader[u]):
                    picked.append(by_uploader[u][idx])
                    progressed = True
                    if len(picked) >= target:
                        break
            if not progressed:
                break
            idx += 1
        selected[cls] = picked

    return selected, stats


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------

def valid_wav(path: Path) -> bool:
    """A file is reusable only if it is a real RIFF/WAVE container of non-trivial size."""
    if not path.exists() or path.stat().st_size < 512:
        return False
    with path.open("rb") as fh:
        head = fh.read(12)
    return head[:4] == b"RIFF" and head[8:12] == b"WAVE"


def probe(path: Path) -> dict:
    """Measure duration / sample rate / channels with ffprobe. Never assume."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels,duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()[:200]}")
    stream = json.loads(out.stdout)["streams"][0]
    duration = stream.get("duration")
    if duration in (None, "N/A"):
        out2 = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60)
        duration = json.loads(out2.stdout)["format"]["duration"]
    return {"duration_sec": round(float(duration), 3),
            "sampling_rate": int(stream["sample_rate"]),
            "channels": int(stream["channels"])}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(session: requests.Session, item: dict) -> dict:
    """Fetch one clip into the flat pool. Returns item + status."""
    fname = item["fname"]
    dest = POOL / f"{fname}.wav"
    if valid_wav(dest):
        return {**item, "status": "cached", "pool_path": dest}
    url = f"{FSD50K_BASE}/{item['fsd50k_split']}/{fname}.wav"
    tmp = dest.with_suffix(".part")
    last = ""
    for attempt in range(3):
        try:
            r = session.get(url, timeout=120, stream=True)
            if r.status_code == 404:
                return {**item, "status": "not_found", "pool_path": None}
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    fh.write(chunk)
            if not valid_wav(tmp):
                last = "not a RIFF/WAVE file"
                tmp.unlink(missing_ok=True)
                time.sleep(1.5 * (attempt + 1))
                continue
            tmp.replace(dest)
            return {**item, "status": "downloaded", "pool_path": dest}
        except Exception as exc:  # noqa: BLE001 - report, retry, never crash the batch
            last = f"{type(exc).__name__}: {exc}"
            tmp.unlink(missing_ok=True)
            time.sleep(1.5 * (attempt + 1))
    return {**item, "status": "failed", "pool_path": None, "error": last}


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch real, licence-checked FSD50K field recordings.")
    ap.add_argument("--check", action="store_true", help="print the plan, download nothing")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=MANIFESTS / "fsd50k_real_rows.csv")
    args = ap.parse_args(argv)

    cfg = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
    allowed = load_allowed_licences(cfg)
    info = load_corpus_info()
    gt_rows = load_ground_truth()
    selected, stats = assign_and_select(cfg, gt_rows, info, allowed)

    total = sum(len(v) for v in selected.values())
    log(f"FSD50K licence-clean clips: {stats['clips_total'] - stats['licence_rejected'] - stats['licence_missing']}"
        f"  (rejected NC/unlicensed: {stats['licence_rejected']}, no metadata: {stats['licence_missing']})")
    log(f"selected for download: {total} clips")
    for cls in cfg["assignment_priority"]:
        cands = stats["assigned"].get(cls, 0)
        sel = len(selected[cls])
        target = cfg["classes"][cls]["real_target"]
        flag = "" if sel >= target else "  <-- SHORT (population exhausted)"
        log(f"  {cls:<24} pool={cands:>5}  selected={sel:>4}/{target:<4}{flag}")

    if args.check:
        log("\n--check: nothing downloaded.")
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        STATS_PATH.write_text(json.dumps(
            {"mode": "check", "generated": date.today().isoformat(),
             "licence_rejected": stats["licence_rejected"],
             "no_metadata": stats["licence_missing"],
             "assigned_pool": dict(stats["assigned"]),
             "selected": {c: len(v) for c, v in selected.items()},
             "targets": {c: cfg["classes"][c]["real_target"] for c in cfg["classes"]}},
            indent=2) + "\n", encoding="utf-8")
        return 0

    POOL.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    # ---- download -----------------------------------------------------------------
    flat = [item for cls in cfg["assignment_priority"] for item in selected[cls]]
    for item in flat:
        item.setdefault("class_label", None)
    # tag each item with its class (selected is keyed by class)
    for cls, items in selected.items():
        for it in items:
            it["class_label"] = cls

    log(f"\ndownloading {len(flat)} clips with {args.workers} workers -> {POOL}")
    session = requests.Session()
    session.headers.update({"User-Agent": "SonicSentinel-AI-dataset/1.0 (competition dataset build)"})
    results: list[dict] = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, session, it): it for it in flat}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 25 == 0 or done == len(flat):
                rate = done / max(time.time() - t0, 1e-6)
                log(f"  {done}/{len(flat)}  ({rate:.1f} clips/s)")

    ok = [r for r in results if r["status"] in ("downloaded", "cached")]
    bad = [r for r in results if r["status"] not in ("downloaded", "cached")]
    log(f"\nfetched ok: {len(ok)}   failed/not-found: {len(bad)}")

    # ---- measure + place + build rows ---------------------------------------------
    rows: list[dict] = []
    # audio ids are assigned AFTER selection, grouped by class, in deterministic order,
    # using the frozen SS-<CODE>-<NNNN> scheme from config/classes.json
    classes_cfg = json.loads((REPO_ROOT / "config" / "classes.json").read_text(encoding="utf-8"))
    code = {c["name"]: c["code"] for c in classes_cfg["classes"]}
    today = date.today().isoformat()
    # Real field recordings are numbered from 1001, synthetic from 0001 (see
    # DATA_DICTIONARY.md). The two producers must never share an id range, because
    # audio_id is the primary key -- a collision would silently drop a row from the
    # assembled manifest and corrupt the per-class counts we report.
    per_class_counter: dict[str, int] = defaultdict(lambda: 1000)
    measure_fail: list[str] = []

    for cls in cfg["assignment_priority"]:
        slug = cls.lower().replace(" ", "_")
        out_dir = ORIGINALS / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in sorted([x for x in ok if x["class_label"] == cls],
                        key=lambda x: (x["uploader"], int(x["fname"]))):
            per_class_counter[cls] += 1
            audio_id = f"SS-{code[cls]}-{per_class_counter[cls]:04d}"
            target = out_dir / f"{audio_id}.wav"
            src = r["pool_path"]
            if target.exists() and valid_wav(target):
                pass
            else:
                try:
                    os.link(src, target)          # hardlink: same bytes, no extra disk
                except OSError:
                    shutil.copy2(src, target)
            try:
                m = probe(target)
            except Exception as exc:  # noqa: BLE001
                measure_fail.append(f"{audio_id}: {exc}")
                target.unlink(missing_ok=True)
                per_class_counter[cls] -= 1
                continue
            rel = target.relative_to(AUDIO_DATASET)
            rows.append({
                "audio_id": audio_id,
                "filename": str(rel),
                "class_label": cls,
                "source": f"FSD50K (Freesound clip {r['fname']})",
                "source_url": FREESOUND_PAGE.format(fname=r["fname"]),
                "licence": r["licence"],
                "author": r["uploader"],
                "date_fetched": today,
                "duration_sec": m["duration_sec"],
                "sampling_rate": m["sampling_rate"],
                "channels": m["channels"],
                "recording_environment": r["environment"],
                "recording_device": "unspecified",     # FSD50K metadata does not state the device
                "approximate_distance": "n/a",         # not recorded by the uploader
                "original_or_augmented": "original",
                "parent_audio_id": "",
                "segment_start_sec": "",
                "segment_end_sec": "",
                "sha256": sha256_of(target),
                "dataset_split": "",
                # extras (preserved by build_split.py)
                "freesound_id": r["fname"],
                "fsd50k_split": r["fsd50k_split"],
                "fsd50k_labels": r["fsd50k_labels"],
                "licence_url": r["licence_url"],
                "environment_basis": r["environment_basis"],
                "fetch_batch": "fsd50k_real_v1",
            })

    # ---- write ---------------------------------------------------------------------
    cols = FROZEN_COLUMNS + EXTRA_COLUMNS
    rows.sort(key=lambda r: r["audio_id"])
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    final = Counter(r["class_label"] for r in rows)
    log(f"\nwrote {len(rows)} rows -> {args.out}")
    for cls in cfg["assignment_priority"]:
        log(f"  {cls:<24} {final.get(cls, 0):>4}")
    env_basis = Counter(r["environment_basis"] for r in rows)
    log(f"environment provenance: {dict(env_basis)}")
    lic = Counter(r["licence"] for r in rows)
    log(f"licences: {dict(lic)}")

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps({
        "generated": today, "mode": "fetch",
        "clips_scanned": stats["clips_total"],
        "licence_rejected_nc_or_unlicensed": stats["licence_rejected"],
        "no_metadata": stats["licence_missing"],
        "candidate_pool_per_class": dict(stats["assigned"]),
        "selected_per_class": dict(final),
        "targets_per_class": {c: cfg["classes"][c]["real_target"] for c in cfg["classes"]},
        "download_failed_or_missing": len(bad),
        "measure_failures": measure_fail[:20],
        "licence_counts": dict(lic),
        "environment_basis": dict(env_basis),
    }, indent=2) + "\n", encoding="utf-8")
    if bad:
        log(f"NOTE: {len(bad)} clips failed — see {STATS_PATH.name}; they are simply not in the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
