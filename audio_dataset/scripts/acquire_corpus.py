#!/usr/bin/env python3
"""Copy CC-BY recordings from ESC-50, UrbanSound8K and FSD50K into
audio_dataset/originals/<class>/ (from SS-<CODE>-0501.wav) and write
audio_dataset/manifests/acquired_rows.csv for assemble_manifest.py.

Only source categories on the allow-lists below are used. Fireworks is not counted as
Gunshot. Duplicates are dropped by sha256, every row has its licence, and dataset_split is
left blank for build_split.py.

    .venv/bin/python audio_dataset/scripts/acquire_corpus.py --per-class 300
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path

import soundfile as sf
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_DATASET = SCRIPT_DIR.parent
REPO_ROOT = AUDIO_DATASET.parent

OUT_MANIFEST = AUDIO_DATASET / "manifests" / "acquired_rows.csv"

# The frozen 20 (audio_dataset/manifest_schema.md), in order, then our extras.
FROZEN_COLUMNS = [
    "audio_id", "filename", "class_label", "source", "source_url", "licence",
    "author", "date_fetched", "duration_sec", "sampling_rate", "channels",
    "recording_environment", "recording_device", "approximate_distance",
    "original_or_augmented", "parent_audio_id", "segment_start_sec",
    "segment_end_sec", "sha256", "dataset_split",
]
EXTRA_COLUMNS = [
    "source_category", "corpus_id", "salience", "environment_basis",
    "fetch_batch", "original_filename", "licence_url", "notes",
]
ALL_COLUMNS = FROZEN_COLUMNS + EXTRA_COLUMNS

ID_START = 501  # FSD50K uses 0001..0150

# Allow-lists: source category -> our class.

# ESC-50 categories (lowercase, underscores) -> our class name.
ESC50_MAP = {
    # Alarm or Siren
    "siren":            "Alarm or Siren",
    "clock_alarm":      "Alarm or Siren",
    # Vehicle Horn
    "car_horn":         "Vehicle Horn",
    # Animal Sound: mammals, birds and insects.
    "dog":              "Animal Sound",
    "rooster":          "Animal Sound",
    "pig":              "Animal Sound",
    "cow":              "Animal Sound",
    "frog":             "Animal Sound",
    "cat":              "Animal Sound",
    "hen":              "Animal Sound",
    "sheep":            "Animal Sound",
    "crickets":         "Animal Sound",
    "chirping_birds":   "Animal Sound",
    "crow":             "Animal Sound",
    "insects":          "Animal Sound",
    # Glass Breaking
    "glass_breaking":   "Glass Breaking",
    # Machinery Fault: real machines/power tools.
    "engine":           "Machinery Fault",
    "chainsaw":         "Machinery Fault",
    "helicopter":       "Machinery Fault",
    "hand_saw":         "Machinery Fault",
    "vacuum_cleaner":   "Machinery Fault",
    "washing_machine":  "Machinery Fault",
    # Background Noise: wind, rain, water, HVAC.
    "wind":             "Background Noise",
    "rain":             "Background Noise",
    "sea_waves":        "Background Noise",
    "thunderstorm":     "Background Noise",
    "water_drops":      "Background Noise",
    "crackling_fire":   "Background Noise",
    # 'fireworks' is not Gunshot.
}

# UrbanSound8K classes -> our class name.
US8K_MAP = {
    "siren":            "Alarm or Siren",
    "car_horn":         "Vehicle Horn",
    "dog_bark":         "Animal Sound",
    "gun_shot":         "Gunshot",
    "jackhammer":       "Machinery Fault",
    "engine_idling":    "Machinery Fault",
    "air_conditioner":  "Background Noise",  # HVAC
    "drilling":         "Machinery Fault",
}

# FSD50K.eval ontology terms -> our class name, unambiguous terms only.
FSD_EVAL_MAP = {
    "Siren":                     "Alarm or Siren",
    "Alarm":                     "Alarm or Siren",
    "Car alarm":                 "Alarm or Siren",
    "Smoke detector":            "Alarm or Siren",
    "Car":                       "Vehicle Horn",
    "Train horn":                "Vehicle Horn",
    "Foghorn":                   "Vehicle Horn",
    "Glass":                     "Glass Breaking",
    "Breaking glass":            "Glass Breaking",
    "Glass shatter":             "Glass Breaking",
    "Shatter":                   "Glass Breaking",
    "Dog":                       "Animal Sound",
    "Bark":                      "Animal Sound",
    "Animal":                    "Animal Sound",
    "Bird":                      "Animal Sound",
    "Cat":                       "Animal Sound",
    "Rooster":                   "Animal Sound",
    "Chicken":                   "Animal Sound",
    "Gunshot, gunfire":          "Gunshot",
    "Machine gun":               "Gunshot",
    "Gunshot":                   "Gunshot",
    "Jackhammer":                "Machinery Fault",
    "Power tool":                "Machinery Fault",
    "Chainsaw":                  "Machinery Fault",
    "Engine":                    "Machinery Fault",
    "Air conditioning":          "Background Noise",
    "Traffic":                   "Background Noise",
    "Environmental noise":       "Background Noise",
    "Wind":                      "Background Noise",
    "Rain":                      "Background Noise",
    # Panic Scream is fear/distress only; yell/shout map to Aggression.
    "Screaming":                 "Panic Scream",
    "Scream":                    "Panic Scream",
    # Aggression: physical violence and hostile shouting.
    "Fight":                     "Aggression",
    "Fighting":                  "Aggression",
    "Yell":                      "Aggression",
    "Shout":                     "Aggression",
    "Argument":                  "Aggression",
    "Quarrel":                   "Aggression",
}

# Source details, one per corpus.

ESC50_PROV = {
    "source": "ESC-50 (Environmental Sound Classification)",
    "source_url": "https://github.com/karolpiczak/ESC-50",
    "licence": "CC-BY-4.0",
    "author": "Karol J. Piczak",
    "licence_url": "https://github.com/karolpiczak/ESC-50/blob/master/LICENSE",
    "fetch_batch": "esc50_v1",
    "recording_environment": "unspecified",
    "environment_basis": "corpus_metadata:ESC-50 field recordings (Freesound clips)",
}

US8K_PROV = {
    "source": "UrbanSound8K (urban sound slices)",
    "source_url": "https://urbansounddataset.weebly.com/urbansound8k.html",
    "licence": "CC-BY-4.0",
    "author": "Justin Salamon, Duncan Jacoby, Juan Pablo Bello",
    "licence_url": "https://creativecommons.org/licenses/by/4.0/",
    "fetch_batch": "urbansound8k_v1",
    "recording_environment": "unspecified",
    "environment_basis": "corpus_metadata:UrbanSound8K urban field recordings",
}

FSD_EVAL_PROV = {
    "source": "FSD50K.eval (recovered from truncated archive)",
    "source_url": "https://zenodo.org/record/4060432",
    "licence": "CC-BY-4.0",
    "author": "Eduardo Fonseca et al.",
    "licence_url": "https://creativecommons.org/licenses/by/4.0/",
    "fetch_batch": "fsd50k_eval_recovered_v1",
    "recording_environment": "unspecified",
    "environment_basis": "corpus_metadata:FSD50K Freesound field recordings",
}



def _audio_meta(blob: bytes, tmp: Path) -> tuple[float, int, int] | None:
    """(duration_sec, sampling_rate, channels), or None if undecodable."""
    try:
        tmp.write_bytes(blob)
        info = sf.info(str(tmp))
    except Exception:
        return None
    if info.frames == 0:
        return None
    return (info.duration, info.samplerate, 1 if info.channels == 1 else 2)


def _passes_quality(blob: bytes, tmp: Path) -> bool:
    """At least 0.5 s and louder than -50 dBFS RMS (same limits as QUALITY_DEFAULTS)."""
    min_dur, silence_rms = 0.5, 10 ** (-50.0 / 20.0)
    try:
        tmp.write_bytes(blob)
        y, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
        mono = y.mean(axis=1)
        if (len(mono) / sr) < min_dur:
            return False
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        return rms > silence_rms
    except Exception:
        return False


def _emit(rows: list[dict], prov: dict, category: str, audio_id: str,
          filename: str, class_label: str, meta: tuple[float, int, int],
          digest: str, extra: dict) -> None:
    dur, sr, ch = meta
    rows.append({
        "audio_id": audio_id,
        "filename": filename,
        "class_label": class_label,
        "source": prov["source"],
        "source_url": prov["source_url"],
        "licence": prov["licence"],
        "author": prov["author"],
        "date_fetched": date.today().isoformat(),
        "duration_sec": round(dur, 3),
        "sampling_rate": sr,
        "channels": ch,
        "recording_environment": prov["recording_environment"],
        "recording_device": "",
        "approximate_distance": "n/a",
        "original_or_augmented": "original",
        "parent_audio_id": "",
        "segment_start_sec": "",
        "segment_end_sec": "",
        "sha256": digest,
        "dataset_split": "",
        "source_category": category,
        "corpus_id": extra.get("corpus_id", ""),
        "salience": extra.get("salience", ""),
        "environment_basis": prov["environment_basis"],
        "fetch_batch": prov["fetch_batch"],
        "original_filename": extra.get("original_filename", ""),
        "licence_url": prov["licence_url"],
    })


def _ingest_blob(rows: list[dict], prov: dict, category: str, class_label: str,
                 slug: str, code: str, next_id: list[int], have_sha: set[str],
                 stats: Counter, skipped: Counter, blob: bytes, extra: dict,
                 tmp: Path, room: dict[str, int]) -> bool:
    """Decode a WAV blob and, if new and audible, save it and add a row. True if accepted."""
    meta = _audio_meta(blob, tmp)
    if meta is None:
        skipped["undecodable"] += 1
        return False
    if not _passes_quality(blob, tmp):
        skipped["quality_too_short_or_silent"] += 1
        return False
    # Hash the bytes we write.
    digest = hashlib.sha256(blob).hexdigest()
    if digest in have_sha:
        skipped["dupe_sha256"] += 1
        return False
    have_sha.add(digest)
    audio_id = f"SS-{code}-{next_id[0]:04d}"
    next_id[0] += 1
    out_dir = AUDIO_DATASET / "originals" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"originals/{slug}/{audio_id}.wav"
    (AUDIO_DATASET / filename).write_bytes(blob)
    stats[class_label] += 1
    _emit(rows, prov, category, audio_id, filename, class_label, meta, digest, extra)
    return True


# sources

def from_esc50(classes: dict, next_id: dict, have_sha: set, stats: Counter,
               skipped: Counter, room: dict[str, int], tmp: Path) -> list[dict]:
    rows: list[dict] = []
    import pandas as pd
    for parquet in sorted((AUDIO_DATASET / "raw_downloads").glob("esc50-train-*.parquet")):
        df = pd.read_parquet(parquet)
        for rec in df.itertuples(index=False):
            label = ESC50_MAP.get(str(rec.category))
            if label is None or room.get(label, 0) <= 0:
                skipped["unmapped:" + str(rec.category)] += 1
                continue
            if _ingest_blob(rows, ESC50_PROV, str(rec.category), label,
                            classes[label]["slug"], classes[label]["code"],
                            next_id[label], have_sha, stats, skipped,
                            rec.audio["bytes"],
                            {"corpus_id": f"esc50:{rec.filename}",
                             "original_filename": str(rec.filename)}, tmp, room):
                room[label] -= 1
    return rows


def from_us8k(classes: dict, next_id: dict, have_sha: set, stats: Counter,
              skipped: Counter, room: dict[str, int], tmp: Path) -> list[dict]:
    rows: list[dict] = []
    par_dir = REPO_ROOT / "data" / "uscraping" / "parquet"
    if not par_dir.exists():
        return rows
    import pandas as pd
    for parquet in sorted(par_dir.glob("train-*.parquet")):
        try:
            df = pd.read_parquet(parquet)
        except Exception:
            skipped["unreadable_parquet:" + parquet.name] += 1
            continue
        # itertuples mangles the `class` column, so rename it first.
        if "class" in df.columns:
            df = df.rename(columns={"class": "us8k_class"})
        for rec in df.itertuples(index=False):
            us8k_class = getattr(rec, "us8k_class", "")
            label = US8K_MAP.get(str(us8k_class))
            if label is None or room.get(label, 0) <= 0:
                skipped["unmapped:" + str(us8k_class)] += 1
                continue
            if _ingest_blob(rows, US8K_PROV, str(us8k_class), label,
                            classes[label]["slug"], classes[label]["code"],
                            next_id[label], have_sha, stats, skipped,
                            rec.audio["bytes"],
                            {"corpus_id": f"us8k:{rec.slice_file_name}",
                             "original_filename": str(rec.slice_file_name),
                             "salience": str(rec.salience)}, tmp, room):
                room[label] -= 1
    return rows


def _load_acquired(have_sha: set) -> list[dict]:
    """Load rows from earlier runs so they are kept, not overwritten."""
    if not OUT_MANIFEST.exists():
        return []
    with OUT_MANIFEST.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("sha256")]
    seen = set()
    keep: list[dict] = []
    for r in rows:
        if r["sha256"] in seen:
            continue
        seen.add(r["sha256"])
        have_sha.add(r["sha256"])
        keep.append(r)
    return keep


def _wipe_acquired(classes: dict) -> None:
    """Delete the clips this script wrote (id >= ID_START). FSD50K clips are left alone."""
    removed = 0
    for name, c in classes.items():
        d = AUDIO_DATASET / "originals" / c["slug"]
        if not d.exists():
            continue
        for wav in d.glob("*.wav"):
            stem = wav.stem  # SS-<CODE>-NNNN
            try:
                num = int(stem.rsplit("-", 1)[-1])
            except ValueError:
                continue
            if num >= ID_START:
                wav.unlink()
                removed += 1
    if OUT_MANIFEST.exists():
        OUT_MANIFEST.unlink()
    print(f"rebuild: removed {removed} previously acquired clips")


def from_fsd_eval(classes: dict, next_id: dict, have_sha: set, stats: Counter,
                  skipped: Counter, room: dict[str, int], tmp: Path) -> list[dict]:
    """FSD50K.eval clips from the partial archive, with their own labels."""
    rows: list[dict] = []
    labels_json = Path("/tmp/eval_labels.json")
    wav_dir = AUDIO_DATASET / "raw_downloads" / "fsd50k_eval_recovered"
    if not (labels_json.exists() and wav_dir.exists()):
        return rows
    labels = json.loads(labels_json.read_text(encoding="utf-8"))
    # label keys carry the .wav suffix, wavs do not
    labels = {k[:-4] if k.endswith(".wav") else k: v for k, v in labels.items()}
    for wav in sorted(wav_dir.glob("*.wav")):
        raw = labels.get(wav.stem)
        # Labels arrive as a comma-joined string of AudioSet terms with underscores.
        terms = [] if not raw else [t.replace("_", " ").strip() for t in str(raw).split(",") if t.strip()]
        if not terms:
            skipped["fsd_eval:no_label"] += 1
            continue
        # first allow-listed term wins
        label = None
        for term in terms:
            if term in FSD_EVAL_MAP and room.get(FSD_EVAL_MAP[term], 0) > 0:
                label = FSD_EVAL_MAP[term]
                break
        if label is None or room.get(label, 0) <= 0:
            skipped["fsd_eval:unmapped:" + ",".join(terms)[:40]] += 1
            continue
        blob = wav.read_bytes()
        if _ingest_blob(rows, FSD_EVAL_PROV, ",".join(terms), label,
                        classes[label]["slug"], classes[label]["code"],
                        next_id[label], have_sha, stats, skipped, blob,
                        {"corpus_id": f"fsd50k_eval:{wav.stem}",
                         "original_filename": wav.name}, tmp, room):
            room[label] -= 1
    return rows


FSD_DEV_PROV = {
    "source": "FSD50K.dev (Freesound field recordings)",
    "source_url": "https://zenodo.org/record/4060432",
    "licence": "CC-BY-4.0",
    "author": "Eduardo Fonseca et al.",
    "licence_url": "https://creativecommons.org/licenses/by/4.0/",
    "fetch_batch": "fsd50k_dev_topup_v1",
    "recording_environment": "unspecified",
    "environment_basis": "corpus_metadata:FSD50K Freesound field recordings",
}


def from_fsd_dev(classes: dict, next_id: dict, have_sha: set, stats: Counter,
                 skipped: Counter, room: dict[str, int], tmp: Path) -> list[dict]:
    """Download FSD50K.dev clips one at a time from the Hugging Face mirror.

    Only fetches clips for classes that still need more, and skips NC or unlicensed clips.
    """
    rows: list[dict] = []
    import urllib.request

    gt_path = AUDIO_DATASET / "raw_downloads" / "fsd50k_gt" / "FSD50K.ground_truth" / "dev.csv"
    info_path = AUDIO_DATASET / "raw_downloads" / "dev_clips_info_FSD50K.json"
    if not gt_path.exists():
        skipped["fsd_dev:no_ground_truth"] += 1
        return rows
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    allowed_licences = {
        "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
        "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
        "http://creativecommons.org/licenses/sampling+/1.0/": "Sampling+-1.0",
    }
    # skip clips we already have
    used_sources: set[str] = set()
    for src_csv in (AUDIO_DATASET / "manifests" / "fsd50k_real_rows.csv", OUT_MANIFEST):
        if src_csv.exists():
            with src_csv.open(newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    used_sources.add(r.get("original_filename", "").strip())

    # 1. build the candidate list: (fname, label) sorted for determinism
    candidates: dict[str, str] = {}
    with gt_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fname = row["fname"]
            terms = [t.strip().replace("_", " ") for t in row["labels"].split(",") if t.strip()]
            label = None
            for term in terms:
                if term in FSD_EVAL_MAP and room.get(FSD_EVAL_MAP[term], 0) > 0:
                    label = FSD_EVAL_MAP[term]
                    break
            if label is None:
                continue
            clip_info = info.get(fname, {})
            licence_url = clip_info.get("license", "")
            if licence_url not in allowed_licences:
                key = "fsd_dev:licence_refused:" + (licence_url or "missing")
                skipped[key] += 1
                continue
            if f"{fname}.wav" in used_sources:
                skipped["fsd_dev:already_used"] += 1
                continue
            candidates[fname] = label
            if sum(1 for l in candidates.values() if l == label) >= room[label]:
                # enough candidates for this class; stop collecting for it
                if all(sum(1 for l in candidates.values() if l == n) >= room[n]
                       for n in classes if room[n] > 0):
                    break

    # Interleave classes so an interrupted run spreads its gains.
    by_class: dict[str, list[tuple[str, str]]] = {}
    for fname, label in sorted(candidates.items()):
        by_class.setdefault(label, []).append(fname)
    plan: list[tuple[str, str]] = []
    remaining = {c: list(v) for c, v in by_class.items()}
    while any(remaining.values()):
        for label in sorted(remaining):
            if remaining[label]:
                plan.append((remaining[label].pop(0), label))

    # 2. fetch, verify, ingest
    base = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips/dev/{}.wav"
    for i, (fname, label) in enumerate(plan, start=1):
        if room[label] <= 0:
            continue
        url = base.format(fname)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                blob = resp.read()
        except Exception as exc:  # noqa: BLE001 - skip the clip
            skipped[f"fsd_dev:download_failed"] += 1
            print(f"  ! {fname}: {exc}")
            continue
        # must decode and not be a duplicate
        if _ingest_blob(rows, FSD_DEV_PROV, f"fsd50k_dev:{info.get(fname, {}).get('title', '')[:40]}",
                        label, classes[label]["slug"], classes[label]["code"],
                        next_id[label], have_sha, stats, skipped, blob,
                        {"corpus_id": f"fsd50k_dev:{fname}",
                         "original_filename": f"{fname}.wav"}, tmp, room):
            room[label] -= 1
        if i % 25 == 0:
            print(f"  fetched {i}/{len(plan)}; gained so far: {dict(stats)}")
    return rows



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=300,
                    help="originals per class to end on (SRS: 300)")
    ap.add_argument("--source", action="append", default=None,
                    choices=["esc50", "us8k", "fsd_eval", "fsd_dev"],
                    help="limit to these sources (repeatable); default: all")
    ap.add_argument("--rebuild", action="store_true",
                    help="delete every clip this script has ever written and "
                         "regenerate acquired_rows.csv from scratch "
                         "(deterministic: ids are stable)")
    args = ap.parse_args(argv)

    cfg = json.loads((REPO_ROOT / "config" / "classes.json").read_text(encoding="utf-8"))
    classes = {c["name"]: c for c in cfg["classes"]}
    for c in classes.values():
        c.setdefault("slug", c["name"].lower().replace(" ", "_"))

    if args.rebuild:
        _wipe_acquired(classes)

    # Count what is already on disk.
    existing: Counter = Counter()
    for name, c in classes.items():
        d = AUDIO_DATASET / "originals" / c["slug"]
        if d.exists():
            existing[name] = len(list(d.glob("*.wav")))
    # skip the empty folder left by an old naming bug
    stray = AUDIO_DATASET / "originals" / "alarm_siren"
    if stray.exists() and not any(stray.iterdir()):
        stray.rmdir()
        print(f"removed empty stray dir: {stray}")

    room = {name: max(0, args.per_class - existing[name]) for name in classes}
    next_id = {name: [ID_START] for name in classes}
    # Start above the highest id in use.
    if OUT_MANIFEST.exists():
        with OUT_MANIFEST.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cls, aid = r.get("class_label", ""), r.get("audio_id", "")
                if cls in next_id and aid.startswith("SS-"):
                    parts = aid.split("-")
                    if len(parts) == 3 and parts[2].isdigit():
                        next_id[cls][0] = max(next_id[cls][0], int(parts[2]) + 1)
    have_sha: set[str] = set()
    stats: Counter = Counter()
    skipped: Counter = Counter()

    # Known hashes from the manifest.
    for src_csv in (AUDIO_DATASET / "manifests" / "fsd50k_real_rows.csv", OUT_MANIFEST):
        if src_csv.exists():
            with src_csv.open(newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("sha256"):
                        have_sha.add(r["sha256"].strip())

    tmp = Path(tempfile.mkdtemp(prefix="ss_acquire_")) / "decode_tmp.wav"

    rows: list[dict] = _load_acquired(have_sha)
    gained_start = len(rows)
    sources = args.source or ["esc50", "us8k", "fsd_eval", "fsd_dev"]
    if "esc50" in sources:
        rows += from_esc50(classes, next_id, have_sha, stats, skipped, room, tmp)
    if "us8k" in sources:
        rows += from_us8k(classes, next_id, have_sha, stats, skipped, room, tmp)
    if "fsd_eval" in sources:
        rows += from_fsd_eval(classes, next_id, have_sha, stats, skipped, room, tmp)
    if "fsd_dev" in sources:
        rows += from_fsd_dev(classes, next_id, have_sha, stats, skipped, room, tmp)
    try:
        tmp.unlink()
        tmp.parent.rmdir()
    except OSError:
        pass

    rows.sort(key=lambda r: (r["class_label"], r["audio_id"]))
    # Older CSVs may have different columns; keep only the schema ones.
    rows = [{k: r.get(k, "") for k in ALL_COLUMNS} for r in rows]
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ALL_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"acquired {len(rows) - gained_start} new real clips "
          f"({len(rows)} total in {OUT_MANIFEST.name}) -> {OUT_MANIFEST}")
    print(f"\n  {'class':<24} {'have':>5} {'gained':>7} {'room left':>10}")
    for name in sorted(classes):
        gained = stats[name]
        mark = "  <-- still short" if room[name] > 0 else ""
        print(f"  {name:<24} {existing[name]:>5} {gained:>7} {room[name]:>10}{mark}")
    if skipped:
        print(f"\nskipped ({sum(skipped.values())}):")
        for k in sorted(skipped, key=lambda k: -skipped[k])[:20]:
            print(f"  {skipped[k]:>6}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
