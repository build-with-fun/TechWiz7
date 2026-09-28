"""Download the pretrained networks used for the Python model's embeddings.

    python tools/fetch_pretrained.py            # PANNs CNN14, ~358 MB into ~/.cache/sonicsentinel/
    python tools/fetch_pretrained.py --ast      # AST (AudioSet), ~350 MB into the Hugging Face cache
    python tools/fetch_pretrained.py --clap     # CLAP (LAION), ~780 MB into the Hugging Face cache
    python tools/fetch_pretrained.py --all      # all three, which the served ensemble needs
    SST_PANNS_CHECKPOINT=/path/file.pth ...     # use a CNN14 copy somewhere else instead

CNN14: Zenodo record 3987831 (Kong et al., PANNs), CC BY 4.0, checked against the MD5
and SHA-256 in feature_extraction/embeddings.py. Zenodo is slow, so it downloads in
parallel, resumable chunks.

AST: MIT/ast-finetuned-audioset-10-10-0.4593 on Hugging Face (BSD-3-Clause), at the revision
pinned in feature_extraction/ast_embeddings.py.

CLAP: laion/larger_clap_general on Hugging Face (Apache-2.0), at the revision pinned in
feature_extraction/clap_embeddings.py.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_extraction.embeddings import (CHECKPOINT_MD5, CHECKPOINT_SHA256,  # noqa: E402
                                           CHECKPOINT_URL, default_checkpoint_path)

SIZE = 358_668_570
PARTS = 12


def fetch_range(url: str, start: int, end: int, target: Path, attempts: int = 8) -> None:
    for _ in range(attempts):
        have = target.stat().st_size if target.exists() else 0
        if start + have > end:
            return
        request = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, target.open("ab") as out:
                while chunk := response.read(1 << 16):
                    out.write(chunk)
        except OSError as exc:
            print(f"  part {target.name}: {exc}; retrying", flush=True)
    if (target.stat().st_size if target.exists() else 0) != end - start + 1:
        raise SystemExit(f"could not download {target.name} after {attempts} attempts")


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_ast() -> None:
    from feature_extraction.ast_embeddings import AST_MODEL, AST_REVISION, AstEmbedder

    AstEmbedder()  # downloads on first use
    print(f"AST {AST_MODEL}@{AST_REVISION[:8]} is in the Hugging Face cache")


def fetch_clap() -> None:
    from feature_extraction.clap_embeddings import CLAP_MODEL, CLAP_REVISION, ClapEmbedder

    ClapEmbedder()  # downloads on first use
    print(f"CLAP {CLAP_MODEL}@{CLAP_REVISION[:8]} is in the Hugging Face cache")


def main() -> None:
    flags = set(sys.argv[1:])
    if "--ast" in flags or "--all" in flags:
        fetch_ast()
    if "--clap" in flags or "--all" in flags:
        fetch_clap()
    if flags & {"--ast", "--clap"} and "--all" not in flags:
        return
    fetch_cnn14()


def fetch_cnn14() -> None:
    target = default_checkpoint_path()
    if target.exists() and digest(target, "sha256") == CHECKPOINT_SHA256:
        print(f"already present and verified: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = target.parent / (target.name + ".parts")
    parts_dir.mkdir(exist_ok=True)
    step = -(-SIZE // PARTS)
    ranges = [(i * step, min(SIZE, (i + 1) * step) - 1) for i in range(PARTS)]
    print(f"downloading {SIZE / 1e6:.0f} MB in {PARTS} parts to {target}", flush=True)
    with ThreadPoolExecutor(PARTS) as pool:
        list(pool.map(lambda r: fetch_range(CHECKPOINT_URL, r[0], r[1],
                                            parts_dir / f"{r[0]:012d}"), ranges))
    with target.open("wb") as out:
        for start, _ in ranges:
            out.write((parts_dir / f"{start:012d}").read_bytes())
    md5, sha = digest(target, "md5"), digest(target, "sha256")
    if md5 != CHECKPOINT_MD5 or sha != CHECKPOINT_SHA256:
        target.unlink()
        raise SystemExit(f"checksum mismatch (md5 {md5}); the download was deleted, run again")
    for part in parts_dir.iterdir():
        part.unlink()
    parts_dir.rmdir()
    print(f"verified md5 {md5} and sha256 {sha}")


if __name__ == "__main__":
    main()
