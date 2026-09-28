"""Features for the ensemble: several embeddings side by side.

The bundle's feature version lists the backbones in order (``ensemble:ast-...+clap-...``),
so the columns always match what the saved model expects.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PREFIX = "ensemble:"


def backbone_module(backbone: str):
    if backbone == "ast":
        from feature_extraction import ast_embeddings as mod
    elif backbone == "clap":
        from feature_extraction import clap_embeddings as mod
    elif backbone == "cnn14":
        from feature_extraction import embeddings as mod
    else:
        raise KeyError(f"unknown backbone {backbone!r}")
    return mod


def _extractor(backbone: str):
    mod = backbone_module(backbone)
    if backbone == "ast":
        return mod.AstFeatureExtractor()
    if backbone == "clap":
        return mod.ClapFeatureExtractor()
    return mod.EmbeddingFeatureExtractor()


def embedding_version(backbones: list[str]) -> str:
    return PREFIX + "+".join(backbone_module(b).EMBEDDING_VERSION for b in backbones)


def backbones_from_version(version: str) -> list[str]:
    """Backbones named in a feature version; raises if one doesn't match this checkout."""
    known = {backbone_module(b).EMBEDDING_VERSION: b for b in ("ast", "clap", "cnn14")}
    parts = version[len(PREFIX):].split("+") if version.startswith(PREFIX) else []
    missing = [p for p in parts if p not in known]
    if not parts or missing:
        raise ValueError(f"cannot rebuild the extractor for {version!r}: unknown parts {missing}")
    return [known[p] for p in parts]


def embedding_columns(backbones: list[str]) -> list[str]:
    return [c for b in backbones for c in backbone_module(b).embedding_columns()]


class EnsembleFeatureExtractor:
    """Runs every backbone on the same audio and joins the results.

    With ``parallel`` the backbones run at the same time, which uses the cores better than
    one network after another (set SST_ENSEMBLE_PARALLEL=0 to switch it off).
    """

    def __init__(self, backbones: list[str], *, parallel: bool | None = None) -> None:
        import os

        self.backbones = list(backbones)
        self.feature_version = embedding_version(self.backbones)
        self._members = [_extractor(b) for b in self.backbones]
        self.parallel = (os.environ.get("SST_ENSEMBLE_PARALLEL", "1") != "0"
                         if parallel is None else parallel)
        self._pool = None

    def extract(self, preprocessed_or_samples: Any, sample_rate: int | None = None) -> np.ndarray:
        if not self.parallel or len(self._members) == 1:
            return np.concatenate([m.extract(preprocessed_or_samples, sample_rate)
                                   for m in self._members])
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(len(self._members), thread_name_prefix="embed")
        jobs = [self._pool.submit(m.extract, preprocessed_or_samples, sample_rate)
                for m in self._members]
        return np.concatenate([job.result() for job in jobs])
