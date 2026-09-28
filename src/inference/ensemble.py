"""The served Python model: one classifier per embedding (AST, CLAP, CNN14), scores averaged.

The feature vector is the embeddings side by side and each member reads its own slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class EnsembleMember:
    """A classifier and the columns it reads."""

    backbone: str
    estimator: Any
    start: int
    stop: int


class SoftVotingEnsemble:
    """Plain average of the members' probabilities."""

    def __init__(self, members: Sequence[EnsembleMember], classes: Sequence[str]) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = list(members)
        self.classes_ = np.asarray([str(c) for c in classes])
        self.n_features_in_ = max(m.stop for m in self.members)
        for member in self.members:
            fitted = [str(c) for c in getattr(member.estimator, "classes_", [])]
            if sorted(fitted) != sorted(self.classes_.tolist()):
                raise ValueError(f"{member.backbone} member was fitted on classes {fitted}")

    def member_proba(self, X: Any) -> dict[str, np.ndarray]:
        """Each member's probabilities in ``classes_`` order."""
        X = np.asarray(X, dtype=np.float64)
        out = {}
        for member in self.members:
            proba = np.asarray(member.estimator.predict_proba(X[:, member.start:member.stop]))
            order = [str(c) for c in member.estimator.classes_]
            out[member.backbone] = proba[:, [order.index(c) for c in self.classes_]]
        return out

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.mean(list(self.member_proba(X).values()), axis=0)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
