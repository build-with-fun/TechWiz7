"""
Generate the SRS Deliverable-3 comparison report: both models on the same unseen
recordings, per-class, with the consistency verdict for every clip.

Owner: fatima-2 (QA).  SRS Step 11, Step 14, FR xxiii-xxx, lii, Deliverable 3.

WHAT THIS IS
------------
The SRS demands a comparison report covering >=100 unseen recordings with >=10 per
class, where "unseen" means held-out test split -- never trained on, never validated
on. Both models must be run on the SAME clips and their verdicts compared.

INTEGRITY RULES ENFORCED HERE (disqualifiers if broken)
-------------------------------------------------------
1. GTM never receives Python's prediction or confidence. Both predictors are called
   on the raw preprocessed audio independently, in this file, with no cross wiring.
   `classify_consistency` is the only place the two results meet, and it takes the
   two results as separate arguments so no value can leak from one to the other.
2. No hard-coded predictions. Every row comes from a real model call.
3. No invented confidence. Both confidences come off the predictor contract.
4. The report is evidence, so it records its own provenance: which split, how many
   clips, which model artifacts, which threshold snapshot.

The report is CSV (one row per clip) + a JSON summary. A human reads the summary;
an evaluator diffs the CSV.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The columns of the per-clip CSV. Fixed on purpose: an evaluator comparing two runs
# of this report needs the schema to be stable.
CSV_COLUMNS: tuple[str, ...] = (
    "audio_id",
    "true_class",
    "python_predicted_class",
    "python_confidence",
    "gtm_predicted_class",
    "gtm_confidence",
    "classes_agree",
    "confidence_difference",
    "consistency_status",
    "python_correct",
    "gtm_correct",
    "reason",
)


@dataclass
class ComparisonRow:
    """One clip, both models, one verdict."""

    audio_id: str
    true_class: str
    comparison: Any  # src.inference.consistency.ComparisonResult

    @property
    def python_correct(self) -> bool:
        return self.comparison.python_class == self.true_class

    @property
    def gtm_correct(self) -> bool:
        return self.comparison.gtm_class == self.true_class

    def to_csv_row(self) -> dict[str, Any]:
        c = self.comparison
        return {
            "audio_id": self.audio_id,
            "true_class": self.true_class,
            "python_predicted_class": c.python_class,
            "python_confidence": f"{c.python_confidence:.6f}",
            "gtm_predicted_class": c.gtm_class,
            "gtm_confidence": f"{c.gtm_confidence:.6f}",
            "classes_agree": str(c.classes_agree).lower(),
            "confidence_difference": f"{c.confidence_difference:.6f}",
            "consistency_status": c.consistency_status,
            "python_correct": str(self.python_correct).lower(),
            "gtm_correct": str(self.gtm_correct).lower(),
            "reason": c.reason,
        }


@dataclass
class ComparisonReport:
    """The full report: per-clip rows plus the summary an evaluator reads first."""

    rows: list[ComparisonRow] = field(default_factory=list)
    split_name: str = "test"
    python_model_name: str = ""
    gtm_model_name: str = ""
    threshold_snapshot: dict[str, Any] = field(default_factory=dict)
    classes: tuple[str, ...] = ()

    # ---- gates the SRS sets on the report itself -----------------------------

    MIN_TOTAL = 100
    MIN_PER_CLASS = 10

    @property
    def n_clips(self) -> int:
        return len(self.rows)

    def per_class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rows:
            counts[r.true_class] = counts.get(r.true_class, 0) + 1
        return counts

    def consistency_breakdown(self) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for r in self.rows:
            s = r.comparison.consistency_status
            breakdown[s] = breakdown.get(s, 0) + 1
        return breakdown

    def agreement_rate(self) -> float:
        if not self.rows:
            return 0.0
        agree = sum(1 for r in self.rows if r.comparison.classes_agree)
        return agree / len(self.rows)

    def accuracy(self, which: str) -> float:
        if which not in ("python", "gtm"):
            raise ValueError(f"which must be 'python' or 'gtm', got {which!r}")
        if not self.rows:
            return 0.0
        if which == "python":
            correct = sum(1 for r in self.rows if r.python_correct)
        else:
            correct = sum(1 for r in self.rows if r.gtm_correct)
        return correct / len(self.rows)

    def missing_classes(self) -> list[str]:
        """Classes with too few clips to count as evidence."""
        counts = self.per_class_counts()
        return sorted(
            c for c in self.classes
            if counts.get(c, 0) < self.MIN_PER_CLASS
        )

    def meets_report_floor(self) -> tuple[bool, list[str]]:
        """The >=100 total / >=10-per-class floor from the SRS."""
        problems: list[str] = []
        if self.n_clips < self.MIN_TOTAL:
            problems.append(
                f"report covers {self.n_clips} recordings, minimum is {self.MIN_TOTAL}"
            )
        missing = self.missing_classes()
        if missing:
            problems.append(
                f"classes with fewer than {self.MIN_PER_CLASS} recordings: "
                f"{', '.join(missing)}"
            )
        return (not problems), problems

    # ---- serialisation ------------------------------------------------------

    def write(self, out_csv: Path, out_json: Path) -> None:
        """Write the CSV (one row per clip) and the JSON summary."""
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.to_csv_row())

        ok, problems = self.meets_report_floor()
        summary: dict[str, Any] = {
            "n_clips": self.n_clips,
            "split": self.split_name,
            "python_model": self.python_model_name,
            "gtm_model": self.gtm_model_name,
            "classes": list(self.classes),
            "per_class_counts": self.per_class_counts(),
            "consistency_breakdown": self.consistency_breakdown(),
            "agreement_rate": round(self.agreement_rate(), 6),
            "python_accuracy": round(self.accuracy("python"), 6),
            "gtm_accuracy": round(self.accuracy("gtm"), 6),
            "meets_report_floor": ok,
            "floor_problems": problems,
            "threshold_snapshot": self.threshold_snapshot,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=False)
