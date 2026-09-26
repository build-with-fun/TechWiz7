"""SRS deliverable 6: the Python vs Teachable Machine comparison report on unseen audio.

    python tools/build_comparison_report.py                 # all 450 test recordings
    python tools/build_comparison_report.py --per-class 10  # the SRS minimum (100 clips)

Every row comes from the same ``AnalysisPipeline.analyse_file`` call the upload route
makes, with persistence switched off. So quality, severity, alert status, review
routing and the final decision are the product's real output, not a re-implementation.
Rows are not hand-edited. The "explanation" column is generated from the two score
distributions and the ground truth, so it can only restate what the models did.

The repeat tracker is reset before every clip. The test recordings are independent
events, and letting three unrelated gunshot files "confirm" each other would overstate
how often an alert is raised.

Outputs (in ``reports/``):
    model_comparison.csv            one row per clip, SRS column list
    model_comparison_summary.json   the headline numbers + provenance
    MODEL_COMPARISON.md             human-readable summary with the worst disagreements
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def test_rows(per_class: int | None) -> list[dict[str, str]]:
    with (ROOT / "audio_dataset" / "manifest_with_split.csv").open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["dataset_split"] == "test" and r["original_or_augmented"] == "original"]
    rows.sort(key=lambda r: (r["class_label"], r["audio_id"]))
    if not per_class:
        return rows
    kept, taken = [], Counter()
    for r in rows:
        if taken[r["class_label"]] < per_class:
            taken[r["class_label"]] += 1
            kept.append(r)
    return kept


def explain(actual: str, py: dict, gtm: dict, agree: bool) -> str:
    """One factual sentence; empty when the models agree and are right."""
    py_cls, gtm_cls = py["predicted_class"], gtm["predicted_class"]
    if agree and py_cls == actual:
        return ""
    if agree:
        return (f"Both models chose {py_cls} (true class {actual}); the true class was "
                f"Python's #{rank(py, actual)} and GTM's #{rank(gtm, actual)} choice.")
    right = [name for name, cls in (("Python", py_cls), ("GTM", gtm_cls)) if cls == actual]
    who = " and ".join(right) + " correct" if right else "neither model correct"
    return (f"Python {py_cls} ({py['confidence']:.2f}) vs GTM {gtm_cls} "
            f"({gtm['confidence']:.2f}); {who}. True class {actual} ranked "
            f"#{rank(py, actual)} by Python and #{rank(gtm, actual)} by GTM.")


def rank(pred: dict, cls: str) -> int:
    ordered = sorted(pred["confidences"].items(), key=lambda kv: -kv[1])
    return 1 + [name for name, _ in ordered].index(cls)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=0,
                        help="clips per class (0 = every test recording)")
    args = parser.parse_args()

    from src.services.pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline.load()
    classes = pipeline.store.class_names()
    rows = test_rows(args.per_class or None)

    header = (["audio_id", "filename", "actual_class", "python_predicted_class",
               "python_confidence"]
              + [f"python_p_{slug(c)}" for c in classes]
              + ["gtm_predicted_class", "gtm_confidence"]
              + [f"gtm_p_{slug(c)}" for c in classes]
              + ["class_match", "consistency_status", "top_class_confidence_difference",
                 "python_top_two_margin", "gtm_top_two_margin", "overlap_flag",
                 "audio_quality", "severity", "alert_status", "manual_review",
                 "review_reasons", "final_class", "final_decision", "python_correct",
                 "gtm_correct", "final_result", "explanation", "python_ms", "gtm_ms",
                 "total_ms"])
    out_rows, failures = [], []
    started = time.time()
    for n, row in enumerate(rows, 1):
        pipeline.tracker.reset()
        path = ROOT / "audio_dataset" / row["filename"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = pipeline.analyse_file(path, audio_id=row["audio_id"])
        if not record.get("ok"):
            failures.append({"audio_id": row["audio_id"],
                             "reason": (record.get("rejection") or record.get("error") or {})})
            continue
        py, gtm = record["predictions"]["python"], record["predictions"]["gtm"]
        cmp_ = record["comparison"]
        actual = row["class_label"]
        final_class = record["decision"]["final_class"]
        review = record["review"]
        if review["required"]:
            final_result = "sent to review"
        else:
            final_result = "correct" if final_class == actual else "incorrect"
        timings = record.get("timings_ms", {})
        out_rows.append(
            [row["audio_id"], Path(row["filename"]).name, actual, py["predicted_class"],
             f"{py['confidence']:.4f}"]
            + [f"{py['confidences'].get(c, 0.0):.4f}" for c in classes]
            + [gtm["predicted_class"], f"{gtm['confidence']:.4f}"]
            + [f"{gtm['confidences'].get(c, 0.0):.4f}" for c in classes]
            + ["match" if cmp_["classes_agree"] else "mismatch", cmp_["consistency_status"],
               f"{cmp_['confidence_difference']:.4f}",
               f"{cmp_['python']['top_two_margin']:.4f}", f"{cmp_['gtm']['top_two_margin']:.4f}",
               "yes" if cmp_.get("overlapping") else "no",
               record["quality"]["verdict"], record["severity"]["severity_display"],
               "raised" if record["alert"]["raised"] else
               ("eligible" if record["alert"]["eligible"] else "none"),
               "required" if review["required"] else "not required",
               "; ".join(review.get("matched", [])), final_class,
               record["decision"]["final_decision"],
               py["predicted_class"] == actual, gtm["predicted_class"] == actual, final_result,
               explain(actual, py, gtm, cmp_["classes_agree"]),
               timings.get("python_model", ""), timings.get("gtm_model", ""),
               timings.get("total", "")])
        if n % 50 == 0:
            print(f"  {n}/{len(rows)} clips, {time.time() - started:.0f}s", flush=True)

    REPORTS.mkdir(exist_ok=True)
    with (REPORTS / "model_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(out_rows)

    col = {name: i for i, name in enumerate(header)}
    n = len(out_rows)
    per_class = defaultdict(lambda: Counter())
    for r in out_rows:
        c = per_class[r[col["actual_class"]]]
        c["n"] += 1
        c["python"] += r[col["python_correct"]] is True
        c["gtm"] += r[col["gtm_correct"]] is True
        c["agree"] += r[col["class_match"]] == "match"
        c["review"] += r[col["manual_review"]] == "required"
    agree = [r for r in out_rows if r[col["class_match"]] == "match"]
    auto = [r for r in out_rows if r[col["manual_review"]] != "required"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": "test", "clips_scored": n, "clips_failed": len(failures),
        "failures": failures,
        "models": record_models(pipeline),
        "python_accuracy": sum(r[col["python_correct"]] is True for r in out_rows) / n,
        "gtm_accuracy": sum(r[col["gtm_correct"]] is True for r in out_rows) / n,
        "agreement_rate": len(agree) / n,
        "accuracy_when_models_agree": (sum(r[col["python_correct"]] is True for r in agree)
                                       / len(agree)) if agree else None,
        "sent_to_manual_review": n - len(auto),
        "auto_decided": len(auto),
        "accuracy_of_auto_decisions": (sum(r[col["final_class"]] == r[col["actual_class"]]
                                           for r in auto) / len(auto)) if auto else None,
        "alerts_raised": sum(r[col["alert_status"]] == "raised" for r in out_rows),
        "consistency_status_counts": dict(Counter(r[col["consistency_status"]] for r in out_rows)),
        "per_class": {k: dict(v) for k, v in sorted(per_class.items())},
        "median_total_ms": sorted(r[col["total_ms"]] for r in out_rows)[n // 2],
    }
    (REPORTS / "model_comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, out_rows, col)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("per_class", "failures", "models")}, indent=2))


def record_models(pipeline) -> dict:
    d = pipeline.models.describe() if hasattr(pipeline.models, "describe") else {}
    return {"python": {k: d.get("python", {}).get(k) for k in ("model_name", "model_version")},
            "gtm": {k: d.get("gtm", {}).get(k) for k in ("model_name", "model_version")}}


def write_markdown(summary: dict, rows: list, col: dict) -> None:
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    lines = [
        "# Model comparison on unseen test recordings",
        "",
        f"Generated {summary['generated_at'][:19]} UTC by `tools/build_comparison_report.py` "
        f"from {summary['clips_scored']} frozen test recordings "
        f"({summary['clips_failed']} failed to analyse). Full rows: `reports/model_comparison.csv`.",
        "",
        f"- Python model: `{summary['models']['python']['model_name']}` "
        f"v{summary['models']['python']['model_version']}",
        f"- Teachable Machine model: v{summary['models']['gtm']['model_version']}",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Python top-1 accuracy | {pct(summary['python_accuracy'])} |",
        f"| GTM top-1 accuracy | {pct(summary['gtm_accuracy'])} |",
        f"| Models agree on the class | {pct(summary['agreement_rate'])} |",
        f"| Accuracy when they agree | {pct(summary['accuracy_when_models_agree'])} |",
        f"| Sent to manual review | {summary['sent_to_manual_review']} |",
        f"| Decided automatically | {summary['auto_decided']} |",
        f"| Accuracy of automatic decisions | {pct(summary['accuracy_of_auto_decisions'])} |",
        f"| Alerts raised (tracker reset per clip) | {summary['alerts_raised']} |",
        f"| Median end-to-end time per clip | {summary['median_total_ms']} ms |",
        "",
        "## Per class",
        "",
        "| Class | Clips | Python correct | GTM correct | Agree | Sent to review |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, c in summary["per_class"].items():
        lines.append(f"| {name} | {c['n']} | {c.get('python', 0)} | {c.get('gtm', 0)} | "
                     f"{c.get('agree', 0)} | {c.get('review', 0)} |")
    lines += ["", "## Consistency status counts", ""]
    for status, count in sorted(summary["consistency_status_counts"].items()):
        lines.append(f"- {status}: {count}")
    worst = sorted((r for r in rows if r[col["class_match"]] == "mismatch"),
                   key=lambda r: -float(r[col["top_class_confidence_difference"]]))[:15]
    lines += ["", "## Largest disagreements", "",
              "Sorted by the absolute top-class confidence difference.", "",
              "| Audio ID | True class | Python | GTM | Difference | Explanation |",
              "|---|---|---|---|---:|---|"]
    for r in worst:
        lines.append(f"| {r[col['audio_id']]} | {r[col['actual_class']]} | "
                     f"{r[col['python_predicted_class']]} {r[col['python_confidence']]} | "
                     f"{r[col['gtm_predicted_class']]} {r[col['gtm_confidence']]} | "
                     f"{r[col['top_class_confidence_difference']]} | {r[col['explanation']]} |")
    (REPORTS / "MODEL_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
