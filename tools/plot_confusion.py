"""Draw a test-set confusion matrix as a PNG for the report.

    python tools/plot_confusion.py python_models/metrics/transfer_test_predictions_current.csv \
        python_models/metrics/confusion_matrix_transfer_test.png --title "CNN14 + MLP"
    python tools/plot_confusion.py gtm_model/gtm_metrics.json gtm_model/confusion_matrix_test.png \
        --title "Teachable Machine"

Input is either the per-clip predictions CSV written by train_transfer.py (columns
``actual`` and ``predicted``) or a metrics JSON with ``labels`` and ``confusion_matrix``.
Rows are the true class and columns the prediction, the same convention as
src/training/evaluation.py, and each cell shows the count out of the row total.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# One-hue sequential ramp (light -> dark blue); zero recedes to near-white.
RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load(path: Path) -> tuple[list[str], np.ndarray]:
    if path.suffix == ".json":
        doc = json.loads(path.read_text())
        return list(doc["labels"]), np.asarray(doc["confusion_matrix"], dtype=int)
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    labels = sorted({r["actual"] for r in rows})
    index = {c: i for i, c in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for r in rows:
        matrix[index[r["actual"]], index[r["predicted"]]] += 1
    return labels, matrix


def plot(labels: list[str], matrix: np.ndarray, title: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)
    share = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    accuracy = np.trace(matrix) / max(matrix.sum(), 1)

    fig, ax = plt.subplots(figsize=(8.6, 7.2), dpi=130)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{title}\n{matrix.sum()} test recordings, accuracy {accuracy:.3f}",
                 fontsize=10.5)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j]:
                ink = "white" if share[i, j] > 0.5 else "#1f2328"
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8, color=ink)
    # Recessive frame: the cells carry the information, not the box around them.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--title", default="Confusion matrix")
    args = parser.parse_args()
    labels, matrix = load(args.source)
    plot(labels, matrix, args.title, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
