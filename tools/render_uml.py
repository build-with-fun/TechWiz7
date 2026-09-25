#!/usr/bin/env python3
"""Render the five SRS-mandated UML diagrams as PNGs into diagrams/.

Pure matplotlib (no plantuml/java dependency), deterministic layout. Diagrams:
DFD (context + level-1), use case, activity, sequence (upload flow), decision flow
(alert state machine).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse, Rectangle, Polygon

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "diagrams"
OUT.mkdir(exist_ok=True)

INK = "#1f2933"
ACCENT = "#0b5cad"
SOFT = "#eef3f8"
WARN = "#b34700"


def _save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def arrow(fig, ax, xy_from, xy_to, label: str = "", rad: float = 0.0, style: str = "-|>") -> None:
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style, mutation_scale=14,
                                 color=INK, lw=1.2,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        mx, my = (xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2
        ax.text(mx, my + 0.12, label, ha="center", va="bottom", fontsize=7.5, color=INK)


# ---------------------------------------------------------------- DFD context
def dfd() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Data Flow Diagram — context and level 1", fontsize=12, weight="bold")

    def ext(x, y, label):  # external entity: rectangle
        ax.add_patch(Rectangle((x - 1.0, y - 0.45), 2.0, 0.9, fc="white", ec=INK, lw=1.4))
        ax.text(x, y, label, ha="center", va="center", fontsize=9)

    def proc(x, y, label):  # process: rounded box
        ax.add_patch(Ellipse((x, y), 2.6, 1.15, fc=SOFT, ec=ACCENT, lw=1.6))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color=ACCENT, weight="bold")

    def store(x, y, label):  # data store: open-ended rectangle
        ax.add_patch(Rectangle((x - 1.3, y - 0.4), 2.6, 0.8, fc="white", ec=INK, lw=1.2))
        ax.plot([x - 1.3, x - 0.75], [y - 0.4, y - 0.4], color="white", lw=2)
        ax.text(x, y, label, ha="center", va="center", fontsize=8)

    ext(1.6, 6.8, "User")
    ext(1.6, 4.6, "Reviewer")
    ext(1.6, 2.2, "Security\nOperator")
    ext(10.4, 6.8, "Admin")

    proc(5.2, 6.8, "P1 Audio Ingestion\n& Preprocessing")
    proc(8.3, 5.6, "P2 Dual-Model\nInference")
    proc(5.2, 3.4, "P3 Comparison\n& Consistency")
    proc(8.3, 2.0, "P4 Alerting &\nEscalation")
    proc(2.9, 5.2, "P5 Manual\nReview")

    store(6.9, 6.9, "D1 Audio Files")
    store(11.0, 3.2, "D2 Events")
    store(3.4, 1.0, "D3 Alerts")
    store(8.0, 7.6, "D4 Model Bundles\n(Python + GTM)")

    arrow(fig, ax, (2.6, 6.8), (4.2, 6.9), "clip upload")
    arrow(fig, ax, (6.2, 6.6), (7.4, 5.6), "clean samples")
    arrow(fig, ax, (9.3, 4.9), (6.2, 3.7), "both predictions")
    arrow(fig, ax, (4.3, 3.3), (2.4, 4.2), "verdicts")
    arrow(fig, ax, (6.2, 3.1), (7.3, 2.2), "event + consistency")
    arrow(fig, ax, (8.3, 2.6), (2.6, 4.7), "critical event", rad=-0.25)
    arrow(fig, ax, (2.6, 2.2), (7.3, 2.0), "ack / dismiss / escalate")
    arrow(fig, ax, (9.4, 6.8), (9.0, 3.3), "retention & config", rad=-0.25)
    arrow(fig, ax, (6.9, 6.6), (5.6, 6.9), "", rad=0.1, style="<|-|>")
    arrow(fig, ax, (11.0, 3.6), (9.6, 4.6), "", rad=0.1, style="<|-|>")
    _save(fig, "dfd.png")


# ------------------------------------------------------------ use case
def use_case() -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_title("Use Case Diagram", fontsize=12, weight="bold")

    actors = {"Normal User": 1.4, "Audio Reviewer": 1.4, "Security Operator": 1.4, "Administrator": 1.4}
    ys = {"Normal User": 7.6, "Audio Reviewer": 5.6, "Security Operator": 3.4, "Administrator": 1.2}

    cases = {
        "Upload Audio Clip": (5.6, 7.9),
        "View Prediction & Spectrogram": (6.0, 6.9),
        "View Model Comparison": (6.2, 6.0),
        "Decide Manual Review": (6.0, 5.1),
        "Acknowledge / Dismiss Alert": (6.3, 4.1),
        "Escalate Alert": (6.1, 3.3),
        "Live Microphone Monitoring": (6.5, 2.4),
        "Manage Users & Config": (8.9, 1.4),
        "Run Retention Purge": (9.0, 0.6),
        "Export Reports (CSV/XLSX)": (9.3, 6.8),
    }

    ax.add_patch(Rectangle((4.4, 0.05), 6.6, 8.5, fc="none", ec=INK, ls="--", lw=1.2))
    for name, (x, y) in cases.items():
        ax.add_patch(Ellipse((x, y), 3.6 if x < 8 else 2.9, 0.78, fc=SOFT, ec=ACCENT, lw=1.3))
        ax.text(x, y, name, ha="center", va="center", fontsize=7.6, color=ACCENT)

    ax.add_patch(Circle((1.4, 7.6), 0.22, fc="white", ec=INK)); ax.plot([1.4, 1.4], [7.38, 6.85], color=INK)
    ax.plot([1.05, 1.75], [7.05, 7.05], color=INK); ax.plot([1.4, 1.4], [7.05, 6.7], color=INK)
    ax.plot([1.4, 1.15], [6.7, 6.3], color=INK); ax.plot([1.4, 1.65], [6.7, 6.7], color=INK)
    ax.text(1.4, 6.35, "Normal User", ha="center", fontsize=8)

    links = {
        "Normal User": ["Upload Audio Clip", "View Prediction & Spectrogram",
                        "View Model Comparison", "Live Microphone Monitoring"],
        "Audio Reviewer": ["Decide Manual Review"],
        "Security Operator": ["Acknowledge / Dismiss Alert", "Escalate Alert"],
        "Administrator": ["Manage Users & Config", "Run Retention Purge", "Export Reports (CSV/XLSX)"],
    }
    for a, ys_ in links.items():
        y = ys.get(a, 1.2)
        for c in ys_:
            tgt = cases.get(c)
            if tgt is None:
                tgt = (8.9, 1.4) if c.startswith("Manage") else (9.0, 6.0)
            arrow(fig, ax, (1.75, y), (tgt[0] - 1.6, tgt[1]), rad=0.06)
    _save(fig, "use_case.png")


# ------------------------------------------------------------ activity
def activity() -> None:
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.set_xlim(0, 10); ax.set_ylim(0, 16); ax.axis("off")
    ax.set_title("Activity Diagram — clip evaluation end to end", fontsize=12, weight="bold")

    def node(x, y, w, h, label, kind="action"):
        if kind == "start":
            ax.add_patch(Circle((x, y), 0.18, fc=INK, ec=INK))
        elif kind == "end":
            ax.add_patch(Circle((x, y), 0.22, fc="none", ec=INK, lw=1.6))
            ax.add_patch(Circle((x, y), 0.13, fc=INK, ec=INK))
        elif kind == "decision":
            ax.add_patch(Polygon([(x, y + 0.55), (x + 1.05, y), (x, y - 0.55), (x - 1.05, y)],
                                 fc="#fff7e6", ec=WARN, lw=1.4))
            ax.text(x, y, label, ha="center", va="center", fontsize=7.4)
        else:
            ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=SOFT, ec=ACCENT, lw=1.3, joinstyle="round"))
            ax.text(x, y, label, ha="center", va="center", fontsize=8)

    X = 5
    node(X, 15.3, 0, 0, "", "start")
    seq = [
        (14.4, "Receive upload / live window"),
        (13.5, "Validate content (magic bytes,\nduration, sample rate)"),
        (12.6, "Preprocess: decode, resample\nto 16 kHz mono"),
        (11.6, "Audio quality verdict\n(Good / Acceptable / Poor / Unusable)"),
        (10.5, "Extract features\n(audiofeat-1.0.0)"),
    ]
    for y, label in seq:
        node(X, y, 4.6, 1.0, label)
        arrow(fig, ax, (X, y + 1.1), (X, y + 0.62))

    node(X, 9.3, 0, 0, "quality\nusable?", "decision")
    arrow(fig, ax, (X, 10.0), (X, 9.85))
    node(X + 3.4, 9.3, 2.9, 0.95, "Quarantine clip;\nrecord unusable event")
    arrow(fig, ax, (X + 1.05, 9.3), (X + 1.95, 9.3), "no")
    arrow(fig, ax, (X, 8.75), (X, 8.35), "yes")

    node(X, 7.8, 4.4, 1.05, "Python model inference\n(class + confidences)")
    arrow(fig, ax, (X, 8.35), (X, 8.32))
    node(X, 6.6, 4.4, 1.05, "GTM model inference\n(independent frontend)")
    arrow(fig, ax, (X, 7.27), (X, 7.12))
    node(X, 5.4, 4.6, 1.05, "Compare: class match,\n|conf diff|, margins")
    arrow(fig, ax, (X, 6.07), (X, 5.92))

    node(X, 4.2, 0, 0, "critical class\n& rules met?", "decision")
    arrow(fig, ax, (X, 4.87), (X, 4.75))
    node(X - 3.1, 4.2, 2.8, 0.95, "Raise alert\n(Open state)")
    node(X + 3.1, 4.2, 3.0, 0.95, "Record event only;\nqueue manual review")
    arrow(fig, ax, (X - 1.05, 4.2), (X - 1.7, 4.2), "yes")
    arrow(fig, ax, (X + 1.05, 4.2), (X + 1.6, 4.2), "no")
    arrow(fig, ax, (X - 3.1, 3.72), (X - 3.1, 3.1))
    arrow(fig, ax, (X + 3.1, 3.72), (X, 3.3), rad=0.1)

    node(X, 2.9, 4.0, 1.0, "Persist event, comparison\nand audit trail")
    node(X, 1.9, 4.2, 0.95, "Notify / display in dashboard")
    arrow(fig, ax, (X, 2.4), (X, 2.37))
    node(X, 0.7, 0, 0, "", "end")
    arrow(fig, ax, (X, 1.42), (X, 0.92))
    _save(fig, "activity.png")


# ------------------------------------------------------------ sequence
def sequence() -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 11); ax.axis("off")
    ax.set_title("Sequence Diagram — upload → dual inference → comparison → alert", fontsize=12, weight="bold")

    lanes = {"User": 1.5, "Web App": 4.3, "Preproc\nService": 6.9, "Python\nModel": 9.2, "GTM\nModel": 11.4, "DB +\nAlerts": 13.0}
    key = lambda s: {"Preproc Service": "Preproc\nService", "Python Model": "Python\nModel",
                     "GTM Model": "GTM\nModel", "DB + Alerts": "DB +\nAlerts"}.get(s, s)
    for name, x in lanes.items():
        ax.add_patch(Rectangle((x - 0.9, 10.1), 1.8, 0.5, fc=SOFT, ec=ACCENT, lw=1.2))
        ax.text(x, 10.35, name, ha="center", va="center", fontsize=8, color=ACCENT, weight="bold")
        ax.plot([x, x], [0.4, 10.1], color="#9aa5b1", lw=1, ls=":")

    msgs = [
        ("User", "Web App", 9.6, "1. POST /api/audio (clip)"),
        ("Web App", "Preproc Service", 8.9, "2. decode + resample 16k"),
        ("Preproc Service", "Web App", 8.8, "3. samples + quality verdict"),
        ("Web App", "Python Model", 7.9, "4. predict_proba(features)"),
        ("Python Model", "Web App", 7.1, "5. class + per-class confidence"),
        ("Web App", "GTM Model", 6.3, "6. predict(samples)  [no python output]"),
        ("GTM Model", "Web App", 5.5, "7. gtm_class + confidences"),
        ("Web App", "Web App", 4.7, "8. consistency verdict"),
        ("Web App", "DB + Alerts", 3.7, "9. persist event + comparison"),
        ("DB + Alerts", "Web App", 2.9, "10. alert_id (if critical)"),
        ("Web App", "User", 1.9, "11. JSON result / redirect"),
    ]
    for src, dst, y, label in msgs:
        xs = lanes[key(src)]; xd = lanes[key(dst)]
        if src == dst:
            ax.annotate("", xy=(xs + 0.55, y), xytext=(xs, y),
                        arrowprops=dict(arrowstyle="-|>", color=INK))
            ax.text(xs + 0.7, y, label, fontsize=7.4, va="center")
        else:
            arrow(fig, ax, (xs, y), (xd, y))
            ax.text((xs + xd) / 2, y + 0.12, label, ha="center", fontsize=7.4)
    _save(fig, "sequence_upload.png")


# ------------------------------------------------------------ decision flow
def decision_flow() -> None:
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Decision Flow — alert lifecycle & comparison verdicts", fontsize=12, weight="bold")

    def dnode(x, y, label, kind="state"):
        if kind == "state":
            ax.add_patch(Rectangle((x - 1.35, y - 0.5), 2.7, 1.0, fc=SOFT, ec=ACCENT, lw=1.4))
            ax.text(x, y, label, ha="center", va="center", fontsize=8.4, color=ACCENT, weight="bold")
        else:
            ax.add_patch(Polygon([(x, y + 0.6), (x + 1.15, y), (x, y - 0.6), (x - 1.15, y)],
                                 fc="#fff7e6", ec=WARN, lw=1.4))
            ax.text(x, y, label, ha="center", va="center", fontsize=7.6)

    dnode(2.2, 8.6, "Event recorded\n(from upload/live)")
    dnode(2.2, 6.4, "Alert Open", "state")
    dnode(6.6, 9.0, "Acknowledged", "state")
    dnode(6.6, 7.2, "Dismissed\n(reason required)", "state")
    dnode(2.2, 5.2, "Escalated", "state")
    dnode(6.8, 4.6, "Closed", "state")
    dnode(10.8, 8.6, "Manual Review\nQueued", "state")
    dnode(10.8, 6.4, "Confirmed /\nOverridden", "state")

    arrow(fig, ax, (3.55, 8.5), (5.25, 8.95), "acknowledge", rad=-0.08)
    arrow(fig, ax, (3.55, 8.35), (5.3, 7.35), "dismiss", rad=0.08)
    arrow(fig, ax, (2.2, 8.1), (2.2, 5.7), "escalate")
    arrow(fig, ax, (3.0, 4.9), (5.6, 4.75), "escalate dismissed→closed", rad=0.05)
    arrow(fig, ax, (6.6, 8.6), (6.8, 5.1), "timeout / closure", rad=-0.1)
    arrow(fig, ax, (7.95, 7.2), (9.6, 8.2), "model disagreement\nor manual rule", rad=0.15)
    arrow(fig, ax, (10.8, 8.1), (10.8, 7.0), "reviewer decides")

    dnode(10.8, 4.2, "Strong /\nAcceptable / Weak\nMatch", "state")
    dnode(10.8, 2.2, "Model Disagreement /\nUncertain Result", "state")
    arrow(fig, ax, (10.8, 5.9), (10.8, 4.8), "class agrees,\nsmall conf diff")
    arrow(fig, ax, (11.9, 5.6), (11.3, 2.9), "class differs or\nlarge conf diff", rad=0.12)
    arrow(fig, ax, (9.6, 4.0), (3.5, 8.35), "auto-acknowledge path", rad=0.35, style="-|>")
    _save(fig, "decision_flow.png")


if __name__ == "__main__":
    dfd(); use_case(); activity(); sequence(); decision_flow()
    print("all diagrams rendered to", OUT)