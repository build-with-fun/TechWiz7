#!/usr/bin/env python3
"""Render the five SRS-mandated UML diagrams as PNGs into diagrams/.

Every diagram is laid out on a fixed grid with straight arrows; ``label_spot`` keeps edge
labels off the boxes. ``--check`` verifies structurally that no two boxes overlap, every
edge stays clear of the boxes it does not connect, and all five PNGs exist.

Pure matplotlib, no plantuml/java/graphviz binary required.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "diagrams"

INK = "#1f2933"
ACCENT = "#0b5cad"
SOFT = "#eef3f8"
WARN = "#b34700"
STORE_BG = "#f7f3ea"
ACTOR_LINE = "#42526b"
EDGES = []   # routed polylines of the current diagram, for --check


# ------------------------------------------------------------------ geometry
@dataclass
class Box:
    """Axis-aligned node with named ports used for edge attachment."""
    cx: float
    cy: float
    w: float
    h: float
    name: str = ""

    @property
    def x0(self):
        return self.cx - self.w / 2

    @property
    def x1(self):
        return self.cx + self.w / 2

    @property
    def y0(self):
        return self.cy - self.h / 2

    @property
    def y1(self):
        return self.cy + self.h / 2

    def port(self, side: str):
        return {"E": (self.x1, self.cy), "W": (self.x0, self.cy),
                "N": (self.cx, self.y1), "S": (self.cx, self.y0)}[side]

    def facing(self, other: "Box"):
        """Port of this box that best faces another box's centre."""
        dx, dy = other.cx - self.cx, other.cy - self.cy
        return "E" if dx >= 0 else "W" if abs(dx) > abs(dy) else ("N" if dy >= 0 else "S")

    def contains(self, x: float, y: float, pad: float = 0.0) -> bool:
        return (self.x0 - pad <= x <= self.x1 + pad
                and self.y0 - pad <= y <= self.y1 + pad)


def overlap(a: Box, b: Box, pad: float = 0.0) -> bool:
    return not (a.x1 + pad <= b.x0 or b.x1 + pad <= a.x0
                or a.y1 + pad <= b.y0 or b.y1 + pad <= a.y0)


def seg_clear_box(p, q, b, pad=0.02):
    """True if segment p-q stays outside box b (Liang-Barsky on padded rect)."""
    x0, x1, y0, y1 = b.x0 - pad, b.x1 + pad, b.y0 - pad, b.y1 + pad
    dx, dy = q[0] - p[0], q[1] - p[1]
    tmin, tmax = 0.0, 1.0
    for p_val, d, lo, hi in ((p[0], dx, x0, x1), (p[1], dy, y0, y1)):
        if abs(d) < 1e-12:
            if p_val < lo or p_val > hi:
                return True
        else:
            t1, t2 = (lo - p_val) / d, (hi - p_val) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin >= tmax:
                return True
    return False


def path_clear(pts, boxes, pad=0.02):
    """Every segment of a routed path must miss every box."""
    for b in boxes:
        for p, q in zip(pts, pts[1:]):
            if not seg_clear_box(p, q, b, pad):
                return False
    return True


def label_spot(pts, boxes, w_label=1.9, h_label=0.34):
    """Find a label anchor that does not overlap any node box."""
    for k in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy) or 1.0
        ox, oy = -dy / L * 0.17, dx / L * 0.17
        for px, py in ((mx + ox, my + oy), (mx - ox, my - oy),
                       (mx, my + 0.15), (mx, my - 0.15)):
            cand = Box(px, py, w_label, h_label)
            if not any(overlap(cand, b, pad=0.04) for b in boxes):
                return (px, py)
    cx = (pts[0][0] + pts[-1][0]) / 2
    return (cx, (pts[0][1] + pts[-1][1]) / 2 + 0.18)


# ------------------------------------------------------------------ drawing
def edge(ax, pts, label="", boxes=(), color=INK, lw=1.2, rad=0.0, fs=7.6,
        dashed=False, arrow="-|>"):
    """Draw a routed orthogonal path with an arrowhead and safe label."""
    if len(pts) >= 2:
        EDGES.append(list(pts))
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=color, lw=lw, solid_capstyle="round", zorder=2,
                ls=(0, (4, 3)) if dashed else "-")
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle=arrow,
                                     mutation_scale=13, color=color, lw=lw, zorder=3))
    if label:
        spot = label_spot(pts, boxes) if len(pts) >= 2 else ((pts[0][0], pts[0][1] + 0.15))
        ax.text(spot[0], spot[1], label, ha="center", va="center", fontsize=fs,
                color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))


def straight(ax, a, b, label="", boxes=(), color=INK, lw=1.2, fs=7.6):
    edge(ax, [a, b], label, boxes, color, lw, fs=fs)


def title(ax, text, sub=""):
    ax.set_title(text, fontsize=12.5, weight="bold", color=INK, pad=24)
    if sub:
        ax.text(0.5, 1.005, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.6, color="#7b8794")


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT / name}")


# --------------------------------------------------------------------- DFD
def dfd():
    """Level-1 data flow: entities left, processes in the middle, stores right.

    Redrawn on 26 Sep with a fixed grid so that every flow is a straight horizontal or
    vertical arrow; the routed version overlapped labels and crossed its own lines.
    """
    fig, ax = plt.subplots(figsize=(14.5, 11.5))
    W, H = 17.0, 13.2
    ax.set_xlim(0, W), ax.set_ylim(0.6, H)
    ax.axis("off")
    title(ax, "Data Flow Diagram — level 1",
          "Rectangles: people outside the system · blue: processes · beige: data stores")

    def entity(cx, cy, label):
        b = Box(cx, cy, 2.9, 1.0, label)
        ax.add_patch(Rectangle((b.x0, b.y0), b.w, b.h, fc="white", ec=INK, lw=2.0, zorder=4))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=9, zorder=5)
        return b

    def process(cx, cy, label):
        b = Box(cx, cy, 4.4, 1.15, label)
        ax.add_patch(FancyBboxRaw(b, SOFT, ACCENT))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=8.8, color=ACCENT,
                weight="bold", zorder=5)
        return b

    def store(cx, cy, key, label, h=1.0):
        b = Box(cx, cy, 3.6, h, label)
        ax.add_patch(Rectangle((b.x0, b.y0), b.w, b.h, fc=STORE_BG, ec=INK, lw=1.3, zorder=4))
        ax.plot([b.x0 + 0.55, b.x0 + 0.55], [b.y0, b.y1], color=INK, lw=1.0, zorder=5)
        ax.text(b.x0 + 0.27, cy, key, ha="center", va="center", fontsize=8.5, weight="bold", zorder=5)
        ax.text(b.x0 + 0.55 + (b.w - 0.55) / 2, cy, label, ha="center", va="center",
                fontsize=8.4, zorder=5)
        return b

    XE, XP, XS = 2.0, 8.4, 14.6
    user = entity(XE, 11.6, "User / operator")
    admin = entity(XE, 7.2, "Administrator")
    security = entity(XE, 4.6, "Security operator")
    reviewer = entity(XE, 2.2, "Audio reviewer")

    p1 = process(XP, 11.6, "P1  Validate &\npreprocess audio")
    p2 = process(XP, 9.4, "P2  Score with both\nmodels (independently)")
    p3 = process(XP, 7.2, "P3  Compare, rules,\nreview routing")
    p4 = process(XP, 4.6, "P4  Alert handling")
    p5 = process(XP, 2.2, "P5  Manual review")

    d1 = store(XS, 11.6, "D1", "Audio files")
    d4 = store(XS, 9.4, "D4", "Model bundles\n(Python + TM)")
    d2 = store(XS, 7.2, "D2", "Events, scores,\nmodel versions")
    d3 = store(XS, 3.4, "D3", "Alerts, reviews,\naudit log", h=3.4)
    boxes = [user, admin, security, reviewer, p1, p2, p3, p4, p5, d1, d4, d2, d3]

    def h_pair(left, right, to_right, to_left):
        """Two horizontal flows between side-by-side nodes, one each way."""
        y_hi, y_lo = left.cy + 0.22, left.cy - 0.22
        if to_right:
            straight(ax, (left.x1, y_hi), (right.x0, y_hi), to_right, boxes)
        if to_left:
            straight(ax, (right.x0, y_lo), (left.x1, y_lo), to_left, boxes)

    def one(a, b, label, y=None):
        y = a.cy if y is None else y
        if a.cx < b.cx:
            straight(ax, (a.x1, y), (b.x0, y), label, boxes)
        else:
            straight(ax, (a.x0, y), (b.x1, y), label, boxes)

    h_pair(user, p1, "clip or 2 s window", "result, event page")
    one(p1, d1, "stored original")
    straight(ax, (p1.cx, p1.y0), (p2.cx, p2.y1), "16 kHz mono samples", boxes)
    one(d4, p2, "loaded at start-up")
    straight(ax, (p2.cx, p2.y0), (p3.cx, p3.y1), "two score lists", boxes)
    one(admin, p3, "thresholds, alert rules")
    one(p3, d2, "event record")
    straight(ax, (p3.cx - 1.2, p3.y0), (p4.cx - 1.2, p4.y1), "confirmed alert", boxes)
    h_pair(security, p4, "ack / dismiss / escalate", "open alerts")
    one(p4, d3, "alert state + audit")
    # Review items bypass alert handling: a straight line to the right of P4.
    xr = p4.x1 + 0.45
    ax.plot([p3.x1 - 0.3, p3.x1 - 0.3, xr], [p3.y0, p3.y0 - 0.35, p3.y0 - 0.35], color=INK, lw=1.2, zorder=2)
    # The line passes under the P4 -> D3 arrow; the gap marks a crossing, not a join.
    gap = 0.14
    ax.plot([xr, xr], [p3.y0 - 0.35, p4.cy + gap], color=INK, lw=1.2, zorder=2)
    ax.plot([xr, xr], [p4.cy - gap, p5.cy + 0.25], color=INK, lw=1.2, zorder=2)
    straight(ax, (xr, p5.cy + 0.25), (p5.x1, p5.cy + 0.25), "", boxes)
    ax.text(xr + 0.12, (p4.y0 + p5.y1) / 2, "uncertain event", fontsize=7.6, va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))
    h_pair(reviewer, p5, "decision, comment", "audio + both models")
    one(p5, d3, "decision (model output kept)", y=p5.cy - 0.25)
    save(fig, "dfd.png")
    return boxes


def FancyBbox(b, fc=SOFT, ec=ACCENT):
    return FancyBboxRaw(b, fc, ec)


def FancyBboxRaw(b, fc, ec):
    return Rectangle((b.x0, b.y0), b.w, b.h, fc=fc, ec=ec, lw=1.5,
                     joinstyle="round", zorder=4)


# --------------------------------------------------------------- use case
def use_case():
    """Five SRS roles and what each may do, taken from ROLE_CAPABILITIES in src/auth.py.

    Redrawn on 26 Sep: the earlier version put the use cases outside the system
    boundary, on top of the actors, and showed only four of the five roles.
    """
    groups = [
        ("Normal user", ["Register, sign in, edit profile", "Upload audio (one or a batch)",
                         "Live microphone monitoring", "View own events: playback,\nwaveform, spectrogram, both models"]),
        ("Audio reviewer", ["Search all events, analytics,\nreports", "Review queue: confirm or\ncorrect the class, comment"]),
        ("Security operator", ["Acknowledge, dismiss or\nescalate alerts", "Alert history"]),
        ("Maintenance operator", ["Edit thresholds and\nalert rules", "Register / activate\nmodel versions"]),
        ("Administrator", ["Manage users and roles", "Export CSV / Excel",
                           "Read the audit trail", "Configure and run retention"]),
    ]
    n_cases = sum(len(c) for _, c in groups)
    step, gap = 1.05, 0.55
    H = n_cases * step + gap * (len(groups) - 1) + 2.4
    W = 14.0
    fig, ax = plt.subplots(figsize=(11.5, H * 0.72))
    ax.set_xlim(0, W), ax.set_ylim(0, H)
    ax.axis("off")
    title(ax, "Use Case Diagram",
          "Every role can also do everything a normal user can; the administrator can do everything")
    ax.add_patch(Rectangle((4.0, 0.3), W - 4.3, H - 1.3, fill=False, ec=INK, lw=1.4,
                           ls=(0, (5, 4)), zorder=1))
    ax.text(4.2, H - 1.25, "SonicSentinel AI", fontsize=9.5, weight="bold", va="top", color=INK)

    boxes = []
    y = H - 2.0
    for actor, cases in groups:
        top = y
        centres = []
        for case in cases:
            e = Box(9.0, y, 5.6, 0.86, case)
            ax.add_patch(Ellipse((e.cx, e.cy), e.w, e.h, fc=SOFT, ec=ACCENT, lw=1.4, zorder=4))
            ax.text(e.cx, e.cy, case, ha="center", va="center", fontsize=7.8, color=INK, zorder=5)
            boxes.append(e)
            centres.append(e)
            y -= step
        ay = (top + y + step) / 2
        stick(ax, 1.6, ay)
        ax.text(1.6, ay - 0.62, actor, ha="center", va="top", fontsize=8.6, weight="bold", color=INK)
        for e in centres:
            ex, ey = ellipse_boundary(e, 1.9, ay)
            ax.plot([1.9, ex], [ay, ey], color=ACTOR_LINE, lw=1.0, zorder=2)
        y -= gap
    save(fig, "use_case.png")
    return boxes


def stick(ax, x, y, s=0.26):
    """UML stick figure centred at (x, y)."""
    ax.add_patch(Circle((x, y + 0.52), s, fc="white", ec=INK, lw=1.4, zorder=4))
    ax.plot([x, x], [y + 0.52 - s, y - 0.02], color=INK, lw=1.4, zorder=4)
    ax.plot([x - 0.3, x + 0.3], [y + 0.26, y + 0.26], color=INK, lw=1.4, zorder=4)
    ax.plot([x, x - 0.26], [y - 0.02, y - 0.5], color=INK, lw=1.4, zorder=4)
    ax.plot([x, x + 0.26], [y - 0.02, y - 0.5], color=INK, lw=1.4, zorder=4)


def ellipse_boundary(e: Box, px, py):
    """Point on ellipse `e` nearest the ray from its centre toward (px, py)."""
    dx, dy = px - e.cx, py - e.cy
    a, b = e.w / 2, e.h / 2
    denom = math.hypot(dx / a, dy / b) or 1.0
    t = 1.0 / denom
    return e.cx + dx * t, e.cy + dy * t


# ---------------------------------------------------------------- activity
def activity():
    fig, ax = plt.subplots(figsize=(10.4, 16.4))
    W, H = 11.6, 19.4
    ax.set_xlim(-0.2, W), ax.set_ylim(0.8, H)
    ax.axis("off")
    title(ax, "Activity Diagram — clip evaluation, end to end",
          "Filled circle = start · ring = end · diamond = decision")

    X = 5.4
    nodes = {}

    def act(key, x, y, label, w=4.9, h=1.05):
        b = Box(x, y, w, h, label)
        nodes[key] = b
        ax.add_patch(FancyBboxRaw(b, SOFT, ACCENT))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.4, zorder=5)
        return b

    def dec(key, x, y, label, w=2.5, h=1.3):
        b = Box(x, y, w, h, label)
        nodes[key] = b
        ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                             fc="#fff7e6", ec=WARN, lw=1.4, zorder=4))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, zorder=5)
        return b

    boxes = []
    top = 18.5
    ax.add_patch(Circle((X, top), 0.2, fc=INK, ec=INK, zorder=4))
    nodes["start"] = Box(X, top, 0.42, 0.42, "start")
    boxes.append(nodes["start"])

    # Order follows AudioPipeline._run and AnalysisPipeline.analyse (corrected 26 Sep:
    # the usable check happens before any model runs, and the Python model now uses
    # CNN14 embeddings rather than the 254 hand-made features).
    flow = ["receive", "validate", "quality"]
    labels = {
        "receive": "Receive upload / 2 s live window",
        "validate": "Decode and validate\n(format, size, duration, integrity)",
        "quality": "Audio quality verdict\n(Good / Acceptable / Poor / Unusable)",
        "preprocess": "Preprocess: high-pass, denoise, trim,\nnormalise, 16 kHz mono",
        "py": "Python model: CNN14 embedding -> MLP\n(scores for all ten classes)",
        "gtm": "Teachable Machine: loudest 1 s -> browser FFT\n(never sees the Python output)",
        "compare": "Compare both models; send to manual review\nif they disagree or confidence/quality is low",
    }
    ys = {"receive": 16.3, "validate": 15.1, "quality": 13.9, "preprocess": 10.9,
          "py": 9.7, "gtm": 8.5, "compare": 7.3}
    SIDE = X + 3.9
    for k in flow:
        act(k, X, ys[k], labels[k])
        boxes.append(nodes[k])

    d1 = dec("d1", X, 12.4, "usable?", w=2.2, h=1.3)
    boxes.append(d1)
    q = act("quarantine", SIDE, 12.4, "Refuse with the reason;\nnothing is classified", w=3.2, h=1.0)
    boxes.append(q)
    for k in ("preprocess", "py", "gtm", "compare"):
        act(k, X, ys[k], labels[k], w=5.6)
        boxes.append(nodes[k])

    d2 = dec("d2", X, 5.6, "alert rule met?\n(class, agreement,\nN windows)", w=3.0, h=1.7)
    boxes.append(d2)
    alert = act("alert", X - 3.9, 5.6, "Raise alert\n(Open state)", w=3.0, h=1.0)
    review = act("review", SIDE, 5.6, "Store the event\nwithout an alert", w=3.2, h=1.0)
    boxes += [alert, review]

    persist = act("persist", X, 3.8, "Persist audio, both score lists,\nversions and audit trail", w=5.2, h=1.0)
    notify = act("notify", X, 2.6, "Show on dashboard, event page, live monitor", w=5.2, h=0.9)
    boxes += [persist, notify]

    end_y = 1.4
    ax.add_patch(Circle((X, end_y), 0.24, fc="white", ec=INK, lw=1.7, zorder=4))
    ax.add_patch(Circle((X, end_y), 0.14, fc=INK, ec=INK, zorder=4))
    nodes["end"] = Box(X, end_y, 0.5, 0.5, "end")
    boxes.append(nodes["end"])

    chain = ["start"] + flow + ["d1"]
    for a, b in zip(chain, chain[1:]):
        A, B = nodes[a], nodes[b]
        straight(ax, A.port("S"), B.port("N"), boxes=boxes)
    straight(ax, d1.port("E"), q.port("W"), "no", boxes=boxes)
    straight(ax, d1.port("S"), nodes["preprocess"].port("N"), "yes", boxes=boxes)
    for a, b in (("preprocess", "py"), ("py", "gtm"), ("gtm", "compare"), ("compare", "d2")):
        straight(ax, nodes[a].port("S"), nodes[b].port("N"), boxes=boxes)

    straight(ax, d2.port("W"), alert.port("E"), "yes", boxes=boxes)
    straight(ax, d2.port("E"), review.port("W"), "no", boxes=boxes)
    straight(ax, alert.port("S"), (alert.cx, persist.cy), boxes=boxes)
    ax.plot([alert.cx, X - 2.6], [persist.cy, persist.cy], color=INK, lw=1.2, zorder=2)
    ax.plot([review.cx, review.cx], [review.y0, persist.cy], color=INK, lw=1.2, zorder=2)
    ax.plot([review.cx, X + 2.6], [persist.cy, persist.cy], color=INK, lw=1.2, zorder=2)
    straight(ax, persist.port("S"), notify.port("N"), boxes=boxes)
    straight(ax, notify.port("S"), (X, end_y + 0.26), boxes=boxes)
    save(fig, "activity.png")
    return boxes


# ---------------------------------------------------------------- sequence
def sequence():
    fig, ax = plt.subplots(figsize=(13.2, 9.0))
    W, H = 15.8, 11.2
    ax.set_xlim(-0.2, W), ax.set_ylim(-0.2, H)
    ax.axis("off")
    title(ax, "Sequence Diagram — upload → dual inference → comparison → alert",
          "Dotted lifelines · numbered messages top to bottom · GTM never sees Python output")

    lanes = [("User", 1.5), ("Web App", 4.7), ("Preproc\nService", 7.7),
             ("Python\nModel", 10.3), ("GTM\nModel", 12.6), ("DB +\nAlerts", 14.6)]
    key = {k: k for k, _ in lanes}
    key.update({"Preproc Service": "Preproc\nService", "Python Model": "Python\nModel",
                "GTM Model": "GTM\nModel", "DB + Alerts": "DB +\nAlerts"})
    pos = {n: x for n, x in lanes}

    heads = {n: Box(x, H - 0.75, 1.5, 0.62, n) for n, x in lanes}
    for n, x in lanes:
        ax.add_patch(FancyBboxRaw(heads[n], SOFT, ACCENT))
        ax.text(x, H - 0.75, n, ha="center", va="center", fontsize=8.2,
                color=ACCENT, weight="bold", zorder=5)
        ax.plot([x, x], [0.45, H - 1.06], color="#9aa5b1", lw=1.0, ls=(0, (2, 3)), zorder=1)

    msgs = [
        ("User", "Web App", 9.6, "1. POST /api/audio/upload (clip)"),
        ("Web App", "Preproc Service", 8.8, "2. decode + resample to 16 kHz"),
        ("Preproc Service", "Web App", 8.0, "3. samples + quality verdict"),
        ("Web App", "Python Model", 7.1, "4. CNN14 embedding -> predict_proba"),
        ("Python Model", "Web App", 6.3, "5. class + per-class confidence"),
        ("Web App", "GTM Model", 5.4, "6. predict(samples)  [no Python output]"),
        ("GTM Model", "Web App", 4.6, "7. gtm_class + confidences"),
        ("Web App", "Web App", 3.8, "8. consistency verdict"),
        ("Web App", "DB + Alerts", 3.0, "9. persist event + comparison"),
        ("DB + Alerts", "Web App", 2.2, "10. alert_id (if critical)"),
        ("Web App", "User", 1.4, "11. JSON result / redirect"),
    ]
    boxes = list(heads.values())
    for src, dst, y, label in msgs:
        if src == dst:
            x = pos[key[src]]
            ax.annotate("", xy=(x + 0.85, y - 0.16), xytext=(x, y - 0.16),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
            ax.text(x + 0.95, y - 0.16, label, fontsize=7.5, va="center", ha="left")
        else:
            xs, xd = pos[key[src]], pos[key[dst]]
            straight(ax, (xs, y), (xd, y), boxes=boxes)
            mx = (xs + xd) / 2
            side = 1 if xd >= xs else -1
            ax.text(mx, y + 0.15, label, ha="center", va="bottom", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
            _ = side

    ax.text(0.2, 0.25, "Note: both models receive the same preprocessed samples and nothing else; the consistency "
                       "verdict is computed only after both have answered.",
            fontsize=7.2, color="#7b8794", style="italic")
    save(fig, "sequence_upload.png")
    return boxes


# ---------------------------------------------------------- decision flow
def decision_flow():
    """The decision path of AnalysisPipeline.analyse(), top to bottom.

    Redrawn on 26 Sep: the earlier version showed an auto-acknowledge step, a timeout
    and a feedback-to-dataset arrow that the app does not have, and its routed edges
    crossed. This one only uses straight arrows, and every box matches a step in
    src/services/pipeline.py or src/api/alerts_api.py.
    """
    fig, ax = plt.subplots(figsize=(12.5, 13.5))
    W, H = 14.0, 15.2
    ax.set_xlim(0, W), ax.set_ylim(0, H)
    ax.axis("off")
    title(ax, "Decision flow for one clip or live window",
          "Every box is a step in src/services/pipeline.py; the bottom row is the alert lifecycle")

    def state(cx, cy, label, w=5.6, h=0.95, fc=SOFT, ec=ACCENT):
        b = Box(cx, cy, w, h, label)
        ax.add_patch(FancyBboxRaw(b, fc, ec))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=8.0,
                color=INK, zorder=5)
        return b

    def dec(cx, cy, label, w=5.6, h=1.5):
        b = Box(cx, cy, w, h, label)
        ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2),
                              (cx - w / 2, cy)], fc="#fff7e6", ec=WARN, lw=1.4, zorder=4))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=7.6, zorder=5)
        return b

    X, SIDE = 4.6, 11.0
    received = state(X, 14.3, "Audio received: an upload, or a 2 s live window")
    usable = dec(X, 12.75, "Decodes, at least 0.5 s long,\nquality not Unusable?")
    rejected = state(SIDE, 12.75, "Refused with the reason\n(422, audited as a failed upload)",
                     w=4.4, fc="#fdecec", ec="#b72f43")
    models = state(X, 11.2, "Python model and Teachable Machine each score\n"
                            "the same preprocessed audio (neither sees the other)", h=1.1)
    compare = state(X, 9.8, "Compare: same class?  |Python top - TM top|,\n"
                            "top-two margin, overlap -> consistency status", h=1.1)
    review = dec(X, 8.25, "Any review condition? disagreement, low confidence,\n"
                          "small margin, poor quality, overlap, near-duplicate")
    queue = state(SIDE, 8.25, "Manual-review queue\n(original model output kept)", w=4.4)
    severity = state(X, 6.75, "Severity from the class rule\n(Informational ... Critical)")
    confirm = dec(X, 5.2, "Alertable class, models agree, quality OK,\n"
                          "N consecutive windows reached?")
    stored = state(SIDE, 5.2, "Stored as an event,\nno alert", w=4.4)
    alert = state(X, 3.65, "Alert raised once, with the recommended action", w=5.6)
    opened = state(X, 2.2, "Open", w=2.2, h=0.8)
    acked = state(1.4, 0.7, "Acknowledged", w=2.4, h=0.8)
    escal = state(X, 0.7, "Escalated", w=2.2, h=0.8)
    dismissed = state(8.4, 0.7, "Dismissed\n(reason required)", w=2.9, h=0.8)

    boxes = [received, usable, rejected, models, compare, review, queue, severity,
             confirm, stored, alert, opened, acked, escal, dismissed]

    def down(a, b, label=""):
        straight(ax, (a.cx, a.y0), (b.cx, b.y1), label, boxes)

    def right(a, b, label):
        straight(ax, (a.x1, a.cy), (b.x0, b.cy), label, boxes)

    down(received, usable)
    right(usable, rejected, "no")
    down(usable, models, "yes")
    down(models, compare)
    down(compare, review)
    right(review, queue, "yes")
    down(review, severity, "either way")
    down(severity, confirm)
    right(confirm, stored, "no")
    down(confirm, alert, "yes")
    down(alert, opened)
    straight(ax, (opened.x0, opened.cy), (acked.cx, acked.y1), "acknowledge", boxes)
    down(opened, escal, "escalate")
    straight(ax, (opened.x1, opened.cy), (dismissed.cx, dismissed.y1), "dismiss", boxes)
    straight(ax, (acked.x1, acked.cy), (escal.x0, escal.cy), "", boxes)
    ax.text(2.55, 0.05, "Acknowledged or escalated alerts can still be dismissed later.",
            fontsize=7.2, color="#52606d")
    save(fig, "decision_flow.png")
    return boxes


# -------------------------------------------------------------------- check
def check(boxes_by_diag, edges_by_diag=None):
    import os
    ok = True
    print("\n-- structural check --")
    for name, boxes in boxes_by_diag.items():
        bad = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if overlap(boxes[i], boxes[j], pad=0.02):
                    bad.append((boxes[i].name, boxes[j].name))
        if bad:
            ok = False
        edges = (edges_by_diag or {}).get(name, [])
        # an edge may only touch the boxes it terminates at; every interior
        # segment must clear every other node (segment-rect intersection).
        bad_e = []
        for pts in edges:
            ends = [b for b in boxes
                    if any(b.contains(p[0], p[1], pad=0.04) for p in (pts[0], pts[-1]))]
            for b in boxes:
                if b in ends:
                    continue
                if not path_clear(pts, [b]):
                    bad_e.append(b.name)
                    break
        if bad_e:
            ok = False
        n_status = "ok" if not bad else f"OVERLAP {bad}"
        e_status = "ok" if not bad_e else f"EDGE-THROUGH-NODE {bad_e}"
        print(f"  {name:<14} {len(boxes):>2} nodes / {len(edges):>2} edges  "
              f"{n_status}  {e_status}")

    missing = [f for f in ("dfd.png", "use_case.png", "activity.png",
                           "sequence_upload.png", "decision_flow.png")
               if not (OUT / f).exists()]
    if missing:
        ok = False
        print("  MISSING:", missing)
    else:
        for f in sorted(os.listdir(OUT)):
            if f.endswith(".png"):
                print(f"  {OUT / f}")
    print("  result:", "PASS" if ok else "FAIL")
    return ok


def main():
    args = sys.argv[1:]
    OUT.mkdir(exist_ok=True)
    print("rendering diagrams ->", OUT)
    names = ["dfd", "use_case", "activity", "sequence", "decision_flow"]
    drawers = [dfd, use_case, activity, sequence, decision_flow]
    got, edges = {}, {}
    for nm, fn in zip(names, drawers):
        EDGES.clear()
        got[nm] = fn()
        edges[nm] = list(EDGES)
    if "--check" in args:
        return 0 if check(got, edges) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
