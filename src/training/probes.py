"""Noise-robustness probing — measuring degradation without lying about it.

The SRS asks for noise robustness alongside the headline metrics. The trap is that a probe
scores degraded TEST audio, and the moment you let that number influence which model you
ship, the test split has leaked into the selection and the reported test accuracy is no
longer an unbiased estimate of anything. It becomes a number that was partly chosen.

So the rule is structural, not a matter of discipline:

    The probe runs ONCE, after `python_models/selection.json` is frozen.

`assert_selection_frozen()` enforces it and returns the hash of the selection file, which
is recorded in the probe report. If the selection changes afterwards, the hash no longer
matches and the recorded probe is visibly stale rather than quietly misleading.

Probe results are built with `probe=True`, and `EvaluationResult.meets_floors` refuses to
certify a probe, so a robustness number can never be pasted in where an accuracy belongs.

Every degradation is seeded from (probe_seed, audio_id, condition), so a re-run reproduces
the same noise realisation byte for byte. A robustness figure nobody can reproduce is a
robustness figure nobody can trust.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.training.evaluation import (
    EvaluationError,
    EvaluationResult,
    evaluate_predictions,
)


class ProbeError(RuntimeError):
    """Raised when a probe would produce a number that cannot be defended."""


# --------------------------------------------------------------------------------------
# Degradations — deterministic, and each one models something that happens for real
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """One degradation to apply to every clip in the probe."""

    name: str
    kind: str                  # "noise" | "gain" | "lowpass"
    level: float               # SNR in dB, gain in dB, or cutoff in Hz
    description: str = ""

    def label(self) -> str:
        if self.kind == "noise":
            return f"{self.name} (SNR {self.level:g} dB)"
        if self.kind == "gain":
            return f"{self.name} ({self.level:+g} dB)"
        return f"{self.name} (lowpass {self.level:g} Hz)"


def _rms(samples: np.ndarray) -> float:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr**2)))


def _seed_for(probe_seed: int, audio_id: str, condition: str) -> int:
    digest = hashlib.sha256(f"{probe_seed}:{audio_id}:{condition}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def add_noise(samples: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise scaled to hit a target signal-to-noise ratio.

    The noise power is derived from the signal's own RMS, so the SNR means the same thing
    for a quiet clip and a loud one. When the signal is digital silence there is no SNR to
    speak of; the signal is returned unchanged and the caller records the quality verdict.
    """
    arr = np.asarray(samples, dtype=np.float64)
    signal_rms = _rms(arr)
    if signal_rms <= 0:
        return arr.astype(np.float32)

    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noise = rng.standard_normal(arr.shape) * noise_rms
    return (arr + noise).astype(np.float32)


def apply_gain(samples: np.ndarray, db: float) -> np.ndarray:
    """A level shift, modelling a distant or over-driven microphone.

    Clipped deliberately: an over-driven real recording clips, and pretending otherwise
    would make the probe easier than the world it is supposed to represent.
    """
    arr = np.asarray(samples, dtype=np.float64) * (10.0 ** (db / 20.0))
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


def lowpass(samples: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Band-limiting, modelling a poor microphone or a lossy codec."""
    from scipy.signal import butter, sosfiltfilt

    nyquist = sr / 2.0
    if not 0 < cutoff_hz < nyquist:
        raise ProbeError(
            f"cutoff {cutoff_hz} Hz is outside (0, {nyquist}) for a {sr} Hz signal"
        )
    sos = butter(6, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return sosfiltfilt(sos, np.asarray(samples, dtype=np.float64)).astype(np.float32)


def degrade(samples: np.ndarray, sr: int, condition: Condition, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if condition.kind == "noise":
        return add_noise(samples, sr, condition.level, rng)
    if condition.kind == "gain":
        return apply_gain(samples, condition.level)
    if condition.kind == "lowpass":
        return lowpass(samples, sr, condition.level)
    raise ProbeError(f"unknown condition kind {condition.kind!r}")


def conditions_from_config(config: Mapping[str, Any]) -> list[Condition]:
    """Build the condition list from config/robustness, never from literals."""
    block = config.get("robustness")
    if not isinstance(block, Mapping):
        raise ProbeError(
            "config/thresholds.json has no 'robustness' block, so the probe conditions "
            "would have to be hard-coded — which is what this project must not do"
        )
    conditions: list[Condition] = []
    for snr in block.get("probe_snr_db", []) or []:
        conditions.append(
            Condition(name="Additive noise", kind="noise", level=float(snr),
                      description="street/crowd noise floor")
        )
    for db in block.get("gain_shift_db", []) or []:
        conditions.append(
            Condition(name="Level shift", kind="gain", level=float(db),
                      description="distant or over-driven microphone")
        )
    for cutoff in block.get("probe_lowpass_hz", []) or []:
        conditions.append(
            Condition(name="Band-limited", kind="lowpass", level=float(cutoff),
                      description="poor microphone or lossy codec")
        )
    if not conditions:
        raise ProbeError("the robustness block configures no probe conditions")
    return conditions


# --------------------------------------------------------------------------------------
# The freeze rule
# --------------------------------------------------------------------------------------

def selection_hash(selection_path: str | Path) -> str:
    path = Path(selection_path)
    if not path.exists():
        raise ProbeError(
            f"no frozen selection at {path}. The probe must not run before the model is "
            "chosen: a probe scores degraded test audio, and selecting on it would leak "
            "the test split into the choice. Write selection.json first."
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_selection_frozen(selection_path: str | Path) -> str:
    """Return the selection hash, or refuse to run a probe at all."""
    digest = selection_hash(selection_path)
    try:
        payload = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProbeError(f"{selection_path} is not valid JSON: {exc}") from exc

    if not payload.get("selected_model"):
        raise ProbeError(
            f"{selection_path} names no selected_model, so there is nothing to freeze"
        )
    if not payload.get("rationale"):
        raise ProbeError(
            f"{selection_path} records no rationale. A selection without a written "
            "rationale is a preference, and the probe would be certifying a preference."
        )
    return digest


# --------------------------------------------------------------------------------------
# The probe
# --------------------------------------------------------------------------------------

@dataclass
class ProbeReport:
    model_name: str
    model_version: str
    selection_hash: str
    baseline_accuracy: float
    baseline_macro_f1: float
    baseline_critical_recall: float
    conditions: list[dict[str, Any]]
    probe_seed: int
    baseline_result: EvaluationResult | None = None

    def worst_condition(self) -> dict[str, Any] | None:
        if not self.conditions:
            return None
        return min(self.conditions, key=lambda c: c["accuracy"])

    def accuracy_drop(self, condition: Mapping[str, Any]) -> float:
        return float(self.baseline_accuracy - condition["accuracy"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "noise-robustness probe",
            "disclaimer": (
                "These conditions score DEGRADED test audio. They measure how gracefully "
                "accuracy degrades and are NOT test accuracy. They did not influence the "
                "model selection; the probe ran after selection.json was frozen, and "
                "selection_hash proves which selection it describes."
            ),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "selection_hash": self.selection_hash,
            "probe_seed": self.probe_seed,
            "baseline": {
                "accuracy": self.baseline_accuracy,
                "macro_f1": self.baseline_macro_f1,
                "critical_recall": self.baseline_critical_recall,
            },
            "conditions": [dict(c) for c in self.conditions],
        }


def run_robustness_probe(
    records: Sequence[Mapping[str, Any]],
    load_audio: Callable[[Mapping[str, Any]], tuple[np.ndarray, int]],
    predict_samples: Callable[[np.ndarray, int, Mapping[str, Any]], Any],
    class_names: Sequence[str],
    critical_classes: Sequence[str],
    *,
    config: Mapping[str, Any],
    selection_path: str | Path,
    model_name: str,
    model_version: str,
    conditions: Sequence[Condition] | None = None,
    label_field: str = "class_label",
) -> ProbeReport:
    """Score degraded copies of every record under every condition.

    `load_audio(record) -> (samples, sample_rate)` reads the clip once; the SAME clip is
    then degraded per condition, so the only thing that differs between conditions is the
    degradation. `predict_samples(samples, sample_rate, record)` runs the model on the
    degraded audio.

    Requires a frozen selection first — see the module docstring for why.
    """
    digest = assert_selection_frozen(selection_path)
    probe_seed = int(config.get("robustness", {}).get("probe_seed", 20260923))
    conditions = list(conditions) if conditions is not None else conditions_from_config(config)

    # Baseline: the same clips, undegraded, through the same path. Without it, a drop of
    # "0.81 accuracy under noise" says nothing at all.
    baseline_result, _ = evaluate_predictions(
        records,
        lambda record: predict_samples(*load_audio(record), record),
        class_names,
        critical_classes,
        label_field=label_field,
        model_name=model_name,
        model_version=model_version,
        probe=True,
        probe_label="baseline (clean audio, probe path)",
        config_snapshot={"probe_seed": probe_seed},
    )
    if baseline_result.n_records != len(records):
        raise ProbeError("baseline did not score every record")

    rows: list[dict[str, Any]] = []
    for condition in conditions:
        def predict_degraded(record: Mapping[str, Any], _c: Condition = condition):
            samples, sr = load_audio(record)
            seed = _seed_for(probe_seed, str(record.get("audio_id", "")), _c.name)
            return predict_samples(degrade(samples, sr, _c, seed), sr, record)

        result, _ = evaluate_predictions(
            records,
            predict_degraded,
            class_names,
            critical_classes,
            label_field=label_field,
            model_name=model_name,
            model_version=model_version,
            probe=True,
            probe_label=condition.label(),
            config_snapshot={
                "probe_seed": probe_seed,
                "condition": condition.name,
                "level": condition.level,
                "selection_hash": digest,
            },
        )
        rows.append(
            {
                "condition": condition.name,
                "label": condition.label(),
                "kind": condition.kind,
                "level": condition.level,
                "description": condition.description,
                "n_records": result.n_records,
                "accuracy": result.accuracy,
                "accuracy_drop": baseline_result.accuracy - result.accuracy,
                "macro_f1": result.macro_f1,
                "macro_f1_drop": baseline_result.macro_f1 - result.macro_f1,
                "critical_recall": result.critical_recall,
                "critical_recall_drop": (
                    baseline_result.critical_recall - result.critical_recall
                ),
                "severe_errors": len(result.severe_errors),
            }
        )

    return ProbeReport(
        model_name=model_name,
        model_version=model_version,
        selection_hash=digest,
        baseline_accuracy=baseline_result.accuracy,
        baseline_macro_f1=baseline_result.macro_f1,
        baseline_critical_recall=baseline_result.critical_recall,
        conditions=rows,
        probe_seed=probe_seed,
        baseline_result=baseline_result,
    )


def write_probe_report(report: ProbeReport, path: str | Path) -> Path:
    """Write it with the provenance that makes it re-checkable."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    markdown = Path(target.with_suffix(".md"))
    lines = [
        f"# Noise-robustness probe — {report.model_name} {report.model_version}",
        "",
        payload["disclaimer"],
        "",
        f"Selection frozen as `{report.selection_hash[:16]}…` "
        f"(probe seed {report.probe_seed}).",
        "",
        f"Baseline on the same clips, undegraded: accuracy "
        f"**{report.baseline_accuracy:.4f}**, macro-F1 "
        f"{report.baseline_macro_f1:.4f}, critical recall "
        f"{report.baseline_critical_recall:.4f}.",
        "",
        "| Condition | Accuracy | Δ accuracy | Macro-F1 | Critical recall | Silent misses |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.conditions:
        lines.append(
            f"| {row['label']} | {row['accuracy']:.4f} | "
            f"{row['accuracy_drop']:+.4f} | {row['macro_f1']:.4f} | "
            f"{row['critical_recall']:.4f} | {row['severe_errors']} |"
        )
    worst = report.worst_condition()
    lines += ["", "## Reading", ""]
    if worst:
        lines.append(
            f"Accuracy is worst under **{worst['label']}**: {worst['accuracy']:.4f}, a drop of "
            f"{report.accuracy_drop(worst):.4f} from the clean baseline on the same clips. "
            f"Critical-class recall falls to {worst['critical_recall']:.4f}, with "
            f"{worst['severe_errors']} critical event(s) not flagged as critical."
        )
    lines += [
        "",
        "This is a measurement of degradation. It is not the model's test accuracy, and it "
        "did not decide which model shipped.",
        "",
    ]
    markdown.write_text("\n".join(lines), encoding="utf-8")
    return target
