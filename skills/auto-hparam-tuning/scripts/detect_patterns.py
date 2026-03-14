"""Training curve pattern detection for AHT.

Analyzes the summary statistics produced by ``analyze_event.summarize_scalar_curve``
and emits structured diagnostic labels (e.g. divergence, plateau, overfitting).

Usage
-----
As a CLI tool::

    python detect_patterns.py EVENT_PATH METRIC_KEY --mode min
    python detect_patterns.py EVENT_PATH train/loss val/loss --mode min

As a library::

    from detect_patterns import detect_patterns, detect_overfitting
    diagnosis = detect_patterns(summary_dict)
    overfitting_report = detect_overfitting(train_summary, val_summary)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

DiagnosisLabel = Literal[
    "divergence",
    "instability",
    "plateau",
    "overfitting",
    "underfitting",
    "early_saturation",
    "slow_training",
    "healthy",
]

SEVERITY_ORDER: dict[DiagnosisLabel, int] = {
    "divergence": 7,
    "instability": 6,
    "overfitting": 5,
    "underfitting": 4,
    "plateau": 3,
    "early_saturation": 2,
    "slow_training": 1,
    "healthy": 0,
}


def _total_steps(s: dict) -> int:
    """Return the number of recorded data points."""
    return int(s.get("count", 0))


def _training_fraction(s: dict, step: int | float | None) -> float:
    """Return how far *step* is into training as a fraction in [0, 1]."""
    if step is None:
        return 1.0
    start = s.get("start_step", 0)
    end = s.get("end_step", start)
    span = end - start
    if span <= 0:
        return 1.0
    return max(0.0, min(1.0, (step - start) / span))


# ---------------------------------------------------------------------------
# Individual pattern detectors
# ---------------------------------------------------------------------------


def _check_divergence(s: dict, *, threshold_factor: float = 3.0) -> dict[str, Any] | None:
    """Detect divergence: metric getting drastically worse.

    Heuristic: ``worsened`` is *True* AND the absolute ``improvement`` is large
    relative to the ``robust_range``, or the ``final_value`` moved far from the
    initial value in the wrong direction.
    """
    if not s.get("worsened", False):
        return None

    robust_range = s.get("robust_range", 1.0) or 1.0
    improvement = abs(s.get("improvement", 0.0))

    # The metric worsened; check if it worsened *a lot*
    if improvement / (robust_range + 1e-12) < threshold_factor:
        return None

    return {
        "label": "divergence",
        "severity": "critical",
        "confidence": "high" if improvement / (robust_range + 1e-12) > 5.0 else "medium",
        "evidence": {
            "worsened": True,
            "improvement": s.get("improvement"),
            "robust_range": robust_range,
            "ratio": round(improvement / (robust_range + 1e-12), 3),
            "initial_value": s.get("initial_value"),
            "final_value": s.get("final_value"),
        },
        "message": (
            f"Metric '{s['key']}' worsened by {improvement:.4g} "
            f"(~{improvement / (robust_range + 1e-12):.1f}x the robust range). "
            "This strongly suggests divergence."
        ),
    }


def _check_instability(
    s: dict,
    *,
    oscillation_threshold: float = 0.5,
    sign_change_rate_threshold: float = 0.3,
) -> dict[str, Any] | None:
    """Detect instability: high oscillation or many sign changes."""
    norm_osc = s.get("normalized_oscillation", 0.0)
    count = _total_steps(s)
    sign_changes = s.get("sign_change_count", 0)
    sign_change_rate = sign_changes / max(count - 1, 1)

    if norm_osc < oscillation_threshold and sign_change_rate < sign_change_rate_threshold:
        return None

    confidence = "high" if (norm_osc > 1.0 or sign_change_rate > 0.5) else "medium"

    return {
        "label": "instability",
        "severity": "warning",
        "confidence": confidence,
        "evidence": {
            "normalized_oscillation": round(norm_osc, 4),
            "sign_change_count": sign_changes,
            "sign_change_rate": round(sign_change_rate, 4),
            "total_steps": count,
        },
        "message": (
            f"Metric '{s['key']}' shows high instability "
            f"(normalized oscillation={norm_osc:.3f}, "
            f"sign change rate={sign_change_rate:.2%}). "
            "Consider reducing learning rate or increasing batch size."
        ),
    }


def _check_plateau(
    s: dict,
    *,
    best_step_fraction_threshold: float = 0.5,
    min_steps: int = 20,
) -> dict[str, Any] | None:
    """Detect plateau: best metric achieved early, no improvement since."""
    count = _total_steps(s)
    if count < min_steps:
        return None

    best_step = s.get("best_step")
    best_frac = _training_fraction(s, best_step)

    if best_frac > best_step_fraction_threshold:
        return None  # best is recent enough

    # Additional check: improvement should be small or the metric should not be
    # worsened — if it actually worsened a lot, the divergence detector handles it.
    improvement = abs(s.get("improvement", 0.0))
    robust_range = s.get("robust_range", 1.0) or 1.0
    if improvement / (robust_range + 1e-12) > 2.0 and s.get("worsened", False):
        return None  # divergence, not plateau

    return {
        "label": "plateau",
        "severity": "warning",
        "confidence": "high" if best_frac < 0.3 else "medium",
        "evidence": {
            "best_step": best_step,
            "end_step": s.get("end_step"),
            "best_step_fraction": round(best_frac, 3),
            "best_value": s.get("best_value"),
            "final_value": s.get("final_value"),
        },
        "message": (
            f"Metric '{s['key']}' peaked at step {best_step} "
            f"({best_frac:.0%} through training) and has not improved since. "
            "Consider learning rate scheduling or early stopping."
        ),
    }


def _check_early_saturation(
    s: dict,
    *,
    saturation_fraction_threshold: float = 0.2,
    min_steps: int = 20,
) -> dict[str, Any] | None:
    """Detect early saturation: metric converges too quickly and stops."""
    count = _total_steps(s)
    if count < min_steps:
        return None

    best_step = s.get("best_step")
    best_frac = _training_fraction(s, best_step)

    if best_frac > saturation_fraction_threshold:
        return None

    # Must have improved from baseline (not diverged)
    if s.get("worsened", False):
        return None

    improvement = abs(s.get("improvement", 0.0))
    if improvement < 1e-8:
        return None  # no improvement at all

    return {
        "label": "early_saturation",
        "severity": "info",
        "confidence": "medium",
        "evidence": {
            "best_step": best_step,
            "end_step": s.get("end_step"),
            "best_step_fraction": round(best_frac, 3),
            "improvement": s.get("improvement"),
        },
        "message": (
            f"Metric '{s['key']}' reached its best value at step {best_step} "
            f"({best_frac:.0%} through training). "
            "This may indicate early saturation — try warmup + cosine schedule."
        ),
    }


def _check_slow_training(
    s: dict,
    *,
    improvement_per_step_threshold: float = 1e-6,
    min_steps: int = 50,
) -> dict[str, Any] | None:
    """Detect slow training: metric improving but very slowly."""
    count = _total_steps(s)
    if count < min_steps:
        return None

    improvement = s.get("improvement", 0.0)
    if improvement <= 0:
        return None  # not improving at all (handled by other detectors)

    robust_range = s.get("robust_range", 1.0) or 1.0
    normalized_rate = (improvement / count) / (robust_range + 1e-12)

    if normalized_rate > improvement_per_step_threshold:
        return None

    # Also check if training is still improving at the end (i.e., not plateaued)
    best_frac = _training_fraction(s, s.get("best_step"))
    if best_frac < 0.7:
        return None  # this is more of a plateau

    return {
        "label": "slow_training",
        "severity": "info",
        "confidence": "medium",
        "evidence": {
            "improvement": improvement,
            "total_steps": count,
            "improvement_per_step": round(improvement / count, 8),
            "normalized_rate": round(normalized_rate, 8),
        },
        "message": (
            f"Metric '{s['key']}' is improving but very slowly "
            f"(normalized rate={normalized_rate:.2e} per step). "
            "Consider increasing learning rate or batch size."
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_patterns(
    summary: dict[str, Any],
    *,
    oscillation_threshold: float = 0.5,
    sign_change_rate_threshold: float = 0.3,
    plateau_fraction: float = 0.5,
    saturation_fraction: float = 0.2,
    divergence_factor: float = 3.0,
    slow_rate_threshold: float = 1e-6,
) -> dict[str, Any]:
    """Run all pattern detectors on one scalar-curve summary.

    Parameters
    ----------
    summary : dict
        Output of ``analyze_event.summarize_scalar_curve``.

    Returns
    -------
    dict
        Structured diagnosis with ``labels``, ``primary_label``, ``details``,
        and ``overall_severity``.
    """
    detectors = [
        lambda s: _check_divergence(s, threshold_factor=divergence_factor),
        lambda s: _check_instability(
            s,
            oscillation_threshold=oscillation_threshold,
            sign_change_rate_threshold=sign_change_rate_threshold,
        ),
        lambda s: _check_plateau(s, best_step_fraction_threshold=plateau_fraction),
        lambda s: _check_early_saturation(s, saturation_fraction_threshold=saturation_fraction),
        lambda s: _check_slow_training(s, improvement_per_step_threshold=slow_rate_threshold),
    ]

    findings: list[dict[str, Any]] = []
    for detector in detectors:
        result = detector(summary)
        if result is not None:
            findings.append(result)

    if not findings:
        findings.append({
            "label": "healthy",
            "severity": "info",
            "confidence": "medium",
            "evidence": {
                "improvement": summary.get("improvement"),
                "normalized_oscillation": summary.get("normalized_oscillation"),
                "best_step_fraction": round(
                    _training_fraction(summary, summary.get("best_step")), 3
                ),
            },
            "message": (
                f"Metric '{summary['key']}' shows healthy training behavior. "
                "No significant issues detected."
            ),
        })

    labels = [f["label"] for f in findings]
    primary = max(findings, key=lambda f: SEVERITY_ORDER.get(f["label"], 0))

    return {
        "key": summary.get("key"),
        "mode": summary.get("mode"),
        "labels": labels,
        "primary_label": primary["label"],
        "overall_severity": primary["severity"],
        "num_issues": len([f for f in findings if f["label"] != "healthy"]),
        "details": findings,
    }


def detect_overfitting(
    train_summary: dict[str, Any],
    val_summary: dict[str, Any],
    *,
    gap_threshold: float = 0.1,
) -> dict[str, Any] | None:
    """Detect overfitting by comparing train and validation curves.

    Parameters
    ----------
    train_summary, val_summary : dict
        Outputs of ``analyze_event.summarize_scalar_curve`` for train and
        validation metrics respectively.
    gap_threshold : float
        Minimum normalized generalization gap to trigger overfitting diagnosis.

    Returns
    -------
    dict or None
        Overfitting diagnosis if detected, else ``None``.
    """
    mode = val_summary.get("mode", "higher")
    train_best = train_summary.get("best_value", 0)
    val_best = val_summary.get("best_value", 0)

    # Compute generalization gap (always positive means train is "better")
    if mode == "higher":
        gap = train_best - val_best
    else:
        gap = val_best - train_best  # for loss, val being higher = gap

    # Normalize by the robust range of the validation metric
    val_range = val_summary.get("robust_range", 1.0) or 1.0
    normalized_gap = gap / (val_range + 1e-12)

    # Check if training is still improving but validation is not
    train_improving = not train_summary.get("worsened", True)
    val_plateaued_or_worsened = (
        val_summary.get("worsened", False)
        or _training_fraction(val_summary, val_summary.get("best_step")) < 0.5
    )

    if normalized_gap < gap_threshold and not (train_improving and val_plateaued_or_worsened):
        return None

    confidence = "high" if (normalized_gap > 0.3 and val_plateaued_or_worsened) else "medium"

    return {
        "label": "overfitting",
        "severity": "warning",
        "confidence": confidence,
        "evidence": {
            "train_best": train_best,
            "val_best": val_best,
            "generalization_gap": round(gap, 6),
            "normalized_gap": round(normalized_gap, 4),
            "train_improving": train_improving,
            "val_plateaued_or_worsened": val_plateaued_or_worsened,
            "train_best_step": train_summary.get("best_step"),
            "val_best_step": val_summary.get("best_step"),
        },
        "message": (
            f"Overfitting detected: train best={train_best:.4g}, "
            f"val best={val_best:.4g} (gap={gap:.4g}, "
            f"normalized gap={normalized_gap:.2f}). "
            "Consider increasing regularization (dropout, weight_decay)."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect training curve patterns from a TensorBoard event file.\n\n"
            "Runs all pattern detectors (divergence, instability, plateau, "
            "overfitting, early saturation, slow training) on the specified "
            "metric(s) and outputs a structured JSON diagnosis.\n\n"
            "When two metric keys are given, the first is treated as the "
            "training metric and the second as the validation metric; this "
            "enables the overfitting detector."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "event_path",
        metavar="EVENT_PATH",
        help="Path to the TensorBoard event file or directory containing it.",
    )
    parser.add_argument(
        "key",
        metavar="KEY",
        nargs="+",
        help=(
            "One or two scalar tags. If two are given the first is the train "
            "metric and the second is the validation metric."
        ),
    )
    parser.add_argument(
        "-m", "--mode",
        default=None,
        metavar="DIR",
        help="Optimization direction: 'higher'/'max' or 'lower'/'min'.",
    )
    parser.add_argument(
        "-s", "--smoothing",
        type=float,
        default=0.0,
        metavar="ALPHA",
        help="EMA smoothing factor in [0, 1). Default: 0.",
    )

    args = parser.parse_args()

    # Import the sibling analyze_event module from the same directory.
    import importlib.util
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "analyze_event", os.path.join(here, "analyze_event.py")
    )
    if spec is None or spec.loader is None:
        print("Error: cannot locate analyze_event.py in the same directory.", file=sys.stderr)
        sys.exit(1)
    ae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ae)

    # Load and summarize curves
    keys = args.key
    summaries = ae.summarize_scalar_curve_from_event(
        event_path=args.event_path,
        key=keys,
        mode=args.mode,
        smoothing=args.smoothing,
    )
    if isinstance(summaries, dict):
        summaries = [summaries]

    results: list[dict[str, Any]] = []
    for summary in summaries:
        diagnosis = detect_patterns(summary)
        results.append(diagnosis)

    # Overfitting detection when train + val keys are given
    overfitting_result = None
    if len(summaries) == 2:
        overfitting_result = detect_overfitting(summaries[0], summaries[1])
        if overfitting_result is not None:
            # Merge into the validation diagnosis
            results[1]["labels"].append("overfitting")
            results[1]["details"].append(overfitting_result)
            # Update primary if overfitting is more severe
            if SEVERITY_ORDER.get("overfitting", 0) > SEVERITY_ORDER.get(
                results[1]["primary_label"], 0
            ):
                results[1]["primary_label"] = "overfitting"
                results[1]["overall_severity"] = "warning"
            results[1]["num_issues"] += 1

    output = results if len(results) > 1 else results[0]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
