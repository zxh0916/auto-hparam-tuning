from typing import Any, Literal

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

OptimizationMode = Literal["higher", "lower"]


def event2dataframe(event_path: str):
    ea = event_accumulator.EventAccumulator(
        event_path,
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.TENSORS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.HISTOGRAMS: 0,
        },
    ).Reload()
    df = pd.DataFrame(columns=ea.Tags()["scalars"])
    df.index.name = "step"
    pbar = tqdm(ea.Tags()["scalars"], desc="Loading scalar tags")
    for scalar in pbar:
        pbar.set_postfix_str(f"tag={scalar}")
        for event in ea.Scalars(scalar):
            df.loc[event.step, scalar] = event.value
    return df


def _normalize_opt_mode(higher_is_better: bool | None = None, mode: str | None = None) -> OptimizationMode:
    """Normalize metric optimization direction into a compact internal literal."""
    if mode is not None:
        normalized = mode.strip().lower()
        if normalized in {"higher", "high", "max", "maximize", "larger", "greater"}:
            return "higher"
        if normalized in {"lower", "low", "min", "minimize", "smaller", "less"}:
            return "lower"
        raise ValueError(f"Unsupported mode={mode!r}. Use 'higher'/'max' or 'lower'/'min'.")

    if higher_is_better is None:
        raise ValueError("Either higher_is_better or mode must be provided.")

    return "higher" if higher_is_better else "lower"


def _series_from_dataframe(df: pd.DataFrame, key: str) -> pd.Series:
    """Extract one scalar column, coerce to float, and drop missing values."""
    if key not in df.columns:
        raise KeyError(f"Key {key!r} not found in dataframe columns.")

    series = pd.to_numeric(df[key], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"Key {key!r} has no valid scalar values.")

    # Keep the step index numeric when possible to make downstream stats easier to read.
    try:
        series.index = pd.Index(pd.to_numeric(series.index), name=series.index.name)
    except Exception:
        pass
    return series.astype(float)


def _smooth_series(series: pd.Series, smoothing: float = 0.0) -> pd.Series:
    """Apply exponential moving average smoothing with TensorBoard-like semantics."""
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must satisfy 0 <= smoothing < 1.")
    if smoothing == 0.0 or len(series) <= 1:
        return series.copy()
    alpha = 1.0 - smoothing
    return series.ewm(alpha=alpha, adjust=False).mean()


def _step_to_python(step: Any) -> Any:
    """Convert pandas/numpy scalar step values into plain Python types for clean serialization."""
    return step.item() if hasattr(step, "item") else step


def summarize_scalar_curve(
    df: pd.DataFrame,
    key: str,
    higher_is_better: bool | None = None,
    smoothing: float = 0.0,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    mode: str | None = None,
) -> dict[str, Any]:
    """
    Compute descriptive statistics for one scalar curve in a dataframe.

    Args:
        df: DataFrame whose index is training step and whose columns are scalar tags.
        key: Scalar tag / column name to summarize.
        higher_is_better: Whether larger values are preferable. Retained for ergonomic compatibility.
        smoothing: Exponential moving average factor in [0, 1). 0 means no smoothing.
        quantile_low: Lower quantile used for robust range estimation.
        quantile_high: Upper quantile used for robust range estimation.
        mode: Optional optimization direction string. Accepted aliases include
            {"higher", "max", "maximize"} and {"lower", "min", "minimize"}.
            If provided, it overrides higher_is_better.

    Returns:
        A dictionary containing the raw and smoothed curve statistics, including
        start/end values, extrema, trend change points, robust range, and an
        oscillation score based on first differences.
    """
    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError("Require 0 <= quantile_low < quantile_high <= 1.")

    optimization_mode = _normalize_opt_mode(higher_is_better=higher_is_better, mode=mode)
    raw_series = _series_from_dataframe(df=df, key=key)
    series = _smooth_series(raw_series, smoothing=smoothing)

    initial_value = float(series.iloc[0])
    final_value = float(series.iloc[-1])
    max_value = float(series.max())
    min_value = float(series.min())
    max_step = _step_to_python(series.idxmax())
    min_step = _step_to_python(series.idxmin())
    delta = final_value - initial_value
    increased = delta > 0
    decreased = delta < 0

    # Interpret "improvement" according to optimization direction so the caller
    # can use the same API for loss-like and accuracy-like metrics.
    improvement = delta if optimization_mode == "higher" else -delta
    worsened = improvement < 0

    diff = series.diff().dropna()
    start_rise_step = None
    start_fall_step = None
    if not diff.empty:
        positive_diff = diff[diff > 0]
        negative_diff = diff[diff < 0]
        if not positive_diff.empty:
            start_rise_step = _step_to_python(positive_diff.index[0])
        if not negative_diff.empty:
            start_fall_step = _step_to_python(negative_diff.index[0])

    q_low_value = float(series.quantile(quantile_low))
    q_high_value = float(series.quantile(quantile_high))
    robust_range = q_high_value - q_low_value
    absolute_range = max_value - min_value

    if diff.empty:
        oscillation = 0.0
        normalized_oscillation = 0.0
        sign_change_count = 0
    else:
        # Oscillation is measured as the total variation after smoothing. The
        # normalized version makes curves with different scales comparable.
        oscillation = float(diff.abs().mean())
        normalized_oscillation = float(oscillation / (robust_range + 1e-12))
        sign_change_count = int((diff.shift(1) * diff < 0).sum())

    best_step = max_step if optimization_mode == "higher" else min_step
    best_value = max_value if optimization_mode == "higher" else min_value
    worst_step = min_step if optimization_mode == "higher" else max_step
    worst_value = min_value if optimization_mode == "higher" else max_value

    return {
        "key": key,
        "mode": optimization_mode,
        "count": int(series.shape[0]),
        "smoothing": float(smoothing),
        "quantile_low": float(quantile_low),
        "quantile_high": float(quantile_high),
        "initial_value": initial_value,
        "final_value": final_value,
        "max_value": max_value,
        "min_value": min_value,
        "best_value": best_value,
        "worst_value": worst_value,
        "delta": float(delta),
        "improvement": float(improvement),
        "increased": bool(increased),
        "decreased": bool(decreased),
        "worsened": bool(worsened),
        "start_step": _step_to_python(series.index[0]),
        "end_step": _step_to_python(series.index[-1]),
        "max_step": max_step,
        "min_step": min_step,
        "best_step": best_step,
        "worst_step": worst_step,
        "start_rise_step": start_rise_step,
        "start_fall_step": start_fall_step,
        "quantile_low_value": q_low_value,
        "quantile_high_value": q_high_value,
        "robust_range": float(robust_range),
        "absolute_range": float(absolute_range),
        "oscillation": float(oscillation),
        "normalized_oscillation": float(normalized_oscillation),
        "sign_change_count": int(sign_change_count),
    }


def summarize_scalar_curve_from_event(
    event_path: str,
    key: str,
    higher_is_better: bool | None = None,
    smoothing: float = 0.0,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    mode: str | None = None,
) -> dict[str, Any]:
    """Load an event file into a dataframe and summarize one scalar curve."""
    df = event2dataframe(event_path)
    return summarize_scalar_curve(
        df=df,
        key=key,
        higher_is_better=higher_is_better,
        smoothing=smoothing,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        mode=mode,
    )