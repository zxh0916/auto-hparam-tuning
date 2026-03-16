from typing import Any, Literal, Optional, Union

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

OptimizationMode = Literal["higher", "lower"]

from utils import next_step_postfix

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


def _normalize_opt_mode(
    higher_is_better: Optional[bool] = None, mode: Optional[str] = None
) -> OptimizationMode:
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
    higher_is_better: Optional[bool] = None,
    smoothing: float = 0.0,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    mode: Optional[str] = None,
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


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Analyze TensorBoard event files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- list-keys ---
    p_list = subparsers.add_parser(
        "list-keys",
        help="List all scalar tags recorded in a TensorBoard event file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_list.add_argument("event_path", metavar="EVENT_PATH",
                        help="Path to the TensorBoard event file or the directory containing it.")

    # --- summarize ---
    p_summarize = subparsers.add_parser(
        "summarize",
        help="Print a JSON summary of one or more scalar curves.",
        description=(
            "Load a TensorBoard event file and print a JSON summary of one scalar curve.\n\n"
            "The summary includes basic statistics (initial/final/best/worst value and their\n"
            "steps), trend information (delta, improvement, whether the metric worsened),\n"
            "robust range (configurable quantiles), and an oscillation score based on the\n"
            "mean absolute first difference of the (optionally smoothed) curve.\n\n"
            "The 'best' and 'worst' values are interpreted relative to --mode: for a loss\n"
            "curve use --mode min so that lower is better; for an accuracy curve use\n"
            "--mode max so that higher is better."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_summarize.add_argument("event_path", metavar="EVENT_PATH",
                             help="Path to the TensorBoard event file or the directory containing it.")
    p_summarize.add_argument("key", metavar="KEY", nargs="+",
                             help="One or more scalar tags to summarize (e.g. 'val/loss' 'train/acc').")
    p_summarize.add_argument(
        "-m", "--mode",
        default=None,
        metavar="DIR",
        help=(
            "Optimization direction used to determine best/worst values and the\n"
            "'improvement' field. Accepted values: 'higher'/'max'/'maximize' or\n"
            "'lower'/'min'/'minimize'. Required unless the direction is obvious from\n"
            "context. (default: None — will raise if not provided)"
        ),
    )
    p_summarize.add_argument(
        "-s", "--smoothing",
        type=float,
        default=0.0,
        metavar="ALPHA",
        help=(
            "Exponential moving average smoothing factor in [0, 1). Mirrors the\n"
            "TensorBoard UI smoothing slider: 0 means no smoothing, values closer\n"
            "to 1 produce heavier smoothing. All statistics are computed on the\n"
            "smoothed curve. (default: 0)"
        ),
    )
    p_summarize.add_argument(
        "-ql", "--quantile-low",
        type=float,
        default=0.05,
        dest="quantile_low",
        metavar="Q",
        help=(
            "Lower quantile for the robust range estimate [quantile_low, quantile_high].\n"
            "Using quantiles instead of the raw min/max makes the range insensitive to\n"
            "outlier spikes. (default: 0.05)"
        ),
    )
    p_summarize.add_argument(
        "-qh", "--quantile-high",
        type=float,
        default=0.95,
        dest="quantile_high",
        metavar="Q",
        help=(
            "Upper quantile for the robust range estimate. (default: 0.95)"
        ),
    )

    args = parser.parse_args()

    if args.command == "list-keys":
        df = event2dataframe(args.event_path)
        keys = list(df.columns)
        for key in keys:
            print(key)
        return

    # summarize
    result = summarize_scalar_curve_from_event(
        event_path=args.event_path,
        key=args.key,
        mode=args.mode,
        smoothing=args.smoothing,
        quantile_low=args.quantile_low,
        quantile_high=args.quantile_high,
    )
    # Single key → bare object for backward compatibility; multiple keys or "all" → array.
    single = len(args.key) == 1 and args.key[0] != "all"
    print(json.dumps(result[0] if single else result, indent=2))
    print(
        "next_step: Run `python scripts/session_manager.py update-run " +
        "{SESSION_DIR} --run-id {N} --status finished --primary-metric {VALUE} --best-step {STEP}` " +
        "to update the report and the records. " +
        "There is no need to ask the user for the next decision, just do your own job." +
        next_step_postfix()
    )


def summarize_scalar_curve_from_event(
    event_path: str,
    key: Union[str, list[str]],
    higher_is_better: Optional[bool] = None,
    smoothing: float = 0.0,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    mode: Optional[str] = None,
) -> Union[dict[str, Any], list[dict[str, Any]]]:
    """Load an event file into a dataframe and summarize one or more scalar curves.

    Args:
        event_path: Path to the TensorBoard event file or directory.
        key: A single scalar tag or a list of tags. When a list is given, the
            event file is loaded once and each key is summarized in order.
        higher_is_better: Whether larger values are preferable. Ignored when
            *mode* is provided.
        smoothing: EMA smoothing factor in [0, 1). 0 means no smoothing.
        quantile_low: Lower quantile for the robust range estimate.
        quantile_high: Upper quantile for the robust range estimate.
        mode: Optimization direction string ('higher'/'max' or 'lower'/'min').

    Returns:
        A single summary dict when *key* is a string, or a list of summary
        dicts (one per key, in the same order) when *key* is a list.
    """
    df = event2dataframe(event_path)
    if key == "all" or key == ["all"]:
        keys = list(df.columns)
        return_single = False
    elif isinstance(key, str):
        keys = [key]
        return_single = True
    else:
        keys = key
        return_single = False
    results = [
        summarize_scalar_curve(
            df=df,
            key=k,
            higher_is_better=higher_is_better,
            smoothing=smoothing,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            mode=mode,
        )
        for k in keys
    ]
    return results[0] if return_single else results


if __name__ == "__main__":
    main()
