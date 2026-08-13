"""
evaluation.py
==============
Part 8: I use this module to evaluate all my models with common accuracy
metrics, forecast plots, error diagnostics, and a comparison against the
strongest benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true, y_pred, epsilon: float = 1e-6) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def smape(y_true, y_pred) -> float:
    """I use symmetric MAPE as well - it's more robust than MAPE when
    y_true can be near zero, which happens here since appliance usage
    can dip close to the baseline load."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1e-6, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def evaluate_all(y_true, forecasts: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    I score every model's forecast against the same true values.

    Parameters
    ----------
    y_true : array-like
        Actual observed values for the forecast horizon.
    forecasts : dict[str, array-like]
        model_name -> forecast array (same length as y_true).

    Returns
    -------
    pd.DataFrame
        One row per model, sorted by RMSE ascending (best first).
    """
    rows = []
    for name, y_pred in forecasts.items():
        rows.append({
            "model": name,
            "RMSE": rmse(y_true, y_pred),
            "MAE": mae(y_true, y_pred),
            "MAPE_%": mape(y_true, y_pred),
            "sMAPE_%": smape(y_true, y_pred),
        })
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def plot_forecast_comparison(
    y_true: pd.Series, forecasts: dict[str, np.ndarray], future_index, save_path: str | None = None
) -> None:
    """I overlay every model's forecast against the true held-out values."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(future_index, y_true, label="Actual", color="black", linewidth=2)
    for name, y_pred in forecasts.items():
        ax.plot(future_index, y_pred, label=name, linestyle="--", alpha=0.8)
    ax.set_title("24-hour forecast comparison across models")
    ax.set_xlabel("Time")
    ax.set_ylabel("Appliances energy use (Wh)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_error_diagnostics(y_true, forecasts: dict[str, np.ndarray], save_path: str | None = None) -> None:
    """I use this bar chart of RMSE per model as a quick 'which model wins' visual for my report."""
    scores = evaluate_all(y_true, forecasts)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(scores["model"], scores["RMSE"], color="steelblue")
    ax.set_ylabel("RMSE (Wh)")
    ax.set_title("Model comparison by RMSE (lower is better)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def strongest_benchmark(scores_df: pd.DataFrame, benchmark_names: list[str]) -> str:
    """I use this to identify my best-performing benchmark model, for the Part 9 Q1/Q2 comparison."""
    bench_scores = scores_df[scores_df["model"].isin(benchmark_names)]
    return bench_scores.sort_values("RMSE").iloc[0]["model"]


def walk_forward_evaluate(
    forecast_fn,
    full_series: pd.Series,
    test_start: pd.Timestamp,
    horizon: int = 24,
    step: int = 24,
) -> pd.DataFrame:
    """
    I use this for rolling-origin ("walk-forward") evaluation: I repeatedly
    forecast `horizon` steps ahead starting at each origin across the test
    period, then average the errors. This is how I reconcile the brief's
    '24 hour forecast horizon' requirement with its '14-day test period'
    requirement (Part 6) - rather than relying on a single 24h forecast, I
    get 14 (or however many `step`-sized windows fit) independent 24h
    forecasts and report the averaged metrics, which I think is far more
    robust than judging a model on one window.

    Parameters
    ----------
    forecast_fn : callable(history: pd.Series, horizon: int) -> np.ndarray
        Any of my benchmark functions, or a wrapper around my SARIMAX/ML/
        foundation model forecasts, with this same signature.
    full_series : pd.Series
        My complete target series (train + test).
    test_start : pd.Timestamp
        Timestamp where my test period begins (e.g. len(series) - 14 days).
    horizon : int
        Forecast horizon per origin (24 for me).
    step : int
        Spacing between successive forecast origins (I use 24 = non-
        overlapping daily windows across my 14-day test period).

    Returns
    -------
    pd.DataFrame
        One row per forecast origin with RMSE/MAE/MAPE for that window.
    """
    results = []
    origin = test_start
    while origin + pd.Timedelta(hours=horizon) <= full_series.index[-1] + pd.Timedelta(hours=1):
        history = full_series.loc[:origin - pd.Timedelta(hours=1)]
        y_true = full_series.loc[origin: origin + pd.Timedelta(hours=horizon - 1)]
        if len(y_true) < horizon:
            break
        y_pred = forecast_fn(history, horizon)
        results.append({
            "origin": origin,
            "RMSE": rmse(y_true.values, y_pred),
            "MAE": mae(y_true.values, y_pred),
            "MAPE_%": mape(y_true.values, y_pred),
        })
        origin += pd.Timedelta(hours=step)
    return pd.DataFrame(results)
