"""
eda_utils.py
============
Part 1 (cont.): I use this module for exploratory analysis and stationarity
testing on my target series (Appliances energy use, Wh).

Functions
---------
plot_series             -> I plot the raw time series
seasonal_decompose_plot -> I decompose it into trend / seasonal / residual
plot_acf_pacf           -> I plot ACF & PACF for diagnostics
run_adf_test            -> I run the Augmented Dickey-Fuller stationarity test
run_kpss_test           -> I run the KPSS stationarity test (complements ADF)
difference_series       -> I apply differencing until the series looks stationary
stationarity_report     -> I summarise every stationarity check in one table
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss


def plot_series(series: pd.Series, title: str, save_path: str | None = None) -> None:
    """I plot the raw time series and optionally save it as a figure."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series.values, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(series.name)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def seasonal_decompose_plot(
    series: pd.Series, period: int = 24, model: str = "additive", save_path: str | None = None
):
    """
    I decompose the series into trend, seasonal and residual components.

    Parameters
    ----------
    series : pd.Series
        My hourly target series.
    period : int
        Seasonal period in observations - I use 24 for daily seasonality
        on hourly data.
    model : str
        'additive' or 'multiplicative'. I use additive since I don't see
        the seasonal swings growing with the level of the series.

    Returns
    -------
    DecomposeResult
        statsmodels decomposition result object (trend/seasonal/resid attrs).
    """
    result = seasonal_decompose(series, model=model, period=period)
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return result


def plot_acf_pacf(series: pd.Series, lags: int = 72, save_path: str | None = None) -> None:
    """I plot ACF and PACF side by side to identify AR/MA order and seasonality."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_adf_test(series: pd.Series, verbose: bool = True) -> dict:
    """
    I run the Augmented Dickey-Fuller test.
    H0: the series has a unit root (non-stationary).
    If I reject H0 (p < 0.05), I treat the series as stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    out = {
        "test_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }
    if verbose:
        print("ADF Test")
        print(f"  Statistic : {out['test_statistic']:.4f}")
        print(f"  p-value   : {out['p_value']:.4f}")
        for k, v in out["critical_values"].items():
            print(f"  Critical value ({k}): {v:.4f}")
        print(f"  => {'Stationary' if out['is_stationary'] else 'Non-stationary'} at 5% level")
    return out


def run_kpss_test(series: pd.Series, regression: str = "c", verbose: bool = True) -> dict:
    """
    I run the KPSS test to complement ADF - it has the opposite null hypothesis.
    H0: the series IS stationary (around a constant/trend).
    If I reject H0 (p < 0.05), I treat the series as non-stationary.
    I use ADF + KPSS together so I'm not relying on a single test.
    """
    stat, p_value, n_lags, crit = kpss(series.dropna(), regression=regression, nlags="auto")
    out = {
        "test_statistic": stat,
        "p_value": p_value,
        "n_lags": n_lags,
        "critical_values": crit,
        "is_stationary": p_value > 0.05,
    }
    if verbose:
        print("KPSS Test")
        print(f"  Statistic : {out['test_statistic']:.4f}")
        print(f"  p-value   : {out['p_value']:.4f} (capped between 0.01-0.1 by statsmodels)")
        print(f"  => {'Stationary' if out['is_stationary'] else 'Non-stationary'} at 5% level")
    return out


def difference_series(series: pd.Series, seasonal_period: int | None = None) -> pd.Series:
    """
    I apply first-order (and optionally seasonal) differencing.

    Parameters
    ----------
    series : pd.Series
    seasonal_period : int, optional
        If I provide this, I also apply seasonal differencing at this lag
        (e.g. 24 for daily).

    Returns
    -------
    pd.Series
        Differenced series (shorter than the input by the differencing order).
    """
    diffed = series.diff().dropna()
    if seasonal_period:
        diffed = diffed.diff(seasonal_period).dropna()
    return diffed


def stationarity_report(series: pd.Series, seasonal_period: int = 24) -> pd.DataFrame:
    """
    I run ADF + KPSS on the raw series, then on first-differenced and
    seasonally-differenced versions, and summarise everything in one table.
    This is how I address the 'perform all the time series analysis tasks
    to test for non-stationarity' requirement in Part 1.
    """
    rows = []

    def _row(name, s):
        adf = run_adf_test(s, verbose=False)
        kp = run_kpss_test(s, verbose=False)
        rows.append({
            "series": name,
            "adf_stat": adf["test_statistic"],
            "adf_p": adf["p_value"],
            "adf_stationary": adf["is_stationary"],
            "kpss_stat": kp["test_statistic"],
            "kpss_p": kp["p_value"],
            "kpss_stationary": kp["is_stationary"],
        })

    _row("raw", series)
    _row("1st difference", difference_series(series))
    _row(f"seasonal diff (lag={seasonal_period})", series.diff(seasonal_period).dropna())
    _row(
        f"1st + seasonal diff (lag={seasonal_period})",
        difference_series(series, seasonal_period=seasonal_period),
    )

    return pd.DataFrame(rows)
