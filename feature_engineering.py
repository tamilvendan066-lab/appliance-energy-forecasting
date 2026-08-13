"""
feature_engineering.py
========================
Part 5: I use this module to add sensor, weather and time-based covariates
for my feature-based ML model (Part 6), and I could reuse it for SARIMAX
exogenous regressors too.

I'm careful throughout NOT to leak future information into a feature (see
Part 9, Q5) - my lag and rolling features only ever use PAST values
relative to the row they're attached to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    I add cyclical time-of-day and day-of-week features.

    I use cyclical (sin/cos) encoding to avoid the false discontinuity of
    raw integers - e.g. without this, hour 23 and hour 0 would look "far
    apart" to the model even though they're one hour apart in reality.
    """
    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    return out


def add_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    """
    I add lagged values of the target - these are always known at forecast
    origin (they're PAST values), so they're safe to use.

    Parameters
    ----------
    lags : list[int]
        I use e.g. [1, 2, 3, 24, 48, 168] - 1-3 hours ago, 1 & 2 days ago,
        and 1 week ago.
    """
    out = df.copy()
    for lag in lags:
        out[f"{target_col}_lag{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame, target_col: str, windows: list[int]
) -> pd.DataFrame:
    """
    I add rolling mean/std features. IMPORTANT: I shift the window by 1
    before rolling, so the feature for row t only ever uses data up to and
    including t-1 - I never let it see the current, unobserved-at-forecast-
    time value.
    """
    out = df.copy()
    shifted = out[target_col].shift(1)
    for w in windows:
        out[f"{target_col}_roll_mean{w}"] = shifted.rolling(window=w).mean()
        out[f"{target_col}_roll_std{w}"] = shifted.rolling(window=w).std()
    return out


def build_feature_set(
    df: pd.DataFrame,
    target_col: str = "Appliances",
    lags: list[int] = (1, 2, 3, 24, 48, 168),
    rolling_windows: list[int] = (3, 24, 168),
    weather_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    My full feature pipeline: time features + lags + rolling stats +
    selected sensor/weather columns already present in the dataset.

    Parameters
    ----------
    weather_cols : list[str], optional
        Existing columns I keep as-is, e.g.
        ["T_out", "RH_out", "Windspeed", "T1", "RH_1"].
        I default to a sensible subset if I don't provide this.

    Returns
    -------
    pd.DataFrame
        Feature matrix with the NaNs from my lag/rolling warm-up dropped.
    """
    if weather_cols is None:
        candidate_cols = ["T_out", "RH_out", "Windspeed", "Press_mm_hg", "T1", "RH_1"]
        weather_cols = [c for c in candidate_cols if c in df.columns]

    out = df[[target_col] + weather_cols].copy()
    out = add_time_features(out)
    out = add_lag_features(out, target_col, list(lags))
    out = add_rolling_features(out, target_col, list(rolling_windows))
    out = out.dropna()
    return out
