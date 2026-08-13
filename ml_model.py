"""
ml_model.py
============
Part 6: I use this module for my feature-based machine-learning model
(XGBoost, with a scikit-learn HistGradientBoostingRegressor fallback if I
don't have xgboost installed).

I forecast multiple steps ahead recursively: to forecast t+1..t+24 I
predict one step, feed that prediction back in to build the next row's lag
features, and repeat. I chose this because it mirrors how I'd actually
deploy the model - I don't have the true future lag values at forecast
time, so I shouldn't pretend I do during evaluation either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    _HAS_XGB = False

from feature_engineering import add_time_features, add_lag_features, add_rolling_features


def get_model(**kwargs):
    """I return an XGBRegressor if it's available, else I fall back to a HistGradientBoostingRegressor."""
    if _HAS_XGB:
        params = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, random_state=42)
        params.update(kwargs)
        return XGBRegressor(**params)
    else:
        print("I don't have xgboost installed - falling back to sklearn HistGradientBoostingRegressor. "
              "I'd run `pip install xgboost` to use the model the brief specifically names.")
        return HistGradientBoostingRegressor(max_depth=5, random_state=42)


def train_ml_model(feature_df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    """I fit my ML model on the full engineered training feature set."""
    X = feature_df[feature_cols]
    y = feature_df[target_col]
    model = get_model()
    model.fit(X, y)
    return model


def feature_importance(model, feature_cols: list[str]) -> pd.Series:
    """I return sorted feature importances - this helps me answer Part 9 Q3."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    raise AttributeError("Model does not expose feature_importances_")


def recursive_forecast(
    model,
    history_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    lags: list[int],
    rolling_windows: list[int],
    weather_cols: list[str],
    horizon: int,
    future_weather: pd.DataFrame | None = None,
) -> pd.Series:
    """
    I recursively forecast `horizon` steps ahead.

    Parameters
    ----------
    history_df : pd.DataFrame
        My full history (target + weather cols) up to the forecast origin.
    future_weather : pd.DataFrame, optional
        Known/forecast weather values for the horizon. If I don't have
        these, I carry the last observed weather values forward as a
        simplifying assumption - I state this explicitly in my report
        (see Part 9 Q5), since it affects whether this counts as a true
        unconditional forecast.

    Returns
    -------
    pd.Series
        Forecast values indexed by future timestamps.
    """
    working = history_df.copy()
    freq = pd.infer_freq(working.index) or "h"
    last_ts = working.index[-1]
    future_index = pd.date_range(last_ts, periods=horizon + 1, freq=freq)[1:]

    preds = []
    for i, ts in enumerate(future_index):
        # I append a new row for this timestep with weather values
        if future_weather is not None and ts in future_weather.index:
            new_row = future_weather.loc[[ts]].copy()
        else:
            new_row = working.iloc[[-1]][weather_cols].copy()
            new_row.index = [ts]
        new_row[target_col] = np.nan  # this is what I'm about to predict
        working = pd.concat([working, new_row])

        feat = add_time_features(working)
        feat = add_lag_features(feat, target_col, lags)
        feat = add_rolling_features(feat, target_col, rolling_windows)

        x_row = feat.loc[[ts], feature_cols]
        y_pred = model.predict(x_row)[0]
        preds.append(y_pred)
        working.loc[ts, target_col] = y_pred  # I feed my prediction back in so the next step's lags can use it

    return pd.Series(preds, index=future_index, name="forecast")
