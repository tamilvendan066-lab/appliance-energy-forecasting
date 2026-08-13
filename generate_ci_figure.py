"""
generate_ci_figure.py
=======================
A fast add-on I wrote: this generates ONLY the SARIMA confidence-interval
figure, reusing the SARIMA order I already found (order=(1,1,3),
seasonal_order=(1,1,1,24)) instead of rerunning my full ~10-minute grid
search again.

I run this after main.py has already run once (so data/energydata_complete.csv
exists locally):

    python generate_ci_figure.py

It produces: figures/07_sarima_forecast_ci.png
"""
import pandas as pd
from data_utils import prepare_dataset
from sarimax_model import fit_best_model, forecast_with_ci, plot_sarima_forecast_with_ci

TARGET = "Appliances"
DAILY_PERIOD = 24
HORIZON = 24
TEST_DAYS = 14

# I reuse the order my grid search already selected - I'd change these if
# my best AIC order came out differently in my own run.
BEST_ORDER = (1, 1, 3)
BEST_SEASONAL_ORDER = (1, 1, 1, DAILY_PERIOD)

print("Loading my cached dataset (should be instant - no redownload)...")
data = prepare_dataset()
target_series = data[TARGET]

test_start = target_series.index[-1] - pd.Timedelta(days=TEST_DAYS) + pd.Timedelta(hours=1)
train_series = target_series.loc[:test_start - pd.Timedelta(hours=1)]
test_series = target_series.loc[test_start:]

print(f"Refitting SARIMA{BEST_ORDER}x{BEST_SEASONAL_ORDER} (single fit, ~30-60s)...")
sarima_fit = fit_best_model(train_series, order=BEST_ORDER, seasonal_order=BEST_SEASONAL_ORDER)

sarima_forecast_df = forecast_with_ci(sarima_fit, steps=HORIZON)
sarima_forecast_df.index = test_series.index[:HORIZON]

plot_sarima_forecast_with_ci(
    test_series.iloc[:HORIZON], sarima_forecast_df,
    save_path="figures/07_sarima_forecast_ci.png",
)

ci_width = (sarima_forecast_df["ci_upper"] - sarima_forecast_df["ci_lower"]).mean()
print(f"\nDone. Saved figures/07_sarima_forecast_ci.png")
print(f"Mean 95% confidence interval width: {ci_width:.1f} Wh")
print("I'll add this figure + the CI width number to my report's SARIMA section.")
