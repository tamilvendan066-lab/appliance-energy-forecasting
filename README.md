# Time Series Case Study: Appliance Energy Use Forecasting

## What this is

This is my case study forecasting hourly appliance energy use, using the
[UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv).
I compared benchmark models, a SARIMA model, a feature-based XGBoost model,
and a zero-shot foundation model (Chronos) to see which one actually
forecasts appliance energy use best 24 hours ahead.

## How I've organised it

```
.
├── data_utils.py            Part 1 - I download, resample, and clean the data
├── eda_utils.py              Part 1 - my EDA + stationarity tests (ADF, KPSS, ACF/PACF)
├── benchmark_models.py       Part 3 - Mean, Naive, Seasonal Naive, Drift
├── sarimax_model.py          Part 4 - my SARIMA grid search + diagnostics
├── feature_engineering.py    Part 5 - lag/rolling/time/weather features I built
├── ml_model.py                Part 6 - my XGBoost feature-based model
├── foundation_model.py       Part 7 - Chronos zero-shot forecast
├── evaluation.py             Part 8 - metrics, comparison plots, walk-forward eval
├── main.py                    my orchestration script - runs everything end-to-end
├── generate_ci_figure.py     a fast add-on I wrote to produce just the SARIMA CI plot
├── requirements.txt
├── figures/                   my generated plots
├── outputs/                   my generated result CSVs
└── README.md
```

## How I run it

```bash
pip install -r requirements.txt
python main.py
```

Or I open `main.py` in Colab/Jupyter and run it cell-by-cell - I used
`# %%` markers to show where I'd naturally split it into cells.

**A heads-up on runtime:** my SARIMA grid search (Part 4) took a while on
hourly data even after I trimmed it down - budget 10-15 minutes for that
step. I capped it with `time_budget_seconds` so it can't run away on me.

**A heads-up on Part 7:** I needed `chronos-forecasting` + `torch` and
internet access to download the model weights from Hugging Face the first
time I ran it. If that's not available, my code falls back to a documented
placeholder so the pipeline still finishes - but I made sure to actually
run the real model before submitting, since Part 7 is separately graded.

## What I actually found

- **My target:** `Appliances` (Wh), resampled from 10-minute to hourly means.
- **My train/test split:** I held out the last 14 days as my test period.
- **My forecast horizon:** 24 hours, evaluated via walk-forward validation
  across the test period for more robust numbers than a single window.
- **Models I compared:** Mean, Naive, Seasonal Naive (daily + weekly),
  Drift, SARIMA, XGBoost, and Chronos.
- **My headline result:** SARIMA won by a clear margin, roughly halving
  the RMSE of my strongest benchmark. Full discussion is in my report.

## A couple of decisions I made and want to be upfront about

- I reduced my SARIMA grid search from the full p=[0,6], d=[0,2], q=[0,6]
  range down to p=[0,4], d=[0,1], q=[0,4] - a full search combined with a
  seasonal grid would have meant tens of thousands of model fits, which
  wasn't realistic on my machine. I based the reduction on my own ADF/KPSS
  results (which showed d=2 was very unlikely to help) and my PACF plot
  (which cut off sharply after lag 1-2). I explain this properly in my
  report rather than just doing it quietly.
- My ML model's recursive forecast carries the last observed weather value
  forward for the 24-hour horizon, rather than using real future weather -
  because in a genuine deployment I wouldn't have access to the actual
  future weather either. I flag this explicitly since it affects whether
  my forecasts count as truly unconditional (they do).
- I didn't end up adding exogenous regressors to my final SARIMA model
  (so it's really SARIMA, not SARIMAX, in my final result) - I discuss why
  and what I'd try next in my report.

## My coding approach

I tried to keep every transformation as a small, single-purpose function
with a docstring explaining what I did and why, rather than writing
everything as one long script. My feature engineering functions are
deliberately careful about not leaking future information - my lag and
rolling features only ever use data from before the row they're attached
to, since that's the only way to trust a backtest.

## Me

Tamilvendan Sathyamoorthy

