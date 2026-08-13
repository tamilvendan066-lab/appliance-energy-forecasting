# [Your Name] — Time Series Case Study: Forecasting Appliance Energy Use
*(Target length: 6–8 pages incl. figures; references ≤ 0.5 page)*

Fill in every `[...]` once you have real numbers/plots from running
`main.py`. Section headers map directly onto the rubric so a marker can
tick boxes fast — keep them.

---

## 1. Introduction (~0.5 page)
- One paragraph: what the dataset is, why forecasting appliance energy use
  matters (smart-home energy management, demand response, grid planning).
- One paragraph: what this report does — models compared, headline finding.

## 2. Data & Exploratory Analysis (~1.5 pages)
- Describe the dataset: source, [N] observations at 10-min resolution,
  resampled to hourly ([N_hourly] points), date range [start]–[end].
- **Figure 1**: raw hourly series plot (`figures/01_raw_series.png`).
- **Figure 2**: seasonal decomposition (`figures/02_decomposition.png`) —
  describe the trend/seasonal/residual components you observe.
- Discuss: is there a clear daily seasonal component? Weekly? Comment on
  what drives the pattern (occupancy/work-day rhythm, etc.).
- **Figure 3**: ACF/PACF (`figures/03_acf_pacf.png`) — what lag structure
  do you see (e.g. spike at lag 24 confirming daily seasonality)?

## 3. Stationarity Analysis (~0.5–1 page)
- Report your ADF and KPSS results on the raw series (from
  `outputs/stationarity_report.csv`): statistic, p-value, conclusion.
- Report the same after 1st-order and seasonal differencing.
- State your conclusion: is the raw series stationary? What differencing
  order did you settle on for SARIMA, and why (cite the ADF/KPSS evidence)?
- *Do not just define ADF/KPSS from a textbook — interpret YOUR results.*

## 4. Forecasting Problem Definition (~0.3 page)
- Target variable: `Appliances` (Wh).
- Forecast horizon: 24 hours.
- Train/test split: last 14 days held out; walk-forward evaluation across
  14 non-overlapping 24h windows.
- Evaluation metrics: RMSE (primary — units match the target, easy to
  interpret in Wh), MAE, MAPE/sMAPE (for scale-independent comparison).
- Briefly justify why RMSE is the primary metric here (penalises large
  errors more, relevant since occasional appliance-use spikes are the
  hardest/most consequential events to forecast).

## 5. Modelling Methods (~1.5–2 pages)
For **each** model, 1 short paragraph: brief method overview (not a
textbook definition — why THIS method, applied to THIS problem):
- **Benchmarks** (Mean, Naive, Seasonal Naive daily/weekly, Drift): why
  these matter as a floor — any "real" model must beat them to be useful.
- **SARIMA(X)**: order selected [(p,d,q)(P,D,Q,m)] via AIC grid search
  (`outputs/sarima_grid_search.csv`), why this order made sense given your
  stationarity/ACF findings. Report residual diagnostics
  (`figures/04_sarima_residuals.png`, Ljung-Box result) — do residuals look
  like white noise? If not, what does that suggest about the model's fit?
- **XGBoost (feature-based)**: which features you engineered (lags,
  rolling stats, time-of-day, weather) and why. Report top-10 feature
  importances (`outputs/ml_feature_importance.csv`) — what does the
  ranking tell you about what actually drives appliance use?
- **Foundation model (Chronos)**: one paragraph on what a zero-shot
  time-series foundation model is and why it's an interesting comparison
  point (no training on this specific series at all).

## 6. Results (~1–1.5 pages)
- **Table 1**: model comparison table (`outputs/model_comparison_single_window.csv`
  or averaged walk-forward numbers) — RMSE/MAE/MAPE/sMAPE per model.
- **Figure 4**: forecast comparison overlay (`figures/05_forecast_comparison.png`).
- **Figure 5**: RMSE bar chart (`figures/06_rmse_comparison.png`).
- 1–2 paragraphs: which model won, by how much, and does that margin feel
  practically meaningful (not just numerically lower)?

## 7. Discussion & Answers to Set Questions (~1.5–2 pages)
Answer all 6 questions explicitly (use sub-headings so they're easy to find):

**Q1. Which benchmark model is strongest, and what does this tell you
about the structure of appliance energy use?**
`[Your answer, citing outputs/model_comparison table]` — e.g. if seasonal
naive (daily or weekly) beats naive/mean/drift, that indicates a strong
recurring daily/weekly rhythm rather than a trending or random-walk series.

**Q2. Does SARIMAX improve on the strongest seasonal benchmark? Are
daily seasonality, autocorrelation, and exogenous variables adequately
captured?**
`[Your answer]` — reference the residual diagnostics; if Ljung-Box still
rejects white noise, say explicitly that some structure remains uncaptured.

**Q3. Does XGBoost improve when lag/rolling/time-of-day/sensor features
are added? Which feature groups are most useful?**
`[Your answer]` — reference the feature importance table directly.

**Q4. Does the foundation model outperform the simpler benchmark,
SARIMAX, and feature-based models? Is any improvement large enough to
justify the extra complexity?**
`[Your answer]` — be honest here: zero-shot foundation models often do NOT
beat a well-tuned feature-based model on a single, well-understood series
with rich exogenous data — that's a legitimate and interesting finding,
not a failure of your work.

**Q5. Which variables would genuinely be known at the forecast origin? If
you used future weather values from the test set, is this a true forecast
or a conditional forecast?**
`[Your answer]` — be explicit: `main.py`'s ML recursive forecast carries
the LAST OBSERVED weather forward rather than using true future weather —
state whether you changed this, and if you used true future weather
anywhere, label those results as conditional (weather-aware) forecasts,
not genuine unconditional forecasts.

**Q6. Based on accuracy, interpretability, uncertainty, computational
cost and ease of deployment, which model would you recommend for
practical smart-home energy forecasting, and why?**
`[Your answer]` — weigh trade-offs explicitly, e.g. SARIMAX gives
interpretable coefficients + native confidence intervals but is slower to
refit; XGBoost is fast and accurate but needs feature engineering upkeep
and gives no native uncertainty; Chronos needs zero training but offers
no interpretability and heavier compute at inference.

## 8. Limitations & Future Improvements (~0.5 page)
- Data limitations (single household, ~4.5 months only, synthetic `rv1`/
  `rv2` columns provide no signal).
- Modelling limitations (weather-forward-fill assumption; SARIMA seasonal
  order fixed rather than fully grid-searched for tractability — state
  this as a documented decision, see `sarimax_model.py` docstring).
- Future work: probabilistic forecasts / prediction intervals for the ML
  and foundation models (only SARIMAX gives these natively here); ensembling
  models; multi-household generalisation; online/incremental retraining.
  Cite 1–2 relevant papers here (e.g. on hybrid statistical+ML forecasting,
  or on the specific foundation model you used).

## References
`[Author, Year, Title, Venue/URL]` — ≤ 0.5 page, consistent citation style
(e.g. IEEE or APA — check what your department expects).
