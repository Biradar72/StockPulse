# 2. Code Walkthrough

File by file, in the order the data flows.

---

## `config/`

### `config/settings.py`
Defines every path (`DATA_DIR`, `PRICE_DIR`, `NEWS_CSV_DIR`, `MODEL_DIR`,
`RESULT_DIR`, …) relative to the project root, and creates the folders on
import.

### `config/stocks.py`
`BSE_STOCKS`: a list of 25 `(ticker, company name, sector)` tuples. Every
runner loops over this list.

---

## `collection/` (getting raw data)

### `collection/price_data.py` *(added)*
* `download(symbol, years)` wraps `yf.download` and flattens yfinance's
  MultiIndex columns.
* `download_stock(ticker, years)` tries `X.BO`, then `X.NS`, then any alias
  (e.g. `TATAMOTORS.BO → TMPV.BO`). It keeps the first result with at least
  500 rows and saves `Date, Open, High, Low, Close, Volume`.
* `download_macro(years)` saves the Close of 8 macro series under the file
  names that `macro_data.py` expects.

### `collection/google_news.py`
* Builds 9 search queries per company (for example `"Infosys" earnings results
  profit`) and reads the Google News RSS feed for each one.
* De-duplicates by a hash of the title.
* `_parse_date` converts the RSS UTC timestamp to Indian time (fixed; see doc 4).
* `save_news` appends to the CSV cache and de-duplicates.
* **Limitation:** Google News RSS returns only about 100 recent articles per
  query. For TCS, 306 of 370 articles were from the last year, so there is
  effectively no news history for the training years.

### `collection/macro_data.py`
`load_macro_data(start, end)` reads each macro CSV and then, **per series on
its own trading calendar**:
1. computes 1-, 5-, and 20-day returns;
2. **lags by one day** the series whose day-*t* close happens after Indian
   market close (S&P 500, crude, gold, USD/INR), to avoid look-ahead;
3. outer-merges all series on Date and forward-fills market holidays.

`MACRO_LEVEL_COLUMNS` lists the raw index levels. These drift over years and
are excluded from the model; only their returns are used.

---

## `processing/` (feature engineering)

### `processing/technical.py`: `add_technical(df)`
Takes the OHLCV frame and adds about 130 columns, using **only current and past
data** (rolling windows, `shift(+n)`):

| Group | Examples | Meaning |
|-------|----------|---------|
| Returns | `ret_1d … ret_60d`, `ret_lag1..5` | % change over *n* days; yesterday's return, etc. |
| Moving averages | `sma_*`, `ema_*` | Smoothed price *(level: not a model input)* |
| Price/MA ratios | `price_sma_ratio_20`, `sma_50_200_ratio` | Where the price sits relative to its trend (scale-free) |
| Volatility | `vol_5 … vol_60`, `atr_pct_14`, `bb_width` | How much the price is moving |
| Oscillators | `rsi_7/14/21`, `stoch_k/d`, `williams_r` | Overbought or oversold |
| MACD | `macd_pct`, `macd_hist_pct` | Trend momentum, as a % of price |
| Candles | `body_pct`, `upper_shadow`, `gap`, `close_location` | Shape of today's bar |
| Volume | `vol_z`, `volume_ratio_20`, `log_volume_ratio_5_50` | Unusual trading activity |
| Momentum / range | `momentum_*`, `distance_high_*`, `drawdown_60` | Distance from recent highs and lows |
| Regime *(added)* | `vol_ratio_5_20`, `ret_skew_20`, `up_days_20`, `ret_1d_z` | Volatility regime and return normalised by volatility |
| Calendar *(added)* | `day_of_week`, `month` | Weekday and month seasonality |
| Binary flags | `close_above_sma_200`, `macd_positive` | Simple yes/no trend states |

`PRICE_LEVEL_COLUMNS` lists the columns measured in rupees or share counts.
They stay in the DataFrame (the indicator-agreement rule uses `sma_20` and
`sma_50`) but are never fed to the model.

### `processing/news.py`
* `prepare_news` cleans the text, scores each article with **VADER** sentiment
  (−1 to +1), and tags it with categories (earnings, management, business,
  regulatory, market, macro) using keyword matching.
* `build_daily_news_features` runs once per trading day. It takes articles
  published before 15:30 IST that day and within the last `NEWS_LOOKBACK_DAYS`
  (30, added), then weights each article by:
  * **category weight** (earnings 1.0 … macro 0.65),
  * **exponential decay** `0.5 ** (days_old / half_life)`, where the half-life
    ranges from 3 days (market news) to 10 days (business news),
  * a **volatility gate** that dampens news in already-volatile periods.
  It outputs `overall_sent`, `news_count`, `news_impact`, and per-category
  sentiment and weight.
* `build_news_features` loops over every price date and returns one row per
  date.

---

## `modeling/` (dataset, training, evaluation)

### `modeling/pipeline.py`
* `load_price_data` / `load_news_data` read the CSVs and coerce types.
* `create_target(df, horizon)` is the **only** place future data is touched:
  `Next_Close = Close.shift(-h)` and `Target = Next_Close > Close`.
  `Future_Return_H` is stored for the training dead-zone filter and is
  excluded from features. The last `h` rows (which have no future) are dropped.
* `add_market_features` *(added)*: `excess_ret_{1,5,20}d` (stock minus NIFTY),
  60-day `beta_60`, `corr_market_60`, and `excess_vol_20`.
* `build_dataset` chains: price → technical → news → macro → market features
  → target.
* `run(ticker, …)`:
  1. `prepare_features` builds X and y, then runs the leakage check again.
  2. Drops warm-up rows (more than 10% of features missing).
  3. Splits chronologically into train / calibration / test, with an embargo
     gap when `TARGET_HORIZON > 1`.
  4. Optionally applies the training dead zone.
  5. `train_model(...)`.
  6. Saves `models/{T}_model.joblib` and `results/{T}_result.json`.

### `modeling/model.py`
Configuration constants at the top: `TEST_RATIO`, `CALIBRATION_RATIO`,
`CONFIDENCE_THRESHOLD`, `AGREEMENT_THRESHOLD`, `MAX_FEATURES`,
`TARGET_HORIZON`, `TRAIN_DEAD_ZONE`, `COVERAGE_LEVELS`.

| Function | What it does |
|----------|--------------|
| `check_for_leakage` | Raises an error if a forbidden column name (`target`, `next_close`, …) is in X |
| `prepare_features` | Drops Date/OHLCV/targets/price levels/macro levels and keeps numeric columns |
| `correlation_pruning` | Drops one of each pair of features with \|corr\| > 0.95 (fitted on train only) |
| `select_features` | Median-imputes, then keeps the top 50 features by mutual information with the target (train only) |
| `build_models` | XGBoost, LightGBM, RandomForest, sklearn GradientBoosting, and LogisticRegression, all heavily regularised |
| `MODEL_WEIGHTS` | Soft-vote weights: xgb .28, lgbm .28, rf .18, gb .14, lr .12 |
| `indicator_agreement` | Share of 10 classic indicators that are bullish (0 to 1) |
| `PlattCalibrator` / `calibrate_probabilities` | Fits a logistic curve on the calibration block to map ensemble scores to probabilities |
| `coverage_accuracy` | Accuracy on the top 10/20/30/50% most confident test days; the cut-off is chosen on the calibration block |
| `train_model` | Runs everything above and computes the metrics and signals |

### `modeling/experiments.py` *(added)*
The walk-forward research harness. It builds all 25 datasets once, then for
each configuration trains on everything before a one-year test block, predicts
that block, rolls forward 5 times, and reports pooled out-of-sample metrics.
This is the most reliable accuracy number in the project. See doc 5.

### `modeling/fit_diagnostics.py`
Only checks whether `data/05_final_dataset/*_final_dataset.csv` files exist.
The current pipeline never writes those files, so this script is a leftover.

---

## Runners and app

* `run_25_stocks.py` calls `pipeline.run` for each stock and writes
  `results/StockPulse_25_Stock_Results.csv`.
* `app/streamlit_app.py` shows that CSV as a table.
