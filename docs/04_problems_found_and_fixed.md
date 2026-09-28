# 4. Problems Found and What Was Changed

Severity: 🔴 breaks results · 🟠 hurts accuracy or evaluation · 🟡 minor

## Data

### 🔴 1. No price data and no downloader
`pipeline.load_price_data` reads `data/01_price_data/bse/{T}.csv`, but no
script in the repo created those files, and the macro CSVs were also missing.
Nothing could run.

**Fix:** added `collection/price_data.py` (yfinance, 10 years, with an NSE
fallback and aliases). Yahoo returns only one row for about half of the `.BO`
symbols, and `TATAMOTORS.BO` no longer exists after the 2025 demerger (it now
maps to `TMPV`).

### 🟠 2. Google News covers almost only the test period
Google News RSS returns about 100 recent articles per query. For TCS, 306 of
370 articles were from 2026. News features are therefore zero for about 8
years of training data and suddenly non-zero in the test year. The model
cannot learn their meaning, and the shift adds noise.

**Status:** documented. The real fix is a historical news source (see doc 6).

### 🟠 3. `news_count` accumulated forever
`filter_market_cutoff` kept *every* article published before day *t*, so
`news_count` could only grow. It measured "how far into the dataset we are",
not "how much news there is".

**Fix:** only articles from the last `NEWS_LOOKBACK_DAYS = 30` are used.

### 🟡 4. News timestamps were UTC but treated as IST
RSS `published_parsed` is UTC, and the 15:30 cutoff is IST, so late-evening
articles were attributed to the wrong day.

**Fix:** `_parse_date` adds +5:30.

### 🟠 5. US market data had look-ahead
The S&P 500 (and crude, gold, and USD/INR bars) for date *t* close **after**
Indian market close on date *t*. Using them to predict `Close(t+1) > Close(t)`
at the close of day *t* uses information you would not have had.

**Fix:** these series are lagged by one of their own trading days
(`CLOSES_AFTER_INDIA`). Returns are also computed on each series' own
calendar before merging.

## Features

### 🔴 6. Non-stationary price-level features fed to the model
Raw `sma_*`, `ema_*`, `atr_*`, `bb_upper/lower/mid`, `macd`,
`volume_sma_*`, and raw index levels (`nifty50`, `sp500`, …) were model
inputs. Tree models cannot extrapolate to price levels they never saw (doc 3,
section 3.2).

**Fix:** `PRICE_LEVEL_COLUMNS` and `MACRO_LEVEL_COLUMNS` are excluded in
`prepare_features`. Scale-free replacements were added (`macd_signal_pct`,
`macd_hist_pct`, and others).

### 🟠 7. Warm-up rows were median-imputed
Rows 1–200 have no `sma_200` (and several other long-window features). They
were kept and median-filled, creating about 200 fake "average" rows.

**Fix:** rows with more than 10% missing features are dropped.

### 🟡 8. New features
Lagged returns, volatility-normalised returns, volatility-regime ratios,
return skew, 60-day drawdown, close location in the bar, day of week, month,
excess return vs NIFTY, beta, and correlation with the market.

## Model

### 🔴 9. DOWN signals required bullish indicators
```python
if confidence >= 0.55 and agreement >= 0.58:     # agreement = share bullish
    signal = "UP" if prob >= 0.5 else "DOWN"
```
A DOWN signal was only allowed when at least 58% of indicators were
**bullish**, so the rule contradicted itself.

**Fix:** UP needs `agreement ≥ 0.58`; DOWN needs `agreement ≤ 0.42`.

### 🟠 10. Isotonic calibration collapsed
On about 240 calibration rows, isotonic regression gave only a few output
levels. For TCS, the old run issued **1 signal in 493 days**, while other
stocks issued almost only UP signals (SUNPHARMA: 238 UP, 2 DOWN).

**Fix:** Platt scaling (`PlattCalibrator`, a class so joblib can save it).

### 🟠 11. Under-regularised models
Depth-4/5 boosting with 400 trees and depth-9 forests on about 1,700 noisy rows
memorise the training years.

**Fix:** shallower trees, larger minimum leaf sizes, stronger L1/L2, and more
subsampling.

### 🟡 12. Embargo for multi-day targets
With `TARGET_HORIZON > 1`, targets overlap across the split boundaries.

**Fix:** a gap of `h−1` rows between train/cal and cal/test.

## Evaluation

### 🔴 13. The old model artifacts contain target leakage
`models/*/feature_columns.json` (from an earlier version of this project)
lists **`"target"` as an input feature**. A model trained with the answer as
an input will show very high accuracy, and none of it is real. If you have
seen high accuracy numbers from this project before, this is the likely
reason. The current pipeline has explicit leakage guards, and a deliberate
leak test (doc 5) shows the harness catches it.

### 🟠 14. Single-split evaluation with no baseline
One stock's test block of about 490 days has ±2.3% noise, and no
"always UP" baseline was reported, so 52–54% numbers looked like skill.

**Fix:** `always_up_accuracy`, calibrated top-X% confidence accuracy, and a
walk-forward harness (`modeling/experiments.py`) over about 30,000
out-of-sample days.

### 🟡 15. Leftovers
* `modeling/fit_diagnostics.py` checks for `data/05_final_dataset` files that
  nothing produces.
* `modeling/pipeline.remove_leakage` only prints and removes nothing. It is
  harmless, because `prepare_features` does the real exclusion.
* `models/*/feature_columns.json` is from an older version. You can delete
  those folders.
