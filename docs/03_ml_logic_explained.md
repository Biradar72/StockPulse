# 3. The ML Logic, Explained

## 3.1 The prediction target

```
Target(t) = 1   if Close(t + h) > Close(t)
            0   otherwise
```

`h = TARGET_HORIZON` (default 1, meaning tomorrow). You stand at the close of
day *t*, know everything up to that moment, and must guess the sign of the
next move.

**Why this is hard:** a large-cap stock's daily return is roughly
"small drift + large random noise". Its daily standard deviation is about
1.5–2%, while its average daily drift is about 0.05%. The sign of tomorrow's
return is therefore dominated by the noise. Thousands of professional traders
compete to remove any predictable pattern. This is the *efficient market*
effect, and doc 5 shows it directly in the numbers.

## 3.2 Features: what the model sees

Every feature must be **known at the close of day t**. There are three families:

1. **Technical** (from the stock's own price and volume): trend
   (price/MA ratios), momentum (returns, RSI, MACD), volatility (ATR, rolling
   std), and volume anomalies.
2. **Market and macro**: NIFTY, Bank NIFTY, and Nikkei returns on day *t*;
   India VIX; and S&P 500, crude, gold, and USD/INR returns for day *t−1*
   (lagged because those markets close after India does). Also the stock's
   excess return over NIFTY and its beta.
3. **News**: VADER sentiment of recent headlines, weighted by category and
   time decay.

### Stationarity: the most important feature-engineering rule

A tree model learns rules like *"if sma_20 < 1800 then …"*. If sma_20 was
between 1,000 and 2,000 in the training years (2016–2023) and between 3,500
and 4,000 in the test year, every test row falls into one leaf, the leaf for
"bigger than anything I have seen". The model is then effectively constant.

So features must be **scale-free**:

| Bad (level) | Good (ratio) |
|-------------|--------------|
| `sma_20 = 3812.4` | `price_sma_ratio_20 = 1.024` |
| `macd = 41.3` | `macd_pct = 0.011` |
| `atr_14 = 55.0` | `atr_pct_14 = 0.014` |
| `nifty50 = 24500` | `nifty50_ret_1d = +0.4%` |

The original code fed about 30 level features to the model. They are now
excluded through `PRICE_LEVEL_COLUMNS` and `MACRO_LEVEL_COLUMNS`.

## 3.3 Chronological split and why random splits lie

```
|──────── TRAIN (≈70%) ────────|── CAL (10%) ──|──── TEST (20%) ────|
2016                                                             2026
```

With time series you must never shuffle. Neighbouring days share almost
identical indicator values (a 20-day moving average changes very little from
day to day). With a random split, the model sees 14 March in training and
15 March in test, and effectively memorises the answer. Shuffled
cross-validation on stock data routinely reports 60–70% accuracy that vanishes
in live trading.

With `h > 1`, the target of day *t* uses prices up to *t+h*, which overlaps
the next block. The pipeline leaves an **embargo** gap of `h−1` rows between
blocks.

## 3.4 Feature reduction

1. **Correlation pruning:** of any two features with |corr| > 0.95, one is
   dropped (for example `ret_5d` and `roc_5` are identical).
2. **Mutual information (SelectKBest, k=50):** keeps the 50 features that
   share the most information with the target on the training set.

Both are fitted **on the training block only**, so no test information leaks
into the choice of features.

## 3.5 The ensemble

| Model | Why it's included |
|-------|-------------------|
| XGBoost, LightGBM | Gradient-boosted trees, strong on tabular data |
| Random Forest | Averages many deep-ish trees; low variance |
| Gradient Boosting (sklearn) | Another boosted-tree variant, for diversity |
| Logistic Regression | Linear; robust when the signal is tiny |

Each outputs `P(UP)`. The final raw score is a **weighted average** (a soft
vote) using `MODEL_WEIGHTS`.

**Regularisation:** with a noisy target, a flexible model memorises noise.
Trees are now shallow (depth 2–3 for boosting), leaves need at least 20–50
samples, rows and columns are subsampled, and L1/L2 penalties are larger.

## 3.6 Calibration

A score of 0.53 from the ensemble does not automatically mean a 53% chance of
UP. **Calibration** fits a mapping `score → true probability` on the
*calibration block* (data the models never trained on):

* **Before:** isotonic regression, a step function. On about 240 rows it
  produced only a handful of distinct values, so often every test day landed
  on the same side of 0.5.
* **Now:** Platt scaling, a smooth logistic curve with 2 parameters. It is
  hard to overfit.

## 3.7 From probability to signal

```
confidence = max(p, 1 − p)
agreement  = share of 10 indicators that are bullish
             (ret_1d>0, Close>sma20, macd>0, macd_hist>0, rsi>50,
              stoch_k>50, williams_r>−50, roc_10>0, Close>sma50, gap>0)

UP        if p ≥ 0.5 and confidence ≥ 0.55 and agreement ≥ 0.58
DOWN      if p < 0.5 and confidence ≥ 0.55 and agreement ≤ 0.42   ← fixed
NO SIGNAL otherwise
```

The idea is to act only when the model is confident and the classic
indicators point the same way.

## 3.8 Metrics: how to read the results

| Metric | Meaning | "Random" value |
|--------|---------|----------------|
| `accuracy` | % of test days where the predicted direction was right | ~50% |
| `always_up_accuracy` | Accuracy of the dumb rule "always predict UP". **Your model must beat this.** | = share of up days |
| `roc_auc` | Ranking quality: is P(UP) higher on days that went up? | 0.50 |
| `acc_top20` | Accuracy on the 20% of days with the highest confidence | ~50% |
| `signal_accuracy` | Accuracy on days with an UP or DOWN signal | ~50% |
| `signal_count` | How many signals were issued (tiny counts, such as 1 signal at 100%, mean nothing) | |

**Statistical noise:** with *n* test days, the standard error of accuracy is
about `0.5/√n`.

* One stock, 480 test days: ±2.3%. A single stock at 54% is **not** evidence
  of skill.
* 25 stocks × 5 walk-forward years, about 30,000 days: ±0.3%. At this scale
  51% is a real (if small) difference.

That is why `modeling/experiments.py` exists.
