# 5. Results and the Accuracy Reality Check

All numbers come from 10 years of daily data (Sept 2016 to Sept 2026) for the
25 stocks.

## 5.1 Single-split results (`run_25_stocks.py`)

Each stock is split 70% train / 10% calibration / 20% test (about 490 test
days per stock). The values below are means over the 25 stocks.

| Metric | Original code | After fixes |
|--------|---------------|-------------|
| Accuracy | 50.5% | 49.8% |
| "Always UP" baseline | 48.6% | 48.8% |
| ROC-AUC | 0.540 | 0.511 |
| Signal accuracy (weighted) | 50.8% (3,149 signals) | 50.6% (1,275 signals) |

Accuracy did not change; both runs are a coin flip. The original's higher
AUC came from look-ahead (section 5.3), not from skill.

## 5.2 Walk-forward experiments (`python -m modeling.experiments`)

These are the most trustworthy numbers in the project. There are 5 folds, and
each fold trains on all data before a one-year test block. Results are pooled
over about 30,600 out-of-sample stock-days (standard error about ±0.3%).
"Pooled" means one model trained on all 25 stocks together, plus
cross-sectional rank features.

| Experiment | Accuracy | Naive baseline* | ROC-AUC | Top-20% conf. | Top-10% conf. |
|---|---|---|---|---|---|
| per-stock, next day | 50.2% | 50.8% | 0.505 | 51.7% | 52.0% |
| per-stock, next day, dead-zone 0.3 | 50.7% | 50.8% | 0.507 | 51.6% | 51.9% |
| pooled, next day, no cross-sectional ranks | 50.3% | 50.8% | 0.506 | 52.2% | 53.2% |
| pooled, next day | 50.5% | 50.8% | 0.508 | 51.6% | 53.5% |
| pooled, next day, dead-zone 0.3 | 50.5% | 50.8% | 0.505 | 51.6% | 54.1% |
| per-stock, 5 days | 50.2% | 52.8% | 0.492 | 50.4% | 49.5% |
| pooled, 5 days | 50.3% | 52.8% | 0.480 | 51.5% | 52.0% |
| pooled, 10 days | 51.0% | 53.6% | 0.491 | 53.4% | 56.2% |
| pooled, 20 days | 53.1% | 54.3% | 0.529 | 57.3% | 62.4% |
| **per-stock, intraday open→close** | **53.5%** | 53.0% | **0.542** | **57.8%** | **60.1%** |
| **pooled, intraday open→close** | **54.7%** | 53.0% | **0.555** | **60.3%** | **61.5%** |

\*The naive baseline is the better of "always UP" and "always DOWN". A model
has to beat it to show any skill.

**How to read this:**

* **Close-to-close direction (1 to 20 days) never beats the naive baseline.**
  Longer horizons look better only because stocks rose more often than they
  fell over 2016–2026, so "always UP" scores 54% at 20 days.
* **The intraday target is the only setup with real, repeatable skill.**
  Details are in section 5.4.

## 5.3 Two sanity checks: is the harness itself working?

| Test | Result | Conclusion |
|------|--------|------------|
| Deliberate leak: future return added as a feature | **100.0%** accuracy, AUC 1.000 | The harness finds signal whenever signal exists |
| Volatility target: will tomorrow's move be bigger than usual? | 55.2% overall, 61.3% top-20% (AUC 0.572) | The same features and models do learn real structure |
| Same-day US data (the original, look-ahead version) | 53.6% (top-10%: 60.2%) | The original's apparent edge was look-ahead |
| US data lagged by one day (fixed) | 50.2% | With the look-ahead removed, the edge disappears |

The ~50% on next-day direction is therefore **not a bug**. It is what the
data contains.

## 5.4 The intraday result (the one real improvement)

The S&P 500's day-*t* session ends at about 01:30 IST, which is **before**
the Indian market opens on day *t+1* at 09:15. That information cannot be
used honestly for a close(t) → close(t+1) prediction, but it *can* be used for:

```
decision time : 09:15 IST, day t+1 (market open)
known         : everything up to close(t), US day-t returns,
                today's opening gap  Open(t+1) / Close(t) − 1
target        : Close(t+1) > Open(t+1)      (buy at open, sell at close)
```

It holds up in every walk-forward year (pooled model):

| Test year (fold) | Accuracy | Always DOWN | Top-20% conf. |
|------------------|----------|-------------|---------------|
| 1 | 53.8% | 52.0% | 58.1% |
| 2 | 54.3% | 52.1% | 60.8% |
| 3 | 55.6% | 53.6% | 62.1% |
| 4 | 55.3% | 53.7% | 61.5% |
| 5 | 54.6% | 53.6% | 59.7% |

The model is not simply calling DOWN. Among the top-20% most confident days,
its **UP** calls were right 57.4% of the time (while only 47% of days go up),
and its DOWN calls were right 60.7% of the time (53% base rate).

Caveats:
* The top-X% cut-off in the experiment harness is the quantile of the test
  predictions' confidence. In live use it must be fixed in advance (as
  `coverage_accuracy` in `model.py` does, using the calibration block).
* Trading costs (brokerage, STT, slippage) are not included. A 55–60% hit
  rate on intraday moves is not automatically profitable.
* This target is currently only in `modeling/experiments.py`. The production
  pipeline still predicts close-to-close (see doc 6).

## 5.5 Why "90% accuracy" is not a real target for daily stock direction

1. **Noise dominates.** The daily volatility of about 1.5–2% is roughly 30–40
   times the average daily drift. Even a perfect model of the *expected*
   return would get the sign right only a little more than half the time.
2. **Competition.** Any pattern that predicted the next day at 60% or better
   would be traded away by funds with far more data and faster execution.
3. **What credible results look like.** Published, out-of-sample results for
   daily direction of liquid stocks are typically **51–56%**. Top quant funds
   make money from edges of this size across thousands of trades, not from
   90% hit rates.
4. **When you see 80–99%, look for a leak.** Common causes:
   * the target (or a future price) in the features, as in this repo's old
     `models/*/feature_columns.json`, which lists `target`;
   * shuffled or random train/test splits on time series;
   * features that use `shift(-1)`, centred rolling windows, or data
     published after the decision time (like same-day US data here);
   * scaling or feature selection fitted on the whole dataset;
   * reporting accuracy on a handful of "signals" (for example "1 signal,
     100% correct").

A better way to measure success for this project:

| Goal | Realistic target |
|------|------------------|
| Next-day close-to-close accuracy | 51–53% is already good |
| Accuracy on the top 10–20% most confident days | 55–62% |
| ROC-AUC | 0.52–0.56 |
| Beats the naive baseline in *every* walk-forward year | Yes / No |
