# 6. Next Steps

Ordered by expected value. Each step should be judged with
`python -m modeling.experiments` (walk-forward), not with a single split.

## 1. Promote the intraday target into the production pipeline
It is the only setup with repeatable skill (54.7%, top-20% 60.3%; doc 5,
section 5.4). Work needed:
* a `TARGET_MODE = "close_to_close" | "open_to_close"` switch in
  `modeling/model.py`;
* in `pipeline.py`, when in `open_to_close` mode, skip the US lag
  (`CLOSES_AFTER_INDIA`), add `next_open_gap`, and build the target from
  `Open(t+1)` and `Close(t+1)`;
* for live prediction, run the model at 09:15–09:20 IST once the opening
  price is known.

## 2. Switch the pipeline to one pooled model for all 25 stocks
The pooled model was consistently a little better than per-stock models
(+0.3 to +1.2 points), and it has 25 times more training rows. Add
cross-sectional ranks (`add_cross_sectional` in `experiments.py`).

## 3. Use walk-forward validation in the production runner
Replace the single 70/10/20 split in `pipeline.run` with the fold logic in
`experiments.walk_forward`, and report per-year accuracy against the naive
baseline.

## 4. Get real historical news
Google News RSS covers only recent months, so the news features are empty
during training. Options: GDELT (free, from 2015), a paid news API with
history, or collecting the RSS feed daily for 1–2 years before relying on it.
Consider FinBERT instead of VADER; VADER was built for social media, not
financial headlines.

## 5. Try other predictable targets
* **Volatility** (is tomorrow's move bigger than usual?) already reaches 55%,
  and 61% on the top 20% of days. It is useful for position sizing, stop
  distances, and options.
* **Relative performance:** will a stock beat the median of the 25 over the
  next 5 days? This removes the market-wide move, which is the noisiest part.

## 6. Evaluate as a strategy, not just as accuracy
Add a simple backtest: take a position only on the top-X% confidence days,
subtract about 0.05–0.1% per trade in costs, and report hit rate, average
return per trade, Sharpe ratio, and maximum drawdown. A model at 55% can be
profitable or unprofitable depending on how large its winning and losing
moves are.

## 7. Housekeeping
* Delete `models/*/feature_columns.json` (the old, leaky version).
* Remove or rewrite `modeling/fit_diagnostics.py`.
* Extend `app/streamlit_app.py` to show the walk-forward table and per-year
  accuracy.
