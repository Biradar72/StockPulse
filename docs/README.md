# StockPulse Documentation

Read these in order:

| # | Document | What you learn |
|---|----------|----------------|
| 1 | [01_project_overview.md](01_project_overview.md) | What the project does, the data flow, and how to run it |
| 2 | [02_code_walkthrough.md](02_code_walkthrough.md) | What every file and function does |
| 3 | [03_ml_logic_explained.md](03_ml_logic_explained.md) | Target, features, models, ensemble, calibration, signals, and metrics, explained from first principles |
| 4 | [04_problems_found_and_fixed.md](04_problems_found_and_fixed.md) | Every bug and design flaw found, why it hurts, and what was changed |
| 5 | [05_results_and_accuracy_reality.md](05_results_and_accuracy_reality.md) | Measured accuracy before and after, walk-forward experiments, and why "90% daily direction accuracy" is not a real target |
| 6 | [06_next_steps.md](06_next_steps.md) | Concrete options for making the system genuinely more useful |

**TL;DR**

* StockPulse tries to predict whether each of 25 large Indian stocks will close
  **higher tomorrow than today**, using technical indicators, macro and market
  data, and Google News sentiment.
* On 10 years of data, the original code scored **50.5%** mean accuracy, the
  same as a coin flip.
* Several real bugs were fixed: non-stationary features, a broken DOWN-signal
  rule, news features that only grew over time, US market data with look-ahead,
  calibration that collapsed into a few steps, a missing price downloader, and
  others. The evaluation is now honest and much more robust.
* Even after the fixes, next-day direction stays at **about 50–51%** in strict
  walk-forward testing. This is not a bug; it is how liquid stock markets
  behave. Sanity checks prove that the harness does find signal when it exists
  (a leak test gives 100%, and a volatility target gives 55–61%).
* The one genuine improvement: predicting **open → close** (decide at the
  09:15 market open, using overnight US moves and the opening gap) reaches
  **54.7%** accuracy (the naive baseline is 53.0%), and **60–62%** on the
  most confident 20% of days. It beats the baseline in all 5 walk-forward
  years (doc 5, section 5.4).
* Any system that reports 90% accuracy on next-day stock direction is almost
  certainly leaking future data. The *old* model files in this repo list
  `target` as an input feature, which is exactly that kind of leak.
