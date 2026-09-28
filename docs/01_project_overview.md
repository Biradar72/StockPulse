# 1. Project Overview

## What StockPulse is

StockPulse is a **binary classification** system. For each of 25 large-cap
Indian stocks (listed in `config/stocks.py`, e.g. RELIANCE, TCS, HDFCBANK), it
answers one question every trading day:

> Will tomorrow's closing price be **higher** than today's closing price?

* `1` means UP (tomorrow's close is above today's close)
* `0` means DOWN or flat

One separate model is trained **per stock**. On top of the raw prediction, it
produces a trading-style **signal** (`UP`, `DOWN`, or `NO SIGNAL`). A signal is
only issued when the model is confident *and* classic technical indicators
agree.

## Data sources

| Source | Code | What it provides | Stored in |
|--------|------|------------------|-----------|
| Yahoo Finance (yfinance) | `collection/price_data.py` *(new)* | Daily Open/High/Low/Close/Volume for the 25 stocks, 10 years | `data/01_price_data/bse/{TICKER}.csv` |
| Yahoo Finance (yfinance) | `collection/price_data.py` *(new)* | Macro series: India VIX, NIFTY 50, NIFTY Bank, USD/INR, crude oil, gold, S&P 500, Nikkei | `data/06_macro_data/*.csv` |
| Google News RSS | `collection/google_news.py` | Headlines and summaries about each company | `data/02_news_data/csv/{TICKER}_news.csv` |

> The original repo had **no price downloader**. The pipeline expected the
> CSVs to exist but nothing created them. `collection/price_data.py` was added
> for this. Yahoo's BSE (`.BO`) history is broken for about half of these
> symbols (it returns only one row), so the downloader falls back to the NSE
> (`.NS`) listing of the same company. Tata Motors demerged in 2025 and now
> trades as `TMPV`.

## End-to-end data flow

```
                ┌──────────────────────┐
 yfinance ────► │ data/01_price_data   │──┐
                └──────────────────────┘  │
                                          ▼
                              processing/technical.py
                              (≈130 indicator columns)
                                          │
 Google News ─► data/02_news_data ─► processing/news.py ─┐
                                          │              │ merge on Date
 yfinance ────► data/06_macro_data ─► collection/macro_data.py
                                          │
                                          ▼
                      modeling/pipeline.py: build_dataset()
                        + add_market_features()  (excess return, beta)
                        + create_target()        (Target = close(t+h) > close(t))
                                          │
                                          ▼
                      chronological split
                   ┌───────────┬─────────────┬───────────┐
                   │  TRAIN    │ CALIBRATION │   TEST    │
                   │  ~70%     │    10%      │   20%     │
                   └───────────┴─────────────┴───────────┘
                                          │
                                          ▼
                      modeling/model.py: train_model()
                        correlation pruning → mutual-info feature selection
                        → 5 models (XGB, LGBM, RF, GB, LR)
                        → weighted soft vote
                        → Platt calibration
                        → prediction + signal (with indicator agreement)
                                          │
                                          ▼
                 models/{TICKER}_model.joblib     results/{TICKER}_result.json
                 results/StockPulse_25_Stock_Results.csv  ──► app/streamlit_app.py
```

## How to run

```bash
cd StockPulse
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt

# 1. Download prices + macro data (about 1 minute)
.venv/bin/python collection/price_data.py

# 2. (optional) Scrape news for a stock
.venv/bin/python collection/google_news.py --ticker TCS.BO \
    --company "Tata Consultancy Services" --sector IT

# 3. Train + evaluate all 25 stocks (single chronological split)
.venv/bin/python run_25_stocks.py

# 4. Robust walk-forward evaluation of modelling choices (about 10 minutes)
.venv/bin/python -m modeling.experiments

# 5. Dashboard
.venv/bin/streamlit run app/streamlit_app.py
```

Always run from inside `StockPulse/`. The code imports `config.*`,
`modeling.*`, and so on as top-level packages.

## Folder map

```
StockPulse/
├── config/          settings.py (paths), stocks.py (the 25 tickers)
├── collection/      data download: prices, macro, Google News
├── processing/      feature engineering: technical indicators, news sentiment
├── modeling/        pipeline (dataset + split), model (training + metrics),
│                    experiments (walk-forward), fit_diagnostics (dataset check)
├── app/             Streamlit results viewer
├── models/          saved models (*.joblib); the old */feature_columns.json
│                    folders are leftovers from an earlier, leaky version
├── results/         per-stock JSON + 25-stock CSV + experiments CSV
├── data/            downloaded data (git-ignored)
└── docs/            you are here
```
