# StockPulse

Direction prediction for 25 large-cap Indian stocks (BSE/NSE), using
technical indicators, macro and market data, and Google News sentiment, with a
5-model ensemble (XGBoost, LightGBM, Random Forest, Gradient Boosting, and
Logistic Regression).

**Start with the documentation in [`docs/`](docs/README.md).** It explains
the logic, the code, the problems that were found and fixed, and honest
walk-forward results.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python collection/price_data.py            # prices + macro data (yfinance)
.venv/bin/python collection/google_news.py --ticker TCS.BO \
    --company "Tata Consultancy Services" --sector IT   # optional news
.venv/bin/python run_25_stocks.py                    # train + evaluate 25 stocks
.venv/bin/python -m modeling.experiments             # walk-forward experiments
.venv/bin/streamlit run app/streamlit_app.py         # dashboard
```

Run all commands from this folder.

## Results at a glance (walk-forward, ~30,600 out-of-sample days)

| Setup | Accuracy | Naive baseline | Top-20% confidence |
|-------|----------|----------------|--------------------|
| Next-day close → close | 50.2–50.5% | 50.8% | 51.6–52.2% |
| Intraday open → close (pooled) | **54.7%** | 53.0% | **60.3%** |

See [docs/05_results_and_accuracy_reality.md](docs/05_results_and_accuracy_reality.md).
