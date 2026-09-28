"""
collection/price_data.py
────────────────────────
Downloads daily OHLCV price data for the 25 BSE stocks and the macro
series that collection/macro_data.py expects, using yfinance.

Output:
    data/01_price_data/bse/{TICKER}.csv     (Date, Open, High, Low, Close, Volume)
    data/06_macro_data/{name}.csv           (Date, Close)

Run:
    python collection/price_data.py                 # everything, 10 years
    python collection/price_data.py --years 15
    python collection/price_data.py --ticker TCS.BO
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import PRICE_DIR
from config.stocks import BSE_STOCKS

BSE_DIR = PRICE_DIR / "bse"
MACRO_DIR = ROOT / "data" / "06_macro_data"

# file name (as expected by macro_data.py) -> Yahoo symbol
MACRO_SYMBOLS = {
    "india_vix": "^INDIAVIX",
    "nifty50": "^NSEI",
    "nifty_bank": "^NSEBANK",
    "usd_inr": "INR=X",
    "crude_oil": "CL=F",
    "gold": "GC=F",
    "sp500": "^GSPC",
    "nikkei": "^N225",
}


def download(symbol, years):
    df = yf.download(
        symbol,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    # yfinance >= 0.2.40 returns MultiIndex columns (field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df


# Yahoo symbols renamed after corporate actions (Tata Motors demerger, 2025)
SYMBOL_ALIASES = {
    "TATAMOTORS.BO": ["TMPV.BO", "TMPV.NS"],
}

MIN_HISTORY_ROWS = 500


def download_stock(ticker, years):
    # Yahoo's .BO history is sometimes truncated to a single row;
    # the NSE listing of the same company is used as a fallback.
    candidates = [ticker, ticker.replace(".BO", ".NS")]
    candidates += SYMBOL_ALIASES.get(ticker, [])
    df, source = pd.DataFrame(), None
    for symbol in candidates:
        df = download(symbol, years)
        if len(df) >= MIN_HISTORY_ROWS:
            source = symbol
            break
        time.sleep(0.5)
    if source is None:
        print(f"[PRICE] {ticker}: no usable history from {candidates}")
        return None
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    # BSE sometimes reports zero-volume holidays / stale rows
    df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
    BSE_DIR.mkdir(parents=True, exist_ok=True)
    path = BSE_DIR / f"{ticker}.csv"
    df.to_csv(path, index=False)
    print(f"[PRICE] {ticker}: {len(df)} rows (source {source}) -> {path}")
    return path


def download_macro(years):
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    for name, symbol in MACRO_SYMBOLS.items():
        df = download(symbol, years)
        if df.empty:
            print(f"[MACRO] {name} ({symbol}): no data returned")
            continue
        path = MACRO_DIR / f"{name}.csv"
        df[["Date", "Close"]].to_csv(path, index=False)
        print(f"[MACRO] {name}: {len(df)} rows -> {path}")
        time.sleep(0.5)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--skip-macro", action="store_true")
    args = ap.parse_args()

    tickers = [args.ticker] if args.ticker else [t for t, _, _ in BSE_STOCKS]
    for t in tickers:
        try:
            download_stock(t, args.years)
        except Exception as e:
            print(f"[PRICE] {t}: failed ({e})")
        time.sleep(0.5)

    if not args.skip_macro:
        download_macro(args.years)
