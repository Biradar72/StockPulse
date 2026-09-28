"""
modeling/experiments.py
───────────────────────
Walk-forward experiments to find out which modelling choices
actually improve out-of-sample accuracy.

Every experiment uses the same protocol:
    - 5 folds; each fold's test block is ~1 year of later data
    - the model is trained only on data BEFORE the test block
      (minus an embargo of `horizon` days)
    - accuracy / ROC-AUC are averaged over all out-of-sample days

Run:
    python -m modeling.experiments
"""
import sys
import warnings

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import collection.macro_data as macro_data
from config.stocks import BSE_STOCKS
from config.settings import RESULT_DIR
from modeling.pipeline import build_dataset
from modeling.model import prepare_features

warnings.filterwarnings("ignore")

N_FOLDS = 5
TEST_DAYS = 245          # ~1 trading year per fold

# Features ranked across the 25 stocks on each date (pooled model)
CROSS_SECTIONAL = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "vol_20",
    "rsi_14",
    "volume_ratio_20",
    "excess_ret_5d",
    "excess_ret_20d",
    "drawdown_60",
    "next_open_gap",     # intraday target only
]


def make_model():
    return LGBMClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.02,
        num_leaves=8,
        min_child_samples=100,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


# ============================================================
# DATA
# ============================================================

def load_all(quiet=True, lag_us=True):
    """
    Build the base dataset for every stock once.

    lag_us=False keeps US-session series (S&P 500, crude, gold,
    USD/INR) on their own date. That is look-ahead for a
    close-to-close target, but legitimate for the intraday target,
    where the decision is made at the next morning's open.
    """
    lagged = set(macro_data.CLOSES_AFTER_INDIA)
    if not lag_us:
        macro_data.CLOSES_AFTER_INDIA.clear()
    try:
        return _load_all(quiet)
    finally:
        macro_data.CLOSES_AFTER_INDIA.update(lagged)


def _load_all(quiet):
    frames = {}
    for ticker, company, sector in BSE_STOCKS:
        stdout = sys.stdout
        if quiet:
            sys.stdout = open("/dev/null", "w")
        try:
            df = build_dataset(ticker, company, sector)
        finally:
            if quiet:
                sys.stdout.close()
                sys.stdout = stdout
        df["ticker"] = ticker
        frames[ticker] = df
        print(f"[DATA] {ticker}: {len(df)} rows")
    return frames


def with_target(df, horizon):
    """Re-create the target for a given horizon from Close."""
    df = df.copy()
    fwd = df["Close"].shift(-horizon) / df["Close"] - 1
    df["Future_Return_H"] = fwd
    df["Target"] = (fwd > 0).astype(int)
    df = df.iloc[:-horizon] if horizon > 0 else df
    return df


def intraday_target(df):
    """
    Row t = decision at the OPEN of day t+1 (09:15 IST).

    Known at that moment: everything up to close t, US day-t
    returns (closed at ~01:30 IST), and the opening gap of t+1.
    Target: Close(t+1) > Open(t+1)  (enter at open, exit at close).
    """
    df = df.copy()
    next_open = df["Open"].shift(-1)
    next_close = df["Close"].shift(-1)
    df["next_open_gap"] = next_open / df["Close"] - 1
    df["next_open_gap_z"] = df["next_open_gap"] / (df["vol_20"] + 1e-9)
    df["Future_Return_H"] = next_close / next_open - 1
    df["Target"] = (next_close > next_open).astype(int)
    return df.iloc[:-1]


def add_cross_sectional(panel):
    """Percentile rank of selected features across stocks per date."""
    cols = [c for c in CROSS_SECTIONAL if c in panel.columns]
    ranks = panel.groupby("Date")[cols].rank(pct=True)
    ranks.columns = [f"cs_rank_{c}" for c in cols]
    return pd.concat([panel, ranks], axis=1)


def split_xy(df):
    stdout = sys.stdout
    sys.stdout = open("/dev/null", "w")
    try:
        X, y = prepare_features(df.drop(columns=["ticker"], errors="ignore"))
    finally:
        sys.stdout.close()
        sys.stdout = stdout
    keep = X.isna().mean(axis=1) <= 0.10
    return X[keep], y[keep], df.loc[keep.values]


# ============================================================
# WALK-FORWARD
# ============================================================

def fold_bounds(dates):
    """Return (train_end_date, test_start_date, test_end_date) per fold."""
    unique = np.sort(pd.unique(dates))
    bounds = []
    for k in range(N_FOLDS, 0, -1):
        test_end = len(unique) - (k - 1) * TEST_DAYS
        test_start = test_end - TEST_DAYS
        bounds.append((unique[test_start], unique[test_end - 1]))
    return unique, bounds


def walk_forward(X, y, rows, horizon, dead_zone=0.0):
    """Generic walk-forward over a (possibly pooled) dataset."""
    dates = rows["Date"].values
    unique, bounds = fold_bounds(dates)
    preds, probs, actual, tickers = [], [], [], []

    for test_start, test_end in bounds:
        start_idx = np.searchsorted(unique, test_start)
        embargo_date = unique[max(start_idx - horizon, 0)]

        train = dates < embargo_date
        test = (dates >= test_start) & (dates <= test_end)

        if dead_zone > 0:
            move = rows["Future_Return_H"].abs().values
            thresh = dead_zone * rows["vol_20"].values * np.sqrt(horizon)
            train = train & (move >= thresh)

        model = make_model()
        model.fit(X.values[train], y.values[train])
        p = model.predict_proba(X.values[test])[:, 1]

        probs.append(p)
        preds.append((p >= 0.5).astype(int))
        actual.append(y.values[test])
        tickers.append(rows["ticker"].values[test])

    p = np.concatenate(probs)
    yhat = np.concatenate(preds)
    yt = np.concatenate(actual)
    tk = np.concatenate(tickers)

    conf = np.abs(p - 0.5)
    top20 = conf >= np.quantile(conf, 0.80)

    return {
        "accuracy": accuracy_score(yt, yhat),
        "always_up": yt.mean(),
        "roc_auc": roc_auc_score(yt, p),
        "acc_top20_conf": accuracy_score(yt[top20], yhat[top20]),
        "n_test": len(yt),
    }, pd.DataFrame({"ticker": tk, "y": yt, "pred": yhat, "prob": p})


def run_per_stock(frames, horizon, dead_zone=0.0, target_fn=None):
    target_fn = target_fn or (lambda df: with_target(df, horizon))
    out = []
    for df in frames.values():
        d = target_fn(df)
        X, y, rows = split_xy(d)
        _, pred = walk_forward(X, y, rows, horizon, dead_zone)
        out.append(pred)
    return summarise(pd.concat(out))


def run_pooled(frames, horizon, dead_zone=0.0, cross_sectional=True,
               target_fn=None):
    target_fn = target_fn or (lambda df: with_target(df, horizon))
    panel = pd.concat(
        [target_fn(df) for df in frames.values()],
        ignore_index=True,
    )
    if cross_sectional:
        panel = add_cross_sectional(panel)
    panel = panel.sort_values(["Date", "ticker"]).reset_index(drop=True)
    X, y, rows = split_xy(panel)
    _, pred = walk_forward(X, y, rows, horizon, dead_zone)
    return summarise(pred)


def summarise(pred):
    conf = np.abs(pred["prob"] - 0.5)
    top20 = conf >= conf.quantile(0.80)
    top10 = conf >= conf.quantile(0.90)
    return {
        "accuracy": accuracy_score(pred["y"], pred["pred"]),
        "always_up": pred["y"].mean(),
        "always_down": 1 - pred["y"].mean(),
        "roc_auc": roc_auc_score(pred["y"], pred["prob"]),
        "acc_top20_conf": accuracy_score(pred["y"][top20], pred["pred"][top20]),
        "acc_top10_conf": accuracy_score(pred["y"][top10], pred["pred"][top10]),
        "n_test": len(pred),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    frames = load_all()
    frames_open = load_all(lag_us=False)

    experiments = [
        ("per-stock  h=1",                 lambda: run_per_stock(frames, 1)),
        ("per-stock  h=1  dead-zone 0.3",  lambda: run_per_stock(frames, 1, 0.3)),
        ("pooled     h=1  (no cs-rank)",   lambda: run_pooled(frames, 1, cross_sectional=False)),
        ("pooled     h=1",                 lambda: run_pooled(frames, 1)),
        ("pooled     h=1  dead-zone 0.3",  lambda: run_pooled(frames, 1, 0.3)),
        ("per-stock  h=5",                 lambda: run_per_stock(frames, 5)),
        ("pooled     h=5",                 lambda: run_pooled(frames, 5)),
        ("pooled     h=5  dead-zone 0.3",  lambda: run_pooled(frames, 5, 0.3)),
        ("pooled     h=10",                lambda: run_pooled(frames, 10)),
        ("pooled     h=20",                lambda: run_pooled(frames, 20)),
        ("per-stock  intraday open->close", lambda: run_per_stock(frames_open, 1, target_fn=intraday_target)),
        ("pooled     intraday open->close", lambda: run_pooled(frames_open, 1, target_fn=intraday_target)),
    ]

    rows = []
    for name, fn in experiments:
        res = fn()
        res["experiment"] = name
        rows.append(res)
        print(
            f"{name:34s} acc={res['accuracy']:.4f} "
            f"always_up={res['always_up']:.4f} "
            f"always_down={res['always_down']:.4f} "
            f"auc={res['roc_auc']:.4f} "
            f"top20={res['acc_top20_conf']:.4f} "
            f"top10={res['acc_top10_conf']:.4f} "
            f"n={res['n_test']}",
            flush=True,
        )

    out = pd.DataFrame(rows).set_index("experiment")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULT_DIR / "experiments_walk_forward.csv")
    print(out.round(4).to_string())
