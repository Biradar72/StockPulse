import numpy as np
import pandas as pd


# ============================================================
# NON-STATIONARY COLUMNS
# ============================================================
#
# These columns are measured in rupees (or raw share counts),
# so their values drift with the price level. A tree trained on
# 2017 prices (e.g. sma_20 = 1500) cannot generalise to 2026
# prices (sma_20 = 4000). They stay in the DataFrame for
# indicator_agreement() and charts, but must never be model
# inputs. Their scale-free ratio versions are used instead.

PRICE_LEVEL_COLUMNS = (
    [f"sma_{n}" for n in [5, 10, 20, 50, 100, 200]]
    + [f"ema_{n}" for n in [5, 10, 12, 20, 26, 50, 100, 200]]
    + [f"atr_{n}" for n in [7, 14, 21]]
    + [f"volume_sma_{n}" for n in [5, 10, 20, 50]]
    + [
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_mid",
        "bb_upper",
        "bb_lower",
    ]
)


def add_technical(df):
    """
    Add technical indicators using only current/past market data.

    IMPORTANT:
    This function must NEVER create:
        Target
        target
        Next_Close
        next_close
        future_return
        future_price
        label

    The prediction target is created separately in pipeline.py.
    """

    x = df.copy()

    # ========================================================
    # BASIC PRICE SERIES
    # ========================================================

    c = pd.to_numeric(x["Close"], errors="coerce")
    h = pd.to_numeric(x["High"], errors="coerce")
    l = pd.to_numeric(x["Low"], errors="coerce")
    o = pd.to_numeric(x["Open"], errors="coerce")
    v = pd.to_numeric(x["Volume"], errors="coerce")

    # ========================================================
    # RETURNS
    # ========================================================

    x["ret_1d"] = c.pct_change()

    for n in [2, 3, 5, 10, 20, 30, 60]:
        x[f"ret_{n}d"] = c.pct_change(n)

    # ========================================================
    # SIMPLE MOVING AVERAGES
    # ========================================================

    for n in [5, 10, 20, 50, 100, 200]:
        x[f"sma_{n}"] = c.rolling(n).mean()

    # ========================================================
    # EXPONENTIAL MOVING AVERAGES
    # ========================================================

    for n in [5, 10, 12, 20, 26, 50, 100, 200]:
        x[f"ema_{n}"] = c.ewm(
            span=n,
            adjust=False
        ).mean()

    # ========================================================
    # PRICE / SMA RATIOS
    # ========================================================

    for n in [5, 10, 20, 50, 100, 200]:

        x[f"price_sma_ratio_{n}"] = (
            c / (x[f"sma_{n}"] + 1e-9)
        )

    # ========================================================
    # PRICE / EMA RATIOS
    # ========================================================

    for n in [5, 10, 12, 20, 26, 50, 100, 200]:

        x[f"price_ema_ratio_{n}"] = (
            c / (x[f"ema_{n}"] + 1e-9)
        )

    # ========================================================
    # VOLATILITY
    # ========================================================

    for n in [5, 10, 20, 30, 60]:

        x[f"vol_{n}"] = (
            x["ret_1d"]
            .rolling(n)
            .std()
        )

    # ========================================================
    # RSI
    # ========================================================

    delta = c.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    for n in [7, 14, 21]:

        avg_gain = gain.rolling(n).mean()
        avg_loss = loss.rolling(n).mean()

        rs = (
            avg_gain
            / (avg_loss + 1e-9)
        )

        x[f"rsi_{n}"] = (
            100
            - (
                100
                / (1 + rs)
            )
        )

    # ========================================================
    # MACD
    # ========================================================

    # These columns now definitely exist:
    # ema_12 and ema_26

    x["macd"] = (
        x["ema_12"]
        - x["ema_26"]
    )

    x["macd_signal"] = (
        x["macd"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    x["macd_hist"] = (
        x["macd"]
        - x["macd_signal"]
    )

    # MACD percentage
    x["macd_pct"] = (
        x["macd"]
        / (c + 1e-9)
    )

    # ========================================================
    # BOLLINGER BANDS
    # ========================================================

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()

    x["bb_mid"] = bb_mid

    x["bb_upper"] = (
        bb_mid
        + 2 * bb_std
    )

    x["bb_lower"] = (
        bb_mid
        - 2 * bb_std
    )

    x["bb_width"] = (
        (
            x["bb_upper"]
            - x["bb_lower"]
        )
        / (bb_mid + 1e-9)
    )

    x["bb_position"] = (
        (c - x["bb_lower"])
        /
        (
            x["bb_upper"]
            - x["bb_lower"]
            + 1e-9
        )
    )

    # ========================================================
    # STOCHASTIC OSCILLATOR
    # ========================================================

    lowest_14 = (
        l.rolling(14)
        .min()
    )

    highest_14 = (
        h.rolling(14)
        .max()
    )

    x["stoch_k"] = (
        100
        * (c - lowest_14)
        /
        (
            highest_14
            - lowest_14
            + 1e-9
        )
    )

    x["stoch_d"] = (
        x["stoch_k"]
        .rolling(3)
        .mean()
    )

    # ========================================================
    # WILLIAMS %R
    # ========================================================

    x["williams_r"] = (
        -100
        * (
            highest_14 - c
        )
        /
        (
            highest_14
            - lowest_14
            + 1e-9
        )
    )

    # ========================================================
    # RATE OF CHANGE
    # ========================================================

    for n in [5, 10, 20, 30, 60]:

        x[f"roc_{n}"] = (
            (c - c.shift(n))
            /
            (c.shift(n) + 1e-9)
        )

    # ========================================================
    # ATR
    # ========================================================

    prev_close = c.shift(1)

    tr1 = h - l

    tr2 = (
        h - prev_close
    ).abs()

    tr3 = (
        l - prev_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    for n in [7, 14, 21]:

        x[f"atr_{n}"] = (
            true_range
            .rolling(n)
            .mean()
        )

        x[f"atr_pct_{n}"] = (
            x[f"atr_{n}"]
            / (c + 1e-9)
        )

    # ========================================================
    # PRICE RANGE / CANDLE FEATURES
    # ========================================================

    x["intraday_range"] = (
        (h - l)
        / (c + 1e-9)
    )

    x["body_pct"] = (
        (c - o)
        / (o + 1e-9)
    )

    x["upper_shadow"] = (
        h
        - np.maximum(o, c)
    ) / (c + 1e-9)

    x["lower_shadow"] = (
        np.minimum(o, c)
        - l
    ) / (c + 1e-9)

    x["close_open_ratio"] = (
        c / (o + 1e-9)
    )

    # ========================================================
    # GAP
    # ========================================================

    x["gap"] = (
        o
        / (prev_close + 1e-9)
    ) - 1

    # ========================================================
    # VOLUME FEATURES
    # ========================================================

    for n in [5, 10, 20, 50]:

        x[f"volume_sma_{n}"] = (
            v.rolling(n).mean()
        )

    x["volume_change"] = (
        v.pct_change()
    )

    volume_mean_20 = (
        v.rolling(20).mean()
    )

    volume_std_20 = (
        v.rolling(20).std()
    )

    x["vol_z"] = (
        (v - volume_mean_20)
        /
        (volume_std_20 + 1e-9)
    )

    x["volume_ratio_20"] = (
        v
        /
        (volume_mean_20 + 1e-9)
    )

    # ========================================================
    # MOMENTUM
    # ========================================================

    for n in [5, 10, 20, 30, 60]:

        x[f"momentum_{n}"] = (
            c
            /
            (c.shift(n) + 1e-9)
        ) - 1

    # ========================================================
    # ROLLING HIGH / LOW
    # ========================================================

    for n in [5, 10, 20, 50]:

        rolling_high = (
            h.rolling(n).max()
        )

        rolling_low = (
            l.rolling(n).min()
        )

        x[f"distance_high_{n}"] = (
            c
            /
            (rolling_high + 1e-9)
        ) - 1

        x[f"distance_low_{n}"] = (
            c
            /
            (rolling_low + 1e-9)
        ) - 1

    # ========================================================
    # TREND RATIOS
    # ========================================================

    x["sma_5_20_ratio"] = (
        x["sma_5"]
        /
        (x["sma_20"] + 1e-9)
    )

    x["sma_20_50_ratio"] = (
        x["sma_20"]
        /
        (x["sma_50"] + 1e-9)
    )

    x["sma_50_200_ratio"] = (
        x["sma_50"]
        /
        (x["sma_200"] + 1e-9)
    )

    x["ema_12_26_ratio"] = (
        x["ema_12"]
        /
        (x["ema_26"] + 1e-9)
    )

    # ========================================================
    # Z-SCORES
    # ========================================================

    for n in [10, 20, 50]:

        mean_n = (
            c.rolling(n).mean()
        )

        std_n = (
            c.rolling(n).std()
        )

        x[f"price_zscore_{n}"] = (
            (c - mean_n)
            /
            (std_n + 1e-9)
        )

    # ========================================================
    # ADDITIONAL TREND FEATURES
    # ========================================================

    x["ema_5_20_ratio"] = (
        x["ema_5"]
        /
        (x["ema_20"] + 1e-9)
    )

    x["ema_20_50_ratio"] = (
        x["ema_20"]
        /
        (x["ema_50"] + 1e-9)
    )

    x["ema_50_200_ratio"] = (
        x["ema_50"]
        /
        (x["ema_200"] + 1e-9)
    )

    # ========================================================
    # DIRECTIONAL FEATURES
    # ========================================================

    x["close_above_sma_20"] = (
        c > x["sma_20"]
    ).astype(int)

    x["close_above_sma_50"] = (
        c > x["sma_50"]
    ).astype(int)

    x["close_above_sma_200"] = (
        c > x["sma_200"]
    ).astype(int)

    x["macd_positive"] = (
        x["macd"] > 0
    ).astype(int)

    x["rsi_above_50"] = (
        x["rsi_14"] > 50
    ).astype(int)

    # ========================================================
    # SCALE-FREE MACD
    # ========================================================

    x["macd_signal_pct"] = (
        x["macd_signal"]
        / (c + 1e-9)
    )

    x["macd_hist_pct"] = (
        x["macd_hist"]
        / (c + 1e-9)
    )

    # ========================================================
    # LAGGED RETURNS / SHORT-TERM REVERSAL
    # ========================================================

    for n in [1, 2, 3, 5]:

        x[f"ret_lag{n}"] = (
            x["ret_1d"].shift(n)
        )

    x["up_days_5"] = (
        (x["ret_1d"] > 0)
        .astype(float)
        .rolling(5)
        .mean()
    )

    x["up_days_20"] = (
        (x["ret_1d"] > 0)
        .astype(float)
        .rolling(20)
        .mean()
    )

    # Return normalised by its own recent volatility
    x["ret_1d_z"] = (
        x["ret_1d"]
        / (x["vol_20"] + 1e-9)
    )

    x["ret_5d_z"] = (
        x["ret_5d"]
        / (x["vol_20"] * np.sqrt(5) + 1e-9)
    )

    # ========================================================
    # CLOSE LOCATION / VOLATILITY REGIME
    # ========================================================

    x["close_location"] = (
        (c - l)
        / (h - l + 1e-9)
    )

    x["vol_ratio_5_20"] = (
        x["vol_5"]
        / (x["vol_20"] + 1e-9)
    )

    x["vol_ratio_20_60"] = (
        x["vol_20"]
        / (x["vol_60"] + 1e-9)
    )

    x["ret_skew_20"] = (
        x["ret_1d"]
        .rolling(20)
        .skew()
    )

    x["drawdown_60"] = (
        c
        / (c.rolling(60).max() + 1e-9)
    ) - 1

    x["log_volume_ratio_5_50"] = np.log(
        (x["volume_sma_5"] + 1)
        / (x["volume_sma_50"] + 1)
    )

    # ========================================================
    # CALENDAR
    # ========================================================

    if "Date" in x.columns:

        dates = pd.to_datetime(
            x["Date"],
            errors="coerce"
        )

        x["day_of_week"] = dates.dt.dayofweek

        x["month"] = dates.dt.month

    # ========================================================
    # CLEANUP
    # ========================================================

    # Defragment after adding columns one by one
    x = x.copy()

    x = x.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return x