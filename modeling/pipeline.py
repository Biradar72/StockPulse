import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from config.settings import (
    PRICE_DIR,
    NEWS_CSV_DIR,
    MODEL_DIR,
    RESULT_DIR
)

from processing.technical import add_technical
from processing.news import build_news_features

from modeling.model import (
    prepare_features,
    train_model,
    MIN_SAMPLES,
    TEST_RATIO,
    CALIBRATION_RATIO,
    TARGET_HORIZON,
    TRAIN_DEAD_ZONE,
    MAX_MISSING_FRACTION
)


# ============================================================
# TARGET LEAKAGE PROTECTION
# ============================================================

LEAKAGE_NAMES = {
    "target",
    "Target",
    "next_close",
    "Next_Close",
    "future_close",
    "Future_Close",
    "future_return",
    "Future_Return",
    "future_price",
    "Future_Price",
    "label",
    "Label"
}


# ============================================================
# LOAD PRICE DATA
# ============================================================

def load_price_data(ticker):

    path = (
        PRICE_DIR
        / "bse"
        / f"{ticker}.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Price file not found: {path}"
        )

    df = pd.read_csv(
        path
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    required = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{ticker}: Missing price columns "
            f"{missing}"
        )

    for c in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    return df


# ============================================================
# LOAD NEWS
# ============================================================

def load_news_data(ticker):

    path = (
        NEWS_CSV_DIR
        / f"{ticker.replace('.', '_')}_news.csv"
    )

    if not path.exists():

        print(
            f"[WARN] News file not found: {path}"
        )

        return pd.DataFrame()

    news = pd.read_csv(
        path
    )

    if news.empty:
        return news

    if "date" in news.columns:

        news["date"] = pd.to_datetime(
            news["date"],
            errors="coerce"
        )

    news = news.dropna(
        subset=["date"]
    )

    return news


# ============================================================
# BUILD TARGET
# ============================================================

def create_target(
    df,
    horizon=TARGET_HORIZON
):

    df = df.copy()

    # ---------------------------------------------------------
    # FUTURE INFORMATION IS CREATED HERE ONLY
    # ---------------------------------------------------------

    df["Next_Close"] = (
        df["Close"].shift(-horizon)
    )

    df["Target"] = (
        df["Next_Close"]
        > df["Close"]
    ).astype(int)

    # Kept only for the training dead-zone filter;
    # excluded from features in prepare_features().
    df["Future_Return_H"] = (
        df["Next_Close"]
        / df["Close"]
    ) - 1

    # Last rows have no future close
    df = df.iloc[:-horizon].copy()

    return df


# ============================================================
# MARKET-RELATIVE FEATURES
# ============================================================

def add_market_features(df):

    """
    Relate the stock to the overall market (NIFTY 50). Most of a
    large-cap stock's daily move is the market's move; the part
    that is specific to the stock is more informative.
    """

    if "nifty50_ret_1d" not in df.columns:

        return df

    df = df.copy()

    market = df["nifty50_ret_1d"]

    for n in [1, 5, 20]:

        if (
            f"ret_{n}d" in df.columns
            and f"nifty50_ret_{n}d" in df.columns
        ):

            df[f"excess_ret_{n}d"] = (
                df[f"ret_{n}d"]
                - df[f"nifty50_ret_{n}d"]
            )

    cov = (
        df["ret_1d"]
        .rolling(60)
        .cov(market)
    )

    var = (
        market
        .rolling(60)
        .var()
    )

    df["beta_60"] = (
        cov
        / (var + 1e-12)
    )

    df["corr_market_60"] = (
        df["ret_1d"]
        .rolling(60)
        .corr(market)
    )

    df["excess_vol_20"] = (
        (df["ret_1d"] - market)
        .rolling(20)
        .std()
    )

    return df


# ============================================================
# REMOVE LEAKAGE
# ============================================================

def remove_leakage(df):

    found = [
        c
        for c in df.columns
        if c in LEAKAGE_NAMES
    ]

    if found:

        print(
            "[LEAKAGE CHECK] "
            f"Removing target columns from features: "
            f"{found}"
        )

    # We DO NOT delete Target here because
    # Target is required as y.
    #
    # We remove only leakage columns from
    # the feature matrix later.

    return df


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(
    ticker,
    company,
    sector
):

    print(
        "\n"
        "===================================================="
    )

    print(
        f"BUILDING DATASET: {ticker}"
    )

    print(
        "===================================================="
    )

    # ---------------------------------------------------------
    # Price
    # ---------------------------------------------------------

    df = load_price_data(
        ticker
    )

    print(
        f"[PRICE] {len(df)} rows"
    )

    # ---------------------------------------------------------
    # Technical indicators
    # ---------------------------------------------------------

    df = add_technical(
        df
    )

    print(
        f"[TECHNICAL] "
        f"{len(df.columns)} total columns"
    )

    # ---------------------------------------------------------
    # News
    # ---------------------------------------------------------

    try:

        news = load_news_data(
            ticker
        )

        if not news.empty:

            print(
                f"[NEWS] "
                f"{len(news)} news records"
            )

            news_features = (
                build_news_features(
                    df,
                    news,
                    ticker
                )
            )

            if (
                news_features is not None
                and not news_features.empty
            ):

                # Ensure Date exists
                if "Date" in news_features.columns:

                    news_features["Date"] = (
                        pd.to_datetime(
                            news_features["Date"],
                            errors="coerce"
                        )
                    )

                    df = df.merge(
                        news_features,
                        on="Date",
                        how="left"
                    )

                else:

                    print(
                        "[WARN] News features "
                        "do not contain Date."
                    )

        else:

            print(
                "[NEWS] No news data available."
            )

    except Exception as e:

        print(
            f"[WARN] News processing failed: {e}"
        )

    # ---------------------------------------------------------
    # Macro data
    # ---------------------------------------------------------

    try:

        from collection.macro_data import (
            load_macro_data
        )

        macro = load_macro_data(
            start_date=df["Date"].min(),
            end_date=df["Date"].max()
        )

        if (
            macro is not None
            and not macro.empty
        ):

            macro["Date"] = pd.to_datetime(
                macro["Date"],
                errors="coerce"
            )

            df = df.merge(
                macro,
                on="Date",
                how="left"
            )

            print(
                f"[MACRO] Added "
                f"{len(macro.columns) - 1} "
                f"macro features."
            )

        else:

            print(
                "[MACRO] No macro data returned."
            )

    except Exception as e:

        print(
            f"[WARN] Macro data skipped: {e}"
        )

    df = add_market_features(
        df
    )

    # ---------------------------------------------------------
    # Create target LAST
    # ---------------------------------------------------------

    df = create_target(
        df
    )

    # ---------------------------------------------------------
    # Leakage protection
    # ---------------------------------------------------------

    df = remove_leakage(
        df
    )

    # ---------------------------------------------------------
    # Remove unusable rows
    # ---------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.reset_index(
        drop=True
    )

    if len(df) < MIN_SAMPLES:

        raise ValueError(
            f"{ticker}: Only {len(df)} usable "
            f"samples. Minimum required = "
            f"{MIN_SAMPLES}"
        )

    print(
        f"[DATASET] "
        f"{len(df)} usable samples"
    )

    return df


# ============================================================
# RUN SINGLE STOCK
# ============================================================

def run(
    ticker,
    company,
    sector
):

    try:

        df = build_dataset(
            ticker,
            company,
            sector
        )

        # -----------------------------------------------------
        # Prepare X and y
        # -----------------------------------------------------

        X, y = prepare_features(
            df,
            target_col="Target"
        )

        # -----------------------------------------------------
        # FINAL LEAKAGE CHECK
        # -----------------------------------------------------

        leakage_found = [
            col
            for col in X.columns
            if (
                col in LEAKAGE_NAMES
                or col.lower()
                in {
                    "target",
                    "next_close",
                    "future_close",
                    "future_return",
                    "future_price",
                    "label"
                }
            )
        ]

        if leakage_found:

            raise ValueError(
                "\n"
                "====================================================\n"
                "STOPPING: DATA LEAKAGE DETECTED\n"
                "====================================================\n"
                f"{ticker}\n"
                f"Forbidden features: "
                f"{leakage_found}\n"
                "===================================================="
            )

        print(
            "[LEAKAGE CHECK] PASS - "
            "No future target columns in X."
        )

        # -----------------------------------------------------
        # Remove rows where features are completely unusable
        # -----------------------------------------------------

        # Drop the warm-up period (long moving averages not yet
        # defined) instead of median-imputing it.
        valid_mask = (
            X.isna().mean(axis=1)
            <= MAX_MISSING_FRACTION
        )

        X = X.loc[
            valid_mask
        ].reset_index(
            drop=True
        )

        y = y.loc[
            valid_mask
        ].reset_index(
            drop=True
        )

        df_model = df.loc[
            valid_mask
        ].reset_index(
            drop=True
        )

        # -----------------------------------------------------
        # Chronological split
        # -----------------------------------------------------

        n = len(X)

        test_size = int(
            n * TEST_RATIO
        )

        calibration_size = int(
            n * CALIBRATION_RATIO
        )

        # With a multi-day horizon the last target of one block
        # overlaps the first days of the next block. An embargo
        # of `horizon` rows between blocks prevents that leak.
        gap = max(
            TARGET_HORIZON - 1,
            0
        )

        train_size = (
            n
            - calibration_size
            - test_size
            - 2 * gap
        )

        if train_size <= 0:

            raise ValueError(
                f"{ticker}: Invalid chronological split."
            )

        print(
            f"[SPLIT] "
            f"Train={train_size}, "
            f"Calibration={calibration_size}, "
            f"Test={test_size}"
        )

        cal_start = train_size + gap

        test_start = (
            cal_start
            + calibration_size
            + gap
        )

        X_train = X.iloc[
            :train_size
        ].copy()

        y_train = y.iloc[
            :train_size
        ].copy()

        X_cal = X.iloc[
            cal_start:
            cal_start + calibration_size
        ].copy()

        y_cal = y.iloc[
            cal_start:
            cal_start + calibration_size
        ].copy()

        X_test = X.iloc[
            test_start:
        ].copy()

        y_test = y.iloc[
            test_start:
        ].copy()

        # Corresponding raw rows for indicator agreement
        test_rows = df_model.iloc[
            test_start:
        ].copy()

        # -----------------------------------------------------
        # Training dead zone (training rows only)
        # -----------------------------------------------------

        if TRAIN_DEAD_ZONE > 0:

            train_rows = df_model.iloc[
                :train_size
            ]

            threshold = (
                TRAIN_DEAD_ZONE
                * train_rows["vol_20"]
                * np.sqrt(TARGET_HORIZON)
            )

            keep = (
                train_rows["Future_Return_H"].abs()
                >= threshold
            ).values

            X_train = X_train.loc[keep]

            y_train = y_train.loc[keep]

            print(
                f"[DEAD ZONE] Dropped {int((~keep).sum())} "
                f"low-move training rows."
            )

        # -----------------------------------------------------
        # Train model
        # -----------------------------------------------------

        (
            metrics,
            bundle
        ) = train_model(
            X_train,
            y_train,
            X_cal,
            y_cal,
            X_test,
            y_test,
            test_rows=test_rows
        )

        # -----------------------------------------------------
        # Add metadata
        # -----------------------------------------------------

        metrics.update({

            "ticker": ticker,

            "company": company,

            "sector": sector,

            "samples": len(df_model),

            "train_samples":
                len(X_train),

            "calibration_samples":
                len(X_cal),

            "test_samples":
                len(X_test),

            "feature_count":
                len(
                    metrics[
                        "selected_features"
                    ]
                ),

            "feature_columns":
                metrics[
                    "selected_features"
                ]
        })

        # -----------------------------------------------------
        # Save model
        # -----------------------------------------------------

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = (
            MODEL_DIR
            / f"{ticker.replace('.', '_')}_model.joblib"
        )

        joblib.dump(
            bundle,
            model_path
        )

        print(
            f"[MODEL SAVED] {model_path}"
        )

        # -----------------------------------------------------
        # Save per-stock result
        # -----------------------------------------------------

        RESULT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        result_path = (
            RESULT_DIR
            / f"{ticker.replace('.', '_')}_result.json"
        )

        with open(
            result_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=2,
                default=str
            )

        print(
            "\n"
            f"[RESULT] {ticker}\n"
            f"Accuracy : "
            f"{metrics['accuracy']:.4f}\n"
            f"Precision: "
            f"{metrics['precision']:.4f}\n"
            f"Recall   : "
            f"{metrics['recall']:.4f}\n"
            f"F1       : "
            f"{metrics['f1']:.4f}\n"
            f"ROC-AUC  : "
            f"{metrics['roc_auc']:.4f}\n"
            f"Signals  : "
            f"{metrics['signal_count']}\n"
            f"Signal Acc: "
            f"{metrics['signal_accuracy']}\n"
            f"Always-UP: "
            f"{metrics['always_up_accuracy']:.4f}\n"
            f"Top-20% confidence acc: "
            f"{metrics['acc_top20']:.4f} "
            f"(n={metrics['n_top20']})\n"
        )

        return metrics

    except Exception as e:

        print(
            f"[ERROR] {ticker}: {e}"
        )

        raise