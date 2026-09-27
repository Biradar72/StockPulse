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
    CALIBRATION_RATIO
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

def create_target(df):

    df = df.copy()

    # ---------------------------------------------------------
    # FUTURE INFORMATION IS CREATED HERE ONLY
    # ---------------------------------------------------------

    df["Next_Close"] = (
        df["Close"].shift(-1)
    )

    df["Target"] = (
        df["Next_Close"]
        > df["Close"]
    ).astype(int)

    # Last row has no future close
    df = df.iloc[:-1].copy()

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

        valid_mask = (
            X.notna().sum(axis=1) > 0
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

        train_size = (
            n
            - calibration_size
            - test_size
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

        X_train = X.iloc[
            :train_size
        ].copy()

        y_train = y.iloc[
            :train_size
        ].copy()

        X_cal = X.iloc[
            train_size:
            train_size + calibration_size
        ].copy()

        y_cal = y.iloc[
            train_size:
            train_size + calibration_size
        ].copy()

        X_test = X.iloc[
            train_size + calibration_size:
        ].copy()

        y_test = y.iloc[
            train_size + calibration_size:
        ].copy()

        # Corresponding raw rows for indicator agreement
        test_rows = df_model.iloc[
            train_size + calibration_size:
        ].copy()

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
        )

        return metrics

    except Exception as e:

        print(
            f"[ERROR] {ticker}: {e}"
        )

        raise