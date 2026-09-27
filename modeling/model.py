import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import RobustScaler

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.isotonic import IsotonicRegression

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_RATIO = 0.20

CALIBRATION_RATIO = 0.10

CONFIDENCE_THRESHOLD = 0.55

AGREEMENT_THRESHOLD = 0.58

MAX_FEATURES = 50

MIN_SAMPLES = 300

MIN_CALIBRATION_SAMPLES = 50


# ============================================================
# LEAKAGE PROTECTION
# ============================================================

BAD_FEATURES = {
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


def check_for_leakage(X):

    bad_features = [
        col
        for col in X.columns
        if (
            col in BAD_FEATURES
            or col.lower() in {
                "target",
                "next_close",
                "future_close",
                "future_return",
                "future_price",
                "label"
            }
        )
    ]

    if bad_features:

        raise ValueError(
            "DATA LEAKAGE DETECTED: "
            f"{bad_features}"
        )

    print(
        "[LEAKAGE CHECK] PASS - "
        "No future target columns found."
    )


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(
    df,
    target_col="Target"
):

    exclude = {
        "Date",

        "Open",
        "High",
        "Low",
        "Close",
        "Volume",

        "Target",
        "target",

        "Next_Close",
        "next_close",

        "future_close",
        "Future_Close",

        "future_return",
        "Future_Return",

        "future_price",
        "Future_Price",

        "label",
        "Label"
    }

    feature_cols = [
        c
        for c in df.columns
        if c not in exclude
    ]

    X = df[
        feature_cols
    ].copy()

    X = X.select_dtypes(
        include=[np.number]
    )

    check_for_leakage(X)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = df[
        target_col
    ].astype(int)

    return X, y


# ============================================================
# CORRELATION PRUNING
# ============================================================

def correlation_pruning(
    X_train,
    X_other,
    threshold=0.95
):

    corr = (
        X_train
        .corr()
        .abs()
    )

    upper = corr.where(
        np.triu(
            np.ones(
                corr.shape
            ),
            k=1
        ).astype(bool)
    )

    drop_cols = [
        column
        for column in upper.columns
        if any(
            upper[column] > threshold
        )
    ]

    X_train_new = (
        X_train
        .drop(
            columns=drop_cols,
            errors="ignore"
        )
    )

    X_other_new = (
        X_other
        .drop(
            columns=drop_cols,
            errors="ignore"
        )
    )

    return (
        X_train_new,
        X_other_new,
        drop_cols
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features(
    X_train,
    y_train,
    X_cal,
    X_test,
    max_features=MAX_FEATURES
):

    # --------------------------------------------------------
    # IMPORTANT:
    # IMPUTE BEFORE SelectKBest
    # --------------------------------------------------------

    selector_imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = (
        selector_imputer
        .fit_transform(
            X_train
        )
    )

    X_cal_imp = (
        selector_imputer
        .transform(
            X_cal
        )
    )

    X_test_imp = (
        selector_imputer
        .transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # CHECK REMAINING NaN
    # --------------------------------------------------------

    if np.isnan(
        X_train_imp
    ).any():

        raise ValueError(
            "NaN remains in training data "
            "after imputation."
        )

    if np.isnan(
        X_cal_imp
    ).any():

        raise ValueError(
            "NaN remains in calibration data "
            "after imputation."
        )

    if np.isnan(
        X_test_imp
    ).any():

        raise ValueError(
            "NaN remains in test data "
            "after imputation."
        )

    n_features = X_train.shape[1]

    # --------------------------------------------------------
    # No feature selection required
    # --------------------------------------------------------

    if n_features <= max_features:

        selected = list(
            X_train.columns
        )

        X_train_selected = pd.DataFrame(
            X_train_imp,
            columns=selected,
            index=X_train.index
        )

        X_cal_selected = pd.DataFrame(
            X_cal_imp,
            columns=selected,
            index=X_cal.index
        )

        X_test_selected = pd.DataFrame(
            X_test_imp,
            columns=selected,
            index=X_test.index
        )

        return (
            X_train_selected,
            X_cal_selected,
            X_test_selected,
            selected,
            selector_imputer
        )

    # --------------------------------------------------------
    # Mutual Information Selection
    # --------------------------------------------------------

    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=min(
            max_features,
            n_features
        )
    )

    selector.fit(
        X_train_imp,
        y_train
    )

    mask = selector.get_support()

    selected = list(
        X_train.columns[mask]
    )

    X_train_selected = pd.DataFrame(
        X_train_imp[:, mask],
        columns=selected,
        index=X_train.index
    )

    X_cal_selected = pd.DataFrame(
        X_cal_imp[:, mask],
        columns=selected,
        index=X_cal.index
    )

    X_test_selected = pd.DataFrame(
        X_test_imp[:, mask],
        columns=selected,
        index=X_test.index
    )

    return (
        X_train_selected,
        X_cal_selected,
        X_test_selected,
        selected,
        selector_imputer
    )


# ============================================================
# BUILD MODELS
# ============================================================

def build_models():

    return {

        "xgb": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.035,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.10,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "lgbm": LGBMClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.035,
            num_leaves=24,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.10,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1
        ),

        "rf": RandomForestClassifier(
            n_estimators=500,
            max_depth=9,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "gb": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.85,
            random_state=RANDOM_STATE
        ),

        "lr": LogisticRegression(
            C=0.35,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE
        )
    }


# ============================================================
# MODEL WEIGHTS
# ============================================================

MODEL_WEIGHTS = {
    "xgb": 0.28,
    "lgbm": 0.28,
    "rf": 0.18,
    "gb": 0.14,
    "lr": 0.12
}


# ============================================================
# INDICATOR AGREEMENT
# ============================================================

def indicator_agreement(row):

    signals = []

    if "ret_1d" in row:

        value = row["ret_1d"]

        if pd.notna(value):

            signals.append(
                int(value > 0)
            )

    if (
        "Close" in row
        and
        "sma_20" in row
    ):

        if pd.notna(row["sma_20"]):

            signals.append(
                int(
                    row["Close"]
                    >
                    row["sma_20"]
                )
            )

    if "macd" in row:

        if pd.notna(row["macd"]):

            signals.append(
                int(
                    row["macd"] > 0
                )
            )

    if "macd_hist" in row:

        if pd.notna(
            row["macd_hist"]
        ):

            signals.append(
                int(
                    row["macd_hist"] > 0
                )
            )

    if "rsi_14" in row:

        if pd.notna(
            row["rsi_14"]
        ):

            signals.append(
                int(
                    row["rsi_14"] > 50
                )
            )

    if "stoch_k" in row:

        if pd.notna(
            row["stoch_k"]
        ):

            signals.append(
                int(
                    row["stoch_k"] > 50
                )
            )

    if "williams_r" in row:

        if pd.notna(
            row["williams_r"]
        ):

            signals.append(
                int(
                    row["williams_r"] > -50
                )
            )

    if "roc_10" in row:

        if pd.notna(
            row["roc_10"]
        ):

            signals.append(
                int(
                    row["roc_10"] > 0
                )
            )

    if (
        "Close" in row
        and
        "sma_50" in row
    ):

        if pd.notna(
            row["sma_50"]
        ):

            signals.append(
                int(
                    row["Close"]
                    >
                    row["sma_50"]
                )
            )

    if "gap" in row:

        if pd.notna(
            row["gap"]
        ):

            signals.append(
                int(
                    row["gap"] > 0
                )
            )

    if not signals:

        return 0.50

    return (
        sum(signals)
        /
        len(signals)
    )


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_probabilities(
    y_cal,
    raw_probabilities
):

    y_cal = np.asarray(
        y_cal
    )

    raw_probabilities = np.asarray(
        raw_probabilities
    )

    if len(y_cal) < MIN_CALIBRATION_SAMPLES:

        return (
            lambda x: np.asarray(x),
            False
        )

    if len(
        np.unique(y_cal)
    ) < 2:

        return (
            lambda x: np.asarray(x),
            False
        )

    if len(
        np.unique(
            np.round(
                raw_probabilities,
                6
            )
        )
    ) < 4:

        return (
            lambda x: np.asarray(x),
            False
        )

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip"
    )

    calibrator.fit(
        raw_probabilities,
        y_cal
    )

    print(
        "[CALIBRATION] "
        "Isotonic calibration applied."
    )

    return (
        calibrator.predict,
        True
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
    X_cal,
    y_cal,
    X_test,
    y_test,
    test_rows=None
):

    check_for_leakage(
        X_train
    )

    # ========================================================
    # CORRELATION PRUNING
    # ========================================================

    (
        X_train,
        X_cal,
        dropped
    ) = correlation_pruning(
        X_train,
        X_cal
    )

    X_test = X_test.drop(
        columns=dropped,
        errors="ignore"
    )

    X_cal = X_cal[
        X_train.columns
    ]

    X_test = X_test[
        X_train.columns
    ]

    print(
        f"[FEATURE PRUNING] "
        f"Removed {len(dropped)} "
        f"highly correlated features."
    )

    # ========================================================
    # FEATURE SELECTION + IMPUTATION
    # ========================================================

    (
        X_train,
        X_cal,
        X_test,
        selected_features,
        selector_imputer
    ) = select_features(
        X_train,
        y_train,
        X_cal,
        X_test
    )

    print(
        f"[FEATURE SELECTION] "
        f"Selected {len(selected_features)} "
        f"features."
    )

    # Data is already imputed
    X_train_imp = X_train.values
    X_cal_imp = X_cal.values
    X_test_imp = X_test.values

    # ========================================================
    # SCALING
    # ========================================================

    scaler = RobustScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train_imp
        )
    )

    X_cal_scaled = (
        scaler.transform(
            X_cal_imp
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test_imp
        )
    )

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    models = build_models()

    probabilities_cal = {}

    probabilities_test = {}

    for name, model in models.items():

        print(
            f"[TRAIN] {name.upper()}"
        )

        if name == "lr":

            model.fit(
                X_train_scaled,
                y_train
            )

            probabilities_cal[name] = (
                model
                .predict_proba(
                    X_cal_scaled
                )[:, 1]
            )

            probabilities_test[name] = (
                model
                .predict_proba(
                    X_test_scaled
                )[:, 1]
            )

        else:

            model.fit(
                X_train_imp,
                y_train
            )

            probabilities_cal[name] = (
                model
                .predict_proba(
                    X_cal_imp
                )[:, 1]
            )

            probabilities_test[name] = (
                model
                .predict_proba(
                    X_test_imp
                )[:, 1]
            )

    # ========================================================
    # SOFT VOTING
    # ========================================================

    raw_cal = np.zeros(
        len(X_cal)
    )

    raw_test = np.zeros(
        len(X_test)
    )

    for name, weight in (
        MODEL_WEIGHTS.items()
    ):

        raw_cal += (
            weight
            *
            probabilities_cal[name]
        )

        raw_test += (
            weight
            *
            probabilities_test[name]
        )

    # ========================================================
    # ISOTONIC CALIBRATION
    # ========================================================

    (
        calibrator,
        calibrated
    ) = calibrate_probabilities(
        y_cal,
        raw_cal
    )

    cal_prob = calibrator(
        raw_cal
    )

    test_prob = calibrator(
        raw_test
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    predictions = (
        test_prob >= 0.50
    ).astype(int)

    # ========================================================
    # INDICATOR AGREEMENT
    # ========================================================

    if test_rows is not None:

        agreements = [
            indicator_agreement(row)
            for _, row
            in test_rows.iterrows()
        ]

    else:

        agreements = [
            0.50
            for _ in range(
                len(test_prob)
            )
        ]

    agreements = np.asarray(
        agreements
    )

    # ========================================================
    # SIGNAL GENERATION
    # ========================================================

    signals = []

    confidences = []

    for prob, agreement in zip(
        test_prob,
        agreements
    ):

        confidence = max(
            prob,
            1 - prob
        )

        confidences.append(
            confidence
        )

        if (
            confidence
            >= CONFIDENCE_THRESHOLD
            and
            agreement
            >= AGREEMENT_THRESHOLD
        ):

            if prob >= 0.50:

                signals.append(
                    "UP"
                )

            else:

                signals.append(
                    "DOWN"
                )

        else:

            signals.append(
                "NO SIGNAL"
            )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y_test,
            test_prob
        )

    except Exception:

        roc_auc = np.nan

    cm = confusion_matrix(
        y_test,
        predictions
    )

    signal_mask = (
        np.asarray(signals)
        !=
        "NO SIGNAL"
    )

    signal_count = int(
        signal_mask.sum()
    )

    signal_rate = (
        signal_count
        /
        len(signals)
        if len(signals) > 0
        else 0
    )

    if signal_count > 0:

        signal_predictions = (
            np.asarray(signals)[
                signal_mask
            ]
            ==
            "UP"
        ).astype(int)

        signal_actual = (
            np.asarray(y_test)[
                signal_mask
            ]
        )

        signal_accuracy = (
            accuracy_score(
                signal_actual,
                signal_predictions
            )
        )

    else:

        signal_accuracy = np.nan

    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            (
                float(roc_auc)
                if not np.isnan(
                    roc_auc
                )
                else np.nan
            ),

        "confusion_matrix":
            cm.tolist(),

        "signal_accuracy":
            (
                float(
                    signal_accuracy
                )
                if not np.isnan(
                    signal_accuracy
                )
                else np.nan
            ),

        "signal_rate":
            float(signal_rate),

        "signal_count":
            signal_count,

        "up_signals":
            int(
                np.sum(
                    np.asarray(signals)
                    == "UP"
                )
            ),

        "down_signals":
            int(
                np.sum(
                    np.asarray(signals)
                    == "DOWN"
                )
            ),

        "no_signal":
            int(
                np.sum(
                    np.asarray(signals)
                    == "NO SIGNAL"
                )
            ),

        "calibration_applied":
            calibrated,

        "selected_features":
            selected_features,

        "model_weights":
            MODEL_WEIGHTS,

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "agreement_threshold":
            AGREEMENT_THRESHOLD,

        "probabilities":
            test_prob.tolist(),

        "predictions":
            predictions.tolist(),

        "actuals":
            np.asarray(
                y_test
            ).tolist(),

        "signals":
            signals,

        "agreements":
            agreements.tolist()
    }

    # ========================================================
    # MODEL BUNDLE
    # ========================================================

    bundle = {

        "models":
            models,

        "selector_imputer":
            selector_imputer,

        "scaler":
            scaler,

        "selected_features":
            selected_features,

        "calibrator":
            calibrator,

        "calibrated":
            calibrated,

        "model_weights":
            MODEL_WEIGHTS,

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "agreement_threshold":
            AGREEMENT_THRESHOLD
    }

    return (
        results,
        bundle
    )