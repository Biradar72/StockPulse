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

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from processing.technical import PRICE_LEVEL_COLUMNS
from collection.macro_data import MACRO_LEVEL_COLUMNS


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

# Predict the direction of the close TARGET_HORIZON trading days
# ahead (1 = next day).
TARGET_HORIZON = 1

# Training rows whose forward move is smaller than
# TRAIN_DEAD_ZONE x (20-day daily volatility) are dropped from
# TRAINING only. Such days are mostly noise and teach the model
# nothing. Calibration and test keep every day, so the reported
# accuracy is not inflated. 0 disables the filter.
TRAIN_DEAD_ZONE = 0.0

# A row is dropped if more than this fraction of its features is
# missing (the first ~200 days, before sma_200 exists).
MAX_MISSING_FRACTION = 0.10

# Accuracy is also reported on the most confident X% of test
# days. The confidence cut-off is chosen on the calibration set,
# never on the test set.
COVERAGE_LEVELS = [0.10, 0.20, 0.30, 0.50]


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
        "Label",

        # Forward return used for the dead-zone filter
        "Future_Return_H"
    }

    # Non-stationary rupee / index levels
    exclude.update(PRICE_LEVEL_COLUMNS)
    exclude.update(MACRO_LEVEL_COLUMNS)

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

    # Daily stock returns are mostly noise, so every model is
    # heavily regularised: shallow trees, large leaves, row and
    # column subsampling. Deep trees memorise the training years
    # and score ~50% on unseen years.

    return {

        "xgb": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.02,
            min_child_weight=20,
            subsample=0.70,
            colsample_bytree=0.60,
            reg_alpha=1.0,
            reg_lambda=5.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "lgbm": LGBMClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.02,
            num_leaves=8,
            min_child_samples=50,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_alpha=1.0,
            reg_lambda=5.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1
        ),

        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=30,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "gb": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=30,
            subsample=0.70,
            random_state=RANDOM_STATE
        ),

        "lr": LogisticRegression(
            C=0.05,
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

class PlattCalibrator:

    """
    Platt scaling: a 1-D logistic regression on the logit of the
    ensemble probability. A class (not a lambda) so the model
    bundle can be saved with joblib. With no fitted model it
    returns probabilities unchanged.
    """

    def __init__(self):

        self.model = None

    @staticmethod
    def _logit(p):

        p = np.clip(
            np.asarray(p, dtype=float),
            1e-6,
            1 - 1e-6
        )

        return np.log(
            p / (1 - p)
        ).reshape(-1, 1)

    def fit(self, raw_probabilities, y):

        self.model = LogisticRegression(
            C=1.0
        )

        self.model.fit(
            self._logit(raw_probabilities),
            y
        )

        return self

    def __call__(self, raw_probabilities):

        if self.model is None:

            return np.asarray(
                raw_probabilities
            )

        return self.model.predict_proba(
            self._logit(raw_probabilities)
        )[:, 1]


def calibrate_probabilities(
    y_cal,
    raw_probabilities
):

    # Isotonic regression on ~240 calibration rows produced a
    # step function with only a few distinct outputs, which often
    # pushed every test day to the same side of 0.5. Platt
    # scaling is smooth and monotone.

    y_cal = np.asarray(
        y_cal
    )

    if (
        len(y_cal) < MIN_CALIBRATION_SAMPLES
        or len(np.unique(y_cal)) < 2
    ):

        return (
            PlattCalibrator(),
            False
        )

    print(
        "[CALIBRATION] "
        "Platt calibration applied."
    )

    return (
        PlattCalibrator().fit(
            raw_probabilities,
            y_cal
        ),
        True
    )


# ============================================================
# ACCURACY BY CONFIDENCE COVERAGE
# ============================================================

def coverage_accuracy(
    cal_prob,
    test_prob,
    y_test
):

    """
    For each coverage level c, find the confidence cut-off that
    keeps the top c fraction of CALIBRATION days, then report the
    accuracy on test days that pass that cut-off.
    """

    cal_conf = np.abs(
        np.asarray(cal_prob) - 0.5
    )

    test_conf = np.abs(
        np.asarray(test_prob) - 0.5
    )

    y_test = np.asarray(
        y_test
    )

    test_pred = (
        np.asarray(test_prob) >= 0.5
    ).astype(int)

    out = {}

    for level in COVERAGE_LEVELS:

        key = f"top{int(level * 100)}"

        cutoff = np.quantile(
            cal_conf,
            1 - level
        )

        mask = test_conf >= cutoff

        out[f"acc_{key}"] = (
            float(
                (test_pred[mask] == y_test[mask]).mean()
            )
            if mask.sum() > 0
            else np.nan
        )

        out[f"n_{key}"] = int(
            mask.sum()
        )

    return out


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

        # agreement is the share of BULLISH indicators, so an UP
        # signal needs agreement >= threshold and a DOWN signal
        # needs agreement <= 1 - threshold.

        if (
            confidence
            >= CONFIDENCE_THRESHOLD
            and
            prob >= 0.50
            and
            agreement
            >= AGREEMENT_THRESHOLD
        ):

            signals.append(
                "UP"
            )

        elif (
            confidence
            >= CONFIDENCE_THRESHOLD
            and
            prob < 0.50
            and
            agreement
            <= 1 - AGREEMENT_THRESHOLD
        ):

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

        "always_up_accuracy":
            float(
                np.mean(
                    np.asarray(y_test)
                )
            ),

        **coverage_accuracy(
            cal_prob,
            test_prob,
            y_test
        ),

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