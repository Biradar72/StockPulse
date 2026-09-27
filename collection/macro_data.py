from pathlib import Path

import pandas as pd


ROOT_DIR = Path(
    __file__
).resolve().parents[1]

MACRO_DIR = (
    ROOT_DIR
    /
    "data"
    /
    "06_macro_data"
)


def _load_csv(
    path
):

    if not path.exists():

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            path
        )

    except Exception as e:

        print(
            f"[WARN] "
            f"Could not read {path}: {e}"
        )

        return pd.DataFrame()

    if "Date" not in df.columns:

        # Try common date names
        date_candidates = [
            "date",
            "DATE",
            "timestamp",
            "Timestamp"
        ]

        found = None

        for col in date_candidates:

            if col in df.columns:

                found = col

                break

        if found is not None:

            df = df.rename(
                columns={
                    found: "Date"
                }
            )

        else:

            return pd.DataFrame()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    )

    df["Date"] = (
        df["Date"]
        .dt.normalize()
    )

    return df


def load_macro_data(
    start_date=None,
    end_date=None
):

    """
    Load locally available macro CSV data.

    Expected directory:

        data/06_macro_data/

    Possible files:

        india_vix.csv
        nifty50.csv
        nifty_bank.csv
        usd_inr.csv
        crude_oil.csv
        gold.csv
        sp500.csv
        nikkei.csv

    The function does not create fake macro values.
    """

    MACRO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = [

        (
            "india_vix.csv",
            "india_vix"
        ),

        (
            "nifty50.csv",
            "nifty50"
        ),

        (
            "nifty_bank.csv",
            "nifty_bank"
        ),

        (
            "usd_inr.csv",
            "usd_inr"
        ),

        (
            "crude_oil.csv",
            "crude_oil"
        ),

        (
            "gold.csv",
            "gold"
        ),

        (
            "sp500.csv",
            "sp500"
        ),

        (
            "nikkei.csv",
            "nikkei"
        )
    ]

    frames = []

    for filename, prefix in files:

        path = (
            MACRO_DIR
            /
            filename
        )

        df = _load_csv(
            path
        )

        if df.empty:

            continue

        # Rename numeric columns
        value_columns = [
            c
            for c in df.columns
            if c != "Date"
        ]

        if not value_columns:

            continue

        # Use first numeric column
        value_col = None

        for col in value_columns:

            converted = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            if converted.notna().sum() > 0:

                value_col = col

                df[col] = converted

                break

        if value_col is None:

            continue

        df = df[
            [
                "Date",
                value_col
            ]
        ].copy()

        df = df.rename(
            columns={
                value_col: prefix
            }
        )

        frames.append(
            df
        )

    if not frames:

        print(
            "[MACRO] "
            "No local macro CSV files found."
        )

        return pd.DataFrame()

    # ========================================================
    # MERGE
    # ========================================================

    macro = frames[0]

    for frame in frames[1:]:

        macro = macro.merge(
            frame,
            on="Date",
            how="outer"
        )

    macro = (
        macro
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # LIMIT DATE RANGE
    # ========================================================

    if start_date is not None:

        start_date = pd.Timestamp(
            start_date
        ).normalize()

        macro = macro[
            macro["Date"]
            >= start_date
        ]

    if end_date is not None:

        end_date = pd.Timestamp(
            end_date
        ).normalize()

        macro = macro[
            macro["Date"]
            <= end_date
        ]

    # ========================================================
    # FORWARD FILL
    # ========================================================

    value_columns = [
        c
        for c in macro.columns
        if c != "Date"
    ]

    if value_columns:

        macro[value_columns] = (
            macro[value_columns]
            .ffill()
        )

    # ========================================================
    # RETURNS
    # ========================================================

    for col in list(
        value_columns
    ):

        macro[
            f"{col}_ret_1d"
        ] = (
            macro[col]
            .pct_change()
        )

    macro = macro.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    return macro