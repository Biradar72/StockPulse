import pandas as pd

from config.stocks import BSE_STOCKS
from config.settings import RESULT_DIR

from modeling.pipeline import run


rows = []

print(
    "Running pipeline on 25 BSE stocks..."
)

for ticker, company, sector in BSE_STOCKS:

    try:

        result = run(
            ticker,
            company,
            sector
        )

        rows.append(
            result
        )

    except Exception as e:

        print(
            f"[ERROR] "
            f"{ticker}: {e}"
        )

        rows.append({

            "ticker":
                ticker,

            "company":
                company,

            "sector":
                sector,

            "error":
                str(e)
        })


# ============================================================
# SAVE 25-STOCK RESULTS
# ============================================================

out = pd.DataFrame(
    rows
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    RESULT_DIR
    /
    "StockPulse_25_Stock_Results.csv"
)

out.to_csv(
    output_path,
    index=False
)

print(
    "\n"
    "===================================================="
)

print(
    "25-STOCK PIPELINE COMPLETE"
)

print(
    "===================================================="
)

print(
    "Saved:",
    output_path
)

print()

print(
    out.to_string(
        index=False
    )
)