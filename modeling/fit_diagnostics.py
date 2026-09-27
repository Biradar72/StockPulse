from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

# fit_diagnostics.py is inside:
# StockPulse_GoogleNews_Complete/modeling/
#
# parents[1] = StockPulse_GoogleNews_Complete/

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "05_final_dataset"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STOCKPULSE — OVERFITTING / UNDERFITTING DIAGNOSTICS")
print("=" * 80)
print(f"Project root:      {ROOT}")
print(f"Dataset directory: {DATA_DIR}")
print()


# ============================================================
# TEST THE PATH
# ============================================================

if not DATA_DIR.exists():
    print("[ERROR] Dataset directory does not exist:")
    print(DATA_DIR)
    raise SystemExit(1)

print("[OK] Dataset directory found.")
print()


# ============================================================
# STOCK LIST
# ============================================================

STOCKS = [
    "RELIANCE.BO",
    "TCS.BO",
    "INFY.BO",
    "WIPRO.BO",
    "HDFCBANK.BO",
    "ICICIBANK.BO",
    "SBIN.BO",
    "AXISBANK.BO",
    "KOTAKBANK.BO",
    "ITC.BO",
    "HINDUNILVR.BO",
    "ASIANPAINT.BO",
    "TATAMOTORS.BO",
    "MARUTI.BO",
    "SUNPHARMA.BO",
    "DRREDDY.BO",
    "BHARTIARTL.BO",
    "TATASTEEL.BO",
    "JSWSTEEL.BO",
    "LT.BO",
    "ULTRACEMCO.BO",
    "BAJFINANCE.BO",
    "ADANIENT.BO",
    "NTPC.BO",
    "ONGC.BO",
]


# ============================================================
# CHECK ALL 25 DATASETS
# ============================================================

found = []
missing = []

for ticker in STOCKS:

    filename = f"{ticker.replace('.', '_')}_final_dataset.csv"
    path = DATA_DIR / filename

    print(f"[CHECKING] {ticker}")

    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)

        print(f"  [FOUND] {filename}")
        print(f"  Size: {size_mb:.2f} MB")

        found.append(ticker)

    else:
        print(f"  [MISSING] {filename}")
        missing.append(ticker)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("DATASET CHECK SUMMARY")
print("=" * 80)

print(f"Total stocks required : {len(STOCKS)}")
print(f"Datasets found        : {len(found)}")
print(f"Datasets missing      : {len(missing)}")

print()

if found:
    print("FOUND:")
    for ticker in found:
        print(f"  ✓ {ticker}")

print()

if missing:
    print("MISSING:")
    for ticker in missing:
        print(f"  ✗ {ticker}")

print()
print("=" * 80)

if len(found) < 25:
    print("WARNING:")
    print(
        "Only the datasets listed above exist in data/05_final_dataset."
    )
    print(
        "The remaining datasets must be generated before running "
        "full 25-stock diagnostics."
    )
else:
    print("All 25 datasets are available.")