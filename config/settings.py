from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = ROOT_DIR / "data"

PRICE_DIR = DATA_DIR / "01_price_data"

NEWS_DIR = DATA_DIR / "02_news_data"

NEWS_CSV_DIR = NEWS_DIR / "csv"

CLEAN_NEWS_DIR = DATA_DIR / "03_clean_news"

NEWS_FEATURE_DIR = DATA_DIR / "04_news_features"

FINAL_DATASET_DIR = DATA_DIR / "05_final_dataset"


# ============================================================
# MODEL / RESULT DIRECTORIES
# ============================================================

MODEL_DIR = ROOT_DIR / "models"

RESULT_DIR = ROOT_DIR / "results"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    DATA_DIR,
    PRICE_DIR,
    NEWS_DIR,
    NEWS_CSV_DIR,
    CLEAN_NEWS_DIR,
    NEWS_FEATURE_DIR,
    FINAL_DATASET_DIR,
    MODEL_DIR,
    RESULT_DIR
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )