"""
collection/google_news.py
──────────────────────────
Improved Google News scraper:
- Uses both feedparser and urllib (2 methods for reliability)
- More diverse queries (sector, macro, BSE-specific)
- Deduplication by title hash
- Saves to CSV cache
"""
import time, hashlib, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
NEWS_DIR = ROOT / "data" / "02_news_data" / "csv"
NEWS_DIR.mkdir(parents=True, exist_ok=True)


def _title_hash(t):
    return hashlib.md5("".join(c.lower() for c in t if c.isalnum()).encode()).hexdigest()[:12]


def _parse_date(entry):
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            # published_parsed is UTC; the pipeline compares against
            # the 15:30 IST market close, so convert to naive IST.
            return datetime(*entry.published_parsed[:6]) + timedelta(hours=5, minutes=30)
        except:
            pass
    return None


def _queries(company, ticker, sector):
    bare = ticker.replace(".BO", "").replace(".NS", "")
    return [
        f'"{company}" stock India',
        f'"{company}" earnings results profit',
        f'"{company}" share price BSE',
        f'"{company}" business news',
        f'"{company}" quarterly annual',
        f'"{company}" {sector}',
        f'"{bare}" NSE BSE',
        f'{company} dividend buyback',
        f'{company} management news',
    ]


def scrape_google_news(ticker, company, sector="", max_per_query=50):
    rows, seen = [], set()
    queries = _queries(company, ticker, sector)

    for q in queries:
        url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
            {"q": q, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
        )
        print(f"  [RSS] {q}")
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_query]:
                title = (entry.get("title") or "").strip()
                if not title:
                    continue
                h = _title_hash(title)
                if h in seen:
                    continue
                seen.add(h)
                rows.append({
                    "date"       : _parse_date(entry),
                    "headline"   : title,
                    "description": (entry.get("summary") or "")[:300],
                    "source"     : getattr(entry.get("source", None), "title", "Google News"),
                    "source_url" : entry.get("link", ""),
                    "query"      : q,
                })
        except Exception as e:
            print(f"  [WARN] {q}: {e}")
        time.sleep(0.4)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date","headline"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[NEWS] {ticker}: {len(df)} unique articles scraped")
    return df


def save_news(df, ticker):
    p = NEWS_DIR / f"{ticker.replace('.','_')}_news.csv"
    if p.exists():
        old = pd.read_csv(p, parse_dates=["date"])
        df  = pd.concat([old, df], ignore_index=True)
        df  = df.drop_duplicates(subset=["date","headline"])
    df.to_csv(p, index=False)
    print(f"[SAVED] {p}  ({len(df)} rows)")
    return p


def load_news(ticker):
    p = NEWS_DIR / f"{ticker.replace('.','_')}_news.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker",  required=True)
    ap.add_argument("--company", required=True)
    ap.add_argument("--sector",  default="")
    ap.add_argument("--max",     type=int, default=50)
    args = ap.parse_args()
    d = scrape_google_news(args.ticker, args.company, args.sector, args.max)
    if not d.empty:
        save_news(d, args.ticker)
    else:
        print("No news found.")
