import re
import numpy as np
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


MARKET_CUTOFF_HOUR = 15
MARKET_CUTOFF_MINUTE = 30

# Only news from the last N days affects a given trading day.
# Without this window news_count grows forever (it counted every
# article ever published), which is a trend, not a signal.
NEWS_LOOKBACK_DAYS = 30

analyzer = SentimentIntensityAnalyzer()


NEWS_CATEGORIES = {

    "earnings": {
        "keywords": [
            "earnings",
            "results",
            "profit",
            "revenue",
            "quarterly results",
            "quarter",
            "financial results"
        ],
        "weight": 1.00,
        "half_life": 5
    },

    "management": {
        "keywords": [
            "management",
            "ceo",
            "cfo",
            "director",
            "resignation",
            "appointment",
            "leadership"
        ],
        "weight": 0.90,
        "half_life": 7
    },

    "business": {
        "keywords": [
            "business",
            "expansion",
            "contract",
            "order",
            "partnership",
            "acquisition",
            "merger",
            "investment"
        ],
        "weight": 0.85,
        "half_life": 10
    },

    "regulatory": {
        "keywords": [
            "sebi",
            "rbi",
            "government",
            "regulator",
            "regulatory",
            "approval",
            "penalty",
            "investigation",
            "court"
        ],
        "weight": 0.90,
        "half_life": 7
    },

    "market": {
        "keywords": [
            "stock",
            "shares",
            "market",
            "investor",
            "buy",
            "sell",
            "upgrade",
            "downgrade",
            "target price"
        ],
        "weight": 0.75,
        "half_life": 3
    },

    "macro": {
        "keywords": [
            "inflation",
            "interest rate",
            "repo rate",
            "crude oil",
            "rupee",
            "dollar",
            "gdp",
            "economy",
            "economic",
            "global market"
        ],
        "weight": 0.65,
        "half_life": 3
    }
}


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_sentiment(text):

    text = clean_text(text)

    if not text:
        return 0.0

    return float(
        analyzer
        .polarity_scores(text)
        ["compound"]
    )


def detect_categories(text):

    text = clean_text(
        text
    ).lower()

    categories = []

    for category, config in NEWS_CATEGORIES.items():

        for keyword in config["keywords"]:

            if keyword.lower() in text:

                categories.append(
                    category
                )

                break

    if not categories:

        categories.append(
            "market"
        )

    return categories


def prepare_news(news):

    if news is None or news.empty:

        return pd.DataFrame()

    news = news.copy()

    if "date" not in news.columns:

        if "Date" in news.columns:

            news["date"] = news["Date"]

        elif "published" in news.columns:

            news["date"] = news["published"]

        else:

            return pd.DataFrame()

    news["date"] = pd.to_datetime(
        news["date"],
        errors="coerce"
    )

    news = news.dropna(
        subset=["date"]
    )

    try:

        if news["date"].dt.tz is not None:

            news["date"] = (
                news["date"]
                .dt.tz_convert(
                    "Asia/Kolkata"
                )
                .dt.tz_localize(None)
            )

    except Exception:
        pass

    if "headline" not in news.columns:

        if "title" in news.columns:

            news["headline"] = (
                news["title"]
            )

        else:

            news["headline"] = ""

    if "description" not in news.columns:

        news["description"] = ""

    news["headline"] = (
        news["headline"]
        .fillna("")
        .astype(str)
    )

    news["description"] = (
        news["description"]
        .fillna("")
        .astype(str)
    )

    news["text"] = (
        news["headline"]
        + " "
        + news["description"]
    ).apply(
        clean_text
    )

    news["sentiment"] = (
        news["text"]
        .apply(get_sentiment)
    )

    news["categories"] = (
        news["text"]
        .apply(detect_categories)
    )

    return news


def filter_market_cutoff(
    news,
    current_date
):

    if news.empty:

        return news

    current_date = pd.Timestamp(
        current_date
    ).normalize()

    cutoff = (
        current_date
        +
        pd.Timedelta(
            hours=15,
            minutes=30
        )
    )

    window_start = (
        current_date
        -
        pd.Timedelta(
            days=NEWS_LOOKBACK_DAYS
        )
    )

    return news[
        (news["date"] <= cutoff)
        &
        (news["date"] > window_start)
    ].copy()


def calculate_decay(
    days_since_event,
    half_life
):

    if days_since_event < 0:

        return 0.0

    return float(
        0.5
        **
        (
            days_since_event
            /
            half_life
        )
    )


def build_daily_news_features(
    news,
    current_date,
    volatility=None
):

    empty = {

        "overall_sent": 0.0,
        "news_count": 0.0,
        "news_impact": 0.0,

        "earnings_sent": 0.0,
        "management_sent": 0.0,
        "business_sent": 0.0,
        "regulatory_sent": 0.0,
        "market_sent": 0.0,
        "macro_sent": 0.0,

        "earnings_count": 0.0,
        "management_count": 0.0,
        "business_count": 0.0,
        "regulatory_count": 0.0,
        "market_count": 0.0,
        "macro_count": 0.0,

        "news_positive_ratio": 0.0,
        "news_negative_ratio": 0.0,
        "news_sentiment_std": 0.0
    }

    if news is None or news.empty:

        return empty

    current_date = pd.Timestamp(
        current_date
    ).normalize()

    events = filter_market_cutoff(
        news,
        current_date
    )

    if events.empty:

        return empty

    event_dates = (
        events["date"]
        .dt.normalize()
    )

    events["days_since_event"] = (
        current_date
        -
        event_dates
    ).dt.days.clip(
        lower=0
    )

    volatility_gate = 1.0

    if volatility is not None:

        try:

            vol = float(
                volatility
            )

            if np.isfinite(vol):

                volatility_gate = (
                    1.0
                    /
                    (
                        1.0
                        +
                        min(
                            max(
                                vol,
                                0.0
                            ),
                            1.0
                        )
                    )
                )

        except Exception:

            pass

    sentiments = (
        events["sentiment"]
        .astype(float)
    )

    empty["news_count"] = float(
        len(events)
    )

    empty["news_positive_ratio"] = float(
        (
            sentiments > 0.05
        ).mean()
    )

    empty["news_negative_ratio"] = float(
        (
            sentiments < -0.05
        ).mean()
    )

    empty["news_sentiment_std"] = float(
        sentiments.std()
    ) if len(sentiments) > 1 else 0.0

    total_weight = 0.0
    total_sentiment = 0.0

    for _, event in events.iterrows():

        sentiment = float(
            event["sentiment"]
        )

        categories = event[
            "categories"
        ]

        if not isinstance(
            categories,
            list
        ):

            categories = [
                "market"
            ]

        for category in categories:

            if category not in NEWS_CATEGORIES:

                continue

            config = (
                NEWS_CATEGORIES[
                    category
                ]
            )

            decay = calculate_decay(
                event[
                    "days_since_event"
                ],
                config[
                    "half_life"
                ]
            )

            weight = (
                config["weight"]
                *
                decay
                *
                volatility_gate
            )

            weighted_sentiment = (
                sentiment
                *
                weight
            )

            empty[
                f"{category}_sent"
            ] += weighted_sentiment

            empty[
                f"{category}_count"
            ] += weight

            total_sentiment += (
                weighted_sentiment
            )

            total_weight += weight

    if total_weight > 0:

        empty["overall_sent"] = (
            total_sentiment
            /
            total_weight
        )

    empty["news_impact"] = (
        total_sentiment
    )

    return empty


def build_news_features(
    price_df,
    news,
    ticker=None
):

    if price_df is None or price_df.empty:

        return pd.DataFrame()

    news = prepare_news(
        news
    )

    dates = pd.to_datetime(
        price_df["Date"],
        errors="coerce"
    ).dt.normalize()

    rows = []

    volatility_series = None

    if "ret_1d" in price_df.columns:

        volatility_series = (
            price_df
            .assign(
                _date=pd.to_datetime(
                    price_df["Date"]
                ).dt.normalize()
            )
            .set_index("_date")
            ["ret_1d"]
            .rolling(20)
            .std()
        )

    for date in dates:

        volatility = None

        if volatility_series is not None:

            try:

                volatility = (
                    volatility_series
                    .get(date)
                )

            except Exception:

                volatility = None

        row = {
            "Date": date
        }

        row.update(
            build_daily_news_features(
                news,
                date,
                volatility
            )
        )

        rows.append(
            row
        )

    return (
        pd.DataFrame(rows)
        .drop_duplicates(
            subset=["Date"]
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )