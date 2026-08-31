from __future__ import annotations

from typing import Any

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from uuid import uuid4
from pathlib import Path
from threading import Lock
from collections import Counter, defaultdict

import asyncio
import io
import math
import os
import time
import json
import re
import shutil
import time
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ml.model_manager import predict_any_stock
from ml.universal_precision_v3 import historical_replay_v3
from ml.x2_live_predictor import predict_x2_hybrid
from ml.multi_model_manager import predict_any_stock_multi
from ml.training_status import start_training, get_training_status
from ml.v9_model_manager import predict_relative_strength, get_v9_status
from ml.analytics_manager import get_model_analytics


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="StockVision AI API",
    version="2.14.1",
    description=(
        "StockVision AI backend for live market data, technical indicators, "
        "BiLSTM forecasting, V9 relative-strength intelligence, model analytics, "
        "OHLC candlestick data, X2 probabilistic ranges, and live prediction validation."
    ),
)


# =========================================================
# CORS
# =========================================================

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# NSE STOCK UNIVERSE
# =========================================================

NSE_EQUITY_CSV_URL = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
)

STOCK_UNIVERSE_CACHE_SECONDS = 6 * 60 * 60

_stock_universe_cache: dict[str, Any] = {
    "loaded_at": 0.0,
    "stocks": [],
}


FALLBACK_STOCKS = [
    {"name": "Reliance Industries", "symbol": "RELIANCE.NS", "short": "RELIANCE"},
    {"name": "Tata Consultancy Services", "symbol": "TCS.NS", "short": "TCS"},
    {"name": "Infosys", "symbol": "INFY.NS", "short": "INFY"},
    {"name": "HDFC Bank", "symbol": "HDFCBANK.NS", "short": "HDFCBANK"},
    {"name": "ICICI Bank", "symbol": "ICICIBANK.NS", "short": "ICICIBANK"},
    {"name": "Axis Bank", "symbol": "AXISBANK.NS", "short": "AXISBANK"},
    {"name": "Kotak Mahindra Bank", "symbol": "KOTAKBANK.NS", "short": "KOTAKBANK"},
    {"name": "Wipro", "symbol": "WIPRO.NS", "short": "WIPRO"},
    {"name": "HCL Technologies", "symbol": "HCLTECH.NS", "short": "HCLTECH"},
    {"name": "Tata Motors", "symbol": "TATAMOTORS.NS", "short": "TATAMOTORS"},
    {"name": "State Bank of India", "symbol": "SBIN.NS", "short": "SBIN"},
    {"name": "ITC", "symbol": "ITC.NS", "short": "ITC"},
    {"name": "Larsen & Toubro", "symbol": "LT.NS", "short": "LT"},
    {"name": "Bharti Airtel", "symbol": "BHARTIARTL.NS", "short": "BHARTIARTL"},
    {"name": "Maruti Suzuki", "symbol": "MARUTI.NS", "short": "MARUTI"},
    {"name": "Sun Pharmaceutical", "symbol": "SUNPHARMA.NS", "short": "SUNPHARMA"},
]


def load_nse_stock_universe(force: bool = False) -> tuple[list[dict[str, Any]], str]:
    """
    Download the current NSE equity-security list from NSE's official archive.

    The result is cached for six hours so the UI gets a large current stock
    universe without repeatedly downloading the CSV.

    If NSE is temporarily unavailable, the app falls back to the original
    StockVision list instead of breaking the dashboard.
    """
    now = time.time()

    cached_stocks = _stock_universe_cache.get("stocks") or []
    loaded_at = float(_stock_universe_cache.get("loaded_at") or 0.0)

    if (
        cached_stocks
        and not force
        and now - loaded_at < STOCK_UNIVERSE_CACHE_SECONDS
    ):
        return cached_stocks, "NSE official cache"

    request = urllib.request.Request(
        NSE_EQUITY_CSV_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/151.0.0.0 Safari/537.36"
            ),
            "Accept": "text/csv,text/plain,*/*",
            "Referer": "https://www.nseindia.com/",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read()

        csv_text = raw.decode("utf-8-sig", errors="replace")
        frame = pd.read_csv(io.StringIO(csv_text))

        # Normalize NSE column names because the CSV can contain spaces.
        frame.columns = [
            str(column).strip().upper()
            for column in frame.columns
        ]

        symbol_column = next(
            (
                column
                for column in frame.columns
                if column in {"SYMBOL", "SYMBOL "}
            ),
            None,
        )

        name_column = next(
            (
                column
                for column in frame.columns
                if column
                in {
                    "NAME OF COMPANY",
                    "COMPANY NAME",
                    "NAME",
                }
            ),
            None,
        )

        series_column = (
            "SERIES"
            if "SERIES" in frame.columns
            else None
        )

        isin_column = next(
            (
                column
                for column in frame.columns
                if "ISIN" in column
            ),
            None,
        )

        if symbol_column is None:
            raise ValueError("NSE stock CSV does not contain a SYMBOL column.")

        # Deduplicate symbols. If NSE has multiple series for the same symbol,
        # prefer the normal EQ series where available.
        by_symbol: dict[str, dict[str, Any]] = {}

        for _, row in frame.iterrows():
            raw_symbol = str(row.get(symbol_column, "") or "").strip().upper()

            if (
                not raw_symbol
                or raw_symbol == "NAN"
            ):
                continue

            company_name = (
                str(row.get(name_column, "") or "").strip()
                if name_column
                else raw_symbol
            )

            if (
                not company_name
                or company_name.upper() == "NAN"
            ):
                company_name = raw_symbol

            series = (
                str(row.get(series_column, "") or "").strip().upper()
                if series_column
                else ""
            )

            isin = (
                str(row.get(isin_column, "") or "").strip()
                if isin_column
                else ""
            )

            item = {
                "name": company_name,
                "symbol": f"{raw_symbol}.NS",
                "short": raw_symbol,
                "series": series or None,
                "isin": (
                    isin
                    if isin and isin.upper() != "NAN"
                    else None
                ),
            }

            existing = by_symbol.get(raw_symbol)

            if (
                existing is None
                or (
                    series == "EQ"
                    and existing.get("series") != "EQ"
                )
            ):
                by_symbol[raw_symbol] = item

        stocks = sorted(
            by_symbol.values(),
            key=lambda item: (
                item["short"].lower(),
                item["name"].lower(),
            ),
        )

        if not stocks:
            raise ValueError("NSE returned an empty stock universe.")

        _stock_universe_cache["stocks"] = stocks
        _stock_universe_cache["loaded_at"] = now

        return stocks, "NSE official"

    except Exception:
        # Preserve the last successfully downloaded list when possible.
        if cached_stocks:
            return cached_stocks, "NSE cached fallback"

        return FALLBACK_STOCKS, "StockVision fallback"


# =========================================================
# SYMBOL HELPERS
# =========================================================

def normalize_symbol(symbol: str) -> str:
    """
    Convert inputs like RELIANCE -> RELIANCE.NS.
    Keep index symbols such as ^NSEI unchanged.
    Keep already-qualified tickers unchanged.
    """
    clean = str(symbol or "").strip().upper().replace(" ", "")

    if not clean:
        raise HTTPException(status_code=400, detail="Stock symbol is required.")

    if clean.startswith("^") or "." in clean:
        return clean

    return f"{clean}.NS"


# =========================================================
# SAFE VALUE HELPERS
# =========================================================

def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default

        return int(float(value))

    except (TypeError, ValueError):
        return default


def round_or_none(value: Any, digits: int = 4) -> float | None:
    number = safe_float(value)

    if number is None:
        return None

    return round(number, digits)


# =========================================================
# DATA HELPERS
# =========================================================

def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output and remove rows without valid OHLC close data.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    frame = df.copy()

    # yfinance can occasionally return MultiIndex columns.
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in frame.columns
        ]

    required = ["Open", "High", "Low", "Close"]

    for column in required:
        if column not in frame.columns:
            return pd.DataFrame()

    frame = frame.dropna(subset=["Close"])

    return frame


def latest_session(intraday: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the most recent trading session from intraday data.
    """
    if intraday.empty:
        return intraday

    index_series = pd.Series(intraday.index, index=intraday.index)

    try:
        dates = index_series.dt.date
    except Exception:
        dates = pd.Series(
            [pd.Timestamp(value).date() for value in intraday.index],
            index=intraday.index,
        )

    latest_date = dates.max()

    return intraday.loc[dates == latest_date]


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(daily_df: pd.DataFrame) -> dict[str, Any]:
    if daily_df.empty or len(daily_df) < 2:
        return {
            "sma20": None,
            "ema20": None,
            "rsi14": None,
            "macd": None,
            "macd_signal": None,
            "daily_return": None,
            "volatility20": None,
        }

    close = daily_df["Close"].astype(float)

    sma20 = close.rolling(20).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    daily_return = close.pct_change() * 100
    volatility20 = daily_return.rolling(20).std()

    return {
        # Main naming used by the current frontend
        "sma20": round_or_none(sma20.iloc[-1], 4),
        "ema20": round_or_none(ema20.iloc[-1], 4),
        "rsi14": round_or_none(rsi14.iloc[-1], 4),
        "macd": round_or_none(macd.iloc[-1], 6),
        "macd_signal": round_or_none(macd_signal.iloc[-1], 6),
        "daily_return": round_or_none(daily_return.iloc[-1], 6),
        "volatility20": round_or_none(volatility20.iloc[-1], 6),

        # Compatibility aliases
        "sma_20": round_or_none(sma20.iloc[-1], 4),
        "ema_20": round_or_none(ema20.iloc[-1], 4),
        "rsi": round_or_none(rsi14.iloc[-1], 4),
    }


# =========================================================
# RANGE CONFIGURATION
# =========================================================

RANGE_CONFIG = {
    "1d": {
        "period": "5d",
        "interval": "5m",
        "latest_only": True,
    },
    "5d": {
        "period": "5d",
        "interval": "30m",
        "latest_only": False,
    },
    "1mo": {
        "period": "1mo",
        "interval": "1d",
        "latest_only": False,
    },
    "3mo": {
        "period": "3mo",
        "interval": "1d",
        "latest_only": False,
    },
    "6mo": {
        "period": "6mo",
        "interval": "1d",
        "latest_only": False,
    },
    "ytd": {
        "period": "ytd",
        "interval": "1d",
        "latest_only": False,
    },
    "1y": {
        "period": "1y",
        "interval": "1d",
        "latest_only": False,
    },
    "5y": {
        "period": "5y",
        "interval": "1wk",
        "latest_only": False,
    },
}


# =========================================================
# CHART FORMATTER
# =========================================================

def format_chart(
    df: pd.DataFrame,
    interval: str,
) -> list[dict[str, Any]]:
    """
    IMPORTANT:
    Every chart point contains both:
      - price: keeps the existing line-chart UI working
      - open/high/low/close: enables the new real candlestick UI
    """
    if df.empty:
        return []

    points: list[dict[str, Any]] = []

    intraday = interval in {
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
    }

    for timestamp, row in df.iterrows():
        ts = pd.Timestamp(timestamp)

        if intraday:
            label = ts.strftime("%H:%M")
        elif interval in {"1wk", "1mo", "3mo"}:
            label = ts.strftime("%d %b %Y")
        else:
            label = ts.strftime("%d %b")

        close = safe_float(row.get("Close"))

        if close is None:
            continue

        points.append(
            {
                "time": label,
                "timestamp": ts.isoformat(),
                "price": round(close, 4),

                # Real OHLC values for candlesticks
                "open": round_or_none(row.get("Open"), 4),
                "high": round_or_none(row.get("High"), 4),
                "low": round_or_none(row.get("Low"), 4),
                "close": round(close, 4),

                "volume": safe_int(row.get("Volume")),
            }
        )

    return points


# =========================================================
# LIVE SESSION SUMMARY
# =========================================================

def get_live_session_summary(
    ticker: yf.Ticker,
    daily_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Pull recent intraday bars for the current/latest available trading session.
    Fall back to the latest daily candle if intraday data is unavailable.
    """
    intraday = clean_history(
        ticker.history(
            period="5d",
            interval="5m",
            auto_adjust=False,
            prepost=False,
        )
    )

    session = latest_session(intraday)

    if not session.empty:
        current_price = safe_float(session["Close"].iloc[-1], 0.0) or 0.0
        day_open = safe_float(session["Open"].iloc[0], current_price) or current_price
        day_high = safe_float(session["High"].max(), current_price) or current_price
        day_low = safe_float(session["Low"].min(), current_price) or current_price
        day_volume = safe_int(session["Volume"].sum())
        latest_market_date = pd.Timestamp(session.index[-1]).strftime("%Y-%m-%d")
    elif not daily_df.empty:
        latest = daily_df.iloc[-1]

        current_price = safe_float(latest.get("Close"), 0.0) or 0.0
        day_open = safe_float(latest.get("Open"), current_price) or current_price
        day_high = safe_float(latest.get("High"), current_price) or current_price
        day_low = safe_float(latest.get("Low"), current_price) or current_price
        day_volume = safe_int(latest.get("Volume"))
        latest_market_date = pd.Timestamp(daily_df.index[-1]).strftime("%Y-%m-%d")
    else:
        raise HTTPException(status_code=404, detail="No market data available.")

    previous_close = None

    if len(daily_df) >= 2:
        latest_daily_date = pd.Timestamp(daily_df.index[-1]).date()

        # If the latest daily candle is today's/current session,
        # previous close is the row before it.
        if latest_market_date and latest_daily_date == pd.Timestamp(latest_market_date).date():
            previous_close = safe_float(daily_df["Close"].iloc[-2])
        else:
            previous_close = safe_float(daily_df["Close"].iloc[-1])

    if previous_close is None or previous_close == 0:
        previous_close = current_price

    change = current_price - previous_close

    change_percent = (
        (change / previous_close) * 100
        if previous_close
        else 0.0
    )

    return {
        "price": round(current_price, 4),
        "open": round(day_open, 4),
        "high": round(day_high, 4),
        "low": round(day_low, 4),
        "volume": day_volume,
        "previous_close": round(previous_close, 4),
        "change": round(change, 4),
        "change_percent": round(change_percent, 4),
        "latest_market_date": latest_market_date,
    }



# =========================================================
# NEWS + SENTIMENT
# =========================================================

POSITIVE_NEWS_WORDS = {
    "beat", "beats", "growth", "grow", "grows", "gain", "gains",
    "surge", "surges", "rise", "rises", "rally", "strong", "record",
    "profit", "profits", "upgrade", "upgrades", "positive", "bullish",
    "expansion", "expands", "investment", "invest", "launch", "launches",
    "wins", "win", "improve", "improves", "improved", "higher",
    "outperform", "outperforms", "deal", "partnership", "approval",
    "approved", "boost", "boosts", "recovery", "resilient", "dividend",
    "buyback", "breakthrough",
}

NEGATIVE_NEWS_WORDS = {
    "miss", "misses", "fall", "falls", "drop", "drops", "decline",
    "declines", "slump", "slumps", "weak", "loss", "losses",
    "downgrade", "downgrades", "negative", "bearish", "risk", "risks",
    "probe", "investigation", "lawsuit", "fraud", "fine", "fines",
    "penalty", "cut", "cuts", "lower", "slowdown", "concern", "concerns",
    "pressure", "uncertainty", "debt", "default", "crash", "warning",
    "warns", "delay", "delays",
}

NEWS_TOPIC_KEYWORDS = {
    "Earnings": {
        "earnings", "profit", "revenue", "results", "quarter",
        "q1", "q2", "q3", "q4",
    },
    "Energy": {
        "energy", "refining", "petroleum", "gas", "solar",
        "hydrogen", "renewable",
    },
    "Oil": {
        "oil", "crude", "brent",
    },
    "Retail": {
        "retail", "consumer", "store", "stores",
    },
    "Telecom": {
        "telecom", "jio", "5g", "arpu", "subscriber", "subscribers",
    },
    "Investment": {
        "investment", "invest", "capex", "acquisition", "stake",
    },
    "Markets": {
        "market", "shares", "stock", "nifty", "sensex", "trading",
    },
    "Regulation": {
        "regulator", "regulatory", "sebi", "government", "policy",
        "approval",
    },
}



# Stock-specific aliases used to reject unrelated Yahoo Finance recommendations.
NEWS_RELEVANCE_ALIASES = {
    "RELIANCE.NS": {
        "reliance industries",
        "reliance",
        "ril",
        "reliance retail",
        "reliance brands",
        "jio",
        "jio platforms",
        "mukesh ambani",
    },
    "TCS.NS": {
        "tata consultancy services",
        "tcs",
    },
    "INFY.NS": {
        "infosys",
        "infy",
    },
    "HDFCBANK.NS": {
        "hdfc bank",
        "hdfcbank",
    },
    "ICICIBANK.NS": {
        "icici bank",
        "icicibank",
    },
    "SBIN.NS": {
        "state bank of india",
        "sbi",
    },
}

# Keep the sentiment/news page focused on recent market context.
NEWS_LOOKBACK_DAYS = 45


def _news_aliases_for_symbol(
    symbol: str,
):
    normalized = normalize_symbol(
        symbol
    )

    aliases = set(
        NEWS_RELEVANCE_ALIASES.get(
            normalized,
            set(),
        )
    )

    short = normalized.replace(
        ".NS",
        "",
    ).lower()

    aliases.add(
        short
    )

    return {
        alias.lower().strip()
        for alias in aliases
        if alias
    }


def _article_is_relevant(
    symbol: str,
    title: str,
    summary: str,
):
    """
    Require a direct mention of the stock/company/known brand alias.

    This removes Yahoo Finance recommendations that are not actually about
    the requested stock.
    """
    text_value = (
        f"{title} {summary}"
        .lower()
    )

    aliases = _news_aliases_for_symbol(
        symbol
    )

    # Jio Financial Services is a separate listed company.
    # Do not treat a standalone Jio Financial article as RELIANCE.NS news
    # unless it also directly mentions Reliance / Reliance Industries / RIL
    # / Reliance Retail / Reliance Brands / Mukesh Ambani.
    if (
        normalize_symbol(
            symbol
        ) == "RELIANCE.NS"
        and "jio financial" in text_value
    ):
        parent_aliases = {
            "reliance industries",
            "reliance",
            "ril",
            "reliance retail",
            "reliance brands",
            "mukesh ambani",
        }

        parent_mentioned = False

        for parent_alias in parent_aliases:
            if (
                len(parent_alias) <= 4
                and parent_alias.replace(
                    " ",
                    "",
                ).isalnum()
            ):
                if re.search(
                    rf"(?<![a-z0-9]){re.escape(parent_alias)}(?![a-z0-9])",
                    text_value,
                ):
                    parent_mentioned = True
                    break
            else:
                if parent_alias in text_value:
                    parent_mentioned = True
                    break

        if not parent_mentioned:
            return False

    for alias in aliases:
        # Short aliases such as RIL should match as a whole token only.
        if len(alias) <= 4 and alias.replace(" ", "").isalnum():
            if re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                text_value,
            ):
                return True

        else:
            if alias in text_value:
                return True

    return False


def _article_is_recent(
    published_at_utc,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
):
    if not published_at_utc:
        # Keep undated items only when Yahoo gives no usable timestamp.
        return True

    try:
        timestamp = pd.Timestamp(
            published_at_utc
        )

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                "UTC"
            )
        else:
            timestamp = timestamp.tz_convert(
                "UTC"
            )

        cutoff = (
            pd.Timestamp.now(
                tz="UTC"
            )
            - pd.Timedelta(
                days=lookback_days
            )
        )

        return timestamp >= cutoff

    except Exception:
        return True


def _relevant_text_for_sentiment(
    symbol: str,
    title: str,
    summary: str,
):
    """
    Prefer sentences that directly mention the requested company/brand.

    This reduces cases where a long article is mainly about another company
    but mentions the requested stock only as a partner or comparison.
    """
    aliases = _news_aliases_for_symbol(
        symbol
    )

    combined = " ".join(
        part
        for part in [
            _clean_news_text(
                title
            ),
            _clean_news_text(
                summary
            ),
        ]
        if part
    )

    if not combined:
        return ""

    sentences = re.split(
        r"(?<=[.!?])\s+",
        combined,
    )

    relevant_sentences = []

    for sentence in sentences:
        sentence_lower = sentence.lower()

        matched = False

        for alias in aliases:
            if len(alias) <= 4 and alias.replace(" ", "").isalnum():
                if re.search(
                    rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                    sentence_lower,
                ):
                    matched = True
                    break
            else:
                if alias in sentence_lower:
                    matched = True
                    break

        if matched:
            relevant_sentences.append(
                sentence
            )

    if relevant_sentences:
        return " ".join(
            relevant_sentences
        )

    # Fallback to headline if the mention was not sentence-splittable.
    return _clean_news_text(
        title
    )


def _clean_news_text(value):
    if value is None:
        return ""

    return str(value).strip()


def _extract_news_timestamp(value):
    if value is None:
        return None

    try:
        if isinstance(value, (int, float)):
            timestamp = pd.Timestamp(
                value,
                unit="s",
                tz="UTC",
            )
        else:
            timestamp = pd.Timestamp(
                value
            )

            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(
                    "UTC"
                )
            else:
                timestamp = timestamp.tz_convert(
                    "UTC"
                )

        return timestamp.isoformat()

    except Exception:
        return None


def _extract_yfinance_news(
    symbol: str,
    limit: int = 50,
):
    """
    Parse both old and newer yfinance news response shapes.
    """
    ticker = yf.Ticker(
        symbol
    )

    raw_items = []

    try:
        raw_items = ticker.get_news(
            count=limit
        )
    except Exception:
        try:
            raw_items = ticker.news
        except Exception:
            raw_items = []

    if not raw_items:
        return []

    parsed = []

    for raw in raw_items[
        :limit
    ]:
        if not isinstance(
            raw,
            dict,
        ):
            continue

        content = raw.get(
            "content"
        )

        if isinstance(
            content,
            dict,
        ):
            title = (
                content.get(
                    "title"
                )
                or content.get(
                    "headline"
                )
                or ""
            )

            summary = (
                content.get(
                    "summary"
                )
                or content.get(
                    "description"
                )
                or ""
            )

            provider = (
                content.get(
                    "provider"
                )
                or content.get(
                    "publisher"
                )
                or {}
            )

            if isinstance(
                provider,
                dict,
            ):
                source = (
                    provider.get(
                        "displayName"
                    )
                    or provider.get(
                        "name"
                    )
                    or "Yahoo Finance"
                )
            else:
                source = (
                    _clean_news_text(
                        provider
                    )
                    or "Yahoo Finance"
                )

            canonical = (
                content.get(
                    "canonicalUrl"
                )
                or content.get(
                    "clickThroughUrl"
                )
                or {}
            )

            if isinstance(
                canonical,
                dict,
            ):
                url = (
                    canonical.get(
                        "url"
                    )
                    or canonical.get(
                        "link"
                    )
                    or ""
                )
            else:
                url = (
                    _clean_news_text(
                        canonical
                    )
                )

            published = (
                content.get(
                    "pubDate"
                )
                or content.get(
                    "displayTime"
                )
                or content.get(
                    "publishedAt"
                )
            )

        else:
            title = (
                raw.get(
                    "title"
                )
                or raw.get(
                    "headline"
                )
                or ""
            )

            summary = (
                raw.get(
                    "summary"
                )
                or raw.get(
                    "description"
                )
                or ""
            )

            source = (
                raw.get(
                    "publisher"
                )
                or raw.get(
                    "provider"
                )
                or "Yahoo Finance"
            )

            if isinstance(
                source,
                dict,
            ):
                source = (
                    source.get(
                        "displayName"
                    )
                    or source.get(
                        "name"
                    )
                    or "Yahoo Finance"
                )

            url = (
                raw.get(
                    "link"
                )
                or raw.get(
                    "url"
                )
                or ""
            )

            published = (
                raw.get(
                    "providerPublishTime"
                )
                or raw.get(
                    "pubDate"
                )
                or raw.get(
                    "publishedAt"
                )
            )

        title = (
            _clean_news_text(
                title
            )
        )

        if not title:
            continue

        parsed.append(
            {
                "title": title,
                "summary": (
                    _clean_news_text(
                        summary
                    )
                ),
                "source": (
                    _clean_news_text(
                        source
                    )
                    or "Yahoo Finance"
                ),
                "url": (
                    _clean_news_text(
                        url
                    )
                ),
                "published_at_utc": (
                    _extract_news_timestamp(
                        published
                    )
                ),
            }
        )

    # De-duplicate by headline.
    unique = []
    seen = set()

    for item in parsed:
        key = (
            item[
                "title"
            ]
            .lower()
            .strip()
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            item
        )

    return unique


def _score_news_sentiment(
    title: str,
    summary: str,
    symbol: str | None = None,
):
    """
    Transparent descriptive sentiment.

    If symbol is supplied, prefer only the sentences that directly mention
    that stock/company/brand. The score is NOT calibrated probability or
    confidence.
    """
    if symbol:
        relevant_text = _relevant_text_for_sentiment(
            symbol,
            title,
            summary,
        )
    else:
        relevant_text = (
            f"{title} {summary}"
        )

    text_value = relevant_text.lower()

    tokens = re.findall(
        r"[a-z0-9]+",
        text_value,
    )

    positive_hits = sum(
        1
        for token in tokens
        if token
        in POSITIVE_NEWS_WORDS
    )

    negative_hits = sum(
        1
        for token in tokens
        if token
        in NEGATIVE_NEWS_WORDS
    )

    raw_score = (
        positive_hits
        - negative_hits
    )

    if raw_score >= 1:
        label = "POSITIVE"

    elif raw_score <= -1:
        label = "NEGATIVE"

    else:
        label = "NEUTRAL"

    normalized_score = float(
        np.tanh(
            raw_score /
            3.0
        )
    )

    return (
        label,
        normalized_score,
    )


def _extract_news_topics(
    title: str,
    summary: str,
):
    text = (
        f"{title} {summary}"
        .lower()
    )

    matched = []

    for (
        topic,
        keywords,
    ) in NEWS_TOPIC_KEYWORDS.items():
        if any(
            keyword in text
            for keyword
            in keywords
        ):
            matched.append(
                topic
            )

    return matched


def build_news_sentiment_payload(
    symbol: str,
):
    normalized = normalize_symbol(
        symbol
    )

    news_items = _extract_yfinance_news(
        normalized,
        limit=50,
    )

    relevant_items = [
        item
        for item in news_items
        if _article_is_relevant(
            normalized,
            item.get(
                "title",
                "",
            ),
            item.get(
                "summary",
                "",
            ),
        )
        and _article_is_recent(
            item.get(
                "published_at_utc"
            )
        )
    ]

    analyzed = []

    for item in relevant_items:
        (
            label,
            score,
        ) = _score_news_sentiment(
            item.get(
                "title",
                "",
            ),
            item.get(
                "summary",
                "",
            ),
            normalized,
        )

        topics = _extract_news_topics(
            item.get(
                "title",
                "",
            ),
            item.get(
                "summary",
                "",
            ),
        )

        analyzed.append(
            {
                **item,
                "sentiment": label,
                "sentiment_score": round(
                    score,
                    6,
                ),
                "topics": topics,
            }
        )

    counts = Counter(
        item[
            "sentiment"
        ]
        for item in analyzed
    )

    total = len(
        analyzed
    )

    def percent(
        label,
    ):
        return (
            counts.get(
                label,
                0,
            )
            / total
            * 100.0
            if total
            else 0.0
        )

    positive_percent = percent(
        "POSITIVE"
    )

    neutral_percent = percent(
        "NEUTRAL"
    )

    negative_percent = percent(
        "NEGATIVE"
    )

    if total:
        overall_score = float(
            np.mean(
                [
                    item[
                        "sentiment_score"
                    ]
                    for item
                    in analyzed
                ]
            )
        )

        if overall_score > 0.12:
            overall_sentiment = (
                "POSITIVE"
            )

        elif overall_score < -0.12:
            overall_sentiment = (
                "NEGATIVE"
            )

        else:
            overall_sentiment = (
                "NEUTRAL"
            )

    else:
        overall_score = None
        overall_sentiment = (
            "NO_DATA"
        )

    source_counts = Counter(
        item[
            "source"
        ]
        for item in analyzed
        if item.get(
            "source"
        )
    )

    sources = []

    for (
        source,
        count,
    ) in source_counts.most_common(
        8
    ):
        sources.append(
            {
                "source": source,
                "count": int(
                    count
                ),
                "percent": round(
                    (
                        count
                        / total
                        * 100.0
                    )
                    if total
                    else 0.0,
                    2,
                ),
            }
        )

    topic_counts = Counter()

    for item in analyzed:
        for topic in item.get(
            "topics",
            [],
        ):
            topic_counts[
                topic
            ] += 1

    trending_topics = [
        {
            "topic": topic,
            "count": int(
                count
            ),
        }
        for (
            topic,
            count,
        ) in topic_counts.most_common(
            8
        )
    ]

    daily = defaultdict(
        lambda: {
            "POSITIVE": 0,
            "NEUTRAL": 0,
            "NEGATIVE": 0,
        }
    )

    for item in analyzed:
        published = item.get(
            "published_at_utc"
        )

        if not published:
            continue

        try:
            day = (
                pd.Timestamp(
                    published
                )
                .date()
                .isoformat()
            )
        except Exception:
            continue

        daily[
            day
        ][
            item[
                "sentiment"
            ]
        ] += 1

    trend = []

    for day in sorted(
        daily.keys()
    )[
        -7:
    ]:
        bucket = daily[
            day
        ]

        day_total = sum(
            bucket.values()
        )

        trend.append(
            {
                "date": day,
                "positive_percent": round(
                    (
                        bucket[
                            "POSITIVE"
                        ]
                        / day_total
                        * 100.0
                    )
                    if day_total
                    else 0.0,
                    2,
                ),
                "neutral_percent": round(
                    (
                        bucket[
                            "NEUTRAL"
                        ]
                        / day_total
                        * 100.0
                    )
                    if day_total
                    else 0.0,
                    2,
                ),
                "negative_percent": round(
                    (
                        bucket[
                            "NEGATIVE"
                        ]
                        / day_total
                        * 100.0
                    )
                    if day_total
                    else 0.0,
                    2,
                ),
                "article_count": int(
                    day_total
                ),
            }
        )

    dominant_topic = (
        trending_topics[
            0
        ][
            "topic"
        ]
        if trending_topics
        else None
    )

    if total == 0:
        mood_summary = (
            "No recent Yahoo Finance news items "
            "were available for this stock."
        )

    elif (
        overall_sentiment
        == "POSITIVE"
    ):
        mood_summary = (
            "Recent coverage leans positive"
            + (
                f", with {dominant_topic.lower()} "
                "appearing frequently."
                if dominant_topic
                else "."
            )
        )

    elif (
        overall_sentiment
        == "NEGATIVE"
    ):
        mood_summary = (
            "Recent coverage leans negative"
            + (
                f", with {dominant_topic.lower()} "
                "appearing frequently."
                if dominant_topic
                else "."
            )
        )

    else:
        mood_summary = (
            "Recent coverage is mixed or neutral"
            + (
                f", with {dominant_topic.lower()} "
                "among the common themes."
                if dominant_topic
                else "."
            )
        )

    return {
        "symbol": normalized,
        "source": (
            "Yahoo Finance via yfinance"
        ),
        "sentiment_method": (
            "Rule-based relevance-filtered headline/summary lexicon. "
            "Scores are descriptive, not calibrated probabilities."
        ),
        "raw_articles_received": int(
            len(
                news_items
            )
        ),
        "relevant_articles_analyzed": int(
            total
        ),
        "lookback_days": int(
            NEWS_LOOKBACK_DAYS
        ),
        "total_articles_analyzed": int(
            total
        ),
        "overall_sentiment": (
            overall_sentiment
        ),
        "overall_score": (
            round(
                overall_score,
                6,
            )
            if overall_score
            is not None
            else None
        ),
        "breakdown": {
            "positive_count": int(
                counts.get(
                    "POSITIVE",
                    0,
                )
            ),
            "neutral_count": int(
                counts.get(
                    "NEUTRAL",
                    0,
                )
            ),
            "negative_count": int(
                counts.get(
                    "NEGATIVE",
                    0,
                )
            ),
            "positive_percent": round(
                positive_percent,
                2,
            ),
            "neutral_percent": round(
                neutral_percent,
                2,
            ),
            "negative_percent": round(
                negative_percent,
                2,
            ),
        },
        "latest_news": analyzed[
            :20
        ],
        "trend": trend,
        "sources": sources,
        "trending_topics": (
            trending_topics
        ),
        "market_mood_summary": (
            mood_summary
        ),
        "generated_at_utc": (
            pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
        ),
    }


# =========================================================
# ROOT
# =========================================================

@app.get("/news-sentiment/{symbol}")
def news_sentiment(
    symbol: str,
):
    try:
        return build_news_sentiment_payload(
            symbol
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "News & sentiment failed for "
                f"{normalize_symbol(symbol)}: {exc}"
            ),
        )


@app.get("/stocks")
def stocks(force_refresh: bool = False):
    """
    Return the current NSE equity-security universe for frontend search,
    Markets, and Stock Comparison.
    """
    items, source = load_nse_stock_universe(force=force_refresh)

    return {
        "count": len(items),
        "source": source,
        "updated_at_epoch": int(
            _stock_universe_cache.get("loaded_at") or 0
        ),
        "stocks": items,
    }


@app.get("/")
def root():
    return {
        "message": "StockVision API is running",
        "version": "2.14.1",
        "engines": {
            "live_market": "ready",
            "candlestick_market": "ready",
            "next_day_prediction": "baseline-safe + X2 probabilistic range",
            "v3_multi_horizon": "available",
            "v9_relative_strength": "available",
            "model_analytics": "available",
            "news_sentiment": "ready",
            "prediction_history": "live/replay history + aligned direction evaluation + full-NSE auto capture",
        },
    }


# =========================================================
# STOCK DATA
# =========================================================

@app.get("/stock/{symbol}")
def get_stock(
    symbol: str,
    range: str = "1d",
):
    normalized = normalize_symbol(symbol)

    selected_range = str(range or "1d").lower()

    config = RANGE_CONFIG.get(selected_range)

    if config is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid range. Supported ranges: "
                + ", ".join(RANGE_CONFIG.keys())
            ),
        )

    try:
        ticker = yf.Ticker(normalized)

        # Daily history is used for reliable previous close + indicators.
        daily_df = clean_history(
            ticker.history(
                period="1y",
                interval="1d",
                auto_adjust=False,
                prepost=False,
            )
        )

        if daily_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for {normalized}.",
            )

        summary = get_live_session_summary(
            ticker=ticker,
            daily_df=daily_df,
        )

        chart_df = clean_history(
            ticker.history(
                period=config["period"],
                interval=config["interval"],
                auto_adjust=False,
                prepost=False,
            )
        )

        if config["latest_only"] and not chart_df.empty:
            chart_df = latest_session(chart_df)

        chart = format_chart(
            chart_df,
            config["interval"],
        )

        indicators = calculate_indicators(daily_df)

        return {
            "symbol": normalized,
            "range": selected_range,

            **summary,

            # Existing frontend
            "chart": chart,

            # Technical indicators
            "indicators": indicators,

            # Compatibility fields for older UI code
            "sma20": indicators.get("sma20"),
            "ema20": indicators.get("ema20"),
            "rsi14": indicators.get("rsi14"),
            "macd": indicators.get("macd"),
            "macd_signal": indicators.get("macd_signal"),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load {normalized}: {exc}",
        )


# =========================================================
# REAL CANDLESTICK DATA
# =========================================================

ALLOWED_CANDLE_PERIODS = {
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "ytd",
    "max",
}

ALLOWED_CANDLE_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}


@app.get("/stock-candles/{symbol}")
def get_stock_candles(
    symbol: str,
    period: str = "3mo",
    interval: str = "1d",
):
    """
    Real OHLC + volume data specifically for a candlestick chart.

    Examples:
      /stock-candles/RELIANCE?period=1d&interval=5m
      /stock-candles/RELIANCE?period=5d&interval=30m
      /stock-candles/RELIANCE?period=3mo&interval=1d
    """
    normalized = normalize_symbol(symbol)

    selected_period = str(period or "3mo").lower()
    selected_interval = str(interval or "1d").lower()

    if selected_period not in ALLOWED_CANDLE_PERIODS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported candlestick period: {selected_period}",
        )

    if selected_interval not in ALLOWED_CANDLE_INTERVALS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported candlestick interval: {selected_interval}",
        )

    try:
        ticker = yf.Ticker(normalized)

        df = clean_history(
            ticker.history(
                period=selected_period,
                interval=selected_interval,
                auto_adjust=False,
                prepost=False,
            )
        )

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No candlestick data found for {normalized}.",
            )

        # For a 1-day intraday chart, keep only the latest actual session.
        if selected_period == "1d" or (
            selected_period == "5d"
            and selected_interval in {"1m", "2m", "5m", "15m"}
        ):
            if selected_period == "1d":
                df = latest_session(df)

        candles = format_chart(
            df,
            selected_interval,
        )

        return {
            "symbol": normalized,
            "period": selected_period,
            "interval": selected_interval,
            "count": len(candles),
            "candles": candles,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load candle data for {normalized}: {exc}",
        )


# =========================================================
# LIVE PREDICTION TRACKER / VALIDATION
# =========================================================

PREDICTION_HISTORY_DIR = Path(__file__).resolve().parent / "data"
PREDICTION_HISTORY_FILE = (
    PREDICTION_HISTORY_DIR / "live_prediction_history.json"
)

PREDICTION_HISTORY_BACKUP_FILE = (
    PREDICTION_HISTORY_DIR / "live_prediction_history.backup.json"
)

PREDICTION_REPLAY_DIR = (
    PREDICTION_HISTORY_DIR / "historical_replay"
)

PREDICTION_REPLAY_LOCK = Lock()

PREDICTION_TRACKED_SYMBOLS_FILE = (
    PREDICTION_HISTORY_DIR / "prediction_tracked_symbols.json"
)

PREDICTION_FULL_UNIVERSE_STATE_FILE = (
    PREDICTION_HISTORY_DIR / "prediction_full_universe_state.json"
)

PREDICTION_HISTORY_LOCK = Lock()
PREDICTION_TRACK_LOCK = Lock()
PREDICTION_FULL_UNIVERSE_LOCK = Lock()

FULL_UNIVERSE_AUTO_CAPTURE = True
FULL_UNIVERSE_DELAY_SECONDS = 1.0

_full_universe_task = None

# Auto-history checks are lightweight because the model runs only when a new
# completed trading-day base date is missing from a tracked stock's history.
AUTO_HISTORY_CHECK_SECONDS = 30 * 60

_prediction_history_worker_task = None
_prediction_history_worker_last_run_utc = None
_prediction_history_worker_last_error = None

# A very small neutral band avoids calling tiny moves directionally meaningful.
PREDICTION_DIRECTION_NEUTRAL_BAND_PCT = 0.10


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)

        if np.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return None


def _direction_from_return(return_pct: float | None) -> str | None:
    if return_pct is None:
        return None

    if return_pct > PREDICTION_DIRECTION_NEUTRAL_BAND_PCT:
        return "BULLISH"

    if return_pct < -PREDICTION_DIRECTION_NEUTRAL_BAND_PCT:
        return "BEARISH"

    return "NEUTRAL"


def _read_prediction_history_file(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    return (
        data
        if isinstance(
            data,
            list,
        )
        else []
    )


def _load_prediction_history() -> list[dict[str, Any]]:
    """
    Load the all-time saved live-prediction history.

    If the main JSON is damaged, automatically fall back to the last known
    good backup instead of silently losing the visible history.
    """
    PREDICTION_HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        return _read_prediction_history_file(
            PREDICTION_HISTORY_FILE
        )

    except Exception:
        try:
            return _read_prediction_history_file(
                PREDICTION_HISTORY_BACKUP_FILE
            )
        except Exception:
            return []


def _atomic_json_replace(
    path: Path,
    payload: Any,
) -> None:
    """
    Windows-safe atomic JSON writer.

    Uses a unique temp filename plus retrying os.replace() so the full-NSE
    worker and normal UI requests do not fight over one shared *.tmp file.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        f"{path.name}.{uuid4().hex}.tmp"
    )

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
            )

            file.flush()

            try:
                os.fsync(
                    file.fileno()
                )
            except Exception:
                pass

        last_error = None

        for attempt in range(
            8
        ):
            try:
                os.replace(
                    temp_path,
                    path,
                )
                return

            except PermissionError as exc:
                last_error = exc
                time.sleep(
                    0.05
                    * (
                        attempt
                        + 1
                    )
                )

        if last_error is not None:
            raise last_error

    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


def _save_prediction_history(
    records: list[dict[str, Any]],
) -> None:
    """
    Atomic save + last-good backup.

    History is intentionally NOT truncated: this file is the user's
    all-time live prediction audit log.
    """
    PREDICTION_HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preserve a last-good copy of the existing main file before replacing it.
    if PREDICTION_HISTORY_FILE.exists():
        try:
            _ = _read_prediction_history_file(
                PREDICTION_HISTORY_FILE
            )

            shutil.copy2(
                PREDICTION_HISTORY_FILE,
                PREDICTION_HISTORY_BACKUP_FILE,
            )
        except Exception:
            # Do not overwrite a good backup with a damaged main file.
            pass

    _atomic_json_replace(
        PREDICTION_HISTORY_FILE,
        records,
    )


def _load_tracked_prediction_symbols() -> list[str]:
    """
    Persistent set of stocks whose daily prediction history should continue
    growing automatically while the StockVision backend is running.
    """
    PREDICTION_HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    symbols = []

    try:
        if PREDICTION_TRACKED_SYMBOLS_FILE.exists():
            with PREDICTION_TRACKED_SYMBOLS_FILE.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(
                    file
                )

            if isinstance(
                data,
                list,
            ):
                symbols.extend(
                    str(item)
                    for item in data
                    if item
                )

    except Exception:
        pass

    # Existing history is also a tracking source. This migrates users from
    # earlier StockVision history versions automatically.
    try:
        symbols.extend(
            item.get(
                "symbol"
            )
            for item in _load_prediction_history()
            if item.get(
                "symbol"
            )
        )
    except Exception:
        pass

    normalized = []

    seen = set()

    for item in symbols:
        try:
            symbol = normalize_symbol(
                item
            )
        except Exception:
            continue

        if symbol in seen:
            continue

        seen.add(
            symbol
        )
        normalized.append(
            symbol
        )

    return sorted(
        normalized
    )


def _save_tracked_prediction_symbols(
    symbols: list[str],
) -> None:
    PREDICTION_HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean = sorted(
        {
            normalize_symbol(
                symbol
            )
            for symbol in symbols
            if symbol
        }
    )

    _atomic_json_replace(
        PREDICTION_TRACKED_SYMBOLS_FILE,
        clean,
    )


def _track_prediction_symbol(
    symbol: str,
) -> list[str]:
    normalized = normalize_symbol(
        symbol
    )

    with PREDICTION_TRACK_LOCK:
        symbols = _load_tracked_prediction_symbols()

        if normalized not in symbols:
            symbols.append(
                normalized
            )

        _save_tracked_prediction_symbols(
            symbols
        )

        return sorted(
            set(
                symbols
            )
        )


def _untrack_prediction_symbol(
    symbol: str,
) -> list[str]:
    normalized = normalize_symbol(
        symbol
    )

    with PREDICTION_TRACK_LOCK:
        symbols = [
            item
            for item in _load_tracked_prediction_symbols()
            if item != normalized
        ]

        _save_tracked_prediction_symbols(
            symbols
        )

        return symbols


def _default_full_universe_state() -> dict[str, Any]:
    return {
        "running": False,
        "session_date": None,
        "started_at_utc": None,
        "updated_at_utc": None,
        "finished_at_utc": None,
        "total": 0,
        "processed": 0,
        "captured": 0,
        "already_saved": 0,
        "failed": 0,
        "data_unavailable": 0,
        "model_data_stale": 0,
        "waiting": 0,
        "last_symbol": None,
        "last_status": None,
        "last_detail": None,
        "completed": False,
        "symbols_with_errors": [],
    }


def _load_full_universe_state() -> dict[str, Any]:
    if not PREDICTION_FULL_UNIVERSE_STATE_FILE.exists():
        return _default_full_universe_state()

    try:
        with PREDICTION_FULL_UNIVERSE_STATE_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(
                file
            )

        state = _default_full_universe_state()

        if isinstance(
            data,
            dict,
        ):
            state.update(
                data
            )

        return state

    except Exception:
        return _default_full_universe_state()


def _save_full_universe_state(
    state: dict[str, Any],
) -> None:
    PREDICTION_HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    state[
        "updated_at_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    _atomic_json_replace(
        PREDICTION_FULL_UNIVERSE_STATE_FILE,
        state,
    )


def _full_universe_session_date() -> str | None:
    """
    Use the latest completed NIFTY 50 session as the batch key.
    """
    try:
        session_date, _ = _latest_daily_market_snapshot(
            "^NSEI"
        )

        return session_date

    except Exception:
        now_ist = datetime.now(
            NSE_TIMEZONE
        )

        # Weekend fallback: use the most recent weekday.
        date_value = now_ist.date()

        while date_value.weekday() >= 5:
            date_value -= timedelta(
                days=1
            )

        return date_value.isoformat()


def _full_universe_status_payload() -> dict[str, Any]:
    state = _load_full_universe_state()

    total = int(
        state.get(
            "total"
        )
        or 0
    )

    processed = int(
        state.get(
            "processed"
        )
        or 0
    )

    percent = (
        round(
            (
                processed
                / total
            )
            * 100.0,
            2,
        )
        if total
        else 0.0
    )

    return {
        **state,
        "progress_percent": percent,
        "auto_capture": FULL_UNIVERSE_AUTO_CAPTURE,
        "rate_limit_delay_seconds": FULL_UNIVERSE_DELAY_SECONDS,
        "note": (
            "StockVision attempts every current NSE symbol. Live inference "
            "supplements stale Yahoo 1d data with the latest completed 5-minute "
            "session when available. Some very new/suspended securities can "
            "still remain unavailable."
        ),
    }


def _latest_daily_market_snapshot(
    symbol: str,
) -> tuple[str | None, float | None]:
    """
    Return the latest COMPLETED daily trading date + close.

    This prevents an unfinished intraday daily candle from becoming the base
    close of a next-trading-day audit record.
    """
    earliest = (
        datetime.now(
            NSE_TIMEZONE
        ).date()
        - timedelta(
            days=25
        )
    ).isoformat()

    try:
        history = _download_daily_validation_history(
            symbol,
            earliest,
        )
    except Exception:
        # Final compatibility fallback.
        history = yf.Ticker(
            symbol
        ).history(
            period="1mo",
            interval="1d",
            auto_adjust=False,
            prepost=False,
        )

        history = _filter_completed_daily_rows(
            history
        )

    if history.empty:
        return None, None

    last_index = history.index[
        -1
    ]

    last_close = _safe_float(
        history[
            "Close"
        ].iloc[
            -1
        ]
    )

    base_date = (
        pd.Timestamp(
            last_index
        )
        .date()
        .isoformat()
    )

    return (
        base_date,
        last_close,
    )


def record_live_prediction(
    symbol: str,
    result: dict[str, Any],
) -> None:
    """
    Save ONE IMMUTABLE live prediction per symbol/base trading date.

    Important:
    - First saved forecast for that base date is frozen as the audit record.
    - Refreshing/searching again on the same base date does not rewrite what
      the model originally predicted.
    - A new market base date creates a new row.
    - Rows are retained all-time; no 50/5000-record truncation is applied here.
    """
    # Live forecast history is a closing-day audit. During an open NSE session
    # the user can still view live predictions, but the immutable daily snapshot
    # is not captured until the session is completed.
    if not _prediction_auto_capture_window_open():
        return

    production_central_price = _safe_float(
        result.get(
            "predicted_price"
        )
    )

    predicted_return = _safe_float(
        result.get(
            "predicted_return_percent"
        )
    )

    current_close = _safe_float(
        result.get(
            "current_close"
        )
    )

    x2_point_price = _safe_float(
        (
            result.get(
                "experimental_x2_point"
            )
            or {}
        ).get(
            "price"
        )
    )

    # This is the actual forecast point shown/audited in history:
    # X2 experimental point when available, otherwise the universal model point.
    forecast_point_price = (
        x2_point_price
        if x2_point_price is not None
        else production_central_price
    )

    if forecast_point_price is None:
        return

    try:
        (
            base_date,
            market_close,
        ) = _latest_daily_market_snapshot(
            symbol
        )

    except Exception:
        base_date = datetime.now(
            NSE_TIMEZONE
        ).date().isoformat()

        market_close = None

    if not base_date:
        base_date = datetime.now(
            NSE_TIMEZONE
        ).date().isoformat()

    # For an audit record, the completed base-date close is authoritative.
    if market_close is not None:
        current_close = market_close

    forecast_return = None

    if (
        current_close is not None
        and current_close != 0
    ):
        forecast_return = (
            (
                forecast_point_price
                - current_close
            )
            / current_close
        ) * 100.0

    if predicted_return is None:
        predicted_return = forecast_return

    now_iso = datetime.now(
        timezone.utc
    ).isoformat()

    expected_range = (
        result.get(
            "expected_range"
        )
        or {}
    )

    coverage_mode = result.get(
        "coverage_mode"
    )

    trained_universe_member = result.get(
        "trained_universe_member"
    )

    if x2_point_price is not None:
        forecast_source = (
            "X2_EXPERIMENTAL_POINT"
        )
    elif (
        coverage_mode
        == "GENERALIZED_UNSEEN_STOCK"
    ):
        forecast_source = (
            "UNIVERSAL_GENERALIZED_POINT"
        )
    else:
        forecast_source = (
            "MODEL_PREDICTED_POINT"
        )

    record = {
        "symbol": symbol,
        "base_date": base_date,
        "recorded_at_utc": now_iso,
        "updated_at_utc": now_iso,

        # Snapshot visible when prediction was saved.
        "current_close": current_close,

        # Production/safe central kept for compatibility.
        "predicted_price": production_central_price,
        "predicted_return_percent": predicted_return,

        # Immutable point used by the all-time audit table.
        "forecast_point_price": forecast_point_price,
        "forecast_return_percent": forecast_return,
        "forecast_source": forecast_source,
        "predicted_direction": _direction_from_return(
            forecast_return
        ),

        "model": (
            result.get(
                "model"
            )
            or "StockVision"
        ),
        "prediction_mode": result.get(
            "prediction_mode"
        ),
        "central_source": result.get(
            "central_source"
        ),
        "coverage_mode": coverage_mode,
        "trained_universe_member": (
            trained_universe_member
        ),
        "symbol_validation_status": result.get(
            "symbol_validation_status"
        ),
        "x2_training_status": result.get(
            "x2_training_status"
        ),

        "experimental_x2_point_price": (
            x2_point_price
        ),

        "expected_range_lower": _safe_float(
            expected_range.get(
                "lower"
            )
        ),
        "expected_range_upper": _safe_float(
            expected_range.get(
                "upper"
            )
        ),
        "historical_range_coverage_percent": _safe_float(
            expected_range.get(
                "holdout_observed_coverage_percent"
            )
        ),

        "status": "PENDING",
        "target_date": None,
        "actual_close": None,
        "actual_return_percent": None,
        "actual_direction": None,
        "direction_correct": None,

        # Compatibility metrics.
        "absolute_error": None,
        "percentage_error": None,
        "experimental_x2_point_absolute_error": None,

        # Clear audit metrics.
        "forecast_difference_actual_minus_prediction": None,
        "forecast_absolute_error": None,
        "forecast_error_percent": None,

        "inside_expected_range": None,
    }

    with PREDICTION_HISTORY_LOCK:
        records = _load_prediction_history()

        match = next(
            (
                item
                for item in records
                if (
                    item.get(
                        "symbol"
                    )
                    == symbol
                    and item.get(
                        "base_date"
                    )
                    == base_date
                )
            ),
            None,
        )

        if match is None:
            records.append(
                record
            )

        else:
            # NEVER replace prediction values after first capture.
            # We may safely add missing schema fields to an older record.
            match[
                "last_seen_at_utc"
            ] = now_iso

            if (
                match.get(
                    "forecast_point_price"
                )
                is None
            ):
                legacy_point = _safe_float(
                    match.get(
                        "experimental_x2_point_price"
                    )
                )

                if legacy_point is None:
                    legacy_point = _safe_float(
                        match.get(
                            "predicted_price"
                        )
                    )

                match[
                    "forecast_point_price"
                ] = legacy_point

            if (
                match.get(
                    "forecast_return_percent"
                )
                is None
            ):
                legacy_base = _safe_float(
                    match.get(
                        "current_close"
                    )
                )

                legacy_forecast = _safe_float(
                    match.get(
                        "forecast_point_price"
                    )
                )

                if (
                    legacy_base is not None
                    and legacy_base != 0
                    and legacy_forecast is not None
                ):
                    match[
                        "forecast_return_percent"
                    ] = (
                        (
                            legacy_forecast
                            - legacy_base
                        )
                        / legacy_base
                    ) * 100.0

        _save_prediction_history(
            records
        )


NSE_TIMEZONE = ZoneInfo(
    "Asia/Kolkata"
)


def _daily_bar_is_completed(
    trading_date,
) -> bool:
    """
    Prevent today's still-open NSE daily candle from being used as an
    'actual close'. Previous trading dates are always considered complete.
    """
    now_ist = datetime.now(
        NSE_TIMEZONE
    )

    today_ist = now_ist.date()

    if trading_date < today_ist:
        return True

    if trading_date > today_ist:
        return False

    # Allow today's candle only after the normal close/closing-auction window.
    return (
        now_ist.hour > 15
        or (
            now_ist.hour == 15
            and now_ist.minute >= 40
        )
    )


def _filter_completed_daily_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = clean_history(
        frame
    )

    if frame.empty:
        return frame

    keep_mask = []

    for value in frame.index:
        trading_date = pd.Timestamp(
            value
        ).date()

        keep_mask.append(
            _daily_bar_is_completed(
                trading_date
            )
        )

    return frame.loc[
        keep_mask
    ]


def _raw_yahoo_chart_daily_history(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Crumb-free Yahoo chart endpoint fallback.

    This is useful when yfinance temporarily logs Invalid Crumb / Unauthorized
    while the public chart endpoint still returns daily OHLC data.
    """
    start_ts = int(
        pd.Timestamp(
            start_date,
            tz="UTC",
        ).timestamp()
    )

    end_ts = int(
        pd.Timestamp(
            end_date,
            tz="UTC",
        ).timestamp()
    )

    encoded_symbol = urllib.parse.quote(
        symbol,
        safe="",
    )

    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{encoded_symbol}"
        f"?period1={start_ts}"
        f"&period2={end_ts}"
        "&interval=1d"
        "&events=history"
        "&includeAdjustedClose=true"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 StockVision/2.11"
            )
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=15,
    ) as response:
        payload = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    chart = payload.get(
        "chart",
        {}
    )

    if chart.get(
        "error"
    ):
        raise RuntimeError(
            str(
                chart.get(
                    "error"
                )
            )
        )

    results = chart.get(
        "result"
    ) or []

    if not results:
        return pd.DataFrame()

    result = results[
        0
    ]

    timestamps = result.get(
        "timestamp"
    ) or []

    quotes = (
        result.get(
            "indicators",
            {}
        )
        .get(
            "quote",
            []
        )
    )

    if (
        not timestamps
        or not quotes
    ):
        return pd.DataFrame()

    quote = quotes[
        0
    ]

    size = len(
        timestamps
    )

    def values(
        key,
    ):
        raw = quote.get(
            key
        ) or []

        if len(
            raw
        ) < size:
            raw = list(
                raw
            ) + [
                None
            ] * (
                size
                - len(
                    raw
                )
            )

        return raw[
            :size
        ]

    index = pd.to_datetime(
        timestamps,
        unit="s",
        utc=True,
    ).tz_convert(
        NSE_TIMEZONE
    )

    frame = pd.DataFrame(
        {
            "Open": values(
                "open"
            ),
            "High": values(
                "high"
            ),
            "Low": values(
                "low"
            ),
            "Close": values(
                "close"
            ),
            "Volume": values(
                "volume"
            ),
        },
        index=index,
    )

    return frame


def _recent_completed_intraday_session_closes(
    symbol: str,
) -> pd.DataFrame:
    """
    Build recent completed NSE session closes from intraday candles.

    Why:
    Yahoo's 1d history can lag by one trading day. On Saturday 29 Aug 2026,
    RELIANCE 1d data was still ending at 27 Aug even though 28 Aug traded.
    Recent 5m candles can still contain the completed 28 Aug session.
    """
    errors = []

    fetchers = [
        (
            "Ticker.history:5d-5m",
            lambda: yf.Ticker(
                symbol
            ).history(
                period="5d",
                interval="5m",
                auto_adjust=False,
                prepost=False,
            ),
        ),
        (
            "yf.download:5d-5m",
            lambda: yf.download(
                symbol,
                period="5d",
                interval="5m",
                auto_adjust=False,
                progress=False,
                threads=False,
            ),
        ),
    ]

    intraday = None
    source_name = None

    for (
        candidate_source,
        fetcher,
    ) in fetchers:
        try:
            frame = clean_history(
                fetcher()
            )

            if not frame.empty:
                intraday = frame
                source_name = candidate_source
                break

            errors.append(
                f"{candidate_source}: empty"
            )

        except Exception as exc:
            errors.append(
                f"{candidate_source}: {exc}"
            )

    if intraday is None or intraday.empty:
        result = pd.DataFrame()

        result.attrs[
            "stockvision_source"
        ] = (
            "intraday unavailable: "
            + " | ".join(
                errors
            )
        )

        return result

    rows = []

    for index, row in intraday.iterrows():
        timestamp = pd.Timestamp(
            index
        )

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                NSE_TIMEZONE
            )
        else:
            timestamp = timestamp.tz_convert(
                NSE_TIMEZONE
            )

        trading_date = timestamp.date()

        if not _daily_bar_is_completed(
            trading_date
        ):
            continue

        close_value = _safe_float(
            row.get(
                "Close"
            )
        )

        if close_value is None:
            continue

        rows.append(
            (
                trading_date.isoformat(),
                timestamp,
                close_value,
            )
        )

    if not rows:
        result = pd.DataFrame()

        result.attrs[
            "stockvision_source"
        ] = source_name or "intraday"

        return result

    # Keep the final intraday candle as the session close proxy.
    latest_by_date = {}

    for (
        trading_date,
        timestamp,
        close_value,
    ) in rows:
        previous = latest_by_date.get(
            trading_date
        )

        if (
            previous is None
            or timestamp > previous[
                0
            ]
        ):
            latest_by_date[
                trading_date
            ] = (
                timestamp,
                close_value,
            )

    output_rows = []

    output_index = []

    for trading_date in sorted(
        latest_by_date
    ):
        (
            timestamp,
            close_value,
        ) = latest_by_date[
            trading_date
        ]

        output_index.append(
            timestamp
        )

        output_rows.append(
            {
                "Open": close_value,
                "High": close_value,
                "Low": close_value,
                "Close": close_value,
                "Volume": None,
            }
        )

    result = pd.DataFrame(
        output_rows,
        index=output_index,
    )

    result.attrs[
        "stockvision_source"
    ] = source_name or "intraday"

    return result


def _merge_daily_with_recent_intraday(
    daily_frame: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    daily_frame = _filter_completed_daily_rows(
        daily_frame
    )

    try:
        intraday_sessions = (
            _recent_completed_intraday_session_closes(
                symbol
            )
        )
    except Exception:
        intraday_sessions = pd.DataFrame()

    if intraday_sessions.empty:
        return daily_frame

    if daily_frame.empty:
        result = intraday_sessions.copy()

        result.attrs[
            "stockvision_source"
        ] = intraday_sessions.attrs.get(
            "stockvision_source",
            "intraday",
        )

        return result

    combined = pd.concat(
        [
            daily_frame,
            intraday_sessions,
        ]
    )

    # Collapse duplicate trading dates, preferring the later appended intraday
    # session close when Yahoo daily data is stale or duplicated.
    by_date = {}

    for index, row in combined.iterrows():
        trading_date = (
            pd.Timestamp(
                index
            )
            .date()
            .isoformat()
        )

        by_date[
            trading_date
        ] = {
            "index": index,
            "row": row,
        }

    ordered = [
        by_date[
            date
        ]
        for date in sorted(
            by_date
        )
    ]

    result = pd.DataFrame(
        [
            item[
                "row"
            ]
            for item in ordered
        ],
        index=[
            item[
                "index"
            ]
            for item in ordered
        ],
    )

    result.attrs[
        "stockvision_source"
    ] = (
        daily_frame.attrs.get(
            "stockvision_source",
            "daily",
        )
        + " + "
        + intraday_sessions.attrs.get(
            "stockvision_source",
            "intraday",
        )
    )

    return result


def _download_daily_validation_history(
    symbol: str,
    earliest_base_date: str,
) -> pd.DataFrame:
    """
    Fetch completed daily candles with multiple yfinance fallbacks.

    The old implementation used one Ticker.history request and silently left
    predictions PENDING whenever that request returned stale/empty data.
    """
    start_date = (
        pd.Timestamp(
            earliest_base_date
        )
        - pd.Timedelta(
            days=5
        )
    ).date().isoformat()

    end_date = (
        datetime.now(
            NSE_TIMEZONE
        ).date()
        + timedelta(
            days=2
        )
    ).isoformat()

    errors = []

    fetchers = [
        (
            "Ticker.history:1y",
            lambda: yf.Ticker(
                symbol
            ).history(
                period="1y",
                interval="1d",
                auto_adjust=False,
                prepost=False,
            ),
        ),
        (
            "Yahoo chart API:start-end",
            lambda: _raw_yahoo_chart_daily_history(
                symbol,
                start_date,
                end_date,
            ),
        ),
        (
            "yf.download:start-end",
            lambda: yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            ),
        ),
        (
            "Ticker.history:start-end",
            lambda: yf.Ticker(
                symbol
            ).history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                prepost=False,
            ),
        ),
        (
            "Ticker.history:1mo",
            lambda: yf.Ticker(
                symbol
            ).history(
                period="1mo",
                interval="1d",
                auto_adjust=False,
                prepost=False,
            ),
        ),
    ]

    for (
        source_name,
        fetcher,
    ) in fetchers:
        try:
            frame = fetcher()

            frame = _filter_completed_daily_rows(
                frame
            )

            if not frame.empty:
                frame.attrs[
                    "stockvision_source"
                ] = source_name

                frame = _merge_daily_with_recent_intraday(
                    frame,
                    symbol,
                )

                frame.attrs[
                    "stockvision_source"
                ] = (
                    frame.attrs.get(
                        "stockvision_source",
                        source_name,
                    )
                )

                return frame

            errors.append(
                f"{source_name}: empty"
            )

        except Exception as exc:
            errors.append(
                f"{source_name}: {exc}"
            )

    raise RuntimeError(
        "No completed Yahoo daily validation data. "
        + " | ".join(
            errors
        )
    )


def _resolve_symbol_predictions(
    symbol: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    symbol_records = [
        item
        for item in records
        if item.get("symbol") == symbol
    ]

    if not symbol_records:
        return records

    valid_base_dates = [
        item.get("base_date")
        for item in symbol_records
        if item.get("base_date")
    ]

    if not valid_base_dates:
        return records

    try:
        market_history = _download_daily_validation_history(
            symbol,
            min(valid_base_dates),
        )

    except Exception as exc:
        checked_at = datetime.now(
            timezone.utc
        ).isoformat()

        for item in records:
            if (
                item.get(
                    "symbol"
                )
                == symbol
                and item.get(
                    "status"
                )
                != "RESOLVED"
            ):
                item[
                    "last_resolution_check_utc"
                ] = checked_at

                item[
                    "resolution_data_status"
                ] = "FETCH_FAILED"

                item[
                    "resolution_note"
                ] = str(
                    exc
                )

        return records

    if market_history.empty:
        return records

    daily_rows: list[tuple[str, float]] = []

    for index, row in market_history.iterrows():
        close_value = _safe_float(
            row.get("Close")
        )

        if close_value is None:
            continue

        daily_rows.append(
            (
                pd.Timestamp(index).date().isoformat(),
                close_value,
            )
        )

    # Sort by trading date and keep the final close for duplicate dates.
    daily_map = {}

    for (
        trading_date,
        close_value,
    ) in daily_rows:
        daily_map[
            trading_date
        ] = close_value

    daily_rows = sorted(
        daily_map.items(),
        key=lambda item: item[
            0
        ],
    )

    latest_available_market_date = (
        daily_rows[
            -1
        ][
            0
        ]
        if daily_rows
        else None
    )

    validation_source = (
        market_history.attrs.get(
            "stockvision_source",
            "yfinance",
        )
    )

    resolution_checked_at = datetime.now(
        timezone.utc
    ).isoformat()

    for item in records:
        if item.get("symbol") != symbol:
            continue

        item[
            "last_resolution_check_utc"
        ] = resolution_checked_at

        item[
            "latest_available_market_date"
        ] = latest_available_market_date

        item[
            "resolution_data_source"
        ] = validation_source

        base_date = item.get("base_date")

        if not base_date:
            continue

        next_rows = [
            (
                trading_date,
                close_value,
            )
            for trading_date, close_value in daily_rows
            if trading_date > base_date
        ]

        if not next_rows:
            # The next completed trading-day close is not available in the
            # fetched daily history yet.
            item[
                "status"
            ] = "PENDING"

            item[
                "resolution_data_status"
            ] = "WAITING_FOR_NEXT_COMPLETED_DAILY_BAR"

            item[
                "resolution_note"
            ] = (
                "No completed Yahoo daily candle later than "
                f"{base_date} is available yet. "
                f"Latest available market date: "
                f"{latest_available_market_date or 'none'}."
            )

            continue

        target_date, actual_close = next_rows[0]

        item[
            "resolved_target_candidate_date"
        ] = target_date

        current_close = _safe_float(
            item.get("current_close")
        )

        # IMPORTANT:
        # Direction must be evaluated against the SAME immutable forecast
        # point that the history UI displays. Older code used
        # `predicted_price` (production/safe central), while the table showed
        # `forecast_point_price`, which could make Direction look WRONG even
        # when the displayed predicted price moved in the correct direction.
        forecast_point_price = _safe_float(
            item.get(
                "forecast_point_price"
            )
        )

        if forecast_point_price is None:
            forecast_point_price = _safe_float(
                item.get(
                    "experimental_x2_point_price"
                )
            )

        predicted_price = _safe_float(
            item.get(
                "predicted_price"
            )
        )

        if forecast_point_price is None:
            forecast_point_price = predicted_price

        forecast_return = _safe_float(
            item.get(
                "forecast_return_percent"
            )
        )

        if (
            forecast_return is None
            and current_close is not None
            and current_close != 0
            and forecast_point_price is not None
        ):
            forecast_return = (
                (
                    forecast_point_price
                    - current_close
                )
                / current_close
            ) * 100.0

        actual_return = None

        if (
            current_close is not None
            and current_close != 0
        ):
            actual_return = (
                (
                    actual_close
                    - current_close
                )
                / current_close
            ) * 100.0

        predicted_direction = _direction_from_return(
            forecast_return
        )

        actual_direction = _direction_from_return(
            actual_return
        )

        # Compatibility metrics now also use the displayed immutable forecast
        # point so all history numbers are internally consistent.
        absolute_error = None
        percentage_error = None

        if forecast_point_price is not None:
            absolute_error = abs(
                forecast_point_price
                - actual_close
            )

            if actual_close != 0:
                percentage_error = (
                    absolute_error
                    / abs(
                        actual_close
                    )
                ) * 100.0

        direction_correct = (
            predicted_direction == actual_direction
            if (
                predicted_direction is not None
                and actual_direction is not None
            )
            else None
        )

        range_lower = _safe_float(
            item.get(
                "expected_range_lower"
            )
        )

        range_upper = _safe_float(
            item.get(
                "expected_range_upper"
            )
        )

        inside_expected_range = None

        if (
            range_lower is not None
            and range_upper is not None
        ):
            low = min(
                range_lower,
                range_upper,
            )

            high = max(
                range_lower,
                range_upper,
            )

            inside_expected_range = (
                low
                <= actual_close
                <= high
            )

        experimental_x2_price = _safe_float(
            item.get(
                "experimental_x2_point_price"
            )
        )

        experimental_x2_error = None

        if experimental_x2_price is not None:
            experimental_x2_error = abs(
                experimental_x2_price
                - actual_close
            )


        forecast_difference = None
        forecast_absolute_error = None
        forecast_error_percent = None

        if forecast_point_price is not None:
            forecast_difference = (
                actual_close
                - forecast_point_price
            )

            forecast_absolute_error = abs(
                forecast_difference
            )

            if actual_close != 0:
                forecast_error_percent = (
                    forecast_absolute_error
                    / abs(
                        actual_close
                    )
                ) * 100.0

        item.update(
            {
                "status": "RESOLVED",
                "resolution_data_status": "RESOLVED_FROM_DAILY_CLOSE",
                "resolution_note": (
                    f"Resolved from completed {target_date} daily close."
                ),
                "target_date": target_date,
                "actual_close": actual_close,
                "actual_return_percent": actual_return,
                "forecast_return_percent": forecast_return,
                "predicted_return_percent": forecast_return,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "direction_correct": direction_correct,
                "absolute_error": absolute_error,
                "percentage_error": percentage_error,

                # All-time audit metrics use the actual forecast point that
                # was saved for the row.
                "forecast_point_price": forecast_point_price,
                "forecast_difference_actual_minus_prediction": (
                    forecast_difference
                ),
                "forecast_absolute_error": forecast_absolute_error,
                "forecast_error_percent": forecast_error_percent,

                "inside_expected_range": inside_expected_range,
                "experimental_x2_point_absolute_error": experimental_x2_error,
            }
        )

    return records


def _prediction_auto_capture_window_open() -> bool:
    """
    Automatic immutable snapshots are captured only when the daily market bar
    can be treated as complete.

    Weekend startup is allowed so a missing Friday snapshot can be captured.
    """
    now_ist = datetime.now(
        NSE_TIMEZONE
    )

    if now_ist.weekday() >= 5:
        return True

    return (
        now_ist.hour > 15
        or (
            now_ist.hour == 15
            and now_ist.minute >= 45
        )
    )


def _run_stock_prediction_engine(
    symbol: str,
) -> dict[str, Any]:
    """
    Shared prediction engine used by both the API and the daily history worker.
    """
    try:
        return predict_x2_hybrid(
            symbol
        )

    except (
        FileNotFoundError,
        ValueError,
    ):
        return predict_any_stock(
            symbol
        )


def _history_has_base_date(
    symbol: str,
    base_date: str,
) -> bool:
    with PREDICTION_HISTORY_LOCK:
        records = _load_prediction_history()

        return any(
            item.get(
                "symbol"
            )
            == symbol
            and item.get(
                "base_date"
            )
            == base_date
            for item in records
        )


def _resolve_history_for_symbol(
    symbol: str,
) -> None:
    with PREDICTION_HISTORY_LOCK:
        records = _load_prediction_history()

        records = _resolve_symbol_predictions(
            symbol,
            records,
        )

        _save_prediction_history(
            records
        )


def _capture_latest_tracked_prediction(
    symbol: str,
    track_symbol: bool = True,
) -> dict[str, Any]:
    """
    Resolve older rows and, when the market snapshot window is open, save one
    immutable forecast for the latest completed base trading date.

    No duplicate is created for the same symbol/base_date.
    """
    normalized = normalize_symbol(
        symbol
    )

    if track_symbol:
        _track_prediction_symbol(
            normalized
        )

    # Resolve old pending rows even when a new snapshot is not yet due.
    _resolve_history_for_symbol(
        normalized
    )

    try:
        (
            base_date,
            _,
        ) = _latest_daily_market_snapshot(
            normalized
        )
    except Exception as exc:
        return {
            "symbol": normalized,
            "status": "DATA_UNAVAILABLE",
            "captured": False,
            "detail": str(
                exc
            ),
        }

    if not base_date:
        return {
            "symbol": normalized,
            "status": "DATA_UNAVAILABLE",
            "captured": False,
            "detail": "No completed daily base date is available.",
        }

    if _history_has_base_date(
        normalized,
        base_date,
    ):
        return {
            "symbol": normalized,
            "status": "ALREADY_SAVED",
            "captured": False,
            "base_date": base_date,
        }

    if not _prediction_auto_capture_window_open():
        return {
            "symbol": normalized,
            "status": "WAITING_FOR_COMPLETED_SESSION",
            "captured": False,
            "base_date": base_date,
        }

    try:
        result = _run_stock_prediction_engine(
            normalized
        )

        model_latest_market_date = (
            result.get(
                "latest_market_date"
            )
            if isinstance(
                result,
                dict,
            )
            else None
        )

        # Do not fabricate a 28-Aug prediction using model features that only
        # reached 27 Aug. A historical audit must preserve what the model
        # actually knew at capture time.
        if (
            model_latest_market_date
            and str(
                model_latest_market_date
            ) < str(
                base_date
            )
        ):
            return {
                "symbol": normalized,
                "status": "MODEL_DATA_STALE",
                "captured": False,
                "base_date": base_date,
                "model_latest_market_date": (
                    model_latest_market_date
                ),
                "detail": (
                    "Latest completed market session is newer than the "
                    "model input date. StockVision will not fabricate a "
                    "retrospective live prediction."
                ),
            }

        record_live_prediction(
            normalized,
            result,
        )

        # Resolve again in case another old row became eligible meanwhile.
        _resolve_history_for_symbol(
            normalized
        )

        return {
            "symbol": normalized,
            "status": "CAPTURED",
            "captured": True,
            "base_date": base_date,
        }

    except Exception as exc:
        return {
            "symbol": normalized,
            "status": "PREDICTION_FAILED",
            "captured": False,
            "base_date": base_date,
            "detail": str(
                exc
            ),
        }


async def _full_universe_capture_worker(
    force_restart: bool = False,
) -> None:
    """
    Attempt one immutable daily prediction snapshot for EVERY current NSE
    security in the official StockVision universe.

    Processing is intentionally sequential + rate-limited to reduce Yahoo
    throttling on a local student project.
    """
    global _full_universe_task

    try:
        if not _prediction_auto_capture_window_open():
            return

        session_date = _full_universe_session_date()

        if not session_date:
            return

        stocks, source = load_nse_stock_universe(
            force=False
        )

        symbols = [
            item.get(
                "symbol"
            )
            for item in stocks
            if item.get(
                "symbol"
            )
        ]

        with PREDICTION_FULL_UNIVERSE_LOCK:
            state = _load_full_universe_state()

            if (
                not force_restart
                and state.get(
                    "completed"
                )
                and state.get(
                    "session_date"
                )
                == session_date
            ):
                return

            # Resume an incomplete batch for the same session.
            if (
                not force_restart
                and state.get(
                    "session_date"
                )
                == session_date
                and not state.get(
                    "completed"
                )
            ):
                processed_symbols = set(
                    state.get(
                        "processed_symbols"
                    )
                    or []
                )

            else:
                processed_symbols = set()

                state = _default_full_universe_state()

                state.update(
                    {
                        "session_date": session_date,
                        "started_at_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "total": len(
                            symbols
                        ),
                        "universe_source": source,
                        "processed_symbols": [],
                    }
                )

            state[
                "running"
            ] = True

            state[
                "completed"
            ] = False

            _save_full_universe_state(
                state
            )

        for symbol in symbols:
            if symbol in processed_symbols:
                continue

            try:
                result = await asyncio.to_thread(
                    _capture_latest_tracked_prediction,
                    symbol,
                    False,
                )

                status = str(
                    result.get(
                        "status"
                    )
                    or "UNKNOWN"
                )

                detail = result.get(
                    "detail"
                )

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                status = "PREDICTION_FAILED"
                detail = str(
                    exc
                )

            with PREDICTION_FULL_UNIVERSE_LOCK:
                state = _load_full_universe_state()

                processed_list = list(
                    state.get(
                        "processed_symbols"
                    )
                    or []
                )

                if symbol not in processed_list:
                    processed_list.append(
                        symbol
                    )

                state[
                    "processed_symbols"
                ] = processed_list

                state[
                    "processed"
                ] = len(
                    processed_list
                )

                state[
                    "last_symbol"
                ] = symbol

                state[
                    "last_status"
                ] = status

                state[
                    "last_detail"
                ] = detail

                if status == "CAPTURED":
                    state[
                        "captured"
                    ] = int(
                        state.get(
                            "captured"
                        )
                        or 0
                    ) + 1

                elif status == "ALREADY_SAVED":
                    state[
                        "already_saved"
                    ] = int(
                        state.get(
                            "already_saved"
                        )
                        or 0
                    ) + 1

                elif status == "DATA_UNAVAILABLE":
                    state[
                        "data_unavailable"
                    ] = int(
                        state.get(
                            "data_unavailable"
                        )
                        or 0
                    ) + 1

                elif status == "MODEL_DATA_STALE":
                    state[
                        "model_data_stale"
                    ] = int(
                        state.get(
                            "model_data_stale"
                        )
                        or 0
                    ) + 1

                elif status == "WAITING_FOR_COMPLETED_SESSION":
                    state[
                        "waiting"
                    ] = int(
                        state.get(
                            "waiting"
                        )
                        or 0
                    ) + 1

                else:
                    state[
                        "failed"
                    ] = int(
                        state.get(
                            "failed"
                        )
                        or 0
                    ) + 1

                    errors = list(
                        state.get(
                            "symbols_with_errors"
                        )
                        or []
                    )

                    if len(
                        errors
                    ) < 200:
                        errors.append(
                            {
                                "symbol": symbol,
                                "status": status,
                                "detail": detail,
                            }
                        )

                    state[
                        "symbols_with_errors"
                    ] = errors

                _save_full_universe_state(
                    state
                )

            await asyncio.sleep(
                FULL_UNIVERSE_DELAY_SECONDS
            )

        with PREDICTION_FULL_UNIVERSE_LOCK:
            state = _load_full_universe_state()

            state[
                "running"
            ] = False

            state[
                "completed"
            ] = True

            state[
                "finished_at_utc"
            ] = datetime.now(
                timezone.utc
            ).isoformat()

            _save_full_universe_state(
                state
            )

    except asyncio.CancelledError:
        with PREDICTION_FULL_UNIVERSE_LOCK:
            state = _load_full_universe_state()

            state[
                "running"
            ] = False

            _save_full_universe_state(
                state
            )

        raise

    except Exception as exc:
        with PREDICTION_FULL_UNIVERSE_LOCK:
            state = _load_full_universe_state()

            state[
                "running"
            ] = False

            state[
                "last_status"
            ] = "WORKER_ERROR"

            state[
                "last_detail"
            ] = str(
                exc
            )

            _save_full_universe_state(
                state
            )


async def _ensure_full_universe_task(
    force_restart: bool = False,
) -> bool:
    global _full_universe_task

    if (
        _full_universe_task is not None
        and not _full_universe_task.done()
    ):
        return False

    _full_universe_task = asyncio.create_task(
        _full_universe_capture_worker(
            force_restart=force_restart
        )
    )

    return True


async def _prediction_history_worker() -> None:
    """
    Background worker for stocks the user has searched/predicted.

    It runs only while the local FastAPI backend is running. This is not a
    cloud 24x7 service.
    """
    global _prediction_history_worker_last_run_utc
    global _prediction_history_worker_last_error

    # Give application startup a moment before hitting market data.
    await asyncio.sleep(
        3
    )

    while True:
        try:
            symbols = _load_tracked_prediction_symbols()

            for symbol in symbols:
                try:
                    await asyncio.to_thread(
                        _capture_latest_tracked_prediction,
                        symbol,
                    )
                except Exception:
                    # One bad/delisted stock must not stop every other stock.
                    continue

            _prediction_history_worker_last_run_utc = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            _prediction_history_worker_last_error = None

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            _prediction_history_worker_last_error = str(
                exc
            )

        await asyncio.sleep(
            AUTO_HISTORY_CHECK_SECONDS
        )


@app.on_event(
    "startup"
)
async def _start_prediction_history_worker() -> None:
    global _prediction_history_worker_task

    if (
        _prediction_history_worker_task
        is None
        or _prediction_history_worker_task.done()
    ):
        _prediction_history_worker_task = asyncio.create_task(
            _prediction_history_worker()
        )

    if (
        FULL_UNIVERSE_AUTO_CAPTURE
        and _prediction_auto_capture_window_open()
    ):
        await _ensure_full_universe_task(
            force_restart=False
        )


@app.on_event(
    "shutdown"
)
async def _stop_prediction_history_worker() -> None:
    global _prediction_history_worker_task

    task = _prediction_history_worker_task

    if task is None:
        return

    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    _prediction_history_worker_task = None


def _prediction_replay_file(
    symbol: str,
) -> Path:
    safe = (
        normalize_symbol(
            symbol
        )
        .replace(
            "^",
            "_INDEX_",
        )
        .replace(
            ".",
            "_",
        )
        .replace(
            "/",
            "_",
        )
    )

    return (
        PREDICTION_REPLAY_DIR
        / f"{safe}.json"
    )


def _load_prediction_replay(
    symbol: str,
) -> list[dict[str, Any]]:
    path = _prediction_replay_file(
        symbol
    )

    if not path.exists():
        return []

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(
                file
            )

        if isinstance(
            payload,
            dict,
        ):
            rows = payload.get(
                "rows"
            )
        else:
            rows = payload

        return (
            rows
            if isinstance(
                rows,
                list,
            )
            else []
        )

    except Exception:
        return []


def _save_prediction_replay(
    symbol: str,
    rows: list[dict[str, Any]],
) -> None:
    PREDICTION_REPLAY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = _prediction_replay_file(
        symbol
    )

    _atomic_json_replace(
        path,
        {
            "symbol": normalize_symbol(
                symbol
            ),
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "mode": "HISTORICAL_MODEL_REPLAY",
            "rows": rows,
        },
    )


def _ensure_prediction_replay(
    symbol: str,
    force: bool = False,
) -> list[dict[str, Any]]:
    """
    Generate a cached two-year historical replay for the selected stock.

    This fills missing historical dates such as 27 Aug with an explicit
    BACKFILLED_MODEL_REPLAY row while preserving genuine LIVE rows separately.
    """
    normalized = normalize_symbol(
        symbol
    )

    with PREDICTION_REPLAY_LOCK:
        existing = _load_prediction_replay(
            normalized
        )

        if (
            existing
            and not force
        ):
            # Refresh when cached replay is clearly behind the latest resolved
            # completed session available in current validation history.
            latest_cached_target = max(
                (
                    item.get(
                        "target_date"
                    )
                    or ""
                )
                for item in existing
            )

            try:
                (
                    latest_market_date,
                    _,
                ) = _latest_daily_market_snapshot(
                    normalized
                )
            except Exception:
                latest_market_date = None

            if (
                latest_market_date
                and latest_cached_target
                >= latest_market_date
            ):
                return existing

        try:
            rows = historical_replay_v3(
                normalized,
                period="2y",
                max_rows=None,
            )
        except Exception:
            return existing

        if rows:
            _save_prediction_replay(
                normalized,
                rows,
            )

            return rows

        return existing


def _merge_live_and_replay_history(
    live_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Live rows always win when both sources contain the same base date.
    Replay rows only fill dates that were never captured live.
    """
    merged = {}

    for item in replay_rows:
        base_date = item.get(
            "base_date"
        )

        if not base_date:
            continue

        merged[
            base_date
        ] = {
            **item,
            "record_type": (
                item.get(
                    "record_type"
                )
                or "HISTORICAL_REPLAY"
            ),
            "history_source": (
                item.get(
                    "history_source"
                )
                or "BACKFILLED_MODEL_REPLAY"
            ),
            "is_live_capture": False,
        }

    for item in live_rows:
        base_date = item.get(
            "base_date"
        )

        if not base_date:
            continue

        merged[
            base_date
        ] = {
            **item,
            "record_type": (
                item.get(
                    "record_type"
                )
                or "LIVE_CAPTURE"
            ),
            "history_source": (
                item.get(
                    "history_source"
                )
                or "LIVE_CAPTURE"
            ),
            "is_live_capture": True,
        }

    rows = list(
        merged.values()
    )

    rows.sort(
        key=lambda item: (
            item.get(
                "base_date"
            )
            or "",
            item.get(
                "target_date"
            )
            or "",
        ),
        reverse=True,
    )

    return rows


def build_prediction_validation(
    symbol: str,
) -> dict[str, Any]:
    with PREDICTION_HISTORY_LOCK:
        records = _load_prediction_history()

        records = _resolve_symbol_predictions(
            symbol,
            records,
        )

        _save_prediction_history(
            records
        )

    live_symbol_records = [
        item
        for item in records
        if item.get("symbol") == symbol
    ]

    live_symbol_records.sort(
        key=lambda item: (
            item.get("base_date") or "",
            item.get("updated_at_utc") or "",
        ),
        reverse=True,
    )

    replay_records = _ensure_prediction_replay(
        symbol,
        force=False,
    )

    symbol_records = _merge_live_and_replay_history(
        live_symbol_records,
        replay_records,
    )

    resolved = [
        item
        for item in symbol_records
        if item.get("status") == "RESOLVED"
    ]

    pending = [
        item
        for item in symbol_records
        if item.get("status") != "RESOLVED"
    ]

    direction_rows = [
        item
        for item in resolved
        if item.get("direction_correct") is not None
    ]

    correct_count = sum(
        1
        for item in direction_rows
        if item.get("direction_correct") is True
    )

    direction_accuracy = (
        (correct_count / len(direction_rows)) * 100.0
        if direction_rows
        else None
    )

    absolute_errors = [
        float(item["absolute_error"])
        for item in resolved
        if _safe_float(item.get("absolute_error")) is not None
    ]

    percentage_errors = [
        float(item["percentage_error"])
        for item in resolved
        if _safe_float(item.get("percentage_error")) is not None
    ]

    mae = (
        float(np.mean(absolute_errors))
        if absolute_errors
        else None
    )

    mape = (
        float(np.mean(percentage_errors))
        if percentage_errors
        else None
    )

    range_rows = [
        item
        for item in resolved
        if item.get(
            "inside_expected_range"
        ) is not None
    ]

    inside_range_count = sum(
        1
        for item in range_rows
        if item.get(
            "inside_expected_range"
        ) is True
    )

    live_range_coverage = (
        (
            inside_range_count
            / len(
                range_rows
            )
        )
        * 100.0
        if range_rows
        else None
    )

    x2_point_errors = [
        float(
            item[
                "experimental_x2_point_absolute_error"
            ]
        )
        for item in resolved
        if _safe_float(
            item.get(
                "experimental_x2_point_absolute_error"
            )
        ) is not None
    ]

    x2_point_mae = (
        float(
            np.mean(
                x2_point_errors
            )
        )
        if x2_point_errors
        else None
    )

    x2_within_20 = (
        (
            sum(
                1
                for error
                in x2_point_errors
                if error <= 20.0
            )
            / len(
                x2_point_errors
            )
        )
        * 100.0
        if x2_point_errors
        else None
    )

    x2_over_30 = (
        (
            sum(
                1
                for error
                in x2_point_errors
                if error > 30.0
            )
            / len(
                x2_point_errors
            )
        )
        * 100.0
        if x2_point_errors
        else None
    )

    return {
        "symbol": symbol,
        "rule": {
            "target": "next NSE trading-day close",
            "direction_neutral_band_percent": (
                PREDICTION_DIRECTION_NEUTRAL_BAND_PCT
            ),
            "correct_definition": (
                "predicted direction equals actual next-day direction"
            ),
        },
        "metrics": {
            "total_predictions": len(symbol_records),
            "resolved_predictions": len(resolved),
            "pending_predictions": len(pending),
            "correct_direction_predictions": correct_count,
            "direction_accuracy_percent": direction_accuracy,
            "mae_price": mae,
            "mape_percent": mape,

            # X2 live validation
            "range_resolved_predictions": len(
                range_rows
            ),
            "inside_expected_range_count": inside_range_count,
            "live_range_coverage_percent": live_range_coverage,
            "experimental_x2_point_mae": x2_point_mae,
            "experimental_x2_within_20_percent": x2_within_20,
            "experimental_x2_over_30_percent": x2_over_30,
        },
        "history_scope": "LIVE_PLUS_HISTORICAL_MODEL_REPLAY",
        "tracking": {
            "tracked": (
                symbol
                in _load_tracked_prediction_symbols()
            ),
            "auto_capture": True,
            "auto_capture_when": (
                "after 15:45 IST on trading weekdays, plus weekend startup "
                "for the latest completed session"
            ),
            "worker_runs_only_while_backend_is_running": True,
            "worker_check_interval_minutes": int(
                AUTO_HISTORY_CHECK_SECONDS
                / 60
            ),
            "worker_last_run_utc": (
                _prediction_history_worker_last_run_utc
            ),
            "worker_last_error": (
                _prediction_history_worker_last_error
            ),
        },
        "history_breakdown": {
            "live_rows": len(
                live_symbol_records
            ),
            "replay_rows_used": sum(
                1
                for item in symbol_records
                if item.get(
                    "history_source"
                )
                == "BACKFILLED_MODEL_REPLAY"
            ),
            "total_rows": len(
                symbol_records
            ),
        },
        "history_note": (
            "LIVE_CAPTURE rows are genuine forecasts saved at the time. "
            "BACKFILLED_MODEL_REPLAY rows fill older missing dates by replaying "
            "the frozen model with information available on each historical "
            "base date. Replay rows are clearly labelled and are not presented "
            "as original live forecasts."
        ),
        "history_start_date": (
            symbol_records[-1].get(
                "base_date"
            )
            if symbol_records
            else None
        ),
        "history_end_date": (
            symbol_records[0].get(
                "base_date"
            )
            if symbol_records
            else None
        ),
        "history": symbol_records,
    }


# =========================================================
# NEXT-DAY BILSTM PREDICTION
# =========================================================

@app.get("/predict/{symbol}")
def predict_stock(symbol: str):
    normalized = normalize_symbol(symbol)

    try:
        # Every stock the user predicts becomes a permanently tracked history
        # symbol for future daily snapshots.
        _track_prediction_symbol(
            normalized
        )

        result = _run_stock_prediction_engine(
            normalized
        )

        # Prediction logging must never make the live prediction endpoint fail.
        try:
            if isinstance(
                result,
                dict,
            ):
                record_live_prediction(
                    normalized,
                    result,
                )

        except Exception:
            pass

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed for {normalized}: {exc}",
        )



def _prediction_history_overview(
    query: str = "",
    page: int = 1,
    page_size: int = 25,
    status: str = "ALL",
) -> dict[str, Any]:
    """
    Join the current NSE universe with saved StockVision prediction history.

    Important:
    - Every NSE stock is visible in the overview, even if the user never
      searched it before.
    - We do NOT fabricate old predictions. Stocks with no captured live
      forecast are shown as NO_HISTORY.
    """
    stocks, universe_source = load_nse_stock_universe(
        force=False
    )

    with PREDICTION_HISTORY_LOCK:
        records = _load_prediction_history()

    by_symbol: dict[str, list[dict[str, Any]]] = {}

    for item in records:
        symbol = item.get(
            "symbol"
        )

        if not symbol:
            continue

        by_symbol.setdefault(
            symbol,
            []
        ).append(
            item
        )

    for symbol_records in by_symbol.values():
        symbol_records.sort(
            key=lambda item: (
                item.get(
                    "base_date"
                )
                or "",
                item.get(
                    "recorded_at_utc"
                )
                or "",
            ),
            reverse=True,
        )

    rows = []

    total_saved_predictions = 0
    total_resolved = 0
    total_pending = 0
    with_history = 0

    for stock in stocks:
        symbol = stock.get(
            "symbol"
        )

        symbol_records = by_symbol.get(
            symbol,
            [],
        )

        saved_count = len(
            symbol_records
        )

        resolved_count = sum(
            1
            for item in symbol_records
            if item.get(
                "status"
            )
            == "RESOLVED"
        )

        pending_count = (
            saved_count
            - resolved_count
        )

        total_saved_predictions += (
            saved_count
        )

        total_resolved += (
            resolved_count
        )

        total_pending += (
            pending_count
        )

        if saved_count:
            with_history += 1

        latest = (
            symbol_records[
                0
            ]
            if symbol_records
            else {}
        )

        forecast = _safe_float(
            latest.get(
                "forecast_point_price"
            )
        )

        if forecast is None:
            forecast = _safe_float(
                latest.get(
                    "experimental_x2_point_price"
                )
            )

        if forecast is None:
            forecast = _safe_float(
                latest.get(
                    "predicted_price"
                )
            )

        actual = _safe_float(
            latest.get(
                "actual_close"
            )
        )

        difference = _safe_float(
            latest.get(
                "forecast_difference_actual_minus_prediction"
            )
        )

        if (
            difference is None
            and actual is not None
            and forecast is not None
        ):
            difference = (
                actual
                - forecast
            )

        abs_error = _safe_float(
            latest.get(
                "forecast_absolute_error"
            )
        )

        if (
            abs_error is None
            and difference is not None
        ):
            abs_error = abs(
                difference
            )

        error_percent = _safe_float(
            latest.get(
                "forecast_error_percent"
            )
        )

        final_value = (
            actual
            if actual is not None
            else forecast
        )

        rows.append(
            {
                "symbol": symbol,
                "short": stock.get(
                    "short"
                ),
                "name": stock.get(
                    "name"
                ),
                "series": stock.get(
                    "series"
                ),
                "has_history": bool(
                    saved_count
                ),
                "saved_records": saved_count,
                "resolved_records": resolved_count,
                "pending_records": pending_count,
                "latest_prediction_date": latest.get(
                    "base_date"
                ),
                "latest_target_date": latest.get(
                    "target_date"
                ),
                "latest_base_close": _safe_float(
                    latest.get(
                        "current_close"
                    )
                ),
                "latest_forecast": forecast,
                "latest_actual_close": actual,
                "latest_difference": difference,
                "latest_absolute_error": abs_error,
                "latest_error_percent": error_percent,
                "latest_final_value": final_value,
                "latest_status": (
                    latest.get(
                        "status"
                    )
                    if latest
                    else "NO_HISTORY"
                ),
                "direction_correct": latest.get(
                    "direction_correct"
                ),
                "inside_expected_range": latest.get(
                    "inside_expected_range"
                ),
            }
        )

    clean_query = str(
        query
        or ""
    ).strip().lower()

    if clean_query:
        rows = [
            row
            for row in rows
            if (
                clean_query
                in str(
                    row.get(
                        "short"
                    )
                    or ""
                ).lower()
                or clean_query
                in str(
                    row.get(
                        "name"
                    )
                    or ""
                ).lower()
                or clean_query
                in str(
                    row.get(
                        "symbol"
                    )
                    or ""
                ).lower()
            )
        ]

    clean_status = str(
        status
        or "ALL"
    ).strip().upper()

    if clean_status == "WITH_HISTORY":
        rows = [
            row
            for row in rows
            if row.get(
                "has_history"
            )
        ]

    elif clean_status == "RESOLVED":
        rows = [
            row
            for row in rows
            if row.get(
                "latest_status"
            )
            == "RESOLVED"
        ]

    elif clean_status == "PENDING":
        rows = [
            row
            for row in rows
            if row.get(
                "latest_status"
            )
            == "PENDING"
        ]

    elif clean_status == "NO_HISTORY":
        rows = [
            row
            for row in rows
            if not row.get(
                "has_history"
            )
        ]

    rows.sort(
        key=lambda row: (
            0
            if row.get(
                "has_history"
            )
            else 1,
            str(
                row.get(
                    "short"
                )
                or ""
            ).lower(),
        )
    )

    page = max(
        1,
        int(
            page
            or 1
        ),
    )

    page_size = min(
        100,
        max(
            10,
            int(
                page_size
                or 25
            ),
        ),
    )

    total_filtered = len(
        rows
    )

    total_pages = max(
        1,
        math.ceil(
            total_filtered
            / page_size
        ),
    )

    page = min(
        page,
        total_pages,
    )

    start = (
        page
        - 1
    ) * page_size

    end = start + page_size

    visible_rows = rows[
        start:end
    ]

    return {
        "universe_source": universe_source,
        "summary": {
            "total_nse_stocks": len(
                stocks
            ),
            "stocks_with_saved_history": with_history,
            "stocks_without_saved_history": (
                len(
                    stocks
                )
                - with_history
            ),
            "total_saved_predictions": total_saved_predictions,
            "resolved_predictions": total_resolved,
            "pending_predictions": total_pending,
        },
        "filters": {
            "query": query,
            "status": clean_status,
        },
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_filtered": total_filtered,
            "total_pages": total_pages,
        },
        "rows": visible_rows,
        "full_universe_capture": _full_universe_status_payload(),
        "note": (
            "All current NSE stocks are visible. NO_HISTORY means StockVision "
            "has not yet captured a real live forecast for that stock. "
            "Historical rows are never fabricated."
        ),
    }


@app.get("/prediction-full-universe/status")
def prediction_full_universe_status():
    return _full_universe_status_payload()


@app.post("/prediction-full-universe/start")
async def prediction_full_universe_start():
    if not _prediction_auto_capture_window_open():
        return {
            **_full_universe_status_payload(),
            "started": False,
            "message": (
                "Full-universe daily capture starts after the NSE session "
                "is completed."
            ),
        }

    started = await _ensure_full_universe_task(
        force_restart=False
    )

    await asyncio.sleep(
        0
    )

    return {
        **_full_universe_status_payload(),
        "started": started,
    }


@app.post("/prediction-full-universe/restart")
async def prediction_full_universe_restart():
    global _full_universe_task

    if (
        _full_universe_task is not None
        and not _full_universe_task.done()
    ):
        _full_universe_task.cancel()

        try:
            await _full_universe_task
        except asyncio.CancelledError:
            pass

    started = await _ensure_full_universe_task(
        force_restart=True
    )

    return {
        **_full_universe_status_payload(),
        "started": started,
    }


@app.get("/prediction-history-overview")
def prediction_history_overview(
    query: str = "",
    page: int = 1,
    page_size: int = 25,
    status: str = "ALL",
):
    try:
        return _prediction_history_overview(
            query=query,
            page=page,
            page_size=page_size,
            status=status,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load all-stock prediction history overview: "
                f"{exc}"
            ),
        )


@app.get("/prediction-tracked")
def prediction_tracked():
    symbols = _load_tracked_prediction_symbols()

    return {
        "count": len(
            symbols
        ),
        "symbols": symbols,
        "auto_capture": True,
        "worker_runs_only_while_backend_is_running": True,
        "worker_check_interval_minutes": int(
            AUTO_HISTORY_CHECK_SECONDS
            / 60
        ),
        "worker_last_run_utc": (
            _prediction_history_worker_last_run_utc
        ),
        "worker_last_error": (
            _prediction_history_worker_last_error
        ),
    }


@app.post("/prediction-track/{symbol}")
def prediction_track(symbol: str):
    normalized = normalize_symbol(
        symbol
    )

    tracked = _track_prediction_symbol(
        normalized
    )

    capture = _capture_latest_tracked_prediction(
        normalized
    )

    return {
        "symbol": normalized,
        "tracked": True,
        "tracked_count": len(
            tracked
        ),
        "capture": capture,
    }


@app.delete("/prediction-track/{symbol}")
def prediction_untrack(symbol: str):
    normalized = normalize_symbol(
        symbol
    )

    symbols = _untrack_prediction_symbol(
        normalized
    )

    return {
        "symbol": normalized,
        "tracked": False,
        "tracked_count": len(
            symbols
        ),
    }


@app.post("/prediction-history-refresh/{symbol}")
def prediction_history_refresh(symbol: str):
    """
    Resolve old rows and capture the latest completed-day forecast if due.
    """
    normalized = normalize_symbol(
        symbol
    )

    _track_prediction_symbol(
        normalized
    )

    capture = _capture_latest_tracked_prediction(
        normalized
    )

    history = build_prediction_validation(
        normalized
    )

    return {
        **history,
        "refresh_result": capture,
    }


@app.get("/prediction-history-debug/{symbol}")
def prediction_history_debug(symbol: str):
    normalized = normalize_symbol(
        symbol
    )

    with PREDICTION_HISTORY_LOCK:
        records = _load_prediction_history()

        symbol_records = [
            item
            for item in records
            if item.get(
                "symbol"
            ) == normalized
        ]

    base_dates = [
        item.get(
            "base_date"
        )
        for item in symbol_records
        if item.get(
            "base_date"
        )
    ]

    daily_rows = []

    data_source = None
    fetch_error = None

    if base_dates:
        try:
            frame = _download_daily_validation_history(
                normalized,
                min(
                    base_dates
                ),
            )

            data_source = frame.attrs.get(
                "stockvision_source",
                "unknown",
            )

            for index, row in frame.iterrows():
                close_value = _safe_float(
                    row.get(
                        "Close"
                    )
                )

                if close_value is None:
                    continue

                daily_rows.append(
                    {
                        "date": (
                            pd.Timestamp(
                                index
                            )
                            .date()
                            .isoformat()
                        ),
                        "close": close_value,
                    }
                )

        except Exception as exc:
            fetch_error = str(
                exc
            )

    return {
        "symbol": normalized,
        "backend_version": "2.11.1",
        "saved_records": len(
            symbol_records
        ),
        "saved_base_dates": base_dates,
        "daily_data_source": data_source,
        "daily_rows": daily_rows,
        "latest_daily_date": (
            daily_rows[
                -1
            ][
                "date"
            ]
            if daily_rows
            else None
        ),
        "fetch_error": fetch_error,
        "intraday_fallback_note": (
            "Recent completed 5-minute sessions are merged into daily "
            "history when Yahoo 1d data is one trading day behind."
        ),
        "note": (
            "A prediction with base_date 2026-08-27 should resolve "
            "when daily_rows contains any completed trading date later "
            "than 2026-08-27, normally 2026-08-28."
        ),
    }


@app.post("/prediction-history-replay-refresh/{symbol}")
def prediction_history_replay_refresh(
    symbol: str,
):
    normalized = normalize_symbol(
        symbol
    )

    try:
        rows = _ensure_prediction_replay(
            normalized,
            force=True,
        )

        result = build_prediction_validation(
            normalized
        )

        return {
            **result,
            "replay_refresh": {
                "symbol": normalized,
                "generated_rows": len(
                    rows
                ),
                "status": "OK",
            },
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Historical replay failed for {normalized}: {exc}"
            ),
        )


@app.get("/prediction-history/{symbol}")
def prediction_history(symbol: str):
    normalized = normalize_symbol(
        symbol
    )

    try:
        _track_prediction_symbol(
            normalized
        )

        return build_prediction_validation(
            normalized
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to load prediction history for "
                f"{normalized}: {exc}"
            ),
        )


@app.get("/prediction-validation/{symbol}")
def prediction_validation(symbol: str):
    normalized = normalize_symbol(symbol)

    try:
        return build_prediction_validation(
            normalized
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to validate predictions for "
                f"{normalized}: {exc}"
            ),
        )


# =========================================================
# MULTI-HORIZON FORECAST
# =========================================================

@app.get("/future-predict/{symbol}")
def future_predict(symbol: str):
    normalized = normalize_symbol(symbol)

    try:
        result = predict_any_stock_multi(normalized)

        return result

    except Exception as exc:
        message = str(exc)

        # Keep model-not-ready errors readable by the frontend.
        status_code = 404 if "model" in message.lower() else 500

        raise HTTPException(
            status_code=status_code,
            detail=message,
        )


# =========================================================
# START MULTI-HORIZON TRAINING
# =========================================================

@app.post("/future-train/{symbol}")
def future_train(symbol: str):
    normalized = normalize_symbol(symbol)

    try:
        return start_training(normalized)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to start training for {normalized}: {exc}",
        )


# =========================================================
# MULTI-HORIZON TRAINING STATUS
# =========================================================

@app.get("/future-status/{symbol}")
def future_status(symbol: str):
    normalized = normalize_symbol(symbol)

    try:
        return get_training_status(normalized)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read training status for {normalized}: {exc}",
        )


# =========================================================
# V9 RELATIVE-STRENGTH PREDICTION
# =========================================================

@app.get("/relative-predict/{symbol}")
def relative_predict(symbol: str):
    normalized = normalize_symbol(symbol)

    try:
        return predict_relative_strength(normalized)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"V9 relative-strength prediction failed for {normalized}: {exc}",
        )


# =========================================================
# V9 STATUS
# =========================================================

@app.get("/relative-status")
def relative_status():
    try:
        return get_v9_status()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read V9 status: {exc}",
        )


# =========================================================
# MODEL ANALYTICS
# =========================================================

@app.get("/model-analytics")
def model_analytics():
    try:
        return get_model_analytics()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load model analytics: {exc}",
        )


# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
