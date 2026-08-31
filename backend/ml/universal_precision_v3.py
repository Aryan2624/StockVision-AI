"""
StockVision Precision V3 — Universal Multi-Stock Next-Day Forecast
=================================================================

Purpose
-------
Train ONE leakage-safe model across a liquid NSE stock universe instead of
training a separate model from only ~2,000 rows for each stock.

Target
------
For each stock/date:

    normalized_next_move
        = (next_close - current_close) / ATR14

At prediction time:

    predicted_rupee_move
        = predicted_normalized_move * current_ATR14

    predicted_close
        = current_close + predicted_rupee_move

Why this helps
--------------
A Rs 10 move has very different meaning for a Rs 300 stock and a Rs 4,000
stock. ATR normalization lets the model learn comparable next-day movement
patterns across many stocks.

Safety / evaluation
-------------------
- all train/validation/test splits are chronological by DATE
- same future date never leaks into training through another stock
- validation chooses ensemble weights, shrinkage, bias and prediction cap
- previous-close baseline = normalized move 0
- 4-fold expanding walk-forward evaluation
- overall metrics + per-stock metrics
- model is ACCEPTED only if:
    * validation beats baseline
    * unseen holdout beats baseline overall
    * walk-forward overall beats baseline
    * >= 3/4 walk-forward folds beat baseline
    * majority of holdout stocks beat their own baseline

No model can guarantee Rs 1-2 next-day accuracy.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

MODEL_VERSION = "precision-v3-universal"
RANDOM_SEED = 42

NSE_TIMEZONE = ZoneInfo("Asia/Kolkata")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

MIN_VALIDATION_IMPROVEMENT_PCT = 0.50
MIN_WALK_FORWARD_FOLDS_BEATING = 3
WALK_FORWARD_FOLDS = 4
WALK_FORWARD_INITIAL_DATE_RATIO = 0.55

DIRECTION_NEUTRAL_PCT = 0.10

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "TATAMOTORS.NS",
    "SBIN.NS",
    "ITC.NS",
    "LT.NS",
    "BHARTIARTL.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
]

BANK_STOCKS = {
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "AXISBANK",
    "KOTAKBANK",
}

IT_STOCKS = {
    "TCS",
    "INFY",
    "HCLTECH",
    "WIPRO",
}

BASE_FEATURES = [
    # Price action
    "RET1",
    "RET2",
    "RET3",
    "RET5",
    "RET10",
    "GAP_PCT",
    "INTRADAY_PCT",
    "RANGE_PCT",
    "CLOSE_LOCATION",

    # Volatility-normalized movement
    "MOVE_ATR_1D",
    "MOVE_ATR_3D",
    "MOVE_ATR_5D",
    "ATR14_PCT",
    "VOL5",
    "VOL10",
    "VOL20",

    # Trend
    "SMA5_DIST",
    "SMA10_DIST",
    "SMA20_DIST",
    "SMA50_DIST",
    "EMA5_DIST",
    "EMA10_DIST",
    "EMA20_DIST",
    "EMA50_DIST",
    "EMA5_20_SPREAD",

    # Momentum
    "RSI14",
    "MACD_PCT",
    "MACD_SIGNAL_PCT",
    "MACD_HIST_PCT",
    "ROC5",
    "ROC10",

    # Volume
    "VOLUME_LOG_CHANGE",
    "VOLUME_Z20",
    "VOLUME_RATIO5_20",

    # Broad market
    "NIFTY_RET1",
    "NIFTY_RET3",
    "NIFTY_RET5",
    "NIFTY_VOL10",
    "RELATIVE_RET1",
    "RELATIVE_RET5",

    # Sector context
    "SECTOR_RET1",
    "SECTOR_RET5",
    "RELATIVE_SECTOR_RET1",

    # Regime
    "ABOVE_SMA20",
    "ABOVE_SMA50",
]

SYMBOL_FEATURES = [
    f"SYMBOL_{symbol.replace('.NS', '')}"
    for symbol in UNIVERSE
]

FEATURES = BASE_FEATURES + SYMBOL_FEATURES


# ============================================================
# SYMBOL / ARTIFACT HELPERS
# ============================================================

def normalize_symbol(symbol: str) -> str:
    clean = str(symbol or "").strip().upper().replace(" ", "")

    if not clean:
        raise ValueError("Stock symbol is required.")

    if clean.startswith("^") or "." in clean:
        return clean

    return f"{clean}.NS"


def short_symbol(symbol: str) -> str:
    return normalize_symbol(symbol).replace(".NS", "")


def artifacts() -> dict[str, Path]:
    return {
        "hist": MODEL_DIR / f"{MODEL_VERSION}_hist.pkl",
        "gbr": MODEL_DIR / f"{MODEL_VERSION}_gbr.pkl",
        "ridge": MODEL_DIR / f"{MODEL_VERSION}_ridge.pkl",
        "metadata": MODEL_DIR / f"{MODEL_VERSION}_metadata.json",
    }


def artifacts_exist() -> bool:
    p = artifacts()

    return all(
        p[key].exists()
        for key in ("hist", "gbr", "ridge", "metadata")
    )


# ============================================================
# DATA HELPERS
# ============================================================

def clean_yfinance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    frame = frame.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        first_level = set(frame.columns.get_level_values(0))

        if "Close" in first_level:
            frame.columns = frame.columns.get_level_values(0)
        else:
            frame.columns = frame.columns.get_level_values(-1)

    required = ["Open", "High", "Low", "Close", "Volume"]

    for column in required:
        if column not in frame.columns:
            raise ValueError(
                f"Missing market column: {column}"
            )

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame[required].dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    frame = frame[
        (frame["Open"] > 0)
        & (frame["High"] > 0)
        & (frame["Low"] > 0)
        & (frame["Close"] > 0)
    ]

    frame = frame[
        ~frame.index.duplicated(keep="last")
    ]

    return frame.sort_index()


def _expected_latest_completed_nse_date() -> pd.Timestamp:
    now_ist = datetime.now(
        NSE_TIMEZONE
    )

    date_value = now_ist.date()

    # Before the normal close window, today's session is not complete.
    if (
        now_ist.weekday() < 5
        and (
            now_ist.hour < 15
            or (
                now_ist.hour == 15
                and now_ist.minute < 45
            )
        )
    ):
        date_value -= timedelta(
            days=1
        )

    while date_value.weekday() >= 5:
        date_value -= timedelta(
            days=1
        )

    return pd.Timestamp(
        date_value
    )


def _append_recent_completed_intraday_session(
    frame: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Yahoo 1d history can lag one completed NSE session. For live inference only,
    supplement the stale daily frame with the most recent completed 5-minute
    session OHLCV when available.
    """
    if frame.empty:
        return frame

    expected_date = (
        _expected_latest_completed_nse_date()
    )

    latest_daily_date = pd.Timestamp(
        frame.index[
            -1
        ]
    ).tz_localize(
        None
    ).normalize()

    if latest_daily_date >= expected_date:
        return frame

    try:
        intraday = yf.Ticker(
            normalize_symbol(
                symbol
            )
        ).history(
            period="5d",
            interval="5m",
            auto_adjust=True,
            prepost=False,
        )

        intraday = clean_yfinance_frame(
            intraday
        )

        if intraday.empty:
            return frame

        intraday_index = pd.to_datetime(
            intraday.index
        )

        if getattr(
            intraday_index,
            "tz",
            None,
        ) is not None:
            intraday_index = intraday_index.tz_convert(
                NSE_TIMEZONE
            )

        session_mask = (
            intraday_index.date
            == expected_date.date()
        )

        session = intraday.loc[
            session_mask
        ]

        if session.empty:
            # Could be an exchange holiday or Yahoo may not have the session.
            return frame

        session = session.sort_index()

        volume = (
            float(
                session[
                    "Volume"
                ].fillna(
                    0
                ).sum()
            )
            if "Volume" in session.columns
            else 0.0
        )

        session_row = pd.DataFrame(
            {
                "Open": [
                    float(
                        session[
                            "Open"
                        ].iloc[
                            0
                        ]
                    )
                ],
                "High": [
                    float(
                        session[
                            "High"
                        ].max()
                    )
                ],
                "Low": [
                    float(
                        session[
                            "Low"
                        ].min()
                    )
                ],
                "Close": [
                    float(
                        session[
                            "Close"
                        ].iloc[
                            -1
                        ]
                    )
                ],
                "Volume": [
                    volume
                ],
            },
            index=[
                expected_date
            ],
        )

        frame = pd.concat(
            [
                frame,
                session_row,
            ]
        )

        frame = frame[
            ~frame.index.duplicated(
                keep="last"
            )
        ].sort_index()

        return frame

    except Exception:
        return frame


def download_daily(
    symbol: str,
    period: str = "8y",
    min_rows: int = 400,
    supplement_recent: bool = False,
) -> pd.DataFrame:
    frame = yf.download(
        normalize_symbol(symbol),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    frame = clean_yfinance_frame(
        frame
    )

    if supplement_recent:
        frame = _append_recent_completed_intraday_session(
            frame,
            symbol,
        )

    if len(frame) < min_rows:
        raise ValueError(
            f"Not enough history for {symbol}: {len(frame)} rows "
            f"(need at least {min_rows})."
        )

    return frame


def sector_proxy_for(symbol: str) -> str:
    short = short_symbol(symbol)

    if short in BANK_STOCKS:
        return "^NSEBANK"

    if short in IT_STOCKS:
        return "^CNXIT"

    return "^NSEI"


# ============================================================
# INDICATORS
# ============================================================

def compute_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(
    frame: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    close = frame["Close"].astype(float)

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def align_context_close(
    frame: pd.DataFrame,
    stock_index: pd.Index,
) -> pd.Series:
    return (
        frame["Close"]
        .astype(float)
        .reindex(stock_index)
        .ffill()
    )


# ============================================================
# STOCK FEATURE ENGINE
# ============================================================

def build_stock_features(
    symbol: str,
    stock: pd.DataFrame,
    nifty: pd.DataFrame,
    sector: pd.DataFrame,
) -> pd.DataFrame:
    symbol = normalize_symbol(symbol)

    close = stock["Close"].astype(float)
    open_ = stock["Open"].astype(float)
    high = stock["High"].astype(float)
    low = stock["Low"].astype(float)
    volume = stock["Volume"].fillna(0).astype(float)

    features = pd.DataFrame(index=stock.index)

    # --------------------------------------------------------
    # PRICE ACTION
    # --------------------------------------------------------

    for window in (1, 2, 3, 5, 10):
        features[f"RET{window}"] = (
            close.pct_change(window)
            * 100.0
        )

    features["GAP_PCT"] = (
        (
            open_
            / close.shift(1).replace(0, np.nan)
        )
        - 1.0
    ) * 100.0

    features["INTRADAY_PCT"] = (
        (
            close
            / open_.replace(0, np.nan)
        )
        - 1.0
    ) * 100.0

    features["RANGE_PCT"] = (
        (high - low)
        / close.replace(0, np.nan)
    ) * 100.0

    features["CLOSE_LOCATION"] = (
        (close - low)
        / (high - low).replace(0, np.nan)
    )

    # --------------------------------------------------------
    # ATR / NORMALIZED MOVE
    # --------------------------------------------------------

    atr14 = compute_atr(
        stock,
        14,
    )

    features["ATR14_PCT"] = (
        atr14
        / close.replace(0, np.nan)
    ) * 100.0

    features["MOVE_ATR_1D"] = (
        close.diff(1)
        / atr14.replace(0, np.nan)
    )

    features["MOVE_ATR_3D"] = (
        close.diff(3)
        / (
            atr14.replace(0, np.nan)
            * np.sqrt(3.0)
        )
    )

    features["MOVE_ATR_5D"] = (
        close.diff(5)
        / (
            atr14.replace(0, np.nan)
            * np.sqrt(5.0)
        )
    )

    daily_return = (
        close.pct_change()
        * 100.0
    )

    for window in (5, 10, 20):
        features[f"VOL{window}"] = (
            daily_return
            .rolling(window)
            .std()
        )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    sma = {}
    ema = {}

    for window in (5, 10, 20, 50):
        sma[window] = (
            close
            .rolling(window)
            .mean()
        )

        ema[window] = (
            close
            .ewm(
                span=window,
                adjust=False,
            )
            .mean()
        )

        features[
            f"SMA{window}_DIST"
        ] = (
            (
                close
                / sma[window].replace(
                    0,
                    np.nan,
                )
            )
            - 1.0
        ) * 100.0

        features[
            f"EMA{window}_DIST"
        ] = (
            (
                close
                / ema[window].replace(
                    0,
                    np.nan,
                )
            )
            - 1.0
        ) * 100.0

    features["EMA5_20_SPREAD"] = (
        (
            ema[5]
            / ema[20].replace(
                0,
                np.nan,
            )
        )
        - 1.0
    ) * 100.0

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    features["RSI14"] = (
        compute_rsi(
            close,
            14,
        )
    )

    ema12 = (
        close
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        close
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    macd = ema12 - ema26

    macd_signal = (
        macd
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    features["MACD_PCT"] = (
        macd
        / close.replace(0, np.nan)
    ) * 100.0

    features["MACD_SIGNAL_PCT"] = (
        macd_signal
        / close.replace(0, np.nan)
    ) * 100.0

    features["MACD_HIST_PCT"] = (
        (macd - macd_signal)
        / close.replace(0, np.nan)
    ) * 100.0

    features["ROC5"] = (
        close.pct_change(5)
        * 100.0
    )

    features["ROC10"] = (
        close.pct_change(10)
        * 100.0
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    log_volume = np.log1p(
        volume
    )

    features["VOLUME_LOG_CHANGE"] = (
        log_volume.diff()
    )

    mean20 = (
        volume
        .rolling(20)
        .mean()
    )

    std20 = (
        volume
        .rolling(20)
        .std()
    )

    features["VOLUME_Z20"] = (
        (volume - mean20)
        / std20.replace(0, np.nan)
    )

    features["VOLUME_RATIO5_20"] = (
        volume.rolling(5).mean()
        / mean20.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # NIFTY CONTEXT
    # --------------------------------------------------------

    nifty_close = align_context_close(
        nifty,
        stock.index,
    )

    nifty_return1 = (
        nifty_close.pct_change()
        * 100.0
    )

    features["NIFTY_RET1"] = (
        nifty_return1
    )

    features["NIFTY_RET3"] = (
        nifty_close.pct_change(3)
        * 100.0
    )

    features["NIFTY_RET5"] = (
        nifty_close.pct_change(5)
        * 100.0
    )

    features["NIFTY_VOL10"] = (
        nifty_return1
        .rolling(10)
        .std()
    )

    features["RELATIVE_RET1"] = (
        features["RET1"]
        - features["NIFTY_RET1"]
    )

    features["RELATIVE_RET5"] = (
        features["RET5"]
        - features["NIFTY_RET5"]
    )

    # --------------------------------------------------------
    # SECTOR CONTEXT
    # --------------------------------------------------------

    sector_close = align_context_close(
        sector,
        stock.index,
    )

    features["SECTOR_RET1"] = (
        sector_close.pct_change()
        * 100.0
    )

    features["SECTOR_RET5"] = (
        sector_close.pct_change(5)
        * 100.0
    )

    features[
        "RELATIVE_SECTOR_RET1"
    ] = (
        features["RET1"]
        - features["SECTOR_RET1"]
    )

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    features["ABOVE_SMA20"] = (
        close > sma[20]
    ).astype(float)

    features["ABOVE_SMA50"] = (
        close > sma[50]
    ).astype(float)

    # --------------------------------------------------------
    # CLIPPING
    # --------------------------------------------------------

    for column in BASE_FEATURES:
        if column in {
            "ABOVE_SMA20",
            "ABOVE_SMA50",
        }:
            continue

        if column == "RSI14":
            features[column] = (
                features[column]
                .clip(0, 100)
            )
        elif column == "CLOSE_LOCATION":
            features[column] = (
                features[column]
                .clip(0, 1)
            )
        elif column == "VOLUME_Z20":
            features[column] = (
                features[column]
                .clip(-8, 8)
            )
        elif column == "VOLUME_RATIO5_20":
            features[column] = (
                features[column]
                .clip(0, 8)
            )
        elif "MOVE_ATR" in column:
            features[column] = (
                features[column]
                .clip(-8, 8)
            )
        else:
            features[column] = (
                features[column]
                .clip(-60, 60)
            )

    # --------------------------------------------------------
    # SYMBOL ONE-HOT
    # --------------------------------------------------------

    active = short_symbol(
        symbol
    )

    for universe_symbol in UNIVERSE:
        short = short_symbol(
            universe_symbol
        )

        features[
            f"SYMBOL_{short}"
        ] = (
            1.0
            if short == active
            else 0.0
        )

    features["ATR14_RS_CURRENT"] = (
        atr14
    )

    return features


# ============================================================
# BUILD UNIVERSAL DATASET
# ============================================================

def download_training_market(
    period: str = "8y",
) -> dict[str, pd.DataFrame]:
    print("Downloading shared market context...")

    data: dict[str, pd.DataFrame] = {}

    data["^NSEI"] = download_daily(
        "^NSEI",
        period,
    )

    try:
        data["^NSEBANK"] = download_daily(
            "^NSEBANK",
            period,
        )
    except Exception:
        data["^NSEBANK"] = data["^NSEI"]

    try:
        data["^CNXIT"] = download_daily(
            "^CNXIT",
            period,
        )
    except Exception:
        data["^CNXIT"] = data["^NSEI"]

    for symbol in UNIVERSE:
        print(f"Downloading {symbol}...")

        try:
            data[symbol] = download_daily(
                symbol,
                period,
            )
        except Exception as exc:
            print(
                f"WARNING: skipping {symbol}: {exc}"
            )

    return data


def build_universal_dataset(
    period: str = "8y",
) -> pd.DataFrame:
    market = download_training_market(
        period
    )

    nifty = market[
        "^NSEI"
    ]

    rows = []

    for symbol in UNIVERSE:
        if symbol not in market:
            continue

        stock = market[
            symbol
        ]

        sector_symbol = sector_proxy_for(
            symbol
        )

        sector = market.get(
            sector_symbol,
            nifty,
        )

        features = build_stock_features(
            symbol,
            stock,
            nifty,
            sector,
        )

        close = stock[
            "Close"
        ].astype(float)

        next_close = (
            close.shift(-1)
        )

        next_residual_rs = (
            next_close - close
        )

        atr = features[
            "ATR14_RS_CURRENT"
        ]

        normalized_move = (
            next_residual_rs
            / atr.replace(0, np.nan)
        )

        target_return_pct = (
            next_residual_rs
            / close.replace(0, np.nan)
        ) * 100.0

        frame = features[
            FEATURES
        ].copy()

        frame["DATE"] = frame.index

        frame["SYMBOL"] = (
            symbol
        )

        frame["BASE_CLOSE"] = (
            close
        )

        frame["ATR14_RS"] = (
            atr
        )

        frame["TARGET_RESIDUAL_RS"] = (
            next_residual_rs
        )

        frame["TARGET_RETURN_PCT"] = (
            target_return_pct
        )

        frame["TARGET_NORMALIZED_MOVE"] = (
            normalized_move
        )

        frame = frame.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        # Protect learners from obvious one-off data errors. The target is not
        # clipped here for evaluation; only the fitted learner target is clipped.
        frame = frame[
            frame[
                "ATR14_RS"
            ] > 0
        ]

        rows.append(
            frame
        )

    if not rows:
        raise ValueError(
            "Could not build any stock dataset."
        )

    dataset = pd.concat(
        rows,
        axis=0,
    )

    dataset = dataset.sort_values(
        [
            "DATE",
            "SYMBOL",
        ]
    ).reset_index(
        drop=True
    )

    if len(dataset) < 5000:
        raise ValueError(
            f"Universal dataset is too small: {len(dataset)} rows."
        )

    return dataset


# ============================================================
# MODELS
# ============================================================

def make_hist_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.035,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=2.5,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=30,
        random_state=RANDOM_SEED,
    )


def make_gbr_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="huber",
        alpha=0.85,
        learning_rate=0.025,
        n_estimators=300,
        max_depth=2,
        min_samples_leaf=35,
        subsample=0.85,
        random_state=RANDOM_SEED,
    )


def make_ridge_model():
    return make_pipeline(
        StandardScaler(),
        Ridge(
            alpha=40.0,
            fit_intercept=True,
        ),
    )


def recency_weights(
    dates: pd.Series,
) -> np.ndarray:
    date_series = pd.to_datetime(
        dates
    )

    minimum = date_series.min()
    maximum = date_series.max()

    total_days = max(
        1,
        (maximum - minimum).days,
    )

    age = (
        date_series - minimum
    ).dt.days.to_numpy(
        dtype=float
    )

    scaled = (
        age
        / total_days
    )

    return (
        0.65
        + 0.70
        * scaled
    )


def fit_models(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    X = frame[
        FEATURES
    ]

    y = frame[
        "TARGET_NORMALIZED_MOVE"
    ].to_numpy(
        dtype=float
    )

    # Only the learner target is clipped. Evaluation always uses the real move.
    target_cap = max(
        1.5,
        float(
            np.percentile(
                np.abs(y),
                99,
            )
        ),
    )

    y_fit = np.clip(
        y,
        -target_cap,
        target_cap,
    )

    weights = recency_weights(
        frame["DATE"]
    )

    hist = make_hist_model()

    hist.fit(
        X,
        y_fit,
        sample_weight=weights,
    )

    gbr = make_gbr_model()

    gbr.fit(
        X,
        y_fit,
        sample_weight=weights,
    )

    ridge = make_ridge_model()

    ridge.fit(
        X,
        y_fit,
        ridge__sample_weight=weights,
    )

    return {
        "hist": hist,
        "gbr": gbr,
        "ridge": ridge,
        "target_cap": target_cap,
    }


def component_predictions(
    models: dict[str, Any],
    X: pd.DataFrame,
) -> dict[str, np.ndarray]:
    return {
        "hist": np.asarray(
            models[
                "hist"
            ].predict(X),
            dtype=float,
        ),
        "gbr": np.asarray(
            models[
                "gbr"
            ].predict(X),
            dtype=float,
        ),
        "ridge": np.asarray(
            models[
                "ridge"
            ].predict(X),
            dtype=float,
        ),
    }


# ============================================================
# METRICS
# ============================================================

def direction_class_from_return(
    return_pct: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        return_pct,
        dtype=float,
    )

    return np.where(
        values
        > DIRECTION_NEUTRAL_PCT,
        1,
        np.where(
            values
            < -DIRECTION_NEUTRAL_PCT,
            -1,
            0,
        ),
    )


def evaluate(
    frame: pd.DataFrame,
    predicted_normalized_move: np.ndarray,
) -> dict[str, float]:
    predicted_normalized_move = np.asarray(
        predicted_normalized_move,
        dtype=float,
    )

    base = frame[
        "BASE_CLOSE"
    ].to_numpy(
        dtype=float
    )

    atr = frame[
        "ATR14_RS"
    ].to_numpy(
        dtype=float
    )

    actual_residual = frame[
        "TARGET_RESIDUAL_RS"
    ].to_numpy(
        dtype=float
    )

    predicted_residual = (
        predicted_normalized_move
        * atr
    )

    actual_close = (
        base
        + actual_residual
    )

    predicted_close = (
        base
        + predicted_residual
    )

    error = np.abs(
        actual_close
        - predicted_close
    )

    actual_return = (
        actual_residual
        / base
        * 100.0
    )

    predicted_return = (
        predicted_residual
        / base
        * 100.0
    )

    return {
        "mae": float(
            np.mean(
                error
            )
        ),
        "median_absolute_error": float(
            np.median(
                error
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        actual_close
                        - predicted_close
                    )
                )
            )
        ),
        "mape_percent": float(
            np.mean(
                error
                / actual_close
            )
            * 100.0
        ),
        "direction_accuracy_percent": float(
            np.mean(
                direction_class_from_return(
                    actual_return
                )
                ==
                direction_class_from_return(
                    predicted_return
                )
            )
            * 100.0
        ),
        "within_2_rupees_percent": float(
            np.mean(
                error <= 2.0
            )
            * 100.0
        ),
        "within_5_rupees_percent": float(
            np.mean(
                error <= 5.0
            )
            * 100.0
        ),
        "within_10_rupees_percent": float(
            np.mean(
                error <= 10.0
            )
            * 100.0
        ),
    }


# ============================================================
# VALIDATION CALIBRATION
# ============================================================

def apply_calibration(
    components: dict[str, np.ndarray],
    calibration: dict[str, Any],
) -> np.ndarray:
    raw = (
        float(
            calibration[
                "hist_weight"
            ]
        )
        * components[
            "hist"
        ]
        +
        float(
            calibration[
                "gbr_weight"
            ]
        )
        * components[
            "gbr"
        ]
        +
        float(
            calibration[
                "ridge_weight"
            ]
        )
        * components[
            "ridge"
        ]
    )

    prediction = (
        float(
            calibration[
                "shrinkage"
            ]
        )
        * raw
        +
        float(
            calibration[
                "bias_normalized"
            ]
        )
    )

    cap = calibration.get(
        "normalized_cap"
    )

    if cap is not None:
        prediction = np.clip(
            prediction,
            -float(cap),
            float(cap),
        )

    return np.asarray(
        prediction,
        dtype=float,
    )


def optimize_calibration(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    components: dict[str, np.ndarray],
) -> dict[str, Any]:
    baseline_prediction = np.zeros(
        len(
            val_frame
        ),
        dtype=float,
    )

    baseline_metrics = evaluate(
        val_frame,
        baseline_prediction,
    )

    baseline_mae = baseline_metrics[
        "mae"
    ]

    train_target = train_frame[
        "TARGET_NORMALIZED_MOVE"
    ].to_numpy(
        dtype=float
    )

    caps: list[
        float | None
    ] = [
        None,
        float(
            np.percentile(
                np.abs(
                    train_target
                ),
                70,
            )
        ),
        float(
            np.percentile(
                np.abs(
                    train_target
                ),
                80,
            )
        ),
        float(
            np.percentile(
                np.abs(
                    train_target
                ),
                90,
            )
        ),
    ]

    best = {
        "hist_weight": 0.0,
        "gbr_weight": 0.0,
        "ridge_weight": 0.0,
        "baseline_weight": 1.0,
        "shrinkage": 0.0,
        "bias_normalized": 0.0,
        "normalized_cap": None,
        "validation_mae": float(
            baseline_mae
        ),
        "baseline_validation_mae": float(
            baseline_mae
        ),
        "validation_improvement_percent": 0.0,
        "accepted": False,
    }

    weights = np.arange(
        0.0,
        1.0001,
        0.10,
    )

    shrinkages = [
        0.25,
        0.50,
        0.75,
        1.00,
    ]

    y_val = val_frame[
        "TARGET_NORMALIZED_MOVE"
    ].to_numpy(
        dtype=float
    )

    for wh in weights:
        for wg in weights:
            for wr in weights:
                total = (
                    wh
                    + wg
                    + wr
                )

                if (
                    total <= 0.0
                    or total > 1.0001
                ):
                    continue

                raw = (
                    wh
                    * components[
                        "hist"
                    ]
                    +
                    wg
                    * components[
                        "gbr"
                    ]
                    +
                    wr
                    * components[
                        "ridge"
                    ]
                )

                for shrinkage in shrinkages:
                    shrunk = (
                        shrinkage
                        * raw
                    )

                    bias = float(
                        np.clip(
                            np.median(
                                y_val
                                - shrunk
                            ),
                            -0.20,
                            0.20,
                        )
                    )

                    for cap in caps:
                        candidate = (
                            shrunk
                            + bias
                        )

                        if cap is not None:
                            candidate = np.clip(
                                candidate,
                                -cap,
                                cap,
                            )

                        candidate_metrics = evaluate(
                            val_frame,
                            candidate,
                        )

                        mae = candidate_metrics[
                            "mae"
                        ]

                        if mae < best[
                            "validation_mae"
                        ]:
                            improvement = (
                                (
                                    baseline_mae
                                    - mae
                                )
                                / baseline_mae
                                * 100.0
                                if baseline_mae
                                > 0
                                else 0.0
                            )

                            best = {
                                "hist_weight": float(
                                    wh
                                ),
                                "gbr_weight": float(
                                    wg
                                ),
                                "ridge_weight": float(
                                    wr
                                ),
                                "baseline_weight": float(
                                    max(
                                        0.0,
                                        1.0
                                        - total,
                                    )
                                ),
                                "shrinkage": float(
                                    shrinkage
                                ),
                                "bias_normalized": float(
                                    bias
                                ),
                                "normalized_cap": (
                                    float(
                                        cap
                                    )
                                    if cap
                                    is not None
                                    else None
                                ),
                                "validation_mae": float(
                                    mae
                                ),
                                "baseline_validation_mae": float(
                                    baseline_mae
                                ),
                                "validation_improvement_percent": float(
                                    improvement
                                ),
                                "accepted": True,
                            }

    if (
        best[
            "validation_improvement_percent"
        ]
        < MIN_VALIDATION_IMPROVEMENT_PCT
    ):
        return {
            "hist_weight": 0.0,
            "gbr_weight": 0.0,
            "ridge_weight": 0.0,
            "baseline_weight": 1.0,
            "shrinkage": 0.0,
            "bias_normalized": 0.0,
            "normalized_cap": None,
            "validation_mae": float(
                baseline_mae
            ),
            "baseline_validation_mae": float(
                baseline_mae
            ),
            "validation_improvement_percent": 0.0,
            "accepted": False,
        }

    return best


# ============================================================
# DATE-BASED SPLITS
# ============================================================

def unique_dates(
    frame: pd.DataFrame,
) -> np.ndarray:
    return np.sort(
        pd.to_datetime(
            frame[
                "DATE"
            ]
        ).dt.normalize().unique()
    )


def split_by_date(
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    dates = unique_dates(
        dataset
    )

    train_end_index = int(
        len(dates)
        * TRAIN_RATIO
    )

    val_end_index = int(
        len(dates)
        * (
            TRAIN_RATIO
            + VAL_RATIO
        )
    )

    train_end_date = dates[
        train_end_index
        - 1
    ]

    val_end_date = dates[
        val_end_index
        - 1
    ]

    date_series = pd.to_datetime(
        dataset[
            "DATE"
        ]
    ).dt.normalize()

    train = dataset[
        date_series
        <= train_end_date
    ].copy()

    val = dataset[
        (date_series
        > train_end_date)
        &
        (date_series
        <= val_end_date)
    ].copy()

    test = dataset[
        date_series
        > val_end_date
    ].copy()

    if min(
        len(train),
        len(val),
        len(test),
    ) < 500:
        raise ValueError(
            "Universal chronological split produced too few rows."
        )

    return train, val, test


# ============================================================
# PER-STOCK METRICS
# ============================================================

def per_stock_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, Any]:
    working = frame[
        [
            "SYMBOL",
            "BASE_CLOSE",
            "ATR14_RS",
            "TARGET_RESIDUAL_RS",
            "TARGET_RETURN_PCT",
        ]
    ].copy()

    working[
        "PREDICTED_NORMALIZED"
    ] = np.asarray(
        prediction,
        dtype=float,
    )

    result = {}

    for symbol, group in working.groupby(
        "SYMBOL",
        sort=True,
    ):
        group_prediction = group[
            "PREDICTED_NORMALIZED"
        ].to_numpy(
            dtype=float
        )

        model_metrics = evaluate(
            group,
            group_prediction,
        )

        baseline_metrics = evaluate(
            group,
            np.zeros(
                len(
                    group
                ),
                dtype=float,
            ),
        )

        improvement = (
            (
                baseline_metrics[
                    "mae"
                ]
                - model_metrics[
                    "mae"
                ]
            )
            / baseline_metrics[
                "mae"
            ]
            * 100.0
            if baseline_metrics[
                "mae"
            ] > 0
            else 0.0
        )

        result[
            symbol
        ] = {
            "samples": int(
                len(
                    group
                )
            ),
            "model": model_metrics,
            "baseline": baseline_metrics,
            "mae_improvement_vs_baseline_percent": float(
                improvement
            ),
            "beats_baseline": bool(
                model_metrics[
                    "mae"
                ]
                < baseline_metrics[
                    "mae"
                ]
            ),
        }

    return result


# ============================================================
# WALK-FORWARD
# ============================================================

def walk_forward_evaluate(
    dataset: pd.DataFrame,
    folds: int = WALK_FORWARD_FOLDS,
) -> dict[str, Any]:
    dates = unique_dates(
        dataset
    )

    initial_index = int(
        len(
            dates
        )
        * WALK_FORWARD_INITIAL_DATE_RATIO
    )

    remaining = (
        len(
            dates
        )
        - initial_index
    )

    fold_date_count = max(
        60,
        remaining // folds,
    )

    fold_results = []

    accumulated_frames = []
    accumulated_predictions = []

    date_series = pd.to_datetime(
        dataset[
            "DATE"
        ]
    ).dt.normalize()

    for fold_index in range(
        folds
    ):
        test_start_index = (
            initial_index
            + fold_index
            * fold_date_count
        )

        if test_start_index >= len(
            dates
        ):
            break

        if fold_index == folds - 1:
            test_end_index = len(
                dates
            )
        else:
            test_end_index = min(
                len(
                    dates
                ),
                test_start_index
                + fold_date_count,
            )

        test_dates = dates[
            test_start_index:test_end_index
        ]

        if len(
            test_dates
        ) < 30:
            continue

        development_end_date = dates[
            test_start_index
            - 1
        ]

        development = dataset[
            date_series
            <= development_end_date
        ].copy()

        development_dates = unique_dates(
            development
        )

        validation_date_count = max(
            60,
            int(
                len(
                    development_dates
                )
                * 0.15
            ),
        )

        if (
            len(
                development_dates
            )
            <= validation_date_count
            + 100
        ):
            continue

        validation_start_date = (
            development_dates[
                -validation_date_count
            ]
        )

        dev_date_series = pd.to_datetime(
            development[
                "DATE"
            ]
        ).dt.normalize()

        train = development[
            dev_date_series
            < validation_start_date
        ].copy()

        val = development[
            dev_date_series
            >= validation_start_date
        ].copy()

        test_start_date = test_dates[
            0
        ]

        test_end_date = test_dates[
            -1
        ]

        test = dataset[
            (date_series
            >= test_start_date)
            &
            (date_series
            <= test_end_date)
        ].copy()

        models = fit_models(
            train
        )

        val_components = component_predictions(
            models,
            val[
                FEATURES
            ],
        )

        calibration = optimize_calibration(
            train,
            val,
            val_components,
        )

        test_components = component_predictions(
            models,
            test[
                FEATURES
            ],
        )

        test_prediction = apply_calibration(
            test_components,
            calibration,
        )

        model_metrics = evaluate(
            test,
            test_prediction,
        )

        baseline_metrics = evaluate(
            test,
            np.zeros(
                len(
                    test
                ),
                dtype=float,
            ),
        )

        fold_results.append(
            {
                "fold": fold_index + 1,
                "test_start": str(
                    pd.Timestamp(
                        test_start_date
                    ).date()
                ),
                "test_end": str(
                    pd.Timestamp(
                        test_end_date
                    ).date()
                ),
                "samples": int(
                    len(
                        test
                    )
                ),
                "model": model_metrics,
                "baseline": baseline_metrics,
                "beats_baseline": bool(
                    model_metrics[
                        "mae"
                    ]
                    < baseline_metrics[
                        "mae"
                    ]
                ),
                "calibration": calibration,
            }
        )

        accumulated_frames.append(
            test
        )

        accumulated_predictions.append(
            test_prediction
        )

    if not fold_results:
        raise ValueError(
            "No valid universal walk-forward folds were created."
        )

    combined_frame = pd.concat(
        accumulated_frames,
        axis=0,
    )

    combined_prediction = np.concatenate(
        accumulated_predictions
    )

    model_overall = evaluate(
        combined_frame,
        combined_prediction,
    )

    baseline_overall = evaluate(
        combined_frame,
        np.zeros(
            len(
                combined_frame
            ),
            dtype=float,
        ),
    )

    improvement = (
        (
            baseline_overall[
                "mae"
            ]
            - model_overall[
                "mae"
            ]
        )
        / baseline_overall[
            "mae"
        ]
        * 100.0
        if baseline_overall[
            "mae"
        ] > 0
        else 0.0
    )

    return {
        "fold_count": int(
            len(
                fold_results
            )
        ),
        "samples": int(
            len(
                combined_frame
            )
        ),
        "folds_beating_baseline": int(
            sum(
                1
                for row
                in fold_results
                if row[
                    "beats_baseline"
                ]
            )
        ),
        "model": model_overall,
        "baseline": baseline_overall,
        "mae_improvement_vs_baseline_percent": float(
            improvement
        ),
        "folds": fold_results,
    }


# ============================================================
# ERROR BANDS
# ============================================================

def error_bands_by_symbol(
    frame: pd.DataFrame,
    predicted_normalized: np.ndarray,
) -> dict[str, Any]:
    working = frame[
        [
            "SYMBOL",
            "ATR14_RS",
            "TARGET_RESIDUAL_RS",
        ]
    ].copy()

    working[
        "PREDICTED_NORMALIZED"
    ] = np.asarray(
        predicted_normalized,
        dtype=float,
    )

    working[
        "PREDICTED_RESIDUAL_RS"
    ] = (
        working[
            "PREDICTED_NORMALIZED"
        ]
        * working[
            "ATR14_RS"
        ]
    )

    working[
        "ERROR_RS"
    ] = (
        working[
            "TARGET_RESIDUAL_RS"
        ]
        - working[
            "PREDICTED_RESIDUAL_RS"
        ]
    )

    result = {}

    overall_error = working[
        "ERROR_RS"
    ].to_numpy(
        dtype=float
    )

    result[
        "__GLOBAL__"
    ] = {
        "q10_rs": float(
            np.quantile(
                overall_error,
                0.10,
            )
        ),
        "q90_rs": float(
            np.quantile(
                overall_error,
                0.90,
            )
        ),
    }

    for symbol, group in working.groupby(
        "SYMBOL"
    ):
        errors = group[
            "ERROR_RS"
        ].to_numpy(
            dtype=float
        )

        if len(
            errors
        ) < 30:
            continue

        result[
            symbol
        ] = {
            "q10_rs": float(
                np.quantile(
                    errors,
                    0.10,
                )
            ),
            "q90_rs": float(
                np.quantile(
                    errors,
                    0.90,
                )
            ),
        }

    return result


# ============================================================
# TRAIN UNIVERSAL MODEL
# ============================================================

def train_universal_v3() -> dict[str, Any]:
    print(
        "\n"
        + "=" * 78
    )
    print(
        " STOCKVISION PRECISION V3 — UNIVERSAL"
    )
    print(
        " Multi-Stock ATR-Normalized Next-Day Forecast"
    )
    print(
        "=" * 78
    )

    dataset = build_universal_dataset(
        period="8y"
    )

    print(
        f"\nUniversal rows: {len(dataset):,}"
    )
    print(
        f"Stocks used: {dataset['SYMBOL'].nunique()}"
    )
    print(
        f"Features: {len(FEATURES)}"
    )
    print(
        f"Date range: "
        f"{pd.Timestamp(dataset['DATE'].min()).date()} -> "
        f"{pd.Timestamp(dataset['DATE'].max()).date()}"
    )

    print(
        "\nRunning universal expanding walk-forward evaluation..."
    )

    walk_forward = walk_forward_evaluate(
        dataset
    )

    train, val, test = split_by_date(
        dataset
    )

    print(
        f"\nChronological rows -> "
        f"Train {len(train):,} | "
        f"Val {len(val):,} | "
        f"Test {len(test):,}"
    )

    models = fit_models(
        train
    )

    val_components = component_predictions(
        models,
        val[
            FEATURES
        ],
    )

    calibration = optimize_calibration(
        train,
        val,
        val_components,
    )

    val_prediction = apply_calibration(
        val_components,
        calibration,
    )

    validation_error_bands = error_bands_by_symbol(
        val,
        val_prediction,
    )

    test_components = component_predictions(
        models,
        test[
            FEATURES
        ],
    )

    test_prediction = apply_calibration(
        test_components,
        calibration,
    )

    model_metrics = evaluate(
        test,
        test_prediction,
    )

    baseline_metrics = evaluate(
        test,
        np.zeros(
            len(
                test
            ),
            dtype=float,
        ),
    )

    test_improvement = (
        (
            baseline_metrics[
                "mae"
            ]
            - model_metrics[
                "mae"
            ]
        )
        / baseline_metrics[
            "mae"
        ]
        * 100.0
        if baseline_metrics[
            "mae"
        ] > 0
        else 0.0
    )

    symbol_metrics = per_stock_metrics(
        test,
        test_prediction,
    )

    stocks_beating = sum(
        1
        for metrics
        in symbol_metrics.values()
        if metrics[
            "beats_baseline"
        ]
    )

    stock_count = len(
        symbol_metrics
    )

    validation_accepted = bool(
        calibration[
            "accepted"
        ]
    )

    holdout_beats = (
        model_metrics[
            "mae"
        ]
        < baseline_metrics[
            "mae"
        ]
    )

    walk_forward_beats = (
        walk_forward[
            "model"
        ][
            "mae"
        ]
        < walk_forward[
            "baseline"
        ][
            "mae"
        ]
    )

    enough_folds_beat = (
        walk_forward[
            "folds_beating_baseline"
        ]
        >= MIN_WALK_FORWARD_FOLDS_BEATING
    )

    majority_stocks_beat = (
        stocks_beating
        > stock_count / 2.0
    )

    accepted = (
        validation_accepted
        and holdout_beats
        and walk_forward_beats
        and enough_folds_beat
        and majority_stocks_beat
    )

    production_status = (
        "ACCEPTED"
        if accepted
        else "EXPERIMENTAL"
    )

    # Production models train on train + validation only.
    development = pd.concat(
        [
            train,
            val,
        ],
        axis=0,
    )

    production_models = fit_models(
        development
    )

    p = artifacts()

    joblib.dump(
        production_models[
            "hist"
        ],
        p[
            "hist"
        ],
    )

    joblib.dump(
        production_models[
            "gbr"
        ],
        p[
            "gbr"
        ],
    )

    joblib.dump(
        production_models[
            "ridge"
        ],
        p[
            "ridge"
        ],
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "model_name": (
            "Precision V3 Universal ATR-Normalized Ensemble"
        ),
        "production_status": production_status,
        "universe": UNIVERSE,
        "feature_count": len(
            FEATURES
        ),
        "features": FEATURES,
        "dataset": {
            "rows": int(
                len(
                    dataset
                )
            ),
            "stocks": int(
                dataset[
                    "SYMBOL"
                ].nunique()
            ),
            "date_start": str(
                pd.Timestamp(
                    dataset[
                        "DATE"
                    ].min()
                ).date()
            ),
            "date_end": str(
                pd.Timestamp(
                    dataset[
                        "DATE"
                    ].max()
                ).date()
            ),
        },
        "calibration": calibration,
        "validation_error_bands": validation_error_bands,
        "holdout_test": {
            "model": model_metrics,
            "baseline": baseline_metrics,
            "mae_improvement_vs_baseline_percent": float(
                test_improvement
            ),
            "beats_baseline": bool(
                holdout_beats
            ),
            "stocks_beating_baseline": int(
                stocks_beating
            ),
            "stock_count": int(
                stock_count
            ),
            "per_stock": symbol_metrics,
        },
        "walk_forward": walk_forward,
        "acceptance": {
            "validation_accepted": validation_accepted,
            "holdout_overall_beats_baseline": bool(
                holdout_beats
            ),
            "walk_forward_overall_beats_baseline": bool(
                walk_forward_beats
            ),
            "at_least_3_of_4_folds_beat_baseline": bool(
                enough_folds_beat
            ),
            "majority_holdout_stocks_beat_baseline": bool(
                majority_stocks_beat
            ),
        },
        "notes": [
            "Target is next-day close residual divided by current ATR14.",
            "All stock/date splits are chronological by date.",
            "Same future dates are never randomly split across stocks.",
            "Historical prices use auto-adjusted data to reduce corporate-action distortion.",
            "Previous close is explicit zero-move baseline.",
            "Rs 1-2 next-day error is measured, never guaranteed.",
        ],
    }

    p[
        "metadata"
    ].write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 78
    )
    print(
        " UNIVERSAL HOLDOUT TEST"
    )
    print(
        "=" * 78
    )
    print(
        f"Model MAE:             Rs {model_metrics['mae']:.2f}"
    )
    print(
        f"Baseline MAE:          Rs {baseline_metrics['mae']:.2f}"
    )
    print(
        f"MAE improvement:       {test_improvement:+.2f}%"
    )
    print(
        f"Median abs error:      Rs {model_metrics['median_absolute_error']:.2f}"
    )
    print(
        f"RMSE:                  Rs {model_metrics['rmse']:.2f}"
    )
    print(
        f"MAPE:                  {model_metrics['mape_percent']:.3f}%"
    )
    print(
        f"Direction accuracy:    {model_metrics['direction_accuracy_percent']:.2f}%"
    )
    print(
        f"Within Rs 2:           {model_metrics['within_2_rupees_percent']:.2f}%"
    )
    print(
        f"Within Rs 5:           {model_metrics['within_5_rupees_percent']:.2f}%"
    )
    print(
        f"Within Rs 10:          {model_metrics['within_10_rupees_percent']:.2f}%"
    )
    print(
        f"Stocks beating baseline: {stocks_beating}/{stock_count}"
    )

    reliance_metrics = symbol_metrics.get(
        "RELIANCE.NS"
    )

    if reliance_metrics:
        print(
            "\nRELIANCE HOLDOUT"
        )
        print(
            f"Model MAE:             Rs {reliance_metrics['model']['mae']:.2f}"
        )
        print(
            f"Baseline MAE:          Rs {reliance_metrics['baseline']['mae']:.2f}"
        )
        print(
            f"Improvement:           "
            f"{reliance_metrics['mae_improvement_vs_baseline_percent']:+.2f}%"
        )
        print(
            f"Within Rs 2:           "
            f"{reliance_metrics['model']['within_2_rupees_percent']:.2f}%"
        )
        print(
            f"Within Rs 5:           "
            f"{reliance_metrics['model']['within_5_rupees_percent']:.2f}%"
        )
        print(
            f"Within Rs 10:          "
            f"{reliance_metrics['model']['within_10_rupees_percent']:.2f}%"
        )

    print(
        "\n"
        + "=" * 78
    )
    print(
        " UNIVERSAL WALK-FORWARD"
    )
    print(
        "=" * 78
    )
    print(
        f"Folds beating baseline: "
        f"{walk_forward['folds_beating_baseline']}/"
        f"{walk_forward['fold_count']}"
    )
    print(
        f"Model MAE:             Rs {walk_forward['model']['mae']:.2f}"
    )
    print(
        f"Baseline MAE:          Rs {walk_forward['baseline']['mae']:.2f}"
    )
    print(
        f"Improvement:           "
        f"{walk_forward['mae_improvement_vs_baseline_percent']:+.2f}%"
    )

    print(
        "\nSelected validation calibration:"
    )
    print(
        f"Hist={calibration['hist_weight']:.2f} | "
        f"GBR={calibration['gbr_weight']:.2f} | "
        f"Ridge={calibration['ridge_weight']:.2f} | "
        f"Baseline={calibration['baseline_weight']:.2f}"
    )
    print(
        f"Shrinkage={calibration['shrinkage']:.2f} | "
        f"Bias={calibration['bias_normalized']:+.4f} ATR | "
        f"Cap={calibration['normalized_cap']}"
    )

    print(
        "\n"
        + "=" * 78
    )
    print(
        f"PRODUCTION STATUS: {production_status}"
    )
    print(
        "=" * 78
    )

    if production_status != "ACCEPTED":
        print(
            "Universal V3 is saved for analysis, but should not be claimed "
            "as a validated improvement yet."
        )

    return metadata


# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================

def load_metadata() -> dict[str, Any]:
    return json.loads(
        artifacts()[
            "metadata"
        ].read_text(
            encoding="utf-8"
        )
    )


def load_models() -> dict[str, Any]:
    p = artifacts()

    return {
        "hist": joblib.load(
            p[
                "hist"
            ]
        ),
        "gbr": joblib.load(
            p[
                "gbr"
            ]
        ),
        "ridge": joblib.load(
            p[
                "ridge"
            ]
        ),
    }


# ============================================================
# LIVE SINGLE-STOCK FEATURE ROW
# ============================================================

def live_feature_row(
    symbol: str,
) -> tuple[
    pd.DataFrame,
    float,
    float,
    pd.Timestamp,
]:
    symbol = normalize_symbol(
        symbol
    )

    stock = download_daily(
        symbol,
        period="2y",
        min_rows=80,
        supplement_recent=True,
    )

    nifty = download_daily(
        "^NSEI",
        period="2y",
        min_rows=80,
        supplement_recent=True,
    )

    sector_symbol = sector_proxy_for(
        symbol
    )

    if sector_symbol == "^NSEI":
        sector = nifty
    else:
        try:
            sector = download_daily(
                sector_symbol,
                period="2y",
                min_rows=80,
                supplement_recent=True,
            )
        except Exception:
            sector = nifty

    feature_frame = build_stock_features(
        symbol,
        stock,
        nifty,
        sector,
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if feature_frame.empty:
        raise ValueError(
            f"No valid latest feature row for {symbol}."
        )

    latest_date = pd.Timestamp(
        feature_frame.index[
            -1
        ]
    )

    X = feature_frame[
        FEATURES
    ].iloc[
        [-1]
    ]

    current_close = float(
        stock.loc[
            stock.index[
                stock.index
                <= latest_date
            ][
                -1
            ],
            "Close",
        ]
    )

    current_atr = float(
        feature_frame[
            "ATR14_RS_CURRENT"
        ].iloc[
            -1
        ]
    )

    return (
        X,
        current_close,
        current_atr,
        latest_date,
    )


# ============================================================
# HISTORICAL REPLAY / BACKFILL
# ============================================================

def historical_replay_v3(
    symbol: str,
    period: str = "2y",
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """
    Re-run the frozen production V3 model across historical feature rows.

    These rows are HISTORICAL REPLAY rows, not original live forecasts.
    They are useful for filling an all-date history table honestly.

    Each row predicts the next available stock trading-day close using only
    information available on the base date.
    """
    symbol = normalize_symbol(
        symbol
    )

    if not artifacts_exist():
        raise FileNotFoundError(
            "Universal V3 model artifacts are not available."
        )

    metadata = load_metadata()
    models = load_models()

    stock = download_daily(
        symbol,
        period=period,
        min_rows=80,
        supplement_recent=True,
    )

    nifty = download_daily(
        "^NSEI",
        period=period,
        min_rows=80,
        supplement_recent=True,
    )

    sector_symbol = sector_proxy_for(
        symbol
    )

    if sector_symbol == "^NSEI":
        sector = nifty
    else:
        try:
            sector = download_daily(
                sector_symbol,
                period=period,
                min_rows=80,
                supplement_recent=True,
            )
        except Exception:
            sector = nifty

    feature_frame = build_stock_features(
        symbol,
        stock,
        nifty,
        sector,
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if feature_frame.empty:
        return []

    # Only feature rows that exist on actual stock trading dates.
    feature_frame = feature_frame[
        feature_frame.index.isin(
            stock.index
        )
    ]

    if feature_frame.empty:
        return []

    X = feature_frame[
        FEATURES
    ]

    components = component_predictions(
        models,
        X,
    )

    normalized_prediction = apply_calibration(
        components,
        metadata[
            "calibration"
        ],
    )

    base_close = (
        stock[
            "Close"
        ]
        .astype(float)
        .reindex(
            feature_frame.index
        )
    )

    atr = (
        feature_frame[
            "ATR14_RS_CURRENT"
        ]
        .astype(float)
    )

    predicted_move = (
        np.asarray(
            normalized_prediction,
            dtype=float,
        )
        * atr.to_numpy(
            dtype=float
        )
    )

    predicted_close = (
        base_close.to_numpy(
            dtype=float
        )
        + predicted_move
    )

    # Map each stock trading date to the next actual stock trading date/close.
    stock_sorted = stock.sort_index()

    next_date_map = {}
    next_close_map = {}

    stock_dates = list(
        stock_sorted.index
    )

    stock_closes = (
        stock_sorted[
            "Close"
        ]
        .astype(float)
        .to_numpy()
    )

    for index in range(
        len(
            stock_dates
        )
        - 1
    ):
        base_index = stock_dates[
            index
        ]

        next_index = stock_dates[
            index + 1
        ]

        next_date_map[
            pd.Timestamp(
                base_index
            )
        ] = pd.Timestamp(
            next_index
        )

        next_close_map[
            pd.Timestamp(
                base_index
            )
        ] = float(
            stock_closes[
                index + 1
            ]
        )

    error_bands = metadata.get(
        "validation_error_bands",
        {},
    )

    band = error_bands.get(
        symbol,
        error_bands.get(
            "__GLOBAL__",
            {
                "q10_rs": 0.0,
                "q90_rs": 0.0,
            },
        ),
    )

    lower_offset = float(
        band.get(
            "q10_rs",
            0.0,
        )
    )

    upper_offset = float(
        band.get(
            "q90_rs",
            0.0,
        )
    )

    rows = []

    for row_index, base_index in enumerate(
        feature_frame.index
    ):
        base_timestamp = pd.Timestamp(
            base_index
        )

        target_timestamp = next_date_map.get(
            base_timestamp
        )

        actual_close = next_close_map.get(
            base_timestamp
        )

        if (
            target_timestamp is None
            or actual_close is None
        ):
            # No next completed trading day yet -> do not fabricate resolution.
            continue

        base_value = float(
            base_close.loc[
                base_index
            ]
        )

        prediction = float(
            predicted_close[
                row_index
            ]
        )

        move = (
            prediction
            - base_value
        )

        predicted_return = (
            move
            / base_value
            * 100.0
            if base_value
            else None
        )

        actual_return = (
            (
                actual_close
                - base_value
            )
            / base_value
            * 100.0
            if base_value
            else None
        )

        difference = (
            actual_close
            - prediction
        )

        absolute_error = abs(
            difference
        )

        error_percent = (
            absolute_error
            / abs(
                actual_close
            )
            * 100.0
            if actual_close
            else None
        )

        predicted_direction = (
            "BULLISH"
            if predicted_return is not None
            and predicted_return > DIRECTION_NEUTRAL_PCT
            else "BEARISH"
            if predicted_return is not None
            and predicted_return < -DIRECTION_NEUTRAL_PCT
            else "NEUTRAL"
        )

        actual_direction = (
            "BULLISH"
            if actual_return is not None
            and actual_return > DIRECTION_NEUTRAL_PCT
            else "BEARISH"
            if actual_return is not None
            and actual_return < -DIRECTION_NEUTRAL_PCT
            else "NEUTRAL"
        )

        range_lower = (
            prediction
            + lower_offset
        )

        range_upper = (
            prediction
            + upper_offset
        )

        range_low = min(
            range_lower,
            range_upper,
        )

        range_high = max(
            range_lower,
            range_upper,
        )

        rows.append(
            {
                "symbol": symbol,
                "base_date": base_timestamp.date().isoformat(),
                "target_date": target_timestamp.date().isoformat(),
                "record_type": "HISTORICAL_REPLAY",
                "history_source": "BACKFILLED_MODEL_REPLAY",
                "is_live_capture": False,
                "status": "RESOLVED",
                "current_close": round(
                    base_value,
                    4,
                ),
                "predicted_price": round(
                    prediction,
                    4,
                ),
                "forecast_point_price": round(
                    prediction,
                    4,
                ),
                "predicted_return_percent": (
                    round(
                        predicted_return,
                        6,
                    )
                    if predicted_return is not None
                    else None
                ),
                "forecast_return_percent": (
                    round(
                        predicted_return,
                        6,
                    )
                    if predicted_return is not None
                    else None
                ),
                "actual_close": round(
                    actual_close,
                    4,
                ),
                "actual_return_percent": (
                    round(
                        actual_return,
                        6,
                    )
                    if actual_return is not None
                    else None
                ),
                "forecast_difference_actual_minus_prediction": round(
                    difference,
                    4,
                ),
                "forecast_absolute_error": round(
                    absolute_error,
                    4,
                ),
                "forecast_error_percent": (
                    round(
                        error_percent,
                        6,
                    )
                    if error_percent is not None
                    else None
                ),
                "absolute_error": round(
                    absolute_error,
                    4,
                ),
                "percentage_error": (
                    round(
                        error_percent,
                        6,
                    )
                    if error_percent is not None
                    else None
                ),
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "direction_correct": (
                    predicted_direction
                    == actual_direction
                ),
                "expected_range_lower": round(
                    range_low,
                    4,
                ),
                "expected_range_upper": round(
                    range_high,
                    4,
                ),
                "inside_expected_range": (
                    range_low
                    <= actual_close
                    <= range_high
                ),
                "resolution_data_status": "HISTORICAL_MODEL_REPLAY",
                "resolution_note": (
                    "Historical replay generated with the frozen production "
                    "model using features available on the base date."
                ),
            }
        )

    rows.sort(
        key=lambda item: item[
            "base_date"
        ],
        reverse=True,
    )

    if (
        max_rows is not None
        and max_rows > 0
    ):
        rows = rows[
            :max_rows
        ]

    return rows


# ============================================================
# PREDICT
# ============================================================

def predict_universal_v3(
    symbol: str,
    auto_train: bool = True,
) -> dict[str, Any]:
    symbol = normalize_symbol(
        symbol
    )

    if not artifacts_exist():
        if not auto_train:
            raise FileNotFoundError(
                "Universal V3 is not trained."
            )

        train_universal_v3()

    metadata = load_metadata()

    models = load_models()

    (
        X,
        current_close,
        current_atr,
        latest_date,
    ) = live_feature_row(
        symbol
    )

    components = component_predictions(
        models,
        X,
    )

    calibration = metadata[
        "calibration"
    ]

    normalized_prediction = float(
        apply_calibration(
            components,
            calibration,
        )[
            0
        ]
    )

    predicted_rupee_move = (
        normalized_prediction
        * current_atr
    )

    predicted_close = (
        current_close
        + predicted_rupee_move
    )

    predicted_return_percent = (
        predicted_rupee_move
        / current_close
        * 100.0
    )

    signal = (
        "BULLISH"
        if predicted_return_percent
        > DIRECTION_NEUTRAL_PCT
        else "BEARISH"
        if predicted_return_percent
        < -DIRECTION_NEUTRAL_PCT
        else "NEUTRAL"
    )

    error_bands = metadata.get(
        "validation_error_bands",
        {}
    )

    band = error_bands.get(
        symbol,
        error_bands.get(
            "__GLOBAL__",
            {
                "q10_rs": 0.0,
                "q90_rs": 0.0,
            },
        ),
    )

    lower = (
        predicted_close
        + float(
            band.get(
                "q10_rs",
                0.0,
            )
        )
    )

    upper = (
        predicted_close
        + float(
            band.get(
                "q90_rs",
                0.0,
            )
        )
    )

    per_stock = (
        metadata.get(
            "holdout_test",
            {}
        )
        .get(
            "per_stock",
            {}
        )
        .get(
            symbol
        )
    )

    trained_universe_member = (
        symbol
        in metadata.get(
            "universe",
            [],
        )
    )

    if not trained_universe_member:
        symbol_status = (
            "UNSEEN_EXPERIMENTAL"
        )
    else:
        symbol_status = (
            "VALIDATED"
            if (
                per_stock
                and per_stock.get(
                    "beats_baseline"
                )
            )
            else "EXPERIMENTAL"
        )

    return {
        "symbol": symbol,
        "latest_market_date": str(
            latest_date.date()
        ),
        "current_close": round(
            current_close,
            4,
        ),
        "predicted_price": round(
            predicted_close,
            4,
        ),
        "predicted_residual_rupees": round(
            predicted_rupee_move,
            4,
        ),
        "predicted_return_percent": round(
            predicted_return_percent,
            6,
        ),
        "trend_signal": signal,
        "expected_range": {
            "lower": round(
                min(
                    lower,
                    upper,
                ),
                4,
            ),
            "upper": round(
                max(
                    lower,
                    upper,
                ),
                4,
            ),
            "basis": (
                "10th-90th percentile validation forecast-error band"
            ),
        },
        "model": metadata[
            "model_name"
        ],
        "model_version": MODEL_VERSION,
        "production_status": metadata[
            "production_status"
        ],
        "symbol_validation_status": symbol_status,
        "trained_universe_member": trained_universe_member,
        "coverage_mode": (
            "TRAINED_UNIVERSE"
            if trained_universe_member
            else "GENERALIZED_UNSEEN_STOCK"
        ),
        "generalization_note": (
            "This symbol was part of the universal training universe."
            if trained_universe_member
            else (
                "This symbol was not part of the original universal training "
                "universe. Stock-agnostic technical and market features are "
                "used with all symbol identity flags set to zero. Treat the "
                "forecast as experimental generalization, not symbol-specific "
                "validated performance."
            )
        ),
        "current_atr14_rupees": round(
            current_atr,
            4,
        ),
        "predicted_normalized_move_atr": round(
            normalized_prediction,
            6,
        ),
        "raw_components_normalized_atr": {
            "hist": round(
                float(
                    components[
                        "hist"
                    ][
                        0
                    ]
                ),
                6,
            ),
            "gradient_boosting": round(
                float(
                    components[
                        "gbr"
                    ][
                        0
                    ]
                ),
                6,
            ),
            "ridge": round(
                float(
                    components[
                        "ridge"
                    ][
                        0
                    ]
                ),
                6,
            ),
            "previous_close_baseline": 0.0,
        },
        "blend": calibration,
        "evaluation": {
            "universal_holdout": metadata.get(
                "holdout_test"
            ),
            "symbol_holdout": per_stock,
            "walk_forward_summary": {
                key: metadata.get(
                    "walk_forward",
                    {}
                ).get(
                    key
                )
                for key in [
                    "fold_count",
                    "samples",
                    "folds_beating_baseline",
                    "model",
                    "baseline",
                    "mae_improvement_vs_baseline_percent",
                ]
            },
        },
        "precision_note": (
            "Universal V3 learns ATR-normalized next-day moves across many "
            "liquid NSE stocks and is accepted only when chronological "
            "validation, holdout and walk-forward results beat the "
            "previous-close baseline. Rs 1-2 accuracy is measured, not "
            "guaranteed."
        ),
    }
