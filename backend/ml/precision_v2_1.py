"""
StockVision Precision V2.1 — Residual MAE Ensemble
==================================================

Goal
----
Predict the next NSE trading-day close as:

    predicted_close = current_close + predicted_residual_rupees

Instead of forcing a neural network to predict the full stock price, V2.1
learns only the correction (residual) that should be applied to today's close.

The model is evaluation-first:
- chronological train / validation / test
- 4-fold expanding walk-forward evaluation
- previous-close baseline = residual Rs 0
- validation-only ensemble weighting
- validation-only shrinkage toward previous close
- automatic production rejection if unseen evaluation does not beat baseline

No model can guarantee a Rs 1-2 next-day error. V2.1 measures how often it
actually lands within Rs 2 / Rs 5 / Rs 10 on unseen data.
"""

from __future__ import annotations

import json
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

MODEL_VERSION = "precision-v2.1"
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

DIRECTION_NEUTRAL_RS = 2.0

MIN_VALIDATION_MAE_IMPROVEMENT_PCT = 0.50
MIN_WALK_FORWARD_FOLDS_BEATING = 3
WALK_FORWARD_FOLDS = 4
WALK_FORWARD_INITIAL_TRAIN_RATIO = 0.55

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
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

    # Rupee scale / realized movement
    "PRICE_LOG",
    "ATR14_RS",
    "ATR14_PCT",
    "RANGE_RS",
    "AVG_RANGE5_RS",
    "AVG_RANGE20_RS",
    "ABS_MOVE5_RS",
    "ABS_MOVE20_RS",

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

    # Volatility
    "VOL5",
    "VOL10",
    "VOL20",

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

    # Sector
    "SECTOR_RET1",
    "SECTOR_RET5",
    "RELATIVE_SECTOR_RET1",

    # Regime
    "ABOVE_SMA20",
    "ABOVE_SMA50",
]


BANK_STOCKS = {
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "AXISBANK",
    "KOTAKBANK",
    "INDUSINDBK",
    "BANKBARODA",
    "PNB",
    "FEDERALBNK",
    "IDFCFIRSTB",
}

IT_STOCKS = {
    "TCS",
    "INFY",
    "HCLTECH",
    "WIPRO",
    "TECHM",
    "LTIM",
    "MPHASIS",
    "COFORGE",
    "PERSISTENT",
}


# ============================================================
# SYMBOL / FILE HELPERS
# ============================================================

def normalize_symbol(symbol: str) -> str:
    clean = str(symbol or "").strip().upper().replace(" ", "")

    if not clean:
        raise ValueError("Stock symbol is required.")

    if clean.startswith("^") or "." in clean:
        return clean

    return f"{clean}.NS"


def safe_name(symbol: str) -> str:
    return (
        normalize_symbol(symbol)
        .replace("^", "IDX_")
        .replace(".", "_")
        .replace("-", "_")
    )


def artifact_paths(symbol: str) -> dict[str, Path]:
    name = safe_name(symbol)

    return {
        "hist": MODEL_DIR / f"{name}_{MODEL_VERSION}_hist.pkl",
        "gbr": MODEL_DIR / f"{name}_{MODEL_VERSION}_gbr.pkl",
        "ridge": MODEL_DIR / f"{name}_{MODEL_VERSION}_ridge.pkl",
        "metadata": MODEL_DIR / f"{name}_{MODEL_VERSION}_metadata.json",
    }


def artifacts_exist(symbol: str) -> bool:
    paths = artifact_paths(symbol)

    return all(
        paths[key].exists()
        for key in ("hist", "gbr", "ridge", "metadata")
    )


# ============================================================
# DATA
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


def download_daily(
    symbol: str,
    period: str = "8y",
) -> pd.DataFrame:
    """
    auto_adjust=True makes old history split/dividend adjusted while the latest
    adjusted Close normally equals the current market Close. This avoids
    corporate-action jumps corrupting the residual target.
    """
    frame = yf.download(
        normalize_symbol(symbol),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    frame = clean_yfinance_frame(frame)

    if len(frame) < 500:
        raise ValueError(
            f"Not enough daily data for {symbol}. "
            f"Only {len(frame)} rows returned."
        )

    return frame


def context_close(
    symbol: str,
    index: pd.Index,
    period: str,
) -> pd.Series:
    try:
        frame = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        frame = clean_yfinance_frame(frame)

        return (
            frame["Close"]
            .astype(float)
            .reindex(index)
            .ffill()
        )

    except Exception:
        return pd.Series(
            np.nan,
            index=index,
            dtype=float,
        )


def sector_proxy_for(symbol: str) -> str:
    short = normalize_symbol(symbol).replace(".NS", "")

    if short in BANK_STOCKS:
        return "^NSEBANK"

    if short in IT_STOCKS:
        return "^CNXIT"

    # Unknown sectors use NIFTY as a broad-market proxy rather than inventing
    # a sector mapping.
    return "^NSEI"


# ============================================================
# FEATURES
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


def build_feature_frame(
    symbol: str,
    stock: pd.DataFrame,
    period: str = "8y",
) -> pd.DataFrame:
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
        (open_ / close.shift(1).replace(0, np.nan)) - 1.0
    ) * 100.0

    features["INTRADAY_PCT"] = (
        (close / open_.replace(0, np.nan)) - 1.0
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
    # RUPEE SCALE
    # --------------------------------------------------------

    atr14 = compute_atr(
        stock,
        14,
    )

    daily_rupee_move = close.diff().abs()

    features["PRICE_LOG"] = np.log(
        close
    )

    features["ATR14_RS"] = atr14

    features["ATR14_PCT"] = (
        atr14
        / close.replace(0, np.nan)
    ) * 100.0

    features["RANGE_RS"] = (
        high - low
    )

    features["AVG_RANGE5_RS"] = (
        (high - low)
        .rolling(5)
        .mean()
    )

    features["AVG_RANGE20_RS"] = (
        (high - low)
        .rolling(20)
        .mean()
    )

    features["ABS_MOVE5_RS"] = (
        daily_rupee_move
        .rolling(5)
        .mean()
    )

    features["ABS_MOVE20_RS"] = (
        daily_rupee_move
        .rolling(20)
        .mean()
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    sma = {}
    ema = {}

    for window in (5, 10, 20, 50):
        sma[window] = close.rolling(
            window
        ).mean()

        features[
            f"SMA{window}_DIST"
        ] = (
            (close / sma[window].replace(0, np.nan)) - 1.0
        ) * 100.0

        ema[window] = close.ewm(
            span=window,
            adjust=False,
        ).mean()

        features[
            f"EMA{window}_DIST"
        ] = (
            (close / ema[window].replace(0, np.nan)) - 1.0
        ) * 100.0

    features["EMA5_20_SPREAD"] = (
        (ema[5] / ema[20].replace(0, np.nan)) - 1.0
    ) * 100.0

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    features["RSI14"] = compute_rsi(
        close,
        14,
    )

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    macd = ema12 - ema26

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
    ).mean()

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
    # VOLATILITY
    # --------------------------------------------------------

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
    # VOLUME
    # --------------------------------------------------------

    log_volume = np.log1p(
        volume
    )

    features["VOLUME_LOG_CHANGE"] = (
        log_volume.diff()
    )

    volume_mean20 = (
        volume
        .rolling(20)
        .mean()
    )

    volume_std20 = (
        volume
        .rolling(20)
        .std()
    )

    features["VOLUME_Z20"] = (
        (volume - volume_mean20)
        / volume_std20.replace(0, np.nan)
    )

    features["VOLUME_RATIO5_20"] = (
        volume.rolling(5).mean()
        / volume_mean20.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # NIFTY CONTEXT
    # --------------------------------------------------------

    nifty_close = context_close(
        "^NSEI",
        stock.index,
        period,
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

    sector_symbol = sector_proxy_for(
        symbol
    )

    if sector_symbol == "^NSEI":
        sector_close = nifty_close
    else:
        sector_close = context_close(
            sector_symbol,
            stock.index,
            period,
        )

        if sector_close.notna().sum() < 100:
            sector_close = nifty_close

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
    # ROBUST CLIPPING
    # --------------------------------------------------------

    percentage_columns = [
        column
        for column in FEATURES
        if (
            "RET" in column
            or "DIST" in column
            or "PCT" in column
            or "ROC" in column
            or "VOL" in column
            or "SPREAD" in column
        )
        and column not in {
            "VOLUME_Z20",
            "VOLUME_RATIO5_20",
        }
    ]

    for column in percentage_columns:
        features[column] = (
            features[column]
            .clip(-60, 60)
        )

    features["RSI14"] = (
        features["RSI14"]
        .clip(0, 100)
    )

    features["CLOSE_LOCATION"] = (
        features["CLOSE_LOCATION"]
        .clip(0, 1)
    )

    features["VOLUME_Z20"] = (
        features["VOLUME_Z20"]
        .clip(-8, 8)
    )

    features["VOLUME_RATIO5_20"] = (
        features["VOLUME_RATIO5_20"]
        .clip(0, 8)
    )

    return features[FEATURES]


# ============================================================
# TARGET DATASET
# ============================================================

def build_dataset(
    symbol: str,
    period: str = "8y",
) -> pd.DataFrame:
    stock = download_daily(
        symbol,
        period,
    )

    features = build_feature_frame(
        symbol,
        stock,
        period,
    )

    close = stock["Close"].astype(float)

    next_close = close.shift(-1)

    residual_rs = (
        next_close - close
    )

    return_pct = (
        residual_rs
        / close.replace(0, np.nan)
    ) * 100.0

    dataset = features.copy()

    dataset["BASE_CLOSE"] = close

    dataset["TARGET_NEXT_CLOSE"] = (
        next_close
    )

    dataset["TARGET_RESIDUAL_RS"] = (
        residual_rs
    )

    dataset["TARGET_RETURN_PCT"] = (
        return_pct
    )

    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(dataset) < 600:
        raise ValueError(
            f"Only {len(dataset)} usable rows after feature engineering."
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
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=30,
        random_state=RANDOM_SEED,
    )


def make_gbr_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="huber",
        alpha=0.85,
        learning_rate=0.025,
        n_estimators=350,
        max_depth=2,
        min_samples_leaf=20,
        subsample=0.85,
        random_state=RANDOM_SEED,
    )


def make_ridge_model():
    return make_pipeline(
        StandardScaler(),
        Ridge(
            alpha=30.0,
            fit_intercept=True,
        ),
    )


def recency_weights(
    n: int,
) -> np.ndarray:
    if n <= 1:
        return np.ones(
            n,
            dtype=float,
        )

    return np.linspace(
        0.65,
        1.35,
        n,
        dtype=float,
    )


def fit_models(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    X = frame[FEATURES]

    y = frame[
        "TARGET_RESIDUAL_RS"
    ].to_numpy(dtype=float)

    # Clip only learner targets. Evaluation always uses real untouched residual.
    cap = max(
        10.0,
        float(
            np.percentile(
                np.abs(y),
                99,
            )
        ),
    )

    y_fit = np.clip(
        y,
        -cap,
        cap,
    )

    weights = recency_weights(
        len(frame)
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
        "training_target_cap_rs": cap,
    }


def component_predictions(
    models: dict[str, Any],
    X: pd.DataFrame,
) -> dict[str, np.ndarray]:
    return {
        "hist": np.asarray(
            models["hist"].predict(X),
            dtype=float,
        ),
        "gbr": np.asarray(
            models["gbr"].predict(X),
            dtype=float,
        ),
        "ridge": np.asarray(
            models["ridge"].predict(X),
            dtype=float,
        ),
    }


# ============================================================
# METRICS
# ============================================================

def residual_direction(
    residual_rs: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        residual_rs,
        dtype=float,
    )

    return np.where(
        values > DIRECTION_NEUTRAL_RS,
        1,
        np.where(
            values < -DIRECTION_NEUTRAL_RS,
            -1,
            0,
        ),
    )


def evaluate(
    y_true_residual: np.ndarray,
    y_pred_residual: np.ndarray,
    base_close: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(
        y_true_residual,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred_residual,
        dtype=float,
    )

    base = np.asarray(
        base_close,
        dtype=float,
    )

    actual_close = (
        base + y_true
    )

    predicted_close = (
        base + y_pred
    )

    absolute_error = np.abs(
        actual_close
        - predicted_close
    )

    return {
        "mae": float(
            np.mean(
                absolute_error
            )
        ),
        "median_absolute_error": float(
            np.median(
                absolute_error
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
                absolute_error
                / actual_close
            )
            * 100.0
        ),
        "direction_accuracy_percent": float(
            np.mean(
                residual_direction(y_true)
                == residual_direction(y_pred)
            )
            * 100.0
        ),
        "within_2_rupees_percent": float(
            np.mean(
                absolute_error <= 2.0
            )
            * 100.0
        ),
        "within_5_rupees_percent": float(
            np.mean(
                absolute_error <= 5.0
            )
            * 100.0
        ),
        "within_10_rupees_percent": float(
            np.mean(
                absolute_error <= 10.0
            )
            * 100.0
        ),
    }


# ============================================================
# VALIDATION-ONLY ENSEMBLE CALIBRATION
# ============================================================

def apply_calibration(
    components: dict[str, np.ndarray],
    calibration: dict[str, Any],
) -> np.ndarray:
    raw = (
        float(
            calibration["hist_weight"]
        )
        * components["hist"]
        +
        float(
            calibration["gbr_weight"]
        )
        * components["gbr"]
        +
        float(
            calibration["ridge_weight"]
        )
        * components["ridge"]
    )

    # baseline weight is implicitly residual Rs 0.
    prediction = (
        float(
            calibration["shrinkage"]
        )
        * raw
        +
        float(
            calibration["bias_rs"]
        )
    )

    cap = calibration.get(
        "residual_cap_rs"
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
    y_val = val_frame[
        "TARGET_RESIDUAL_RS"
    ].to_numpy(dtype=float)

    base_val = val_frame[
        "BASE_CLOSE"
    ].to_numpy(dtype=float)

    baseline_prediction = np.zeros_like(
        y_val
    )

    baseline_metrics = evaluate(
        y_val,
        baseline_prediction,
        base_val,
    )

    baseline_mae = baseline_metrics[
        "mae"
    ]

    train_residual = train_frame[
        "TARGET_RESIDUAL_RS"
    ].to_numpy(dtype=float)

    absolute_train = np.abs(
        train_residual
    )

    cap_candidates: list[
        float | None
    ] = [
        None,
        float(
            np.percentile(
                absolute_train,
                70,
            )
        ),
        float(
            np.percentile(
                absolute_train,
                80,
            )
        ),
        float(
            np.percentile(
                absolute_train,
                90,
            )
        ),
        float(
            np.percentile(
                absolute_train,
                95,
            )
        ),
    ]

    best = {
        "hist_weight": 0.0,
        "gbr_weight": 0.0,
        "ridge_weight": 0.0,
        "baseline_weight": 1.0,
        "shrinkage": 0.0,
        "bias_rs": 0.0,
        "residual_cap_rs": None,
        "validation_mae": float(
            baseline_mae
        ),
        "baseline_validation_mae": float(
            baseline_mae
        ),
        "validation_improvement_percent": 0.0,
        "accepted": False,
    }

    # Coarse ensemble search to avoid validation overfitting.
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

    for hist_weight in weights:
        for gbr_weight in weights:
            for ridge_weight in weights:
                weight_sum = (
                    hist_weight
                    + gbr_weight
                    + ridge_weight
                )

                if (
                    weight_sum <= 0.0
                    or weight_sum > 1.0001
                ):
                    continue

                raw = (
                    hist_weight
                    * components["hist"]
                    +
                    gbr_weight
                    * components["gbr"]
                    +
                    ridge_weight
                    * components["ridge"]
                )

                for shrinkage in shrinkages:
                    shrunk = (
                        shrinkage
                        * raw
                    )

                    # Robust residual bias from validation only.
                    bias = float(
                        np.clip(
                            np.median(
                                y_val
                                - shrunk
                            ),
                            -3.0,
                            3.0,
                        )
                    )

                    for cap in cap_candidates:
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
                            y_val,
                            candidate,
                            base_val,
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
                                if baseline_mae > 0
                                else 0.0
                            )

                            best = {
                                "hist_weight": float(
                                    hist_weight
                                ),
                                "gbr_weight": float(
                                    gbr_weight
                                ),
                                "ridge_weight": float(
                                    ridge_weight
                                ),
                                "baseline_weight": float(
                                    max(
                                        0.0,
                                        1.0
                                        - weight_sum,
                                    )
                                ),
                                "shrinkage": float(
                                    shrinkage
                                ),
                                "bias_rs": float(
                                    bias
                                ),
                                "residual_cap_rs": (
                                    float(cap)
                                    if cap is not None
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
        < MIN_VALIDATION_MAE_IMPROVEMENT_PCT
    ):
        return {
            "hist_weight": 0.0,
            "gbr_weight": 0.0,
            "ridge_weight": 0.0,
            "baseline_weight": 1.0,
            "shrinkage": 0.0,
            "bias_rs": 0.0,
            "residual_cap_rs": None,
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


def calibration_error_band(
    val_frame: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, float]:
    """
    Empirical validation residual interval. This is not a guarantee; it is
    only an observed validation error band.
    """
    actual_residual = val_frame[
        "TARGET_RESIDUAL_RS"
    ].to_numpy(dtype=float)

    error = (
        actual_residual
        - prediction
    )

    return {
        "error_q10_rs": float(
            np.quantile(
                error,
                0.10,
            )
        ),
        "error_q90_rs": float(
            np.quantile(
                error,
                0.90,
            )
        ),
    }


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    n = len(dataset)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n
        * (
            TRAIN_RATIO
            + VAL_RATIO
        )
    )

    train = dataset.iloc[
        :train_end
    ].copy()

    val = dataset.iloc[
        train_end:val_end
    ].copy()

    test = dataset.iloc[
        val_end:
    ].copy()

    if min(
        len(train),
        len(val),
        len(test),
    ) < 100:
        raise ValueError(
            "Not enough rows for chronological train/validation/test split."
        )

    return train, val, test


# ============================================================
# WALK-FORWARD EVALUATION
# ============================================================

def walk_forward_evaluate(
    dataset: pd.DataFrame,
    folds: int = WALK_FORWARD_FOLDS,
) -> dict[str, Any]:
    n = len(dataset)

    initial_train_end = int(
        n
        * WALK_FORWARD_INITIAL_TRAIN_RATIO
    )

    remaining = (
        n
        - initial_train_end
    )

    fold_size = max(
        80,
        remaining // folds,
    )

    fold_results = []

    all_true = []
    all_pred = []
    all_base = []

    for fold_index in range(
        folds
    ):
        test_start = (
            initial_train_end
            + fold_index
            * fold_size
        )

        test_end = (
            n
            if fold_index
            == folds - 1
            else min(
                n,
                test_start
                + fold_size,
            )
        )

        if (
            test_end
            - test_start
            < 50
        ):
            continue

        development = dataset.iloc[
            :test_start
        ].copy()

        test = dataset.iloc[
            test_start:test_end
        ].copy()

        val_size = max(
            80,
            int(
                len(development)
                * 0.15
            ),
        )

        train = development.iloc[
            :-val_size
        ].copy()

        val = development.iloc[
            -val_size:
        ].copy()

        if len(train) < 300:
            continue

        models = fit_models(
            train
        )

        val_components = (
            component_predictions(
                models,
                val[FEATURES],
            )
        )

        calibration = (
            optimize_calibration(
                train,
                val,
                val_components,
            )
        )

        test_components = (
            component_predictions(
                models,
                test[FEATURES],
            )
        )

        test_prediction = (
            apply_calibration(
                test_components,
                calibration,
            )
        )

        y_test = test[
            "TARGET_RESIDUAL_RS"
        ].to_numpy(dtype=float)

        base_test = test[
            "BASE_CLOSE"
        ].to_numpy(dtype=float)

        model_metrics = evaluate(
            y_test,
            test_prediction,
            base_test,
        )

        baseline_metrics = evaluate(
            y_test,
            np.zeros_like(
                y_test
            ),
            base_test,
        )

        fold_results.append(
            {
                "fold": fold_index + 1,
                "test_start": str(
                    test.index[
                        0
                    ].date()
                ),
                "test_end": str(
                    test.index[
                        -1
                    ].date()
                ),
                "samples": int(
                    len(test)
                ),
                "model": model_metrics,
                "baseline": baseline_metrics,
                "beats_baseline": bool(
                    model_metrics["mae"]
                    < baseline_metrics["mae"]
                ),
                "calibration": calibration,
            }
        )

        all_true.append(
            y_test
        )

        all_pred.append(
            test_prediction
        )

        all_base.append(
            base_test
        )

    if not fold_results:
        raise ValueError(
            "No valid walk-forward folds were created."
        )

    y_all = np.concatenate(
        all_true
    )

    pred_all = np.concatenate(
        all_pred
    )

    base_all = np.concatenate(
        all_base
    )

    model_overall = evaluate(
        y_all,
        pred_all,
        base_all,
    )

    baseline_overall = evaluate(
        y_all,
        np.zeros_like(
            y_all
        ),
        base_all,
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
                y_all
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
# TRAIN FINAL MODEL
# ============================================================

def train_precision_v2_1(
    symbol: str,
) -> dict[str, Any]:
    symbol = normalize_symbol(
        symbol
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        " STOCKVISION PRECISION V2.1"
    )
    print(
        " Residual MAE Ensemble + Previous-Close Safety Gate"
    )
    print(
        "=" * 72
    )
    print(
        f"Symbol: {symbol}"
    )

    dataset = build_dataset(
        symbol,
        period="8y",
    )

    print(
        f"Rows: {len(dataset)} | "
        f"Features: {len(FEATURES)}"
    )

    print(
        "\nRunning expanding walk-forward evaluation..."
    )

    walk_forward = (
        walk_forward_evaluate(
            dataset
        )
    )

    train, val, test = (
        chronological_split(
            dataset
        )
    )

    models = fit_models(
        train
    )

    val_components = (
        component_predictions(
            models,
            val[FEATURES],
        )
    )

    calibration = (
        optimize_calibration(
            train,
            val,
            val_components,
        )
    )

    val_prediction = (
        apply_calibration(
            val_components,
            calibration,
        )
    )

    error_band = (
        calibration_error_band(
            val,
            val_prediction,
        )
    )

    test_components = (
        component_predictions(
            models,
            test[FEATURES],
        )
    )

    test_prediction = (
        apply_calibration(
            test_components,
            calibration,
        )
    )

    y_test = test[
        "TARGET_RESIDUAL_RS"
    ].to_numpy(dtype=float)

    base_test = test[
        "BASE_CLOSE"
    ].to_numpy(dtype=float)

    model_metrics = evaluate(
        y_test,
        test_prediction,
        base_test,
    )

    baseline_metrics = evaluate(
        y_test,
        np.zeros_like(
            y_test
        ),
        base_test,
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

    close_hit_rates_not_worse = (
        model_metrics[
            "within_5_rupees_percent"
        ]
        >= baseline_metrics[
            "within_5_rupees_percent"
        ]
        and
        model_metrics[
            "within_10_rupees_percent"
        ]
        >= baseline_metrics[
            "within_10_rupees_percent"
        ]
    )

    accepted = (
        calibration[
            "accepted"
        ]
        and holdout_beats
        and walk_forward_beats
        and enough_folds_beat
        and close_hit_rates_not_worse
    )

    production_status = (
        "ACCEPTED"
        if accepted
        else "EXPERIMENTAL"
    )

    # Refit production learners on train + validation only.
    development = pd.concat(
        [
            train,
            val,
        ]
    )

    production_models = (
        fit_models(
            development
        )
    )

    paths = artifact_paths(
        symbol
    )

    joblib.dump(
        production_models[
            "hist"
        ],
        paths[
            "hist"
        ],
    )

    joblib.dump(
        production_models[
            "gbr"
        ],
        paths[
            "gbr"
        ],
    )

    joblib.dump(
        production_models[
            "ridge"
        ],
        paths[
            "ridge"
        ],
    )

    metadata = {
        "symbol": symbol,
        "model_version": MODEL_VERSION,
        "model_name": (
            "Precision V2.1 Residual MAE Ensemble"
        ),
        "production_status": production_status,
        "feature_count": len(
            FEATURES
        ),
        "features": FEATURES,
        "sector_proxy": sector_proxy_for(
            symbol
        ),
        "direction_neutral_rupees": (
            DIRECTION_NEUTRAL_RS
        ),
        "calibration": calibration,
        "validation_error_band": error_band,
        "holdout_test": {
            "model": model_metrics,
            "baseline": baseline_metrics,
            "mae_improvement_vs_baseline_percent": float(
                test_improvement
            ),
            "beats_baseline": bool(
                holdout_beats
            ),
        },
        "walk_forward": walk_forward,
        "acceptance": {
            "validation_gate_accepted": bool(
                calibration[
                    "accepted"
                ]
            ),
            "holdout_beats_baseline": bool(
                holdout_beats
            ),
            "walk_forward_beats_baseline": bool(
                walk_forward_beats
            ),
            "at_least_3_of_4_folds_beat_baseline": bool(
                enough_folds_beat
            ),
            "within_5_and_10_not_worse_than_baseline": bool(
                close_hit_rates_not_worse
            ),
        },
        "date_ranges": {
            "train_start": str(
                train.index[
                    0
                ].date()
            ),
            "train_end": str(
                train.index[
                    -1
                ].date()
            ),
            "validation_start": str(
                val.index[
                    0
                ].date()
            ),
            "validation_end": str(
                val.index[
                    -1
                ].date()
            ),
            "test_start": str(
                test.index[
                    0
                ].date()
            ),
            "test_end": str(
                test.index[
                    -1
                ].date()
            ),
        },
        "notes": [
            "Target is next-close minus current-close in rupees.",
            "Historical data is auto-adjusted to reduce corporate-action distortion.",
            "All evaluation is chronological.",
            "Ensemble weights, shrinkage, bias and cap are chosen only on validation data.",
            "Previous-close baseline is residual Rs 0.",
            "Rs 1-2 error is measured, not guaranteed.",
        ],
    }

    paths[
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
        + "=" * 72
    )
    print(
        " HOLDOUT TEST"
    )
    print(
        "=" * 72
    )

    print(
        f"Precision V2.1 MAE:    Rs {model_metrics['mae']:.2f}"
    )
    print(
        f"Previous-close MAE:    Rs {baseline_metrics['mae']:.2f}"
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
        f"Within Rs 2:           {model_metrics['within_2_rupees_percent']:.2f}% "
        f"(baseline {baseline_metrics['within_2_rupees_percent']:.2f}%)"
    )
    print(
        f"Within Rs 5:           {model_metrics['within_5_rupees_percent']:.2f}% "
        f"(baseline {baseline_metrics['within_5_rupees_percent']:.2f}%)"
    )
    print(
        f"Within Rs 10:          {model_metrics['within_10_rupees_percent']:.2f}% "
        f"(baseline {baseline_metrics['within_10_rupees_percent']:.2f}%)"
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        " WALK-FORWARD"
    )
    print(
        "=" * 72
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
        f"Bias=Rs {calibration['bias_rs']:+.2f} | "
        f"Cap={calibration['residual_cap_rs']}"
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        f"PRODUCTION STATUS: {production_status}"
    )
    print(
        "=" * 72
    )

    if production_status != "ACCEPTED":
        print(
            "V2.1 is saved for analysis but should not replace the "
            "baseline as a validated production forecast yet."
        )

    return metadata


# ============================================================
# PREDICT
# ============================================================

def load_metadata(
    symbol: str,
) -> dict[str, Any]:
    return json.loads(
        artifact_paths(
            symbol
        )[
            "metadata"
        ].read_text(
            encoding="utf-8"
        )
    )


def load_models(
    symbol: str,
) -> dict[str, Any]:
    paths = artifact_paths(
        symbol
    )

    return {
        "hist": joblib.load(
            paths[
                "hist"
            ]
        ),
        "gbr": joblib.load(
            paths[
                "gbr"
            ]
        ),
        "ridge": joblib.load(
            paths[
                "ridge"
            ]
        ),
    }


def predict_precision_v2_1(
    symbol: str,
    auto_train: bool = True,
) -> dict[str, Any]:
    symbol = normalize_symbol(
        symbol
    )

    if not artifacts_exist(
        symbol
    ):
        if not auto_train:
            raise FileNotFoundError(
                f"Precision V2.1 is not trained for {symbol}."
            )

        train_precision_v2_1(
            symbol
        )

    metadata = load_metadata(
        symbol
    )

    models = load_models(
        symbol
    )

    stock = download_daily(
        symbol,
        period="2y",
    )

    feature_frame = (
        build_feature_frame(
            symbol,
            stock,
            period="2y",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if feature_frame.empty:
        raise ValueError(
            "No valid latest feature row."
        )

    latest_date = (
        feature_frame.index[
            -1
        ]
    )

    X_latest = feature_frame[
        FEATURES
    ].iloc[
        [-1]
    ]

    components = (
        component_predictions(
            models,
            X_latest,
        )
    )

    calibration = metadata[
        "calibration"
    ]

    predicted_residual = float(
        apply_calibration(
            components,
            calibration,
        )[0]
    )

    matching_index = stock.index[
        stock.index
        <= latest_date
    ]

    if len(
        matching_index
    ) == 0:
        raise ValueError(
            "Could not resolve latest market close."
        )

    current_close = float(
        stock.loc[
            matching_index[
                -1
            ],
            "Close",
        ]
    )

    predicted_close = (
        current_close
        + predicted_residual
    )

    predicted_return_percent = (
        predicted_residual
        / current_close
        * 100.0
    )

    signal = (
        "BULLISH"
        if predicted_residual
        > DIRECTION_NEUTRAL_RS
        else "BEARISH"
        if predicted_residual
        < -DIRECTION_NEUTRAL_RS
        else "NEUTRAL"
    )

    error_band = metadata.get(
        "validation_error_band",
        {},
    )

    lower = (
        predicted_close
        + float(
            error_band.get(
                "error_q10_rs",
                0.0,
            )
        )
    )

    upper = (
        predicted_close
        + float(
            error_band.get(
                "error_q90_rs",
                0.0,
            )
        )
    )

    holdout = metadata.get(
        "holdout_test",
        {},
    )

    walk_forward = metadata.get(
        "walk_forward",
        {},
    )

    return {
        "symbol": symbol,
        "latest_market_date": str(
            pd.Timestamp(
                latest_date
            ).date()
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
            predicted_residual,
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
        "raw_components_rupees": {
            "hist": round(
                float(
                    components[
                        "hist"
                    ][
                        0
                    ]
                ),
                4,
            ),
            "gradient_boosting": round(
                float(
                    components[
                        "gbr"
                    ][
                        0
                    ]
                ),
                4,
            ),
            "ridge": round(
                float(
                    components[
                        "ridge"
                    ][
                        0
                    ]
                ),
                4,
            ),
            "previous_close_baseline": 0.0,
        },
        "blend": calibration,
        "evaluation": {
            "holdout_test": holdout,
            "walk_forward_summary": {
                "fold_count": walk_forward.get(
                    "fold_count"
                ),
                "samples": walk_forward.get(
                    "samples"
                ),
                "folds_beating_baseline": walk_forward.get(
                    "folds_beating_baseline"
                ),
                "model": walk_forward.get(
                    "model"
                ),
                "baseline": walk_forward.get(
                    "baseline"
                ),
                "mae_improvement_vs_baseline_percent": walk_forward.get(
                    "mae_improvement_vs_baseline_percent"
                ),
            },
        },
        "precision_note": (
            "V2.1 predicts the next-close residual in rupees and is accepted "
            "only when validation, holdout and walk-forward evaluation support "
            "an improvement over the previous-close baseline. Rs 1-2 error is "
            "measured, not guaranteed."
        ),
    }
