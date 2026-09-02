"""
StockVision Precision Next-Day Model
------------------------------------
Goal:
    Predict the NEXT trading-day close while explicitly protecting against
    large errors from an unstable neural-network forecast.

Key idea:
    1) Predict next-day RETURN, not raw price.
    2) Use a BiLSTM + Ridge ensemble.
    3) Keep the previous-close forecast as a zero-return safety baseline.
    4) Choose ensemble weights ONLY on chronological validation data.
    5) If the ensemble cannot beat the naive previous-close baseline on
       validation MAE, automatically fall back to the baseline.
    6) Report unseen test metrics honestly.

This design can reduce avoidable model error, but it cannot guarantee that a
stock forecast will be within Rs 1-2 of the real next close.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIG
# ============================================================

LOOKBACK = 40
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
MAX_EPOCHS = 70
BATCH_SIZE = 32
RANDOM_SEED = 42

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "DAILY_RETURN",
    "GAP_RETURN",
    "HIGH_LOW_RANGE",
    "OPEN_CLOSE_RETURN",
    "VOLUME_LOG_CHANGE",
    "SMA5_DISTANCE",
    "SMA10_DISTANCE",
    "SMA20_DISTANCE",
    "EMA10_DISTANCE",
    "EMA20_DISTANCE",
    "RSI14",
    "MACD_PERCENT",
    "MACD_SIGNAL_PERCENT",
    "VOLATILITY10",
    "VOLATILITY20",
    "MOMENTUM3",
    "MOMENTUM5",
    "MOMENTUM10",
    "ATR14_PERCENT",
    "NIFTY_RETURN",
    "NIFTY_MOMENTUM5",
    "RELATIVE_RETURN",
]

tf.keras.utils.set_random_seed(RANDOM_SEED)


# ============================================================
# PATHS
# ============================================================

def safe_name(symbol: str) -> str:
    return (
        symbol.upper()
        .replace("^", "IDX_")
        .replace(".", "_")
        .replace("-", "_")
    )


def artifact_paths(symbol: str) -> dict[str, Path]:
    name = safe_name(symbol)

    return {
        "model": MODEL_DIR / f"{name}_precision.keras",
        "scaler": MODEL_DIR / f"{name}_precision_feature_scaler.pkl",
        "ridge": MODEL_DIR / f"{name}_precision_ridge.pkl",
        "metadata": MODEL_DIR / f"{name}_precision_metadata.json",
    }


def artifacts_exist(symbol: str) -> bool:
    paths = artifact_paths(symbol)

    return all(
        paths[key].exists()
        for key in ("model", "scaler", "ridge", "metadata")
    )


# ============================================================
# DATA
# ============================================================

def normalize_symbol(symbol: str) -> str:
    clean = str(symbol or "").strip().upper().replace(" ", "")

    if not clean:
        raise ValueError("Stock symbol is required.")

    if clean.startswith("^") or "." in clean:
        return clean

    return f"{clean}.NS"


def clean_yfinance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    frame = frame.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        # yfinance may return either (field, ticker) or (ticker, field).
        first_level = set(frame.columns.get_level_values(0))

        if "Close" in first_level:
            frame.columns = frame.columns.get_level_values(0)
        else:
            frame.columns = frame.columns.get_level_values(-1)

    required = ["Open", "High", "Low", "Close", "Volume"]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(f"Missing market columns: {missing}")

    frame = frame[required].copy()

    for column in required:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    frame = frame[
        (frame["Close"] > 0)
        & (frame["High"] > 0)
        & (frame["Low"] > 0)
    ]

    frame = frame[~frame.index.duplicated(keep="last")]

    return frame.sort_index()


def download_history(
    symbol: str,
    period: str = "10y",
) -> pd.DataFrame:
    symbol = normalize_symbol(symbol)

    frame = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    frame = clean_yfinance_frame(frame)

    if len(frame) < 500:
        raise ValueError(
            f"Not enough daily history for {symbol}. "
            f"Only {len(frame)} rows were returned."
        )

    return frame


def download_nifty_context(
    stock_index: pd.Index,
) -> pd.Series:
    try:
        nifty = yf.download(
            "^NSEI",
            period="10y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        nifty = clean_yfinance_frame(nifty)

        close = nifty["Close"].astype(float)

        nifty_return = close.pct_change() * 100.0

        return nifty_return.reindex(
            stock_index
        ).ffill()

    except Exception:
        # Context is useful, but the stock model should still train if Yahoo
        # temporarily fails to return NIFTY data.
        return pd.Series(
            0.0,
            index=stock_index,
            dtype=float,
        )


# ============================================================
# FEATURES
# ============================================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
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

    return 100 - (100 / (1 + rs))


def compute_atr_percent(frame: pd.DataFrame, period: int = 14) -> pd.Series:
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

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return (atr / close.replace(0, np.nan)) * 100.0


def make_feature_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    close = frame["Close"].astype(float)
    open_ = frame["Open"].astype(float)
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    volume = frame["Volume"].fillna(0).astype(float)

    daily_return = close.pct_change() * 100.0

    features = pd.DataFrame(index=frame.index)

    features["DAILY_RETURN"] = daily_return

    features["GAP_RETURN"] = (
        (open_ / close.shift(1).replace(0, np.nan)) - 1.0
    ) * 100.0

    features["HIGH_LOW_RANGE"] = (
        (high / low.replace(0, np.nan)) - 1.0
    ) * 100.0

    features["OPEN_CLOSE_RETURN"] = (
        (close / open_.replace(0, np.nan)) - 1.0
    ) * 100.0

    features["VOLUME_LOG_CHANGE"] = (
        np.log1p(volume).diff()
    )

    for window in (5, 10, 20):
        sma = close.rolling(window).mean()

        features[f"SMA{window}_DISTANCE"] = (
            (close / sma.replace(0, np.nan)) - 1.0
        ) * 100.0

    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()

    features["EMA10_DISTANCE"] = (
        (close / ema10.replace(0, np.nan)) - 1.0
    ) * 100.0

    features["EMA20_DISTANCE"] = (
        (close / ema20.replace(0, np.nan)) - 1.0
    ) * 100.0

    features["RSI14"] = compute_rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    features["MACD_PERCENT"] = (
        macd / close.replace(0, np.nan)
    ) * 100.0

    features["MACD_SIGNAL_PERCENT"] = (
        macd_signal / close.replace(0, np.nan)
    ) * 100.0

    features["VOLATILITY10"] = daily_return.rolling(10).std()
    features["VOLATILITY20"] = daily_return.rolling(20).std()

    for window in (3, 5, 10):
        features[f"MOMENTUM{window}"] = (
            (close / close.shift(window).replace(0, np.nan)) - 1.0
        ) * 100.0

    features["ATR14_PERCENT"] = compute_atr_percent(frame, 14)

    nifty_return = download_nifty_context(frame.index)

    features["NIFTY_RETURN"] = nifty_return

    nifty_index_proxy = (
        1.0 + nifty_return.fillna(0.0) / 100.0
    ).cumprod()

    features["NIFTY_MOMENTUM5"] = (
        (nifty_index_proxy / nifty_index_proxy.shift(5)) - 1.0
    ) * 100.0

    features["RELATIVE_RETURN"] = (
        features["DAILY_RETURN"]
        - features["NIFTY_RETURN"]
    )

    # Winsorize extreme bad ticks / one-off data glitches.
    bounded = {
        "DAILY_RETURN": 20,
        "GAP_RETURN": 20,
        "HIGH_LOW_RANGE": 20,
        "OPEN_CLOSE_RETURN": 20,
        "VOLUME_LOG_CHANGE": 5,
        "SMA5_DISTANCE": 30,
        "SMA10_DISTANCE": 30,
        "SMA20_DISTANCE": 40,
        "EMA10_DISTANCE": 30,
        "EMA20_DISTANCE": 40,
        "MACD_PERCENT": 15,
        "MACD_SIGNAL_PERCENT": 15,
        "VOLATILITY10": 15,
        "VOLATILITY20": 15,
        "MOMENTUM3": 30,
        "MOMENTUM5": 40,
        "MOMENTUM10": 50,
        "ATR14_PERCENT": 20,
        "NIFTY_RETURN": 15,
        "NIFTY_MOMENTUM5": 30,
        "RELATIVE_RETURN": 20,
    }

    for column, bound in bounded.items():
        features[column] = features[column].clip(
            lower=-bound,
            upper=bound,
        )

    features["RSI14"] = features["RSI14"].clip(
        lower=0,
        upper=100,
    )

    return features[FEATURE_COLUMNS]


# ============================================================
# SUPERVISED DATASET
# ============================================================

def build_supervised_dataset(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    features = make_feature_frame(frame)

    close = frame["Close"].astype(float)

    target_return = (
        (close.shift(-1) / close.replace(0, np.nan)) - 1.0
    ) * 100.0

    dataset = features.copy()

    dataset["TARGET_NEXT_RETURN"] = target_return
    dataset["BASE_CLOSE"] = close

    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    return dataset


def create_sequences(
    dataset: pd.DataFrame,
    lookback: int = LOOKBACK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = dataset[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    targets = dataset["TARGET_NEXT_RETURN"].to_numpy(
        dtype=np.float32
    )

    base_close = dataset["BASE_CLOSE"].to_numpy(
        dtype=np.float64
    )

    dates = dataset.index.to_numpy()

    X = []
    y = []
    base = []
    sample_dates = []

    for index in range(lookback - 1, len(dataset)):
        X.append(
            values[
                index - lookback + 1:
                index + 1
            ]
        )

        y.append(targets[index])
        base.append(base_close[index])
        sample_dates.append(dates[index])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(base, dtype=np.float64),
        np.asarray(sample_dates),
    )


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    base_close: np.ndarray,
    dates: np.ndarray,
) -> dict[str, tuple[np.ndarray, ...]]:
    n = len(X)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    if train_end < 300 or val_end - train_end < 80 or n - val_end < 80:
        raise ValueError(
            "Insufficient samples after chronological split."
        )

    return {
        "train": (
            X[:train_end],
            y[:train_end],
            base_close[:train_end],
            dates[:train_end],
        ),
        "val": (
            X[train_end:val_end],
            y[train_end:val_end],
            base_close[train_end:val_end],
            dates[train_end:val_end],
        ),
        "test": (
            X[val_end:],
            y[val_end:],
            base_close[val_end:],
            dates[val_end:],
        ),
    }


def scale_sequences(
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    n, steps, features = X.shape

    flat = X.reshape(-1, features)

    scaled = scaler.transform(flat)

    return scaled.reshape(
        n,
        steps,
        features,
    ).astype(np.float32)


# ============================================================
# MODEL
# ============================================================

def build_bilstm(
    input_shape: tuple[int, int],
) -> keras.Model:
    inputs = keras.Input(
        shape=input_shape,
        name="market_sequence",
    )

    x = layers.Bidirectional(
        layers.LSTM(
            48,
            return_sequences=True,
            recurrent_dropout=0.0,
        )
    )(inputs)

    x = layers.Dropout(0.12)(x)

    x = layers.Bidirectional(
        layers.LSTM(
            24,
            return_sequences=False,
            recurrent_dropout=0.0,
        )
    )(x)

    x = layers.Dropout(0.10)(x)

    x = layers.Dense(
        24,
        activation="swish",
    )(x)

    x = layers.Dropout(0.08)(x)

    output = layers.Dense(
        1,
        activation="linear",
        name="next_day_return_pct",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name="StockVisionPrecisionBiLSTM",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=3e-4,
            clipnorm=1.0,
        ),
        loss=keras.losses.Huber(
            delta=0.75,
        ),
        metrics=[
            keras.metrics.MeanAbsoluteError(
                name="return_mae"
            )
        ],
    )

    return model


# ============================================================
# CALIBRATION / ENSEMBLE
# ============================================================

def price_mae(
    true_return_pct: np.ndarray,
    predicted_return_pct: np.ndarray,
    base_close: np.ndarray,
) -> float:
    actual_price = base_close * (
        1.0 + true_return_pct / 100.0
    )

    predicted_price = base_close * (
        1.0 + predicted_return_pct / 100.0
    )

    return float(
        np.mean(
            np.abs(
                actual_price - predicted_price
            )
        )
    )


def price_rmse(
    true_return_pct: np.ndarray,
    predicted_return_pct: np.ndarray,
    base_close: np.ndarray,
) -> float:
    actual_price = base_close * (
        1.0 + true_return_pct / 100.0
    )

    predicted_price = base_close * (
        1.0 + predicted_return_pct / 100.0
    )

    return float(
        np.sqrt(
            np.mean(
                np.square(
                    actual_price - predicted_price
                )
            )
        )
    )


def direction_accuracy(
    true_return_pct: np.ndarray,
    predicted_return_pct: np.ndarray,
    neutral_band: float = 0.10,
) -> float:
    true_direction = np.where(
        true_return_pct > neutral_band,
        1,
        np.where(
            true_return_pct < -neutral_band,
            -1,
            0,
        ),
    )

    pred_direction = np.where(
        predicted_return_pct > neutral_band,
        1,
        np.where(
            predicted_return_pct < -neutral_band,
            -1,
            0,
        ),
    )

    return float(
        np.mean(
            true_direction == pred_direction
        ) * 100.0
    )


def optimize_precision_blend(
    y_val: np.ndarray,
    base_val: np.ndarray,
    lstm_val: np.ndarray,
    ridge_val: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """
    Blend:
        final_return = w_lstm * LSTM + w_ridge * Ridge + baseline_weight * 0

    The previous-close baseline is return=0.

    We search weights only on validation data, then refuse to use an ensemble
    that does not beat the baseline there.
    """
    baseline_prediction = np.zeros_like(y_val)

    baseline_mae = price_mae(
        y_val,
        baseline_prediction,
        base_val,
    )

    abs_train = np.abs(y_train)

    cap_candidates = [
        None,
        float(np.percentile(abs_train, 60)),
        float(np.percentile(abs_train, 70)),
        float(np.percentile(abs_train, 80)),
        float(np.percentile(abs_train, 90)),
        float(np.percentile(abs_train, 95)),
    ]

    best = {
        "w_lstm": 0.0,
        "w_ridge": 0.0,
        "baseline_weight": 1.0,
        "bias": 0.0,
        "cap_pct": None,
        "validation_mae": baseline_mae,
        "baseline_validation_mae": baseline_mae,
    }

    # 0.05 resolution gives enough flexibility without fitting too finely.
    grid = np.arange(
        0.0,
        1.0001,
        0.05,
    )

    for w_lstm in grid:
        for w_ridge in grid:
            if w_lstm + w_ridge > 1.0001:
                continue

            blended = (
                w_lstm * lstm_val
                + w_ridge * ridge_val
            )

            # Median residual is robust to a few large market jumps.
            bias = float(
                np.median(
                    y_val - blended
                )
            )

            # Avoid allowing calibration bias alone to invent a large move.
            bias = float(
                np.clip(
                    bias,
                    -0.35,
                    0.35,
                )
            )

            calibrated = blended + bias

            for cap in cap_candidates:
                prediction = calibrated.copy()

                if cap is not None:
                    prediction = np.clip(
                        prediction,
                        -cap,
                        cap,
                    )

                mae = price_mae(
                    y_val,
                    prediction,
                    base_val,
                )

                if mae < best["validation_mae"]:
                    best = {
                        "w_lstm": float(w_lstm),
                        "w_ridge": float(w_ridge),
                        "baseline_weight": float(
                            1.0 - w_lstm - w_ridge
                        ),
                        "bias": bias,
                        "cap_pct": (
                            float(cap)
                            if cap is not None
                            else None
                        ),
                        "validation_mae": float(mae),
                        "baseline_validation_mae": float(
                            baseline_mae
                        ),
                    }

    # Safety rule:
    # If the learned signal does not improve validation MAE by at least 0.5%,
    # do not pretend it is useful. Use the robust previous-close baseline.
    if (
        best["validation_mae"]
        > baseline_mae * 0.995
    ):
        best = {
            "w_lstm": 0.0,
            "w_ridge": 0.0,
            "baseline_weight": 1.0,
            "bias": 0.0,
            "cap_pct": None,
            "validation_mae": baseline_mae,
            "baseline_validation_mae": baseline_mae,
        }

    best["validation_improvement_percent"] = float(
        (
            baseline_mae
            - best["validation_mae"]
        )
        / baseline_mae
        * 100.0
        if baseline_mae > 0
        else 0.0
    )

    return best


def apply_calibration(
    lstm_prediction: np.ndarray,
    ridge_prediction: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    prediction = (
        float(calibration["w_lstm"])
        * np.asarray(lstm_prediction, dtype=float)
        + float(calibration["w_ridge"])
        * np.asarray(ridge_prediction, dtype=float)
        + float(calibration.get("bias", 0.0))
    )

    cap = calibration.get("cap_pct")

    if cap is not None:
        prediction = np.clip(
            prediction,
            -float(cap),
            float(cap),
        )

    return prediction


# ============================================================
# TRAIN
# ============================================================

def train_precision_model(
    symbol: str,
    verbose: int = 1,
) -> dict[str, Any]:
    symbol = normalize_symbol(symbol)

    print("\n" + "=" * 62)
    print(" STOCKVISION PRECISION NEXT-DAY MODEL")
    print(" BiLSTM + Ridge + Previous-Close Safety Baseline")
    print("=" * 62)
    print(f"Symbol: {symbol}")

    frame = download_history(
        symbol,
        period="10y",
    )

    dataset = build_supervised_dataset(
        frame
    )

    X, y, base_close, dates = create_sequences(
        dataset,
        LOOKBACK,
    )

    split = chronological_split(
        X,
        y,
        base_close,
        dates,
    )

    X_train, y_train, base_train, dates_train = split["train"]
    X_val, y_val, base_val, dates_val = split["val"]
    X_test, y_test, base_test, dates_test = split["test"]

    print(
        f"Samples -> Train: {len(X_train)}, "
        f"Val: {len(X_val)}, Test: {len(X_test)}"
    )

    scaler = StandardScaler()

    scaler.fit(
        X_train.reshape(
            -1,
            X_train.shape[-1],
        )
    )

    X_train_s = scale_sequences(
        scaler,
        X_train,
    )

    X_val_s = scale_sequences(
        scaler,
        X_val,
    )

    X_test_s = scale_sequences(
        scaler,
        X_test,
    )

    # Winsorize only the TRAINING TARGET fed to the learner. Evaluation keeps
    # the real un-clipped return.
    train_target_cap = float(
        max(
            2.0,
            np.percentile(
                np.abs(y_train),
                99,
            ),
        )
    )

    y_train_fit = np.clip(
        y_train,
        -train_target_cap,
        train_target_cap,
    )

    model = build_bilstm(
        input_shape=(
            LOOKBACK,
            len(FEATURE_COLUMNS),
        )
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=1e-4,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=2e-5,
            verbose=1,
        ),
    ]

    print("\nTraining BiLSTM residual-return model...")

    model.fit(
        X_train_s,
        y_train_fit,
        validation_data=(
            X_val_s,
            y_val,
        ),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose,
    )

    # A simple regularized linear model provides a very different inductive
    # bias. We blend it with the BiLSTM only if validation supports it.
    ridge = Ridge(
        alpha=20.0,
        fit_intercept=True,
    )

    ridge.fit(
        X_train_s[:, -1, :],
        y_train_fit,
    )

    lstm_val = model.predict(
        X_val_s,
        verbose=0,
    ).reshape(-1)

    ridge_val = ridge.predict(
        X_val_s[:, -1, :]
    ).reshape(-1)

    calibration = optimize_precision_blend(
        y_val=y_val,
        base_val=base_val,
        lstm_val=lstm_val,
        ridge_val=ridge_val,
        y_train=y_train,
    )

    lstm_test = model.predict(
        X_test_s,
        verbose=0,
    ).reshape(-1)

    ridge_test = ridge.predict(
        X_test_s[:, -1, :]
    ).reshape(-1)

    final_test = apply_calibration(
        lstm_test,
        ridge_test,
        calibration,
    )

    baseline_test = np.zeros_like(
        y_test
    )

    test_mae = price_mae(
        y_test,
        final_test,
        base_test,
    )

    baseline_mae = price_mae(
        y_test,
        baseline_test,
        base_test,
    )

    test_rmse = price_rmse(
        y_test,
        final_test,
        base_test,
    )

    baseline_rmse = price_rmse(
        y_test,
        baseline_test,
        base_test,
    )

    direction_acc = direction_accuracy(
        y_test,
        final_test,
    )

    baseline_direction_acc = direction_accuracy(
        y_test,
        baseline_test,
    )

    mape = float(
        np.mean(
            np.abs(
                (
                    (
                        base_test
                        * (1 + y_test / 100.0)
                    )
                    -
                    (
                        base_test
                        * (1 + final_test / 100.0)
                    )
                )
                /
                (
                    base_test
                    * (1 + y_test / 100.0)
                )
            )
        )
        * 100.0
    )

    improvement = float(
        (
            baseline_mae - test_mae
        )
        / baseline_mae
        * 100.0
        if baseline_mae > 0
        else 0.0
    )

    test_status = (
        "BEATS_BASELINE"
        if test_mae < baseline_mae
        else "BELOW_BASELINE"
    )

    metadata = {
        "symbol": symbol,
        "model_version": "precision-v1",
        "model_name": (
            "Adaptive Precision BiLSTM + Ridge + Previous-Close Baseline"
        ),
        "lookback_days": LOOKBACK,
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "train_target_cap_percent": train_target_cap,
        "calibration": calibration,
        "date_ranges": {
            "train_start": str(
                pd.Timestamp(dates_train[0]).date()
            ),
            "train_end": str(
                pd.Timestamp(dates_train[-1]).date()
            ),
            "validation_start": str(
                pd.Timestamp(dates_val[0]).date()
            ),
            "validation_end": str(
                pd.Timestamp(dates_val[-1]).date()
            ),
            "test_start": str(
                pd.Timestamp(dates_test[0]).date()
            ),
            "test_end": str(
                pd.Timestamp(dates_test[-1]).date()
            ),
        },
        "test_metrics": {
            "price_mae": test_mae,
            "price_rmse": test_rmse,
            "price_mape_percent": mape,
            "direction_accuracy_percent": direction_acc,
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse,
            "baseline_direction_accuracy_percent": (
                baseline_direction_acc
            ),
            "mae_improvement_vs_baseline_percent": improvement,
            "status": test_status,
        },
        "notes": [
            "Target is next-day percentage return, not raw price.",
            "All train/validation/test splits are chronological.",
            "Feature scaler is fitted only on training data.",
            "Blend weights are selected only on validation data.",
            "The model automatically falls back to previous close if the learned signal does not improve validation MAE.",
            "No model can guarantee Rs 1-2 next-day price error.",
        ],
    }

    paths = artifact_paths(symbol)

    model.save(
        paths["model"]
    )

    joblib.dump(
        scaler,
        paths["scaler"],
    )

    joblib.dump(
        ridge,
        paths["ridge"],
    )

    paths["metadata"].write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 62)
    print(" UNSEEN TEST RESULTS")
    print("=" * 62)
    print(f"Precision model MAE:   Rs {test_mae:.2f}")
    print(f"Naive baseline MAE:    Rs {baseline_mae:.2f}")
    print(f"Precision model RMSE:  Rs {test_rmse:.2f}")
    print(f"MAPE:                  {mape:.3f}%")
    print(f"Direction accuracy:    {direction_acc:.2f}%")
    print(f"MAE improvement:       {improvement:+.2f}%")
    print(f"Status:                {test_status}")
    print("\nValidation-selected blend:")
    print(
        f"  BiLSTM={calibration['w_lstm']:.2f}, "
        f"Ridge={calibration['w_ridge']:.2f}, "
        f"PreviousClose={calibration['baseline_weight']:.2f}"
    )
    print(
        f"  Bias={calibration['bias']:+.4f}% | "
        f"Cap={calibration['cap_pct']}"
    )
    print("=" * 62)

    return metadata


# ============================================================
# PREDICT
# ============================================================

def load_metadata(symbol: str) -> dict[str, Any]:
    path = artifact_paths(symbol)["metadata"]

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def predict_precision(
    symbol: str,
    auto_train: bool = True,
) -> dict[str, Any]:
    symbol = normalize_symbol(symbol)

    if not artifacts_exist(symbol):
        if not auto_train:
            raise FileNotFoundError(
                f"Precision model not trained for {symbol}."
            )

        train_precision_model(
            symbol,
            verbose=0,
        )

    paths = artifact_paths(symbol)

    metadata = load_metadata(symbol)

    model = keras.models.load_model(
        paths["model"],
        compile=False,
    )

    scaler: StandardScaler = joblib.load(
        paths["scaler"]
    )

    ridge: Ridge = joblib.load(
        paths["ridge"]
    )

    # Enough recent data to construct indicators and the final 40-day sequence.
    frame = download_history(
        symbol,
        period="2y",
    )

    features = make_feature_frame(
        frame
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(features) < LOOKBACK:
        raise ValueError(
            f"Not enough valid feature rows for {symbol}."
        )

    latest_features = features[
        FEATURE_COLUMNS
    ].iloc[-LOOKBACK:].to_numpy(
        dtype=np.float32
    )

    X = latest_features[
        np.newaxis,
        :,
        :,
    ]

    X_scaled = scale_sequences(
        scaler,
        X,
    )

    lstm_return = float(
        model.predict(
            X_scaled,
            verbose=0,
        ).reshape(-1)[0]
    )

    ridge_return = float(
        ridge.predict(
            X_scaled[:, -1, :]
        ).reshape(-1)[0]
    )

    calibration = metadata["calibration"]

    final_return = float(
        apply_calibration(
            np.asarray([lstm_return]),
            np.asarray([ridge_return]),
            calibration,
        )[0]
    )

    # Use the close corresponding to the last feature row.
    latest_date = features.index[-1]

    matching_index = frame.index[
        frame.index <= latest_date
    ]

    if len(matching_index) == 0:
        raise ValueError(
            "Could not resolve latest completed close."
        )

    current_close = float(
        frame.loc[
            matching_index[-1],
            "Close",
        ]
    )

    predicted_price = current_close * (
        1.0 + final_return / 100.0
    )

    neutral_band = 0.10

    trend_signal = (
        "BULLISH"
        if final_return > neutral_band
        else "BEARISH"
        if final_return < -neutral_band
        else "NEUTRAL"
    )

    test_metrics = metadata.get(
        "test_metrics",
        {},
    )

    return {
        "symbol": symbol,
        "latest_market_date": str(
            pd.Timestamp(
                matching_index[-1]
            ).date()
        ),
        "current_close": round(
            current_close,
            4,
        ),
        "predicted_price": round(
            predicted_price,
            4,
        ),
        "predicted_return_percent": round(
            final_return,
            6,
        ),
        "trend_signal": trend_signal,
        "model": metadata.get(
            "model_name",
            "Adaptive Precision BiLSTM",
        ),
        "model_version": metadata.get(
            "model_version",
            "precision-v1",
        ),
        "raw_components": {
            "bilstm_return_percent": round(
                lstm_return,
                6,
            ),
            "ridge_return_percent": round(
                ridge_return,
                6,
            ),
            "previous_close_return_percent": 0.0,
        },
        "blend": {
            "bilstm_weight": calibration["w_lstm"],
            "ridge_weight": calibration["w_ridge"],
            "previous_close_weight": calibration[
                "baseline_weight"
            ],
            "bias_percent": calibration["bias"],
            "return_cap_percent": calibration["cap_pct"],
        },
        "evaluation": {
            "test_price_mae": test_metrics.get(
                "price_mae"
            ),
            "test_price_rmse": test_metrics.get(
                "price_rmse"
            ),
            "test_mape_percent": test_metrics.get(
                "price_mape_percent"
            ),
            "test_direction_accuracy_percent": test_metrics.get(
                "direction_accuracy_percent"
            ),
            "baseline_test_mae": test_metrics.get(
                "baseline_mae"
            ),
            "mae_improvement_vs_baseline_percent": test_metrics.get(
                "mae_improvement_vs_baseline_percent"
            ),
            "status": test_metrics.get(
                "status"
            ),
        },
        "precision_note": (
            "The model minimizes next-day price MAE using validation-selected "
            "BiLSTM/Ridge/baseline blending. Rs 1-2 accuracy is not guaranteed."
        ),
    }
