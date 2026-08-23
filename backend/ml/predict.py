import os
import sys

import joblib
import numpy as np
import tensorflow as tf


# =========================================================
# IMPORTS
# Works both from FastAPI package mode
# and direct command-line execution
# =========================================================

try:
    from .data import get_historical_data

    from .features import (
        add_features,
        FEATURE_COLUMNS,
        SEQUENCE_LENGTH,
    )

except ImportError:
    from data import get_historical_data

    from features import (
        add_features,
        FEATURE_COLUMNS,
        SEQUENCE_LENGTH,
    )


# =========================================================
# MODEL PATHS
# =========================================================

def get_model_paths(symbol):
    safe_symbol = (
        symbol
        .upper()
        .replace(".", "_")
        .replace("^", "")
    )

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    models_dir = os.path.join(
        base_dir,
        "models"
    )

    model_path = os.path.join(
        models_dir,
        f"{safe_symbol}.keras"
    )

    feature_scaler_path = os.path.join(
        models_dir,
        f"{safe_symbol}_feature_scaler.pkl"
    )

    target_scaler_path = os.path.join(
        models_dir,
        f"{safe_symbol}_target_scaler.pkl"
    )

    return (
        model_path,
        feature_scaler_path,
        target_scaler_path,
    )


# =========================================================
# PREDICT NEXT TRADING DAY
# =========================================================

def predict_next_day(symbol):
    symbol = symbol.upper().strip()

    (
        model_path,
        feature_scaler_path,
        target_scaler_path,
    ) = get_model_paths(symbol)

    # -----------------------------------------------------
    # Check trained files
    # -----------------------------------------------------

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found for {symbol}. "
            f"Train the model first."
        )

    if not os.path.exists(feature_scaler_path):
        raise FileNotFoundError(
            f"Feature scaler not found for {symbol}"
        )

    if not os.path.exists(target_scaler_path):
        raise FileNotFoundError(
            f"Target scaler not found for {symbol}"
        )

    # -----------------------------------------------------
    # Load BiLSTM and scalers
    # -----------------------------------------------------

    model = tf.keras.models.load_model(
        model_path
    )

    feature_scaler = joblib.load(
        feature_scaler_path
    )

    target_scaler = joblib.load(
        target_scaler_path
    )

    # -----------------------------------------------------
    # Fetch real historical data
    # -----------------------------------------------------

    raw_data = get_historical_data(
        symbol,
        period="1y"
    )

    data = add_features(
        raw_data
    )

    if len(data) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Not enough historical data for {symbol}. "
            f"Need at least {SEQUENCE_LENGTH} usable rows."
        )

    # -----------------------------------------------------
    # Latest 60-day feature sequence
    # -----------------------------------------------------

    latest_sequence = (
        data[FEATURE_COLUMNS]
        .iloc[-SEQUENCE_LENGTH:]
    )

    scaled_sequence = (
        feature_scaler.transform(
            latest_sequence
        )
    )

    X = np.array(
        [scaled_sequence],
        dtype=np.float32
    )

    # -----------------------------------------------------
    # Predict next-day return
    # -----------------------------------------------------

    predicted_scaled_return = model.predict(
        X,
        verbose=0
    )

    predicted_return = (
        target_scaler
        .inverse_transform(
            predicted_scaled_return
        )
        .flatten()[0]
    )

    predicted_return = float(
        predicted_return
    )

    # -----------------------------------------------------
    # Current daily close
    # -----------------------------------------------------

    current_close = float(
        data["Close"].iloc[-1]
    )

    # -----------------------------------------------------
    # Convert predicted return to predicted price
    # -----------------------------------------------------

    predicted_price = (
        current_close
        * (1 + predicted_return)
    )

    expected_change_percent = (
        predicted_return * 100
    )

    # -----------------------------------------------------
    # Raw direction
    # -----------------------------------------------------

    if predicted_return > 0:
        direction = "UP"

    elif predicted_return < 0:
        direction = "DOWN"

    else:
        direction = "FLAT"

    # -----------------------------------------------------
    # Better trend signal
    #
    # Tiny changes should not be shown as
    # strongly bullish or bearish.
    # -----------------------------------------------------

    if expected_change_percent >= 0.25:
        trend_signal = "BULLISH"

    elif expected_change_percent <= -0.25:
        trend_signal = "BEARISH"

    else:
        trend_signal = "NEUTRAL"

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    return {
        "symbol": symbol,

        "current_close": round(
            current_close,
            2
        ),

        "predicted_price": round(
            predicted_price,
            2
        ),

        "predicted_return_percent": round(
            expected_change_percent,
            4
        ),

        "direction": direction,

        "trend_signal": trend_signal,

        "model": "BiLSTM",

        "lookback_days": SEQUENCE_LENGTH,

        "features_used": len(
            FEATURE_COLUMNS
        ),
    }


# =========================================================
# COMMAND LINE TEST
# =========================================================

if __name__ == "__main__":
    symbol = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "RELIANCE.NS"
    )

    result = predict_next_day(
        symbol
    )

    print("\n====================================")
    print(" STOCKVISION BiLSTM PREDICTION")
    print("====================================")

    print(
        "\nSymbol:",
        result["symbol"]
    )

    print(
        "Current Close:",
        f'₹{result["current_close"]:.2f}'
    )

    print(
        "Predicted Next Close:",
        f'₹{result["predicted_price"]:.2f}'
    )

    print(
        "Expected Change:",
        f'{result["predicted_return_percent"]:.4f}%'
    )

    print(
        "Direction:",
        result["direction"]
    )

    print(
        "Trend Signal:",
        result["trend_signal"]
    )

    print(
        "Model:",
        result["model"]
    )

    print(
        "Lookback:",
        result["lookback_days"],
        "trading days"
    )

    print(
        "Features:",
        result["features_used"]
    )