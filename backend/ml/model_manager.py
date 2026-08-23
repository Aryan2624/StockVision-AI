import os
import subprocess
import sys

from .predict import (
    predict_next_day,
    get_model_paths,
)

from .data import get_historical_data


def normalize_symbol(symbol: str):
    """
    Convert a simple NSE symbol into Yahoo Finance format.

    RELIANCE -> RELIANCE.NS
    TCS      -> TCS.NS
    INFY.NS  -> INFY.NS
    """

    symbol = symbol.upper().strip()

    if not symbol:
        raise ValueError(
            "Stock symbol cannot be empty."
        )

    if "." not in symbol:
        symbol = f"{symbol}.NS"

    return symbol


def model_exists(symbol: str):
    """
    Check if the BiLSTM model and its
    two scalers already exist.
    """

    (
        model_path,
        feature_scaler_path,
        target_scaler_path,
    ) = get_model_paths(symbol)

    return (
        os.path.exists(model_path)
        and os.path.exists(feature_scaler_path)
        and os.path.exists(target_scaler_path)
    )


def validate_stock(symbol: str):
    """
    Check whether Yahoo Finance contains
    real historical data for this symbol.
    """

    try:
        data = get_historical_data(
            symbol,
            period="1y"
        )

        if data.empty:
            raise ValueError(
                f"No market data found for {symbol}"
            )

        return True

    except Exception as e:
        raise ValueError(
            f"Invalid or unsupported stock: {symbol}. "
            f"{str(e)}"
        )


def train_stock_model(symbol: str):
    """
    Train a new BiLSTM model using the same
    Python environment running StockVision.
    """

    ml_directory = os.path.dirname(
        os.path.abspath(__file__)
    )

    backend_directory = os.path.dirname(
        ml_directory
    )

    train_file = os.path.join(
        ml_directory,
        "train.py"
    )

    print("\n======================================")
    print(" STOCKVISION AUTO TRAINING")
    print("======================================")

    print(
        f"\nStock: {symbol}"
    )

    print(
        "No existing model found."
    )

    print(
        "Training new BiLSTM model...\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            train_file,
            symbol,
        ],
        cwd=backend_directory
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Training failed for {symbol}"
        )

    if not model_exists(symbol):
        raise RuntimeError(
            f"Training finished but model files "
            f"were not created for {symbol}"
        )

    print(
        f"\nModel successfully created for {symbol}"
    )


def predict_any_stock(symbol: str):
    """
    Complete StockVision prediction flow.

    1. Normalize ticker
    2. Check model
    3. Validate stock if model is missing
    4. Train model if required
    5. Generate BiLSTM prediction
    """

    symbol = normalize_symbol(
        symbol
    )

    trained_now = False

    if not model_exists(symbol):

        validate_stock(
            symbol
        )

        train_stock_model(
            symbol
        )

        trained_now = True

    prediction = predict_next_day(
        symbol
    )

    prediction[
        "trained_now"
    ] = trained_now

    prediction[
        "model_status"
    ] = (
        "New model trained"
        if trained_now
        else "Existing model loaded"
    )

    return prediction