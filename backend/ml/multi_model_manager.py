import os

from .data import get_historical_data

from .multi_train import (
    train_multi_horizon_model,
    get_multi_model_paths,
)

from .multi_predict import (
    predict_multi_horizon,
)


# =========================================================
# NORMALIZE SYMBOL
# =========================================================

def normalize_symbol(symbol: str):
    symbol = symbol.upper().strip()

    if not symbol:
        raise ValueError(
            "Stock symbol cannot be empty."
        )

    if (
        "." not in symbol
        and not symbol.startswith("^")
    ):
        symbol = f"{symbol}.NS"

    return symbol


# =========================================================
# CHECK IF MULTI-HORIZON MODEL EXISTS
# =========================================================

def multi_model_exists(symbol: str):
    symbol = normalize_symbol(symbol)

    (
        model_path,
        feature_scaler_path,
        target_scaler_path,
        calibration_path,
        metrics_path,
    ) = get_multi_model_paths(
        symbol
    )

    required_files = [
        model_path,
        feature_scaler_path,
        target_scaler_path,
        calibration_path,
        metrics_path,
    ]

    return all(
        os.path.exists(path)
        for path in required_files
    )


# =========================================================
# VALIDATE STOCK
# =========================================================

def validate_stock(symbol: str):
    symbol = normalize_symbol(symbol)

    try:
        print(
            f"\nChecking market data for {symbol}..."
        )

        data = get_historical_data(
            symbol,
            period="10y"
        )

    except Exception as e:
        raise ValueError(
            f"Unable to find market data for {symbol}. "
            f"{str(e)}"
        )

    if data is None or data.empty:
        raise ValueError(
            f"Invalid or unsupported stock: {symbol}"
        )

    print(
        f"Historical rows found: {len(data)}"
    )

    if len(data) < 500:
        raise ValueError(
            f"{symbol} does not have enough historical "
            f"data for multi-horizon BiLSTM training. "
            f"Found only {len(data)} rows."
        )

    return True


# =========================================================
# TRAIN MULTI-HORIZON MODEL
# =========================================================

def train_multi_model(symbol: str):
    symbol = normalize_symbol(symbol)

    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION AUTO MULTI-HORIZON TRAINING"
    )

    print(
        "=============================================="
    )

    print(
        f"\nStock: {symbol}"
    )

    print(
        "\nMulti-horizon model does not exist."
    )

    print(
        "Validating selected stock..."
    )

    validate_stock(
        symbol
    )

    print(
        "\nStock validation successful."
    )

    print(
        "\nPreparing automatic BiLSTM training..."
    )

    print(
        "\nForecast outputs:"
    )

    print(
        "1 Day"
    )

    print(
        "3 Days"
    )

    print(
        "5 Days"
    )

    print(
        "10 Days"
    )

    print(
        "\nStarting training...\n"
    )

    train_multi_horizon_model(
        symbol
    )

    # =====================================================
    # VERIFY FILES
    # =====================================================

    if not multi_model_exists(
        symbol
    ):
        raise RuntimeError(
            f"Training completed for {symbol}, "
            f"but one or more required model files "
            f"were not created."
        )

    print(
        "\n=============================================="
    )

    print(
        " AUTO TRAINING COMPLETE"
    )

    print(
        "=============================================="
    )

    print(
        f"\nModel ready for: {symbol}"
    )


# =========================================================
# PREDICT ANY STOCK
# =========================================================

def predict_any_stock_multi(symbol: str):
    """
    Main StockVision multi-horizon manager.

    Workflow:

    User selects stock
           |
           v
    Check saved model
           |
       +---+---+
       |       |
      YES      NO
       |       |
       |    Validate stock
       |       |
       |    Train BiLSTM
       |       |
       |    Save model
       |       |
       +---+---+
           |
           v
       Generate
    future forecast
    """

    symbol = normalize_symbol(
        symbol
    )

    trained_now = False

    # =====================================================
    # MODEL EXISTS
    # =====================================================

    if multi_model_exists(
        symbol
    ):

        print(
            "\n=============================================="
        )

        print(
            " STOCKVISION MULTI-HORIZON MODEL"
        )

        print(
            "=============================================="
        )

        print(
            f"\nStock: {symbol}"
        )

        print(
            "Existing model found."
        )

        print(
            "Loading saved model..."
        )

    # =====================================================
    # MODEL DOES NOT EXIST
    # =====================================================

    else:

        train_multi_model(
            symbol
        )

        trained_now = True

    # =====================================================
    # GENERATE PREDICTION
    # =====================================================

    result = predict_multi_horizon(
        symbol
    )

    # =====================================================
    # ADD MODEL STATUS
    # =====================================================

    result[
        "trained_now"
    ] = trained_now

    if trained_now:

        result[
            "model_status"
        ] = (
            "New multi-horizon model trained"
        )

    else:

        result[
            "model_status"
        ] = (
            "Existing multi-horizon model loaded"
        )

    return result