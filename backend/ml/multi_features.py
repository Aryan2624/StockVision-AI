import numpy as np
import pandas as pd


# =========================================================
# MULTI-HORIZON CONFIGURATION
# =========================================================

SEQUENCE_LENGTH = 60

HORIZONS = [
    1,
    3,
    5,
    10,
]


FEATURE_COLUMNS = [
    "DAILY_RETURN",
    "HIGH_LOW_RANGE",
    "OPEN_CLOSE_RETURN",
    "VOLUME_CHANGE",
    "SMA20_DISTANCE",
    "EMA20_DISTANCE",
    "RSI14",
    "MACD_PERCENT",
    "MACD_SIGNAL_PERCENT",
    "VOLATILITY20",
    "MOMENTUM5",
    "MOMENTUM10",
]


TARGET_COLUMNS = [
    "RETURN_1D",
    "RETURN_3D",
    "RETURN_5D",
    "RETURN_10D",
]


# =========================================================
# RSI
# =========================================================

def calculate_rsi(
    close,
    period=14
):
    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = (
        -delta.clip(
            upper=0
        )
    )

    average_gain = (
        gain
        .rolling(
            window=period
        )
        .mean()
    )

    average_loss = (
        loss
        .rolling(
            window=period
        )
        .mean()
    )

    rs = (
        average_gain
        /
        average_loss
    )

    rsi = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    return rsi


# =========================================================
# MARKET FEATURES
# =========================================================

def add_market_features(
    data
):
    """
    Create the 12 StockVision features.

    IMPORTANT:
    This function DOES NOT create prediction targets.

    That means the newest available market row
    remains available for live prediction.
    """

    df = data.copy()


    # -----------------------------------------------------
    # Ensure numeric data
    # -----------------------------------------------------

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]


    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # =====================================================
    # 1. DAILY RETURN
    # =====================================================

    df["DAILY_RETURN"] = (
        df["Close"]
        .pct_change()
    )


    # =====================================================
    # 2. HIGH / LOW RANGE
    # =====================================================

    df["HIGH_LOW_RANGE"] = (
        (
            df["High"]
            -
            df["Low"]
        )
        /
        df["Close"]
    )


    # =====================================================
    # 3. OPEN / CLOSE RETURN
    # =====================================================

    df["OPEN_CLOSE_RETURN"] = (
        (
            df["Close"]
            -
            df["Open"]
        )
        /
        df["Open"]
    )


    # =====================================================
    # 4. VOLUME CHANGE
    # =====================================================

    df["VOLUME_CHANGE"] = (
        df["Volume"]
        .pct_change()
    )


    # =====================================================
    # 5. SMA 20
    # =====================================================

    df["SMA20"] = (
        df["Close"]
        .rolling(
            window=20
        )
        .mean()
    )


    df["SMA20_DISTANCE"] = (
        (
            df["Close"]
            /
            df["SMA20"]
        )
        -
        1
    )


    # =====================================================
    # 6. EMA 20
    # =====================================================

    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )


    df["EMA20_DISTANCE"] = (
        (
            df["Close"]
            /
            df["EMA20"]
        )
        -
        1
    )


    # =====================================================
    # 7. RSI 14
    # =====================================================

    df["RSI14"] = (
        calculate_rsi(
            df["Close"],
            period=14
        )
    )


    # =====================================================
    # 8 + 9. MACD
    # =====================================================

    ema12 = (
        df["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )


    ema26 = (
        df["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )


    macd = (
        ema12
        -
        ema26
    )


    macd_signal = (
        macd
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )


    df["MACD_PERCENT"] = (
        macd
        /
        df["Close"]
    )


    df["MACD_SIGNAL_PERCENT"] = (
        macd_signal
        /
        df["Close"]
    )


    # =====================================================
    # 10. VOLATILITY
    # =====================================================

    df["VOLATILITY20"] = (
        df["DAILY_RETURN"]
        .rolling(
            window=20
        )
        .std()
    )


    # =====================================================
    # 11. MOMENTUM 5
    # =====================================================

    df["MOMENTUM5"] = (
        (
            df["Close"]
            /
            df["Close"].shift(5)
        )
        -
        1
    )


    # =====================================================
    # 12. MOMENTUM 10
    # =====================================================

    df["MOMENTUM10"] = (
        (
            df["Close"]
            /
            df["Close"].shift(10)
        )
        -
        1
    )


    # =====================================================
    # CLEAN INVALID VALUES
    # =====================================================

    df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
        inplace=True
    )


    # Only remove rows where the actual
    # MODEL FEATURES are unavailable.
    #
    # We DO NOT drop the latest row because
    # of a future target.

    df.dropna(
        subset=FEATURE_COLUMNS,
        inplace=True
    )


    return df


# =========================================================
# CREATE FUTURE TARGETS
# =========================================================

def add_multi_horizon_targets(
    data
):
    """
    Create cumulative future return targets:

    RETURN_1D
    RETURN_3D
    RETURN_5D
    RETURN_10D

    Example:

    RETURN_5D =
        Close after 5 trading days
        --------------------------
        Current Close
        minus 1
    """

    df = data.copy()


    for horizon in HORIZONS:

        column_name = (
            f"RETURN_{horizon}D"
        )


        df[column_name] = (
            (
                df["Close"]
                .shift(
                    -horizon
                )
                /
                df["Close"]
            )
            -
            1
        )


    df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
        inplace=True
    )


    return df


# =========================================================
# TRAINING DATA
# =========================================================

def prepare_training_frame(
    raw_data
):
    """
    Used during model training.

    1. Create features.
    2. Create future targets.
    3. Remove rows whose future price is not yet known.
    """

    df = add_market_features(
        raw_data
    )


    df = add_multi_horizon_targets(
        df
    )


    df.dropna(
        subset=TARGET_COLUMNS,
        inplace=True
    )


    return df


# =========================================================
# LIVE PREDICTION DATA
# =========================================================

def prepare_prediction_frame(
    raw_data
):
    """
    Used when predicting today.

    Future targets are NOT created.

    This lets StockVision use the most recent
    available trading day.
    """

    return add_market_features(
        raw_data
    )


# =========================================================
# QUICK TEST
# =========================================================

if __name__ == "__main__":

    try:

        from .data import (
            get_historical_data
        )

    except ImportError:

        from data import (
            get_historical_data
        )


    symbol = "RELIANCE.NS"


    raw = get_historical_data(
        symbol,
        period="10y"
    )


    training_df = (
        prepare_training_frame(
            raw
        )
    )


    prediction_df = (
        prepare_prediction_frame(
            raw
        )
    )


    print(
        "\n======================================"
    )

    print(
        " STOCKVISION MULTI-HORIZON FEATURES"
    )

    print(
        "======================================"
    )


    print(
        f"\nSymbol: {symbol}"
    )


    print(
        f"Feature count: {len(FEATURE_COLUMNS)}"
    )


    print(
        f"Horizons: {HORIZONS}"
    )


    print(
        f"Sequence length: {SEQUENCE_LENGTH}"
    )


    print(
        f"\nTraining rows: {len(training_df)}"
    )


    print(
        f"Prediction rows: {len(prediction_df)}"
    )


    print(
        "\nFeatures:"
    )


    for feature in FEATURE_COLUMNS:
        print(
            "-",
            feature
        )


    print(
        "\nTargets:"
    )


    for target in TARGET_COLUMNS:
        print(
            "-",
            target
        )


    print(
        "\nLatest prediction feature row:"
    )


    print(
        prediction_df[
            FEATURE_COLUMNS
        ].tail(
            1
        )
    )


    print(
        "\nLatest training targets:"
    )


    print(
        training_df[
            TARGET_COLUMNS
        ].tail(
            1
        )
    )