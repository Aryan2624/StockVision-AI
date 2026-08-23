import numpy as np
import pandas as pd


# =========================================================
# STOCKVISION BiLSTM V4 CONFIG
# =========================================================

SEQUENCE_LENGTH = 60

HORIZONS = [
    1,
    3,
    5,
    10,
]


# =========================================================
# V4 FEATURES
# =========================================================

FEATURE_COLUMNS = [

    # Short-term returns
    "RETURN_1",
    "RETURN_2",
    "RETURN_3",
    "RETURN_5",
    "RETURN_10",

    # Intraday behaviour
    "HIGH_LOW_RANGE",
    "OPEN_CLOSE_RETURN",

    # Trend position
    "SMA10_DISTANCE",
    "SMA20_DISTANCE",
    "SMA50_DISTANCE",
    "EMA10_DISTANCE",
    "EMA20_DISTANCE",

    # Momentum
    "MOMENTUM_3",
    "MOMENTUM_5",
    "MOMENTUM_10",
    "MOMENTUM_20",

    # RSI
    "RSI14_NORMALIZED",

    # MACD
    "MACD_PERCENT",
    "MACD_SIGNAL_PERCENT",
    "MACD_HIST_PERCENT",

    # Volatility
    "VOLATILITY_5",
    "VOLATILITY_10",
    "VOLATILITY_20",

    # ATR
    "ATR14_PERCENT",

    # Bollinger
    "BOLLINGER_POSITION",
    "BOLLINGER_WIDTH",

    # Volume
    "VOLUME_CHANGE",
    "VOLUME_ZSCORE_20",

    # Candle structure
    "BODY_PERCENT",
    "UPPER_WICK_PERCENT",
    "LOWER_WICK_PERCENT",
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
    period=14,
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

    avg_gain = (
        gain
        .rolling(
            window=period
        )
        .mean()
    )

    avg_loss = (
        loss
        .rolling(
            window=period
        )
        .mean()
    )

    rs = (
        avg_gain
        /
        avg_loss
    )

    return (
        100
        -
        (
            100
            /
            (
                1
                +
                rs
            )
        )
    )


# =========================================================
# TRUE RANGE / ATR
# =========================================================

def calculate_atr(
    df,
    period=14,
):
    previous_close = (
        df["Close"]
        .shift(1)
    )

    high_low = (
        df["High"]
        -
        df["Low"]
    )

    high_previous = (
        df["High"]
        -
        previous_close
    ).abs()

    low_previous = (
        df["Low"]
        -
        previous_close
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_previous,
            low_previous,
        ],
        axis=1,
    ).max(
        axis=1
    )

    atr = (
        true_range
        .rolling(
            window=period
        )
        .mean()
    )

    return atr


# =========================================================
# ADD V4 FEATURES
# =========================================================

def add_market_features_v4(
    raw_data,
):
    df = raw_data.copy()


    # =====================================================
    # NUMERIC CONVERSION
    # =====================================================

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in required_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


    # =====================================================
    # RETURNS
    # =====================================================

    df["RETURN_1"] = (
        df["Close"]
        .pct_change(1)
    )

    df["RETURN_2"] = (
        df["Close"]
        .pct_change(2)
    )

    df["RETURN_3"] = (
        df["Close"]
        .pct_change(3)
    )

    df["RETURN_5"] = (
        df["Close"]
        .pct_change(5)
    )

    df["RETURN_10"] = (
        df["Close"]
        .pct_change(10)
    )


    # =====================================================
    # INTRADAY RANGE
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
    # MOVING AVERAGES
    # =====================================================

    sma10 = (
        df["Close"]
        .rolling(
            window=10
        )
        .mean()
    )

    sma20 = (
        df["Close"]
        .rolling(
            window=20
        )
        .mean()
    )

    sma50 = (
        df["Close"]
        .rolling(
            window=50
        )
        .mean()
    )


    df["SMA10_DISTANCE"] = (
        df["Close"]
        /
        sma10
        -
        1
    )

    df["SMA20_DISTANCE"] = (
        df["Close"]
        /
        sma20
        -
        1
    )

    df["SMA50_DISTANCE"] = (
        df["Close"]
        /
        sma50
        -
        1
    )


    # =====================================================
    # EMA
    # =====================================================

    ema10 = (
        df["Close"]
        .ewm(
            span=10,
            adjust=False,
        )
        .mean()
    )

    ema20 = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )


    df["EMA10_DISTANCE"] = (
        df["Close"]
        /
        ema10
        -
        1
    )

    df["EMA20_DISTANCE"] = (
        df["Close"]
        /
        ema20
        -
        1
    )


    # =====================================================
    # MOMENTUM
    # =====================================================

    df["MOMENTUM_3"] = (
        df["Close"]
        /
        df["Close"].shift(3)
        -
        1
    )

    df["MOMENTUM_5"] = (
        df["Close"]
        /
        df["Close"].shift(5)
        -
        1
    )

    df["MOMENTUM_10"] = (
        df["Close"]
        /
        df["Close"].shift(10)
        -
        1
    )

    df["MOMENTUM_20"] = (
        df["Close"]
        /
        df["Close"].shift(20)
        -
        1
    )


    # =====================================================
    # RSI
    # =====================================================

    rsi14 = calculate_rsi(
        df["Close"],
        period=14,
    )

    # Convert RSI from 0-100 to roughly -1 to +1

    df["RSI14_NORMALIZED"] = (
        (
            rsi14
            -
            50
        )
        /
        50
    )


    # =====================================================
    # MACD
    # =====================================================

    ema12 = (
        df["Close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        df["Close"]
        .ewm(
            span=26,
            adjust=False,
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
            adjust=False,
        )
        .mean()
    )

    macd_hist = (
        macd
        -
        macd_signal
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

    df["MACD_HIST_PERCENT"] = (
        macd_hist
        /
        df["Close"]
    )


    # =====================================================
    # VOLATILITY
    # =====================================================

    daily_return = (
        df["Close"]
        .pct_change()
    )


    df["VOLATILITY_5"] = (
        daily_return
        .rolling(
            window=5
        )
        .std()
    )

    df["VOLATILITY_10"] = (
        daily_return
        .rolling(
            window=10
        )
        .std()
    )

    df["VOLATILITY_20"] = (
        daily_return
        .rolling(
            window=20
        )
        .std()
    )


    # =====================================================
    # ATR
    # =====================================================

    atr14 = calculate_atr(
        df,
        period=14,
    )


    df["ATR14_PERCENT"] = (
        atr14
        /
        df["Close"]
    )


    # =====================================================
    # BOLLINGER BANDS
    # =====================================================

    bollinger_middle = (
        sma20
    )

    bollinger_std = (
        df["Close"]
        .rolling(
            window=20
        )
        .std()
    )

    bollinger_upper = (
        bollinger_middle
        +
        2
        *
        bollinger_std
    )

    bollinger_lower = (
        bollinger_middle
        -
        2
        *
        bollinger_std
    )


    bollinger_denominator = (
        bollinger_upper
        -
        bollinger_lower
    )


    df["BOLLINGER_POSITION"] = (
        (
            df["Close"]
            -
            bollinger_lower
        )
        /
        bollinger_denominator
    )


    df["BOLLINGER_WIDTH"] = (
        bollinger_denominator
        /
        bollinger_middle
    )


    # =====================================================
    # VOLUME
    # =====================================================

    df["VOLUME_CHANGE"] = (
        df["Volume"]
        .pct_change()
    )


    volume_mean20 = (
        df["Volume"]
        .rolling(
            window=20
        )
        .mean()
    )


    volume_std20 = (
        df["Volume"]
        .rolling(
            window=20
        )
        .std()
    )


    df["VOLUME_ZSCORE_20"] = (
        (
            df["Volume"]
            -
            volume_mean20
        )
        /
        volume_std20
    )


    # =====================================================
    # CANDLE STRUCTURE
    # =====================================================

    candle_range = (
        df["High"]
        -
        df["Low"]
    )


    candle_range_safe = (
        candle_range
        .replace(
            0,
            np.nan
        )
    )


    body = (
        df["Close"]
        -
        df["Open"]
    )


    upper_body = pd.concat(
        [
            df["Open"],
            df["Close"],
        ],
        axis=1,
    ).max(
        axis=1
    )


    lower_body = pd.concat(
        [
            df["Open"],
            df["Close"],
        ],
        axis=1,
    ).min(
        axis=1
    )


    upper_wick = (
        df["High"]
        -
        upper_body
    )


    lower_wick = (
        lower_body
        -
        df["Low"]
    )


    df["BODY_PERCENT"] = (
        body
        /
        candle_range_safe
    )


    df["UPPER_WICK_PERCENT"] = (
        upper_wick
        /
        candle_range_safe
    )


    df["LOWER_WICK_PERCENT"] = (
        lower_wick
        /
        candle_range_safe
    )


    # =====================================================
    # CLEANING
    # =====================================================

    df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
        inplace=True,
    )


    df.dropna(
        subset=FEATURE_COLUMNS,
        inplace=True,
    )


    return df


# =========================================================
# FUTURE TARGETS
# =========================================================

def add_targets_v4(
    data,
):
    df = data.copy()


    for horizon in HORIZONS:

        target_name = (
            f"RETURN_{horizon}D"
        )


        df[target_name] = (
            df["Close"]
            .shift(
                -horizon
            )
            /
            df["Close"]
            -
            1
        )


    df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
        inplace=True,
    )


    return df


# =========================================================
# TRAINING FRAME
# =========================================================

def prepare_training_frame_v4(
    raw_data,
):
    df = (
        add_market_features_v4(
            raw_data
        )
    )


    df = add_targets_v4(
        df
    )


    df.dropna(
        subset=TARGET_COLUMNS,
        inplace=True,
    )


    return df


# =========================================================
# LIVE PREDICTION FRAME
# =========================================================

def prepare_prediction_frame_v4(
    raw_data,
):
    return add_market_features_v4(
        raw_data
    )


# =========================================================
# TEST
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


    symbol = (
        "RELIANCE.NS"
    )


    raw_data = (
        get_historical_data(
            symbol,
            period="10y"
        )
    )


    training_data = (
        prepare_training_frame_v4(
            raw_data
        )
    )


    prediction_data = (
        prepare_prediction_frame_v4(
            raw_data
        )
    )


    print(
        "\n=========================================="
    )

    print(
        " STOCKVISION BiLSTM V4 FEATURES"
    )

    print(
        "=========================================="
    )


    print(
        f"\nSymbol: {symbol}"
    )


    print(
        f"Feature Count: {len(FEATURE_COLUMNS)}"
    )


    print(
        f"Horizons: {HORIZONS}"
    )


    print(
        f"Sequence Length: {SEQUENCE_LENGTH}"
    )


    print(
        f"\nTraining Rows: {len(training_data)}"
    )


    print(
        f"Prediction Rows: {len(prediction_data)}"
    )


    print(
        "\nFEATURES:"
    )


    for feature in FEATURE_COLUMNS:

        print(
            "-",
            feature
        )


    print(
        "\nTARGETS:"
    )


    for target in TARGET_COLUMNS:

        print(
            "-",
            target
        )


    print(
        "\nLatest Feature Row:"
    )


    print(
        prediction_data[
            FEATURE_COLUMNS
        ].tail(
            1
        )
    )