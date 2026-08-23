import numpy as np
import pandas as pd


# =========================================================
# STOCKVISION V5 CONFIG
# =========================================================

SEQUENCE_LENGTH = 60

HORIZONS = [
    1,
    3,
    5,
    10,
]


# =========================================================
# STOCK + MARKET CONTEXT FEATURES
# =========================================================

FEATURE_COLUMNS = [

    # -----------------------------------------
    # STOCK RETURNS
    # -----------------------------------------

    "RETURN_1",
    "RETURN_3",
    "RETURN_5",
    "RETURN_10",

    # -----------------------------------------
    # STOCK DAILY BEHAVIOUR
    # -----------------------------------------

    "HIGH_LOW_RANGE",
    "OPEN_CLOSE_RETURN",

    # -----------------------------------------
    # TREND
    # -----------------------------------------

    "SMA20_DISTANCE",
    "EMA20_DISTANCE",

    # -----------------------------------------
    # MOMENTUM / INDICATORS
    # -----------------------------------------

    "RSI14_NORMALIZED",
    "MACD_HIST_PERCENT",

    # -----------------------------------------
    # RISK / VOLATILITY
    # -----------------------------------------

    "VOLATILITY_10",
    "VOLATILITY_20",
    "ATR14_PERCENT",

    # -----------------------------------------
    # VOLUME
    # -----------------------------------------

    "VOLUME_ZSCORE_20",

    # -----------------------------------------
    # BOLLINGER POSITION
    # -----------------------------------------

    "BOLLINGER_POSITION",

    # =========================================
    # NEW: NIFTY MARKET CONTEXT
    # =========================================

    "NIFTY_RETURN_1",
    "NIFTY_RETURN_5",
    "NIFTY_SMA20_DISTANCE",
    "NIFTY_VOLATILITY_20",

    # =========================================
    # STOCK VS MARKET
    # =========================================

    "RELATIVE_RETURN_1",
    "RELATIVE_RETURN_5",
    "ROLLING_BETA_20",
]


RETURN_TARGET_COLUMNS = [
    "RETURN_1D",
    "RETURN_3D",
    "RETURN_5D",
    "RETURN_10D",
]


DIRECTION_TARGET_COLUMNS = [
    "DIRECTION_1D",
    "DIRECTION_3D",
    "DIRECTION_5D",
    "DIRECTION_10D",
]


# =========================================================
# RSI
# =========================================================

def calculate_rsi(
    close,
    period=14,
):

    delta = (
        close.diff()
    )


    gain = (
        delta.clip(
            lower=0
        )
    )


    loss = (
        -delta.clip(
            upper=0
        )
    )


    average_gain = (
        gain
        .rolling(
            period
        )
        .mean()
    )


    average_loss = (
        loss
        .rolling(
            period
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
            (
                1
                +
                rs
            )
        )
    )


    return rsi


# =========================================================
# ATR
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


    return (
        true_range
        .rolling(
            period
        )
        .mean()
    )


# =========================================================
# ALIGN STOCK + NIFTY
# =========================================================

def combine_stock_and_market(
    stock_data,
    nifty_data,
):

    stock = (
        stock_data
        .copy()
        .sort_index()
    )


    nifty = (
        nifty_data
        .copy()
        .sort_index()
    )


    # -----------------------------------------------------
    # Keep required columns
    # -----------------------------------------------------

    stock = stock[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ].copy()


    nifty = nifty[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ].copy()


    # -----------------------------------------------------
    # Rename NIFTY columns
    # -----------------------------------------------------

    nifty = nifty.rename(
        columns={
            "Open":
                "NIFTY_OPEN",

            "High":
                "NIFTY_HIGH",

            "Low":
                "NIFTY_LOW",

            "Close":
                "NIFTY_CLOSE",

            "Volume":
                "NIFTY_VOLUME",
        }
    )


    # -----------------------------------------------------
    # Inner join
    #
    # Only dates available for both stock and market
    # -----------------------------------------------------

    combined = (
        stock.join(
            nifty,
            how="inner",
        )
    )


    return combined


# =========================================================
# ADD STOCK FEATURES
# =========================================================

def add_stock_features(
    df,
):

    data = (
        df.copy()
    )


    # =====================================================
    # RETURNS
    # =====================================================

    data["RETURN_1"] = (
        data["Close"]
        .pct_change(1)
    )


    data["RETURN_3"] = (
        data["Close"]
        .pct_change(3)
    )


    data["RETURN_5"] = (
        data["Close"]
        .pct_change(5)
    )


    data["RETURN_10"] = (
        data["Close"]
        .pct_change(10)
    )


    # =====================================================
    # DAILY RANGE
    # =====================================================

    data["HIGH_LOW_RANGE"] = (
        (
            data["High"]
            -
            data["Low"]
        )
        /
        data["Close"]
    )


    data["OPEN_CLOSE_RETURN"] = (
        (
            data["Close"]
            -
            data["Open"]
        )
        /
        data["Open"]
    )


    # =====================================================
    # SMA / EMA
    # =====================================================

    sma20 = (
        data["Close"]
        .rolling(
            20
        )
        .mean()
    )


    ema20 = (
        data["Close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )


    data["SMA20_DISTANCE"] = (
        data["Close"]
        /
        sma20
        -
        1
    )


    data["EMA20_DISTANCE"] = (
        data["Close"]
        /
        ema20
        -
        1
    )


    # =====================================================
    # RSI
    # =====================================================

    rsi = (
        calculate_rsi(
            data["Close"],
            14,
        )
    )


    data["RSI14_NORMALIZED"] = (
        (
            rsi
            -
            50
        )
        /
        50
    )


    # =====================================================
    # MACD HISTOGRAM
    # =====================================================

    ema12 = (
        data["Close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )


    ema26 = (
        data["Close"]
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


    data["MACD_HIST_PERCENT"] = (
        macd_hist
        /
        data["Close"]
    )


    # =====================================================
    # VOLATILITY
    # =====================================================

    daily_return = (
        data["Close"]
        .pct_change()
    )


    data["VOLATILITY_10"] = (
        daily_return
        .rolling(
            10
        )
        .std()
    )


    data["VOLATILITY_20"] = (
        daily_return
        .rolling(
            20
        )
        .std()
    )


    # =====================================================
    # ATR
    # =====================================================

    atr14 = (
        calculate_atr(
            data,
            14,
        )
    )


    data["ATR14_PERCENT"] = (
        atr14
        /
        data["Close"]
    )


    # =====================================================
    # VOLUME Z-SCORE
    # =====================================================

    volume_mean = (
        data["Volume"]
        .rolling(
            20
        )
        .mean()
    )


    volume_std = (
        data["Volume"]
        .rolling(
            20
        )
        .std()
    )


    data["VOLUME_ZSCORE_20"] = (
        (
            data["Volume"]
            -
            volume_mean
        )
        /
        volume_std
    )


    # =====================================================
    # BOLLINGER POSITION
    # =====================================================

    bollinger_std = (
        data["Close"]
        .rolling(
            20
        )
        .std()
    )


    upper_band = (
        sma20
        +
        2
        *
        bollinger_std
    )


    lower_band = (
        sma20
        -
        2
        *
        bollinger_std
    )


    band_width = (
        upper_band
        -
        lower_band
    )


    data["BOLLINGER_POSITION"] = (
        (
            data["Close"]
            -
            lower_band
        )
        /
        band_width
    )


    return data


# =========================================================
# ADD NIFTY MARKET CONTEXT
# =========================================================

def add_market_context(
    df,
):

    data = (
        df.copy()
    )


    # =====================================================
    # NIFTY RETURNS
    # =====================================================

    data["NIFTY_RETURN_1"] = (
        data["NIFTY_CLOSE"]
        .pct_change(1)
    )


    data["NIFTY_RETURN_5"] = (
        data["NIFTY_CLOSE"]
        .pct_change(5)
    )


    # =====================================================
    # NIFTY TREND
    # =====================================================

    nifty_sma20 = (
        data["NIFTY_CLOSE"]
        .rolling(
            20
        )
        .mean()
    )


    data[
        "NIFTY_SMA20_DISTANCE"
    ] = (
        data["NIFTY_CLOSE"]
        /
        nifty_sma20
        -
        1
    )


    # =====================================================
    # NIFTY VOLATILITY
    # =====================================================

    nifty_daily_return = (
        data["NIFTY_CLOSE"]
        .pct_change()
    )


    data[
        "NIFTY_VOLATILITY_20"
    ] = (
        nifty_daily_return
        .rolling(
            20
        )
        .std()
    )


    # =====================================================
    # STOCK RELATIVE STRENGTH
    # =====================================================

    data["RELATIVE_RETURN_1"] = (
        data["RETURN_1"]
        -
        data["NIFTY_RETURN_1"]
    )


    data["RELATIVE_RETURN_5"] = (
        data["RETURN_5"]
        -
        data["NIFTY_RETURN_5"]
    )


    # =====================================================
    # ROLLING BETA
    #
    # Beta =
    # Cov(stock, market)
    # ------------------
    # Var(market)
    # =====================================================

    stock_return = (
        data["RETURN_1"]
    )


    market_return = (
        data["NIFTY_RETURN_1"]
    )


    rolling_covariance = (
        stock_return
        .rolling(
            20
        )
        .cov(
            market_return
        )
    )


    rolling_market_variance = (
        market_return
        .rolling(
            20
        )
        .var()
    )


    data["ROLLING_BETA_20"] = (
        rolling_covariance
        /
        rolling_market_variance
    )


    return data


# =========================================================
# CLEAN FEATURES
# =========================================================

def clean_features(
    df,
):

    data = (
        df.copy()
    )


    data.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
        inplace=True,
    )


    data.dropna(
        subset=FEATURE_COLUMNS,
        inplace=True,
    )


    return data


# =========================================================
# CREATE RETURN + DIRECTION TARGETS
# =========================================================

def add_targets(
    df,
):

    data = (
        df.copy()
    )


    for horizon in HORIZONS:

        return_column = (
            f"RETURN_{horizon}D"
        )


        direction_column = (
            f"DIRECTION_{horizon}D"
        )


        future_return = (
            data["Close"]
            .shift(
                -horizon
            )
            /
            data["Close"]
            -
            1
        )


        data[
            return_column
        ] = (
            future_return
        )


        # -------------------------------------------------
        # Preserve NaN for rows without future target.
        # Do NOT accidentally convert missing future rows
        # into DOWN = 0.
        # -------------------------------------------------

        data[
            direction_column
        ] = (
            future_return
            .gt(
                0
            )
            .astype(
                float
            )
            .where(
                future_return.notna()
            )
        )


    return data


# =========================================================
# TRAINING FRAME
# =========================================================

def prepare_training_frame_v5(
    stock_data,
    nifty_data,
):

    data = (
        combine_stock_and_market(
            stock_data,
            nifty_data,
        )
    )


    data = (
        add_stock_features(
            data
        )
    )


    data = (
        add_market_context(
            data
        )
    )


    data = (
        clean_features(
            data
        )
    )


    data = (
        add_targets(
            data
        )
    )


    data.dropna(
        subset=(
            RETURN_TARGET_COLUMNS
            +
            DIRECTION_TARGET_COLUMNS
        ),
        inplace=True,
    )


    return data


# =========================================================
# PREDICTION FRAME
# =========================================================

def prepare_prediction_frame_v5(
    stock_data,
    nifty_data,
):

    data = (
        combine_stock_and_market(
            stock_data,
            nifty_data,
        )
    )


    data = (
        add_stock_features(
            data
        )
    )


    data = (
        add_market_context(
            data
        )
    )


    data = (
        clean_features(
            data
        )
    )


    return data


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


    stock_symbol = (
        "RELIANCE.NS"
    )


    market_symbol = (
        "^NSEI"
    )


    print(
        "\nDownloading stock data..."
    )


    stock_data = (
        get_historical_data(
            stock_symbol,
            period="10y",
        )
    )


    print(
        "Downloading NIFTY 50 data..."
    )


    nifty_data = (
        get_historical_data(
            market_symbol,
            period="10y",
        )
    )


    training_data = (
        prepare_training_frame_v5(
            stock_data,
            nifty_data,
        )
    )


    prediction_data = (
        prepare_prediction_frame_v5(
            stock_data,
            nifty_data,
        )
    )


    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V5 MARKET CONTEXT FEATURES"
    )

    print(
        "=============================================="
    )


    print(
        f"\nStock: {stock_symbol}"
    )


    print(
        f"Market: {market_symbol}"
    )


    print(
        "\nFeature Count:",
        len(
            FEATURE_COLUMNS
        )
    )


    print(
        "Sequence Length:",
        SEQUENCE_LENGTH
    )


    print(
        "Horizons:",
        HORIZONS
    )


    print(
        "\nTraining Rows:",
        len(
            training_data
        )
    )


    print(
        "Prediction Rows:",
        len(
            prediction_data
        )
    )


    print(
        "\nDate Range:"
    )


    print(
        prediction_data.index[
            0
        ].date(),
        "→",
        prediction_data.index[
            -1
        ].date(),
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
        "\nRETURN TARGETS:"
    )


    for target in RETURN_TARGET_COLUMNS:

        print(
            "-",
            target
        )


    print(
        "\nDIRECTION TARGETS:"
    )


    for target in DIRECTION_TARGET_COLUMNS:

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
        ]
        .tail(
            1
        )
    )


    print(
        "\nLatest Training Target Row:"
    )


    print(
        training_data[
            RETURN_TARGET_COLUMNS
            +
            DIRECTION_TARGET_COLUMNS
        ]
        .tail(
            1
        )
    )