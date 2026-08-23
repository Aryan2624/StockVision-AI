import os
import json
import time

import numpy as np
import pandas as pd


# =========================================================
# STOCKVISION IMPORT
# =========================================================

try:
    from .data import get_historical_data

except ImportError:
    from data import get_historical_data


# =========================================================
# V8 CONFIG
# =========================================================

TARGET_HORIZON = 5

# Dynamic target:
# threshold = ATR_FACTOR × current ATR%
ATR_FACTOR = 0.75


CLASS_NAMES = {
    0: "BEARISH",
    1: "NEUTRAL",
    2: "BULLISH",
}


# =========================================================
# STOCK + SECTOR MAP
# =========================================================

STOCK_SECTORS = {

    "RELIANCE.NS": "ENERGY",

    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "WIPRO.NS": "IT",
    "HCLTECH.NS": "IT",

    "HDFCBANK.NS": "BANKING",
    "ICICIBANK.NS": "BANKING",
    "AXISBANK.NS": "BANKING",
    "SBIN.NS": "BANKING",
    "KOTAKBANK.NS": "BANKING",

    "ITC.NS": "CONSUMER",

    "LT.NS": "INDUSTRIAL",

    "BHARTIARTL.NS": "TELECOM",

    "MARUTI.NS": "AUTO",

    "SUNPHARMA.NS": "PHARMA",
}


STOCKS = list(
    STOCK_SECTORS.keys()
)


MARKET_SYMBOL = "^NSEI"


# =========================================================
# FINAL V8 FEATURES
# =========================================================

FEATURE_COLUMNS = [

    # -----------------------------------------------------
    # STOCK RETURNS
    # -----------------------------------------------------

    "RETURN_1",
    "RETURN_3",
    "RETURN_5",
    "RETURN_10",
    "RETURN_20",

    # -----------------------------------------------------
    # CANDLE / RANGE
    # -----------------------------------------------------

    "HIGH_LOW_RANGE",
    "OPEN_CLOSE_RETURN",
    "GAP_RETURN",

    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    "SMA10_DISTANCE",
    "SMA20_DISTANCE",
    "SMA50_DISTANCE",

    "EMA10_DISTANCE",
    "EMA20_DISTANCE",

    "TREND_SLOPE_5",
    "TREND_SLOPE_20",

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    "RSI14_NORMALIZED",
    "MACD_PERCENT",
    "MACD_HIST_PERCENT",
    "MOMENTUM_ACCELERATION_5",

    # -----------------------------------------------------
    # VOLATILITY
    # -----------------------------------------------------

    "VOLATILITY_5",
    "VOLATILITY_10",
    "VOLATILITY_20",

    "VOLATILITY_RATIO_5_20",
    "DOWNSIDE_VOLATILITY_20",

    "ATR14_PERCENT",

    # -----------------------------------------------------
    # BOLLINGER
    # -----------------------------------------------------

    "BOLLINGER_POSITION",
    "BOLLINGER_WIDTH",

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    "VOLUME_CHANGE_1",
    "VOLUME_ZSCORE_20",
    "VOLUME_RATIO_20",

    # -----------------------------------------------------
    # NIFTY CONTEXT
    # -----------------------------------------------------

    "NIFTY_RETURN_1",
    "NIFTY_RETURN_5",
    "NIFTY_RETURN_20",

    "NIFTY_SMA20_DISTANCE",
    "NIFTY_SMA50_DISTANCE",

    "NIFTY_VOLATILITY_20",
    "NIFTY_TREND_SLOPE_20",

    # -----------------------------------------------------
    # STOCK VS NIFTY
    # -----------------------------------------------------

    "RELATIVE_RETURN_1",
    "RELATIVE_RETURN_5",
    "RELATIVE_RETURN_20",

    "ROLLING_BETA_20",
    "ROLLING_CORRELATION_20",
    "BETA_CHANGE_5",

    # -----------------------------------------------------
    # SECTOR CONTEXT
    # -----------------------------------------------------

    "SECTOR_RETURN_1",
    "SECTOR_RETURN_5",
    "SECTOR_VOLATILITY_20",
    "SECTOR_TREND_20",

    "RELATIVE_TO_SECTOR_1",
    "RELATIVE_TO_SECTOR_5",

    # -----------------------------------------------------
    # MARKET BREADTH
    # -----------------------------------------------------

    "BREADTH_POSITIVE_1D",
    "BREADTH_POSITIVE_5D",
    "BREADTH_ABOVE_SMA20",
]


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATASETS_DIR = os.path.join(
    BASE_DIR,
    "datasets",
)


os.makedirs(
    DATASETS_DIR,
    exist_ok=True,
)


CSV_PATH = os.path.join(
    DATASETS_DIR,
    "stockvision_v8_advanced_dataset.csv",
)


METADATA_PATH = os.path.join(
    DATASETS_DIR,
    "stockvision_v8_advanced_metadata.json",
)


# =========================================================
# NORMALIZE YAHOO DATA
# =========================================================

def normalize_data(
    data,
):

    df = data.copy()


    # Handle MultiIndex just in case
    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):

        df.columns = [
            column[0]
            if isinstance(column, tuple)
            else column

            for column in df.columns
        ]


    df.index = pd.to_datetime(
        df.index
    )


    # Remove timezone if present
    try:

        if df.index.tz is not None:

            df.index = (
                df.index
                .tz_localize(None)
            )

    except Exception:
        pass


    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]


    missing = [

        column

        for column in required

        if column not in df.columns
    ]


    if missing:

        raise ValueError(
            "Missing OHLCV columns: "
            +
            ", ".join(missing)
        )


    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


    df = df[
        required
    ]


    df.sort_index(
        inplace=True
    )


    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]


    return df


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


    average_gain = (
        gain
        .rolling(period)
        .mean()
    )


    average_loss = (
        loss
        .rolling(period)
        .mean()
    )


    rs = (
        average_gain
        /
        average_loss.replace(
            0,
            np.nan,
        )
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


    true_range = pd.concat(
        [

            df["High"]
            -
            df["Low"],


            (
                df["High"]
                -
                previous_close
            ).abs(),


            (
                df["Low"]
                -
                previous_close
            ).abs(),
        ],

        axis=1,
    ).max(
        axis=1
    )


    atr = (
        true_range
        .rolling(period)
        .mean()
    )


    return atr


# =========================================================
# MARKET FEATURES
# =========================================================

def build_market_features(
    nifty_data,
):

    market = normalize_data(
        nifty_data
    )


    close = market[
        "Close"
    ]


    market_features = pd.DataFrame(
        index=market.index
    )


    market_features[
        "NIFTY_RETURN_1"
    ] = (
        close.pct_change(1)
    )


    market_features[
        "NIFTY_RETURN_5"
    ] = (
        close.pct_change(5)
    )


    market_features[
        "NIFTY_RETURN_20"
    ] = (
        close.pct_change(20)
    )


    nifty_sma20 = (
        close
        .rolling(20)
        .mean()
    )


    nifty_sma50 = (
        close
        .rolling(50)
        .mean()
    )


    market_features[
        "NIFTY_SMA20_DISTANCE"
    ] = (
        close
        /
        nifty_sma20
        -
        1
    )


    market_features[
        "NIFTY_SMA50_DISTANCE"
    ] = (
        close
        /
        nifty_sma50
        -
        1
    )


    market_features[
        "NIFTY_VOLATILITY_20"
    ] = (

        market_features[
            "NIFTY_RETURN_1"
        ]

        .rolling(20)
        .std()
    )


    market_features[
        "NIFTY_TREND_SLOPE_20"
    ] = (
        nifty_sma20
        /
        nifty_sma20.shift(5)
        -
        1
    )


    return market_features


# =========================================================
# BUILD INDIVIDUAL STOCK FEATURES
# =========================================================

def build_stock_features(
    symbol,
    sector,
    stock_data,
    market_features,
):

    df = normalize_data(
        stock_data
    )


    close = df[
        "Close"
    ]


    open_price = df[
        "Open"
    ]


    high = df[
        "High"
    ]


    low = df[
        "Low"
    ]


    volume = df[
        "Volume"
    ]


    # =====================================================
    # RETURNS
    # =====================================================

    df[
        "RETURN_1"
    ] = (
        close.pct_change(1)
    )


    df[
        "RETURN_3"
    ] = (
        close.pct_change(3)
    )


    df[
        "RETURN_5"
    ] = (
        close.pct_change(5)
    )


    df[
        "RETURN_10"
    ] = (
        close.pct_change(10)
    )


    df[
        "RETURN_20"
    ] = (
        close.pct_change(20)
    )


    # =====================================================
    # CANDLE / RANGE FEATURES
    # =====================================================

    df[
        "HIGH_LOW_RANGE"
    ] = (
        high
        -
        low
    ) / close


    df[
        "OPEN_CLOSE_RETURN"
    ] = (
        close
        -
        open_price
    ) / open_price


    df[
        "GAP_RETURN"
    ] = (
        open_price
        /
        close.shift(1)
        -
        1
    )


    # =====================================================
    # MOVING AVERAGES
    # =====================================================

    sma10 = (
        close
        .rolling(10)
        .mean()
    )


    sma20 = (
        close
        .rolling(20)
        .mean()
    )


    sma50 = (
        close
        .rolling(50)
        .mean()
    )


    ema10 = (
        close
        .ewm(
            span=10,
            adjust=False,
        )
        .mean()
    )


    ema20 = (
        close
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )


    df[
        "SMA10_DISTANCE"
    ] = (
        close
        /
        sma10
        -
        1
    )


    df[
        "SMA20_DISTANCE"
    ] = (
        close
        /
        sma20
        -
        1
    )


    df[
        "SMA50_DISTANCE"
    ] = (
        close
        /
        sma50
        -
        1
    )


    df[
        "EMA10_DISTANCE"
    ] = (
        close
        /
        ema10
        -
        1
    )


    df[
        "EMA20_DISTANCE"
    ] = (
        close
        /
        ema20
        -
        1
    )


    # =====================================================
    # TREND SLOPE
    # =====================================================

    df[
        "TREND_SLOPE_5"
    ] = (
        sma10
        /
        sma10.shift(5)
        -
        1
    )


    df[
        "TREND_SLOPE_20"
    ] = (
        sma20
        /
        sma20.shift(5)
        -
        1
    )


    # =====================================================
    # RSI
    # =====================================================

    rsi14 = calculate_rsi(
        close,
        14,
    )


    df[
        "RSI14_NORMALIZED"
    ] = (
        rsi14
        -
        50
    ) / 50


    # =====================================================
    # MACD
    # =====================================================

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


    macd_histogram = (
        macd
        -
        macd_signal
    )


    df[
        "MACD_PERCENT"
    ] = (
        macd
        /
        close
    )


    df[
        "MACD_HIST_PERCENT"
    ] = (
        macd_histogram
        /
        close
    )


    # =====================================================
    # MOMENTUM ACCELERATION
    #
    # Current 5-day return minus previous
    # non-overlapping 5-day return.
    # =====================================================

    previous_5d_return = (
        close.shift(5)
        /
        close.shift(10)
        -
        1
    )


    df[
        "MOMENTUM_ACCELERATION_5"
    ] = (
        df[
            "RETURN_5"
        ]
        -
        previous_5d_return
    )


    # =====================================================
    # VOLATILITY
    # =====================================================

    df[
        "VOLATILITY_5"
    ] = (
        df[
            "RETURN_1"
        ]
        .rolling(5)
        .std()
    )


    df[
        "VOLATILITY_10"
    ] = (
        df[
            "RETURN_1"
        ]
        .rolling(10)
        .std()
    )


    df[
        "VOLATILITY_20"
    ] = (
        df[
            "RETURN_1"
        ]
        .rolling(20)
        .std()
    )


    df[
        "VOLATILITY_RATIO_5_20"
    ] = (
        df[
            "VOLATILITY_5"
        ]
        /
        df[
            "VOLATILITY_20"
        ]
    )


    downside_return = (

        df[
            "RETURN_1"
        ]

        .where(
            df[
                "RETURN_1"
            ]
            <
            0,

            0.0,
        )
    )


    df[
        "DOWNSIDE_VOLATILITY_20"
    ] = (
        downside_return
        .rolling(20)
        .std()
    )


    # =====================================================
    # ATR
    # =====================================================

    atr14 = calculate_atr(
        df,
        14,
    )


    df[
        "ATR14_PERCENT"
    ] = (
        atr14
        /
        close
    )


    # =====================================================
    # BOLLINGER BANDS
    # =====================================================

    rolling_std20 = (
        close
        .rolling(20)
        .std()
    )


    upper_band = (
        sma20
        +
        2
        *
        rolling_std20
    )


    lower_band = (
        sma20
        -
        2
        *
        rolling_std20
    )


    band_width = (
        upper_band
        -
        lower_band
    )


    df[
        "BOLLINGER_POSITION"
    ] = (
        close
        -
        lower_band
    ) / band_width


    df[
        "BOLLINGER_WIDTH"
    ] = (
        band_width
        /
        sma20
    )


    # =====================================================
    # VOLUME
    # =====================================================

    df[
        "VOLUME_CHANGE_1"
    ] = (
        volume.pct_change(1)
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


    df[
        "VOLUME_ZSCORE_20"
    ] = (
        volume
        -
        volume_mean20
    ) / volume_std20


    df[
        "VOLUME_RATIO_20"
    ] = (
        volume
        /
        volume_mean20
    )


    # =====================================================
    # ADD NIFTY CONTEXT
    # =====================================================

    df = df.join(
        market_features,
        how="left",
    )


    # =====================================================
    # STOCK RELATIVE TO NIFTY
    # =====================================================

    df[
        "RELATIVE_RETURN_1"
    ] = (
        df[
            "RETURN_1"
        ]
        -
        df[
            "NIFTY_RETURN_1"
        ]
    )


    df[
        "RELATIVE_RETURN_5"
    ] = (
        df[
            "RETURN_5"
        ]
        -
        df[
            "NIFTY_RETURN_5"
        ]
    )


    df[
        "RELATIVE_RETURN_20"
    ] = (
        df[
            "RETURN_20"
        ]
        -
        df[
            "NIFTY_RETURN_20"
        ]
    )


    # =====================================================
    # ROLLING BETA
    # =====================================================

    covariance = (

        df[
            "RETURN_1"
        ]

        .rolling(20)

        .cov(
            df[
                "NIFTY_RETURN_1"
            ]
        )
    )


    market_variance = (

        df[
            "NIFTY_RETURN_1"
        ]

        .rolling(20)

        .var()
    )


    df[
        "ROLLING_BETA_20"
    ] = (
        covariance
        /
        market_variance
    )


    # =====================================================
    # ROLLING CORRELATION
    # =====================================================

    df[
        "ROLLING_CORRELATION_20"
    ] = (

        df[
            "RETURN_1"
        ]

        .rolling(20)

        .corr(
            df[
                "NIFTY_RETURN_1"
            ]
        )
    )


    df[
        "BETA_CHANGE_5"
    ] = (
        df[
            "ROLLING_BETA_20"
        ]
        -
        df[
            "ROLLING_BETA_20"
        ]
        .shift(5)
    )


    # =====================================================
    # VOLATILITY-ADJUSTED V8 TARGET
    #
    # IMPORTANT:
    # threshold uses CURRENT ATR only.
    # No future information is used in threshold.
    # =====================================================

    future_return_5d = (
        close.shift(
            -TARGET_HORIZON
        )
        /
        close
        -
        1
    )


    dynamic_threshold = (
        ATR_FACTOR
        *
        df[
            "ATR14_PERCENT"
        ]
    )


    df[
        "TARGET_RETURN_5D"
    ] = (
        future_return_5d
    )


    df[
        "TARGET_THRESHOLD"
    ] = (
        dynamic_threshold
    )


    target_class = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )


    valid_target = (

        future_return_5d.notna()

        &

        dynamic_threshold.notna()
    )


    # BEARISH
    target_class.loc[
        valid_target
        &
        (
            future_return_5d
            <
            -dynamic_threshold
        )
    ] = 0


    # NEUTRAL
    target_class.loc[
        valid_target
        &
        (
            future_return_5d
            >=
            -dynamic_threshold
        )
        &
        (
            future_return_5d
            <=
            dynamic_threshold
        )
    ] = 1


    # BULLISH
    target_class.loc[
        valid_target
        &
        (
            future_return_5d
            >
            dynamic_threshold
        )
    ] = 2


    df[
        "TARGET_CLASS"
    ] = (
        target_class
    )


    # =====================================================
    # IDENTIFICATION
    # =====================================================

    df[
        "SYMBOL"
    ] = symbol


    df[
        "SECTOR"
    ] = sector


    df.index.name = "Date"


    return df


# =========================================================
# LEAVE-ONE-OUT SECTOR FEATURES
#
# We intentionally exclude the current stock
# from its sector average.
#
# This avoids:
#
# "RELIANCE sector return"
# simply becoming RELIANCE's own return.
# =========================================================

def add_sector_features(
    dataset,
):

    df = dataset.copy()


    sector_source_columns = {

        "RETURN_1":
            "SECTOR_RETURN_1",

        "RETURN_5":
            "SECTOR_RETURN_5",

        "VOLATILITY_20":
            "SECTOR_VOLATILITY_20",

        "SMA20_DISTANCE":
            "SECTOR_TREND_20",
    }


    for (
        source_column,
        destination_column,

    ) in sector_source_columns.items():


        grouped = df.groupby(
            [
                "Date",
                "SECTOR",
            ]
        )[
            source_column
        ]


        group_sum = (
            grouped
            .transform(
                "sum"
            )
        )


        group_count = (
            grouped
            .transform(
                "count"
            )
        )


        peer_value = (

            group_sum
            -
            df[
                source_column
            ]

        ) / (

            group_count
            -
            1
        )


        # =================================================
        # SINGLE-STOCK SECTORS
        #
        # If no peer stock exists, use corresponding
        # broad-market information instead.
        # =================================================

        if destination_column == "SECTOR_RETURN_1":

            fallback = df[
                "NIFTY_RETURN_1"
            ]


        elif destination_column == "SECTOR_RETURN_5":

            fallback = df[
                "NIFTY_RETURN_5"
            ]


        elif destination_column == "SECTOR_VOLATILITY_20":

            fallback = df[
                "NIFTY_VOLATILITY_20"
            ]


        else:

            fallback = df[
                "NIFTY_SMA20_DISTANCE"
            ]


        df[
            destination_column
        ] = (
            peer_value
            .where(
                group_count
                >
                1,

                fallback,
            )
        )


    # =====================================================
    # STOCK RELATIVE TO SECTOR
    # =====================================================

    df[
        "RELATIVE_TO_SECTOR_1"
    ] = (
        df[
            "RETURN_1"
        ]
        -
        df[
            "SECTOR_RETURN_1"
        ]
    )


    df[
        "RELATIVE_TO_SECTOR_5"
    ] = (
        df[
            "RETURN_5"
        ]
        -
        df[
            "SECTOR_RETURN_5"
        ]
    )


    return df


# =========================================================
# MARKET BREADTH
# =========================================================

def add_market_breadth(
    dataset,
):

    df = dataset.copy()


    positive_1d = (
        df[
            "RETURN_1"
        ]
        >
        0
    ).astype(float)


    positive_5d = (
        df[
            "RETURN_5"
        ]
        >
        0
    ).astype(float)


    above_sma20 = (
        df[
            "SMA20_DISTANCE"
        ]
        >
        0
    ).astype(float)


    df[
        "BREADTH_POSITIVE_1D"
    ] = (
        positive_1d
        .groupby(
            df[
                "Date"
            ]
        )
        .transform(
            "mean"
        )
    )


    df[
        "BREADTH_POSITIVE_5D"
    ] = (
        positive_5d
        .groupby(
            df[
                "Date"
            ]
        )
        .transform(
            "mean"
        )
    )


    df[
        "BREADTH_ABOVE_SMA20"
    ] = (
        above_sma20
        .groupby(
            df[
                "Date"
            ]
        )
        .transform(
            "mean"
        )
    )


    return df


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

def calculate_distribution(
    df,
):

    counts = (
        df[
            "TARGET_CLASS"
        ]
        .value_counts()
        .sort_index()
    )


    total = len(
        df
    )


    result = {}


    for class_id in range(
        3
    ):

        count = int(
            counts.get(
                class_id,
                0,
            )
        )


        percentage = (

            count
            /
            total
            *
            100

            if total > 0

            else 0
        )


        result[
            CLASS_NAMES[
                class_id
            ]
        ] = {

            "count":
                count,

            "percentage":
                round(
                    percentage,
                    2,
                ),
        }


    return result


# =========================================================
# BUILD V8 DATASET
# =========================================================

def build_v8_dataset():

    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V8 ADVANCED DATASET"
    )

    print(
        "=============================================="
    )


    print(
        "\nStocks:",
        len(
            STOCKS
        )
    )


    print(
        "Features:",
        len(
            FEATURE_COLUMNS
        )
    )


    print(
        "Target Horizon:",
        TARGET_HORIZON,
        "trading days",
    )


    print(
        "Target Type:"
    )


    print(
        f"Dynamic threshold = {ATR_FACTOR} × ATR14%"
    )


    # =====================================================
    # DOWNLOAD NIFTY
    # =====================================================

    print(
        "\nDownloading NIFTY 50..."
    )


    nifty_data = get_historical_data(
        MARKET_SYMBOL,
        period="10y",
    )


    market_features = (
        build_market_features(
            nifty_data
        )
    )


    all_frames = []

    failed_stocks = []


    # =====================================================
    # DOWNLOAD EACH STOCK
    # =====================================================

    for symbol in STOCKS:

        sector = STOCK_SECTORS[
            symbol
        ]


        print(
            f"\nDownloading {symbol} [{sector}]..."
        )


        try:

            stock_data = get_historical_data(
                symbol,
                period="10y",
            )


            stock_features = (
                build_stock_features(
                    symbol=
                        symbol,

                    sector=
                        sector,

                    stock_data=
                        stock_data,

                    market_features=
                        market_features,
                )
            )


            stock_features = (
                stock_features
                .reset_index()
            )


            print(
                "Raw feature rows:",
                len(
                    stock_features
                )
            )


            all_frames.append(
                stock_features
            )


        except Exception as error:

            print(
                f"FAILED: {error}"
            )


            failed_stocks.append(
                {
                    "symbol":
                        symbol,

                    "error":
                        str(
                            error
                        ),
                }
            )


        time.sleep(
            0.25
        )


    if not all_frames:

        raise RuntimeError(
            "No V8 stock data was created."
        )


    # =====================================================
    # COMBINE STOCKS
    # =====================================================

    dataset = pd.concat(
        all_frames,
        ignore_index=True,
    )


    dataset[
        "Date"
    ] = pd.to_datetime(
        dataset[
            "Date"
        ]
    )


    # =====================================================
    # SECTOR FEATURES
    # =====================================================

    print(
        "\nBuilding leave-one-out sector context..."
    )


    dataset = (
        add_sector_features(
            dataset
        )
    )


    # =====================================================
    # MARKET BREADTH
    # =====================================================

    print(
        "Building market breadth features..."
    )


    dataset = (
        add_market_breadth(
            dataset
        )
    )


    # =====================================================
    # CLEAN INF
    # =====================================================

    dataset.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
        inplace=True,
    )


    # =====================================================
    # KEEP VALID TARGET + ALL FEATURES
    # =====================================================

    required_columns = [

        *FEATURE_COLUMNS,

        "TARGET_RETURN_5D",
        "TARGET_THRESHOLD",
        "TARGET_CLASS",
    ]


    before_drop = len(
        dataset
    )


    dataset.dropna(
        subset=
            required_columns,
        inplace=True,
    )


    after_drop = len(
        dataset
    )


    dataset[
        "TARGET_CLASS"
    ] = (
        dataset[
            "TARGET_CLASS"
        ]
        .astype(
            int
        )
    )


    dataset[
        "TARGET_LABEL"
    ] = (
        dataset[
            "TARGET_CLASS"
        ]
        .map(
            CLASS_NAMES
        )
    )


    # =====================================================
    # SORT
    # =====================================================

    dataset.sort_values(
        by=[
            "Date",
            "SYMBOL",
        ],
        inplace=True,
    )


    dataset.reset_index(
        drop=True,
        inplace=True,
    )


    # =====================================================
    # FINAL COLUMNS
    # =====================================================

    output_columns = [

        "Date",
        "SYMBOL",
        "SECTOR",

        "Open",
        "High",
        "Low",
        "Close",
        "Volume",

        *FEATURE_COLUMNS,

        "TARGET_THRESHOLD",
        "TARGET_RETURN_5D",
        "TARGET_CLASS",
        "TARGET_LABEL",
    ]


    dataset = dataset[
        output_columns
    ]


    # =====================================================
    # GLOBAL CLASS DISTRIBUTION
    # =====================================================

    distribution = (
        calculate_distribution(
            dataset
        )
    )


    majority_class = max(

        distribution,

        key=lambda name:
            distribution[
                name
            ][
                "count"
            ],
    )


    majority_accuracy = (
        distribution[
            majority_class
        ][
            "percentage"
        ]
    )


    # =====================================================
    # THRESHOLD STATS
    # =====================================================

    threshold_percent = (
        dataset[
            "TARGET_THRESHOLD"
        ]
        *
        100
    )


    # =====================================================
    # PRINT SUMMARY
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " V8 DATASET SUMMARY"
    )

    print(
        "=============================================="
    )


    print(
        "\nSuccessful Stocks:",
        dataset[
            "SYMBOL"
        ]
        .nunique()
    )


    print(
        "Failed Stocks:",
        len(
            failed_stocks
        )
    )


    print(
        "\nRows before cleaning:",
        before_drop
    )


    print(
        "Usable Rows:",
        after_drop
    )


    print(
        "Feature Count:",
        len(
            FEATURE_COLUMNS
        )
    )


    print(
        "Date Range:",
        dataset[
            "Date"
        ]
        .min()
        .date(),
        "→",
        dataset[
            "Date"
        ]
        .max()
        .date(),
    )


    print(
        "\nDynamic Threshold:"
    )


    print(
        "Average:",
        f"{threshold_percent.mean():.2f}%"
    )


    print(
        "Median:",
        f"{threshold_percent.median():.2f}%"
    )


    print(
        "Minimum:",
        f"{threshold_percent.min():.2f}%"
    )


    print(
        "Maximum:",
        f"{threshold_percent.max():.2f}%"
    )


    # =====================================================
    # GLOBAL CLASS DISTRIBUTION
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " V8 GLOBAL CLASS DISTRIBUTION"
    )

    print(
        "=============================================="
    )


    for class_name in [
        "BEARISH",
        "NEUTRAL",
        "BULLISH",
    ]:

        values = (
            distribution[
                class_name
            ]
        )


        print(
            f"\n{class_name}:"
        )


        print(
            "Count:",
            values[
                "count"
            ]
        )


        print(
            "Percentage:",
            f'{values["percentage"]:.2f}%'
        )


    print(
        "\nMajority Baseline:"
    )


    print(
        majority_class,
        f"{majority_accuracy:.2f}%"
    )


    # =====================================================
    # PER-SECTOR DISTRIBUTION
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " PER-SECTOR DISTRIBUTION"
    )

    print(
        "=============================================="
    )


    sectors = sorted(
        dataset[
            "SECTOR"
        ]
        .unique()
    )


    for sector in sectors:

        sector_data = dataset[
            dataset[
                "SECTOR"
            ]
            ==
            sector
        ]


        sector_distribution = (
            calculate_distribution(
                sector_data
            )
        )


        print(
            f"\n{sector}"
        )


        print(
            "Rows:",
            len(
                sector_data
            )
        )


        for class_name in [
            "BEARISH",
            "NEUTRAL",
            "BULLISH",
        ]:

            values = (
                sector_distribution[
                    class_name
                ]
            )


            print(

                f'{class_name:<8}: '

                f'{values["count"]:>5} '

                f'({values["percentage"]:.2f}%)'
            )


    # =====================================================
    # SAVE CSV
    # =====================================================

    dataset.to_csv(
        CSV_PATH,
        index=False,
    )


    # =====================================================
    # METADATA
    # =====================================================

    metadata = {

        "dataset":
            "StockVision V8 Advanced Dataset",

        "target_horizon_days":
            TARGET_HORIZON,

        "target_type":
            "ATR volatility adjusted 3-class",

        "atr_factor":
            ATR_FACTOR,

        "market_symbol":
            MARKET_SYMBOL,

        "stocks":
            STOCKS,

        "stock_sectors":
            STOCK_SECTORS,

        "feature_count":
            len(
                FEATURE_COLUMNS
            ),

        "feature_columns":
            FEATURE_COLUMNS,

        "usable_rows":
            int(
                len(
                    dataset
                )
            ),

        "date_start":
            str(
                dataset[
                    "Date"
                ]
                .min()
                .date()
            ),

        "date_end":
            str(
                dataset[
                    "Date"
                ]
                .max()
                .date()
            ),

        "threshold_statistics_percent": {

            "mean":
                round(
                    float(
                        threshold_percent.mean()
                    ),
                    4,
                ),

            "median":
                round(
                    float(
                        threshold_percent.median()
                    ),
                    4,
                ),

            "min":
                round(
                    float(
                        threshold_percent.min()
                    ),
                    4,
                ),

            "max":
                round(
                    float(
                        threshold_percent.max()
                    ),
                    4,
                ),
        },

        "class_distribution":
            distribution,

        "majority_class":
            majority_class,

        "majority_baseline_accuracy_percent":
            majority_accuracy,

        "failed_stocks":
            failed_stocks,
    }


    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )


    # =====================================================
    # FINAL
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " V8 DATASET SAVED"
    )

    print(
        "=============================================="
    )


    print(
        "\nCSV:"
    )


    print(
        CSV_PATH
    )


    print(
        "\nMetadata:"
    )


    print(
        METADATA_PATH
    )


    print(
        "\nStockVision V8 dataset creation complete."
    )


    return dataset


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    build_v8_dataset()