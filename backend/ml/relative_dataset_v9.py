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
# V9 CONFIG
# =========================================================

TARGET_HORIZON = 5

EXCESS_RETURN_THRESHOLD = 0.01

NUM_CLASSES = 3

CLASS_NAMES = {
    0: "UNDERPERFORM",
    1: "NEUTRAL",
    2: "OUTPERFORM",
}


MARKET_SYMBOL = "^NSEI"


# =========================================================
# STOCKS
# =========================================================

STOCKS = [
    "RELIANCE.NS",

    "TCS.NS",
    "INFY.NS",
    "WIPRO.NS",
    "HCLTECH.NS",

    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "AXISBANK.NS",
    "SBIN.NS",
    "KOTAKBANK.NS",

    "ITC.NS",
    "LT.NS",
    "BHARTIARTL.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
]


# =========================================================
# CLEAN V9 FEATURE SET
#
# Deliberately smaller than V8.
# =========================================================

FEATURE_COLUMNS = [

    # =====================================================
    # STOCK RETURN / MOMENTUM
    # =====================================================

    "RETURN_1",
    "RETURN_3",
    "RETURN_5",
    "RETURN_10",
    "RETURN_20",

    # =====================================================
    # STOCK TREND
    # =====================================================

    "SMA20_DISTANCE",
    "SMA50_DISTANCE",
    "EMA20_DISTANCE",
    "TREND_20",

    # =====================================================
    # MOMENTUM / TECHNICAL
    # =====================================================

    "RSI14_NORMALIZED",
    "MACD_HIST_PERCENT",

    # =====================================================
    # VOLATILITY
    # =====================================================

    "VOLATILITY_10",
    "VOLATILITY_20",
    "ATR14_PERCENT",
    "VOLATILITY_RATIO",

    # =====================================================
    # VOLUME
    # =====================================================

    "VOLUME_CHANGE",
    "VOLUME_ZSCORE_20",

    # =====================================================
    # NIFTY CONTEXT
    # =====================================================

    "NIFTY_RETURN_1",
    "NIFTY_RETURN_5",
    "NIFTY_RETURN_10",
    "NIFTY_RETURN_20",

    "NIFTY_SMA20_DISTANCE",
    "NIFTY_VOLATILITY_20",

    # =====================================================
    # STOCK VS MARKET
    # =====================================================

    "RELATIVE_RETURN_1",
    "RELATIVE_RETURN_5",
    "RELATIVE_RETURN_10",
    "RELATIVE_RETURN_20",

    "ROLLING_BETA_20",
    "ROLLING_CORRELATION_20",
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
    "stockvision_v9_relative_dataset.csv",
)


METADATA_PATH = os.path.join(
    DATASETS_DIR,
    "stockvision_v9_relative_metadata.json",
)


# =========================================================
# NORMALIZE DATA
# =========================================================

def normalize_data(data):

    df = data.copy()


    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):

        df.columns = [
            column[0]
            if isinstance(
                column,
                tuple,
            )
            else column

            for column in df.columns
        ]


    df.index = pd.to_datetime(
        df.index
    )


    try:

        if df.index.tz is not None:

            df.index = (
                df.index
                .tz_localize(None)
            )

    except Exception:
        pass


    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]


    missing = [
        column

        for column in required_columns

        if column not in df.columns
    ]


    if missing:

        raise ValueError(
            "Missing OHLCV columns: "
            +
            ", ".join(
                missing
            )
        )


    for column in required_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


    df = df[
        required_columns
    ].copy()


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


    return (
        true_range
        .rolling(period)
        .mean()
    )


# =========================================================
# BUILD NIFTY FEATURES
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


    features = pd.DataFrame(
        index=market.index
    )


    features[
        "NIFTY_CLOSE"
    ] = close


    features[
        "NIFTY_RETURN_1"
    ] = (
        close.pct_change(1)
    )


    features[
        "NIFTY_RETURN_5"
    ] = (
        close.pct_change(5)
    )


    features[
        "NIFTY_RETURN_10"
    ] = (
        close.pct_change(10)
    )


    features[
        "NIFTY_RETURN_20"
    ] = (
        close.pct_change(20)
    )


    sma20 = (
        close
        .rolling(20)
        .mean()
    )


    features[
        "NIFTY_SMA20_DISTANCE"
    ] = (
        close
        /
        sma20
        -
        1
    )


    features[
        "NIFTY_VOLATILITY_20"
    ] = (
        features[
            "NIFTY_RETURN_1"
        ]
        .rolling(20)
        .std()
    )


    return features


# =========================================================
# BUILD STOCK FEATURES
# =========================================================

def build_stock_dataset(
    symbol,
    stock_data,
    market_features,
):

    df = normalize_data(
        stock_data
    )


    # =====================================================
    # JOIN NIFTY ON TRADING DATES
    # =====================================================

    df = df.join(
        market_features,
        how="inner",
    )


    close = df[
        "Close"
    ]


    volume = df[
        "Volume"
    ]


    # =====================================================
    # STOCK RETURNS
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
    # TREND
    # =====================================================

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


    ema20 = (
        close
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
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
        "EMA20_DISTANCE"
    ] = (
        close
        /
        ema20
        -
        1
    )


    df[
        "TREND_20"
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
        period=14,
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
        "MACD_HIST_PERCENT"
    ] = (
        macd_histogram
        /
        close
    )


    # =====================================================
    # VOLATILITY
    # =====================================================

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
        "VOLATILITY_RATIO"
    ] = (
        df[
            "VOLATILITY_10"
        ]
        /
        df[
            "VOLATILITY_20"
        ]
    )


    atr14 = calculate_atr(
        df,
        period=14,
    )


    df[
        "ATR14_PERCENT"
    ] = (
        atr14
        /
        close
    )


    # =====================================================
    # VOLUME
    # =====================================================

    df[
        "VOLUME_CHANGE"
    ] = (
        volume.pct_change(1)
    )


    volume_mean = (
        volume
        .rolling(20)
        .mean()
    )


    volume_std = (
        volume
        .rolling(20)
        .std()
    )


    df[
        "VOLUME_ZSCORE_20"
    ] = (
        volume
        -
        volume_mean
    ) / volume_std


    # =====================================================
    # RELATIVE STRENGTH
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
        "RELATIVE_RETURN_10"
    ] = (
        df[
            "RETURN_10"
        ]
        -
        df[
            "NIFTY_RETURN_10"
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


    # =====================================================
    # FUTURE STOCK RETURN
    # =====================================================

    stock_future_return = (
        close.shift(
            -TARGET_HORIZON
        )
        /
        close
        -
        1
    )


    # =====================================================
    # FUTURE NIFTY RETURN
    #
    # Uses the same aligned trading dates.
    # =====================================================

    nifty_future_return = (
        df[
            "NIFTY_CLOSE"
        ]
        .shift(
            -TARGET_HORIZON
        )
        /
        df[
            "NIFTY_CLOSE"
        ]
        -
        1
    )


    # =====================================================
    # V9 TARGET:
    #
    # EXCESS RETURN =
    # Stock Future Return - NIFTY Future Return
    # =====================================================

    excess_return = (
        stock_future_return
        -
        nifty_future_return
    )


    df[
        "TARGET_STOCK_RETURN_5D"
    ] = (
        stock_future_return
    )


    df[
        "TARGET_NIFTY_RETURN_5D"
    ] = (
        nifty_future_return
    )


    df[
        "TARGET_EXCESS_RETURN_5D"
    ] = (
        excess_return
    )


    # =====================================================
    # CLASSIFICATION
    # =====================================================

    target_class = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )


    valid_target = (
        excess_return.notna()
    )


    # UNDERPERFORM
    target_class.loc[
        valid_target
        &
        (
            excess_return
            <
            -EXCESS_RETURN_THRESHOLD
        )
    ] = 0


    # NEUTRAL
    target_class.loc[
        valid_target
        &
        (
            excess_return
            >=
            -EXCESS_RETURN_THRESHOLD
        )
        &
        (
            excess_return
            <=
            EXCESS_RETURN_THRESHOLD
        )
    ] = 1


    # OUTPERFORM
    target_class.loc[
        valid_target
        &
        (
            excess_return
            >
            EXCESS_RETURN_THRESHOLD
        )
    ] = 2


    df[
        "TARGET_CLASS"
    ] = (
        target_class
    )


    df[
        "SYMBOL"
    ] = (
        symbol
    )


    df.index.name = "Date"


    return df


# =========================================================
# DISTRIBUTION
# =========================================================

def calculate_distribution(
    dataset,
):

    counts = (
        dataset[
            "TARGET_CLASS"
        ]
        .value_counts()
        .sort_index()
    )


    total = len(
        dataset
    )


    distribution = {}


    for class_id in range(
        NUM_CLASSES
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

            else 0.0
        )


        distribution[
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


    return distribution


# =========================================================
# BUILD UNIVERSAL V9 DATASET
# =========================================================

def build_v9_dataset():

    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V9 RELATIVE-STRENGTH DATASET"
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
        "Relative Threshold:",
        f"±{EXCESS_RETURN_THRESHOLD * 100:.2f}%"
    )


    # =====================================================
    # NIFTY
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
    # EACH STOCK
    # =====================================================

    for symbol in STOCKS:

        print(
            f"\nDownloading {symbol}..."
        )


        try:

            stock_data = (
                get_historical_data(
                    symbol,
                    period="10y",
                )
            )


            stock_frame = (
                build_stock_dataset(
                    symbol=
                        symbol,

                    stock_data=
                        stock_data,

                    market_features=
                        market_features,
                )
            )


            stock_frame = (
                stock_frame
                .reset_index()
            )


            print(
                "Raw rows:",
                len(
                    stock_frame
                )
            )


            all_frames.append(
                stock_frame
            )


        except Exception as error:

            print(
                f"FAILED {symbol}: {error}"
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
            "No V9 stock data was created."
        )


    # =====================================================
    # COMBINE
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


    dataset.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
        inplace=True,
    )


    # =====================================================
    # CLEAN
    # =====================================================

    before_cleaning = len(
        dataset
    )


    required_columns = [

        *FEATURE_COLUMNS,

        "TARGET_STOCK_RETURN_5D",
        "TARGET_NIFTY_RETURN_5D",
        "TARGET_EXCESS_RETURN_5D",
        "TARGET_CLASS",
    ]


    dataset.dropna(
        subset=
            required_columns,
        inplace=True,
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

        "Open",
        "High",
        "Low",
        "Close",
        "Volume",

        *FEATURE_COLUMNS,

        "TARGET_STOCK_RETURN_5D",
        "TARGET_NIFTY_RETURN_5D",
        "TARGET_EXCESS_RETURN_5D",

        "TARGET_CLASS",
        "TARGET_LABEL",
    ]


    dataset = dataset[
        output_columns
    ]


    # =====================================================
    # DISTRIBUTION
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
    # EXCESS RETURN STATISTICS
    # =====================================================

    excess_percent = (
        dataset[
            "TARGET_EXCESS_RETURN_5D"
        ]
        *
        100
    )


    # =====================================================
    # SUMMARY
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " V9 DATASET SUMMARY"
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
        "\nRows Before Cleaning:",
        before_cleaning
    )


    print(
        "Usable Rows:",
        len(
            dataset
        )
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
        "\n5D Excess Return Statistics:"
    )


    print(
        "Mean:",
        f"{excess_percent.mean():.2f}%"
    )


    print(
        "Median:",
        f"{excess_percent.median():.2f}%"
    )


    print(
        "Std:",
        f"{excess_percent.std():.2f}%"
    )


    # =====================================================
    # GLOBAL DISTRIBUTION
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " V9 GLOBAL CLASS DISTRIBUTION"
    )

    print(
        "=============================================="
    )


    for class_name in [
        "UNDERPERFORM",
        "NEUTRAL",
        "OUTPERFORM",
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
    # PER STOCK
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " PER-STOCK V9 DISTRIBUTION"
    )

    print(
        "=============================================="
    )


    for symbol in STOCKS:

        stock_subset = dataset[
            dataset[
                "SYMBOL"
            ]
            ==
            symbol
        ]


        if stock_subset.empty:

            continue


        stock_distribution = (
            calculate_distribution(
                stock_subset
            )
        )


        print(
            f"\n{symbol}"
        )


        print(
            "Rows:",
            len(
                stock_subset
            )
        )


        for class_name in [
            "UNDERPERFORM",
            "NEUTRAL",
            "OUTPERFORM",
        ]:

            values = (
                stock_distribution[
                    class_name
                ]
            )


            print(
                f'{class_name:<12}: '
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
            "StockVision V9 Relative Strength Dataset",

        "market_symbol":
            MARKET_SYMBOL,

        "stocks":
            STOCKS,

        "target_horizon_days":
            TARGET_HORIZON,

        "target":
            "5-day stock excess return vs NIFTY 50",

        "excess_return_threshold_percent":
            EXCESS_RETURN_THRESHOLD
            *
            100,

        "class_mapping": {
            str(
                key
            ):
                value

            for (
                key,
                value
            ) in CLASS_NAMES.items()
        },

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

        "class_distribution":
            distribution,

        "majority_class":
            majority_class,

        "majority_baseline_accuracy_percent":
            majority_accuracy,

        "excess_return_statistics_percent": {

            "mean":
                round(
                    float(
                        excess_percent.mean()
                    ),
                    4,
                ),

            "median":
                round(
                    float(
                        excess_percent.median()
                    ),
                    4,
                ),

            "std":
                round(
                    float(
                        excess_percent.std()
                    ),
                    4,
                ),
        },

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
    # COMPLETE
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " V9 DATASET SAVED"
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
        "\nStockVision V9 dataset creation complete."
    )


    return dataset


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    build_v9_dataset()