import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


SEQUENCE_LENGTH = 60


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


def add_features(data: pd.DataFrame):
    """
    Create technical/market features from real OHLCV data.

    Target:
        NEXT_RETURN = percentage return of the NEXT trading day.
    """

    df = data.copy()

    # -------------------------------------------------
    # Daily return
    # -------------------------------------------------

    df["DAILY_RETURN"] = df["Close"].pct_change()

    # -------------------------------------------------
    # Intraday price range
    # -------------------------------------------------

    df["HIGH_LOW_RANGE"] = (
        (df["High"] - df["Low"])
        / df["Close"]
    )

    # -------------------------------------------------
    # Open -> Close return
    # -------------------------------------------------

    df["OPEN_CLOSE_RETURN"] = (
        (df["Close"] - df["Open"])
        / df["Open"]
    )

    # -------------------------------------------------
    # Volume change
    # -------------------------------------------------

    df["VOLUME_CHANGE"] = (
        df["Volume"]
        .replace(0, np.nan)
        .pct_change()
    )

    # -------------------------------------------------
    # SMA 20
    # -------------------------------------------------

    df["SMA20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    df["SMA20_DISTANCE"] = (
        df["Close"] / df["SMA20"]
    ) - 1

    # -------------------------------------------------
    # EMA 20
    # -------------------------------------------------

    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA20_DISTANCE"] = (
        df["Close"] / df["EMA20"]
    ) - 1

    # -------------------------------------------------
    # RSI 14
    # -------------------------------------------------

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = (
        gain
        .rolling(window=14)
        .mean()
    )

    average_loss = (
        loss
        .rolling(window=14)
        .mean()
    )

    rs = (
        average_gain
        / average_loss.replace(0, np.nan)
    )

    df["RSI14"] = (
        100
        - (100 / (1 + rs))
    )

    # -------------------------------------------------
    # MACD
    # -------------------------------------------------

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

    macd = ema12 - ema26

    macd_signal = (
        macd
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # Convert MACD into relative values instead
    # of absolute rupee values.
    df["MACD_PERCENT"] = (
        macd / df["Close"]
    )

    df["MACD_SIGNAL_PERCENT"] = (
        macd_signal / df["Close"]
    )

    # -------------------------------------------------
    # 20-day volatility
    # -------------------------------------------------

    df["VOLATILITY20"] = (
        df["DAILY_RETURN"]
        .rolling(window=20)
        .std()
    )

    # -------------------------------------------------
    # Momentum
    # -------------------------------------------------

    df["MOMENTUM5"] = (
        df["Close"]
        .pct_change(periods=5)
    )

    df["MOMENTUM10"] = (
        df["Close"]
        .pct_change(periods=10)
    )

    # -------------------------------------------------
    # TARGET
    #
    # Example:
    #
    # today's close = ₹1300
    # tomorrow close = ₹1313
    #
    # target = +1%
    # -------------------------------------------------

    df["NEXT_RETURN"] = (
        df["Close"]
        .shift(-1)
        / df["Close"]
    ) - 1

    # -------------------------------------------------
    # Clean invalid rows
    # -------------------------------------------------

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.dropna(inplace=True)

    return df


def _create_sequences(
    data,
    target_start,
    target_end,
    feature_scaler,
    target_scaler,
    sequence_length
):
    """
    Creates sequences such that:

    Previous 60 trading days
            ↓
         BiLSTM
            ↓
    Next-day return
    """

    X = []
    y = []

    current_closes = []
    actual_next_closes = []

    scaled_features = (
        feature_scaler.transform(
            data[FEATURE_COLUMNS]
        )
    )

    scaled_targets = (
        target_scaler.transform(
            data[["NEXT_RETURN"]]
        )
        .reshape(-1)
    )

    first_index = max(
        target_start,
        sequence_length - 1
    )

    for i in range(
        first_index,
        target_end
    ):

        sequence_start = (
            i - sequence_length + 1
        )

        sequence_end = i + 1

        X.append(
            scaled_features[
                sequence_start:sequence_end
            ]
        )

        y.append(
            scaled_targets[i]
        )

        current_close = float(
            data["Close"].iloc[i]
        )

        actual_return = float(
            data["NEXT_RETURN"].iloc[i]
        )

        actual_next_close = (
            current_close
            * (1 + actual_return)
        )

        current_closes.append(
            current_close
        )

        actual_next_closes.append(
            actual_next_close
        )

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(
            current_closes,
            dtype=np.float32
        ),
        np.array(
            actual_next_closes,
            dtype=np.float32
        )
    )


def prepare_data_splits(
    data: pd.DataFrame,
    sequence_length: int = 60,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15
):
    """
    Leakage-free chronological split.

    Oldest 70% = Training
    Next   15% = Validation
    Latest 15% = Testing

    Feature scaler and target scaler are fitted
    ONLY on the training period.
    """

    total_rows = len(data)

    train_end = int(
        total_rows * train_ratio
    )

    validation_end = int(
        total_rows
        * (
            train_ratio
            + validation_ratio
        )
    )

    # -------------------------------------------------
    # Training section only
    # -------------------------------------------------

    training_data = data.iloc[
        :train_end
    ]

    # -------------------------------------------------
    # StandardScaler works better here because
    # return-based features are centered around zero.
    # -------------------------------------------------

    feature_scaler = StandardScaler()

    target_scaler = StandardScaler()

    feature_scaler.fit(
        training_data[
            FEATURE_COLUMNS
        ]
    )

    target_scaler.fit(
        training_data[
            ["NEXT_RETURN"]
        ]
    )

    # -------------------------------------------------
    # Train sequences
    # -------------------------------------------------

    (
        X_train,
        y_train,
        current_train,
        actual_train
    ) = _create_sequences(
        data=data,
        target_start=0,
        target_end=train_end,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        sequence_length=sequence_length
    )

    # -------------------------------------------------
    # Validation sequences
    # -------------------------------------------------

    (
        X_val,
        y_val,
        current_val,
        actual_val
    ) = _create_sequences(
        data=data,
        target_start=train_end,
        target_end=validation_end,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        sequence_length=sequence_length
    )

    # -------------------------------------------------
    # Test sequences
    # -------------------------------------------------

    (
        X_test,
        y_test,
        current_test,
        actual_test
    ) = _create_sequences(
        data=data,
        target_start=validation_end,
        target_end=total_rows,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        sequence_length=sequence_length
    )

    return {
        "X_train": X_train,
        "y_train": y_train,

        "X_val": X_val,
        "y_val": y_val,

        "X_test": X_test,
        "y_test": y_test,

        "current_close_test":
            current_test,

        "actual_next_close_test":
            actual_test,

        "feature_scaler":
            feature_scaler,

        "target_scaler":
            target_scaler,

        "train_end":
            train_end,

        "validation_end":
            validation_end,
    }