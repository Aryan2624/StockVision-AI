import os
import sys
import joblib
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from tensorflow.keras.losses import Huber


from data import get_historical_data

from features import (
    add_features,
    prepare_data_splits,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH
)


EPOCHS = 80
BATCH_SIZE = 32


# Makes results more reproducible
tf.keras.utils.set_random_seed(42)


def build_bilstm_model(
    sequence_length,
    number_of_features
):

    model = Sequential([

        Input(
            shape=(
                sequence_length,
                number_of_features
            )
        ),

        Bidirectional(
            LSTM(
                64,
                return_sequences=True
            )
        ),

        Dropout(0.25),

        Bidirectional(
            LSTM(
                32,
                return_sequences=False
            )
        ),

        Dropout(0.25),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.10),

        Dense(1)
    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005
        ),

        # More robust than pure MSE for
        # occasional large market moves
        loss=Huber(),

        metrics=[
            "mae"
        ]
    )

    return model


def train_model(symbol):

    symbol = symbol.upper()

    print("\n======================================")
    print(" STOCKVISION - BiLSTM V2")
    print(" NEXT-DAY RETURN PREDICTION")
    print("======================================")

    print(
        "\nStock:",
        symbol
    )

    # ======================================
    # 1. REAL HISTORICAL STOCK DATA
    # ======================================

    print(
        "\nDownloading real historical data..."
    )

    raw_data = get_historical_data(
        symbol,
        period="10y"
    )

    print(
        "Historical rows:",
        len(raw_data)
    )

    # ======================================
    # 2. FEATURE ENGINEERING
    # ======================================

    print(
        "\nCreating market features..."
    )

    data = add_features(
        raw_data
    )

    print(
        "Usable rows:",
        len(data)
    )

    print(
        "Features:",
        len(FEATURE_COLUMNS)
    )

    for feature in FEATURE_COLUMNS:
        print(
            " -",
            feature
        )

    # ======================================
    # 3. CHRONOLOGICAL DATA SPLIT
    # ======================================

    print(
        "\nPreparing leakage-free sequences..."
    )

    prepared = prepare_data_splits(
        data=data,
        sequence_length=SEQUENCE_LENGTH
    )

    X_train = prepared[
        "X_train"
    ]

    y_train = prepared[
        "y_train"
    ]

    X_val = prepared[
        "X_val"
    ]

    y_val = prepared[
        "y_val"
    ]

    X_test = prepared[
        "X_test"
    ]

    y_test = prepared[
        "y_test"
    ]

    current_close_test = prepared[
        "current_close_test"
    ]

    actual_next_close_test = prepared[
        "actual_next_close_test"
    ]

    feature_scaler = prepared[
        "feature_scaler"
    ]

    target_scaler = prepared[
        "target_scaler"
    ]

    print(
        "\nTrain:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    # ======================================
    # 4. BUILD BiLSTM
    # ======================================

    model = build_bilstm_model(

        SEQUENCE_LENGTH,

        len(
            FEATURE_COLUMNS
        )
    )

    print(
        "\nModel architecture:"
    )

    model.summary()

    # ======================================
    # 5. SAVE PATHS
    # ======================================

    safe_symbol = (
        symbol
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

    os.makedirs(
        models_dir,
        exist_ok=True
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

    # ======================================
    # 6. CALLBACKS
    # ======================================

    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=12,

        restore_best_weights=True,

        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=5,

        min_lr=0.00001,

        verbose=1
    )

    checkpoint = ModelCheckpoint(

        filepath=model_path,

        monitor="val_loss",

        save_best_only=True,

        verbose=1
    )

    # ======================================
    # 7. TRAIN
    # ======================================

    print(
        "\nStarting BiLSTM V2 training...\n"
    )

    model.fit(

        X_train,

        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        shuffle=False,

        callbacks=[
            early_stopping,
            reduce_lr,
            checkpoint
        ],

        verbose=1
    )

    # Save final best-restored model
    model.save(
        model_path
    )

    # ======================================
    # 8. PREDICT RETURNS
    # ======================================

    print(
        "\nEvaluating on unseen test data..."
    )

    predicted_scaled_returns = (
        model.predict(
            X_test,
            verbose=0
        )
    )

    predicted_returns = (
        target_scaler
        .inverse_transform(
            predicted_scaled_returns
        )
        .flatten()
    )

    actual_returns = (
        target_scaler
        .inverse_transform(
            y_test.reshape(-1, 1)
        )
        .flatten()
    )

    # ======================================
    # 9. TURN RETURN INTO STOCK PRICE
    # ======================================

    predicted_prices = (

        current_close_test

        * (
            1
            + predicted_returns
        )
    )

    actual_prices = (
        actual_next_close_test
    )

    # ======================================
    # 10. PRICE METRICS
    # ======================================

    mae = mean_absolute_error(
        actual_prices,
        predicted_prices
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_prices,
            predicted_prices
        )
    )

    r2_price = r2_score(
        actual_prices,
        predicted_prices
    )

    mape = (
        np.mean(
            np.abs(
                (
                    actual_prices
                    - predicted_prices
                )
                / actual_prices
            )
        )
        * 100
    )

    # ======================================
    # 11. RETURN METRICS
    # ======================================

    return_mae = (
        mean_absolute_error(
            actual_returns,
            predicted_returns
        )
        * 100
    )

    return_rmse = (
        np.sqrt(
            mean_squared_error(
                actual_returns,
                predicted_returns
            )
        )
        * 100
    )

    # ======================================
    # 12. DIRECTION ACCURACY
    # ======================================

    actual_direction = (
        actual_returns > 0
    )

    predicted_direction = (
        predicted_returns > 0
    )

    direction_accuracy = (
        np.mean(
            actual_direction
            == predicted_direction
        )
        * 100
    )

    # ======================================
    # 13. NAIVE BASELINE
    #
    # Baseline assumption:
    # tomorrow close = today close
    # ======================================

    baseline_predictions = (
        current_close_test
    )

    baseline_mae = (
        mean_absolute_error(
            actual_prices,
            baseline_predictions
        )
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            actual_prices,
            baseline_predictions
        )
    )

    # ======================================
    # 14. RESULTS
    # ======================================

    print("\n======================================")
    print(" BiLSTM V2 FINAL TEST RESULTS")
    print("======================================")

    print(
        f"Price MAE: ₹{mae:.2f}"
    )

    print(
        f"Price RMSE: ₹{rmse:.2f}"
    )

    print(
        f"Price R²: {r2_price:.4f}"
    )

    print(
        f"Price MAPE: {mape:.2f}%"
    )

    print(
        f"Return MAE: "
        f"{return_mae:.4f}%"
    )

    print(
        f"Return RMSE: "
        f"{return_rmse:.4f}%"
    )

    print(
        f"Direction Accuracy: "
        f"{direction_accuracy:.2f}%"
    )

    # ======================================
    # BASELINE
    # ======================================

    print("\n======================================")
    print(" NAIVE BASELINE")
    print("======================================")

    print(
        f"Baseline MAE: "
        f"₹{baseline_mae:.2f}"
    )

    print(
        f"Baseline RMSE: "
        f"₹{baseline_rmse:.2f}"
    )

    # ======================================
    # COMPARISON
    # ======================================

    print("\n======================================")
    print(" MODEL vs BASELINE")
    print("======================================")

    if mae < baseline_mae:

        improvement = (
            (
                baseline_mae
                - mae
            )
            / baseline_mae
        ) * 100

        print(
            "BiLSTM beats baseline MAE."
        )

        print(
            f"MAE improvement: "
            f"{improvement:.2f}%"
        )

    else:

        difference = (
            (
                mae
                - baseline_mae
            )
            / baseline_mae
        ) * 100

        print(
            "BiLSTM does NOT beat baseline yet."
        )

        print(
            f"BiLSTM MAE is "
            f"{difference:.2f}% worse."
        )

    # ======================================
    # 15. SAVE SCALERS
    # ======================================

    joblib.dump(
        feature_scaler,
        feature_scaler_path
    )

    joblib.dump(
        target_scaler,
        target_scaler_path
    )

    print("\nSaved model:")

    print(
        model_path
    )

    print(
        "\nSaved feature scaler:"
    )

    print(
        feature_scaler_path
    )

    print(
        "\nSaved target scaler:"
    )

    print(
        target_scaler_path
    )

    # ======================================
    # 16. SAMPLE PREDICTIONS
    # ======================================

    print("\n======================================")
    print(" SAMPLE PREDICTIONS")
    print("======================================")

    sample_count = min(
        15,
        len(actual_prices)
    )

    for i in range(
        sample_count
    ):

        predicted_signal = (
            "UP"
            if predicted_returns[i] > 0
            else "DOWN"
        )

        actual_signal = (
            "UP"
            if actual_returns[i] > 0
            else "DOWN"
        )

        print(

            f"Current: "
            f"₹{current_close_test[i]:.2f}"

            f" | Actual Next: "
            f"₹{actual_prices[i]:.2f}"

            f" | Predicted Next: "
            f"₹{predicted_prices[i]:.2f}"

            f" | Pred Return: "
            f"{predicted_returns[i] * 100:.2f}%"

            f" | Pred: "
            f"{predicted_signal}"

            f" | Actual: "
            f"{actual_signal}"
        )


if __name__ == "__main__":

    symbol = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "RELIANCE.NS"
    )

    train_model(
        symbol
    )