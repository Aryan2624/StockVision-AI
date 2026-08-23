import os
import sys
import json

import joblib
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)


# =========================================================
# IMPORTS
# =========================================================

try:
    from .data import get_historical_data

    from .multi_features import (
        prepare_training_frame,
        FEATURE_COLUMNS,
        TARGET_COLUMNS,
        HORIZONS,
        SEQUENCE_LENGTH,
    )

except ImportError:
    from data import get_historical_data

    from multi_features import (
        prepare_training_frame,
        FEATURE_COLUMNS,
        TARGET_COLUMNS,
        HORIZONS,
        SEQUENCE_LENGTH,
    )


# =========================================================
# RANDOM SEEDS
# =========================================================

np.random.seed(42)

tf.random.set_seed(42)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


def safe_symbol_name(symbol):
    return (
        symbol
        .upper()
        .replace(".", "_")
        .replace("^", "")
    )


def get_multi_model_paths(symbol):
    safe_symbol = safe_symbol_name(
        symbol
    )

    model_path = os.path.join(
        MODELS_DIR,
        f"{safe_symbol}_multi.keras"
    )

    feature_scaler_path = os.path.join(
        MODELS_DIR,
        f"{safe_symbol}_multi_feature_scaler.pkl"
    )

    target_scaler_path = os.path.join(
        MODELS_DIR,
        f"{safe_symbol}_multi_target_scaler.pkl"
    )

    calibration_path = os.path.join(
        MODELS_DIR,
        f"{safe_symbol}_multi_calibration.json"
    )

    metrics_path = os.path.join(
        MODELS_DIR,
        f"{safe_symbol}_multi_metrics.json"
    )

    return (
        model_path,
        feature_scaler_path,
        target_scaler_path,
        calibration_path,
        metrics_path,
    )


# =========================================================
# CREATE SEQUENCES
# =========================================================

def create_sequences(
    scaled_features,
    scaled_targets,
    original_targets,
    close_prices,
    train_end,
    validation_end,
):
    X_train = []
    y_train = []

    X_validation = []
    y_validation = []
    y_validation_original = []
    close_validation = []

    X_test = []
    y_test = []
    y_test_original = []
    close_test = []


    for index in range(
        SEQUENCE_LENGTH - 1,
        len(scaled_features)
    ):

        start_index = (
            index
            -
            SEQUENCE_LENGTH
            +
            1
        )

        sequence = (
            scaled_features[
                start_index:
                index + 1
            ]
        )

        target_scaled = (
            scaled_targets[
                index
            ]
        )

        target_original = (
            original_targets[
                index
            ]
        )

        current_close = (
            close_prices[
                index
            ]
        )


        # ---------------------------------------------
        # TRAIN
        # ---------------------------------------------

        if index < train_end:

            X_train.append(
                sequence
            )

            y_train.append(
                target_scaled
            )


        # ---------------------------------------------
        # VALIDATION
        # ---------------------------------------------

        elif index < validation_end:

            X_validation.append(
                sequence
            )

            y_validation.append(
                target_scaled
            )

            y_validation_original.append(
                target_original
            )

            close_validation.append(
                current_close
            )


        # ---------------------------------------------
        # TEST
        # ---------------------------------------------

        else:

            X_test.append(
                sequence
            )

            y_test.append(
                target_scaled
            )

            y_test_original.append(
                target_original
            )

            close_test.append(
                current_close
            )


    return {
        "X_train": np.asarray(
            X_train,
            dtype=np.float32
        ),

        "y_train": np.asarray(
            y_train,
            dtype=np.float32
        ),

        "X_validation": np.asarray(
            X_validation,
            dtype=np.float32
        ),

        "y_validation": np.asarray(
            y_validation,
            dtype=np.float32
        ),

        "y_validation_original": np.asarray(
            y_validation_original,
            dtype=np.float32
        ),

        "close_validation": np.asarray(
            close_validation,
            dtype=np.float32
        ),

        "X_test": np.asarray(
            X_test,
            dtype=np.float32
        ),

        "y_test": np.asarray(
            y_test,
            dtype=np.float32
        ),

        "y_test_original": np.asarray(
            y_test_original,
            dtype=np.float32
        ),

        "close_test": np.asarray(
            close_test,
            dtype=np.float32
        ),
    }


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_multi_horizon_data(
    symbol
):
    print(
        "\nDownloading historical market data..."
    )

    raw_data = get_historical_data(
        symbol,
        period="10y"
    )


    data = prepare_training_frame(
        raw_data
    )


    if len(data) < 500:
        raise ValueError(
            f"Not enough historical data for {symbol}. "
            f"Found only {len(data)} usable rows."
        )


    feature_values = (
        data[
            FEATURE_COLUMNS
        ]
        .values
        .astype(
            np.float32
        )
    )


    target_values = (
        data[
            TARGET_COLUMNS
        ]
        .values
        .astype(
            np.float32
        )
    )


    close_prices = (
        data[
            "Close"
        ]
        .values
        .astype(
            np.float32
        )
    )


    total_rows = len(
        data
    )


    # Chronological:
    # 70% train
    # 15% validation
    # 15% test

    train_end = int(
        total_rows * 0.70
    )

    validation_end = int(
        total_rows * 0.85
    )


    # =====================================================
    # IMPORTANT:
    # fit scalers ONLY on training data
    # =====================================================

    feature_scaler = (
        StandardScaler()
    )

    target_scaler = (
        StandardScaler()
    )


    feature_scaler.fit(
        feature_values[
            :train_end
        ]
    )


    target_scaler.fit(
        target_values[
            :train_end
        ]
    )


    scaled_features = (
        feature_scaler.transform(
            feature_values
        )
    )


    scaled_targets = (
        target_scaler.transform(
            target_values
        )
    )


    sequences = create_sequences(
        scaled_features=scaled_features,

        scaled_targets=scaled_targets,

        original_targets=target_values,

        close_prices=close_prices,

        train_end=train_end,

        validation_end=validation_end,
    )


    return (
        data,
        sequences,
        feature_scaler,
        target_scaler,
    )


# =========================================================
# MODEL
# =========================================================

def build_multi_output_model():
    model = Sequential(
        [
            Input(
                shape=(
                    SEQUENCE_LENGTH,
                    len(
                        FEATURE_COLUMNS
                    ),
                )
            ),

            Bidirectional(
                LSTM(
                    64,
                    return_sequences=True
                )
            ),

            Dropout(
                0.25
            ),

            Bidirectional(
                LSTM(
                    32
                )
            ),

            Dropout(
                0.25
            ),

            Dense(
                32,
                activation="relu"
            ),

            Dropout(
                0.10
            ),

            # -----------------------------------------
            # 4 OUTPUTS:
            #
            # 1D
            # 3D
            # 5D
            # 10D
            # -----------------------------------------

            Dense(
                len(
                    HORIZONS
                ),
                activation="linear"
            ),
        ]
    )


    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005
    )


    model.compile(
        optimizer=optimizer,

        loss=tf.keras.losses.Huber(),

        metrics=[
            "mae"
        ],
    )


    return model


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    actual_returns,
    predicted_returns,
    current_close
):
    results = {}


    for column_index, horizon in enumerate(
        HORIZONS
    ):

        actual_return = (
            actual_returns[
                :,
                column_index
            ]
        )


        predicted_return = (
            predicted_returns[
                :,
                column_index
            ]
        )


        actual_price = (
            current_close
            *
            (
                1
                +
                actual_return
            )
        )


        predicted_price = (
            current_close
            *
            (
                1
                +
                predicted_return
            )
        )


        baseline_price = (
            current_close
        )


        # ---------------------------------------------
        # RETURN METRICS
        # ---------------------------------------------

        return_errors = (
            predicted_return
            -
            actual_return
        )


        return_mae = (
            np.mean(
                np.abs(
                    return_errors
                )
            )
            *
            100
        )


        return_rmse = (
            np.sqrt(
                np.mean(
                    return_errors ** 2
                )
            )
            *
            100
        )


        # ---------------------------------------------
        # PRICE METRICS
        # ---------------------------------------------

        price_errors = (
            predicted_price
            -
            actual_price
        )


        price_mae = (
            np.mean(
                np.abs(
                    price_errors
                )
            )
        )


        price_rmse = np.sqrt(
            np.mean(
                price_errors ** 2
            )
        )


        valid_actual = (
            actual_price != 0
        )


        if np.any(
            valid_actual
        ):
            mape = (
                np.mean(
                    np.abs(
                        (
                            actual_price[
                                valid_actual
                            ]
                            -
                            predicted_price[
                                valid_actual
                            ]
                        )
                        /
                        actual_price[
                            valid_actual
                        ]
                    )
                )
                *
                100
            )
        else:
            mape = 0.0


        ss_res = np.sum(
            (
                actual_price
                -
                predicted_price
            ) ** 2
        )


        ss_total = np.sum(
            (
                actual_price
                -
                np.mean(
                    actual_price
                )
            ) ** 2
        )


        if ss_total == 0:
            r2 = 0.0
        else:
            r2 = (
                1
                -
                (
                    ss_res
                    /
                    ss_total
                )
            )


        # ---------------------------------------------
        # DIRECTION ACCURACY
        # ---------------------------------------------

        actual_direction = (
            actual_return > 0
        )


        predicted_direction = (
            predicted_return > 0
        )


        direction_accuracy = (
            np.mean(
                actual_direction
                ==
                predicted_direction
            )
            *
            100
        )


        # ---------------------------------------------
        # NAIVE BASELINE
        #
        # Future close = today's close
        # ---------------------------------------------

        baseline_error = (
            baseline_price
            -
            actual_price
        )


        baseline_mae = (
            np.mean(
                np.abs(
                    baseline_error
                )
            )
        )


        baseline_rmse = np.sqrt(
            np.mean(
                baseline_error ** 2
            )
        )


        improvement_percent = (
            (
                baseline_mae
                -
                price_mae
            )
            /
            baseline_mae
            *
            100
            if baseline_mae != 0
            else 0.0
        )


        results[
            f"{horizon}D"
        ] = {
            "return_mae_percent": round(
                float(
                    return_mae
                ),
                4
            ),

            "return_rmse_percent": round(
                float(
                    return_rmse
                ),
                4
            ),

            "price_mae": round(
                float(
                    price_mae
                ),
                2
            ),

            "price_rmse": round(
                float(
                    price_rmse
                ),
                2
            ),

            "mape_percent": round(
                float(
                    mape
                ),
                4
            ),

            "r2": round(
                float(
                    r2
                ),
                4
            ),

            "direction_accuracy_percent": round(
                float(
                    direction_accuracy
                ),
                2
            ),

            "baseline_mae": round(
                float(
                    baseline_mae
                ),
                2
            ),

            "baseline_rmse": round(
                float(
                    baseline_rmse
                ),
                2
            ),

            "improvement_vs_baseline_percent": round(
                float(
                    improvement_percent
                ),
                2
            ),

            "beats_baseline": bool(
                price_mae
                <
                baseline_mae
            ),
        }


    return results


# =========================================================
# VALIDATION CALIBRATION
# =========================================================

def calculate_calibration(
    actual_returns,
    predicted_returns
):
    """
    Calculate empirical forecast uncertainty
    from validation prediction errors.

    q80 = 80th percentile absolute error
    q90 = 90th percentile absolute error

    These are validation-based estimated
    error ranges, NOT guaranteed confidence
    intervals.
    """

    calibration = {}


    absolute_errors = np.abs(
        predicted_returns
        -
        actual_returns
    )


    for index, horizon in enumerate(
        HORIZONS
    ):

        horizon_errors = (
            absolute_errors[
                :,
                index
            ]
            *
            100
        )


        q80 = np.quantile(
            horizon_errors,
            0.80
        )


        q90 = np.quantile(
            horizon_errors,
            0.90
        )


        median_error = np.median(
            horizon_errors
        )


        calibration[
            f"{horizon}D"
        ] = {
            "median_absolute_error_percent": round(
                float(
                    median_error
                ),
                4
            ),

            "q80_absolute_error_percent": round(
                float(
                    q80
                ),
                4
            ),

            "q90_absolute_error_percent": round(
                float(
                    q90
                ),
                4
            ),
        }


    return calibration


# =========================================================
# TRAIN
# =========================================================

def train_multi_horizon_model(
    symbol
):
    symbol = (
        symbol
        .upper()
        .strip()
    )


    if (
        "." not in symbol
        and not symbol.startswith(
            "^"
        )
    ):
        symbol = (
            f"{symbol}.NS"
        )


    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION MULTI-HORIZON BiLSTM"
    )

    print(
        "=============================================="
    )


    print(
        f"\nStock: {symbol}"
    )


    print(
        "Forecast horizons:",
        HORIZONS
    )


    print(
        "Features:",
        len(
            FEATURE_COLUMNS
        )
    )


    print(
        "Lookback:",
        SEQUENCE_LENGTH,
        "trading days"
    )


    (
        data,
        sequences,
        feature_scaler,
        target_scaler,
    ) = prepare_multi_horizon_data(
        symbol
    )


    X_train = (
        sequences[
            "X_train"
        ]
    )

    y_train = (
        sequences[
            "y_train"
        ]
    )


    X_validation = (
        sequences[
            "X_validation"
        ]
    )

    y_validation = (
        sequences[
            "y_validation"
        ]
    )


    X_test = (
        sequences[
            "X_test"
        ]
    )


    print(
        "\nDataset:"
    )

    print(
        "Usable rows:",
        len(
            data
        )
    )


    print(
        "Train X:",
        X_train.shape
    )


    print(
        "Train y:",
        y_train.shape
    )


    print(
        "Validation X:",
        X_validation.shape
    )


    print(
        "Test X:",
        X_test.shape
    )


    if (
        len(X_train) == 0
        or len(X_validation) == 0
        or len(X_test) == 0
    ):
        raise ValueError(
            "Train/validation/test sequence creation failed."
        )


    (
        model_path,
        feature_scaler_path,
        target_scaler_path,
        calibration_path,
        metrics_path,
    ) = get_multi_model_paths(
        symbol
    )


    model = (
        build_multi_output_model()
    )


    print(
        "\nModel architecture:"
    )

    model.summary()


    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1,
        ),

        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


    print(
        "\nStarting training...\n"
    )


    model.fit(
        X_train,
        y_train,

        validation_data=(
            X_validation,
            y_validation,
        ),

        epochs=80,

        batch_size=32,

        shuffle=False,

        callbacks=callbacks,

        verbose=1,
    )


    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    model = (
        tf.keras.models.load_model(
            model_path
        )
    )


    # =====================================================
    # VALIDATION PREDICTIONS
    # =====================================================

    validation_prediction_scaled = (
        model.predict(
            X_validation,
            verbose=0
        )
    )


    validation_prediction = (
        target_scaler
        .inverse_transform(
            validation_prediction_scaled
        )
    )


    validation_actual = (
        sequences[
            "y_validation_original"
        ]
    )


    # =====================================================
    # CALIBRATION / ERROR RANGE
    # =====================================================

    calibration = (
        calculate_calibration(
            validation_actual,
            validation_prediction
        )
    )


    # =====================================================
    # TEST PREDICTIONS
    # =====================================================

    test_prediction_scaled = (
        model.predict(
            X_test,
            verbose=0
        )
    )


    test_prediction = (
        target_scaler
        .inverse_transform(
            test_prediction_scaled
        )
    )


    test_actual = (
        sequences[
            "y_test_original"
        ]
    )


    test_close = (
        sequences[
            "close_test"
        ]
    )


    metrics = calculate_metrics(
        actual_returns=test_actual,

        predicted_returns=test_prediction,

        current_close=test_close,
    )


    # =====================================================
    # SAVE SCALERS
    # =====================================================

    joblib.dump(
        feature_scaler,
        feature_scaler_path
    )


    joblib.dump(
        target_scaler,
        target_scaler_path
    )


    # =====================================================
    # SAVE CALIBRATION
    # =====================================================

    with open(
        calibration_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            calibration,
            file,
            indent=4
        )


    # =====================================================
    # SAVE METRICS
    # =====================================================

    metrics_file_data = {
        "symbol": symbol,

        "model": "Multi-Horizon BiLSTM",

        "lookback_days": SEQUENCE_LENGTH,

        "features_used": len(
            FEATURE_COLUMNS
        ),

        "horizons": HORIZONS,

        "rows": len(
            data
        ),

        "metrics": metrics,
    }


    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics_file_data,
            file,
            indent=4
        )


    # =====================================================
    # RESULTS
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " MULTI-HORIZON TEST RESULTS"
    )

    print(
        "=============================================="
    )


    for horizon in HORIZONS:

        key = (
            f"{horizon}D"
        )

        result = (
            metrics[
                key
            ]
        )


        print(
            f"\n---------- {key} FORECAST ----------"
        )


        print(
            "Return MAE:",
            f'{result["return_mae_percent"]:.4f}%'
        )


        print(
            "Return RMSE:",
            f'{result["return_rmse_percent"]:.4f}%'
        )


        print(
            "Price MAE:",
            f'₹{result["price_mae"]:.2f}'
        )


        print(
            "Price RMSE:",
            f'₹{result["price_rmse"]:.2f}'
        )


        print(
            "MAPE:",
            f'{result["mape_percent"]:.4f}%'
        )


        print(
            "R²:",
            result["r2"]
        )


        print(
            "Direction Accuracy:",
            f'{result["direction_accuracy_percent"]:.2f}%'
        )


        print(
            "Naive Baseline MAE:",
            f'₹{result["baseline_mae"]:.2f}'
        )


        print(
            "Improvement vs Baseline:",
            f'{result["improvement_vs_baseline_percent"]:.2f}%'
        )


        print(
            "Beats Baseline:",
            result["beats_baseline"]
        )


    print(
        "\n=============================================="
    )

    print(
        " VALIDATION-BASED FORECAST ERROR RANGES"
    )

    print(
        "=============================================="
    )


    for horizon in HORIZONS:

        key = (
            f"{horizon}D"
        )

        values = (
            calibration[
                key
            ]
        )


        print(
            f"\n{key}:"
        )


        print(
            "Median absolute error:",
            f'{values["median_absolute_error_percent"]:.4f}%'
        )


        print(
            "80% error band:",
            f'±{values["q80_absolute_error_percent"]:.4f}%'
        )


        print(
            "90% error band:",
            f'±{values["q90_absolute_error_percent"]:.4f}%'
        )


    print(
        "\n=============================================="
    )

    print(
        " FILES SAVED"
    )

    print(
        "=============================================="
    )


    print(
        "\nModel:"
    )

    print(
        model_path
    )


    print(
        "\nFeature scaler:"
    )

    print(
        feature_scaler_path
    )


    print(
        "\nTarget scaler:"
    )

    print(
        target_scaler_path
    )


    print(
        "\nCalibration:"
    )

    print(
        calibration_path
    )


    print(
        "\nMetrics:"
    )

    print(
        metrics_path
    )


    print(
        "\nMulti-horizon training complete."
    )


# =========================================================
# COMMAND LINE
# =========================================================

if __name__ == "__main__":

    symbol = (
        sys.argv[1]
        if len(
            sys.argv
        ) > 1
        else "RELIANCE.NS"
    )


    train_multi_horizon_model(
        symbol
    )