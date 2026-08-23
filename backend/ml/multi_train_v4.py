import os
import sys
import json

import joblib
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    LayerNormalization,
    Concatenate,
)

from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)


# =========================================================
# IMPORTS
# =========================================================

try:

    from .data import (
        get_historical_data
    )

    from .multi_features_v4 import (
        prepare_training_frame_v4,
        FEATURE_COLUMNS,
        TARGET_COLUMNS,
        HORIZONS,
        SEQUENCE_LENGTH,
    )

except ImportError:

    from data import (
        get_historical_data
    )

    from multi_features_v4 import (
        prepare_training_frame_v4,
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


# =========================================================
# SYMBOL
# =========================================================

def normalize_symbol(symbol):

    symbol = (
        symbol
        .upper()
        .strip()
    )

    if not symbol:

        raise ValueError(
            "Stock symbol cannot be empty."
        )

    if (
        "." not in symbol
        and
        not symbol.startswith("^")
    ):

        symbol = (
            f"{symbol}.NS"
        )

    return symbol


def safe_symbol_name(symbol):

    return (
        symbol
        .upper()
        .replace(
            ".",
            "_"
        )
        .replace(
            "^",
            ""
        )
    )


# =========================================================
# V4 PATHS
# =========================================================

def get_v4_model_paths(symbol):

    safe_symbol = (
        safe_symbol_name(
            symbol
        )
    )


    model_path = os.path.join(
        MODELS_DIR,
        f"{safe_symbol}_v4.keras"
    )


    feature_scaler_path = os.path.join(
        MODELS_DIR,
        f"{safe_symbol}_v4_feature_scaler.pkl"
    )


    target_scaler_path = os.path.join(
        MODELS_DIR,
        f"{safe_symbol}_v4_target_scaler.pkl"
    )


    calibration_path = os.path.join(
        MODELS_DIR,
        f"{safe_symbol}_v4_calibration.json"
    )


    metrics_path = os.path.join(
        MODELS_DIR,
        f"{safe_symbol}_v4_metrics.json"
    )


    return (
        model_path,
        feature_scaler_path,
        target_scaler_path,
        calibration_path,
        metrics_path,
    )


# =========================================================
# V3 METRICS PATH
# =========================================================

def get_v3_metrics_path(symbol):

    safe_symbol = (
        safe_symbol_name(
            symbol
        )
    )

    return os.path.join(
        MODELS_DIR,
        f"{safe_symbol}_multi_metrics.json"
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
        len(
            scaled_features
        )
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


        # =================================================
        # TRAIN
        # =================================================

        if index < train_end:

            X_train.append(
                sequence
            )

            y_train.append(
                target_scaled
            )


        # =================================================
        # VALIDATION
        # =================================================

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


        # =================================================
        # TEST
        # =================================================

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
# PREPARE V4 DATA
# =========================================================

def prepare_v4_data(symbol):

    print(
        "\nDownloading 10 years historical data..."
    )


    raw_data = get_historical_data(
        symbol,
        period="10y"
    )


    data = (
        prepare_training_frame_v4(
            raw_data
        )
    )


    if len(data) < 500:

        raise ValueError(
            f"Not enough historical data for {symbol}. "
            f"Found {len(data)} usable rows."
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


    # =====================================================
    # CHRONOLOGICAL SPLIT
    #
    # 70% train
    # 15% validation
    # 15% test
    # =====================================================

    train_end = int(
        total_rows *
        0.70
    )


    validation_end = int(
        total_rows *
        0.85
    )


    # =====================================================
    # SCALERS
    #
    # Fit ONLY on training rows.
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


    sequences = (
        create_sequences(

            scaled_features=
                scaled_features,

            scaled_targets=
                scaled_targets,

            original_targets=
                target_values,

            close_prices=
                close_prices,

            train_end=
                train_end,

            validation_end=
                validation_end,
        )
    )


    return (
        data,
        sequences,
        feature_scaler,
        target_scaler,
    )


# =========================================================
# BUILD BiLSTM V4
# =========================================================

def build_v4_model():

    inputs = Input(
        shape=(
            SEQUENCE_LENGTH,
            len(
                FEATURE_COLUMNS
            ),
        ),
        name="market_sequence"
    )


    # =====================================================
    # BiLSTM BLOCK 1
    # =====================================================

    x = Bidirectional(
        LSTM(
            96,
            return_sequences=True,
            kernel_regularizer=l2(
                0.0001
            ),
        ),
        name="bilstm_1"
    )(
        inputs
    )


    x = LayerNormalization(
        name="layer_norm_1"
    )(
        x
    )


    x = Dropout(
        0.25,
        name="dropout_1"
    )(
        x
    )


    # =====================================================
    # BiLSTM BLOCK 2
    # =====================================================

    x = Bidirectional(
        LSTM(
            48,
            return_sequences=False,
            kernel_regularizer=l2(
                0.0001
            ),
        ),
        name="bilstm_2"
    )(
        x
    )


    x = LayerNormalization(
        name="layer_norm_2"
    )(
        x
    )


    x = Dropout(
        0.25,
        name="dropout_2"
    )(
        x
    )


    # =====================================================
    # SHARED DENSE REPRESENTATION
    # =====================================================

    shared = Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(
            0.0001
        ),
        name="shared_dense"
    )(
        x
    )


    shared = Dropout(
        0.20,
        name="shared_dropout"
    )(
        shared
    )


    # =====================================================
    # HORIZON-SPECIFIC HEADS
    # =====================================================

    horizon_outputs = []


    for horizon in HORIZONS:

        head = Dense(
            24,
            activation="relu",
            name=(
                f"head_{horizon}d_dense"
            )
        )(
            shared
        )


        head = Dropout(
            0.10,
            name=(
                f"head_{horizon}d_dropout"
            )
        )(
            head
        )


        output = Dense(
            1,
            activation="linear",
            name=(
                f"return_{horizon}d"
            )
        )(
            head
        )


        horizon_outputs.append(
            output
        )


    # =====================================================
    # COMBINE 4 FORECASTS
    # =====================================================

    outputs = Concatenate(
        name="forecast_returns"
    )(
        horizon_outputs
    )


    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="StockVision_BiLSTM_V4"
    )


    optimizer = (
        tf.keras.optimizers.Adam(
            learning_rate=0.0003,
            clipnorm=1.0,
        )
    )


    model.compile(

        optimizer=
            optimizer,

        loss=
            tf.keras.losses.Huber(),

        metrics=[
            "mae"
        ],
    )


    return model


# =========================================================
# SAFE CORRELATION
# =========================================================

def safe_correlation(
    actual,
    predicted,
):

    try:

        if (
            len(actual) < 2
            or
            np.std(actual) == 0
            or
            np.std(predicted) == 0
        ):

            return 0.0


        correlation = (
            np.corrcoef(
                actual,
                predicted
            )[
                0,
                1
            ]
        )


        if not np.isfinite(
            correlation
        ):

            return 0.0


        return float(
            correlation
        )


    except Exception:

        return 0.0


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    actual_returns,
    predicted_returns,
    current_close,
):

    results = {}


    for (
        column_index,
        horizon,

    ) in enumerate(
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


        # =================================================
        # PRICE RECONSTRUCTION
        # =================================================

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


        # =================================================
        # RETURN ERRORS
        # =================================================

        return_error = (
            predicted_return
            -
            actual_return
        )


        return_mae = (
            np.mean(
                np.abs(
                    return_error
                )
            )
            *
            100
        )


        return_rmse = (
            np.sqrt(
                np.mean(
                    return_error ** 2
                )
            )
            *
            100
        )


        # =================================================
        # PRICE ERRORS
        # =================================================

        price_error = (
            predicted_price
            -
            actual_price
        )


        price_mae = (
            np.mean(
                np.abs(
                    price_error
                )
            )
        )


        price_rmse = (
            np.sqrt(
                np.mean(
                    price_error ** 2
                )
            )
        )


        # =================================================
        # MAPE
        # =================================================

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


        # =================================================
        # R2
        # =================================================

        ss_res = (
            np.sum(
                (
                    actual_price
                    -
                    predicted_price
                )
                ** 2
            )
        )


        ss_total = (
            np.sum(
                (
                    actual_price
                    -
                    np.mean(
                        actual_price
                    )
                )
                ** 2
            )
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


        # =================================================
        # DIRECTION ACCURACY
        # =================================================

        actual_direction = (
            actual_return >
            0
        )


        predicted_direction = (
            predicted_return >
            0
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


        # =================================================
        # ACTIONABLE DIRECTION ACCURACY
        #
        # Evaluate forecasts where predicted move
        # magnitude >= 0.25%.
        #
        # We ALSO show coverage so this cannot
        # misleadingly hide ignored predictions.
        # =================================================

        actionable_threshold = (
            0.0025
        )


        actionable_mask = (
            np.abs(
                predicted_return
            )
            >=
            actionable_threshold
        )


        actionable_count = int(
            np.sum(
                actionable_mask
            )
        )


        total_count = len(
            predicted_return
        )


        actionable_coverage = (
            (
                actionable_count
                /
                total_count
            )
            *
            100

            if total_count >
            0

            else 0.0
        )


        if actionable_count > 0:

            actionable_direction_accuracy = (
                np.mean(
                    actual_direction[
                        actionable_mask
                    ]
                    ==
                    predicted_direction[
                        actionable_mask
                    ]
                )
                *
                100
            )

        else:

            actionable_direction_accuracy = (
                0.0
            )


        # =================================================
        # NAIVE BASELINE
        #
        # Tomorrow/future price = current price.
        # =================================================

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


        baseline_rmse = (
            np.sqrt(
                np.mean(
                    baseline_error ** 2
                )
            )
        )


        improvement_vs_baseline = (
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


        # =================================================
        # CORRELATION
        # =================================================

        return_correlation = (
            safe_correlation(
                actual_return,
                predicted_return,
            )
        )


        # =================================================
        # PREDICTION BIAS
        # =================================================

        predicted_mean_return = (
            np.mean(
                predicted_return
            )
            *
            100
        )


        actual_mean_return = (
            np.mean(
                actual_return
            )
            *
            100
        )


        # =================================================
        # RESULTS
        # =================================================

        results[
            f"{horizon}D"
        ] = {

            "return_mae_percent":
                round(
                    float(
                        return_mae
                    ),
                    4,
                ),

            "return_rmse_percent":
                round(
                    float(
                        return_rmse
                    ),
                    4,
                ),

            "price_mae":
                round(
                    float(
                        price_mae
                    ),
                    2,
                ),

            "price_rmse":
                round(
                    float(
                        price_rmse
                    ),
                    2,
                ),

            "mape_percent":
                round(
                    float(
                        mape
                    ),
                    4,
                ),

            "r2":
                round(
                    float(
                        r2
                    ),
                    4,
                ),

            "direction_accuracy_percent":
                round(
                    float(
                        direction_accuracy
                    ),
                    2,
                ),

            "actionable_direction_accuracy_percent":
                round(
                    float(
                        actionable_direction_accuracy
                    ),
                    2,
                ),

            "actionable_coverage_percent":
                round(
                    float(
                        actionable_coverage
                    ),
                    2,
                ),

            "baseline_mae":
                round(
                    float(
                        baseline_mae
                    ),
                    2,
                ),

            "baseline_rmse":
                round(
                    float(
                        baseline_rmse
                    ),
                    2,
                ),

            "improvement_vs_baseline_percent":
                round(
                    float(
                        improvement_vs_baseline
                    ),
                    2,
                ),

            "beats_baseline":
                bool(
                    price_mae
                    <
                    baseline_mae
                ),

            "return_correlation":
                round(
                    float(
                        return_correlation
                    ),
                    4,
                ),

            "predicted_mean_return_percent":
                round(
                    float(
                        predicted_mean_return
                    ),
                    4,
                ),

            "actual_mean_return_percent":
                round(
                    float(
                        actual_mean_return
                    ),
                    4,
                ),
        }


    return results


# =========================================================
# CALIBRATION
# =========================================================

def calculate_calibration(
    actual_returns,
    predicted_returns,
):

    calibration = {}


    absolute_errors = (
        np.abs(
            predicted_returns
            -
            actual_returns
        )
        *
        100
    )


    for (
        index,
        horizon,

    ) in enumerate(
        HORIZONS
    ):


        errors = (
            absolute_errors[
                :,
                index
            ]
        )


        calibration[
            f"{horizon}D"
        ] = {

            "median_absolute_error_percent":
                round(
                    float(
                        np.median(
                            errors
                        )
                    ),
                    4,
                ),

            "q80_absolute_error_percent":
                round(
                    float(
                        np.quantile(
                            errors,
                            0.80,
                        )
                    ),
                    4,
                ),

            "q90_absolute_error_percent":
                round(
                    float(
                        np.quantile(
                            errors,
                            0.90,
                        )
                    ),
                    4,
                ),
        }


    return calibration


# =========================================================
# LOAD V3 REFERENCE METRICS
# =========================================================

def load_v3_metrics(
    symbol,
):

    path = (
        get_v3_metrics_path(
            symbol
        )
    )


    if not os.path.exists(
        path
    ):

        return None


    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            data = (
                json.load(
                    file
                )
            )


        return data.get(
            "metrics"
        )


    except Exception:

        return None


# =========================================================
# PRINT V3 VS V4
# =========================================================

def print_v3_comparison(
    v4_metrics,
    v3_metrics,
):

    if not v3_metrics:

        print(
            "\nNo V3 metrics file found. "
            "Skipping V3 reference comparison."
        )

        return


    print(
        "\n=============================================="
    )

    print(
        " V4 vs V3 REFERENCE COMPARISON"
    )

    print(
        "=============================================="
    )


    print(
        "\nNote: V4 uses more features and therefore "
        "its usable date range differs slightly from V3."
    )

    print(
        "The naive baseline comparison remains the "
        "main fair test within the V4 test sample."
    )


    for horizon in HORIZONS:

        key = (
            f"{horizon}D"
        )


        v4 = (
            v4_metrics.get(
                key,
                {}
            )
        )


        v3 = (
            v3_metrics.get(
                key,
                {}
            )
        )


        if not v3:

            continue


        print(
            f"\n---------- {key} ----------"
        )


        print(
            "V3 MAPE:",
            f'{v3.get("mape_percent", 0):.4f}%'
        )


        print(
            "V4 MAPE:",
            f'{v4.get("mape_percent", 0):.4f}%'
        )


        print(
            "V3 Direction:",
            f'{v3.get("direction_accuracy_percent", 0):.2f}%'
        )


        print(
            "V4 Direction:",
            f'{v4.get("direction_accuracy_percent", 0):.2f}%'
        )


        print(
            "V3 Beats Baseline:",
            v3.get(
                "beats_baseline"
            )
        )


        print(
            "V4 Beats Baseline:",
            v4.get(
                "beats_baseline"
            )
        )


# =========================================================
# TRAIN V4
# =========================================================

def train_v4_model(symbol):

    symbol = (
        normalize_symbol(
            symbol
        )
    )


    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION BiLSTM V4"
    )

    print(
        "=============================================="
    )


    print(
        f"\nStock: {symbol}"
    )


    print(
        "Forecast Horizons:",
        HORIZONS
    )


    print(
        "Feature Count:",
        len(
            FEATURE_COLUMNS
        )
    )


    print(
        "Lookback:",
        SEQUENCE_LENGTH,
        "trading days"
    )


    # =====================================================
    # DATA
    # =====================================================

    (
        data,
        sequences,
        feature_scaler,
        target_scaler,

    ) = prepare_v4_data(
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
        "Validation y:",
        y_validation.shape
    )


    print(
        "Test X:",
        X_test.shape
    )


    if (
        len(
            X_train
        ) == 0
        or
        len(
            X_validation
        ) == 0
        or
        len(
            X_test
        ) == 0
    ):

        raise ValueError(
            "V4 train/validation/test sequence creation failed."
        )


    # =====================================================
    # PATHS
    # =====================================================

    (
        model_path,
        feature_scaler_path,
        target_scaler_path,
        calibration_path,
        metrics_path,

    ) = get_v4_model_paths(
        symbol
    )


    # =====================================================
    # MODEL
    # =====================================================

    model = (
        build_v4_model()
    )


    print(
        "\nV4 Architecture:"
    )


    model.summary()


    # =====================================================
    # CALLBACKS
    # =====================================================

    callbacks = [

        EarlyStopping(

            monitor=
                "val_loss",

            patience=
                12,

            restore_best_weights=
                True,

            verbose=
                1,
        ),


        ReduceLROnPlateau(

            monitor=
                "val_loss",

            factor=
                0.5,

            patience=
                5,

            min_lr=
                0.000005,

            verbose=
                1,
        ),


        ModelCheckpoint(

            filepath=
                model_path,

            monitor=
                "val_loss",

            save_best_only=
                True,

            verbose=
                1,
        ),
    ]


    # =====================================================
    # TRAIN
    # =====================================================

    print(
        "\nStarting V4 training...\n"
    )


    model.fit(

        X_train,

        y_train,

        validation_data=(
            X_validation,
            y_validation,
        ),

        epochs=
            80,

        batch_size=
            32,

        shuffle=
            False,

        callbacks=
            callbacks,

        verbose=
            1,
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
    # VALIDATION
    # =====================================================

    validation_prediction_scaled = (
        model.predict(
            X_validation,
            verbose=0
        )
    )


    validation_prediction = (
        target_scaler.inverse_transform(
            validation_prediction_scaled
        )
    )


    validation_actual = (
        sequences[
            "y_validation_original"
        ]
    )


    calibration = (
        calculate_calibration(

            actual_returns=
                validation_actual,

            predicted_returns=
                validation_prediction,
        )
    )


    # =====================================================
    # TEST
    # =====================================================

    test_prediction_scaled = (
        model.predict(
            X_test,
            verbose=0
        )
    )


    test_prediction = (
        target_scaler.inverse_transform(
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


    metrics = (
        calculate_metrics(

            actual_returns=
                test_actual,

            predicted_returns=
                test_prediction,

            current_close=
                test_close,
        )
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
        encoding="utf-8",
    ) as file:

        json.dump(
            calibration,
            file,
            indent=4,
        )


    # =====================================================
    # SAVE METRICS
    # =====================================================

    metrics_payload = {

        "symbol":
            symbol,

        "model":
            "StockVision BiLSTM V4",

        "model_version":
            "V4",

        "lookback_days":
            SEQUENCE_LENGTH,

        "features_used":
            len(
                FEATURE_COLUMNS
            ),

        "horizons":
            HORIZONS,

        "rows":
            len(
                data
            ),

        "feature_columns":
            FEATURE_COLUMNS,

        "metrics":
            metrics,
    }


    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_payload,
            file,
            indent=4,
        )


    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V4 TEST RESULTS"
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
            f"\n---------- {key} ----------"
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
            result[
                "r2"
            ]
        )


        print(
            "Direction Accuracy:",
            f'{result["direction_accuracy_percent"]:.2f}%'
        )


        print(
            "Actionable Direction Accuracy:",
            f'{result["actionable_direction_accuracy_percent"]:.2f}%'
        )


        print(
            "Actionable Coverage:",
            f'{result["actionable_coverage_percent"]:.2f}%'
        )


        print(
            "Return Correlation:",
            result[
                "return_correlation"
            ]
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
            result[
                "beats_baseline"
            ]
        )


    # =====================================================
    # CALIBRATION
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " V4 VALIDATION ERROR RANGES"
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
            "Median Error:",
            f'{values["median_absolute_error_percent"]:.4f}%'
        )


        print(
            "80% Error Band:",
            f'±{values["q80_absolute_error_percent"]:.4f}%'
        )


        print(
            "90% Error Band:",
            f'±{values["q90_absolute_error_percent"]:.4f}%'
        )


    # =====================================================
    # V3 REFERENCE
    # =====================================================

    v3_metrics = (
        load_v3_metrics(
            symbol
        )
    )


    print_v3_comparison(
        v4_metrics=
            metrics,

        v3_metrics=
            v3_metrics,
    )


    # =====================================================
    # SAVED FILES
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " V4 FILES SAVED"
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
        "\nFeature Scaler:"
    )

    print(
        feature_scaler_path
    )


    print(
        "\nTarget Scaler:"
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
        "\nV4 training complete."
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

        else
        "RELIANCE.NS"
    )


    train_v4_model(
        symbol
    )