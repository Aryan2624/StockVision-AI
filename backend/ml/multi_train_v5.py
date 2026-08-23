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
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)


# =========================================================
# STOCKVISION IMPORTS
# =========================================================

try:

    from .data import (
        get_historical_data,
    )

    from .context_features_v5 import (
        prepare_training_frame_v5,
        FEATURE_COLUMNS,
        RETURN_TARGET_COLUMNS,
        DIRECTION_TARGET_COLUMNS,
        HORIZONS,
        SEQUENCE_LENGTH,
    )

except ImportError:

    from data import (
        get_historical_data,
    )

    from context_features_v5 import (
        prepare_training_frame_v5,
        FEATURE_COLUMNS,
        RETURN_TARGET_COLUMNS,
        DIRECTION_TARGET_COLUMNS,
        HORIZONS,
        SEQUENCE_LENGTH,
    )


# =========================================================
# RANDOM SEEDS
# =========================================================

np.random.seed(42)

tf.random.set_seed(42)


# =========================================================
# CONFIG
# =========================================================

MAX_HORIZON = max(
    HORIZONS
)


BASE_DIR = os.path.dirname(
    os.path.abspath(
        __file__
    )
)


MODELS_DIR = os.path.join(
    BASE_DIR,
    "models",
)


os.makedirs(
    MODELS_DIR,
    exist_ok=True,
)


# =========================================================
# SYMBOL HELPERS
# =========================================================

def normalize_symbol(
    symbol,
):

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


def safe_symbol_name(
    symbol,
):

    return (
        symbol
        .upper()
        .replace(
            ".",
            "_",
        )
        .replace(
            "^",
            "",
        )
    )


# =========================================================
# V5 FILE PATHS
# =========================================================

def get_v5_paths(
    symbol,
):

    safe_symbol = (
        safe_symbol_name(
            symbol
        )
    )


    return {

        "model":
            os.path.join(
                MODELS_DIR,
                f"{safe_symbol}_v5.keras",
            ),


        "feature_scaler":
            os.path.join(
                MODELS_DIR,
                f"{safe_symbol}_v5_feature_scaler.pkl",
            ),


        "target_scaler":
            os.path.join(
                MODELS_DIR,
                f"{safe_symbol}_v5_target_scaler.pkl",
            ),


        "calibration":
            os.path.join(
                MODELS_DIR,
                f"{safe_symbol}_v5_calibration.json",
            ),


        "metrics":
            os.path.join(
                MODELS_DIR,
                f"{safe_symbol}_v5_metrics.json",
            ),
    }


# =========================================================
# BUILD SEQUENCES
# =========================================================

def build_samples(
    scaled_features,
    scaled_returns,
    original_returns,
    direction_targets,
    close_prices,
    positions,
):

    X = []

    y_return_scaled = []

    y_return_original = []

    y_direction = []

    closes = []


    for position in positions:

        start = (
            position
            -
            SEQUENCE_LENGTH
            +
            1
        )


        if start < 0:

            continue


        sequence = (
            scaled_features[
                start:
                position + 1
            ]
        )


        if (
            len(
                sequence
            )
            !=
            SEQUENCE_LENGTH
        ):

            continue


        X.append(
            sequence
        )


        y_return_scaled.append(
            scaled_returns[
                position
            ]
        )


        y_return_original.append(
            original_returns[
                position
            ]
        )


        y_direction.append(
            direction_targets[
                position
            ]
        )


        closes.append(
            close_prices[
                position
            ]
        )


    return {

        "X":
            np.asarray(
                X,
                dtype=np.float32,
            ),


        "y_return_scaled":
            np.asarray(
                y_return_scaled,
                dtype=np.float32,
            ),


        "y_return_original":
            np.asarray(
                y_return_original,
                dtype=np.float32,
            ),


        "y_direction":
            np.asarray(
                y_direction,
                dtype=np.float32,
            ),


        "close":
            np.asarray(
                closes,
                dtype=np.float32,
            ),
    }


# =========================================================
# PREPARE V5 DATA
# =========================================================

def prepare_v5_data(
    symbol,
):

    print(
        "\nDownloading stock data..."
    )


    stock_data = (
        get_historical_data(
            symbol,
            period="10y",
        )
    )


    print(
        "Downloading NIFTY 50 context..."
    )


    nifty_data = (
        get_historical_data(
            "^NSEI",
            period="10y",
        )
    )


    data = (
        prepare_training_frame_v5(
            stock_data,
            nifty_data,
        )
        .sort_index()
    )


    if len(data) < 500:

        raise ValueError(
            f"Not enough usable data for {symbol}. "
            f"Found {len(data)} rows."
        )


    # =====================================================
    # ARRAYS
    # =====================================================

    feature_values = (
        data[
            FEATURE_COLUMNS
        ]
        .values
        .astype(
            np.float32
        )
    )


    return_values = (
        data[
            RETURN_TARGET_COLUMNS
        ]
        .values
        .astype(
            np.float32
        )
    )


    direction_values = (
        data[
            DIRECTION_TARGET_COLUMNS
        ]
        .values
        .astype(
            np.float32
        )
    )


    close_values = (
        data[
            "Close"
        ]
        .values
        .astype(
            np.float32
        )
    )


    total_rows = (
        len(
            data
        )
    )


    # =====================================================
    # CHRONOLOGICAL SPLIT
    #
    # 70% train
    # 15% validation
    # 15% test
    # =====================================================

    train_boundary = int(
        total_rows
        *
        0.70
    )


    validation_boundary = int(
        total_rows
        *
        0.85
    )


    # =====================================================
    # PURGE TARGET LEAKAGE
    #
    # Last 10 training targets can contain prices
    # from validation period because RETURN_10D
    # looks 10 trading days ahead.
    #
    # We therefore remove MAX_HORIZON rows before
    # each evaluation boundary.
    # =====================================================

    train_positions = (
        np.arange(
            SEQUENCE_LENGTH - 1,
            train_boundary
            -
            MAX_HORIZON,
        )
    )


    validation_positions = (
        np.arange(
            train_boundary,
            validation_boundary
            -
            MAX_HORIZON,
        )
    )


    test_positions = (
        np.arange(
            validation_boundary,
            total_rows,
        )
    )


    if (
        len(
            train_positions
        ) < 500
        or
        len(
            validation_positions
        ) < 50
        or
        len(
            test_positions
        ) < 30
    ):

        raise ValueError(
            "Insufficient samples after V5 purged split."
        )


    # =====================================================
    # SCALERS
    #
    # Fit only using historical training information.
    # =====================================================

    feature_scaler = (
        StandardScaler()
    )


    target_scaler = (
        StandardScaler()
    )


    feature_scaler.fit(
        feature_values[
            :train_boundary
        ]
    )


    target_scaler.fit(
        return_values[
            train_positions
        ]
    )


    scaled_features = (
        feature_scaler
        .transform(
            feature_values
        )
        .astype(
            np.float32
        )
    )


    scaled_returns = (
        target_scaler
        .transform(
            return_values
        )
        .astype(
            np.float32
        )
    )


    # =====================================================
    # SAMPLE SETS
    # =====================================================

    train_set = (
        build_samples(

            scaled_features,
            scaled_returns,
            return_values,
            direction_values,
            close_values,
            train_positions,
        )
    )


    validation_set = (
        build_samples(

            scaled_features,
            scaled_returns,
            return_values,
            direction_values,
            close_values,
            validation_positions,
        )
    )


    test_set = (
        build_samples(

            scaled_features,
            scaled_returns,
            return_values,
            direction_values,
            close_values,
            test_positions,
        )
    )


    return (

        data,

        train_set,

        validation_set,

        test_set,

        feature_scaler,

        target_scaler,
    )


# =========================================================
# BUILD V5 MULTI-TASK BiLSTM
# =========================================================

def build_v5_model():

    inputs = Input(

        shape=(

            SEQUENCE_LENGTH,

            len(
                FEATURE_COLUMNS
            ),
        ),

        name=
            "market_sequence",
    )


    # =====================================================
    # BiLSTM BLOCK 1
    # =====================================================

    x = Bidirectional(

        LSTM(

            64,

            return_sequences=
                True,
        ),

        name=
            "bilstm_1",

    )(
        inputs
    )


    x = LayerNormalization(

        name=
            "layer_norm_1",

    )(
        x
    )


    x = Dropout(

        0.25,

        name=
            "dropout_1",

    )(
        x
    )


    # =====================================================
    # BiLSTM BLOCK 2
    # =====================================================

    x = Bidirectional(

        LSTM(

            32,

            return_sequences=
                False,
        ),

        name=
            "bilstm_2",

    )(
        x
    )


    x = LayerNormalization(

        name=
            "layer_norm_2",

    )(
        x
    )


    x = Dropout(

        0.25,

        name=
            "dropout_2",

    )(
        x
    )


    # =====================================================
    # SHARED REPRESENTATION
    # =====================================================

    shared = Dense(

        48,

        activation=
            "relu",

        name=
            "shared_dense",

    )(
        x
    )


    shared = Dropout(

        0.15,

        name=
            "shared_dropout",

    )(
        shared
    )


    # =====================================================
    # RETURN REGRESSION HEAD
    # =====================================================

    regression = Dense(

        32,

        activation=
            "relu",

        name=
            "regression_dense",

    )(
        shared
    )


    regression = Dropout(

        0.10,

        name=
            "regression_dropout",

    )(
        regression
    )


    return_output = Dense(

        len(
            HORIZONS
        ),

        activation=
            "linear",

        name=
            "return_output",

    )(
        regression
    )


    # =====================================================
    # DIRECTION CLASSIFICATION HEAD
    # =====================================================

    classification = Dense(

        32,

        activation=
            "relu",

        name=
            "direction_dense",

    )(
        shared
    )


    classification = Dropout(

        0.10,

        name=
            "direction_dropout",

    )(
        classification
    )


    direction_output = Dense(

        len(
            HORIZONS
        ),

        activation=
            "sigmoid",

        name=
            "direction_output",

    )(
        classification
    )


    # =====================================================
    # MODEL
    # =====================================================

    model = Model(

        inputs=
            inputs,

        outputs=[
            return_output,
            direction_output,
        ],

        name=
            "StockVision_V5_MultiTask_BiLSTM",
    )


    optimizer = (
        tf.keras.optimizers.Adam(

            learning_rate=
                0.0004,

            clipnorm=
                1.0,
        )
    )


    # =====================================================
    # MULTI-TASK LOSS
    #
    # Regression remains primary.
    # Direction classification contributes 40%.
    # =====================================================

    model.compile(

        optimizer=
            optimizer,


        loss={

            "return_output":
                tf.keras.losses.Huber(),


            "direction_output":
                tf.keras.losses.BinaryCrossentropy(),
        },


        loss_weights={

            "return_output":
                1.0,

            "direction_output":
                0.40,
        },


        metrics={

            "return_output": [

                tf.keras.metrics.MeanAbsoluteError(
                    name="mae"
                ),
            ],


            "direction_output": [

                tf.keras.metrics.BinaryAccuracy(
                    name="accuracy"
                ),
            ],
        },
    )


    return model


# =========================================================
# SAFE CORRELATION
# =========================================================

def safe_correlation(
    actual,
    predicted,
):

    if (
        len(
            actual
        ) < 2
        or
        np.std(
            actual
        ) == 0
        or
        np.std(
            predicted
        ) == 0
    ):

        return 0.0


    correlation = (
        np.corrcoef(
            actual,
            predicted,
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


# =========================================================
# TEST METRICS
# =========================================================

def calculate_metrics(
    actual_returns,
    predicted_returns,
    actual_directions,
    predicted_direction_probabilities,
    current_close,
    train_directions,
):

    metrics = {}


    for (
        index,
        horizon,

    ) in enumerate(
        HORIZONS
    ):


        # =================================================
        # VALUES
        # =================================================

        actual_return = (
            actual_returns[
                :,
                index
            ]
        )


        predicted_return = (
            predicted_returns[
                :,
                index
            ]
        )


        actual_direction = (
            actual_directions[
                :,
                index
            ]
            .astype(
                int
            )
        )


        direction_probability = (
            predicted_direction_probabilities[
                :,
                index
            ]
        )


        classifier_direction = (
            direction_probability
            >=
            0.50
        ).astype(
            int
        )


        regression_direction = (
            predicted_return
            >
            0
        ).astype(
            int
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
        # RETURN ERROR
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
                    return_error
                    **
                    2
                )
            )

            *
            100
        )


        # =================================================
        # PRICE ERROR
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
                    price_error
                    **
                    2
                )
            )
        )


        # =================================================
        # NAIVE PRICE BASELINE
        #
        # Future price = current price
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
                    baseline_error
                    **
                    2
                )
            )
        )


        if baseline_mae > 0:

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
            )

        else:

            improvement_vs_baseline = (
                0.0
            )


        # =================================================
        # MAPE
        # =================================================

        valid_price = (
            actual_price
            !=
            0
        )


        if np.any(
            valid_price
        ):

            mape = (

                np.mean(

                    np.abs(

                        (

                            actual_price[
                                valid_price
                            ]

                            -

                            predicted_price[
                                valid_price
                            ]

                        )

                        /

                        actual_price[
                            valid_price
                        ]
                    )
                )

                *
                100
            )

        else:

            mape = 0.0


        # =================================================
        # R²
        # =================================================

        ss_res = (
            np.sum(

                (
                    actual_price
                    -
                    predicted_price
                )

                **
                2
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

                **
                2
            )
        )


        if ss_total > 0:

            r2 = (

                1

                -

                (
                    ss_res
                    /
                    ss_total
                )
            )

        else:

            r2 = 0.0


        # =================================================
        # REGRESSION DIRECTION ACCURACY
        #
        # Direction inferred from predicted return sign.
        # =================================================

        regression_direction_accuracy = (

            np.mean(

                regression_direction

                ==

                actual_direction
            )

            *
            100
        )


        # =================================================
        # CLASSIFIER DIRECTION ACCURACY
        #
        # Dedicated sigmoid classification head.
        # =================================================

        classifier_accuracy = (

            np.mean(

                classifier_direction

                ==

                actual_direction
            )

            *
            100
        )


        # =================================================
        # DIRECTION BASELINE
        #
        # Always predict majority training class.
        # =================================================

        train_up_rate = (

            np.mean(

                train_directions[
                    :,
                    index
                ]
            )
        )


        majority_class = (

            1

            if train_up_rate >= 0.5

            else 0
        )


        majority_baseline_accuracy = (

            np.mean(

                actual_direction

                ==

                majority_class
            )

            *
            100
        )


        classifier_improvement_pp = (

            classifier_accuracy

            -

            majority_baseline_accuracy
        )


        # =================================================
        # HIGH-CONFIDENCE CLASSIFIER SIGNAL
        #
        # Probability >= 60%
        # or probability <= 40%
        # =================================================

        confidence = (

            np.abs(

                direction_probability

                -
                0.5
            )

            *
            2
        )


        confident_mask = (
            confidence
            >=
            0.20
        )


        confident_count = int(

            np.sum(
                confident_mask
            )
        )


        if len(
            actual_direction
        ) > 0:

            confidence_coverage = (

                confident_count

                /

                len(
                    actual_direction
                )

                *
                100
            )

        else:

            confidence_coverage = (
                0.0
            )


        if confident_count > 0:

            confident_accuracy = (

                np.mean(

                    classifier_direction[
                        confident_mask
                    ]

                    ==

                    actual_direction[
                        confident_mask
                    ]
                )

                *
                100
            )

        else:

            confident_accuracy = (
                0.0
            )


        # =================================================
        # BRIER SCORE
        #
        # Lower is better.
        #
        # Measures quality of probability forecasts.
        # =================================================

        brier_score = (

            np.mean(

                (
                    direction_probability

                    -

                    actual_direction
                )

                **
                2
            )
        )


        # =================================================
        # RETURN CORRELATION
        # =================================================

        correlation = (
            safe_correlation(
                actual_return,
                predicted_return,
            )
        )


        # =================================================
        # RESULT
        # =================================================

        metrics[
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


            "return_correlation":
                round(
                    float(
                        correlation
                    ),
                    4,
                ),


            "regression_direction_accuracy_percent":
                round(
                    float(
                        regression_direction_accuracy
                    ),
                    2,
                ),


            "classifier_direction_accuracy_percent":
                round(
                    float(
                        classifier_accuracy
                    ),
                    2,
                ),


            "majority_direction_baseline_percent":
                round(
                    float(
                        majority_baseline_accuracy
                    ),
                    2,
                ),


            "classifier_improvement_vs_direction_baseline_pp":
                round(
                    float(
                        classifier_improvement_pp
                    ),
                    2,
                ),


            "high_confidence_direction_accuracy_percent":
                round(
                    float(
                        confident_accuracy
                    ),
                    2,
                ),


            "high_confidence_coverage_percent":
                round(
                    float(
                        confidence_coverage
                    ),
                    2,
                ),


            "brier_score":
                round(
                    float(
                        brier_score
                    ),
                    4,
                ),


            "naive_price_baseline_mae":
                round(
                    float(
                        baseline_mae
                    ),
                    2,
                ),


            "naive_price_baseline_rmse":
                round(
                    float(
                        baseline_rmse
                    ),
                    2,
                ),


            "improvement_vs_price_baseline_percent":
                round(
                    float(
                        improvement_vs_baseline
                    ),
                    2,
                ),


            "beats_price_baseline":
                bool(
                    price_mae
                    <
                    baseline_mae
                ),


            "actual_up_rate_percent":
                round(
                    float(

                        np.mean(
                            actual_direction
                        )

                        *
                        100
                    ),
                    2,
                ),


            "predicted_up_probability_mean_percent":
                round(
                    float(

                        np.mean(
                            direction_probability
                        )

                        *
                        100
                    ),
                    2,
                ),
        }


    return metrics


# =========================================================
# VALIDATION ERROR CALIBRATION
# =========================================================

def calculate_calibration(
    actual_returns,
    predicted_returns,
):

    errors = (

        np.abs(

            predicted_returns

            -

            actual_returns
        )

        *
        100
    )


    calibration = {}


    for (
        index,
        horizon,

    ) in enumerate(
        HORIZONS
    ):


        horizon_errors = (
            errors[
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
                            horizon_errors
                        )
                    ),
                    4,
                ),


            "q80_absolute_error_percent":
                round(
                    float(
                        np.quantile(
                            horizon_errors,
                            0.80,
                        )
                    ),
                    4,
                ),


            "q90_absolute_error_percent":
                round(
                    float(
                        np.quantile(
                            horizon_errors,
                            0.90,
                        )
                    ),
                    4,
                ),
        }


    return calibration


# =========================================================
# PRINT TEST METRICS
# =========================================================

def print_metrics(
    metrics,
):

    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V5 TEST RESULTS"
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
            "Price MAE:",
            f'₹{result["price_mae"]:.2f}'
        )


        print(
            "MAPE:",
            f'{result["mape_percent"]:.4f}%'
        )


        print(
            "Return Correlation:",
            result[
                "return_correlation"
            ]
        )


        print(
            "Regression Direction Accuracy:",
            f'{result["regression_direction_accuracy_percent"]:.2f}%'
        )


        print(
            "Classifier Direction Accuracy:",
            f'{result["classifier_direction_accuracy_percent"]:.2f}%'
        )


        print(
            "Direction Baseline:",
            f'{result["majority_direction_baseline_percent"]:.2f}%'
        )


        print(
            "Classifier Improvement:",
            f'{result["classifier_improvement_vs_direction_baseline_pp"]:+.2f} pp'
        )


        print(
            "High-Confidence Direction Accuracy:",
            f'{result["high_confidence_direction_accuracy_percent"]:.2f}%'
        )


        print(
            "High-Confidence Coverage:",
            f'{result["high_confidence_coverage_percent"]:.2f}%'
        )


        print(
            "Brier Score:",
            result[
                "brier_score"
            ]
        )


        print(
            "Naive Price Baseline MAE:",
            f'₹{result["naive_price_baseline_mae"]:.2f}'
        )


        print(
            "Price Baseline Improvement:",
            f'{result["improvement_vs_price_baseline_percent"]:.2f}%'
        )


        print(
            "Beats Price Baseline:",
            result[
                "beats_price_baseline"
            ]
        )


# =========================================================
# TRAIN V5
# =========================================================

def train_v5_model(
    symbol,
):

    symbol = (
        normalize_symbol(
            symbol
        )
    )


    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V5 MULTI-TASK BiLSTM"
    )

    print(
        "=============================================="
    )


    print(
        f"\nStock: {symbol}"
    )


    print(
        "Market Context: ^NSEI"
    )


    print(
        "Features:",
        len(
            FEATURE_COLUMNS
        )
    )


    print(
        "Horizons:",
        HORIZONS
    )


    print(
        "Lookback:",
        SEQUENCE_LENGTH
    )


    # =====================================================
    # DATA
    # =====================================================

    (
        data,
        train_set,
        validation_set,
        test_set,
        feature_scaler,
        target_scaler,

    ) = prepare_v5_data(
        symbol
    )


    print(
        "\nDataset:"
    )


    print(
        "Usable Rows:",
        len(
            data
        )
    )


    print(
        "Train:",
        train_set[
            "X"
        ].shape
    )


    print(
        "Validation:",
        validation_set[
            "X"
        ].shape
    )


    print(
        "Test:",
        test_set[
            "X"
        ].shape
    )


    # =====================================================
    # PATHS
    # =====================================================

    paths = (
        get_v5_paths(
            symbol
        )
    )


    # =====================================================
    # MODEL
    # =====================================================

    tf.keras.backend.clear_session()


    model = (
        build_v5_model()
    )


    print(
        "\nV5 Architecture:"
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
                10,

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
                4,

            min_lr=
                0.000005,

            verbose=
                1,
        ),


        ModelCheckpoint(

            filepath=
                paths[
                    "model"
                ],

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
        "\nStarting V5 training...\n"
    )


    history = (
        model.fit(

            train_set[
                "X"
            ],


            {

                "return_output":
                    train_set[
                        "y_return_scaled"
                    ],


                "direction_output":
                    train_set[
                        "y_direction"
                    ],
            },


            validation_data=(

                validation_set[
                    "X"
                ],


                {

                    "return_output":
                        validation_set[
                            "y_return_scaled"
                        ],


                    "direction_output":
                        validation_set[
                            "y_direction"
                        ],
                },
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
    )


    best_epoch = int(

        np.argmin(

            history.history[
                "val_loss"
            ]
        )

        +
        1
    )


    # =====================================================
    # LOAD BEST SAVED MODEL
    # =====================================================

    model = (
        tf.keras.models.load_model(

            paths[
                "model"
            ]
        )
    )


    # =====================================================
    # VALIDATION PREDICTION
    # =====================================================

    validation_predictions = (
        model.predict(

            validation_set[
                "X"
            ],

            verbose=0,
        )
    )


    validation_return_scaled = (
        validation_predictions[
            0
        ]
    )


    validation_return_prediction = (
        target_scaler
        .inverse_transform(
            validation_return_scaled
        )
    )


    calibration = (
        calculate_calibration(

            validation_set[
                "y_return_original"
            ],

            validation_return_prediction,
        )
    )


    # =====================================================
    # TEST PREDICTION
    # =====================================================

    test_predictions = (
        model.predict(

            test_set[
                "X"
            ],

            verbose=0,
        )
    )


    test_return_scaled = (
        test_predictions[
            0
        ]
    )


    test_direction_probability = (
        test_predictions[
            1
        ]
    )


    test_return_prediction = (
        target_scaler
        .inverse_transform(
            test_return_scaled
        )
    )


    # =====================================================
    # METRICS
    # =====================================================

    metrics = (
        calculate_metrics(

            actual_returns=
                test_set[
                    "y_return_original"
                ],


            predicted_returns=
                test_return_prediction,


            actual_directions=
                test_set[
                    "y_direction"
                ],


            predicted_direction_probabilities=
                test_direction_probability,


            current_close=
                test_set[
                    "close"
                ],


            train_directions=
                train_set[
                    "y_direction"
                ],
        )
    )


    # =====================================================
    # SAVE SCALERS
    # =====================================================

    joblib.dump(

        feature_scaler,

        paths[
            "feature_scaler"
        ],
    )


    joblib.dump(

        target_scaler,

        paths[
            "target_scaler"
        ],
    )


    # =====================================================
    # SAVE CALIBRATION
    # =====================================================

    with open(

        paths[
            "calibration"
        ],

        "w",

        encoding=
            "utf-8",

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
            "StockVision V5 Multi-Task BiLSTM",


        "market_context":
            "^NSEI",


        "features_used":
            len(
                FEATURE_COLUMNS
            ),


        "feature_columns":
            FEATURE_COLUMNS,


        "lookback_days":
            SEQUENCE_LENGTH,


        "horizons":
            HORIZONS,


        "max_horizon_purge_days":
            MAX_HORIZON,


        "best_epoch":
            best_epoch,


        "train_samples":
            int(
                len(
                    train_set[
                        "X"
                    ]
                )
            ),


        "validation_samples":
            int(
                len(
                    validation_set[
                        "X"
                    ]
                )
            ),


        "test_samples":
            int(
                len(
                    test_set[
                        "X"
                    ]
                )
            ),


        "metrics":
            metrics,
    }


    with open(

        paths[
            "metrics"
        ],

        "w",

        encoding=
            "utf-8",

    ) as file:

        json.dump(

            metrics_payload,

            file,

            indent=4,
        )


    # =====================================================
    # PRINT TEST RESULTS
    # =====================================================

    print_metrics(
        metrics
    )


    # =====================================================
    # PRINT CALIBRATION
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " V5 VALIDATION ERROR RANGES"
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
    # FILES
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " V5 FILES SAVED"
    )

    print(
        "=============================================="
    )


    print(
        "\nModel:"
    )

    print(
        paths[
            "model"
        ]
    )


    print(
        "\nFeature Scaler:"
    )

    print(
        paths[
            "feature_scaler"
        ]
    )


    print(
        "\nTarget Scaler:"
    )

    print(
        paths[
            "target_scaler"
        ]
    )


    print(
        "\nCalibration:"
    )

    print(
        paths[
            "calibration"
        ]
    )


    print(
        "\nMetrics:"
    )

    print(
        paths[
            "metrics"
        ]
    )


    print(
        "\nV5 training complete."
    )


# =========================================================
# COMMAND LINE
# =========================================================

if __name__ == "__main__":

    symbol = (

        sys.argv[
            1
        ]

        if len(
            sys.argv
        ) > 1

        else
        "RELIANCE.NS"
    )


    train_v5_model(
        symbol
    )