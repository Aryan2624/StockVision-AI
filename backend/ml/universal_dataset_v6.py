import os
import json
import random

import joblib
import numpy as np
import pandas as pd
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
# IMPORT V6 CONFIG
# =========================================================

try:

    from .universal_dataset_v6 import (
        FEATURE_COLUMNS,
        TARGET_HORIZON,
        CLASS_NAMES,
        CSV_PATH,
    )

except ImportError:

    from universal_dataset_v6 import (
        FEATURE_COLUMNS,
        TARGET_HORIZON,
        CLASS_NAMES,
        CSV_PATH,
    )


# =========================================================
# CONFIG
# =========================================================

SEQUENCE_LENGTH = 60

NUM_CLASSES = 3

RANDOM_SEED = 42


np.random.seed(
    RANDOM_SEED
)

random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODELS_DIR = os.path.join(
    BASE_DIR,
    "models",
)


os.makedirs(
    MODELS_DIR,
    exist_ok=True,
)


MODEL_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_universal_v6.keras",
)


SCALER_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_universal_v6_scaler.pkl",
)


METRICS_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_universal_v6_metrics.json",
)


METADATA_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_universal_v6_metadata.json",
)


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    if not os.path.exists(
        CSV_PATH
    ):

        raise FileNotFoundError(
            f"V6 dataset not found:\n{CSV_PATH}"
        )


    data = pd.read_csv(
        CSV_PATH
    )


    data["Date"] = pd.to_datetime(
        data["Date"]
    )


    data.sort_values(
        by=[
            "SYMBOL",
            "Date",
        ],
        inplace=True,
    )


    data.reset_index(
        drop=True,
        inplace=True,
    )


    return data


# =========================================================
# DATE SPLIT
#
# Global chronological split:
#
# 70% training
# 15% validation
# 15% testing
#
# A 5-trading-day purge is added before
# validation and test boundaries.
# =========================================================

def create_date_split(
    data,
):

    unique_dates = np.array(
        sorted(
            data["Date"]
            .unique()
        )
    )


    total_dates = len(
        unique_dates
    )


    train_boundary = int(
        total_dates
        *
        0.70
    )


    validation_boundary = int(
        total_dates
        *
        0.85
    )


    if (
        train_boundary
        <= TARGET_HORIZON
        or
        validation_boundary
        <= train_boundary
    ):

        raise ValueError(
            "Dataset is too small for chronological splitting."
        )


    # -----------------------------------------------------
    # Purged training end
    # -----------------------------------------------------

    train_last_index = (
        train_boundary
        -
        TARGET_HORIZON
        -
        1
    )


    # -----------------------------------------------------
    # Validation begins after train boundary
    # and stops 5 trading days before test.
    # -----------------------------------------------------

    validation_start_index = (
        train_boundary
    )


    validation_last_index = (
        validation_boundary
        -
        TARGET_HORIZON
        -
        1
    )


    test_start_index = (
        validation_boundary
    )


    train_last_date = pd.Timestamp(
        unique_dates[
            train_last_index
        ]
    )


    validation_start_date = pd.Timestamp(
        unique_dates[
            validation_start_index
        ]
    )


    validation_last_date = pd.Timestamp(
        unique_dates[
            validation_last_index
        ]
    )


    test_start_date = pd.Timestamp(
        unique_dates[
            test_start_index
        ]
    )


    return {

        "unique_dates":
            unique_dates,

        "train_last_date":
            train_last_date,

        "validation_start_date":
            validation_start_date,

        "validation_last_date":
            validation_last_date,

        "test_start_date":
            test_start_date,
    }


# =========================================================
# FIT FEATURE SCALER
# =========================================================

def fit_feature_scaler(
    data,
    split,
):

    training_rows = data[
        data["Date"]
        <=
        split[
            "train_last_date"
        ]
    ]


    scaler = (
        StandardScaler()
    )


    scaler.fit(
        training_rows[
            FEATURE_COLUMNS
        ].values
    )


    return scaler


# =========================================================
# BUILD SEQUENCES
#
# CRITICAL:
#
# Sequences are created separately for every SYMBOL.
#
# RELIANCE → RELIANCE only
# TCS      → TCS only
#
# No company mixing.
# =========================================================

def build_sequences(
    data,
    scaler,
    split,
):

    X_train = []
    y_train = []


    X_validation = []
    y_validation = []


    X_test = []
    y_test = []


    train_symbols = []
    validation_symbols = []
    test_symbols = []


    train_dates = []
    validation_dates = []
    test_dates = []


    symbols = sorted(
        data["SYMBOL"]
        .unique()
    )


    for symbol in symbols:

        stock_data = (
            data[
                data["SYMBOL"]
                ==
                symbol
            ]
            .sort_values(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )


        feature_values = (
            scaler
            .transform(
                stock_data[
                    FEATURE_COLUMNS
                ].values
            )
            .astype(
                np.float32
            )
        )


        labels = (
            stock_data[
                "TARGET_CLASS"
            ]
            .values
            .astype(
                np.int32
            )
        )


        dates = (
            stock_data[
                "Date"
            ]
            .tolist()
        )


        for position in range(
            SEQUENCE_LENGTH - 1,
            len(
                stock_data
            ),
        ):

            start = (
                position
                -
                SEQUENCE_LENGTH
                +
                1
            )


            sequence = (
                feature_values[
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


            sample_date = pd.Timestamp(
                dates[
                    position
                ]
            )


            target = int(
                labels[
                    position
                ]
            )


            # =================================================
            # TRAIN
            # =================================================

            if (
                sample_date
                <=
                split[
                    "train_last_date"
                ]
            ):

                X_train.append(
                    sequence
                )

                y_train.append(
                    target
                )

                train_symbols.append(
                    symbol
                )

                train_dates.append(
                    sample_date
                )


            # =================================================
            # VALIDATION
            # =================================================

            elif (
                sample_date
                >=
                split[
                    "validation_start_date"
                ]
                and
                sample_date
                <=
                split[
                    "validation_last_date"
                ]
            ):

                X_validation.append(
                    sequence
                )

                y_validation.append(
                    target
                )

                validation_symbols.append(
                    symbol
                )

                validation_dates.append(
                    sample_date
                )


            # =================================================
            # TEST
            # =================================================

            elif (
                sample_date
                >=
                split[
                    "test_start_date"
                ]
            ):

                X_test.append(
                    sequence
                )

                y_test.append(
                    target
                )

                test_symbols.append(
                    symbol
                )

                test_dates.append(
                    sample_date
                )


    return {

        "X_train":
            np.asarray(
                X_train,
                dtype=np.float32,
            ),

        "y_train":
            np.asarray(
                y_train,
                dtype=np.int32,
            ),

        "X_validation":
            np.asarray(
                X_validation,
                dtype=np.float32,
            ),

        "y_validation":
            np.asarray(
                y_validation,
                dtype=np.int32,
            ),

        "X_test":
            np.asarray(
                X_test,
                dtype=np.float32,
            ),

        "y_test":
            np.asarray(
                y_test,
                dtype=np.int32,
            ),

        "test_symbols":
            np.asarray(
                test_symbols
            ),

        "test_dates":
            test_dates,
    }


# =========================================================
# CLASS WEIGHTS
#
# Helps compensate for the slightly smaller
# NEUTRAL class.
# =========================================================

def calculate_class_weights(
    y_train,
):

    total = len(
        y_train
    )


    class_weights = {}


    for class_id in range(
        NUM_CLASSES
    ):

        count = int(
            np.sum(
                y_train
                ==
                class_id
            )
        )


        if count == 0:

            class_weights[
                class_id
            ] = 1.0

        else:

            class_weights[
                class_id
            ] = (

                total

                /

                (
                    NUM_CLASSES
                    *
                    count
                )
            )


    return class_weights


# =========================================================
# MODEL
# =========================================================

def build_v6_model():

    inputs = Input(

        shape=(
            SEQUENCE_LENGTH,
            len(
                FEATURE_COLUMNS
            ),
        ),

        name=
            "universal_market_sequence",
    )


    # =====================================================
    # BiLSTM 1
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
    # BiLSTM 2
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
    # CLASSIFIER
    # =====================================================

    x = Dense(

        48,

        activation=
            "relu",

        name=
            "dense_1",

    )(
        x
    )


    x = Dropout(

        0.20,

        name=
            "dense_dropout",

    )(
        x
    )


    output = Dense(

        NUM_CLASSES,

        activation=
            "softmax",

        name=
            "market_class",

    )(
        x
    )


    model = Model(

        inputs=
            inputs,

        outputs=
            output,

        name=
            "StockVision_Universal_V6",
    )


    optimizer = (
        tf.keras.optimizers.Adam(

            learning_rate=
                0.0003,

            clipnorm=
                1.0,
        )
    )


    model.compile(

        optimizer=
            optimizer,

        loss=
            tf.keras.losses.SparseCategoricalCrossentropy(),

        metrics=[

            tf.keras.metrics.SparseCategoricalAccuracy(
                name=
                    "accuracy"
            ),
        ],
    )


    return model


# =========================================================
# CONFUSION MATRIX
# =========================================================

def calculate_confusion_matrix(
    actual,
    predicted,
):

    matrix = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=int,
    )


    for actual_class, predicted_class in zip(
        actual,
        predicted,
    ):

        matrix[
            int(
                actual_class
            ),
            int(
                predicted_class
            )
        ] += 1


    return matrix


# =========================================================
# CLASSIFICATION METRICS
# =========================================================

def calculate_metrics(
    actual,
    probabilities,
    training_labels,
):

    predicted = (
        np.argmax(
            probabilities,
            axis=1,
        )
    )


    # =====================================================
    # ACCURACY
    # =====================================================

    accuracy = (

        np.mean(
            predicted
            ==
            actual
        )

        *
        100
    )


    # =====================================================
    # MAJORITY BASELINE
    # =====================================================

    training_counts = (
        np.bincount(
            training_labels,
            minlength=
                NUM_CLASSES,
        )
    )


    majority_class = int(
        np.argmax(
            training_counts
        )
    )


    baseline_accuracy = (

        np.mean(
            actual
            ==
            majority_class
        )

        *
        100
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    confusion = (
        calculate_confusion_matrix(
            actual,
            predicted,
        )
    )


    class_metrics = {}


    f1_scores = []

    recalls = []


    for class_id in range(
        NUM_CLASSES
    ):

        true_positive = (
            confusion[
                class_id,
                class_id
            ]
        )


        false_positive = (

            confusion[
                :,
                class_id
            ].sum()

            -
            true_positive
        )


        false_negative = (

            confusion[
                class_id,
                :
            ].sum()

            -
            true_positive
        )


        precision_denominator = (
            true_positive
            +
            false_positive
        )


        recall_denominator = (
            true_positive
            +
            false_negative
        )


        precision = (

            true_positive
            /
            precision_denominator

            if precision_denominator > 0

            else 0.0
        )


        recall = (

            true_positive
            /
            recall_denominator

            if recall_denominator > 0

            else 0.0
        )


        if (
            precision
            +
            recall
        ) > 0:

            f1 = (

                2
                *
                precision
                *
                recall

                /
                (
                    precision
                    +
                    recall
                )
            )

        else:

            f1 = 0.0


        f1_scores.append(
            f1
        )


        recalls.append(
            recall
        )


        class_metrics[
            CLASS_NAMES[
                class_id
            ]
        ] = {

            "precision_percent":
                round(
                    precision
                    *
                    100,
                    2,
                ),

            "recall_percent":
                round(
                    recall
                    *
                    100,
                    2,
                ),

            "f1_percent":
                round(
                    f1
                    *
                    100,
                    2,
                ),
        }


    macro_f1 = (
        np.mean(
            f1_scores
        )
        *
        100
    )


    balanced_accuracy = (
        np.mean(
            recalls
        )
        *
        100
    )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = (
        np.max(
            probabilities,
            axis=1,
        )
    )


    high_confidence_mask = (
        confidence
        >=
        0.60
    )


    high_confidence_count = int(
        np.sum(
            high_confidence_mask
        )
    )


    confidence_coverage = (

        high_confidence_count

        /
        len(
            actual
        )

        *
        100
    )


    if high_confidence_count > 0:

        high_confidence_accuracy = (

            np.mean(

                predicted[
                    high_confidence_mask
                ]

                ==

                actual[
                    high_confidence_mask
                ]
            )

            *
            100
        )

    else:

        high_confidence_accuracy = 0.0


    # =====================================================
    # MULTICLASS BRIER SCORE
    # =====================================================

    one_hot_actual = np.eye(
        NUM_CLASSES
    )[
        actual
    ]


    brier_score = (
        np.mean(
            np.sum(

                (
                    probabilities
                    -
                    one_hot_actual
                )
                ** 2,

                axis=1,
            )
        )
    )


    return {

        "accuracy_percent":
            round(
                float(
                    accuracy
                ),
                2,
            ),

        "majority_class":
            CLASS_NAMES[
                majority_class
            ],

        "majority_baseline_accuracy_percent":
            round(
                float(
                    baseline_accuracy
                ),
                2,
            ),

        "improvement_vs_majority_baseline_pp":
            round(
                float(
                    accuracy
                    -
                    baseline_accuracy
                ),
                2,
            ),

        "macro_f1_percent":
            round(
                float(
                    macro_f1
                ),
                2,
            ),

        "balanced_accuracy_percent":
            round(
                float(
                    balanced_accuracy
                ),
                2,
            ),

        "high_confidence_accuracy_percent":
            round(
                float(
                    high_confidence_accuracy
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

        "class_metrics":
            class_metrics,

        "confusion_matrix":
            confusion.tolist(),
    }


# =========================================================
# PER-STOCK TEST ACCURACY
# =========================================================

def calculate_per_stock_accuracy(
    actual,
    probabilities,
    symbols,
):

    predicted = np.argmax(
        probabilities,
        axis=1,
    )


    results = {}


    for symbol in sorted(
        np.unique(
            symbols
        )
    ):

        mask = (
            symbols
            ==
            symbol
        )


        if not np.any(
            mask
        ):

            continue


        accuracy = (

            np.mean(
                predicted[
                    mask
                ]
                ==
                actual[
                    mask
                ]
            )

            *
            100
        )


        results[
            symbol
        ] = round(
            float(
                accuracy
            ),
            2,
        )


    return results


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(
    metrics,
    per_stock,
):

    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION UNIVERSAL V6 TEST RESULTS"
    )

    print(
        "=============================================="
    )


    print(
        "\nOverall Accuracy:",
        f'{metrics["accuracy_percent"]:.2f}%'
    )


    print(
        "Majority Baseline:",
        f'{metrics["majority_baseline_accuracy_percent"]:.2f}%'
    )


    print(
        "Improvement:",
        f'{metrics["improvement_vs_majority_baseline_pp"]:+.2f} pp'
    )


    print(
        "Macro F1:",
        f'{metrics["macro_f1_percent"]:.2f}%'
    )


    print(
        "Balanced Accuracy:",
        f'{metrics["balanced_accuracy_percent"]:.2f}%'
    )


    print(
        "\nHigh-Confidence Accuracy:",
        f'{metrics["high_confidence_accuracy_percent"]:.2f}%'
    )


    print(
        "High-Confidence Coverage:",
        f'{metrics["high_confidence_coverage_percent"]:.2f}%'
    )


    print(
        "Brier Score:",
        metrics[
            "brier_score"
        ]
    )


    print(
        "\n=============================================="
    )

    print(
        " PER-CLASS RESULTS"
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
            metrics[
                "class_metrics"
            ][
                class_name
            ]
        )


        print(
            f"\n{class_name}"
        )


        print(
            "Precision:",
            f'{values["precision_percent"]:.2f}%'
        )


        print(
            "Recall:",
            f'{values["recall_percent"]:.2f}%'
        )


        print(
            "F1:",
            f'{values["f1_percent"]:.2f}%'
        )


    print(
        "\n=============================================="
    )

    print(
        " CONFUSION MATRIX"
    )

    print(
        "=============================================="
    )


    print(
        "\nRows = Actual"
    )

    print(
        "Columns = Predicted"
    )


    print(
        "\n          BEAR     NEUTRAL     BULL"
    )


    matrix = (
        metrics[
            "confusion_matrix"
        ]
    )


    labels = [
        "BEAR",
        "NEUTRAL",
        "BULL",
    ]


    for index, row in enumerate(
        matrix
    ):

        print(
            f'{labels[index]:<8}'
            f'{row[0]:>8}'
            f'{row[1]:>12}'
            f'{row[2]:>10}'
        )


    print(
        "\n=============================================="
    )

    print(
        " PER-STOCK TEST ACCURACY"
    )

    print(
        "=============================================="
    )


    for symbol, accuracy in per_stock.items():

        print(
            f"{symbol:<18}"
            f"{accuracy:>7.2f}%"
        )


# =========================================================
# TRAIN
# =========================================================

def train_v6():

    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION UNIVERSAL V6"
    )

    print(
        "=============================================="
    )


    print(
        "\nModel Goal:"
    )

    print(
        "Predict 5-day BEARISH / NEUTRAL / BULLISH movement"
    )


    print(
        "\nLoading universal dataset..."
    )


    data = (
        load_dataset()
    )


    print(
        "Rows:",
        len(
            data
        )
    )


    print(
        "Stocks:",
        data[
            "SYMBOL"
        ]
        .nunique()
    )


    print(
        "Features:",
        len(
            FEATURE_COLUMNS
        )
    )


    # =====================================================
    # SPLIT
    # =====================================================

    split = (
        create_date_split(
            data
        )
    )


    print(
        "\nChronological Split:"
    )


    print(
        "Training ends:",
        split[
            "train_last_date"
        ].date()
    )


    print(
        "Validation starts:",
        split[
            "validation_start_date"
        ].date()
    )


    print(
        "Validation ends:",
        split[
            "validation_last_date"
        ].date()
    )


    print(
        "Test starts:",
        split[
            "test_start_date"
        ].date()
    )


    print(
        f"Purge Gap: {TARGET_HORIZON} trading days"
    )


    # =====================================================
    # SCALER
    # =====================================================

    scaler = (
        fit_feature_scaler(
            data,
            split,
        )
    )


    # =====================================================
    # SEQUENCES
    # =====================================================

    print(
        "\nBuilding stock-isolated sequences..."
    )


    sequences = (
        build_sequences(
            data,
            scaler,
            split,
        )
    )


    print(
        "\nSequence Shapes:"
    )


    print(
        "Train:",
        sequences[
            "X_train"
        ].shape
    )


    print(
        "Validation:",
        sequences[
            "X_validation"
        ].shape
    )


    print(
        "Test:",
        sequences[
            "X_test"
        ].shape
    )


    # =====================================================
    # CLASS WEIGHTS
    # =====================================================

    class_weights = (
        calculate_class_weights(
            sequences[
                "y_train"
            ]
        )
    )


    print(
        "\nTraining Class Weights:"
    )


    for class_id in range(
        NUM_CLASSES
    ):

        print(
            CLASS_NAMES[
                class_id
            ],
            ":",
            round(
                class_weights[
                    class_id
                ],
                4,
            )
        )


    # =====================================================
    # MODEL
    # =====================================================

    tf.keras.backend.clear_session()


    model = (
        build_v6_model()
    )


    print(
        "\nV6 Architecture:"
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
                7,

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
                3,

            min_lr=
                0.000005,

            verbose=
                1,
        ),


        ModelCheckpoint(

            filepath=
                MODEL_PATH,

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
        "\nStarting Universal V6 training..."
    )


    history = (
        model.fit(

            sequences[
                "X_train"
            ],

            sequences[
                "y_train"
            ],

            validation_data=(

                sequences[
                    "X_validation"
                ],

                sequences[
                    "y_validation"
                ],
            ),

            epochs=
                50,

            batch_size=
                64,

            shuffle=
                True,

            class_weight=
                class_weights,

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
    # LOAD BEST MODEL
    # =====================================================

    model = (
        tf.keras.models.load_model(
            MODEL_PATH
        )
    )


    # =====================================================
    # TEST
    # =====================================================

    print(
        "\nRunning final test evaluation..."
    )


    probabilities = (
        model.predict(

            sequences[
                "X_test"
            ],

            batch_size=
                128,

            verbose=
                0,
        )
    )


    metrics = (
        calculate_metrics(

            actual=
                sequences[
                    "y_test"
                ],

            probabilities=
                probabilities,

            training_labels=
                sequences[
                    "y_train"
                ],
        )
    )


    per_stock = (
        calculate_per_stock_accuracy(

            actual=
                sequences[
                    "y_test"
                ],

            probabilities=
                probabilities,

            symbols=
                sequences[
                    "test_symbols"
                ],
        )
    )


    # =====================================================
    # SAVE SCALER
    # =====================================================

    joblib.dump(
        scaler,
        SCALER_PATH,
    )


    # =====================================================
    # SAVE METRICS
    # =====================================================

    result_payload = {

        "model":
            "StockVision Universal V6 BiLSTM",

        "target":
            "5-Day Market Movement Classification",

        "classes":
            CLASS_NAMES,

        "features_used":
            len(
                FEATURE_COLUMNS
            ),

        "feature_columns":
            FEATURE_COLUMNS,

        "sequence_length":
            SEQUENCE_LENGTH,

        "target_horizon_days":
            TARGET_HORIZON,

        "best_epoch":
            best_epoch,

        "train_samples":
            int(
                len(
                    sequences[
                        "X_train"
                    ]
                )
            ),

        "validation_samples":
            int(
                len(
                    sequences[
                        "X_validation"
                    ]
                )
            ),

        "test_samples":
            int(
                len(
                    sequences[
                        "X_test"
                    ]
                )
            ),

        "metrics":
            metrics,

        "per_stock_test_accuracy":
            per_stock,
    }


    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result_payload,
            file,
            indent=4,
        )


    # =====================================================
    # SAVE MODEL METADATA
    # =====================================================

    metadata = {

        "model_version":
            "V6",

        "model_type":
            "Universal Multi-Stock BiLSTM Classifier",

        "stocks_seen":
            sorted(
                data[
                    "SYMBOL"
                ]
                .unique()
                .tolist()
            ),

        "feature_columns":
            FEATURE_COLUMNS,

        "sequence_length":
            SEQUENCE_LENGTH,

        "target_horizon_days":
            TARGET_HORIZON,

        "target_definition": {

            "BEARISH":
                "< -1%",

            "NEUTRAL":
                "-1% to +1%",

            "BULLISH":
                "> +1%",
        },

        "train_last_date":
            str(
                split[
                    "train_last_date"
                ].date()
            ),

        "validation_start_date":
            str(
                split[
                    "validation_start_date"
                ].date()
            ),

        "validation_last_date":
            str(
                split[
                    "validation_last_date"
                ].date()
            ),

        "test_start_date":
            str(
                split[
                    "test_start_date"
                ].date()
            ),
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
    # PRINT
    # =====================================================

    print_results(
        metrics,
        per_stock,
    )


    print(
        "\n=============================================="
    )

    print(
        " V6 FILES SAVED"
    )

    print(
        "=============================================="
    )


    print(
        "\nModel:"
    )

    print(
        MODEL_PATH
    )


    print(
        "\nScaler:"
    )

    print(
        SCALER_PATH
    )


    print(
        "\nMetrics:"
    )

    print(
        METRICS_PATH
    )


    print(
        "\nMetadata:"
    )

    print(
        METADATA_PATH
    )


    print(
        "\nUniversal V6 training complete."
    )


# =========================================================
# COMMAND LINE
# =========================================================

if __name__ == "__main__":

    train_v6()