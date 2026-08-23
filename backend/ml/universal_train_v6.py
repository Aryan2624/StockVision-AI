import os

# Reduce unnecessary TensorFlow console messages
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
# STOCKVISION FEATURE IMPORT
# =========================================================

try:

    from .context_features_v5 import (
        FEATURE_COLUMNS,
    )

except ImportError:

    from context_features_v5 import (
        FEATURE_COLUMNS,
    )


# =========================================================
# V6 CONFIG
# =========================================================

TARGET_HORIZON = 5

SEQUENCE_LENGTH = 60

NUM_CLASSES = 3

RANDOM_SEED = 42


# =========================================================
# CLASS MAPPING
#
# 0 = BEARISH
# 1 = NEUTRAL
# 2 = BULLISH
# =========================================================

CLASS_NAMES = {
    0: "BEARISH",
    1: "NEUTRAL",
    2: "BULLISH",
}


# =========================================================
# RANDOM SEEDS
# =========================================================

np.random.seed(
    RANDOM_SEED
)

random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


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


MODELS_DIR = os.path.join(
    BASE_DIR,
    "models",
)


os.makedirs(
    MODELS_DIR,
    exist_ok=True,
)


CSV_PATH = os.path.join(
    DATASETS_DIR,
    "stockvision_v6_universal_dataset.csv",
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


PREDICTIONS_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_universal_v6_test_predictions.csv",
)


# =========================================================
# LOAD UNIVERSAL DATASET
# =========================================================

def load_dataset():

    if not os.path.exists(
        CSV_PATH
    ):

        raise FileNotFoundError(
            "\nV6 universal dataset was not found:\n"
            f"{CSV_PATH}\n\n"
            "Create the universal dataset first."
        )


    data = pd.read_csv(
        CSV_PATH
    )


    required_columns = [

        "Date",
        "SYMBOL",
        "TARGET_CLASS",

        *FEATURE_COLUMNS,
    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in data.columns
    ]


    if missing_columns:

        raise ValueError(
            "Dataset is missing required columns:\n"
            +
            ", ".join(
                missing_columns
            )
        )


    data["Date"] = pd.to_datetime(
        data["Date"]
    )


    data["TARGET_CLASS"] = (

        pd.to_numeric(
            data["TARGET_CLASS"],
            errors="coerce",
        )
    )


    # =====================================================
    # ENSURE FEATURES ARE NUMERIC
    # =====================================================

    for feature in FEATURE_COLUMNS:

        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce",
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
        subset=[
            "Date",
            "SYMBOL",
            "TARGET_CLASS",
            *FEATURE_COLUMNS,
        ],
        inplace=True,
    )


    data["TARGET_CLASS"] = (
        data["TARGET_CLASS"]
        .astype(
            np.int32
        )
    )


    invalid_classes = data[
        ~data[
            "TARGET_CLASS"
        ].isin(
            [
                0,
                1,
                2,
            ]
        )
    ]


    if not invalid_classes.empty:

        raise ValueError(
            "TARGET_CLASS contains values outside 0, 1, 2."
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
# GLOBAL CHRONOLOGICAL SPLIT
#
# 70% train
# 15% validation
# 15% test
#
# We purge 5 trading dates before validation
# and test because the target uses a future
# 5-day stock return.
# =========================================================

def create_date_split(
    data,
):

    unique_dates = np.array(
        sorted(
            data[
                "Date"
            ].unique()
        )
    )


    total_dates = len(
        unique_dates
    )


    if total_dates < 500:

        raise ValueError(
            "Not enough unique dates for V6 training."
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


    # =====================================================
    # PURGED TRAIN END
    # =====================================================

    train_last_index = (
        train_boundary
        -
        TARGET_HORIZON
        -
        1
    )


    # =====================================================
    # VALIDATION
    # =====================================================

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


    # =====================================================
    # TEST
    # =====================================================

    test_start_index = (
        validation_boundary
    )


    if (
        train_last_index <= 0
        or
        validation_last_index <= validation_start_index
        or
        test_start_index >= total_dates
    ):

        raise ValueError(
            "Invalid chronological split."
        )


    return {

        "total_unique_dates":
            int(
                total_dates
            ),


        "train_last_date":
            pd.Timestamp(
                unique_dates[
                    train_last_index
                ]
            ),


        "validation_start_date":
            pd.Timestamp(
                unique_dates[
                    validation_start_index
                ]
            ),


        "validation_last_date":
            pd.Timestamp(
                unique_dates[
                    validation_last_index
                ]
            ),


        "test_start_date":
            pd.Timestamp(
                unique_dates[
                    test_start_index
                ]
            ),
    }


# =========================================================
# FIT STANDARD SCALER
#
# Fit ONLY using training-period rows.
# =========================================================

def fit_feature_scaler(
    data,
    split,
):

    training_rows = data[
        data[
            "Date"
        ]
        <=
        split[
            "train_last_date"
        ]
    ]


    if training_rows.empty:

        raise ValueError(
            "No rows available for scaler fitting."
        )


    scaler = (
        StandardScaler()
    )


    scaler.fit(
        training_rows[
            FEATURE_COLUMNS
        ]
        .values
    )


    return scaler


# =========================================================
# BUILD STOCK-ISOLATED SEQUENCES
#
# IMPORTANT:
#
# A sequence can only contain observations
# from ONE stock.
#
# RELIANCE -> RELIANCE
# TCS      -> TCS
# INFY     -> INFY
#
# Never:
#
# RELIANCE day 59
# TCS day 60
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

    test_symbols = []
    test_dates = []


    symbols = sorted(
        data[
            "SYMBOL"
        ]
        .unique()
    )


    for symbol in symbols:

        stock_data = (

            data[
                data[
                    "SYMBOL"
                ]
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

            scaler.transform(

                stock_data[
                    FEATURE_COLUMNS
                ]
                .values
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


        # =================================================
        # CREATE 60-DAY WINDOWS
        # =================================================

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


            if len(
                sequence
            ) != SEQUENCE_LENGTH:

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


    result = {

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


    # =====================================================
    # CHECK
    # =====================================================

    if len(
        result[
            "X_train"
        ]
    ) == 0:

        raise ValueError(
            "No training sequences were created."
        )


    if len(
        result[
            "X_validation"
        ]
    ) == 0:

        raise ValueError(
            "No validation sequences were created."
        )


    if len(
        result[
            "X_test"
        ]
    ) == 0:

        raise ValueError(
            "No test sequences were created."
        )


    return result


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

def print_class_distribution(
    labels,
    title,
):

    print(
        f"\n{title}"
    )


    total = len(
        labels
    )


    for class_id in range(
        NUM_CLASSES
    ):

        count = int(
            np.sum(
                labels
                ==
                class_id
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


        print(

            f"{CLASS_NAMES[class_id]:<8}: "
            f"{count:>6} "
            f"({percentage:.2f}%)"
        )


# =========================================================
# CLASS WEIGHTS
#
# Gives the smaller classes more importance
# during training.
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


        if count <= 0:

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
# BUILD UNIVERSAL BiLSTM
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
    # DENSE CLASSIFICATION BLOCK
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


    # =====================================================
    # THREE-CLASS SOFTMAX OUTPUT
    # =====================================================

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
                name="accuracy"
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


    for (
        actual_class,
        predicted_class,

    ) in zip(
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
# TEST METRICS
# =========================================================

def calculate_metrics(
    actual,
    probabilities,
    training_labels,
):

    predicted = np.argmax(
        probabilities,
        axis=1,
    )


    # =====================================================
    # OVERALL ACCURACY
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
    # MAJORITY CLASS BASELINE
    # =====================================================

    training_counts = np.bincount(

        training_labels,

        minlength=
            NUM_CLASSES,
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


    # =====================================================
    # PER CLASS METRICS
    # =====================================================

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


        if precision_denominator > 0:

            precision = (
                true_positive
                /
                precision_denominator
            )

        else:

            precision = 0.0


        if recall_denominator > 0:

            recall = (
                true_positive
                /
                recall_denominator
            )

        else:

            recall = 0.0


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
                    float(
                        precision
                        *
                        100
                    ),
                    2,
                ),


            "recall_percent":
                round(
                    float(
                        recall
                        *
                        100
                    ),
                    2,
                ),


            "f1_percent":
                round(
                    float(
                        f1
                        *
                        100
                    ),
                    2,
                ),
        }


    # =====================================================
    # MACRO F1
    # =====================================================

    macro_f1 = (

        np.mean(
            f1_scores
        )

        *
        100
    )


    # =====================================================
    # BALANCED ACCURACY
    # =====================================================

    balanced_accuracy = (

        np.mean(
            recalls
        )

        *
        100
    )


    # =====================================================
    # HIGH CONFIDENCE
    #
    # Softmax probability >= 60%
    # =====================================================

    confidence = np.max(
        probabilities,
        axis=1,
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


    if len(
        actual
    ) > 0:

        confidence_coverage = (

            high_confidence_count

            /

            len(
                actual
            )

            *
            100
        )

    else:

        confidence_coverage = 0.0


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
    #
    # Lower = better probability quality.
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

                **
                2,

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


        stock_actual = (
            actual[
                mask
            ]
        )


        stock_predicted = (
            predicted[
                mask
            ]
        )


        accuracy = (

            np.mean(
                stock_predicted
                ==
                stock_actual
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
# SAVE TEST PREDICTIONS
# =========================================================

def save_test_predictions(
    sequences,
    probabilities,
):

    predicted_classes = np.argmax(
        probabilities,
        axis=1,
    )


    rows = []


    for index in range(
        len(
            predicted_classes
        )
    ):

        actual_class = int(
            sequences[
                "y_test"
            ][
                index
            ]
        )


        predicted_class = int(
            predicted_classes[
                index
            ]
        )


        rows.append(
            {

                "Date":
                    str(
                        pd.Timestamp(
                            sequences[
                                "test_dates"
                            ][
                                index
                            ]
                        ).date()
                    ),


                "SYMBOL":
                    str(
                        sequences[
                            "test_symbols"
                        ][
                            index
                        ]
                    ),


                "ACTUAL_CLASS":
                    actual_class,


                "ACTUAL_LABEL":
                    CLASS_NAMES[
                        actual_class
                    ],


                "PREDICTED_CLASS":
                    predicted_class,


                "PREDICTED_LABEL":
                    CLASS_NAMES[
                        predicted_class
                    ],


                "BEARISH_PROBABILITY":
                    float(
                        probabilities[
                            index,
                            0
                        ]
                    ),


                "NEUTRAL_PROBABILITY":
                    float(
                        probabilities[
                            index,
                            1
                        ]
                    ),


                "BULLISH_PROBABILITY":
                    float(
                        probabilities[
                            index,
                            2
                        ]
                    ),


                "CONFIDENCE":
                    float(
                        np.max(
                            probabilities[
                                index
                            ]
                        )
                    ),
            }
        )


    prediction_df = pd.DataFrame(
        rows
    )


    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )


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
        "Baseline Class:",
        metrics[
            "majority_class"
        ]
    )


    print(
        "Improvement:",
        f'{metrics["improvement_vs_majority_baseline_pp"]:+.2f} pp'
    )


    print(
        "\nMacro F1:",
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
        "Multiclass Brier Score:",
        metrics[
            "brier_score"
        ]
    )


    # =====================================================
    # PER CLASS
    # =====================================================

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


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

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
        "\n            BEAR     NEUTRAL      BULL"
    )


    matrix = (
        metrics[
            "confusion_matrix"
        ]
    )


    row_labels = [
        "BEAR",
        "NEUTRAL",
        "BULL",
    ]


    for index, row in enumerate(
        matrix
    ):

        print(

            f'{row_labels[index]:<10}'

            f'{row[0]:>8}'

            f'{row[1]:>12}'

            f'{row[2]:>10}'
        )


    # =====================================================
    # PER STOCK
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " PER-STOCK TEST ACCURACY"
    )

    print(
        "=============================================="
    )


    for (
        symbol,
        accuracy,

    ) in per_stock.items():

        print(

            f"{symbol:<18}"

            f"{accuracy:>7.2f}%"
        )


# =========================================================
# TRAIN UNIVERSAL V6
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


    # =====================================================
    # LOAD DATA
    # =====================================================

    data = (
        load_dataset()
    )


    print(
        "\nRows:",
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


    print(
        "Sequence Length:",
        SEQUENCE_LENGTH
    )


    print(
        "Target Horizon:",
        TARGET_HORIZON,
        "days",
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
        "\n=============================================="
    )

    print(
        " CHRONOLOGICAL SPLIT"
    )

    print(
        "=============================================="
    )


    print(
        "\nTraining ends:",
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
        "Target Purge Gap:",
        TARGET_HORIZON,
        "trading dates",
    )


    # =====================================================
    # SCALER
    # =====================================================

    print(
        "\nFitting feature scaler on training period only..."
    )


    scaler = (
        fit_feature_scaler(
            data,
            split,
        )
    )


    # =====================================================
    # BUILD SEQUENCES
    # =====================================================

    print(
        "\nBuilding stock-isolated 60-day sequences..."
    )


    sequences = (
        build_sequences(
            data,
            scaler,
            split,
        )
    )


    print(
        "\n=============================================="
    )

    print(
        " SEQUENCE SHAPES"
    )

    print(
        "=============================================="
    )


    print(
        "\nTrain:",
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
    # CLASS DISTRIBUTIONS
    # =====================================================

    print_class_distribution(
        sequences[
            "y_train"
        ],
        "TRAINING CLASS DISTRIBUTION",
    )


    print_class_distribution(
        sequences[
            "y_validation"
        ],
        "VALIDATION CLASS DISTRIBUTION",
    )


    print_class_distribution(
        sequences[
            "y_test"
        ],
        "TEST CLASS DISTRIBUTION",
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
        "\n=============================================="
    )

    print(
        " TRAINING CLASS WEIGHTS"
    )

    print(
        "=============================================="
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
        "\n=============================================="
    )

    print(
        " V6 ARCHITECTURE"
    )

    print(
        "==============================================\n"
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
        "\n=============================================="
    )

    print(
        " STARTING UNIVERSAL V6 TRAINING"
    )

    print(
        "==============================================\n"
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


    # =====================================================
    # BEST EPOCH
    # =====================================================

    best_epoch = int(

        np.argmin(

            history.history[
                "val_loss"
            ]
        )

        +

        1
    )


    print(
        "\nBest Epoch:",
        best_epoch
    )


    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    print(
        "\nLoading best saved V6 model..."
    )


    model = (
        tf.keras.models.load_model(
            MODEL_PATH
        )
    )


    # =====================================================
    # FINAL TEST
    # =====================================================

    print(
        "\nRunning final out-of-sample test..."
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


    # =====================================================
    # METRICS
    # =====================================================

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


    # =====================================================
    # PER STOCK ACCURACY
    # =====================================================

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
    # SAVE TEST PREDICTIONS
    # =====================================================

    save_test_predictions(
        sequences,
        probabilities,
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


        "classes": {
            str(
                key
            ):
                value

            for (
                key,
                value
            ) in CLASS_NAMES.items()
        },


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

        encoding=
            "utf-8",

    ) as file:

        json.dump(
            result_payload,
            file,
            indent=4,
        )


    # =====================================================
    # SAVE METADATA
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

        encoding=
            "utf-8",

    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )


    # =====================================================
    # PRINT FINAL RESULTS
    # =====================================================

    print_results(
        metrics,
        per_stock,
    )


    # =====================================================
    # FILES
    # =====================================================

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
        "\nTest Predictions:"
    )

    print(
        PREDICTIONS_PATH
    )


    print(
        "\n=============================================="
    )

    print(
        " UNIVERSAL V6 TRAINING COMPLETE"
    )

    print(
        "=============================================="
    )


# =========================================================
# COMMAND LINE
# =========================================================

if __name__ == "__main__":

    train_v6()