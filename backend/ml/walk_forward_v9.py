import os
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# =========================================================
# V9 IMPORT
# =========================================================

try:
    from .relative_dataset_v9 import (
        FEATURE_COLUMNS,
    )

except ImportError:
    from relative_dataset_v9 import (
        FEATURE_COLUMNS,
    )


# =========================================================
# CONFIG
# =========================================================

TARGET_HORIZON = 5

NUM_CLASSES = 3

RANDOM_STATE = 42

CONFIDENCE_THRESHOLD = 0.60


CLASS_NAMES = {
    0: "UNDERPERFORM",
    1: "NEUTRAL",
    2: "OUTPERFORM",
}


# Validation year -> Test year
FOLDS = [
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
    (2025, 2026),
]


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "stockvision_v9_relative_dataset.csv",
)


MODELS_DIR = os.path.join(
    BASE_DIR,
    "models",
)


os.makedirs(
    MODELS_DIR,
    exist_ok=True,
)


RESULT_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_walk_forward_results.json",
)


PREDICTIONS_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_walk_forward_predictions.csv",
)


# =========================================================
# LOAD DATA
# =========================================================

def load_dataset():

    if not os.path.exists(
        DATASET_PATH
    ):

        raise FileNotFoundError(
            f"V9 dataset not found:\n{DATASET_PATH}"
        )


    data = pd.read_csv(
        DATASET_PATH
    )


    data["Date"] = pd.to_datetime(
        data["Date"]
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
            "Missing columns:\n"
            +
            ", ".join(
                missing_columns
            )
        )


    for feature in FEATURE_COLUMNS:

        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce",
        )


    data["TARGET_CLASS"] = pd.to_numeric(
        data["TARGET_CLASS"],
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
        .astype(int)
    )


    data.sort_values(
        by=[
            "Date",
            "SYMBOL",
        ],
        inplace=True,
    )


    data.reset_index(
        drop=True,
        inplace=True,
    )


    return data


# =========================================================
# CLASS WEIGHTS / SAMPLE WEIGHTS
# =========================================================

def calculate_sample_weights(
    labels,
):

    labels = np.asarray(
        labels,
        dtype=int,
    )


    counts = np.bincount(
        labels,
        minlength=NUM_CLASSES,
    )


    total = len(labels)


    class_weights = {}


    for class_id in range(
        NUM_CLASSES
    ):

        if counts[class_id] == 0:

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
                    counts[class_id]
                )
            )


    sample_weights = np.array(
        [
            class_weights[int(label)]
            for label in labels
        ],
        dtype=float,
    )


    return (
        class_weights,
        sample_weights,
    )


# =========================================================
# BUILD MODELS
# =========================================================

def build_logistic():

    return LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_random_forest():

    return RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def build_hist_gradient():

    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


# =========================================================
# TRAIN A MODEL
# =========================================================

def fit_model(
    model_name,
    X_train,
    y_train,
):

    scaler = None


    if model_name == "Logistic Regression":

        scaler = StandardScaler()

        X_train_final = (
            scaler.fit_transform(
                X_train
            )
        )


        model = build_logistic()


        model.fit(
            X_train_final,
            y_train,
        )


    elif model_name == "Random Forest":

        model = build_random_forest()


        model.fit(
            X_train,
            y_train,
        )


    elif model_name == "Hist Gradient Boosting":

        model = build_hist_gradient()


        (
            _,
            sample_weights,

        ) = calculate_sample_weights(
            y_train
        )


        model.fit(
            X_train,
            y_train,
            sample_weight=
                sample_weights,
        )


    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


    return (
        model,
        scaler,
    )


# =========================================================
# PREDICT PROBABILITIES
# =========================================================

def predict_probabilities(
    model,
    scaler,
    X,
):

    if scaler is not None:

        X = scaler.transform(
            X
        )


    return model.predict_proba(
        X
    )


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    actual,
    probabilities,
):

    predicted = np.argmax(
        probabilities,
        axis=1,
    )


    accuracy = (
        accuracy_score(
            actual,
            predicted,
        )
        *
        100
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            actual,
            predicted,
        )
        *
        100
    )


    macro_f1 = (
        f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0,
        )
        *
        100
    )


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=[
                0,
                1,
                2,
            ],
            zero_division=0,
        )
    )


    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[
            0,
            1,
            2,
        ],
    )


    class_metrics = {}


    for class_id in range(
        NUM_CLASSES
    ):

        class_metrics[
            CLASS_NAMES[class_id]
        ] = {

            "precision_percent":
                round(
                    float(
                        precision[class_id]
                        *
                        100
                    ),
                    2,
                ),

            "recall_percent":
                round(
                    float(
                        recall[class_id]
                        *
                        100
                    ),
                    2,
                ),

            "f1_percent":
                round(
                    float(
                        f1[class_id]
                        *
                        100
                    ),
                    2,
                ),

            "support":
                int(
                    support[class_id]
                ),
        }


    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = np.max(
        probabilities,
        axis=1,
    )


    high_confidence_mask = (
        confidence
        >=
        CONFIDENCE_THRESHOLD
    )


    high_confidence_count = int(
        high_confidence_mask.sum()
    )


    coverage = (
        high_confidence_count
        /
        len(actual)
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
    # BRIER SCORE
    # =====================================================

    one_hot_actual = np.eye(
        NUM_CLASSES
    )[
        actual
    ]


    brier = np.mean(
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


    return {

        "accuracy_percent":
            round(
                float(accuracy),
                2,
            ),

        "balanced_accuracy_percent":
            round(
                float(
                    balanced_accuracy
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
                    coverage
                ),
                2,
            ),

        "brier_score":
            round(
                float(
                    brier
                ),
                4,
            ),

        "class_metrics":
            class_metrics,

        "confusion_matrix":
            matrix.tolist(),
    }


# =========================================================
# BASELINE
# =========================================================

def calculate_majority_baseline(
    train_labels,
    test_labels,
):

    counts = np.bincount(
        train_labels,
        minlength=NUM_CLASSES,
    )


    majority_class = int(
        np.argmax(
            counts
        )
    )


    predictions = np.full(
        len(test_labels),
        majority_class,
        dtype=int,
    )


    accuracy = (
        np.mean(
            predictions
            ==
            test_labels
        )
        *
        100
    )


    return (
        majority_class,
        float(accuracy),
        predictions,
    )


# =========================================================
# PRINT METRICS
# =========================================================

def print_metrics(
    metrics,
):

    print(
        "Accuracy:",
        f'{metrics["accuracy_percent"]:.2f}%'
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
        "High Confidence Accuracy:",
        f'{metrics["high_confidence_accuracy_percent"]:.2f}%'
    )


    print(
        "High Confidence Coverage:",
        f'{metrics["high_confidence_coverage_percent"]:.2f}%'
    )


    print(
        "Brier Score:",
        metrics[
            "brier_score"
        ]
    )


# =========================================================
# CREATE ONE WALK-FORWARD FOLD
# =========================================================

def create_fold(
    data,
    validation_year,
    test_year,
):

    unique_dates = np.array(
        sorted(
            data[
                "Date"
            ].unique()
        )
    )


    date_to_index = {
        pd.Timestamp(date): index
        for index, date
        in enumerate(unique_dates)
    }


    validation_dates = [
        pd.Timestamp(date)
        for date in unique_dates
        if pd.Timestamp(date).year
        ==
        validation_year
    ]


    test_dates = [
        pd.Timestamp(date)
        for date in unique_dates
        if pd.Timestamp(date).year
        ==
        test_year
    ]


    if not validation_dates:

        raise ValueError(
            f"No validation dates found for {validation_year}"
        )


    if not test_dates:

        raise ValueError(
            f"No test dates found for {test_year}"
        )


    validation_start = min(
        validation_dates
    )


    test_start = min(
        test_dates
    )


    validation_start_index = (
        date_to_index[
            validation_start
        ]
    )


    test_start_index = (
        date_to_index[
            test_start
        ]
    )


    # =====================================================
    # PURGE BEFORE VALIDATION
    # =====================================================

    selection_train_last_index = (
        validation_start_index
        -
        TARGET_HORIZON
        -
        1
    )


    # =====================================================
    # PURGE BEFORE TEST
    # =====================================================

    validation_last_index = (
        test_start_index
        -
        TARGET_HORIZON
        -
        1
    )


    if (
        selection_train_last_index
        < 0
        or
        validation_last_index
        <=
        validation_start_index
    ):

        raise ValueError(
            f"Invalid split for {validation_year} → {test_year}"
        )


    selection_train_last_date = pd.Timestamp(
        unique_dates[
            selection_train_last_index
        ]
    )


    validation_last_date = pd.Timestamp(
        unique_dates[
            validation_last_index
        ]
    )


    # =====================================================
    # MODEL SELECTION TRAIN DATA
    # =====================================================

    selection_train = data[
        data[
            "Date"
        ]
        <=
        selection_train_last_date
    ].copy()


    # =====================================================
    # VALIDATION DATA
    # =====================================================

    validation = data[
        (
            data[
                "Date"
            ]
            >=
            validation_start
        )
        &
        (
            data[
                "Date"
            ]
            <=
            validation_last_date
        )
    ].copy()


    # =====================================================
    # FINAL TRAIN DATA
    #
    # All available history up to the purged
    # boundary before the test year.
    # =====================================================

    final_train = data[
        data[
            "Date"
        ]
        <=
        validation_last_date
    ].copy()


    # =====================================================
    # TEST YEAR
    # =====================================================

    test = data[
        data[
            "Date"
        ].dt.year
        ==
        test_year
    ].copy()


    return {

        "selection_train":
            selection_train,

        "validation":
            validation,

        "final_train":
            final_train,

        "test":
            test,

        "selection_train_last_date":
            selection_train_last_date,

        "validation_start":
            validation_start,

        "validation_last_date":
            validation_last_date,

        "test_start":
            test_start,
    }


# =========================================================
# RUN WALK FORWARD
# =========================================================

def run_walk_forward():

    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V9 PURGED WALK-FORWARD"
    )

    print(
        "=============================================="
    )


    data = load_dataset()


    print(
        "\nRows:",
        len(data)
    )


    print(
        "Stocks:",
        data[
            "SYMBOL"
        ].nunique()
    )


    print(
        "Features:",
        len(
            FEATURE_COLUMNS
        )
    )


    model_names = [
        "Logistic Regression",
        "Random Forest",
        "Hist Gradient Boosting",
    ]


    all_fold_results = []

    all_prediction_rows = []


    pooled_actual = []

    pooled_probabilities = []

    pooled_baseline_predictions = []


    # =====================================================
    # EACH FOLD
    # =====================================================

    for (
        validation_year,
        test_year,

    ) in FOLDS:

        print(
            "\n\n=============================================="
        )

        print(
            f" FOLD: VALIDATION {validation_year} → TEST {test_year}"
        )

        print(
            "=============================================="
        )


        fold = create_fold(
            data,
            validation_year,
            test_year,
        )


        selection_train = fold[
            "selection_train"
        ]


        validation = fold[
            "validation"
        ]


        final_train = fold[
            "final_train"
        ]


        test = fold[
            "test"
        ]


        print(
            "\nSelection Train Ends:",
            fold[
                "selection_train_last_date"
            ].date()
        )


        print(
            "Validation:",
            fold[
                "validation_start"
            ].date(),
            "→",
            fold[
                "validation_last_date"
            ].date(),
        )


        print(
            "Test Starts:",
            fold[
                "test_start"
            ].date()
        )


        print(
            "\nSelection Train Rows:",
            len(
                selection_train
            )
        )


        print(
            "Validation Rows:",
            len(
                validation
            )
        )


        print(
            "Final Train Rows:",
            len(
                final_train
            )
        )


        print(
            "Test Rows:",
            len(
                test
            )
        )


        # =================================================
        # ARRAYS
        # =================================================

        X_selection_train = (
            selection_train[
                FEATURE_COLUMNS
            ]
            .values
        )


        y_selection_train = (
            selection_train[
                "TARGET_CLASS"
            ]
            .values
        )


        X_validation = (
            validation[
                FEATURE_COLUMNS
            ]
            .values
        )


        y_validation = (
            validation[
                "TARGET_CLASS"
            ]
            .values
        )


        # =================================================
        # VALIDATION MODEL SELECTION
        # =================================================

        validation_results = {}


        for model_name in model_names:

            print(
                f"\nTraining {model_name}..."
            )


            (
                model,
                scaler,

            ) = fit_model(
                model_name,
                X_selection_train,
                y_selection_train,
            )


            validation_probabilities = (
                predict_probabilities(
                    model,
                    scaler,
                    X_validation,
                )
            )


            validation_metrics = (
                calculate_metrics(
                    y_validation,
                    validation_probabilities,
                )
            )


            validation_results[
                model_name
            ] = (
                validation_metrics
            )


            print(
                f"{model_name} Validation:"
            )


            print_metrics(
                validation_metrics
            )


        # =================================================
        # SELECT BY MACRO F1
        # =================================================

        selected_model_name = max(

            validation_results,

            key=lambda name:
                validation_results[
                    name
                ][
                    "macro_f1_percent"
                ],
        )


        print(
            "\nSelected Model:",
            selected_model_name
        )


        print(
            "Selected Validation Macro F1:",
            f'{validation_results[selected_model_name]["macro_f1_percent"]:.2f}%'
        )


        # =================================================
        # FINAL REFIT
        # =================================================

        X_final_train = (
            final_train[
                FEATURE_COLUMNS
            ]
            .values
        )


        y_final_train = (
            final_train[
                "TARGET_CLASS"
            ]
            .values
        )


        X_test = (
            test[
                FEATURE_COLUMNS
            ]
            .values
        )


        y_test = (
            test[
                "TARGET_CLASS"
            ]
            .values
        )


        (
            final_model,
            final_scaler,

        ) = fit_model(
            selected_model_name,
            X_final_train,
            y_final_train,
        )


        test_probabilities = (
            predict_probabilities(
                final_model,
                final_scaler,
                X_test,
            )
        )


        test_metrics = (
            calculate_metrics(
                y_test,
                test_probabilities,
            )
        )


        # =================================================
        # BASELINE
        # =================================================

        (
            majority_class,
            baseline_accuracy,
            baseline_predictions,

        ) = calculate_majority_baseline(
            y_final_train,
            y_test,
        )


        print(
            "\n----------------------------------------------"
        )

        print(
            f" TEST {test_year}"
        )

        print(
            "----------------------------------------------"
        )


        print(
            "Majority Baseline:",
            CLASS_NAMES[
                majority_class
            ],
            f"{baseline_accuracy:.2f}%"
        )


        print_metrics(
            test_metrics
        )


        print(
            "Improvement vs Baseline:",
            f'{test_metrics["accuracy_percent"] - baseline_accuracy:+.2f} pp'
        )


        # =================================================
        # STORE POOLED RESULTS
        # =================================================

        pooled_actual.extend(
            y_test.tolist()
        )


        pooled_probabilities.extend(
            test_probabilities.tolist()
        )


        pooled_baseline_predictions.extend(
            baseline_predictions.tolist()
        )


        predicted_classes = np.argmax(
            test_probabilities,
            axis=1,
        )


        for index in range(
            len(test)
        ):

            row = test.iloc[
                index
            ]


            actual_class = int(
                y_test[index]
            )


            predicted_class = int(
                predicted_classes[index]
            )


            all_prediction_rows.append(
                {

                    "fold_test_year":
                        test_year,

                    "Date":
                        str(
                            pd.Timestamp(
                                row["Date"]
                            ).date()
                        ),

                    "SYMBOL":
                        row["SYMBOL"],

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

                    "UNDERPERFORM_PROBABILITY":
                        float(
                            test_probabilities[
                                index,
                                0
                            ]
                        ),

                    "NEUTRAL_PROBABILITY":
                        float(
                            test_probabilities[
                                index,
                                1
                            ]
                        ),

                    "OUTPERFORM_PROBABILITY":
                        float(
                            test_probabilities[
                                index,
                                2
                            ]
                        ),

                    "CONFIDENCE":
                        float(
                            np.max(
                                test_probabilities[
                                    index
                                ]
                            )
                        ),

                    "SELECTED_MODEL":
                        selected_model_name,
                }
            )


        all_fold_results.append(
            {

                "validation_year":
                    validation_year,

                "test_year":
                    test_year,

                "selected_model":
                    selected_model_name,

                "selection_train_last_date":
                    str(
                        fold[
                            "selection_train_last_date"
                        ].date()
                    ),

                "validation_start":
                    str(
                        fold[
                            "validation_start"
                        ].date()
                    ),

                "validation_last_date":
                    str(
                        fold[
                            "validation_last_date"
                        ].date()
                    ),

                "test_start":
                    str(
                        fold[
                            "test_start"
                        ].date()
                    ),

                "validation_results":
                    validation_results,

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

                "test_metrics":
                    test_metrics,
            }
        )


    # =====================================================
    # POOLED WALK-FORWARD METRICS
    # =====================================================

    pooled_actual = np.asarray(
        pooled_actual,
        dtype=int,
    )


    pooled_probabilities = np.asarray(
        pooled_probabilities,
        dtype=float,
    )


    pooled_baseline_predictions = np.asarray(
        pooled_baseline_predictions,
        dtype=int,
    )


    pooled_metrics = calculate_metrics(
        pooled_actual,
        pooled_probabilities,
    )


    pooled_baseline_accuracy = (
        np.mean(
            pooled_baseline_predictions
            ==
            pooled_actual
        )
        *
        100
    )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print(
        "\n\n=============================================="
    )

    print(
        " V9 AGGREGATED WALK-FORWARD RESULTS"
    )

    print(
        "=============================================="
    )


    print(
        "\nPooled Test Samples:",
        len(
            pooled_actual
        )
    )


    print(
        "\nPooled Majority Baseline:",
        f"{pooled_baseline_accuracy:.2f}%"
    )


    print(
        "Model Accuracy:",
        f'{pooled_metrics["accuracy_percent"]:.2f}%'
    )


    print(
        "Improvement:",
        f'{pooled_metrics["accuracy_percent"] - pooled_baseline_accuracy:+.2f} pp'
    )


    print(
        "Macro F1:",
        f'{pooled_metrics["macro_f1_percent"]:.2f}%'
    )


    print(
        "Balanced Accuracy:",
        f'{pooled_metrics["balanced_accuracy_percent"]:.2f}%'
    )


    print(
        "High Confidence Accuracy:",
        f'{pooled_metrics["high_confidence_accuracy_percent"]:.2f}%'
    )


    print(
        "High Confidence Coverage:",
        f'{pooled_metrics["high_confidence_coverage_percent"]:.2f}%'
    )


    print(
        "Brier Score:",
        pooled_metrics[
            "brier_score"
        ]
    )


    # =====================================================
    # PER CLASS SUMMARY
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " AGGREGATED PER-CLASS RESULTS"
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
            pooled_metrics[
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
        " AGGREGATED CONFUSION MATRIX"
    )

    print(
        "=============================================="
    )


    matrix = (
        pooled_metrics[
            "confusion_matrix"
        ]
    )


    print(
        "\n                  UNDER     NEUTRAL       OVER"
    )


    row_names = [
        "UNDER",
        "NEUTRAL",
        "OVER",
    ]


    for index, row in enumerate(
        matrix
    ):

        print(
            f"{row_names[index]:<12}"
            f"{row[0]:>10}"
            f"{row[1]:>12}"
            f"{row[2]:>12}"
        )


    # =====================================================
    # SAVE PREDICTIONS
    # =====================================================

    prediction_df = pd.DataFrame(
        all_prediction_rows
    )


    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )


    # =====================================================
    # SAVE JSON
    # =====================================================

    output = {

        "experiment":
            "StockVision V9 Purged Walk-Forward",

        "target":
            "5-day excess return versus NIFTY 50",

        "target_threshold_percent":
            1.0,

        "feature_count":
            len(
                FEATURE_COLUMNS
            ),

        "feature_columns":
            FEATURE_COLUMNS,

        "purge_days":
            TARGET_HORIZON,

        "folds":
            all_fold_results,

        "aggregated": {

            "pooled_test_samples":
                int(
                    len(
                        pooled_actual
                    )
                ),

            "pooled_majority_baseline_accuracy_percent":
                round(
                    float(
                        pooled_baseline_accuracy
                    ),
                    2,
                ),

            "metrics":
                pooled_metrics,
        },
    }


    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )


    print(
        "\n=============================================="
    )

    print(
        " V9 WALK-FORWARD COMPLETE"
    )

    print(
        "=============================================="
    )


    print(
        "\nResults:"
    )


    print(
        RESULT_PATH
    )


    print(
        "\nPredictions:"
    )


    print(
        PREDICTIONS_PATH
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    run_walk_forward()