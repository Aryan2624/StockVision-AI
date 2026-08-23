import os
import json

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

import joblib


# =========================================================
# V8 IMPORT
# =========================================================

try:
    from .advanced_dataset_v8 import (
        FEATURE_COLUMNS,
        TARGET_HORIZON,
    )

except ImportError:
    from advanced_dataset_v8 import (
        FEATURE_COLUMNS,
        TARGET_HORIZON,
    )


# =========================================================
# CONFIG
# =========================================================

NUM_CLASSES = 3

RANDOM_STATE = 42

HIGH_CONFIDENCE_THRESHOLD = 0.60


CLASS_NAMES = {
    0: "BEARISH",
    1: "NEUTRAL",
    2: "BULLISH",
}


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "stockvision_v8_advanced_dataset.csv",
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
    "stockvision_v8_benchmark_results.json",
)


FINAL_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v8_benchmark_best_model.pkl",
)


FINAL_SCALER_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v8_benchmark_scaler.pkl",
)


# =========================================================
# LOAD DATA
# =========================================================

def load_dataset():

    if not os.path.exists(
        DATASET_PATH
    ):
        raise FileNotFoundError(
            f"V8 dataset not found:\n{DATASET_PATH}"
        )


    df = pd.read_csv(
        DATASET_PATH
    )


    df["Date"] = pd.to_datetime(
        df["Date"]
    )


    required_columns = [
        "Date",
        "SYMBOL",
        "SECTOR",
        "TARGET_CLASS",
        *FEATURE_COLUMNS,
    ]


    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]


    if missing:

        raise ValueError(
            "Missing columns:\n"
            +
            ", ".join(missing)
        )


    for feature in FEATURE_COLUMNS:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )


    df["TARGET_CLASS"] = pd.to_numeric(
        df["TARGET_CLASS"],
        errors="coerce",
    )


    df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
        inplace=True,
    )


    df.dropna(
        subset=[
            "Date",
            "TARGET_CLASS",
            *FEATURE_COLUMNS,
        ],
        inplace=True,
    )


    df["TARGET_CLASS"] = (
        df["TARGET_CLASS"]
        .astype(int)
    )


    df.sort_values(
        [
            "Date",
            "SYMBOL",
        ],
        inplace=True,
    )


    df.reset_index(
        drop=True,
        inplace=True,
    )


    return df


# =========================================================
# CHRONOLOGICAL SPLIT
#
# 70% train
# 15% validation
# 15% test
#
# 5-day target => 5-trading-date purge.
# =========================================================

def create_split(
    df,
):

    dates = np.array(
        sorted(
            df["Date"]
            .unique()
        )
    )


    total_dates = len(
        dates
    )


    train_boundary = int(
        total_dates * 0.70
    )


    validation_boundary = int(
        total_dates * 0.85
    )


    train_last_index = (
        train_boundary
        -
        TARGET_HORIZON
        -
        1
    )


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
        dates[
            train_last_index
        ]
    )


    validation_start_date = pd.Timestamp(
        dates[
            validation_start_index
        ]
    )


    validation_last_date = pd.Timestamp(
        dates[
            validation_last_index
        ]
    )


    test_start_date = pd.Timestamp(
        dates[
            test_start_index
        ]
    )


    train_df = df[
        df["Date"]
        <=
        train_last_date
    ].copy()


    validation_df = df[
        (
            df["Date"]
            >=
            validation_start_date
        )
        &
        (
            df["Date"]
            <=
            validation_last_date
        )
    ].copy()


    test_df = df[
        df["Date"]
        >=
        test_start_date
    ].copy()


    return (
        train_df,
        validation_df,
        test_df,
        {
            "train_last_date":
                str(
                    train_last_date.date()
                ),

            "validation_start_date":
                str(
                    validation_start_date.date()
                ),

            "validation_last_date":
                str(
                    validation_last_date.date()
                ),

            "test_start_date":
                str(
                    test_start_date.date()
                ),
        }
    )


# =========================================================
# SAMPLE WEIGHTS
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


    total = len(
        labels
    )


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
            class_weights[
                int(label)
            ]

            for label in labels
        ],
        dtype=float,
    )


    return (
        class_weights,
        sample_weights,
    )


# =========================================================
# BASELINE
# =========================================================

def calculate_majority_baseline(
    training_labels,
    evaluation_labels,
):

    counts = np.bincount(
        training_labels,
        minlength=NUM_CLASSES,
    )


    majority_class = int(
        np.argmax(
            counts
        )
    )


    accuracy = (
        np.mean(
            evaluation_labels
            ==
            majority_class
        )
        *
        100
    )


    return (
        majority_class,
        float(
            accuracy
        ),
    )


# =========================================================
# METRICS
# =========================================================

def evaluate_predictions(
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
            CLASS_NAMES[
                class_id
            ]
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


    confidence = np.max(
        probabilities,
        axis=1,
    )


    high_confidence_mask = (
        confidence
        >=
        HIGH_CONFIDENCE_THRESHOLD
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
# PRINT METRICS
# =========================================================

def print_metrics(
    title,
    metrics,
    baseline,
):

    print(
        "\n=============================================="
    )

    print(
        f" {title}"
    )

    print(
        "=============================================="
    )


    print(
        "\nAccuracy:",
        f'{metrics["accuracy_percent"]:.2f}%'
    )


    print(
        "Majority Baseline:",
        f"{baseline:.2f}%"
    )


    print(
        "Improvement:",
        f'{metrics["accuracy_percent"] - baseline:+.2f} pp'
    )


    print(
        "Balanced Accuracy:",
        f'{metrics["balanced_accuracy_percent"]:.2f}%'
    )


    print(
        "Macro F1:",
        f'{metrics["macro_f1_percent"]:.2f}%'
    )


    print(
        "\nHigh Confidence Accuracy:",
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
# MODEL BUILDERS
# =========================================================

def build_logistic():

    return LogisticRegression(
        max_iter=2500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_random_forest():

    return RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def build_hist_gradient():

    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


# =========================================================
# TRAIN CANDIDATE ON TRAIN ONLY
# =========================================================

def train_candidate(
    model_name,
    X_train,
    y_train,
    X_validation,
):

    if model_name == "Logistic Regression":

        scaler = StandardScaler()

        X_train_final = (
            scaler.fit_transform(
                X_train
            )
        )

        X_validation_final = (
            scaler.transform(
                X_validation
            )
        )

        model = build_logistic()

        model.fit(
            X_train_final,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_validation_final
            )
        )

        return (
            model,
            scaler,
            probabilities,
        )


    if model_name == "Random Forest":

        model = build_random_forest()

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_validation
            )
        )

        return (
            model,
            None,
            probabilities,
        )


    if model_name == "Hist Gradient Boosting":

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

        probabilities = (
            model.predict_proba(
                X_validation
            )
        )

        return (
            model,
            None,
            probabilities,
        )


    raise ValueError(
        f"Unknown model: {model_name}"
    )


# =========================================================
# FINAL REFIT
# =========================================================

def final_refit(
    model_name,
    X_combined,
    y_combined,
    X_test,
):

    if model_name == "Logistic Regression":

        scaler = StandardScaler()

        X_combined_final = (
            scaler.fit_transform(
                X_combined
            )
        )

        X_test_final = (
            scaler.transform(
                X_test
            )
        )

        model = build_logistic()

        model.fit(
            X_combined_final,
            y_combined,
        )

        probabilities = (
            model.predict_proba(
                X_test_final
            )
        )


    elif model_name == "Random Forest":

        scaler = None

        model = build_random_forest()

        model.fit(
            X_combined,
            y_combined,
        )

        probabilities = (
            model.predict_proba(
                X_test
            )
        )


    elif model_name == "Hist Gradient Boosting":

        scaler = None

        model = build_hist_gradient()

        (
            _,
            sample_weights,

        ) = calculate_sample_weights(
            y_combined
        )

        model.fit(
            X_combined,
            y_combined,
            sample_weight=
                sample_weights,
        )

        probabilities = (
            model.predict_proba(
                X_test
            )
        )


    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


    return (
        model,
        scaler,
        probabilities,
    )


# =========================================================
# RUN V8 BENCHMARK
# =========================================================

def run_benchmark():

    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V8 PROPER BENCHMARK"
    )

    print(
        "=============================================="
    )


    df = load_dataset()


    print(
        "\nRows:",
        len(df)
    )


    print(
        "Stocks:",
        df[
            "SYMBOL"
        ].nunique()
    )


    print(
        "Features:",
        len(
            FEATURE_COLUMNS
        )
    )


    (
        train_df,
        validation_df,
        test_df,
        split_info,

    ) = create_split(
        df
    )


    print(
        "\nChronological Split:"
    )


    print(
        "Train ends:",
        split_info[
            "train_last_date"
        ]
    )


    print(
        "Validation:",
        split_info[
            "validation_start_date"
        ],
        "→",
        split_info[
            "validation_last_date"
        ],
    )


    print(
        "Test starts:",
        split_info[
            "test_start_date"
        ]
    )


    print(
        "\nTrain Rows:",
        len(train_df)
    )


    print(
        "Validation Rows:",
        len(validation_df)
    )


    print(
        "Test Rows:",
        len(test_df)
    )


    # =====================================================
    # DATA ARRAYS
    # =====================================================

    X_train = train_df[
        FEATURE_COLUMNS
    ].values


    y_train = train_df[
        "TARGET_CLASS"
    ].values


    X_validation = validation_df[
        FEATURE_COLUMNS
    ].values


    y_validation = validation_df[
        "TARGET_CLASS"
    ].values


    X_test = test_df[
        FEATURE_COLUMNS
    ].values


    y_test = test_df[
        "TARGET_CLASS"
    ].values


    # =====================================================
    # VALIDATION BASELINE
    # =====================================================

    (
        validation_majority,
        validation_baseline,

    ) = calculate_majority_baseline(
        y_train,
        y_validation,
    )


    print(
        "\nValidation Majority Class:",
        CLASS_NAMES[
            validation_majority
        ]
    )


    print(
        "Validation Baseline:",
        f"{validation_baseline:.2f}%"
    )


    # =====================================================
    # CANDIDATES
    # =====================================================

    model_names = [
        "Logistic Regression",
        "Random Forest",
        "Hist Gradient Boosting",
    ]


    validation_results = {}


    # =====================================================
    # MODEL SELECTION USING VALIDATION ONLY
    # =====================================================

    for model_name in model_names:

        print(
            f"\nTraining {model_name}..."
        )


        (
            _,
            _,
            probabilities,

        ) = train_candidate(
            model_name=
                model_name,

            X_train=
                X_train,

            y_train=
                y_train,

            X_validation=
                X_validation,
        )


        metrics = evaluate_predictions(
            y_validation,
            probabilities,
        )


        validation_results[
            model_name
        ] = metrics


        print_metrics(
            title=
                f"{model_name} - VALIDATION",

            metrics=
                metrics,

            baseline=
                validation_baseline,
        )


    # =====================================================
    # SELECT BEST USING VALIDATION MACRO F1
    # =====================================================

    best_model_name = max(

        validation_results,

        key=lambda name:
            validation_results[
                name
            ][
                "macro_f1_percent"
            ],
    )


    print(
        "\n=============================================="
    )

    print(
        " VALIDATION WINNER"
    )

    print(
        "=============================================="
    )


    print(
        "\nModel:",
        best_model_name
    )


    print(
        "Validation Accuracy:",
        f'{validation_results[best_model_name]["accuracy_percent"]:.2f}%'
    )


    print(
        "Validation Macro F1:",
        f'{validation_results[best_model_name]["macro_f1_percent"]:.2f}%'
    )


    print(
        "Validation Balanced Accuracy:",
        f'{validation_results[best_model_name]["balanced_accuracy_percent"]:.2f}%'
    )


    # =====================================================
    # FINAL TRAINING SET
    #
    # Training + validation.
    # Purged dates remain excluded because they are
    # not present in either dataframe.
    # =====================================================

    combined_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )


    X_combined = combined_df[
        FEATURE_COLUMNS
    ].values


    y_combined = combined_df[
        "TARGET_CLASS"
    ].values


    # =====================================================
    # TEST BASELINE
    # =====================================================

    (
        test_majority,
        test_baseline,

    ) = calculate_majority_baseline(
        y_combined,
        y_test,
    )


    print(
        "\nFinal Training Rows:",
        len(
            combined_df
        )
    )


    print(
        "Test Majority Class:",
        CLASS_NAMES[
            test_majority
        ]
    )


    print(
        "Test Majority Baseline:",
        f"{test_baseline:.2f}%"
    )


    # =====================================================
    # REFIT VALIDATION WINNER
    # =====================================================

    print(
        f"\nRetraining {best_model_name} "
        "on Train + Validation..."
    )


    (
        final_model,
        final_scaler,
        test_probabilities,

    ) = final_refit(
        model_name=
            best_model_name,

        X_combined=
            X_combined,

        y_combined=
            y_combined,

        X_test=
            X_test,
    )


    # =====================================================
    # FINAL TEST
    # =====================================================

    test_metrics = evaluate_predictions(
        y_test,
        test_probabilities,
    )


    print(
        "\n\n=============================================="
    )

    print(
        " FINAL V8 OUT-OF-SAMPLE TEST"
    )

    print(
        "=============================================="
    )


    print(
        "\nSelected Model:",
        best_model_name
    )


    print_metrics(
        title=
            "FINAL TEST RESULT",

        metrics=
            test_metrics,

        baseline=
            test_baseline,
    )


    # =====================================================
    # CLASS METRICS
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " PER-CLASS TEST RESULTS"
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
            test_metrics[
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
        " TEST CONFUSION MATRIX"
    )

    print(
        "=============================================="
    )


    matrix = (
        test_metrics[
            "confusion_matrix"
        ]
    )


    print(
        "\n            BEAR     NEUTRAL      BULL"
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
            f"{labels[index]:<10}"
            f"{row[0]:>8}"
            f"{row[1]:>12}"
            f"{row[2]:>10}"
        )


    # =====================================================
    # SAVE BENCHMARK MODEL
    # =====================================================

    joblib.dump(
        final_model,
        FINAL_MODEL_PATH,
    )


    if final_scaler is not None:

        joblib.dump(
            final_scaler,
            FINAL_SCALER_PATH,
        )


    # =====================================================
    # SAVE RESULTS
    # =====================================================

    result_payload = {

        "experiment":
            "StockVision V8 Proper Benchmark",

        "feature_count":
            len(
                FEATURE_COLUMNS
            ),

        "features":
            FEATURE_COLUMNS,

        "target":
            "ATR-adjusted 5-day 3-class movement",

        "split":
            split_info,

        "validation_majority_class":
            CLASS_NAMES[
                validation_majority
            ],

        "validation_baseline_accuracy_percent":
            round(
                validation_baseline,
                2,
            ),

        "validation_results":
            validation_results,

        "selected_model":
            best_model_name,

        "test_majority_class":
            CLASS_NAMES[
                test_majority
            ],

        "test_baseline_accuracy_percent":
            round(
                test_baseline,
                2,
            ),

        "final_test_metrics":
            test_metrics,
    }


    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result_payload,
            file,
            indent=4,
        )


    print(
        "\n=============================================="
    )

    print(
        " V8 BENCHMARK COMPLETE"
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
        "\nBenchmark Model:"
    )


    print(
        FINAL_MODEL_PATH
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    run_benchmark()