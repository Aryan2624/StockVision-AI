import os
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# =========================================================
# FEATURE IMPORT
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
# CONFIG
# =========================================================

TARGET_HORIZON = 5

RANDOM_STATE = 42

NUM_CLASSES = 3


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
    "stockvision_v6_universal_dataset.csv",
)


MODELS_DIR = os.path.join(
    BASE_DIR,
    "models",
)


RESULT_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v6_benchmark_results.json",
)


V6_METRICS_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_universal_v6_metrics.json",
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    if not os.path.exists(
        DATASET_PATH
    ):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
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
            "Missing columns: "
            +
            ", ".join(
                missing
            )
        )


    for column in FEATURE_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
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
# Same principle as V6:
#
# 70% train
# 15% validation
# 15% test
#
# 5-day purge before validation/test boundaries.
# =========================================================

def chronological_split(
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
        total_dates
        *
        0.70
    )


    validation_boundary = int(
        total_dates
        *
        0.85
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
# CLASS WEIGHT → SAMPLE WEIGHT
# =========================================================

def create_sample_weights(
    labels,
):

    labels = np.asarray(
        labels
    )


    total = len(
        labels
    )


    counts = np.bincount(
        labels,
        minlength=
            NUM_CLASSES,
    )


    class_weights = {}


    for class_id in range(
        NUM_CLASSES
    ):

        if counts[
            class_id
        ] > 0:

            class_weights[
                class_id
            ] = (

                total

                /

                (
                    NUM_CLASSES
                    *
                    counts[
                        class_id
                    ]
                )
            )

        else:

            class_weights[
                class_id
            ] = 1.0


    sample_weights = np.array(
        [
            class_weights[
                int(
                    label
                )
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
# EVALUATION
# =========================================================

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test,
):

    predictions = model.predict(
        X_test
    )


    accuracy = (
        accuracy_score(
            y_test,
            predictions,
        )
        *
        100
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions,
        )
        *
        100
    )


    macro_f1 = (
        f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
        *
        100
    )


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            labels=[
                0,
                1,
                2,
            ],
            zero_division=0,
        )
    )


    matrix = confusion_matrix(
        y_test,
        predictions,
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
                        precision[
                            class_id
                        ]
                        *
                        100
                    ),
                    2,
                ),

            "recall_percent":
                round(
                    float(
                        recall[
                            class_id
                        ]
                        *
                        100
                    ),
                    2,
                ),

            "f1_percent":
                round(
                    float(
                        f1[
                            class_id
                        ]
                        *
                        100
                    ),
                    2,
                ),

            "support":
                int(
                    support[
                        class_id
                    ]
                ),
        }


    return {

        "model":
            model_name,

        "accuracy_percent":
            round(
                float(
                    accuracy
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

        "macro_f1_percent":
            round(
                float(
                    macro_f1
                ),
                2,
            ),

        "class_metrics":
            class_metrics,

        "confusion_matrix":
            matrix.tolist(),
    }


# =========================================================
# PRINT RESULT
# =========================================================

def print_result(
    result,
    baseline_accuracy,
):

    print(
        "\n=============================================="
    )

    print(
        f' {result["model"]}'
    )

    print(
        "=============================================="
    )


    print(
        "\nAccuracy:",
        f'{result["accuracy_percent"]:.2f}%'
    )


    print(
        "Majority Baseline:",
        f"{baseline_accuracy:.2f}%"
    )


    improvement = (
        result[
            "accuracy_percent"
        ]
        -
        baseline_accuracy
    )


    print(
        "Improvement:",
        f"{improvement:+.2f} pp"
    )


    print(
        "Balanced Accuracy:",
        f'{result["balanced_accuracy_percent"]:.2f}%'
    )


    print(
        "Macro F1:",
        f'{result["macro_f1_percent"]:.2f}%'
    )


    print(
        "\nPER CLASS:"
    )


    for class_name in [
        "BEARISH",
        "NEUTRAL",
        "BULLISH",
    ]:

        values = (
            result[
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
        "\nConfusion Matrix:"
    )


    matrix = result[
        "confusion_matrix"
    ]


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


# =========================================================
# RUN BENCHMARK
# =========================================================

def run_benchmark():

    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V6 MODEL BENCHMARK"
    )

    print(
        "=============================================="
    )


    df = load_data()


    print(
        "\nRows:",
        len(
            df
        )
    )


    print(
        "Stocks:",
        df[
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

    (
        train_df,
        validation_df,
        test_df,
        split_info,

    ) = chronological_split(
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
        "\nTraining Rows:",
        len(
            train_df
        )
    )


    print(
        "Validation Rows:",
        len(
            validation_df
        )
    )


    print(
        "Test Rows:",
        len(
            test_df
        )
    )


    # =====================================================
    # X / Y
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
    # BASELINE
    # =====================================================

    training_counts = np.bincount(
        y_train,
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
            y_test
            ==
            majority_class
        )
        *
        100
    )


    print(
        "\nMajority Class:",
        CLASS_NAMES[
            majority_class
        ]
    )


    print(
        "Test Majority Baseline:",
        f"{baseline_accuracy:.2f}%"
    )


    # =====================================================
    # CLASS WEIGHTS
    # =====================================================

    (
        class_weights,
        gradient_sample_weights,

    ) = create_sample_weights(
        y_train
    )


    print(
        "\nClass Weights:"
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
    # STANDARD SCALER FOR LOGISTIC REGRESSION
    # =====================================================

    scaler = StandardScaler()


    X_train_scaled = scaler.fit_transform(
        X_train
    )


    X_validation_scaled = scaler.transform(
        X_validation
    )


    X_test_scaled = scaler.transform(
        X_test
    )


    results = []


    # =====================================================
    # 1. LOGISTIC REGRESSION
    # =====================================================

    print(
        "\nTraining Logistic Regression..."
    )


    logistic_model = LogisticRegression(

        max_iter=
            2000,

        class_weight=
            "balanced",

        random_state=
            RANDOM_STATE,
    )


    logistic_model.fit(
        X_train_scaled,
        y_train,
    )


    logistic_result = evaluate_model(

        "Logistic Regression",

        logistic_model,

        X_test_scaled,

        y_test,
    )


    results.append(
        logistic_result
    )


    print_result(
        logistic_result,
        baseline_accuracy,
    )


    # =====================================================
    # 2. RANDOM FOREST
    # =====================================================

    print(
        "\nTraining Random Forest..."
    )


    random_forest = RandomForestClassifier(

        n_estimators=
            300,

        max_depth=
            12,

        min_samples_leaf=
            10,

        max_features=
            "sqrt",

        class_weight=
            "balanced",

        n_jobs=
            -1,

        random_state=
            RANDOM_STATE,
    )


    random_forest.fit(
        X_train,
        y_train,
    )


    random_forest_result = evaluate_model(

        "Random Forest",

        random_forest,

        X_test,

        y_test,
    )


    results.append(
        random_forest_result
    )


    print_result(
        random_forest_result,
        baseline_accuracy,
    )


    # =====================================================
    # 3. GRADIENT BOOSTING
    # =====================================================

    print(
        "\nTraining Gradient Boosting..."
    )


    gradient_boosting = GradientBoostingClassifier(

        n_estimators=
            150,

        learning_rate=
            0.05,

        max_depth=
            3,

        min_samples_leaf=
            10,

        subsample=
            0.85,

        random_state=
            RANDOM_STATE,
    )


    gradient_boosting.fit(

        X_train,

        y_train,

        sample_weight=
            gradient_sample_weights,
    )


    gradient_result = evaluate_model(

        "Gradient Boosting",

        gradient_boosting,

        X_test,

        y_test,
    )


    results.append(
        gradient_result
    )


    print_result(
        gradient_result,
        baseline_accuracy,
    )


    # =====================================================
    # LOAD V6 BiLSTM RESULT
    # =====================================================

    bilstm_result = None


    if os.path.exists(
        V6_METRICS_PATH
    ):

        try:

            with open(
                V6_METRICS_PATH,
                "r",
                encoding="utf-8",
            ) as file:

                v6_data = json.load(
                    file
                )


            metrics = v6_data.get(
                "metrics",
                {}
            )


            bilstm_result = {

                "model":
                    "Universal V6 BiLSTM",

                "accuracy_percent":
                    metrics.get(
                        "accuracy_percent"
                    ),

                "balanced_accuracy_percent":
                    metrics.get(
                        "balanced_accuracy_percent"
                    ),

                "macro_f1_percent":
                    metrics.get(
                        "macro_f1_percent"
                    ),
            }


        except Exception as error:

            print(
                "\nCould not load V6 metrics:",
                error
            )


    # =====================================================
    # SUMMARY
    # =====================================================

    print(
        "\n\n=============================================="
    )

    print(
        " FINAL BENCHMARK SUMMARY"
    )

    print(
        "=============================================="
    )


    print(
        "\nMajority Baseline:",
        f"{baseline_accuracy:.2f}%"
    )


    print(
        "\nMODEL RESULTS"
    )


    for result in results:

        print(
            f'\n{result["model"]}'
        )

        print(
            "Accuracy:",
            f'{result["accuracy_percent"]:.2f}%'
        )

        print(
            "Macro F1:",
            f'{result["macro_f1_percent"]:.2f}%'
        )

        print(
            "Balanced Accuracy:",
            f'{result["balanced_accuracy_percent"]:.2f}%'
        )


    if bilstm_result:

        print(
            "\nUniversal V6 BiLSTM"
        )

        print(
            "Accuracy:",
            f'{bilstm_result["accuracy_percent"]:.2f}%'
        )

        print(
            "Macro F1:",
            f'{bilstm_result["macro_f1_percent"]:.2f}%'
        )

        print(
            "Balanced Accuracy:",
            f'{bilstm_result["balanced_accuracy_percent"]:.2f}%'
        )


    # =====================================================
    # BEST CLASSICAL MODEL
    # =====================================================

    best_model = max(

        results,

        key=lambda item:
            item[
                "macro_f1_percent"
            ],
    )


    print(
        "\n=============================================="
    )

    print(
        " BEST CLASSICAL MODEL"
    )

    print(
        "=============================================="
    )


    print(
        "\nModel:",
        best_model[
            "model"
        ]
    )


    print(
        "Accuracy:",
        f'{best_model["accuracy_percent"]:.2f}%'
    )


    print(
        "Macro F1:",
        f'{best_model["macro_f1_percent"]:.2f}%'
    )


    print(
        "Balanced Accuracy:",
        f'{best_model["balanced_accuracy_percent"]:.2f}%'
    )


    # =====================================================
    # SAVE
    # =====================================================

    output = {

        "experiment":
            "StockVision V6 Classical Model Benchmark",

        "features":
            FEATURE_COLUMNS,

        "feature_count":
            len(
                FEATURE_COLUMNS
            ),

        "target":
            "5-Day Bearish / Neutral / Bullish",

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

        "split":
            split_info,

        "results":
            results,

        "v6_bilstm":
            bilstm_result,

        "best_classical_model":
            best_model[
                "model"
            ],
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
        "\nResults saved:"
    )


    print(
        RESULT_PATH
    )


    print(
        "\n=============================================="
    )

    print(
        " V6 BENCHMARK COMPLETE"
    )

    print(
        "=============================================="
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    run_benchmark()