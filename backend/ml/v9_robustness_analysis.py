import os
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.inspection import (
    permutation_importance,
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

NUM_CLASSES = 3

TARGET_HORIZON = 5

RANDOM_STATE = 42

RELATIVE_MOMENTUM_THRESHOLD = 0.01


CLASS_NAMES = {
    0: "UNDERPERFORM",
    1: "NEUTRAL",
    2: "OUTPERFORM",
}


CLASS_TO_ID = {
    value: key
    for key, value
    in CLASS_NAMES.items()
}


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODELS_DIR = os.path.join(
    BASE_DIR,
    "models",
)


DATASETS_DIR = os.path.join(
    BASE_DIR,
    "datasets",
)


PREDICTIONS_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_walk_forward_predictions.csv",
)


WALK_FORWARD_RESULTS_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_walk_forward_results.json",
)


DATASET_PATH = os.path.join(
    DATASETS_DIR,
    "stockvision_v9_relative_dataset.csv",
)


RESULT_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_robustness_results.json",
)


PER_STOCK_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_robustness_per_stock.csv",
)


FEATURE_IMPORTANCE_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_feature_importance.csv",
)


# =========================================================
# LOAD WALK-FORWARD PREDICTIONS
# =========================================================

def load_predictions():

    if not os.path.exists(
        PREDICTIONS_PATH
    ):
        raise FileNotFoundError(
            f"Predictions not found:\n{PREDICTIONS_PATH}"
        )


    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )


    required_columns = [
        "fold_test_year",
        "Date",
        "SYMBOL",
        "ACTUAL_CLASS",
        "PREDICTED_CLASS",
        "UNDERPERFORM_PROBABILITY",
        "NEUTRAL_PROBABILITY",
        "OUTPERFORM_PROBABILITY",
        "CONFIDENCE",
    ]


    missing = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]


    if missing:
        raise ValueError(
            "Prediction CSV missing columns:\n"
            +
            ", ".join(missing)
        )


    predictions["Date"] = pd.to_datetime(
        predictions["Date"]
    )


    predictions["fold_test_year"] = (
        predictions["fold_test_year"]
        .astype(int)
    )


    predictions["ACTUAL_CLASS"] = (
        predictions["ACTUAL_CLASS"]
        .astype(int)
    )


    predictions["PREDICTED_CLASS"] = (
        predictions["PREDICTED_CLASS"]
        .astype(int)
    )


    predictions["CONFIDENCE"] = (
        pd.to_numeric(
            predictions["CONFIDENCE"],
            errors="coerce",
        )
    )


    return predictions


# =========================================================
# LOAD V9 DATASET
# =========================================================

def load_dataset():

    if not os.path.exists(
        DATASET_PATH
    ):
        raise FileNotFoundError(
            f"V9 dataset not found:\n{DATASET_PATH}"
        )


    dataset = pd.read_csv(
        DATASET_PATH
    )


    dataset["Date"] = pd.to_datetime(
        dataset["Date"]
    )


    required_columns = [
        "Date",
        "SYMBOL",
        "TARGET_CLASS",
        *FEATURE_COLUMNS,
    ]


    # Remove accidental duplicates from required list
    required_columns = list(
        dict.fromkeys(
            required_columns
        )
    )


    missing = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]


    if missing:
        raise ValueError(
            "Dataset missing columns:\n"
            +
            ", ".join(missing)
        )


    for feature in FEATURE_COLUMNS:

        dataset[feature] = pd.to_numeric(
            dataset[feature],
            errors="coerce",
        )


    dataset["TARGET_CLASS"] = pd.to_numeric(
        dataset["TARGET_CLASS"],
        errors="coerce",
    )


    dataset.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
        inplace=True,
    )


    dataset.dropna(
        subset=[
            "Date",
            "SYMBOL",
            "TARGET_CLASS",
            *FEATURE_COLUMNS,
        ],
        inplace=True,
    )


    dataset["TARGET_CLASS"] = (
        dataset["TARGET_CLASS"]
        .astype(int)
    )


    dataset.sort_values(
        by=[
            "Date",
            "SYMBOL",
        ],
        inplace=True,
    )


    dataset.reset_index(
        drop=True,
        inplace=True,
    )


    return dataset


# =========================================================
# LOAD WALK-FORWARD RESULTS
# =========================================================

def load_walk_forward_results():

    if not os.path.exists(
        WALK_FORWARD_RESULTS_PATH
    ):
        raise FileNotFoundError(
            "Walk-forward JSON not found:\n"
            f"{WALK_FORWARD_RESULTS_PATH}"
        )


    with open(
        WALK_FORWARD_RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        results = json.load(
            file
        )


    return results


# =========================================================
# MERGE PREDICTIONS + V9 FEATURES
#
# IMPORTANT FIX:
#
# RELATIVE_RETURN_5 is already inside FEATURE_COLUMNS.
#
# We therefore DO NOT add it separately.
#
# Otherwise pandas creates two columns with the
# same name and:
#
# data["RELATIVE_RETURN_5"]
#
# becomes a DataFrame instead of a Series.
# =========================================================

def merge_data(
    predictions,
    dataset,
):

    feature_columns = [
        "Date",
        "SYMBOL",
        "TARGET_CLASS",
        *FEATURE_COLUMNS,
    ]


    # =====================================================
    # REMOVE DUPLICATE COLUMN NAMES SAFELY
    # =====================================================

    feature_columns = list(
        dict.fromkeys(
            feature_columns
        )
    )


    feature_data = dataset[
        feature_columns
    ].copy()


    feature_data = (
        feature_data
        .drop_duplicates(
            subset=[
                "Date",
                "SYMBOL",
            ],
            keep="last",
        )
    )


    merged = predictions.merge(

        feature_data,

        on=[
            "Date",
            "SYMBOL",
        ],

        how="left",

        validate="many_to_one",
    )


    # =====================================================
    # CHECK MATCHING
    # =====================================================

    if merged["TARGET_CLASS"].isna().any():

        missing_rows = int(
            merged["TARGET_CLASS"]
            .isna()
            .sum()
        )


        raise ValueError(
            f"{missing_rows} prediction rows could not "
            "be matched with the V9 dataset."
        )


    # =====================================================
    # VERIFY TARGETS
    # =====================================================

    mismatch = (
        merged["ACTUAL_CLASS"]
        !=
        merged["TARGET_CLASS"]
        .astype(int)
    )


    mismatch_count = int(
        mismatch.sum()
    )


    if mismatch_count > 0:

        print(
            "\nWARNING:"
        )

        print(
            mismatch_count,
            "prediction targets do not match dataset targets."
        )

    else:

        print(
            "\nPrediction targets successfully matched "
            "with V9 dataset."
        )


    return merged


# =========================================================
# ADD WALK-FORWARD MAJORITY BASELINE
# =========================================================

def add_majority_baseline(
    data,
    walk_forward_results,
):

    fold_mapping = {}


    for fold in walk_forward_results["folds"]:

        test_year = int(
            fold["test_year"]
        )


        majority_name = (
            fold["majority_class"]
        )


        if majority_name not in CLASS_TO_ID:

            raise ValueError(
                f"Unknown majority class: {majority_name}"
            )


        majority_id = (
            CLASS_TO_ID[
                majority_name
            ]
        )


        fold_mapping[
            test_year
        ] = majority_id


    data = data.copy()


    data["MAJORITY_BASELINE_CLASS"] = (
        data["fold_test_year"]
        .map(
            fold_mapping
        )
    )


    if data[
        "MAJORITY_BASELINE_CLASS"
    ].isna().any():

        raise ValueError(
            "Could not map majority baseline "
            "for every walk-forward fold."
        )


    data["MAJORITY_BASELINE_CLASS"] = (
        data["MAJORITY_BASELINE_CLASS"]
        .astype(int)
    )


    return data


# =========================================================
# RELATIVE MOMENTUM BASELINE
#
# Previous 5-day excess performance:
#
# < -1%  => UNDERPERFORM
# -1..1% => NEUTRAL
# > +1%  => OUTPERFORM
#
# Uses historical information only.
# =========================================================

def add_relative_momentum_baseline(
    data,
):

    data = data.copy()


    if "RELATIVE_RETURN_5" not in data.columns:

        raise ValueError(
            "RELATIVE_RETURN_5 was not found after merge."
        )


    relative_return = pd.to_numeric(
        data["RELATIVE_RETURN_5"],
        errors="coerce",
    )


    # =====================================================
    # SAFETY CHECK
    #
    # Must be a single Series, not duplicate columns.
    # =====================================================

    if isinstance(
        relative_return,
        pd.DataFrame,
    ):

        raise ValueError(
            "RELATIVE_RETURN_5 still exists more than once."
        )


    if relative_return.isna().any():

        raise ValueError(
            "RELATIVE_RETURN_5 contains missing values."
        )


    relative_values = (
        relative_return
        .to_numpy(
            dtype=float
        )
    )


    # Default = NEUTRAL
    momentum_class = np.ones(
        len(data),
        dtype=np.int32,
    )


    # UNDERPERFORM
    momentum_class[
        relative_values
        <
        -RELATIVE_MOMENTUM_THRESHOLD
    ] = 0


    # OUTPERFORM
    momentum_class[
        relative_values
        >
        RELATIVE_MOMENTUM_THRESHOLD
    ] = 2


    data[
        "MOMENTUM_BASELINE_CLASS"
    ] = momentum_class


    return data


# =========================================================
# CLASSIFICATION METRICS
# =========================================================

def calculate_metrics(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=int,
    )


    predicted = np.asarray(
        predicted,
        dtype=int,
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


    return {

        "samples":
            int(
                len(actual)
            ),

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
# OVERALL BASELINE COMPARISON
# =========================================================

def overall_comparison(
    data,
):

    actual = (
        data["ACTUAL_CLASS"]
        .values
    )


    model_metrics = (
        calculate_metrics(
            actual,
            data[
                "PREDICTED_CLASS"
            ].values,
        )
    )


    majority_metrics = (
        calculate_metrics(
            actual,
            data[
                "MAJORITY_BASELINE_CLASS"
            ].values,
        )
    )


    momentum_metrics = (
        calculate_metrics(
            actual,
            data[
                "MOMENTUM_BASELINE_CLASS"
            ].values,
        )
    )


    print(
        "\n=============================================="
    )

    print(
        " OVERALL ROBUSTNESS BASELINES"
    )

    print(
        "=============================================="
    )


    print(
        "\nUniform Random Chance Accuracy:"
    )

    print(
        "33.33%"
    )


    print(
        "\nWalk-Forward Majority Baseline:"
    )

    print(
        "Accuracy:",
        f'{majority_metrics["accuracy_percent"]:.2f}%'
    )

    print(
        "Macro F1:",
        f'{majority_metrics["macro_f1_percent"]:.2f}%'
    )

    print(
        "Balanced Accuracy:",
        f'{majority_metrics["balanced_accuracy_percent"]:.2f}%'
    )


    print(
        "\nRelative Momentum Baseline:"
    )

    print(
        "Accuracy:",
        f'{momentum_metrics["accuracy_percent"]:.2f}%'
    )

    print(
        "Macro F1:",
        f'{momentum_metrics["macro_f1_percent"]:.2f}%'
    )

    print(
        "Balanced Accuracy:",
        f'{momentum_metrics["balanced_accuracy_percent"]:.2f}%'
    )


    print(
        "\nV9 Walk-Forward Model:"
    )

    print(
        "Accuracy:",
        f'{model_metrics["accuracy_percent"]:.2f}%'
    )

    print(
        "Macro F1:",
        f'{model_metrics["macro_f1_percent"]:.2f}%'
    )

    print(
        "Balanced Accuracy:",
        f'{model_metrics["balanced_accuracy_percent"]:.2f}%'
    )


    print(
        "\nV9 vs Majority:",
        f'{model_metrics["accuracy_percent"] - majority_metrics["accuracy_percent"]:+.2f} pp'
    )


    print(
        "V9 vs Momentum:",
        f'{model_metrics["accuracy_percent"] - momentum_metrics["accuracy_percent"]:+.2f} pp'
    )


    return {

        "uniform_random_accuracy_percent":
            33.33,

        "majority":
            majority_metrics,

        "relative_momentum":
            momentum_metrics,

        "v9_model":
            model_metrics,

        "v9_improvement_vs_majority_pp":
            round(
                model_metrics[
                    "accuracy_percent"
                ]
                -
                majority_metrics[
                    "accuracy_percent"
                ],
                2,
            ),

        "v9_improvement_vs_momentum_pp":
            round(
                model_metrics[
                    "accuracy_percent"
                ]
                -
                momentum_metrics[
                    "accuracy_percent"
                ],
                2,
            ),
    }


# =========================================================
# PER-STOCK PERFORMANCE
# =========================================================

def analyze_per_stock(
    data,
):

    rows = []


    for symbol in sorted(
        data["SYMBOL"]
        .unique()
    ):

        stock_data = data[
            data["SYMBOL"]
            ==
            symbol
        ]


        actual = (
            stock_data[
                "ACTUAL_CLASS"
            ]
            .values
        )


        model_metrics = (
            calculate_metrics(
                actual,
                stock_data[
                    "PREDICTED_CLASS"
                ].values,
            )
        )


        majority_metrics = (
            calculate_metrics(
                actual,
                stock_data[
                    "MAJORITY_BASELINE_CLASS"
                ].values,
            )
        )


        momentum_metrics = (
            calculate_metrics(
                actual,
                stock_data[
                    "MOMENTUM_BASELINE_CLASS"
                ].values,
            )
        )


        rows.append(
            {

                "SYMBOL":
                    symbol,

                "Samples":
                    int(
                        len(stock_data)
                    ),

                "Model_Accuracy":
                    model_metrics[
                        "accuracy_percent"
                    ],

                "Model_Macro_F1":
                    model_metrics[
                        "macro_f1_percent"
                    ],

                "Model_Balanced_Accuracy":
                    model_metrics[
                        "balanced_accuracy_percent"
                    ],

                "Majority_Accuracy":
                    majority_metrics[
                        "accuracy_percent"
                    ],

                "Momentum_Accuracy":
                    momentum_metrics[
                        "accuracy_percent"
                    ],

                "Improvement_vs_Majority":
                    round(
                        model_metrics[
                            "accuracy_percent"
                        ]
                        -
                        majority_metrics[
                            "accuracy_percent"
                        ],
                        2,
                    ),

                "Improvement_vs_Momentum":
                    round(
                        model_metrics[
                            "accuracy_percent"
                        ]
                        -
                        momentum_metrics[
                            "accuracy_percent"
                        ],
                        2,
                    ),
            }
        )


    result = pd.DataFrame(
        rows
    )


    result.sort_values(
        by=
            "Improvement_vs_Majority",

        ascending=False,

        inplace=True,
    )


    result.reset_index(
        drop=True,
        inplace=True,
    )


    print(
        "\n=============================================="
    )

    print(
        " PER-STOCK ROBUSTNESS"
    )

    print(
        "=============================================="
    )


    print(
        "\n"
        "SYMBOL              MODEL   MAJORITY   MOMENTUM   "
        "VS_MAJ    MACRO_F1"
    )


    for _, row in result.iterrows():

        print(
            f'{row["SYMBOL"]:<18}'
            f'{row["Model_Accuracy"]:>7.2f}%'
            f'{row["Majority_Accuracy"]:>10.2f}%'
            f'{row["Momentum_Accuracy"]:>11.2f}%'
            f'{row["Improvement_vs_Majority"]:>9.2f}'
            f'{row["Model_Macro_F1"]:>11.2f}%'
        )


    stocks_beating_majority = int(
        (
            result[
                "Improvement_vs_Majority"
            ]
            >
            0
        )
        .sum()
    )


    stocks_beating_majority_3pp = int(
        (
            result[
                "Improvement_vs_Majority"
            ]
            >=
            3.0
        )
        .sum()
    )


    stocks_beating_momentum = int(
        (
            result[
                "Improvement_vs_Momentum"
            ]
            >
            0
        )
        .sum()
    )


    print(
        "\nStocks beating Majority:",
        f"{stocks_beating_majority}/{len(result)}"
    )


    print(
        "Stocks beating Majority by >=3 pp:",
        f"{stocks_beating_majority_3pp}/{len(result)}"
    )


    print(
        "Stocks beating Momentum:",
        f"{stocks_beating_momentum}/{len(result)}"
    )


    return (
        result,

        {
            "stocks_beating_majority":
                stocks_beating_majority,

            "stocks_beating_majority_by_3pp":
                stocks_beating_majority_3pp,

            "stocks_beating_momentum":
                stocks_beating_momentum,

            "total_stocks":
                int(
                    len(result)
                ),
        }
    )


# =========================================================
# BUILD NON-OVERLAPPING SUBSET
#
# Future target horizon = 5 days.
#
# Test all offsets:
#
# offset 0 -> rows 0, 5, 10...
# offset 1 -> rows 1, 6, 11...
# offset 2 -> rows 2, 7, 12...
# offset 3 -> rows 3, 8, 13...
# offset 4 -> rows 4, 9, 14...
# =========================================================

def build_non_overlapping_subset(
    data,
    offset,
):

    pieces = []


    grouped = data.groupby(
        [
            "SYMBOL",
            "fold_test_year",
        ]
    )


    for (
        symbol,
        test_year,
    ), group in grouped:

        group = (
            group
            .sort_values(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )


        selected = (
            group.iloc[
                offset::TARGET_HORIZON
            ]
        )


        if not selected.empty:

            pieces.append(
                selected
            )


    if not pieces:

        raise ValueError(
            f"No non-overlapping samples "
            f"created for offset {offset}."
        )


    return pd.concat(
        pieces,
        ignore_index=True,
    )


# =========================================================
# NON-OVERLAPPING ROBUSTNESS
# =========================================================

def analyze_non_overlapping(
    data,
):

    results = []


    print(
        "\n=============================================="
    )

    print(
        " NON-OVERLAPPING 5-DAY ROBUSTNESS"
    )

    print(
        "=============================================="
    )


    print(
        "\nOffset  Samples   Model   Majority   Momentum   "
        "MacroF1   Balanced"
    )


    for offset in range(
        TARGET_HORIZON
    ):

        subset = (
            build_non_overlapping_subset(
                data,
                offset,
            )
        )


        actual = (
            subset[
                "ACTUAL_CLASS"
            ]
            .values
        )


        model_metrics = (
            calculate_metrics(
                actual,
                subset[
                    "PREDICTED_CLASS"
                ].values,
            )
        )


        majority_metrics = (
            calculate_metrics(
                actual,
                subset[
                    "MAJORITY_BASELINE_CLASS"
                ].values,
            )
        )


        momentum_metrics = (
            calculate_metrics(
                actual,
                subset[
                    "MOMENTUM_BASELINE_CLASS"
                ].values,
            )
        )


        result = {

            "offset":
                int(
                    offset
                ),

            "samples":
                int(
                    len(subset)
                ),

            "model_accuracy_percent":
                model_metrics[
                    "accuracy_percent"
                ],

            "majority_accuracy_percent":
                majority_metrics[
                    "accuracy_percent"
                ],

            "momentum_accuracy_percent":
                momentum_metrics[
                    "accuracy_percent"
                ],

            "model_macro_f1_percent":
                model_metrics[
                    "macro_f1_percent"
                ],

            "model_balanced_accuracy_percent":
                model_metrics[
                    "balanced_accuracy_percent"
                ],

            "improvement_vs_majority_pp":
                round(
                    model_metrics[
                        "accuracy_percent"
                    ]
                    -
                    majority_metrics[
                        "accuracy_percent"
                    ],
                    2,
                ),

            "improvement_vs_momentum_pp":
                round(
                    model_metrics[
                        "accuracy_percent"
                    ]
                    -
                    momentum_metrics[
                        "accuracy_percent"
                    ],
                    2,
                ),
        }


        results.append(
            result
        )


        print(
            f'{offset:>6}'
            f'{len(subset):>9}'
            f'{model_metrics["accuracy_percent"]:>8.2f}%'
            f'{majority_metrics["accuracy_percent"]:>11.2f}%'
            f'{momentum_metrics["accuracy_percent"]:>11.2f}%'
            f'{model_metrics["macro_f1_percent"]:>10.2f}%'
            f'{model_metrics["balanced_accuracy_percent"]:>10.2f}%'
        )


    model_accuracies = np.array(
        [
            row[
                "model_accuracy_percent"
            ]
            for row in results
        ],
        dtype=float,
    )


    majority_accuracies = np.array(
        [
            row[
                "majority_accuracy_percent"
            ]
            for row in results
        ],
        dtype=float,
    )


    momentum_accuracies = np.array(
        [
            row[
                "momentum_accuracy_percent"
            ]
            for row in results
        ],
        dtype=float,
    )


    macro_f1_values = np.array(
        [
            row[
                "model_macro_f1_percent"
            ]
            for row in results
        ],
        dtype=float,
    )


    balanced_values = np.array(
        [
            row[
                "model_balanced_accuracy_percent"
            ]
            for row in results
        ],
        dtype=float,
    )


    average_model = float(
        model_accuracies.mean()
    )


    average_majority = float(
        majority_accuracies.mean()
    )


    average_momentum = float(
        momentum_accuracies.mean()
    )


    print(
        "\nAverage Across 5 Offsets:"
    )


    print(
        "Model Accuracy:",
        f"{average_model:.2f}%"
    )


    print(
        "Majority Accuracy:",
        f"{average_majority:.2f}%"
    )


    print(
        "Momentum Accuracy:",
        f"{average_momentum:.2f}%"
    )


    print(
        "Improvement vs Majority:",
        f"{average_model - average_majority:+.2f} pp"
    )


    print(
        "Improvement vs Momentum:",
        f"{average_model - average_momentum:+.2f} pp"
    )


    print(
        "Average Macro F1:",
        f"{macro_f1_values.mean():.2f}%"
    )


    print(
        "Average Balanced Accuracy:",
        f"{balanced_values.mean():.2f}%"
    )


    summary = {

        "average_model_accuracy_percent":
            round(
                average_model,
                2,
            ),

        "average_majority_accuracy_percent":
            round(
                average_majority,
                2,
            ),

        "average_momentum_accuracy_percent":
            round(
                average_momentum,
                2,
            ),

        "average_improvement_vs_majority_pp":
            round(
                average_model
                -
                average_majority,
                2,
            ),

        "average_improvement_vs_momentum_pp":
            round(
                average_model
                -
                average_momentum,
                2,
            ),

        "average_macro_f1_percent":
            round(
                float(
                    macro_f1_values.mean()
                ),
                2,
            ),

        "average_balanced_accuracy_percent":
            round(
                float(
                    balanced_values.mean()
                ),
                2,
            ),

        "minimum_model_accuracy_percent":
            round(
                float(
                    model_accuracies.min()
                ),
                2,
            ),

        "maximum_model_accuracy_percent":
            round(
                float(
                    model_accuracies.max()
                ),
                2,
            ),
    }


    return (
        results,
        summary,
    )


# =========================================================
# CONFIDENCE ANALYSIS
# =========================================================

def analyze_confidence(
    data,
):

    actual = (
        data[
            "ACTUAL_CLASS"
        ]
        .to_numpy(
            dtype=int
        )
    )


    predicted = (
        data[
            "PREDICTED_CLASS"
        ]
        .to_numpy(
            dtype=int
        )
    )


    confidence = (
        data[
            "CONFIDENCE"
        ]
        .to_numpy(
            dtype=float
        )
    )


    thresholds = [
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    ]


    results = []


    print(
        "\n=============================================="
    )

    print(
        " CONFIDENCE ROBUSTNESS"
    )

    print(
        "=============================================="
    )


    print(
        "\nThreshold   Coverage   Accuracy   Samples"
    )


    for threshold in thresholds:

        mask = (
            confidence
            >=
            threshold
        )


        sample_count = int(
            mask.sum()
        )


        coverage = (
            sample_count
            /
            len(data)
            *
            100
        )


        if sample_count > 0:

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

        else:

            accuracy = 0.0


        results.append(
            {

                "threshold":
                    float(
                        threshold
                    ),

                "samples":
                    sample_count,

                "coverage_percent":
                    round(
                        float(
                            coverage
                        ),
                        2,
                    ),

                "accuracy_percent":
                    round(
                        float(
                            accuracy
                        ),
                        2,
                    ),
            }
        )


        print(
            f'{threshold:>9.0%}'
            f'{coverage:>11.2f}%'
            f'{accuracy:>11.2f}%'
            f'{sample_count:>10}'
        )


    return results


# =========================================================
# FEATURE IMPORTANCE
#
# IMPORTANT:
#
# We use only old historical data:
#
# Training -> before 2021
# Validation -> 2021
#
# Walk-forward test years 2022-2026
# are NOT used to calculate importance.
# =========================================================

def analyze_feature_importance(
    dataset,
    walk_forward_results,
):

    first_fold = (
        walk_forward_results[
            "folds"
        ][
            0
        ]
    )


    train_end = pd.Timestamp(
        first_fold[
            "selection_train_last_date"
        ]
    )


    validation_start = pd.Timestamp(
        first_fold[
            "validation_start"
        ]
    )


    validation_end = pd.Timestamp(
        first_fold[
            "validation_last_date"
        ]
    )


    training = dataset[
        dataset[
            "Date"
        ]
        <=
        train_end
    ].copy()


    validation = dataset[
        (
            dataset[
                "Date"
            ]
            >=
            validation_start
        )
        &
        (
            dataset[
                "Date"
            ]
            <=
            validation_end
        )
    ].copy()


    if training.empty:

        raise ValueError(
            "Feature importance training dataset is empty."
        )


    if validation.empty:

        raise ValueError(
            "Feature importance validation dataset is empty."
        )


    print(
        "\n=============================================="
    )

    print(
        " V9 FEATURE IMPORTANCE"
    )

    print(
        "=============================================="
    )


    print(
        "\nDiagnostic Training:"
    )


    print(
        training[
            "Date"
        ]
        .min()
        .date(),
        "→",
        training[
            "Date"
        ]
        .max()
        .date(),
    )


    print(
        "Training Rows:",
        len(
            training
        )
    )


    print(
        "\nDiagnostic Validation:"
    )


    print(
        validation[
            "Date"
        ]
        .min()
        .date(),
        "→",
        validation[
            "Date"
        ]
        .max()
        .date(),
    )


    print(
        "Validation Rows:",
        len(
            validation
        )
    )


    X_train = (
        training[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )


    y_train = (
        training[
            "TARGET_CLASS"
        ]
        .to_numpy(
            dtype=int
        )
    )


    X_validation = (
        validation[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )


    y_validation = (
        validation[
            "TARGET_CLASS"
        ]
        .to_numpy(
            dtype=int
        )
    )


    # =====================================================
    # DIAGNOSTIC RANDOM FOREST
    # =====================================================

    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=12,

        min_samples_leaf=10,

        max_features="sqrt",

        class_weight="balanced",

        n_jobs=-1,

        random_state=RANDOM_STATE,
    )


    print(
        "\nTraining diagnostic Random Forest..."
    )


    model.fit(
        X_train,
        y_train,
    )


    validation_prediction = (
        model.predict(
            X_validation
        )
    )


    validation_accuracy = (
        accuracy_score(
            y_validation,
            validation_prediction,
        )
        *
        100
    )


    validation_macro_f1 = (
        f1_score(
            y_validation,
            validation_prediction,
            average="macro",
            zero_division=0,
        )
        *
        100
    )


    validation_balanced = (
        balanced_accuracy_score(
            y_validation,
            validation_prediction,
        )
        *
        100
    )


    print(
        "Diagnostic Validation Accuracy:",
        f"{validation_accuracy:.2f}%"
    )


    print(
        "Diagnostic Validation Macro F1:",
        f"{validation_macro_f1:.2f}%"
    )


    print(
        "Diagnostic Balanced Accuracy:",
        f"{validation_balanced:.2f}%"
    )


    # =====================================================
    # PERMUTATION IMPORTANCE SAMPLE
    # =====================================================

    if len(
        validation
    ) > 3000:

        sampled_validation = (
            validation.sample(
                n=3000,
                random_state=
                    RANDOM_STATE,
            )
        )

    else:

        sampled_validation = (
            validation.copy()
        )


    X_importance = (
        sampled_validation[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )


    y_importance = (
        sampled_validation[
            "TARGET_CLASS"
        ]
        .to_numpy(
            dtype=int
        )
    )


    print(
        "\nPermutation Importance Samples:",
        len(
            sampled_validation
        )
    )


    print(
        "Calculating permutation importance..."
    )


    importance = (
        permutation_importance(

            model,

            X_importance,

            y_importance,

            scoring=
                "f1_macro",

            n_repeats=5,

            random_state=
                RANDOM_STATE,

            n_jobs=-1,
        )
    )


    importance_df = pd.DataFrame(
        {

            "FEATURE":
                FEATURE_COLUMNS,

            "IMPORTANCE_MEAN":
                importance[
                    "importances_mean"
                ],

            "IMPORTANCE_STD":
                importance[
                    "importances_std"
                ],
        }
    )


    importance_df.sort_values(
        by=
            "IMPORTANCE_MEAN",

        ascending=False,

        inplace=True,
    )


    importance_df.reset_index(
        drop=True,
        inplace=True,
    )


    print(
        "\nTOP 15 V9 FEATURES"
    )


    print(
        "\nRank  Feature                         Importance"
    )


    for index, row in (
        importance_df
        .head(15)
        .iterrows()
    ):

        print(
            f'{index + 1:>4}  '
            f'{row["FEATURE"]:<30}'
            f'{row["IMPORTANCE_MEAN"]:>10.5f}'
        )


    return (
        importance_df,

        {
            "accuracy_percent":
                round(
                    float(
                        validation_accuracy
                    ),
                    2,
                ),

            "macro_f1_percent":
                round(
                    float(
                        validation_macro_f1
                    ),
                    2,
                ),

            "balanced_accuracy_percent":
                round(
                    float(
                        validation_balanced
                    ),
                    2,
                ),
        }
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

def print_final_summary(
    overall,
    stock_summary,
    non_overlap_summary,
):

    print(
        "\n\n=============================================="
    )

    print(
        " V9 ROBUSTNESS FINAL SUMMARY"
    )

    print(
        "=============================================="
    )


    print(
        "\nFull Walk-Forward:"
    )


    print(
        "Model Accuracy:",
        f'{overall["v9_model"]["accuracy_percent"]:.2f}%'
    )


    print(
        "Majority:",
        f'{overall["majority"]["accuracy_percent"]:.2f}%'
    )


    print(
        "Momentum:",
        f'{overall["relative_momentum"]["accuracy_percent"]:.2f}%'
    )


    print(
        "Improvement vs Majority:",
        f'{overall["v9_improvement_vs_majority_pp"]:+.2f} pp'
    )


    print(
        "Improvement vs Momentum:",
        f'{overall["v9_improvement_vs_momentum_pp"]:+.2f} pp'
    )


    print(
        "\nStocks beating majority:",
        f'{stock_summary["stocks_beating_majority"]}/'
        f'{stock_summary["total_stocks"]}'
    )


    print(
        "Stocks beating majority by >=3pp:",
        f'{stock_summary["stocks_beating_majority_by_3pp"]}/'
        f'{stock_summary["total_stocks"]}'
    )


    print(
        "Stocks beating momentum:",
        f'{stock_summary["stocks_beating_momentum"]}/'
        f'{stock_summary["total_stocks"]}'
    )


    print(
        "\nNon-overlapping average:"
    )


    print(
        "Model:",
        f'{non_overlap_summary["average_model_accuracy_percent"]:.2f}%'
    )


    print(
        "Majority:",
        f'{non_overlap_summary["average_majority_accuracy_percent"]:.2f}%'
    )


    print(
        "Momentum:",
        f'{non_overlap_summary["average_momentum_accuracy_percent"]:.2f}%'
    )


    print(
        "Non-overlap improvement vs majority:",
        f'{non_overlap_summary["average_improvement_vs_majority_pp"]:+.2f} pp'
    )


    print(
        "Non-overlap improvement vs momentum:",
        f'{non_overlap_summary["average_improvement_vs_momentum_pp"]:+.2f} pp'
    )


    print(
        "Non-overlap Macro F1:",
        f'{non_overlap_summary["average_macro_f1_percent"]:.2f}%'
    )


    print(
        "Non-overlap Balanced Accuracy:",
        f'{non_overlap_summary["average_balanced_accuracy_percent"]:.2f}%'
    )


# =========================================================
# RUN ANALYSIS
# =========================================================

def run_analysis():

    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V9 ROBUSTNESS ANALYSIS"
    )

    print(
        "=============================================="
    )


    # =====================================================
    # LOAD FILES
    # =====================================================

    predictions = (
        load_predictions()
    )


    dataset = (
        load_dataset()
    )


    walk_forward_results = (
        load_walk_forward_results()
    )


    print(
        "\nPrediction Rows:",
        len(
            predictions
        )
    )


    print(
        "Dataset Rows:",
        len(
            dataset
        )
    )


    print(
        "Stocks:",
        predictions[
            "SYMBOL"
        ]
        .nunique()
    )


    print(
        "Test Years:",
        sorted(
            predictions[
                "fold_test_year"
            ]
            .unique()
            .tolist()
        )
    )


    # =====================================================
    # MERGE WALK-FORWARD PREDICTIONS WITH FEATURES
    # =====================================================

    data = merge_data(
        predictions,
        dataset,
    )


    print(
        "Merged Rows:",
        len(
            data
        )
    )


    # =====================================================
    # ADD BASELINES
    # =====================================================

    data = add_majority_baseline(
        data,
        walk_forward_results,
    )


    data = add_relative_momentum_baseline(
        data
    )


    # =====================================================
    # 1. OVERALL BASELINE COMPARISON
    # =====================================================

    overall = overall_comparison(
        data
    )


    # =====================================================
    # 2. PER-STOCK ANALYSIS
    # =====================================================

    (
        per_stock_df,
        stock_summary,

    ) = analyze_per_stock(
        data
    )


    # =====================================================
    # 3. NON-OVERLAPPING 5-DAY ANALYSIS
    # =====================================================

    (
        non_overlap_results,
        non_overlap_summary,

    ) = analyze_non_overlapping(
        data
    )


    # =====================================================
    # 4. CONFIDENCE ANALYSIS
    # =====================================================

    confidence_results = (
        analyze_confidence(
            data
        )
    )


    # =====================================================
    # 5. FEATURE IMPORTANCE
    # =====================================================

    (
        importance_df,
        diagnostic_metrics,

    ) = analyze_feature_importance(
        dataset,
        walk_forward_results,
    )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print_final_summary(
        overall,
        stock_summary,
        non_overlap_summary,
    )


    # =====================================================
    # SAVE PER-STOCK CSV
    # =====================================================

    per_stock_df.to_csv(
        PER_STOCK_PATH,
        index=False,
    )


    # =====================================================
    # SAVE FEATURE IMPORTANCE
    # =====================================================

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )


    # =====================================================
    # SAVE JSON
    # =====================================================

    output = {

        "experiment":
            "StockVision V9 Robustness Analysis",

        "overall":
            overall,

        "stock_summary":
            stock_summary,

        "per_stock":
            per_stock_df.to_dict(
                orient="records"
            ),

        "non_overlapping_offsets":
            non_overlap_results,

        "non_overlapping_summary":
            non_overlap_summary,

        "confidence_analysis":
            confidence_results,

        "feature_importance": {

            "diagnostic_validation_metrics":
                diagnostic_metrics,

            "top_features":
                importance_df
                .head(20)
                .to_dict(
                    orient="records"
                ),
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


    # =====================================================
    # FILES
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " ROBUSTNESS ANALYSIS SAVED"
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
        "\nPer Stock:"
    )


    print(
        PER_STOCK_PATH
    )


    print(
        "\nFeature Importance:"
    )


    print(
        FEATURE_IMPORTANCE_PATH
    )


    print(
        "\n=============================================="
    )

    print(
        " V9 ROBUSTNESS ANALYSIS COMPLETE"
    )

    print(
        "=============================================="
    )


# =========================================================
# COMMAND LINE
# =========================================================

if __name__ == "__main__":

    run_analysis()