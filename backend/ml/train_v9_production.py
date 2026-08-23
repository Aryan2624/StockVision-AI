import os
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)


# =========================================================
# V9 IMPORTS
# =========================================================

try:
    from .relative_dataset_v9 import (
        FEATURE_COLUMNS,
        CLASS_NAMES,
        TARGET_HORIZON,
        EXCESS_RETURN_THRESHOLD,
        MARKET_SYMBOL,
    )

except ImportError:
    from relative_dataset_v9 import (
        FEATURE_COLUMNS,
        CLASS_NAMES,
        TARGET_HORIZON,
        EXCESS_RETURN_THRESHOLD,
        MARKET_SYMBOL,
    )


# =========================================================
# CONFIG
# =========================================================

RANDOM_STATE = 42

NUM_CLASSES = 3


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


MODEL_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_production_model.pkl",
)


METADATA_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_production_metadata.json",
)


ROBUSTNESS_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_robustness_results.json",
)


# =========================================================
# LOAD DATASET
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


    required_columns = list(
        dict.fromkeys(
            required_columns
        )
    )


    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]


    if missing:

        raise ValueError(
            "Missing columns:\n"
            +
            ", ".join(
                missing
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
# SAMPLE WEIGHTS
#
# Keeps the 3 classes reasonably balanced during training.
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
# BUILD FINAL V9 MODEL
#
# Hist Gradient Boosting is used for the production model.
#
# Reasons:
# - worked well in recent walk-forward folds
# - handles nonlinear relationships
# - no feature scaler needed
# - relatively fast for live project retraining
# =========================================================

def build_model():

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
# LOAD HISTORICAL RESEARCH RESULTS
# =========================================================

def load_research_metrics():

    if not os.path.exists(
        ROBUSTNESS_PATH
    ):

        return None


    try:

        with open(
            ROBUSTNESS_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            robustness = json.load(
                file
            )


        overall = robustness.get(
            "overall",
            {}
        )


        model_results = overall.get(
            "v9_model",
            {}
        )


        majority_results = overall.get(
            "majority",
            {}
        )


        momentum_results = overall.get(
            "relative_momentum",
            {}
        )


        return {

            "walk_forward_accuracy_percent":
                model_results.get(
                    "accuracy_percent"
                ),

            "walk_forward_macro_f1_percent":
                model_results.get(
                    "macro_f1_percent"
                ),

            "walk_forward_balanced_accuracy_percent":
                model_results.get(
                    "balanced_accuracy_percent"
                ),

            "majority_baseline_accuracy_percent":
                majority_results.get(
                    "accuracy_percent"
                ),

            "momentum_baseline_accuracy_percent":
                momentum_results.get(
                    "accuracy_percent"
                ),

            "improvement_vs_majority_pp":
                overall.get(
                    "v9_improvement_vs_majority_pp"
                ),

            "improvement_vs_momentum_pp":
                overall.get(
                    "v9_improvement_vs_momentum_pp"
                ),
        }


    except Exception as error:

        print(
            "\nWarning:"
        )

        print(
            "Could not load robustness metrics:",
            error
        )


        return None


# =========================================================
# TRAIN FINAL PRODUCTION MODEL
# =========================================================

def train_production_model():

    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V9 PRODUCTION TRAINING"
    )

    print(
        "=============================================="
    )


    data = load_dataset()


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
        "Date Range:",
        data[
            "Date"
        ]
        .min()
        .date(),
        "→",
        data[
            "Date"
        ]
        .max()
        .date(),
    )


    # =====================================================
    # FEATURES / TARGET
    # =====================================================

    X = (
        data[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )


    y = (
        data[
            "TARGET_CLASS"
        ]
        .to_numpy(
            dtype=int
        )
    )


    # =====================================================
    # CLASS DISTRIBUTION
    # =====================================================

    counts = np.bincount(
        y,
        minlength=NUM_CLASSES,
    )


    print(
        "\nTraining Class Distribution:"
    )


    for class_id in range(
        NUM_CLASSES
    ):

        count = int(
            counts[
                class_id
            ]
        )


        percentage = (
            count
            /
            len(y)
            *
            100
        )


        print(
            f"{CLASS_NAMES[class_id]:<14}"
            f"{count:>7} "
            f"({percentage:.2f}%)"
        )


    # =====================================================
    # WEIGHTS
    # =====================================================

    (
        class_weights,
        sample_weights,

    ) = calculate_sample_weights(
        y
    )


    print(
        "\nClass Weights:"
    )


    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"{CLASS_NAMES[class_id]:<14}"
            f"{class_weights[class_id]:.4f}"
        )


    # =====================================================
    # BUILD MODEL
    # =====================================================

    model = build_model()


    print(
        "\nTraining final V9 model..."
    )


    print(
        "Model: HistGradientBoostingClassifier"
    )


    # =====================================================
    # FINAL TRAIN
    #
    # At this stage model selection/evaluation has already
    # been completed using purged walk-forward testing.
    #
    # Production model now uses all currently labelled
    # historical samples.
    # =====================================================

    model.fit(
        X,
        y,
        sample_weight=
            sample_weights,
    )


    print(
        "\nTraining complete."
    )


    # =====================================================
    # TRAINING-DATA SANITY CHECK
    #
    # IMPORTANT:
    # This is NOT reported as real model accuracy.
    #
    # Real performance comes from the previously completed
    # walk-forward experiment.
    # =====================================================

    training_prediction = (
        model.predict(
            X
        )
    )


    training_accuracy = (
        np.mean(
            training_prediction
            ==
            y
        )
        *
        100
    )


    print(
        "\nTraining-data sanity accuracy:",
        f"{training_accuracy:.2f}%"
    )


    print(
        "(Do NOT use this as project evaluation accuracy.)"
    )


    # =====================================================
    # SAVE MODEL
    # =====================================================

    joblib.dump(
        model,
        MODEL_PATH,
    )


    # =====================================================
    # RESEARCH METRICS
    # =====================================================

    research_metrics = (
        load_research_metrics()
    )


    # =====================================================
    # METADATA
    # =====================================================

    metadata = {

        "project":
            "StockVision AI",

        "model_version":
            "V9",

        "engine":
            "Relative Strength Intelligence",

        "production_model":
            "HistGradientBoostingClassifier",

        "benchmark":
            "NIFTY 50",

        "benchmark_symbol":
            MARKET_SYMBOL,

        "target":
            (
                "5-day stock excess return "
                "relative to NIFTY 50"
            ),

        "target_horizon_trading_days":
            TARGET_HORIZON,

        "threshold_percent":
            EXCESS_RETURN_THRESHOLD
            *
            100,

        "classes": {

            "0":
                CLASS_NAMES[
                    0
                ],

            "1":
                CLASS_NAMES[
                    1
                ],

            "2":
                CLASS_NAMES[
                    2
                ],
        },

        "training_rows":
            int(
                len(
                    data
                )
            ),

        "training_stocks":
            sorted(
                data[
                    "SYMBOL"
                ]
                .unique()
                .tolist()
            ),

        "training_stock_count":
            int(
                data[
                    "SYMBOL"
                ]
                .nunique()
            ),

        "training_start_date":
            str(
                data[
                    "Date"
                ]
                .min()
                .date()
            ),

        "training_end_date":
            str(
                data[
                    "Date"
                ]
                .max()
                .date()
            ),

        "feature_count":
            len(
                FEATURE_COLUMNS
            ),

        "feature_columns":
            FEATURE_COLUMNS,

        "class_weights": {

            str(
                class_id
            ):
                float(
                    class_weights[
                        class_id
                    ]
                )

            for class_id in range(
                NUM_CLASSES
            )
        },

        "training_sanity_accuracy_percent":
            round(
                float(
                    training_accuracy
                ),
                2,
            ),

        "research_evaluation":
            research_metrics,

        "evaluation_note":
            (
                "Production training accuracy is not "
                "used as the reported model performance. "
                "Reported performance must come from "
                "purged walk-forward V9 evaluation."
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
    # COMPLETE
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " V9 PRODUCTION MODEL SAVED"
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
        "\nMetadata:"
    )


    print(
        METADATA_PATH
    )


    if research_metrics:

        print(
            "\n=============================================="
        )

        print(
            " VERIFIED V9 WALK-FORWARD PERFORMANCE"
        )

        print(
            "=============================================="
        )


        print(
            "\nAccuracy:",
            research_metrics.get(
                "walk_forward_accuracy_percent"
            ),
            "%"
        )


        print(
            "Macro F1:",
            research_metrics.get(
                "walk_forward_macro_f1_percent"
            ),
            "%"
        )


        print(
            "Balanced Accuracy:",
            research_metrics.get(
                "walk_forward_balanced_accuracy_percent"
            ),
            "%"
        )


        print(
            "Improvement vs Majority:",
            research_metrics.get(
                "improvement_vs_majority_pp"
            ),
            "pp"
        )


        print(
            "Improvement vs Momentum:",
            research_metrics.get(
                "improvement_vs_momentum_pp"
            ),
            "pp"
        )


    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V9 PRODUCTION READY"
    )

    print(
        "=============================================="
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    train_production_model()