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


# =========================================================
# V9 IMPORT
# =========================================================

try:
    from .relative_dataset_v9 import FEATURE_COLUMNS

except ImportError:
    from relative_dataset_v9 import FEATURE_COLUMNS


# =========================================================
# CONFIG
# =========================================================

TARGET_HORIZON = 5

NUM_CLASSES = 3

RANDOM_STATE = 42

RELATIVE_MOMENTUM_THRESHOLD = 0.01


# Model must beat best validation baseline by at least
# this amount of Macro F1 before the gate trusts it.
GATE_MARGIN_PP = 0.50


MIN_GROUP_VALIDATION_ROWS = 200


CLASS_NAMES = {
    0: "UNDERPERFORM",
    1: "NEUTRAL",
    2: "OUTPERFORM",
}


# =========================================================
# WALK-FORWARD FOLDS
# =========================================================

FOLDS = [
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
    (2025, 2026),
]


# =========================================================
# GATING GROUPS
#
# We use broader groups instead of every tiny individual
# sector because many sectors contain only one stock.
#
# This gives more reliable validation samples.
# =========================================================

SYMBOL_GROUP = {

    # BANKING
    "HDFCBANK.NS": "BANKING",
    "ICICIBANK.NS": "BANKING",
    "AXISBANK.NS": "BANKING",
    "SBIN.NS": "BANKING",
    "KOTAKBANK.NS": "BANKING",

    # IT
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "WIPRO.NS": "IT",
    "HCLTECH.NS": "IT",

    # OTHER
    "RELIANCE.NS": "OTHER",
    "ITC.NS": "OTHER",
    "LT.NS": "OTHER",
    "BHARTIARTL.NS": "OTHER",
    "MARUTI.NS": "OTHER",
    "SUNPHARMA.NS": "OTHER",
}


GROUPS = [
    "BANKING",
    "IT",
    "OTHER",
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
    "stockvision_v9_1_gated_walk_forward_results.json",
)


PREDICTIONS_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_1_gated_predictions.csv",
)


PER_STOCK_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_1_gated_per_stock.csv",
)


# =========================================================
# LOAD DATA
# =========================================================

def load_dataset():

    if not os.path.exists(DATASET_PATH):

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
        "RELATIVE_RETURN_5",
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
            ", ".join(missing)
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


    data["RELATIVE_RETURN_5"] = pd.to_numeric(
        data["RELATIVE_RETURN_5"],
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


    # =====================================================
    # ADD GATING GROUP
    # =====================================================

    data["GATE_GROUP"] = (
        data["SYMBOL"]
        .map(
            SYMBOL_GROUP
        )
    )


    if data["GATE_GROUP"].isna().any():

        unknown_symbols = (
            data.loc[
                data["GATE_GROUP"].isna(),
                "SYMBOL",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Missing gate group for symbols: "
            +
            ", ".join(unknown_symbols)
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
# CLASS / SAMPLE WEIGHTS
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


    for class_id in range(NUM_CLASSES):

        if counts[class_id] == 0:

            class_weights[class_id] = 1.0

        else:

            class_weights[class_id] = (
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
# MODELS
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
# FIT MODEL
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
# PREDICT MODEL
# =========================================================

def predict_model(
    model,
    scaler,
    X,
):

    if scaler is not None:

        X = scaler.transform(
            X
        )


    probabilities = model.predict_proba(
        X
    )


    predicted = np.argmax(
        probabilities,
        axis=1,
    )


    return (
        predicted.astype(int),
        probabilities,
    )


# =========================================================
# METRICS
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


    for class_id in range(NUM_CLASSES):

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
# MAJORITY CLASS
# =========================================================

def get_majority_class(
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


    return int(
        np.argmax(
            counts
        )
    )


# =========================================================
# MOMENTUM BASELINE
# =========================================================

def momentum_predictions(
    relative_return_5,
):

    values = np.asarray(
        relative_return_5,
        dtype=float,
    )


    predictions = np.ones(
        len(values),
        dtype=int,
    )


    predictions[
        values
        <
        -RELATIVE_MOMENTUM_THRESHOLD
    ] = 0


    predictions[
        values
        >
        RELATIVE_MOMENTUM_THRESHOLD
    ] = 2


    return predictions


# =========================================================
# CREATE PURGED WALK-FORWARD FOLD
# =========================================================

def create_fold(
    data,
    validation_year,
    test_year,
):

    unique_dates = np.array(
        sorted(
            data["Date"]
            .unique()
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
            f"No validation data for {validation_year}"
        )


    if not test_dates:

        raise ValueError(
            f"No test data for {test_year}"
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
    # PURGE 5 TRADING DAYS BEFORE VALIDATION
    # =====================================================

    selection_train_last_index = (
        validation_start_index
        -
        TARGET_HORIZON
        -
        1
    )


    # =====================================================
    # PURGE 5 TRADING DAYS BEFORE TEST
    # =====================================================

    validation_last_index = (
        test_start_index
        -
        TARGET_HORIZON
        -
        1
    )


    if selection_train_last_index < 0:

        raise ValueError(
            "Invalid selection training boundary."
        )


    if (
        validation_last_index
        <=
        validation_start_index
    ):

        raise ValueError(
            "Invalid validation boundary."
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


    selection_train = data[
        data["Date"]
        <=
        selection_train_last_date
    ].copy()


    validation = data[
        (
            data["Date"]
            >=
            validation_start
        )
        &
        (
            data["Date"]
            <=
            validation_last_date
        )
    ].copy()


    final_train = data[
        data["Date"]
        <=
        validation_last_date
    ].copy()


    test = data[
        data["Date"]
        .dt.year
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
# SELECT BEST GLOBAL ML MODEL
#
# Uses validation data only.
# =========================================================

def select_global_model(
    selection_train,
    validation,
):

    X_train = (
        selection_train[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )


    y_train = (
        selection_train[
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


    model_names = [
        "Logistic Regression",
        "Random Forest",
        "Hist Gradient Boosting",
    ]


    results = {}


    predictions = {}


    print(
        "\nGlobal Model Selection:"
    )


    for model_name in model_names:

        print(
            f"\nTraining {model_name}..."
        )


        (
            model,
            scaler,

        ) = fit_model(
            model_name,
            X_train,
            y_train,
        )


        (
            predicted,
            probabilities,

        ) = predict_model(
            model,
            scaler,
            X_validation,
        )


        metrics = calculate_metrics(
            y_validation,
            predicted,
        )


        results[
            model_name
        ] = metrics


        predictions[
            model_name
        ] = predicted


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


    best_model_name = max(

        results,

        key=lambda name:
            results[
                name
            ][
                "macro_f1_percent"
            ],
    )


    print(
        "\nSelected Global Model:",
        best_model_name
    )


    print(
        "Validation Macro F1:",
        f'{results[best_model_name]["macro_f1_percent"]:.2f}%'
    )


    return (
        best_model_name,
        results,
        predictions[
            best_model_name
        ],
    )


# =========================================================
# VALIDATION GATE
#
# For each group:
#
# 1. Evaluate selected ML model.
# 2. Evaluate relative-momentum baseline.
# 3. Evaluate historical majority baseline.
#
# Only validation data makes this decision.
#
# Test year is never used here.
# =========================================================

def create_validation_gates(
    selection_train,
    validation,
    validation_model_predictions,
):

    validation = validation.copy()


    validation[
        "MODEL_PREDICTION"
    ] = validation_model_predictions


    majority_class = get_majority_class(
        selection_train[
            "TARGET_CLASS"
        ].values
    )


    validation[
        "MAJORITY_PREDICTION"
    ] = majority_class


    validation[
        "MOMENTUM_PREDICTION"
    ] = momentum_predictions(
        validation[
            "RELATIVE_RETURN_5"
        ].values
    )


    gate_decisions = {}


    print(
        "\n=============================================="
    )

    print(
        " VALIDATION GATE DECISIONS"
    )

    print(
        "=============================================="
    )


    for group_name in GROUPS:

        group_data = validation[
            validation[
                "GATE_GROUP"
            ]
            ==
            group_name
        ].copy()


        print(
            f"\n{group_name}"
        )


        print(
            "Validation Rows:",
            len(
                group_data
            )
        )


        if len(
            group_data
        ) < MIN_GROUP_VALIDATION_ROWS:

            gate_decisions[
                group_name
            ] = {

                "action":
                    "MOMENTUM",

                "reason":
                    "Insufficient validation rows",
            }


            print(
                "Action: MOMENTUM"
            )


            print(
                "Reason: insufficient validation rows"
            )


            continue


        actual = (
            group_data[
                "TARGET_CLASS"
            ]
            .to_numpy(
                dtype=int
            )
        )


        model_metrics = calculate_metrics(
            actual,
            group_data[
                "MODEL_PREDICTION"
            ].values,
        )


        momentum_metrics = calculate_metrics(
            actual,
            group_data[
                "MOMENTUM_PREDICTION"
            ].values,
        )


        majority_metrics = calculate_metrics(
            actual,
            group_data[
                "MAJORITY_PREDICTION"
            ].values,
        )


        model_score = (
            model_metrics[
                "macro_f1_percent"
            ]
        )


        momentum_score = (
            momentum_metrics[
                "macro_f1_percent"
            ]
        )


        majority_score = (
            majority_metrics[
                "macro_f1_percent"
            ]
        )


        print(
            "ML Macro F1:",
            f"{model_score:.2f}%"
        )


        print(
            "Momentum Macro F1:",
            f"{momentum_score:.2f}%"
        )


        print(
            "Majority Macro F1:",
            f"{majority_score:.2f}%"
        )


        # =================================================
        # BEST BASELINE
        # =================================================

        if (
            momentum_score
            >=
            majority_score
        ):

            best_baseline_name = (
                "MOMENTUM"
            )


            best_baseline_score = (
                momentum_score
            )

        else:

            best_baseline_name = (
                "MAJORITY"
            )


            best_baseline_score = (
                majority_score
            )


        # =================================================
        # GATE DECISION
        # =================================================

        if (
            model_score
            >=
            best_baseline_score
            +
            GATE_MARGIN_PP
        ):

            action = "MODEL"

            reason = (
                "ML beats best validation baseline "
                f"by at least {GATE_MARGIN_PP:.2f} pp Macro F1"
            )

        else:

            action = (
                best_baseline_name
            )

            reason = (
                "ML did not beat best validation "
                "baseline by required margin"
            )


        gate_decisions[
            group_name
        ] = {

            "action":
                action,

            "validation_rows":
                int(
                    len(group_data)
                ),

            "model_metrics":
                model_metrics,

            "momentum_metrics":
                momentum_metrics,

            "majority_metrics":
                majority_metrics,

            "best_baseline":
                best_baseline_name,

            "model_macro_f1":
                round(
                    float(
                        model_score
                    ),
                    2,
                ),

            "best_baseline_macro_f1":
                round(
                    float(
                        best_baseline_score
                    ),
                    2,
                ),

            "reason":
                reason,
        }


        print(
            "Gate Action:",
            action
        )


    return gate_decisions


# =========================================================
# APPLY GATE TO TEST YEAR
# =========================================================

def apply_gate(
    test,
    model_predictions,
    momentum_preds,
    majority_class,
    gate_decisions,
):

    gated_predictions = np.zeros(
        len(test),
        dtype=int,
    )


    actions = []


    for index in range(
        len(test)
    ):

        group_name = (
            test.iloc[index][
                "GATE_GROUP"
            ]
        )


        decision = (
            gate_decisions[
                group_name
            ][
                "action"
            ]
        )


        actions.append(
            decision
        )


        if decision == "MODEL":

            gated_predictions[
                index
            ] = (
                model_predictions[
                    index
                ]
            )


        elif decision == "MOMENTUM":

            gated_predictions[
                index
            ] = (
                momentum_preds[
                    index
                ]
            )


        elif decision == "MAJORITY":

            gated_predictions[
                index
            ] = majority_class


        else:

            raise ValueError(
                f"Unknown gate action: {decision}"
            )


    return (
        gated_predictions,
        actions,
    )


# =========================================================
# PRINT METRICS
# =========================================================

def print_metrics(
    title,
    metrics,
):

    print(
        f"\n{title}"
    )


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


# =========================================================
# PER-STOCK AGGREGATED ANALYSIS
# =========================================================

def analyze_per_stock(
    predictions,
):

    rows = []


    for symbol in sorted(
        predictions[
            "SYMBOL"
        ]
        .unique()
    ):

        stock = predictions[
            predictions[
                "SYMBOL"
            ]
            ==
            symbol
        ]


        actual = (
            stock[
                "ACTUAL_CLASS"
            ]
            .to_numpy(
                dtype=int
            )
        )


        gated_metrics = calculate_metrics(
            actual,
            stock[
                "GATED_PREDICTION"
            ].values,
        )


        base_metrics = calculate_metrics(
            actual,
            stock[
                "BASE_MODEL_PREDICTION"
            ].values,
        )


        momentum_metrics = calculate_metrics(
            actual,
            stock[
                "MOMENTUM_PREDICTION"
            ].values,
        )


        majority_metrics = calculate_metrics(
            actual,
            stock[
                "MAJORITY_PREDICTION"
            ].values,
        )


        rows.append(
            {

                "SYMBOL":
                    symbol,

                "GROUP":
                    stock[
                        "GATE_GROUP"
                    ]
                    .iloc[0],

                "Samples":
                    int(
                        len(stock)
                    ),

                "Gated_Accuracy":
                    gated_metrics[
                        "accuracy_percent"
                    ],

                "Gated_Macro_F1":
                    gated_metrics[
                        "macro_f1_percent"
                    ],

                "Base_Model_Accuracy":
                    base_metrics[
                        "accuracy_percent"
                    ],

                "Momentum_Accuracy":
                    momentum_metrics[
                        "accuracy_percent"
                    ],

                "Majority_Accuracy":
                    majority_metrics[
                        "accuracy_percent"
                    ],

                "Gated_vs_Base":
                    round(
                        gated_metrics[
                            "accuracy_percent"
                        ]
                        -
                        base_metrics[
                            "accuracy_percent"
                        ],
                        2,
                    ),

                "Gated_vs_Momentum":
                    round(
                        gated_metrics[
                            "accuracy_percent"
                        ]
                        -
                        momentum_metrics[
                            "accuracy_percent"
                        ],
                        2,
                    ),

                "Gated_vs_Majority":
                    round(
                        gated_metrics[
                            "accuracy_percent"
                        ]
                        -
                        majority_metrics[
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
            "Gated_Accuracy",

        ascending=False,

        inplace=True,
    )


    result.reset_index(
        drop=True,
        inplace=True,
    )


    return result


# =========================================================
# RUN V9.1
# =========================================================

def run_v9_1():

    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V9.1 VALIDATION-GATED"
    )

    print(
        " PURGED WALK-FORWARD"
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
        "Gate Groups:",
        GROUPS
    )


    print(
        "Gate Margin:",
        f"{GATE_MARGIN_PP:.2f} pp Macro F1"
    )


    all_fold_results = []

    prediction_rows = []


    pooled_actual = []

    pooled_base = []

    pooled_gated = []

    pooled_momentum = []

    pooled_majority = []


    # =====================================================
    # EACH WALK-FORWARD FOLD
    # =====================================================

    for (
        validation_year,
        test_year,

    ) in FOLDS:

        print(
            "\n\n=============================================="
        )

        print(
            f" FOLD: VALIDATION {validation_year}"
            f" → TEST {test_year}"
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
        ].copy()


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
        # STEP A — GLOBAL MODEL SELECTION
        # =================================================

        (
            selected_model_name,
            validation_model_results,
            validation_model_predictions,

        ) = select_global_model(
            selection_train,
            validation,
        )


        # =================================================
        # STEP B — CREATE VALIDATION GATES
        # =================================================

        gate_decisions = (
            create_validation_gates(
                selection_train=
                    selection_train,

                validation=
                    validation,

                validation_model_predictions=
                    validation_model_predictions,
            )
        )


        # =================================================
        # STEP C — RETRAIN SELECTED MODEL
        # USING ALL AVAILABLE PAST DATA
        # =================================================

        X_final_train = (
            final_train[
                FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )


        y_final_train = (
            final_train[
                "TARGET_CLASS"
            ]
            .to_numpy(
                dtype=int
            )
        )


        X_test = (
            test[
                FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )


        y_test = (
            test[
                "TARGET_CLASS"
            ]
            .to_numpy(
                dtype=int
            )
        )


        print(
            "\nRetraining Selected Model:"
        )


        print(
            selected_model_name
        )


        (
            final_model,
            final_scaler,

        ) = fit_model(
            selected_model_name,
            X_final_train,
            y_final_train,
        )


        (
            base_test_predictions,
            base_test_probabilities,

        ) = predict_model(
            final_model,
            final_scaler,
            X_test,
        )


        # =================================================
        # TEST BASELINES
        # =================================================

        final_majority_class = (
            get_majority_class(
                y_final_train
            )
        )


        majority_test_predictions = np.full(
            len(test),
            final_majority_class,
            dtype=int,
        )


        momentum_test_predictions = (
            momentum_predictions(
                test[
                    "RELATIVE_RETURN_5"
                ]
                .values
            )
        )


        # =================================================
        # APPLY VALIDATION GATE
        # =================================================

        (
            gated_test_predictions,
            gate_actions,

        ) = apply_gate(
            test=
                test,

            model_predictions=
                base_test_predictions,

            momentum_preds=
                momentum_test_predictions,

            majority_class=
                final_majority_class,

            gate_decisions=
                gate_decisions,
        )


        # =================================================
        # TEST METRICS
        # =================================================

        base_metrics = calculate_metrics(
            y_test,
            base_test_predictions,
        )


        gated_metrics = calculate_metrics(
            y_test,
            gated_test_predictions,
        )


        momentum_metrics = calculate_metrics(
            y_test,
            momentum_test_predictions,
        )


        majority_metrics = calculate_metrics(
            y_test,
            majority_test_predictions,
        )


        print(
            "\n=============================================="
        )

        print(
            f" TEST {test_year} RESULTS"
        )

        print(
            "=============================================="
        )


        print_metrics(
            "Majority Baseline",
            majority_metrics,
        )


        print_metrics(
            "Momentum Baseline",
            momentum_metrics,
        )


        print_metrics(
            "Ungated V9 Model",
            base_metrics,
        )


        print_metrics(
            "V9.1 Gated Model",
            gated_metrics,
        )


        print(
            "\nV9.1 vs Majority:",
            f'{gated_metrics["accuracy_percent"] - majority_metrics["accuracy_percent"]:+.2f} pp'
        )


        print(
            "V9.1 vs Momentum:",
            f'{gated_metrics["accuracy_percent"] - momentum_metrics["accuracy_percent"]:+.2f} pp'
        )


        print(
            "V9.1 vs Ungated V9:",
            f'{gated_metrics["accuracy_percent"] - base_metrics["accuracy_percent"]:+.2f} pp'
        )


        # =================================================
        # PRINT GATE USAGE
        # =================================================

        action_counts = (
            pd.Series(
                gate_actions
            )
            .value_counts()
            .to_dict()
        )


        print(
            "\nGate Usage:"
        )


        for action_name in [
            "MODEL",
            "MOMENTUM",
            "MAJORITY",
        ]:

            count = int(
                action_counts.get(
                    action_name,
                    0,
                )
            )


            percentage = (
                count
                /
                len(test)
                *
                100
            )


            print(
                f"{action_name}: "
                f"{count} "
                f"({percentage:.2f}%)"
            )


        # =================================================
        # STORE POOLED ARRAYS
        # =================================================

        pooled_actual.extend(
            y_test.tolist()
        )


        pooled_base.extend(
            base_test_predictions.tolist()
        )


        pooled_gated.extend(
            gated_test_predictions.tolist()
        )


        pooled_momentum.extend(
            momentum_test_predictions.tolist()
        )


        pooled_majority.extend(
            majority_test_predictions.tolist()
        )


        # =================================================
        # STORE EACH TEST PREDICTION
        # =================================================

        for index in range(
            len(test)
        ):

            row = test.iloc[
                index
            ]


            prediction_rows.append(
                {

                    "test_year":
                        int(
                            test_year
                        ),

                    "Date":
                        str(
                            pd.Timestamp(
                                row[
                                    "Date"
                                ]
                            ).date()
                        ),

                    "SYMBOL":
                        row[
                            "SYMBOL"
                        ],

                    "GATE_GROUP":
                        row[
                            "GATE_GROUP"
                        ],

                    "SELECTED_MODEL":
                        selected_model_name,

                    "GATE_ACTION":
                        gate_actions[
                            index
                        ],

                    "ACTUAL_CLASS":
                        int(
                            y_test[
                                index
                            ]
                        ),

                    "ACTUAL_LABEL":
                        CLASS_NAMES[
                            int(
                                y_test[
                                    index
                                ]
                            )
                        ],

                    "BASE_MODEL_PREDICTION":
                        int(
                            base_test_predictions[
                                index
                            ]
                        ),

                    "GATED_PREDICTION":
                        int(
                            gated_test_predictions[
                                index
                            ]
                        ),

                    "GATED_LABEL":
                        CLASS_NAMES[
                            int(
                                gated_test_predictions[
                                    index
                                ]
                            )
                        ],

                    "MOMENTUM_PREDICTION":
                        int(
                            momentum_test_predictions[
                                index
                            ]
                        ),

                    "MAJORITY_PREDICTION":
                        int(
                            majority_test_predictions[
                                index
                            ]
                        ),

                    "PROB_UNDERPERFORM":
                        float(
                            base_test_probabilities[
                                index,
                                0
                            ]
                        ),

                    "PROB_NEUTRAL":
                        float(
                            base_test_probabilities[
                                index,
                                1
                            ]
                        ),

                    "PROB_OUTPERFORM":
                        float(
                            base_test_probabilities[
                                index,
                                2
                            ]
                        ),
                }
            )


        # =================================================
        # SAVE FOLD RESULT
        # =================================================

        all_fold_results.append(
            {

                "validation_year":
                    int(
                        validation_year
                    ),

                "test_year":
                    int(
                        test_year
                    ),

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

                "global_validation_results":
                    validation_model_results,

                "gate_decisions":
                    gate_decisions,

                "test_metrics": {

                    "majority":
                        majority_metrics,

                    "momentum":
                        momentum_metrics,

                    "ungated_v9":
                        base_metrics,

                    "gated_v9_1":
                        gated_metrics,
                },

                "gate_usage":
                    {
                        key:
                            int(value)

                        for (
                            key,
                            value
                        ) in action_counts.items()
                    },
            }
        )


    # =====================================================
    # POOLED RESULTS
    # =====================================================

    pooled_actual = np.asarray(
        pooled_actual,
        dtype=int,
    )


    pooled_base = np.asarray(
        pooled_base,
        dtype=int,
    )


    pooled_gated = np.asarray(
        pooled_gated,
        dtype=int,
    )


    pooled_momentum = np.asarray(
        pooled_momentum,
        dtype=int,
    )


    pooled_majority = np.asarray(
        pooled_majority,
        dtype=int,
    )


    base_metrics = calculate_metrics(
        pooled_actual,
        pooled_base,
    )


    gated_metrics = calculate_metrics(
        pooled_actual,
        pooled_gated,
    )


    momentum_metrics = calculate_metrics(
        pooled_actual,
        pooled_momentum,
    )


    majority_metrics = calculate_metrics(
        pooled_actual,
        pooled_majority,
    )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print(
        "\n\n=============================================="
    )

    print(
        " V9.1 AGGREGATED WALK-FORWARD RESULTS"
    )

    print(
        "=============================================="
    )


    print(
        "\nTotal Future Test Samples:",
        len(
            pooled_actual
        )
    )


    print_metrics(
        "Majority Baseline",
        majority_metrics,
    )


    print_metrics(
        "Relative Momentum Baseline",
        momentum_metrics,
    )


    print_metrics(
        "Original Ungated V9",
        base_metrics,
    )


    print_metrics(
        "V9.1 Validation-Gated",
        gated_metrics,
    )


    print(
        "\n=============================================="
    )

    print(
        " V9.1 IMPROVEMENTS"
    )

    print(
        "=============================================="
    )


    print(
        "\nVs Majority:",
        f'{gated_metrics["accuracy_percent"] - majority_metrics["accuracy_percent"]:+.2f} pp'
    )


    print(
        "Vs Momentum:",
        f'{gated_metrics["accuracy_percent"] - momentum_metrics["accuracy_percent"]:+.2f} pp'
    )


    print(
        "Vs Original V9:",
        f'{gated_metrics["accuracy_percent"] - base_metrics["accuracy_percent"]:+.2f} pp'
    )


    # =====================================================
    # PER CLASS
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " V9.1 PER-CLASS RESULTS"
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
            gated_metrics[
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
        " V9.1 CONFUSION MATRIX"
    )

    print(
        "=============================================="
    )


    matrix = (
        gated_metrics[
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
        prediction_rows
    )


    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )


    # =====================================================
    # PER STOCK
    # =====================================================

    per_stock_df = analyze_per_stock(
        prediction_df
    )


    per_stock_df.to_csv(
        PER_STOCK_PATH,
        index=False,
    )


    print(
        "\n=============================================="
    )

    print(
        " V9.1 PER-STOCK RESULTS"
    )

    print(
        "=============================================="
    )


    print(
        "\n"
        "SYMBOL              GROUP      GATED   "
        "BASE    MOMENTUM   MAJORITY"
    )


    for _, row in (
        per_stock_df.iterrows()
    ):

        print(
            f'{row["SYMBOL"]:<18}'
            f'{row["GROUP"]:<10}'
            f'{row["Gated_Accuracy"]:>7.2f}%'
            f'{row["Base_Model_Accuracy"]:>8.2f}%'
            f'{row["Momentum_Accuracy"]:>11.2f}%'
            f'{row["Majority_Accuracy"]:>11.2f}%'
        )


    # =====================================================
    # COUNT STOCKS
    # =====================================================

    stocks_beating_base = int(
        (
            per_stock_df[
                "Gated_vs_Base"
            ]
            >
            0
        )
        .sum()
    )


    stocks_beating_momentum = int(
        (
            per_stock_df[
                "Gated_vs_Momentum"
            ]
            >
            0
        )
        .sum()
    )


    stocks_beating_majority = int(
        (
            per_stock_df[
                "Gated_vs_Majority"
            ]
            >
            0
        )
        .sum()
    )


    print(
        "\nStocks where V9.1 beats original V9:",
        f"{stocks_beating_base}/15"
    )


    print(
        "Stocks where V9.1 beats Momentum:",
        f"{stocks_beating_momentum}/15"
    )


    print(
        "Stocks where V9.1 beats Majority:",
        f"{stocks_beating_majority}/15"
    )


    # =====================================================
    # SAVE JSON
    # =====================================================

    output = {

        "experiment":
            "StockVision V9.1 Validation-Gated Walk-Forward",

        "method":
            "Validation-only group gating with purged walk-forward",

        "target":
            "5-day stock excess return vs NIFTY 50",

        "target_horizon_days":
            TARGET_HORIZON,

        "gate_margin_macro_f1_pp":
            GATE_MARGIN_PP,

        "gate_groups":
            GROUPS,

        "symbol_group_mapping":
            SYMBOL_GROUP,

        "feature_count":
            len(
                FEATURE_COLUMNS
            ),

        "folds":
            all_fold_results,

        "aggregated": {

            "samples":
                int(
                    len(
                        pooled_actual
                    )
                ),

            "majority":
                majority_metrics,

            "momentum":
                momentum_metrics,

            "ungated_v9":
                base_metrics,

            "gated_v9_1":
                gated_metrics,

            "improvement_vs_majority_pp":
                round(
                    gated_metrics[
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
                    gated_metrics[
                        "accuracy_percent"
                    ]
                    -
                    momentum_metrics[
                        "accuracy_percent"
                    ],
                    2,
                ),

            "improvement_vs_ungated_v9_pp":
                round(
                    gated_metrics[
                        "accuracy_percent"
                    ]
                    -
                    base_metrics[
                        "accuracy_percent"
                    ],
                    2,
                ),

            "stocks_beating_ungated_v9":
                stocks_beating_base,

            "stocks_beating_momentum":
                stocks_beating_momentum,

            "stocks_beating_majority":
                stocks_beating_majority,
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
    # COMPLETE
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        " V9.1 WALK-FORWARD COMPLETE"
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


    print(
        "\nPer Stock:"
    )


    print(
        PER_STOCK_PATH
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    run_v9_1()