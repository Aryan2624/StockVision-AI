import os
import sys
import json

import joblib
import numpy as np


# =========================================================
# IMPORTS
# =========================================================

try:
    from .data import get_historical_data

    from .relative_dataset_v9 import (
        FEATURE_COLUMNS,
        CLASS_NAMES,
        MARKET_SYMBOL,
        TARGET_HORIZON,
        EXCESS_RETURN_THRESHOLD,
        STOCKS,
        build_market_features,
        build_stock_dataset,
    )

except ImportError:
    from data import get_historical_data

    from relative_dataset_v9 import (
        FEATURE_COLUMNS,
        CLASS_NAMES,
        MARKET_SYMBOL,
        TARGET_HORIZON,
        EXCESS_RETURN_THRESHOLD,
        STOCKS,
        build_market_features,
        build_stock_dataset,
    )


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


MODEL_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_production_model.pkl",
)


METADATA_PATH = os.path.join(
    MODELS_DIR,
    "stockvision_v9_production_metadata.json",
)


# =========================================================
# CACHE
# =========================================================

_MODEL_CACHE = None

_METADATA_CACHE = None


# =========================================================
# NORMALIZE SYMBOL
# =========================================================

def normalize_symbol(symbol):

    symbol = (
        str(symbol)
        .strip()
        .upper()
    )


    if not symbol:

        raise ValueError(
            "Stock symbol cannot be empty."
        )


    # Already contains exchange/index suffix
    if (
        "." in symbol
        or
        symbol.startswith("^")
    ):

        return symbol


    # Default project market = NSE
    return f"{symbol}.NS"


# =========================================================
# LOAD MODEL
# =========================================================

def load_v9_model():

    global _MODEL_CACHE


    if _MODEL_CACHE is not None:

        return _MODEL_CACHE


    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "V9 production model not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run:\n"
            "python ml\\train_v9_production.py"
        )


    _MODEL_CACHE = joblib.load(
        MODEL_PATH
    )


    return _MODEL_CACHE


# =========================================================
# LOAD METADATA
# =========================================================

def load_v9_metadata():

    global _METADATA_CACHE


    if _METADATA_CACHE is not None:

        return _METADATA_CACHE


    if not os.path.exists(
        METADATA_PATH
    ):

        raise FileNotFoundError(
            "V9 production metadata not found:\n"
            f"{METADATA_PATH}"
        )


    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        _METADATA_CACHE = json.load(
            file
        )


    return _METADATA_CACHE


# =========================================================
# GET LIVE FEATURE ROW
# =========================================================

def prepare_latest_features(
    symbol,
):

    symbol = normalize_symbol(
        symbol
    )


    print(
        f"Downloading live history for {symbol}..."
    )


    # Enough history for 50-day and rolling indicators.
    stock_data = get_historical_data(
        symbol,
        period="1y",
    )


    print(
        f"Downloading benchmark {MARKET_SYMBOL}..."
    )


    nifty_data = get_historical_data(
        MARKET_SYMBOL,
        period="1y",
    )


    # =====================================================
    # CREATE NIFTY FEATURES
    # =====================================================

    market_features = build_market_features(
        nifty_data
    )


    # =====================================================
    # CREATE EXACT SAME V9 FEATURES
    # =====================================================

    stock_frame = build_stock_dataset(
        symbol=
            symbol,

        stock_data=
            stock_data,

        market_features=
            market_features,
    )


    # =====================================================
    # CLEAN FEATURE ROWS ONLY
    #
    # IMPORTANT:
    # Latest rows do NOT have future targets.
    #
    # That is expected because live prediction happens
    # before the next 5 trading days exist.
    #
    # Therefore we drop NaNs using FEATURE_COLUMNS only.
    # =====================================================

    prediction_frame = (
        stock_frame
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna(
            subset=
                FEATURE_COLUMNS
        )
        .copy()
    )


    if prediction_frame.empty:

        raise ValueError(
            f"Not enough usable historical data for {symbol}."
        )


    latest = (
        prediction_frame
        .iloc[-1]
    )


    latest_date = (
        prediction_frame
        .index[-1]
    )


    X = (
        latest[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
        .reshape(
            1,
            -1,
        )
    )


    if X.shape[1] != len(
        FEATURE_COLUMNS
    ):

        raise ValueError(
            "V9 feature count mismatch."
        )


    # =====================================================
    # CURRENT RELATIVE-STRENGTH INFORMATION
    # =====================================================

    stock_return_5 = float(
        latest[
            "RETURN_5"
        ]
    )


    nifty_return_5 = float(
        latest[
            "NIFTY_RETURN_5"
        ]
    )


    relative_return_5 = float(
        latest[
            "RELATIVE_RETURN_5"
        ]
    )


    return {

        "symbol":
            symbol,

        "date":
            latest_date,

        "features":
            X,

        "stock_return_5":
            stock_return_5,

        "nifty_return_5":
            nifty_return_5,

        "relative_return_5":
            relative_return_5,
    }


# =========================================================
# PREDICT
# =========================================================

def predict_v9(
    symbol,
):

    symbol = normalize_symbol(
        symbol
    )


    model = load_v9_model()

    metadata = load_v9_metadata()


    prepared = prepare_latest_features(
        symbol
    )


    X = prepared[
        "features"
    ]


    # =====================================================
    # PROBABILITIES
    # =====================================================

    raw_probabilities = (
        model.predict_proba(
            X
        )[0]
    )


    model_classes = (
        model.classes_
        .astype(int)
    )


    probability_map = {

        int(class_id):
            float(
                probability
            )

        for class_id, probability
        in zip(
            model_classes,
            raw_probabilities,
        )
    }


    underperform_probability = (
        probability_map.get(
            0,
            0.0,
        )
    )


    neutral_probability = (
        probability_map.get(
            1,
            0.0,
        )
    )


    outperform_probability = (
        probability_map.get(
            2,
            0.0,
        )
    )


    # =====================================================
    # PREDICTED CLASS
    # =====================================================

    predicted_class = int(
        model.predict(
            X
        )[0]
    )


    signal = CLASS_NAMES[
        predicted_class
    ]


    top_probability = max(
        underperform_probability,
        neutral_probability,
        outperform_probability,
    )


    # =====================================================
    # RESEARCH SCOPE
    # =====================================================

    validated_stock = (
        symbol
        in
        STOCKS
    )


    research_metrics = (
        metadata.get(
            "research_evaluation"
        )
        or
        {}
    )


    # =====================================================
    # RESPONSE
    # =====================================================

    result = {

        "symbol":
            symbol,

        "as_of":
            str(
                prepared[
                    "date"
                ].date()
            ),

        "engine":
            "StockVision V9 Relative Strength Intelligence",

        "benchmark":
            "NIFTY 50",

        "benchmark_symbol":
            MARKET_SYMBOL,

        "horizon":
            f"{TARGET_HORIZON} trading days",

        "signal":
            signal,

        "signal_class":
            predicted_class,

        "probabilities": {

            "underperform":
                round(
                    underperform_probability,
                    4,
                ),

            "neutral":
                round(
                    neutral_probability,
                    4,
                ),

            "outperform":
                round(
                    outperform_probability,
                    4,
                ),
        },

        # Do not call this calibrated confidence.
        "top_probability":
            round(
                top_probability,
                4,
            ),

        "probability_note":
            (
                "Raw model probability score. "
                "It is not a calibrated certainty estimate."
            ),

        "current_market_context": {

            "stock_previous_5d_return_percent":
                round(
                    prepared[
                        "stock_return_5"
                    ]
                    *
                    100,
                    2,
                ),

            "nifty_previous_5d_return_percent":
                round(
                    prepared[
                        "nifty_return_5"
                    ]
                    *
                    100,
                    2,
                ),

            "previous_5d_relative_strength_percent":
                round(
                    prepared[
                        "relative_return_5"
                    ]
                    *
                    100,
                    2,
                ),
        },

        "target_definition": {

            "underperform":
                (
                    f"Future stock excess return < "
                    f"-{EXCESS_RETURN_THRESHOLD * 100:.2f}%"
                ),

            "neutral":
                (
                    f"Future excess return between "
                    f"-{EXCESS_RETURN_THRESHOLD * 100:.2f}% "
                    f"and +{EXCESS_RETURN_THRESHOLD * 100:.2f}%"
                ),

            "outperform":
                (
                    f"Future stock excess return > "
                    f"+{EXCESS_RETURN_THRESHOLD * 100:.2f}%"
                ),
        },

        "research_scope": {

            "stock_in_original_15_stock_evaluation":
                validated_stock,

            "note":
                (
                    "Performance metrics are validated on the "
                    "original 15-stock V9 research universe."
                    if validated_stock

                    else

                    "This symbol was not part of the original "
                    "15-stock V9 evaluation universe, so the "
                    "reported historical V9 metrics should not "
                    "be assumed to apply equally to this stock."
                ),
        },

        "walk_forward_evaluation": {

            "accuracy_percent":
                research_metrics.get(
                    "walk_forward_accuracy_percent"
                ),

            "macro_f1_percent":
                research_metrics.get(
                    "walk_forward_macro_f1_percent"
                ),

            "balanced_accuracy_percent":
                research_metrics.get(
                    "walk_forward_balanced_accuracy_percent"
                ),

            "improvement_vs_majority_pp":
                research_metrics.get(
                    "improvement_vs_majority_pp"
                ),

            "improvement_vs_momentum_pp":
                research_metrics.get(
                    "improvement_vs_momentum_pp"
                ),
        },
    }


    return result


# =========================================================
# CLI TEST
# =========================================================

def main():

    if len(
        sys.argv
    ) < 2:

        symbol = "RELIANCE.NS"

    else:

        symbol = sys.argv[
            1
        ]


    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION V9 LIVE PREDICTION"
    )

    print(
        "=============================================="
    )


    result = predict_v9(
        symbol
    )


    print(
        "\n"
        +
        json.dumps(
            result,
            indent=4,
        )
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()