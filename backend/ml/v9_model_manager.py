import threading

try:
    from .v9_predict import (
        predict_v9,
        normalize_symbol,
        load_v9_model,
        load_v9_metadata,
    )

except ImportError:
    from v9_predict import (
        predict_v9,
        normalize_symbol,
        load_v9_model,
        load_v9_metadata,
    )


# =========================================================
# LOCK
#
# Keeps model access safe if multiple API requests
# arrive at nearly the same time.
# =========================================================

_V9_LOCK = threading.Lock()


# =========================================================
# MODEL STATUS
# =========================================================

def get_v9_status():

    try:

        model = load_v9_model()

        metadata = load_v9_metadata()


        return {

            "status":
                "ready",

            "model_version":
                metadata.get(
                    "model_version",
                    "V9",
                ),

            "engine":
                metadata.get(
                    "engine",
                    "Relative Strength Intelligence",
                ),

            "production_model":
                metadata.get(
                    "production_model"
                ),

            "benchmark":
                metadata.get(
                    "benchmark",
                    "NIFTY 50",
                ),

            "feature_count":
                metadata.get(
                    "feature_count"
                ),

            "training_stock_count":
                metadata.get(
                    "training_stock_count"
                ),

            "training_end_date":
                metadata.get(
                    "training_end_date"
                ),

            "model_loaded":
                model is not None,
        }


    except Exception as error:

        return {

            "status":
                "error",

            "model_version":
                "V9",

            "error":
                str(
                    error
                ),
        }


# =========================================================
# RELATIVE PREDICTION
# =========================================================

def predict_relative_strength(
    symbol,
):

    normalized_symbol = normalize_symbol(
        symbol
    )


    with _V9_LOCK:

        result = predict_v9(
            normalized_symbol
        )


    # =====================================================
    # SIMPLE DISPLAY NOTE
    #
    # This is NOT calibrated confidence.
    #
    # We only compare how close the top two raw
    # probabilities are so the frontend can avoid
    # presenting a close decision too strongly.
    # =====================================================

    probabilities = result.get(
        "probabilities",
        {}
    )


    probability_values = sorted(
        [
            float(
                probabilities.get(
                    "underperform",
                    0.0,
                )
            ),

            float(
                probabilities.get(
                    "neutral",
                    0.0,
                )
            ),

            float(
                probabilities.get(
                    "outperform",
                    0.0,
                )
            ),
        ],
        reverse=True,
    )


    if len(
        probability_values
    ) >= 2:

        probability_gap = (
            probability_values[0]
            -
            probability_values[1]
        )

    else:

        probability_gap = 0.0


    result[
        "top_probability_gap"
    ] = round(
        float(
            probability_gap
        ),
        4,
    )


    result[
        "display_note"
    ] = (
        "Relative-strength signal versus NIFTY 50. "
        "Probability values are raw model scores, "
        "not guaranteed confidence levels."
    )


    return result


# =========================================================
# COMMAND-LINE TEST
# =========================================================

if __name__ == "__main__":

    import json
    import sys


    symbol = (
        sys.argv[1]

        if len(
            sys.argv
        ) > 1

        else

        "RELIANCE.NS"
    )


    print(
        json.dumps(
            predict_relative_strength(
                symbol
            ),
            indent=4,
        )
    )