import os
import sys
import json

import joblib
import numpy as np
import tensorflow as tf


# =========================================================
# IMPORTS
# =========================================================

try:
    from .data import get_historical_data

    from .multi_features import (
        prepare_prediction_frame,
        FEATURE_COLUMNS,
        HORIZONS,
        SEQUENCE_LENGTH,
    )

    from .multi_train import (
        get_multi_model_paths,
    )

except ImportError:
    from data import get_historical_data

    from multi_features import (
        prepare_prediction_frame,
        FEATURE_COLUMNS,
        HORIZONS,
        SEQUENCE_LENGTH,
    )

    from multi_train import (
        get_multi_model_paths,
    )


# =========================================================
# NORMALIZE SYMBOL
# =========================================================

def normalize_symbol(symbol):
    symbol = (
        symbol
        .upper()
        .strip()
    )

    if not symbol:
        raise ValueError(
            "Stock symbol cannot be empty."
        )

    if (
        "." not in symbol
        and not symbol.startswith("^")
    ):
        symbol = (
            f"{symbol}.NS"
        )

    return symbol


# =========================================================
# LOAD JSON
# =========================================================

def load_json_file(path):
    if not os.path.exists(path):
        return {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# =========================================================
# SIGNAL
# =========================================================

def calculate_signal(
    lower_return_percent,
    upper_return_percent
):
    """
    Conservative signal logic.

    BULLISH:
    entire 80% estimated range is above 0.

    BEARISH:
    entire 80% estimated range is below 0.

    NEUTRAL:
    estimated range crosses 0.
    """

    if lower_return_percent > 0:
        return "BULLISH"

    if upper_return_percent < 0:
        return "BEARISH"

    return "NEUTRAL"


# =========================================================
# MODEL EVALUATION STATUS
# =========================================================

def evaluation_status(
    metrics
):
    beats_baseline = (
        metrics.get(
            "beats_baseline",
            False
        )
    )

    direction_accuracy = float(
        metrics.get(
            "direction_accuracy_percent",
            0
        )
    )

    if (
        beats_baseline
        and direction_accuracy >= 55
    ):
        return "STRONGER"

    if (
        beats_baseline
        or direction_accuracy >= 52
    ):
        return "MODERATE"

    return "EXPERIMENTAL"


# =========================================================
# MULTI-HORIZON PREDICTION
# =========================================================

def predict_multi_horizon(
    symbol
):
    symbol = normalize_symbol(
        symbol
    )

    (
        model_path,
        feature_scaler_path,
        target_scaler_path,
        calibration_path,
        metrics_path,
    ) = get_multi_model_paths(
        symbol
    )


    # =====================================================
    # CHECK FILES
    # =====================================================

    required_files = [
        model_path,
        feature_scaler_path,
        target_scaler_path,
    ]


    missing_files = [
        path
        for path in required_files
        if not os.path.exists(path)
    ]


    if missing_files:
        raise FileNotFoundError(
            f"Multi-horizon model not found for {symbol}. "
            f"Train it first using: "
            f"python ml\\multi_train.py {symbol}"
        )


    # =====================================================
    # LOAD MODEL
    # =====================================================

    model = (
        tf.keras.models.load_model(
            model_path,
            compile=False
        )
    )


    feature_scaler = joblib.load(
        feature_scaler_path
    )


    target_scaler = joblib.load(
        target_scaler_path
    )


    calibration = load_json_file(
        calibration_path
    )


    metrics_file = load_json_file(
        metrics_path
    )


    all_metrics = (
        metrics_file.get(
            "metrics",
            {}
        )
    )


    # =====================================================
    # GET LATEST REAL DATA
    # =====================================================

    raw_data = get_historical_data(
        symbol,
        period="2y"
    )


    data = prepare_prediction_frame(
        raw_data
    )


    if len(data) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Not enough usable historical data for {symbol}. "
            f"Need at least {SEQUENCE_LENGTH} rows."
        )


    # =====================================================
    # CURRENT CLOSE
    # =====================================================

    current_close = float(
        data[
            "Close"
        ].iloc[-1]
    )


    latest_date = (
        data.index[-1]
        .strftime(
            "%Y-%m-%d"
        )
    )


    # =====================================================
    # LAST 60 DAYS
    # =====================================================

    latest_sequence = (
        data[
            FEATURE_COLUMNS
        ]
        .iloc[
            -SEQUENCE_LENGTH:
        ]
    )


    scaled_sequence = (
        feature_scaler.transform(
            latest_sequence
        )
    )


    X = np.asarray(
        [
            scaled_sequence
        ],
        dtype=np.float32
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    predicted_scaled_returns = (
        model.predict(
            X,
            verbose=0
        )
    )


    predicted_returns = (
        target_scaler
        .inverse_transform(
            predicted_scaled_returns
        )
        .flatten()
    )


    # =====================================================
    # CREATE FORECASTS
    # =====================================================

    forecasts = []


    for index, horizon in enumerate(
        HORIZONS
    ):

        key = (
            f"{horizon}D"
        )


        predicted_return = float(
            predicted_returns[
                index
            ]
        )


        predicted_return_percent = (
            predicted_return
            *
            100
        )


        predicted_price = (
            current_close
            *
            (
                1
                +
                predicted_return
            )
        )


        # =================================================
        # VALIDATION ERROR BAND
        # =================================================

        calibration_data = (
            calibration.get(
                key,
                {}
            )
        )


        q80 = float(
            calibration_data.get(
                "q80_absolute_error_percent",
                0
            )
        )


        q90 = float(
            calibration_data.get(
                "q90_absolute_error_percent",
                0
            )
        )


        lower_return_80 = (
            predicted_return_percent
            -
            q80
        )


        upper_return_80 = (
            predicted_return_percent
            +
            q80
        )


        lower_return_90 = (
            predicted_return_percent
            -
            q90
        )


        upper_return_90 = (
            predicted_return_percent
            +
            q90
        )


        # =================================================
        # RANGE PRICES
        # =================================================

        lower_price_80 = (
            current_close
            *
            (
                1
                +
                lower_return_80
                /
                100
            )
        )


        upper_price_80 = (
            current_close
            *
            (
                1
                +
                upper_return_80
                /
                100
            )
        )


        lower_price_90 = (
            current_close
            *
            (
                1
                +
                lower_return_90
                /
                100
            )
        )


        upper_price_90 = (
            current_close
            *
            (
                1
                +
                upper_return_90
                /
                100
            )
        )


        # =================================================
        # SIGNAL
        # =================================================

        signal = calculate_signal(
            lower_return_percent=
                lower_return_80,

            upper_return_percent=
                upper_return_80,
        )


        # =================================================
        # TEST METRICS
        # =================================================

        horizon_metrics = (
            all_metrics.get(
                key,
                {}
            )
        )


        model_status = (
            evaluation_status(
                horizon_metrics
            )
        )


        # =================================================
        # FORECAST OBJECT
        # =================================================

        forecasts.append(
            {
                "horizon_days":
                    horizon,

                "horizon":
                    key,

                "expected_move_percent":
                    round(
                        predicted_return_percent,
                        4
                    ),

                "predicted_price":
                    round(
                        predicted_price,
                        2
                    ),

                "signal":
                    signal,

                "estimated_range_80": {
                    "lower_percent":
                        round(
                            lower_return_80,
                            4
                        ),

                    "upper_percent":
                        round(
                            upper_return_80,
                            4
                        ),

                    "lower_price":
                        round(
                            lower_price_80,
                            2
                        ),

                    "upper_price":
                        round(
                            upper_price_80,
                            2
                        ),
                },

                "estimated_range_90": {
                    "lower_percent":
                        round(
                            lower_return_90,
                            4
                        ),

                    "upper_percent":
                        round(
                            upper_return_90,
                            4
                        ),

                    "lower_price":
                        round(
                            lower_price_90,
                            2
                        ),

                    "upper_price":
                        round(
                            upper_price_90,
                            2
                        ),
                },

                "validation_error_80_percent":
                    round(
                        q80,
                        4
                    ),

                "validation_error_90_percent":
                    round(
                        q90,
                        4
                    ),

                "evaluation_status":
                    model_status,

                "test_direction_accuracy_percent":
                    horizon_metrics.get(
                        "direction_accuracy_percent"
                    ),

                "test_mape_percent":
                    horizon_metrics.get(
                        "mape_percent"
                    ),

                "beats_naive_baseline":
                    horizon_metrics.get(
                        "beats_baseline"
                    ),
            }
        )


    # =====================================================
    # RESPONSE
    # =====================================================

    return {
        "symbol":
            symbol,

        "latest_market_date":
            latest_date,

        "current_close":
            round(
                current_close,
                2
            ),

        "model":
            "Multi-Horizon BiLSTM",

        "lookback_days":
            SEQUENCE_LENGTH,

        "features_used":
            len(
                FEATURE_COLUMNS
            ),

        "forecast_horizons":
            HORIZONS,

        "forecast_unit":
            "percent_return",

        "forecasts":
            forecasts,

        "important_note":
            (
                "Forecast ranges are estimated from "
                "historical validation errors and are "
                "not guaranteed confidence intervals."
            ),
    }


# =========================================================
# TERMINAL DISPLAY
# =========================================================

def print_prediction(
    result
):
    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION FUTURE FORECAST"
    )

    print(
        "=============================================="
    )


    print(
        f'\nSymbol: {result["symbol"]}'
    )


    print(
        "Latest Market Date:",
        result[
            "latest_market_date"
        ]
    )


    print(
        "Current Close:",
        f'₹{result["current_close"]:.2f}'
    )


    print(
        "\nModel:",
        result[
            "model"
        ]
    )


    print(
        "Lookback:",
        result[
            "lookback_days"
        ],
        "days"
    )


    print(
        "Features:",
        result[
            "features_used"
        ]
    )


    print(
        "\n=============================================="
    )

    print(
        " FORECASTS"
    )

    print(
        "=============================================="
    )


    for forecast in result[
        "forecasts"
    ]:

        print(
            f'\n---------- {forecast["horizon"]} ----------'
        )


        move = forecast[
            "expected_move_percent"
        ]


        print(
            "Expected Move:",
            f'{move:+.4f}%'
        )


        print(
            "Predicted Price:",
            f'₹{forecast["predicted_price"]:.2f}'
        )


        range_80 = forecast[
            "estimated_range_80"
        ]


        print(
            "Estimated 80% Range:",
            (
                f'{range_80["lower_percent"]:+.4f}% '
                f'to '
                f'{range_80["upper_percent"]:+.4f}%'
            )
        )


        print(
            "Estimated Price Range:",
            (
                f'₹{range_80["lower_price"]:.2f} '
                f'to '
                f'₹{range_80["upper_price"]:.2f}'
            )
        )


        print(
            "Signal:",
            forecast[
                "signal"
            ]
        )


        print(
            "Evaluation:",
            forecast[
                "evaluation_status"
            ]
        )


        print(
            "Direction Accuracy:",
            forecast[
                "test_direction_accuracy_percent"
            ]
        )


        print(
            "Beats Baseline:",
            forecast[
                "beats_naive_baseline"
            ]
        )


# =========================================================
# COMMAND LINE
# =========================================================

if __name__ == "__main__":

    symbol = (
        sys.argv[1]
        if len(
            sys.argv
        ) > 1
        else "RELIANCE.NS"
    )


    result = predict_multi_horizon(
        symbol
    )


    print_prediction(
        result
    )