import os
import sys
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)


# =========================================================
# IMPORT STOCKVISION V3 DATA PIPELINE
# =========================================================

try:

    from .data import (
        get_historical_data,
    )

    from .multi_features import (
        prepare_training_frame,
        FEATURE_COLUMNS,
        TARGET_COLUMNS,
        HORIZONS,
        SEQUENCE_LENGTH,
    )

except ImportError:

    from data import (
        get_historical_data,
    )

    from multi_features import (
        prepare_training_frame,
        FEATURE_COLUMNS,
        TARGET_COLUMNS,
        HORIZONS,
        SEQUENCE_LENGTH,
    )


# =========================================================
# CONFIG
# =========================================================

np.random.seed(42)

tf.random.set_seed(42)


MAX_HORIZON = max(
    HORIZONS
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODELS_DIR = os.path.join(
    BASE_DIR,
    "models",
)


os.makedirs(
    MODELS_DIR,
    exist_ok=True,
)


# =========================================================
# SYMBOL HELPERS
# =========================================================

def normalize_symbol(
    symbol,
):

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
        and
        not symbol.startswith("^")
    ):

        symbol = (
            f"{symbol}.NS"
        )


    return symbol


def safe_symbol_name(
    symbol,
):

    return (
        symbol
        .replace(
            ".",
            "_",
        )
        .replace(
            "^",
            "",
        )
        .upper()
    )


# =========================================================
# V3-LIKE BiLSTM
# =========================================================

def build_model():

    model = Sequential(
        [

            Input(
                shape=(
                    SEQUENCE_LENGTH,
                    len(
                        FEATURE_COLUMNS
                    ),
                )
            ),


            Bidirectional(
                LSTM(
                    64,
                    return_sequences=True,
                )
            ),


            Dropout(
                0.25
            ),


            Bidirectional(
                LSTM(
                    32
                )
            ),


            Dropout(
                0.25
            ),


            Dense(
                32,
                activation="relu",
            ),


            Dropout(
                0.10
            ),


            Dense(
                len(
                    HORIZONS
                ),
                activation="linear",
            ),
        ]
    )


    optimizer = (
        tf.keras.optimizers.Adam(
            learning_rate=0.0005
        )
    )


    model.compile(

        optimizer=
            optimizer,

        loss=
            tf.keras.losses.Huber(),

        metrics=[
            "mae"
        ],
    )


    return model


# =========================================================
# CREATE ONE SEQUENCE
# =========================================================

def create_sequence(
    scaled_features,
    position,
):

    start = (
        position
        -
        SEQUENCE_LENGTH
        +
        1
    )


    if start < 0:

        return None


    sequence = (
        scaled_features[
            start:
            position + 1
        ]
    )


    if len(
        sequence
    ) != SEQUENCE_LENGTH:

        return None


    return sequence


# =========================================================
# BUILD SAMPLE SET
# =========================================================

def build_samples(
    scaled_features,
    scaled_targets,
    original_targets,
    close_prices,
    dates,
    positions,
):

    X = []

    y_scaled = []

    y_original = []

    closes = []

    sample_dates = []


    for position in positions:

        sequence = (
            create_sequence(
                scaled_features,
                position,
            )
        )


        if sequence is None:

            continue


        X.append(
            sequence
        )


        y_scaled.append(
            scaled_targets[
                position
            ]
        )


        y_original.append(
            original_targets[
                position
            ]
        )


        closes.append(
            close_prices[
                position
            ]
        )


        sample_dates.append(
            dates[
                position
            ]
        )


    return {

        "X": np.asarray(
            X,
            dtype=np.float32,
        ),

        "y_scaled": np.asarray(
            y_scaled,
            dtype=np.float32,
        ),

        "y_original": np.asarray(
            y_original,
            dtype=np.float32,
        ),

        "close": np.asarray(
            closes,
            dtype=np.float32,
        ),

        "dates":
            sample_dates,
    }


# =========================================================
# METRICS
# =========================================================

def calculate_horizon_metrics(
    actual_returns,
    predicted_returns,
    current_close,
):

    results = {}


    for (
        column_index,
        horizon,
    ) in enumerate(
        HORIZONS
    ):

        actual = (
            actual_returns[
                :,
                column_index
            ]
        )


        predicted = (
            predicted_returns[
                :,
                column_index
            ]
        )


        actual_price = (
            current_close
            *
            (
                1
                +
                actual
            )
        )


        predicted_price = (
            current_close
            *
            (
                1
                +
                predicted
            )
        )


        baseline_price = (
            current_close
        )


        # ================================================
        # RETURN ERROR
        # ================================================

        return_error = (
            predicted
            -
            actual
        )


        return_mae = (
            np.mean(
                np.abs(
                    return_error
                )
            )
            *
            100
        )


        return_rmse = (
            np.sqrt(
                np.mean(
                    return_error ** 2
                )
            )
            *
            100
        )


        # ================================================
        # PRICE ERROR
        # ================================================

        price_error = (
            predicted_price
            -
            actual_price
        )


        price_mae = (
            np.mean(
                np.abs(
                    price_error
                )
            )
        )


        price_rmse = (
            np.sqrt(
                np.mean(
                    price_error ** 2
                )
            )
        )


        # ================================================
        # NAIVE BASELINE
        # ================================================

        baseline_error = (
            baseline_price
            -
            actual_price
        )


        baseline_mae = (
            np.mean(
                np.abs(
                    baseline_error
                )
            )
        )


        baseline_rmse = (
            np.sqrt(
                np.mean(
                    baseline_error ** 2
                )
            )
        )


        if baseline_mae > 0:

            improvement = (
                (
                    baseline_mae
                    -
                    price_mae
                )
                /
                baseline_mae
                *
                100
            )

        else:

            improvement = 0.0


        # ================================================
        # DIRECTION ACCURACY
        # ================================================

        direction_accuracy = (
            np.mean(
                (
                    actual >
                    0
                )
                ==
                (
                    predicted >
                    0
                )
            )
            *
            100
        )


        # ================================================
        # CORRELATION
        # ================================================

        if (
            len(
                actual
            ) >= 2
            and
            np.std(
                actual
            ) > 0
            and
            np.std(
                predicted
            ) > 0
        ):

            correlation = (
                np.corrcoef(
                    actual,
                    predicted,
                )[
                    0,
                    1
                ]
            )


            if not np.isfinite(
                correlation
            ):

                correlation = 0.0

        else:

            correlation = 0.0


        # ================================================
        # MAPE
        # ================================================

        valid = (
            actual_price != 0
        )


        if np.any(
            valid
        ):

            mape = (
                np.mean(
                    np.abs(
                        (
                            actual_price[
                                valid
                            ]
                            -
                            predicted_price[
                                valid
                            ]
                        )
                        /
                        actual_price[
                            valid
                        ]
                    )
                )
                *
                100
            )

        else:

            mape = 0.0


        results[
            f"{horizon}D"
        ] = {

            "return_mae_percent":
                round(
                    float(
                        return_mae
                    ),
                    4,
                ),

            "return_rmse_percent":
                round(
                    float(
                        return_rmse
                    ),
                    4,
                ),

            "price_mae":
                round(
                    float(
                        price_mae
                    ),
                    2,
                ),

            "price_rmse":
                round(
                    float(
                        price_rmse
                    ),
                    2,
                ),

            "mape_percent":
                round(
                    float(
                        mape
                    ),
                    4,
                ),

            "direction_accuracy_percent":
                round(
                    float(
                        direction_accuracy
                    ),
                    2,
                ),

            "return_correlation":
                round(
                    float(
                        correlation
                    ),
                    4,
                ),

            "baseline_mae":
                round(
                    float(
                        baseline_mae
                    ),
                    2,
                ),

            "baseline_rmse":
                round(
                    float(
                        baseline_rmse
                    ),
                    2,
                ),

            "improvement_vs_baseline_percent":
                round(
                    float(
                        improvement
                    ),
                    2,
                ),

            "beats_baseline":
                bool(
                    price_mae
                    <
                    baseline_mae
                ),
        }


    return results


# =========================================================
# PRINT METRICS
# =========================================================

def print_metrics(
    title,
    metrics,
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


    for horizon in HORIZONS:

        key = (
            f"{horizon}D"
        )


        result = (
            metrics[
                key
            ]
        )


        print(
            f"\n---------- {key} ----------"
        )


        print(
            "Return MAE:",
            f'{result["return_mae_percent"]:.4f}%'
        )


        print(
            "Price MAE:",
            f'₹{result["price_mae"]:.2f}'
        )


        print(
            "MAPE:",
            f'{result["mape_percent"]:.4f}%'
        )


        print(
            "Direction Accuracy:",
            f'{result["direction_accuracy_percent"]:.2f}%'
        )


        print(
            "Return Correlation:",
            result[
                "return_correlation"
            ]
        )


        print(
            "Naive Baseline MAE:",
            f'₹{result["baseline_mae"]:.2f}'
        )


        print(
            "Improvement vs Baseline:",
            f'{result["improvement_vs_baseline_percent"]:.2f}%'
        )


        print(
            "Beats Baseline:",
            result[
                "beats_baseline"
            ]
        )


# =========================================================
# PREPARE WALK-FORWARD FOLDS
# =========================================================

def find_available_folds(
    data,
    number_of_folds=5,
):

    years = sorted(
        data.index.year.unique()
    )


    folds = []


    # We require:
    #
    # training history
    # previous year as validation
    # current year as test

    for test_year in years:

        validation_year = (
            test_year
            -
            1
        )


        if (
            validation_year
            not in years
        ):

            continue


        test_rows = (
            data.index.year
            ==
            test_year
        )


        validation_rows = (
            data.index.year
            ==
            validation_year
        )


        train_rows = (
            data.index.year
            <
            validation_year
        )


        if (
            np.sum(
                test_rows
            ) <
            40
        ):

            continue


        if (
            np.sum(
                validation_rows
            ) <
            100
        ):

            continue


        if (
            np.sum(
                train_rows
            ) <
            700
        ):

            continue


        folds.append(
            {
                "validation_year":
                    int(
                        validation_year
                    ),

                "test_year":
                    int(
                        test_year
                    ),
            }
        )


    return folds[
        -number_of_folds:
    ]


# =========================================================
# RUN WALK-FORWARD BACKTEST
# =========================================================

def run_walk_forward(
    symbol,
    number_of_folds=5,
):

    symbol = (
        normalize_symbol(
            symbol
        )
    )


    print(
        "\n=============================================="
    )

    print(
        " STOCKVISION WALK-FORWARD BACKTEST"
    )

    print(
        "=============================================="
    )


    print(
        f"\nStock: {symbol}"
    )


    print(
        "Model: V3-like Multi-Horizon BiLSTM"
    )


    print(
        "Features:",
        len(
            FEATURE_COLUMNS
        )
    )


    print(
        "Horizons:",
        HORIZONS
    )


    print(
        "Lookback:",
        SEQUENCE_LENGTH
    )


    print(
        "\nDownloading historical data..."
    )


    raw_data = (
        get_historical_data(
            symbol,
            period="10y",
        )
    )


    data = (
        prepare_training_frame(
            raw_data
        )
    )


    data = (
        data.sort_index()
    )


    print(
        "Usable rows:",
        len(
            data
        )
    )


    print(
        "Date range:",
        data.index[
            0
        ].date(),
        "→",
        data.index[
            -1
        ].date(),
    )


    # =====================================================
    # FOLDS
    # =====================================================

    folds = (
        find_available_folds(
            data,
            number_of_folds=
                number_of_folds,
        )
    )


    if not folds:

        raise ValueError(
            "Unable to create walk-forward folds."
        )


    print(
        "\nWalk-forward folds:"
    )


    for fold in folds:

        print(
            f'Validation {fold["validation_year"]}'
            f' → Test {fold["test_year"]}'
        )


    # =====================================================
    # RAW ARRAYS
    # =====================================================

    feature_values = (
        data[
            FEATURE_COLUMNS
        ]
        .values
        .astype(
            np.float32
        )
    )


    target_values = (
        data[
            TARGET_COLUMNS
        ]
        .values
        .astype(
            np.float32
        )
    )


    close_values = (
        data[
            "Close"
        ]
        .values
        .astype(
            np.float32
        )
    )


    dates = list(
        data.index
    )


    years = (
        data.index.year
        .to_numpy()
    )


    all_fold_results = []

    all_predictions = []


    # =====================================================
    # EACH WALK-FORWARD FOLD
    # =====================================================

    for (
        fold_number,
        fold,
    ) in enumerate(
        folds,
        start=1,
    ):

        validation_year = (
            fold[
                "validation_year"
            ]
        )


        test_year = (
            fold[
                "test_year"
            ]
        )


        print(
            "\n\n##############################################"
        )

        print(
            f" FOLD {fold_number}/{len(folds)}"
        )

        print(
            "##############################################"
        )


        print(
            f"\nValidation year: {validation_year}"
        )


        print(
            f"Test year: {test_year}"
        )


        # =================================================
        # LOCATE YEAR BOUNDARIES
        # =================================================

        validation_positions = np.where(
            years ==
            validation_year
        )[
            0
        ]


        test_positions = np.where(
            years ==
            test_year
        )[
            0
        ]


        if (
            len(
                validation_positions
            ) == 0
            or
            len(
                test_positions
            ) == 0
        ):

            continue


        validation_start = int(
            validation_positions[
                0
            ]
        )


        test_start = int(
            test_positions[
                0
            ]
        )


        test_end = int(
            test_positions[
                -1
            ]
            +
            1
        )


        # =================================================
        # PURGED TRAIN / VALIDATION
        #
        # Important:
        #
        # RETURN_10D uses prices up to 10 future trading
        # days.
        #
        # Therefore the last 10 target rows before a new
        # evaluation period are excluded.
        #
        # This prevents future target values from leaking
        # across the boundary.
        # =================================================

        train_target_end = (
            validation_start
            -
            MAX_HORIZON
        )


        validation_target_end = (
            test_start
            -
            MAX_HORIZON
        )


        train_sample_positions = np.arange(
            SEQUENCE_LENGTH - 1,
            train_target_end,
        )


        validation_sample_positions = np.arange(
            validation_start,
            validation_target_end,
        )


        test_sample_positions = np.arange(
            test_start,
            test_end,
        )


        if (
            len(
                train_sample_positions
            ) <
            500
        ):

            print(
                "Skipping fold: insufficient training samples."
            )

            continue


        if (
            len(
                validation_sample_positions
            ) <
            50
        ):

            print(
                "Skipping fold: insufficient validation samples."
            )

            continue


        if (
            len(
                test_sample_positions
            ) <
            30
        ):

            print(
                "Skipping fold: insufficient test samples."
            )

            continue


        # =================================================
        # FIT SCALERS ON TRAIN DATA ONLY
        # =================================================

        feature_scaler = (
            StandardScaler()
        )


        target_scaler = (
            StandardScaler()
        )


        feature_scaler.fit(
            feature_values[
                :train_target_end
            ]
        )


        target_scaler.fit(
            target_values[
                train_sample_positions
            ]
        )


        scaled_features = (
            feature_scaler.transform(
                feature_values
            )
        )


        scaled_targets = (
            target_scaler.transform(
                target_values
            )
        )


        # =================================================
        # BUILD SEQUENCES
        # =================================================

        train_set = (
            build_samples(

                scaled_features,
                scaled_targets,
                target_values,
                close_values,
                dates,
                train_sample_positions,
            )
        )


        validation_set = (
            build_samples(

                scaled_features,
                scaled_targets,
                target_values,
                close_values,
                dates,
                validation_sample_positions,
            )
        )


        test_set = (
            build_samples(

                scaled_features,
                scaled_targets,
                target_values,
                close_values,
                dates,
                test_sample_positions,
            )
        )


        print(
            "\nSamples:"
        )


        print(
            "Train:",
            train_set[
                "X"
            ].shape
        )


        print(
            "Validation:",
            validation_set[
                "X"
            ].shape
        )


        print(
            "Test:",
            test_set[
                "X"
            ].shape
        )


        # =================================================
        # RESET MODEL STATE
        # =================================================

        tf.keras.backend.clear_session()


        np.random.seed(
            42
            +
            fold_number
        )


        tf.random.set_seed(
            42
            +
            fold_number
        )


        model = (
            build_model()
        )


        callbacks = [

            EarlyStopping(

                monitor=
                    "val_loss",

                patience=
                    6,

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
                    0.00001,

                verbose=
                    1,
            ),
        ]


        # =================================================
        # TRAIN THIS FOLD
        # =================================================

        print(
            "\nTraining fold..."
        )


        history = (
            model.fit(

                train_set[
                    "X"
                ],

                train_set[
                    "y_scaled"
                ],

                validation_data=(

                    validation_set[
                        "X"
                    ],

                    validation_set[
                        "y_scaled"
                    ],
                ),

                epochs=
                    50,

                batch_size=
                    32,

                shuffle=
                    False,

                callbacks=
                    callbacks,

                verbose=
                    1,
            )
        )


        best_epoch = int(
            np.argmin(
                history.history[
                    "val_loss"
                ]
            )
            +
            1
        )


        # =================================================
        # TEST PREDICTIONS
        # =================================================

        prediction_scaled = (
            model.predict(

                test_set[
                    "X"
                ],

                verbose=0,
            )
        )


        prediction_returns = (
            target_scaler.inverse_transform(
                prediction_scaled
            )
        )


        actual_returns = (
            test_set[
                "y_original"
            ]
        )


        current_close = (
            test_set[
                "close"
            ]
        )


        metrics = (
            calculate_horizon_metrics(

                actual_returns=
                    actual_returns,

                predicted_returns=
                    prediction_returns,

                current_close=
                    current_close,
            )
        )


        print_metrics(

            f"FOLD {fold_number} — TEST YEAR {test_year}",

            metrics,
        )


        all_fold_results.append(
            {

                "fold":
                    fold_number,

                "validation_year":
                    validation_year,

                "test_year":
                    test_year,

                "best_epoch":
                    best_epoch,

                "train_samples":
                    int(
                        len(
                            train_set[
                                "X"
                            ]
                        )
                    ),

                "validation_samples":
                    int(
                        len(
                            validation_set[
                                "X"
                            ]
                        )
                    ),

                "test_samples":
                    int(
                        len(
                            test_set[
                                "X"
                            ]
                        )
                    ),

                "metrics":
                    metrics,
            }
        )


        # =================================================
        # SAVE INDIVIDUAL PREDICTIONS IN MEMORY
        # =================================================

        for (
            sample_index,
            date,
        ) in enumerate(
            test_set[
                "dates"
            ]
        ):

            row = {

                "date":
                    str(
                        pd.Timestamp(
                            date
                        ).date()
                    ),

                "test_year":
                    test_year,

                "current_close":
                    float(
                        current_close[
                            sample_index
                        ]
                    ),
            }


            for (
                horizon_index,
                horizon,
            ) in enumerate(
                HORIZONS
            ):

                actual_return = float(

                    actual_returns[
                        sample_index,
                        horizon_index
                    ]
                )


                predicted_return = float(

                    prediction_returns[
                        sample_index,
                        horizon_index
                    ]
                )


                row[
                    f"actual_return_{horizon}d"
                ] = (
                    actual_return
                    *
                    100
                )


                row[
                    f"predicted_return_{horizon}d"
                ] = (
                    predicted_return
                    *
                    100
                )


                row[
                    f"actual_price_{horizon}d"
                ] = (
                    float(
                        current_close[
                            sample_index
                        ]
                    )
                    *
                    (
                        1
                        +
                        actual_return
                    )
                )


                row[
                    f"predicted_price_{horizon}d"
                ] = (
                    float(
                        current_close[
                            sample_index
                        ]
                    )
                    *
                    (
                        1
                        +
                        predicted_return
                    )
                )


            all_predictions.append(
                row
            )


    # =====================================================
    # ENSURE FOLDS COMPLETED
    # =====================================================

    if not all_predictions:

        raise RuntimeError(
            "No walk-forward predictions were produced."
        )


    # =====================================================
    # AGGREGATED WALK-FORWARD METRICS
    # =====================================================

    prediction_df = pd.DataFrame(
        all_predictions
    )


    aggregated_actual = []

    aggregated_predicted = []


    for horizon in HORIZONS:

        aggregated_actual.append(

            prediction_df[
                f"actual_return_{horizon}d"
            ].to_numpy()
            /
            100
        )


        aggregated_predicted.append(

            prediction_df[
                f"predicted_return_{horizon}d"
            ].to_numpy()
            /
            100
        )


    aggregated_actual = (
        np.column_stack(
            aggregated_actual
        )
    )


    aggregated_predicted = (
        np.column_stack(
            aggregated_predicted
        )
    )


    aggregated_close = (
        prediction_df[
            "current_close"
        ]
        .to_numpy()
    )


    aggregate_metrics = (
        calculate_horizon_metrics(

            actual_returns=
                aggregated_actual,

            predicted_returns=
                aggregated_predicted,

            current_close=
                aggregated_close,
        )
    )


    print_metrics(
        "AGGREGATED WALK-FORWARD RESULTS",
        aggregate_metrics,
    )


    # =====================================================
    # SAVE RESULTS
    # =====================================================

    safe_symbol = (
        safe_symbol_name(
            symbol
        )
    )


    json_path = os.path.join(

        MODELS_DIR,

        f"{safe_symbol}_walk_forward.json",
    )


    csv_path = os.path.join(

        MODELS_DIR,

        f"{safe_symbol}_walk_forward_predictions.csv",
    )


    result_payload = {

        "symbol":
            symbol,

        "model":
            "V3-like Multi-Horizon BiLSTM",

        "evaluation":
            "Purged Walk-Forward Backtesting",

        "features_used":
            len(
                FEATURE_COLUMNS
            ),

        "lookback_days":
            SEQUENCE_LENGTH,

        "horizons":
            HORIZONS,

        "max_horizon_purge_days":
            MAX_HORIZON,

        "folds":
            all_fold_results,

        "aggregate_metrics":
            aggregate_metrics,
    }


    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result_payload,
            file,
            indent=4,
        )


    prediction_df.to_csv(
        csv_path,
        index=False,
    )


    print(
        "\n=============================================="
    )

    print(
        " WALK-FORWARD COMPLETE"
    )

    print(
        "=============================================="
    )


    print(
        "\nCompleted folds:",
        len(
            all_fold_results
        )
    )


    print(
        "\nMetrics:"
    )

    print(
        json_path
    )


    print(
        "\nPredictions:"
    )

    print(
        csv_path
    )


    print(
        "\nImportant:"
    )


    print(
        "These results come from historical out-of-sample "
        "walk-forward predictions."
    )


    print(
        "The model is retrained separately for each fold."
    )


    print(
        "Training/validation boundaries are purged by "
        f"{MAX_HORIZON} trading days to reduce target leakage."
    )


# =========================================================
# COMMAND LINE
# =========================================================

if __name__ == "__main__":

    symbol = (

        sys.argv[
            1
        ]

        if len(
            sys.argv
        ) >
        1

        else
        "RELIANCE.NS"
    )


    run_walk_forward(
        symbol,
        number_of_folds=5,
    )