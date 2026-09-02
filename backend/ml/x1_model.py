"""
StockVision Precision X1 — Context-Enriched Next-Day Forecast
=============================================================

This trains on the CSV produced by:

    python -m ml.x1_data_collector RELIANCE.NS

Input file:
    backend/ml/data/x1_RELIANCE_NS.csv

Target:
    TARGET_MOVE_ATR
        = (next_close - current_close) / current_ATR14

Prediction:
    predicted_rupee_move
        = predicted_move_atr * current_ATR14

    predicted_close
        = current_close + predicted_rupee_move

Models:
    - HistGradientBoostingRegressor (absolute error)
    - GradientBoostingRegressor (Huber)
    - Ridge regression

Safety:
    - validation-only ensemble weights
    - validation-only shrinkage toward zero move
    - previous-close baseline = zero normalized move
    - chronological holdout test
    - 4-fold expanding walk-forward evaluation
    - ACCEPTED only if it genuinely beats baseline

This code does NOT guarantee Rs 1-2 error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

MODEL_VERSION = "precision-x1"
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

MIN_VALIDATION_IMPROVEMENT_PCT = 0.50
MIN_WALK_FORWARD_FOLDS_BEATING = 3
WALK_FORWARD_FOLDS = 4

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "TARGET_MOVE_ATR"

NON_FEATURE_COLUMNS = {
    "DATE",
    "SYMBOL",
    "TARGET_NEXT_CLOSE",
    "TARGET_RESIDUAL_RS",
    "TARGET_RETURN_PCT",
    "TARGET_MOVE_ATR",
}

CORE_REQUIRED_COLUMNS = {
    "DATE",
    "SYMBOL",
    "BASE_CLOSE",
    "BASE_ATR14_RS",
    "TARGET_NEXT_CLOSE",
    "TARGET_RESIDUAL_RS",
    "TARGET_RETURN_PCT",
    "TARGET_MOVE_ATR",
}


# ============================================================
# PATHS
# ============================================================

def normalize_symbol(symbol: str) -> str:
    clean = str(symbol or "").strip().upper().replace(" ", "")

    if not clean:
        raise ValueError("Stock symbol is required.")

    if "." in clean:
        return clean

    return f"{clean}.NS"


def safe_name(symbol: str) -> str:
    return normalize_symbol(symbol).replace(".", "_").replace("-", "_")


def dataset_path(symbol: str) -> Path:
    return DATA_DIR / f"x1_{safe_name(symbol)}.csv"


def artifact_paths(symbol: str) -> dict[str, Path]:
    name = safe_name(symbol)

    return {
        "hist": MODEL_DIR / f"{name}_{MODEL_VERSION}_hist.pkl",
        "gbr": MODEL_DIR / f"{name}_{MODEL_VERSION}_gbr.pkl",
        "ridge": MODEL_DIR / f"{name}_{MODEL_VERSION}_ridge.pkl",
        "metadata": MODEL_DIR / f"{name}_{MODEL_VERSION}_metadata.json",
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(symbol: str) -> pd.DataFrame:
    path = dataset_path(symbol)

    if not path.exists():
        raise FileNotFoundError(
            f"X1 dataset not found: {path}\n"
            f"Run: python -m ml.x1_data_collector {normalize_symbol(symbol)}"
        )

    frame = pd.read_csv(
        path
    )

    missing = CORE_REQUIRED_COLUMNS.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    frame["DATE"] = pd.to_datetime(
        frame["DATE"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=["DATE"]
    ).sort_values(
        "DATE"
    ).reset_index(
        drop=True
    )

    numeric_columns = [
        column
        for column in frame.columns
        if column not in {"DATE", "SYMBOL"}
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Collector already returns training-ready rows, but this keeps the model
    # robust if the CSV was edited manually.
    all_nan_columns = [
        column
        for column in frame.columns
        if column not in {"DATE", "SYMBOL"}
        and frame[column].isna().all()
    ]

    if all_nan_columns:
        frame = frame.drop(
            columns=all_nan_columns
        )

    frame = frame.dropna().reset_index(
        drop=True
    )

    if len(frame) < 700:
        raise ValueError(
            f"Only {len(frame)} usable X1 rows."
        )

    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
        and column != "SYMBOL"
    ]

    # BASE_CLOSE and BASE_ATR14_RS are intentionally allowed.
    # They are known at forecast time and help the model understand scale/regime.
    return columns


# ============================================================
# MODELS
# ============================================================

def make_hist_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.03,
        max_iter=350,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=30,
        random_state=RANDOM_SEED,
    )


def make_gbr_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="huber",
        alpha=0.85,
        learning_rate=0.02,
        n_estimators=350,
        max_depth=2,
        min_samples_leaf=20,
        subsample=0.85,
        random_state=RANDOM_SEED,
    )


def make_ridge_model():
    return make_pipeline(
        StandardScaler(),
        Ridge(
            alpha=35.0,
            fit_intercept=True,
        ),
    )


def recency_weights(
    dates: pd.Series,
) -> np.ndarray:
    dates = pd.to_datetime(
        dates
    )

    minimum = dates.min()
    maximum = dates.max()

    span_days = max(
        1,
        (maximum - minimum).days,
    )

    age = (
        dates - minimum
    ).dt.days.to_numpy(
        dtype=float
    )

    scaled = (
        age / span_days
    )

    return (
        0.70
        + 0.60 * scaled
    )


def fit_models(
    frame: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    X = frame[
        features
    ]

    y = frame[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    target_cap = max(
        1.5,
        float(
            np.percentile(
                np.abs(y),
                99,
            )
        ),
    )

    y_fit = np.clip(
        y,
        -target_cap,
        target_cap,
    )

    weights = recency_weights(
        frame["DATE"]
    )

    hist = make_hist_model()

    hist.fit(
        X,
        y_fit,
        sample_weight=weights,
    )

    gbr = make_gbr_model()

    gbr.fit(
        X,
        y_fit,
        sample_weight=weights,
    )

    ridge = make_ridge_model()

    ridge.fit(
        X,
        y_fit,
        ridge__sample_weight=weights,
    )

    return {
        "hist": hist,
        "gbr": gbr,
        "ridge": ridge,
        "target_cap": target_cap,
    }


def component_predictions(
    models: dict[str, Any],
    X: pd.DataFrame,
) -> dict[str, np.ndarray]:
    return {
        "hist": np.asarray(
            models["hist"].predict(X),
            dtype=float,
        ),
        "gbr": np.asarray(
            models["gbr"].predict(X),
            dtype=float,
        ),
        "ridge": np.asarray(
            models["ridge"].predict(X),
            dtype=float,
        ),
    }


# ============================================================
# METRICS
# ============================================================

def evaluate(
    frame: pd.DataFrame,
    predicted_move_atr: np.ndarray,
) -> dict[str, float]:
    predicted_move_atr = np.asarray(
        predicted_move_atr,
        dtype=float,
    )

    base_close = frame[
        "BASE_CLOSE"
    ].to_numpy(
        dtype=float
    )

    atr = frame[
        "BASE_ATR14_RS"
    ].to_numpy(
        dtype=float
    )

    actual_residual = frame[
        "TARGET_RESIDUAL_RS"
    ].to_numpy(
        dtype=float
    )

    predicted_residual = (
        predicted_move_atr
        * atr
    )

    actual_close = (
        base_close
        + actual_residual
    )

    predicted_close = (
        base_close
        + predicted_residual
    )

    error = np.abs(
        actual_close
        - predicted_close
    )

    actual_return = (
        actual_residual
        / base_close
        * 100.0
    )

    predicted_return = (
        predicted_residual
        / base_close
        * 100.0
    )

    def direction(
        values: np.ndarray,
    ) -> np.ndarray:
        return np.where(
            values > 0.10,
            1,
            np.where(
                values < -0.10,
                -1,
                0,
            ),
        )

    return {
        "mae": float(
            np.mean(error)
        ),
        "median_absolute_error": float(
            np.median(error)
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        actual_close
                        - predicted_close
                    )
                )
            )
        ),
        "mape_percent": float(
            np.mean(
                error
                / actual_close
            )
            * 100.0
        ),
        "direction_accuracy_percent": float(
            np.mean(
                direction(
                    actual_return
                )
                ==
                direction(
                    predicted_return
                )
            )
            * 100.0
        ),
        "within_2_rupees_percent": float(
            np.mean(
                error <= 2.0
            )
            * 100.0
        ),
        "within_5_rupees_percent": float(
            np.mean(
                error <= 5.0
            )
            * 100.0
        ),
        "within_10_rupees_percent": float(
            np.mean(
                error <= 10.0
            )
            * 100.0
        ),
    }


# ============================================================
# VALIDATION CALIBRATION
# ============================================================

def apply_calibration(
    components: dict[str, np.ndarray],
    calibration: dict[str, Any],
) -> np.ndarray:
    raw = (
        float(
            calibration[
                "hist_weight"
            ]
        )
        * components[
            "hist"
        ]
        +
        float(
            calibration[
                "gbr_weight"
            ]
        )
        * components[
            "gbr"
        ]
        +
        float(
            calibration[
                "ridge_weight"
            ]
        )
        * components[
            "ridge"
        ]
    )

    prediction = (
        float(
            calibration[
                "shrinkage"
            ]
        )
        * raw
        +
        float(
            calibration[
                "bias_atr"
            ]
        )
    )

    cap = calibration.get(
        "cap_atr"
    )

    if cap is not None:
        prediction = np.clip(
            prediction,
            -float(cap),
            float(cap),
        )

    return np.asarray(
        prediction,
        dtype=float,
    )


def optimize_calibration(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    components: dict[str, np.ndarray],
) -> dict[str, Any]:
    baseline = np.zeros(
        len(
            val_frame
        ),
        dtype=float,
    )

    baseline_mae = evaluate(
        val_frame,
        baseline,
    )[
        "mae"
    ]

    y_train = train_frame[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    y_val = val_frame[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    cap_candidates: list[
        float | None
    ] = [
        None,
        float(
            np.percentile(
                np.abs(
                    y_train
                ),
                70,
            )
        ),
        float(
            np.percentile(
                np.abs(
                    y_train
                ),
                80,
            )
        ),
        float(
            np.percentile(
                np.abs(
                    y_train
                ),
                90,
            )
        ),
    ]

    best = {
        "hist_weight": 0.0,
        "gbr_weight": 0.0,
        "ridge_weight": 0.0,
        "baseline_weight": 1.0,
        "shrinkage": 0.0,
        "bias_atr": 0.0,
        "cap_atr": None,
        "validation_mae": float(
            baseline_mae
        ),
        "baseline_validation_mae": float(
            baseline_mae
        ),
        "validation_improvement_percent": 0.0,
        "accepted": False,
    }

    weights = np.arange(
        0.0,
        1.0001,
        0.10,
    )

    shrinkages = [
        0.25,
        0.50,
        0.75,
        1.00,
    ]

    for wh in weights:
        for wg in weights:
            for wr in weights:
                total = (
                    wh
                    + wg
                    + wr
                )

                if (
                    total <= 0.0
                    or total > 1.0001
                ):
                    continue

                raw = (
                    wh
                    * components["hist"]
                    +
                    wg
                    * components["gbr"]
                    +
                    wr
                    * components["ridge"]
                )

                for shrinkage in shrinkages:
                    shrunk = (
                        shrinkage
                        * raw
                    )

                    bias = float(
                        np.clip(
                            np.median(
                                y_val
                                - shrunk
                            ),
                            -0.20,
                            0.20,
                        )
                    )

                    for cap in cap_candidates:
                        candidate = (
                            shrunk
                            + bias
                        )

                        if cap is not None:
                            candidate = np.clip(
                                candidate,
                                -cap,
                                cap,
                            )

                        mae = evaluate(
                            val_frame,
                            candidate,
                        )[
                            "mae"
                        ]

                        if mae < best[
                            "validation_mae"
                        ]:
                            improvement = (
                                (
                                    baseline_mae
                                    - mae
                                )
                                / baseline_mae
                                * 100.0
                                if baseline_mae
                                > 0
                                else 0.0
                            )

                            best = {
                                "hist_weight": float(
                                    wh
                                ),
                                "gbr_weight": float(
                                    wg
                                ),
                                "ridge_weight": float(
                                    wr
                                ),
                                "baseline_weight": float(
                                    max(
                                        0.0,
                                        1.0
                                        - total,
                                    )
                                ),
                                "shrinkage": float(
                                    shrinkage
                                ),
                                "bias_atr": float(
                                    bias
                                ),
                                "cap_atr": (
                                    float(
                                        cap
                                    )
                                    if cap
                                    is not None
                                    else None
                                ),
                                "validation_mae": float(
                                    mae
                                ),
                                "baseline_validation_mae": float(
                                    baseline_mae
                                ),
                                "validation_improvement_percent": float(
                                    improvement
                                ),
                                "accepted": True,
                            }

    if (
        best[
            "validation_improvement_percent"
        ]
        < MIN_VALIDATION_IMPROVEMENT_PCT
    ):
        return {
            "hist_weight": 0.0,
            "gbr_weight": 0.0,
            "ridge_weight": 0.0,
            "baseline_weight": 1.0,
            "shrinkage": 0.0,
            "bias_atr": 0.0,
            "cap_atr": None,
            "validation_mae": float(
                baseline_mae
            ),
            "baseline_validation_mae": float(
                baseline_mae
            ),
            "validation_improvement_percent": 0.0,
            "accepted": False,
        }

    return best


# ============================================================
# SPLITS
# ============================================================

def chronological_split(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    n = len(
        frame
    )

    train_end = int(
        n
        * TRAIN_RATIO
    )

    val_end = int(
        n
        * (
            TRAIN_RATIO
            + VAL_RATIO
        )
    )

    train = frame.iloc[
        :train_end
    ].copy()

    val = frame.iloc[
        train_end:val_end
    ].copy()

    test = frame.iloc[
        val_end:
    ].copy()

    if min(
        len(
            train
        ),
        len(
            val
        ),
        len(
            test
        ),
    ) < 100:
        raise ValueError(
            "Not enough rows for chronological X1 split."
        )

    return (
        train,
        val,
        test,
    )


# ============================================================
# WALK-FORWARD
# ============================================================

def walk_forward_evaluate(
    frame: pd.DataFrame,
    features: list[str],
    folds: int = WALK_FORWARD_FOLDS,
) -> dict[str, Any]:
    n = len(
        frame
    )

    initial = int(
        n
        * 0.55
    )

    fold_size = max(
        80,
        (
            n
            - initial
        )
        // folds,
    )

    fold_results = []

    combined_frames = []
    combined_predictions = []

    for fold_index in range(
        folds
    ):
        test_start = (
            initial
            + fold_index
            * fold_size
        )

        if test_start >= n:
            break

        test_end = (
            n
            if fold_index
            == folds - 1
            else min(
                n,
                test_start
                + fold_size,
            )
        )

        if (
            test_end
            - test_start
            < 50
        ):
            continue

        development = frame.iloc[
            :test_start
        ].copy()

        test = frame.iloc[
            test_start:test_end
        ].copy()

        val_size = max(
            80,
            int(
                len(
                    development
                )
                * 0.15
            ),
        )

        train = development.iloc[
            :-val_size
        ].copy()

        val = development.iloc[
            -val_size:
        ].copy()

        if len(
            train
        ) < 300:
            continue

        models = fit_models(
            train,
            features,
        )

        val_components = component_predictions(
            models,
            val[
                features
            ],
        )

        calibration = optimize_calibration(
            train,
            val,
            val_components,
        )

        test_components = component_predictions(
            models,
            test[
                features
            ],
        )

        prediction = apply_calibration(
            test_components,
            calibration,
        )

        model_metrics = evaluate(
            test,
            prediction,
        )

        baseline_metrics = evaluate(
            test,
            np.zeros(
                len(
                    test
                ),
                dtype=float,
            ),
        )

        fold_results.append(
            {
                "fold": fold_index + 1,
                "test_start": str(
                    test[
                        "DATE"
                    ].iloc[
                        0
                    ].date()
                ),
                "test_end": str(
                    test[
                        "DATE"
                    ].iloc[
                        -1
                    ].date()
                ),
                "samples": int(
                    len(
                        test
                    )
                ),
                "model": model_metrics,
                "baseline": baseline_metrics,
                "beats_baseline": bool(
                    model_metrics[
                        "mae"
                    ]
                    < baseline_metrics[
                        "mae"
                    ]
                ),
                "calibration": calibration,
            }
        )

        combined_frames.append(
            test
        )

        combined_predictions.append(
            prediction
        )

    if not fold_results:
        raise ValueError(
            "No valid X1 walk-forward folds."
        )

    combined_frame = pd.concat(
        combined_frames,
        axis=0,
    )

    combined_prediction = np.concatenate(
        combined_predictions
    )

    model_metrics = evaluate(
        combined_frame,
        combined_prediction,
    )

    baseline_metrics = evaluate(
        combined_frame,
        np.zeros(
            len(
                combined_frame
            ),
            dtype=float,
        ),
    )

    improvement = (
        (
            baseline_metrics[
                "mae"
            ]
            - model_metrics[
                "mae"
            ]
        )
        / baseline_metrics[
            "mae"
        ]
        * 100.0
        if baseline_metrics[
            "mae"
        ] > 0
        else 0.0
    )

    return {
        "fold_count": int(
            len(
                fold_results
            )
        ),
        "samples": int(
            len(
                combined_frame
            )
        ),
        "folds_beating_baseline": int(
            sum(
                1
                for row
                in fold_results
                if row[
                    "beats_baseline"
                ]
            )
        ),
        "model": model_metrics,
        "baseline": baseline_metrics,
        "mae_improvement_vs_baseline_percent": float(
            improvement
        ),
        "folds": fold_results,
    }


# ============================================================
# ERROR BAND
# ============================================================

def validation_error_band(
    val_frame: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, float]:
    atr = val_frame[
        "BASE_ATR14_RS"
    ].to_numpy(
        dtype=float
    )

    predicted_residual = (
        np.asarray(
            prediction,
            dtype=float,
        )
        * atr
    )

    actual_residual = val_frame[
        "TARGET_RESIDUAL_RS"
    ].to_numpy(
        dtype=float
    )

    error = (
        actual_residual
        - predicted_residual
    )

    return {
        "q10_rs": float(
            np.quantile(
                error,
                0.10,
            )
        ),
        "q90_rs": float(
            np.quantile(
                error,
                0.90,
            )
        ),
    }


# ============================================================
# TRAIN
# ============================================================

def train_x1(
    symbol: str,
) -> dict[str, Any]:
    symbol = normalize_symbol(
        symbol
    )

    print(
        "\n"
        + "=" * 74
    )
    print(
        " STOCKVISION PRECISION X1"
    )
    print(
        " Context-Enriched Next-Day Forecast"
    )
    print(
        "=" * 74
    )
    print(
        f"Symbol: {symbol}"
    )

    frame = load_dataset(
        symbol
    )

    features = feature_columns(
        frame
    )

    print(
        f"Rows: {len(frame)} | "
        f"Features: {len(features)}"
    )

    print(
        f"Date range: "
        f"{frame['DATE'].min().date()} -> "
        f"{frame['DATE'].max().date()}"
    )

    print(
        "\nRunning expanding walk-forward evaluation..."
    )

    walk_forward = walk_forward_evaluate(
        frame,
        features,
    )

    train, val, test = chronological_split(
        frame
    )

    models = fit_models(
        train,
        features,
    )

    val_components = component_predictions(
        models,
        val[
            features
        ],
    )

    calibration = optimize_calibration(
        train,
        val,
        val_components,
    )

    val_prediction = apply_calibration(
        val_components,
        calibration,
    )

    band = validation_error_band(
        val,
        val_prediction,
    )

    test_components = component_predictions(
        models,
        test[
            features
        ],
    )

    test_prediction = apply_calibration(
        test_components,
        calibration,
    )

    model_metrics = evaluate(
        test,
        test_prediction,
    )

    baseline_metrics = evaluate(
        test,
        np.zeros(
            len(
                test
            ),
            dtype=float,
        ),
    )

    test_improvement = (
        (
            baseline_metrics[
                "mae"
            ]
            - model_metrics[
                "mae"
            ]
        )
        / baseline_metrics[
            "mae"
        ]
        * 100.0
        if baseline_metrics[
            "mae"
        ] > 0
        else 0.0
    )

    validation_accepted = bool(
        calibration[
            "accepted"
        ]
    )

    holdout_beats = (
        model_metrics[
            "mae"
        ]
        < baseline_metrics[
            "mae"
        ]
    )

    walk_forward_beats = (
        walk_forward[
            "model"
        ][
            "mae"
        ]
        < walk_forward[
            "baseline"
        ][
            "mae"
        ]
    )

    enough_folds = (
        walk_forward[
            "folds_beating_baseline"
        ]
        >= MIN_WALK_FORWARD_FOLDS_BEATING
    )

    close_hit_rates_not_worse = (
        model_metrics[
            "within_5_rupees_percent"
        ]
        >= baseline_metrics[
            "within_5_rupees_percent"
        ]
        and
        model_metrics[
            "within_10_rupees_percent"
        ]
        >= baseline_metrics[
            "within_10_rupees_percent"
        ]
    )

    accepted = (
        validation_accepted
        and holdout_beats
        and walk_forward_beats
        and enough_folds
        and close_hit_rates_not_worse
    )

    status = (
        "ACCEPTED"
        if accepted
        else "EXPERIMENTAL"
    )

    # Retrain production learners on train + validation after all evaluation.
    development = pd.concat(
        [
            train,
            val,
        ],
        axis=0,
    )

    production_models = fit_models(
        development,
        features,
    )

    p = artifact_paths(
        symbol
    )

    joblib.dump(
        production_models[
            "hist"
        ],
        p[
            "hist"
        ],
    )

    joblib.dump(
        production_models[
            "gbr"
        ],
        p[
            "gbr"
        ],
    )

    joblib.dump(
        production_models[
            "ridge"
        ],
        p[
            "ridge"
        ],
    )

    metadata = {
        "symbol": symbol,
        "model_version": MODEL_VERSION,
        "model_name": (
            "Precision X1 Context-Enriched Ensemble"
        ),
        "production_status": status,
        "features": features,
        "feature_count": len(
            features
        ),
        "calibration": calibration,
        "validation_error_band": band,
        "holdout_test": {
            "model": model_metrics,
            "baseline": baseline_metrics,
            "mae_improvement_vs_baseline_percent": float(
                test_improvement
            ),
            "beats_baseline": bool(
                holdout_beats
            ),
        },
        "walk_forward": walk_forward,
        "acceptance": {
            "validation_accepted": validation_accepted,
            "holdout_beats_baseline": bool(
                holdout_beats
            ),
            "walk_forward_beats_baseline": bool(
                walk_forward_beats
            ),
            "at_least_3_of_4_folds_beat_baseline": bool(
                enough_folds
            ),
            "within_5_and_10_not_worse": bool(
                close_hit_rates_not_worse
            ),
        },
        "date_ranges": {
            "train_start": str(
                train[
                    "DATE"
                ].iloc[
                    0
                ].date()
            ),
            "train_end": str(
                train[
                    "DATE"
                ].iloc[
                    -1
                ].date()
            ),
            "validation_start": str(
                val[
                    "DATE"
                ].iloc[
                    0
                ].date()
            ),
            "validation_end": str(
                val[
                    "DATE"
                ].iloc[
                    -1
                ].date()
            ),
            "test_start": str(
                test[
                    "DATE"
                ].iloc[
                    0
                ].date()
            ),
            "test_end": str(
                test[
                    "DATE"
                ].iloc[
                    -1
                ].date()
            ),
        },
    }

    p[
        "metadata"
    ].write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 74
    )
    print(
        " X1 HOLDOUT TEST"
    )
    print(
        "=" * 74
    )
    print(
        f"X1 MAE:                Rs {model_metrics['mae']:.2f}"
    )
    print(
        f"Baseline MAE:          Rs {baseline_metrics['mae']:.2f}"
    )
    print(
        f"MAE improvement:       {test_improvement:+.2f}%"
    )
    print(
        f"Median abs error:      Rs {model_metrics['median_absolute_error']:.2f}"
    )
    print(
        f"RMSE:                  Rs {model_metrics['rmse']:.2f}"
    )
    print(
        f"MAPE:                  {model_metrics['mape_percent']:.3f}%"
    )
    print(
        f"Direction accuracy:    {model_metrics['direction_accuracy_percent']:.2f}%"
    )
    print(
        f"Within Rs 2:           {model_metrics['within_2_rupees_percent']:.2f}% "
        f"(baseline {baseline_metrics['within_2_rupees_percent']:.2f}%)"
    )
    print(
        f"Within Rs 5:           {model_metrics['within_5_rupees_percent']:.2f}% "
        f"(baseline {baseline_metrics['within_5_rupees_percent']:.2f}%)"
    )
    print(
        f"Within Rs 10:          {model_metrics['within_10_rupees_percent']:.2f}% "
        f"(baseline {baseline_metrics['within_10_rupees_percent']:.2f}%)"
    )

    print(
        "\n"
        + "=" * 74
    )
    print(
        " X1 WALK-FORWARD"
    )
    print(
        "=" * 74
    )
    print(
        f"Folds beating baseline: "
        f"{walk_forward['folds_beating_baseline']}/"
        f"{walk_forward['fold_count']}"
    )
    print(
        f"Model MAE:             Rs {walk_forward['model']['mae']:.2f}"
    )
    print(
        f"Baseline MAE:          Rs {walk_forward['baseline']['mae']:.2f}"
    )
    print(
        f"Improvement:           "
        f"{walk_forward['mae_improvement_vs_baseline_percent']:+.2f}%"
    )

    print(
        "\nSelected validation calibration:"
    )
    print(
        f"Hist={calibration['hist_weight']:.2f} | "
        f"GBR={calibration['gbr_weight']:.2f} | "
        f"Ridge={calibration['ridge_weight']:.2f} | "
        f"Baseline={calibration['baseline_weight']:.2f}"
    )
    print(
        f"Shrinkage={calibration['shrinkage']:.2f} | "
        f"Bias={calibration['bias_atr']:+.4f} ATR | "
        f"Cap={calibration['cap_atr']}"
    )

    print(
        "\n"
        + "=" * 74
    )
    print(
        f"PRODUCTION STATUS: {status}"
    )
    print(
        "=" * 74
    )

    if status != "ACCEPTED":
        print(
            "X1 is saved for analysis, but do not replace the current "
            "production prediction endpoint with it yet."
        )

    return metadata
