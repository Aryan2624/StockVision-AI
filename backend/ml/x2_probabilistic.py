"""
StockVision Precision X2 — Probabilistic Next-Day Forecast
==========================================================

Purpose
-------
X2 is not another attempt to pretend that tomorrow's exact close is known.

It produces:
1. a central next-day forecast,
2. an empirically calibrated 80% price range,
3. explicit large-miss statistics:
   within Rs 5 / 10 / 15 / 20 / 30,
   over Rs 20 / 30 / 40.

Input
-----
Uses the leakage-aware X1 CSV already produced by:

    python -m ml.x1_data_collector RELIANCE.NS

Target
------
TARGET_MOVE_ATR =
    (next_close - current_close) / current_ATR14

Point models
------------
- HistGradientBoostingRegressor with absolute-error loss
- GradientBoostingRegressor with median/quantile loss (alpha=0.50)
- Ridge

Range models
------------
- GradientBoostingRegressor q10
- GradientBoostingRegressor q90
- validation calibration of the interval width

Important
---------
No model can guarantee that every future close is within Rs 20.
X2 measures how often that actually happens and how often misses exceed Rs 30/40.
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

MODEL_VERSION = "precision-x2"
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

TARGET_COVERAGE = 0.80

MIN_POINT_VALIDATION_IMPROVEMENT_PCT = 0.25
MIN_WALK_FORWARD_FOLDS_BEATING = 3

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
# PATHS / SYMBOL
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
        "median": MODEL_DIR / f"{name}_{MODEL_VERSION}_median.pkl",
        "ridge": MODEL_DIR / f"{name}_{MODEL_VERSION}_ridge.pkl",
        "q10": MODEL_DIR / f"{name}_{MODEL_VERSION}_q10.pkl",
        "q90": MODEL_DIR / f"{name}_{MODEL_VERSION}_q90.pkl",
        "metadata": MODEL_DIR / f"{name}_{MODEL_VERSION}_metadata.json",
    }


# ============================================================
# DATA
# ============================================================

def load_dataset(symbol: str) -> pd.DataFrame:
    path = dataset_path(symbol)

    if not path.exists():
        raise FileNotFoundError(
            f"X1 dataset not found: {path}\n"
            f"Run: python -m ml.x1_data_collector {normalize_symbol(symbol)}"
        )

    frame = pd.read_csv(path)

    missing = CORE_REQUIRED_COLUMNS.difference(frame.columns)

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

    for column in frame.columns:
        if column not in {"DATE", "SYMBOL"}:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    )

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

    frame = frame.dropna().reset_index(drop=True)

    if len(frame) < 700:
        raise ValueError(
            f"Only {len(frame)} usable rows."
        )

    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
        and column != "SYMBOL"
    ]


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


def make_median_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=0.50,
        learning_rate=0.025,
        n_estimators=350,
        max_depth=2,
        min_samples_leaf=20,
        subsample=0.85,
        random_state=RANDOM_SEED,
    )


def make_quantile_model(alpha: float) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        learning_rate=0.025,
        n_estimators=400,
        max_depth=2,
        min_samples_leaf=20,
        subsample=0.90,
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


def recency_weights(dates: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(dates)

    minimum = dates.min()
    maximum = dates.max()

    total_days = max(
        1,
        (maximum - minimum).days,
    )

    age = (
        dates - minimum
    ).dt.days.to_numpy(dtype=float)

    scaled = age / total_days

    return 0.70 + 0.60 * scaled


def fit_models(
    frame: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    X = frame[features]

    y = frame[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

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

    median = make_median_model()
    median.fit(
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

    q10 = make_quantile_model(0.10)
    q10.fit(
        X,
        y_fit,
        sample_weight=weights,
    )

    q90 = make_quantile_model(0.90)
    q90.fit(
        X,
        y_fit,
        sample_weight=weights,
    )

    return {
        "hist": hist,
        "median": median,
        "ridge": ridge,
        "q10": q10,
        "q90": q90,
        "target_cap": target_cap,
    }


def point_components(
    models: dict[str, Any],
    X: pd.DataFrame,
) -> dict[str, np.ndarray]:
    return {
        "hist": np.asarray(
            models["hist"].predict(X),
            dtype=float,
        ),
        "median": np.asarray(
            models["median"].predict(X),
            dtype=float,
        ),
        "ridge": np.asarray(
            models["ridge"].predict(X),
            dtype=float,
        ),
    }


def raw_quantiles(
    models: dict[str, Any],
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(
        models["q10"].predict(X),
        dtype=float,
    )

    upper = np.asarray(
        models["q90"].predict(X),
        dtype=float,
    )

    # Quantile crossing can happen. Fix ordering without inventing values.
    corrected_lower = np.minimum(lower, upper)
    corrected_upper = np.maximum(lower, upper)

    return (
        corrected_lower,
        corrected_upper,
    )


# ============================================================
# POINT METRICS
# ============================================================

def point_metrics(
    frame: pd.DataFrame,
    predicted_move_atr: np.ndarray,
) -> dict[str, float]:
    predicted_move_atr = np.asarray(
        predicted_move_atr,
        dtype=float,
    )

    base = frame["BASE_CLOSE"].to_numpy(dtype=float)
    atr = frame["BASE_ATR14_RS"].to_numpy(dtype=float)

    actual_residual = frame[
        "TARGET_RESIDUAL_RS"
    ].to_numpy(dtype=float)

    predicted_residual = (
        predicted_move_atr
        * atr
    )

    actual_close = base + actual_residual
    predicted_close = base + predicted_residual

    error = np.abs(
        actual_close
        - predicted_close
    )

    def pct(mask: np.ndarray) -> float:
        return float(
            np.mean(mask)
            * 100.0
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
        "within_5_rupees_percent": pct(error <= 5.0),
        "within_10_rupees_percent": pct(error <= 10.0),
        "within_15_rupees_percent": pct(error <= 15.0),
        "within_20_rupees_percent": pct(error <= 20.0),
        "within_30_rupees_percent": pct(error <= 30.0),
        "over_20_rupees_percent": pct(error > 20.0),
        "over_30_rupees_percent": pct(error > 30.0),
        "over_40_rupees_percent": pct(error > 40.0),
        "p90_absolute_error_rupees": float(
            np.quantile(
                error,
                0.90,
            )
        ),
        "p95_absolute_error_rupees": float(
            np.quantile(
                error,
                0.95,
            )
        ),
        "max_absolute_error_rupees": float(
            np.max(error)
        ),
    }


def tail_objective(
    frame: pd.DataFrame,
    predicted_move_atr: np.ndarray,
) -> float:
    """
    Tail-aware validation objective.

    It still prioritizes MAE but gives extra cost to large misses over Rs 20
    and especially over Rs 30.
    """
    predicted_move_atr = np.asarray(
        predicted_move_atr,
        dtype=float,
    )

    base = frame["BASE_CLOSE"].to_numpy(dtype=float)
    atr = frame["BASE_ATR14_RS"].to_numpy(dtype=float)

    actual_residual = frame[
        "TARGET_RESIDUAL_RS"
    ].to_numpy(dtype=float)

    predicted_close = (
        base
        + predicted_move_atr
        * atr
    )

    actual_close = (
        base
        + actual_residual
    )

    error = np.abs(
        actual_close
        - predicted_close
    )

    mae = np.mean(error)

    penalty_20 = np.mean(
        np.maximum(
            error - 20.0,
            0.0,
        )
    )

    penalty_30 = np.mean(
        np.maximum(
            error - 30.0,
            0.0,
        )
    )

    return float(
        mae
        + 0.20 * penalty_20
        + 0.35 * penalty_30
    )


# ============================================================
# POINT CALIBRATION
# ============================================================

def apply_point_calibration(
    components: dict[str, np.ndarray],
    calibration: dict[str, Any],
) -> np.ndarray:
    raw = (
        float(calibration["hist_weight"])
        * components["hist"]
        +
        float(calibration["median_weight"])
        * components["median"]
        +
        float(calibration["ridge_weight"])
        * components["ridge"]
    )

    prediction = (
        float(calibration["shrinkage"])
        * raw
        +
        float(calibration["bias_atr"])
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


def optimize_point_calibration(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    components: dict[str, np.ndarray],
) -> dict[str, Any]:
    baseline = np.zeros(
        len(val_frame),
        dtype=float,
    )

    baseline_objective = tail_objective(
        val_frame,
        baseline,
    )

    baseline_mae = point_metrics(
        val_frame,
        baseline,
    )["mae"]

    y_train = train_frame[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    y_val = val_frame[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    cap_candidates: list[float | None] = [
        None,
        float(
            np.percentile(
                np.abs(y_train),
                70,
            )
        ),
        float(
            np.percentile(
                np.abs(y_train),
                80,
            )
        ),
        float(
            np.percentile(
                np.abs(y_train),
                90,
            )
        ),
    ]

    best = {
        "hist_weight": 0.0,
        "median_weight": 0.0,
        "ridge_weight": 0.0,
        "baseline_weight": 1.0,
        "shrinkage": 0.0,
        "bias_atr": 0.0,
        "cap_atr": None,
        "validation_tail_objective": float(
            baseline_objective
        ),
        "baseline_validation_tail_objective": float(
            baseline_objective
        ),
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
        for wm in weights:
            for wr in weights:
                total = wh + wm + wr

                if (
                    total <= 0.0
                    or total > 1.0001
                ):
                    continue

                raw = (
                    wh * components["hist"]
                    +
                    wm * components["median"]
                    +
                    wr * components["ridge"]
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

                        objective = tail_objective(
                            val_frame,
                            candidate,
                        )

                        if objective < best[
                            "validation_tail_objective"
                        ]:
                            mae = point_metrics(
                                val_frame,
                                candidate,
                            )["mae"]

                            improvement = (
                                (
                                    baseline_objective
                                    - objective
                                )
                                / baseline_objective
                                * 100.0
                                if baseline_objective > 0
                                else 0.0
                            )

                            best = {
                                "hist_weight": float(wh),
                                "median_weight": float(wm),
                                "ridge_weight": float(wr),
                                "baseline_weight": float(
                                    max(
                                        0.0,
                                        1.0 - total,
                                    )
                                ),
                                "shrinkage": float(
                                    shrinkage
                                ),
                                "bias_atr": float(
                                    bias
                                ),
                                "cap_atr": (
                                    float(cap)
                                    if cap is not None
                                    else None
                                ),
                                "validation_tail_objective": float(
                                    objective
                                ),
                                "baseline_validation_tail_objective": float(
                                    baseline_objective
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
        best["validation_improvement_percent"]
        < MIN_POINT_VALIDATION_IMPROVEMENT_PCT
    ):
        return {
            "hist_weight": 0.0,
            "median_weight": 0.0,
            "ridge_weight": 0.0,
            "baseline_weight": 1.0,
            "shrinkage": 0.0,
            "bias_atr": 0.0,
            "cap_atr": None,
            "validation_tail_objective": float(
                baseline_objective
            ),
            "baseline_validation_tail_objective": float(
                baseline_objective
            ),
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
# INTERVAL CALIBRATION / METRICS
# ============================================================

def calibrate_interval(
    val_frame: pd.DataFrame,
    raw_lower_atr: np.ndarray,
    raw_upper_atr: np.ndarray,
) -> dict[str, float]:
    y = val_frame[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    lower = np.asarray(
        raw_lower_atr,
        dtype=float,
    )

    upper = np.asarray(
        raw_upper_atr,
        dtype=float,
    )

    scores = np.maximum(
        lower - y,
        y - upper,
    )

    scores = np.maximum(
        scores,
        0.0,
    )

    # Empirical validation calibration. We deliberately avoid claiming a
    # mathematical time-series coverage guarantee.
    adjustment = float(
        np.quantile(
            scores,
            TARGET_COVERAGE,
        )
    )

    return {
        "target_coverage": TARGET_COVERAGE,
        "adjustment_atr": adjustment,
    }


def apply_interval_calibration(
    raw_lower_atr: np.ndarray,
    raw_upper_atr: np.ndarray,
    calibration: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    adjustment = float(
        calibration[
            "adjustment_atr"
        ]
    )

    lower = np.asarray(
        raw_lower_atr,
        dtype=float,
    ) - adjustment

    upper = np.asarray(
        raw_upper_atr,
        dtype=float,
    ) + adjustment

    corrected_lower = np.minimum(
        lower,
        upper,
    )

    corrected_upper = np.maximum(
        lower,
        upper,
    )

    return (
        corrected_lower,
        corrected_upper,
    )


def interval_metrics(
    frame: pd.DataFrame,
    lower_atr: np.ndarray,
    upper_atr: np.ndarray,
) -> dict[str, float]:
    lower_atr = np.asarray(
        lower_atr,
        dtype=float,
    )

    upper_atr = np.asarray(
        upper_atr,
        dtype=float,
    )

    base = frame["BASE_CLOSE"].to_numpy(dtype=float)
    atr = frame["BASE_ATR14_RS"].to_numpy(dtype=float)

    actual_close = frame[
        "TARGET_NEXT_CLOSE"
    ].to_numpy(dtype=float)

    lower_price = (
        base
        + lower_atr * atr
    )

    upper_price = (
        base
        + upper_atr * atr
    )

    lower_price, upper_price = (
        np.minimum(
            lower_price,
            upper_price,
        ),
        np.maximum(
            lower_price,
            upper_price,
        ),
    )

    covered = (
        (actual_close >= lower_price)
        & (actual_close <= upper_price)
    )

    width = (
        upper_price
        - lower_price
    )

    return {
        "coverage_percent": float(
            np.mean(covered)
            * 100.0
        ),
        "average_width_rupees": float(
            np.mean(width)
        ),
        "median_width_rupees": float(
            np.median(width)
        ),
        "p90_width_rupees": float(
            np.quantile(
                width,
                0.90,
            )
        ),
    }


# ============================================================
# SPLIT
# ============================================================

def chronological_split(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    n = len(frame)

    train_end = int(
        n * TRAIN_RATIO
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

    return (
        train,
        val,
        test,
    )


# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward_evaluate(
    frame: pd.DataFrame,
    features: list[str],
    folds: int = 4,
) -> dict[str, Any]:
    n = len(frame)

    initial = int(
        n * 0.55
    )

    fold_size = max(
        80,
        (
            n - initial
        )
        // folds,
    )

    fold_results = []

    combined_frames = []
    combined_point = []
    combined_lower = []
    combined_upper = []

    for fold_index in range(folds):
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
                len(development)
                * 0.15
            ),
        )

        train = development.iloc[
            :-val_size
        ].copy()

        val = development.iloc[
            -val_size:
        ].copy()

        if len(train) < 300:
            continue

        models = fit_models(
            train,
            features,
        )

        val_components = point_components(
            models,
            val[features],
        )

        point_calibration = optimize_point_calibration(
            train,
            val,
            val_components,
        )

        val_q10, val_q90 = raw_quantiles(
            models,
            val[features],
        )

        range_calibration = calibrate_interval(
            val,
            val_q10,
            val_q90,
        )

        test_components = point_components(
            models,
            test[features],
        )

        point_prediction = apply_point_calibration(
            test_components,
            point_calibration,
        )

        test_q10, test_q90 = raw_quantiles(
            models,
            test[features],
        )

        lower, upper = apply_interval_calibration(
            test_q10,
            test_q90,
            range_calibration,
        )

        model_metrics = point_metrics(
            test,
            point_prediction,
        )

        baseline_metrics = point_metrics(
            test,
            np.zeros(
                len(test),
                dtype=float,
            ),
        )

        range_metrics = interval_metrics(
            test,
            lower,
            upper,
        )

        fold_results.append(
            {
                "fold": fold_index + 1,
                "test_start": str(
                    test["DATE"].iloc[0].date()
                ),
                "test_end": str(
                    test["DATE"].iloc[-1].date()
                ),
                "samples": int(
                    len(test)
                ),
                "point_model": model_metrics,
                "baseline": baseline_metrics,
                "range": range_metrics,
                "point_beats_baseline": bool(
                    model_metrics["mae"]
                    < baseline_metrics["mae"]
                ),
                "point_calibration": point_calibration,
                "range_calibration": range_calibration,
            }
        )

        combined_frames.append(test)
        combined_point.append(point_prediction)
        combined_lower.append(lower)
        combined_upper.append(upper)

    if not fold_results:
        raise ValueError(
            "No valid X2 walk-forward folds."
        )

    combined_frame = pd.concat(
        combined_frames,
        axis=0,
    )

    combined_point_prediction = np.concatenate(
        combined_point
    )

    combined_lower_prediction = np.concatenate(
        combined_lower
    )

    combined_upper_prediction = np.concatenate(
        combined_upper
    )

    model_metrics = point_metrics(
        combined_frame,
        combined_point_prediction,
    )

    baseline_metrics = point_metrics(
        combined_frame,
        np.zeros(
            len(combined_frame),
            dtype=float,
        ),
    )

    range_metrics = interval_metrics(
        combined_frame,
        combined_lower_prediction,
        combined_upper_prediction,
    )

    improvement = (
        (
            baseline_metrics["mae"]
            - model_metrics["mae"]
        )
        / baseline_metrics["mae"]
        * 100.0
        if baseline_metrics["mae"] > 0
        else 0.0
    )

    return {
        "fold_count": int(
            len(fold_results)
        ),
        "samples": int(
            len(combined_frame)
        ),
        "folds_beating_baseline": int(
            sum(
                1
                for row
                in fold_results
                if row[
                    "point_beats_baseline"
                ]
            )
        ),
        "point_model": model_metrics,
        "baseline": baseline_metrics,
        "range": range_metrics,
        "mae_improvement_vs_baseline_percent": float(
            improvement
        ),
        "folds": fold_results,
    }


# ============================================================
# TRAIN
# ============================================================

def train_x2(symbol: str) -> dict[str, Any]:
    symbol = normalize_symbol(symbol)

    print(
        "\n"
        + "=" * 76
    )
    print(
        " STOCKVISION PRECISION X2"
    )
    print(
        " Probabilistic Next-Day Forecast + Large-Miss Analysis"
    )
    print(
        "=" * 76
    )

    frame = load_dataset(symbol)
    features = feature_columns(frame)

    print(
        f"Symbol: {symbol}"
    )
    print(
        f"Rows: {len(frame)} | Features: {len(features)}"
    )
    print(
        f"Date range: "
        f"{frame['DATE'].min().date()} -> "
        f"{frame['DATE'].max().date()}"
    )

    print(
        "\nRunning probabilistic walk-forward evaluation..."
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

    val_components = point_components(
        models,
        val[features],
    )

    point_calibration = optimize_point_calibration(
        train,
        val,
        val_components,
    )

    val_q10, val_q90 = raw_quantiles(
        models,
        val[features],
    )

    range_calibration = calibrate_interval(
        val,
        val_q10,
        val_q90,
    )

    test_components = point_components(
        models,
        test[features],
    )

    point_prediction = apply_point_calibration(
        test_components,
        point_calibration,
    )

    test_q10, test_q90 = raw_quantiles(
        models,
        test[features],
    )

    lower, upper = apply_interval_calibration(
        test_q10,
        test_q90,
        range_calibration,
    )

    model_metrics = point_metrics(
        test,
        point_prediction,
    )

    baseline_metrics = point_metrics(
        test,
        np.zeros(
            len(test),
            dtype=float,
        ),
    )

    range_metrics = interval_metrics(
        test,
        lower,
        upper,
    )

    test_improvement = (
        (
            baseline_metrics["mae"]
            - model_metrics["mae"]
        )
        / baseline_metrics["mae"]
        * 100.0
        if baseline_metrics["mae"] > 0
        else 0.0
    )

    point_beats_baseline = (
        model_metrics["mae"]
        < baseline_metrics["mae"]
    )

    within20_not_worse = (
        model_metrics[
            "within_20_rupees_percent"
        ]
        >= baseline_metrics[
            "within_20_rupees_percent"
        ]
    )

    over30_not_worse = (
        model_metrics[
            "over_30_rupees_percent"
        ]
        <= baseline_metrics[
            "over_30_rupees_percent"
        ]
    )

    walk_forward_point_beats = (
        walk_forward[
            "point_model"
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

    holdout_range_ok = (
        72.0
        <= range_metrics[
            "coverage_percent"
        ]
        <= 92.0
    )

    walk_range_ok = (
        70.0
        <= walk_forward[
            "range"
        ][
            "coverage_percent"
        ]
        <= 92.0
    )

    if (
        point_calibration["accepted"]
        and point_beats_baseline
        and within20_not_worse
        and over30_not_worse
        and walk_forward_point_beats
        and enough_folds
        and holdout_range_ok
        and walk_range_ok
    ):
        status = "ACCEPTED_POINT_AND_RANGE"

    elif (
        not point_calibration["accepted"]
        and holdout_range_ok
        and walk_range_ok
    ):
        status = "ACCEPTED_RANGE_ONLY"

    else:
        status = "EXPERIMENTAL"

    # Production learners use train + validation only after evaluation.
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

    paths = artifact_paths(
        symbol
    )

    for key in (
        "hist",
        "median",
        "ridge",
        "q10",
        "q90",
    ):
        joblib.dump(
            production_models[key],
            paths[key],
        )

    metadata = {
        "symbol": symbol,
        "model_version": MODEL_VERSION,
        "model_name": (
            "Precision X2 Probabilistic Forecast"
        ),
        "production_status": status,
        "features": features,
        "feature_count": len(features),
        "point_calibration": point_calibration,
        "range_calibration": range_calibration,
        "holdout": {
            "point_model": model_metrics,
            "baseline": baseline_metrics,
            "range": range_metrics,
            "mae_improvement_vs_baseline_percent": float(
                test_improvement
            ),
        },
        "walk_forward": walk_forward,
        "acceptance": {
            "point_validation_accepted": bool(
                point_calibration["accepted"]
            ),
            "point_holdout_beats_baseline": bool(
                point_beats_baseline
            ),
            "within_20_not_worse_than_baseline": bool(
                within20_not_worse
            ),
            "over_30_not_worse_than_baseline": bool(
                over30_not_worse
            ),
            "walk_forward_point_beats_baseline": bool(
                walk_forward_point_beats
            ),
            "at_least_3_of_4_point_folds_beat_baseline": bool(
                enough_folds
            ),
            "holdout_range_coverage_reasonable": bool(
                holdout_range_ok
            ),
            "walk_forward_range_coverage_reasonable": bool(
                walk_range_ok
            ),
        },
        "notes": [
            "X2 cannot guarantee every prediction within Rs 20.",
            "The central prediction is selected with a tail-aware validation objective.",
            "The range is an empirical validation-calibrated 80% interval.",
            "Because this is time-series data, range coverage is measured empirically rather than claimed as a mathematical guarantee.",
        ],
    }

    paths[
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
        + "=" * 76
    )
    print(
        " X2 POINT FORECAST — HOLDOUT"
    )
    print(
        "=" * 76
    )

    print(
        f"X2 MAE:                Rs {model_metrics['mae']:.2f}"
    )
    print(
        f"Baseline MAE:          Rs {baseline_metrics['mae']:.2f}"
    )
    print(
        f"MAE improvement:       {test_improvement:+.2f}%"
    )
    print(
        f"Median error:          Rs {model_metrics['median_absolute_error']:.2f}"
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
        f"Within Rs 15:          {model_metrics['within_15_rupees_percent']:.2f}% "
        f"(baseline {baseline_metrics['within_15_rupees_percent']:.2f}%)"
    )
    print(
        f"Within Rs 20:          {model_metrics['within_20_rupees_percent']:.2f}% "
        f"(baseline {baseline_metrics['within_20_rupees_percent']:.2f}%)"
    )
    print(
        f"Within Rs 30:          {model_metrics['within_30_rupees_percent']:.2f}% "
        f"(baseline {baseline_metrics['within_30_rupees_percent']:.2f}%)"
    )
    print(
        f"Over Rs 20:            {model_metrics['over_20_rupees_percent']:.2f}% "
        f"(baseline {baseline_metrics['over_20_rupees_percent']:.2f}%)"
    )
    print(
        f"Over Rs 30:            {model_metrics['over_30_rupees_percent']:.2f}% "
        f"(baseline {baseline_metrics['over_30_rupees_percent']:.2f}%)"
    )
    print(
        f"Over Rs 40:            {model_metrics['over_40_rupees_percent']:.2f}% "
        f"(baseline {baseline_metrics['over_40_rupees_percent']:.2f}%)"
    )
    print(
        f"90th percentile error: Rs {model_metrics['p90_absolute_error_rupees']:.2f}"
    )
    print(
        f"95th percentile error: Rs {model_metrics['p95_absolute_error_rupees']:.2f}"
    )

    print(
        "\n"
        + "=" * 76
    )
    print(
        " X2 80% RANGE — HOLDOUT"
    )
    print(
        "=" * 76
    )

    print(
        f"Observed coverage:     {range_metrics['coverage_percent']:.2f}%"
    )
    print(
        f"Average range width:   Rs {range_metrics['average_width_rupees']:.2f}"
    )
    print(
        f"Median range width:    Rs {range_metrics['median_width_rupees']:.2f}"
    )
    print(
        f"90th pct range width:  Rs {range_metrics['p90_width_rupees']:.2f}"
    )

    print(
        "\n"
        + "=" * 76
    )
    print(
        " X2 WALK-FORWARD"
    )
    print(
        "=" * 76
    )

    print(
        f"Point folds beating baseline: "
        f"{walk_forward['folds_beating_baseline']}/"
        f"{walk_forward['fold_count']}"
    )
    print(
        f"Point model MAE:       Rs {walk_forward['point_model']['mae']:.2f}"
    )
    print(
        f"Baseline MAE:          Rs {walk_forward['baseline']['mae']:.2f}"
    )
    print(
        f"Point improvement:     "
        f"{walk_forward['mae_improvement_vs_baseline_percent']:+.2f}%"
    )
    print(
        f"Range coverage:        {walk_forward['range']['coverage_percent']:.2f}%"
    )
    print(
        f"Avg range width:       Rs {walk_forward['range']['average_width_rupees']:.2f}"
    )

    print(
        "\nSelected point calibration:"
    )
    print(
        f"Hist={point_calibration['hist_weight']:.2f} | "
        f"Median={point_calibration['median_weight']:.2f} | "
        f"Ridge={point_calibration['ridge_weight']:.2f} | "
        f"Baseline={point_calibration['baseline_weight']:.2f}"
    )
    print(
        f"Shrinkage={point_calibration['shrinkage']:.2f} | "
        f"Bias={point_calibration['bias_atr']:+.4f} ATR | "
        f"Cap={point_calibration['cap_atr']}"
    )

    print(
        "\n"
        + "=" * 76
    )
    print(
        f"PRODUCTION STATUS: {status}"
    )
    print(
        "=" * 76
    )

    if status == "ACCEPTED_RANGE_ONLY":
        print(
            "The learned exact-price point forecast was not accepted, but "
            "the empirical uncertainty range passed the configured range checks."
        )

    elif status == "EXPERIMENTAL":
        print(
            "Do not integrate X2 yet. Review the point-error tail and range "
            "coverage before changing the production endpoint."
        )

    return metadata
