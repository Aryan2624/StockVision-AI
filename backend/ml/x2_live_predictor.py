"""
StockVision X2 Live Hybrid Predictor
====================================

Production policy
-----------------
- X2 exact point forecast is used as the primary predicted_price ONLY when
  X2 has status ACCEPTED_POINT_AND_RANGE.
- If the exact point model is not accepted but the historical X2 range is
  healthy, predicted_price remains baseline-safe (current close), while the
  X2 experimental point and 80% expected range are returned separately.
- This keeps the API honest while still exposing the useful uncertainty layer.

Requires:
    ml/x1_data_collector.py
    ml/x2_probabilistic.py
    trained X2 artifacts in ml/models/
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .x1_data_collector import (
    ENERGY_PEER_BASKET,
    ENERGY_STOCKS,
    GLOBAL_CONTEXT,
    build_equal_weight_peer_index,
    download_daily,
    make_context_features,
    make_stock_features,
    normalize_symbol,
    optional_download,
    sector_proxy_for,
    short_symbol,
)

from .x2_probabilistic import (
    apply_interval_calibration,
    apply_point_calibration,
    artifact_paths,
    point_components,
    raw_quantiles,
)


def _load_x2(symbol: str) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = artifact_paths(symbol)

    required = [
        "hist",
        "median",
        "ridge",
        "q10",
        "q90",
        "metadata",
    ]

    missing = [
        key
        for key in required
        if not paths[key].exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"X2 is not trained for {normalize_symbol(symbol)}. "
            f"Missing artifacts: {', '.join(missing)}"
        )

    metadata = json.loads(
        paths["metadata"].read_text(
            encoding="utf-8"
        )
    )

    models = {
        "hist": joblib.load(paths["hist"]),
        "median": joblib.load(paths["median"]),
        "ridge": joblib.load(paths["ridge"]),
        "q10": joblib.load(paths["q10"]),
        "q90": joblib.load(paths["q90"]),
    }

    return metadata, models


def _build_live_feature_row(
    symbol: str,
    required_features: list[str],
) -> tuple[pd.DataFrame, float, float, pd.Timestamp]:
    """
    Rebuild exactly the same X1 feature family for the latest available NSE
    trading day, but WITHOUT requiring a next-day target.
    """
    symbol = normalize_symbol(symbol)

    stock = download_daily(
        symbol,
        period="2y",
        minimum_rows=120,
    )

    nifty = download_daily(
        "^NSEI",
        period="2y",
        minimum_rows=120,
    )

    sector_symbol = sector_proxy_for(symbol)

    if sector_symbol == "^NSEI":
        sector = nifty

    else:
        sector = optional_download(
            sector_symbol,
            "LIVE_SECTOR",
            "2y",
        )

        if sector is None:
            if short_symbol(symbol) in ENERGY_STOCKS:
                sector = build_equal_weight_peer_index(
                    ENERGY_PEER_BASKET,
                    period="2y",
                    minimum_members=3,
                )

            if sector is None:
                sector = nifty

    india_vix = optional_download(
        "^INDIAVIX",
        "LIVE_INDIA_VIX",
        "2y",
    )

    foreign: dict[
        str,
        pd.DataFrame | None
    ] = {}

    for label, ticker in GLOBAL_CONTEXT.items():
        foreign[label] = optional_download(
            ticker,
            f"LIVE_{label}",
            "2y",
        )

    stock_features = make_stock_features(
        stock
    )

    context_features = make_context_features(
        stock.index,
        nifty,
        sector,
        india_vix,
        foreign,
    )

    frame = pd.concat(
        [
            stock_features,
            context_features,
        ],
        axis=1,
    )

    frame[
        "STOCK_MINUS_NIFTY_RET1"
    ] = (
        frame[
            "STOCK_RET_1D"
        ]
        - frame[
            "NIFTY_RET_1D"
        ]
    )

    frame[
        "STOCK_MINUS_NIFTY_RET5"
    ] = (
        frame[
            "STOCK_RET_5D"
        ]
        - frame[
            "NIFTY_RET_5D"
        ]
    )

    frame[
        "STOCK_MINUS_SECTOR_RET1"
    ] = (
        frame[
            "STOCK_RET_1D"
        ]
        - frame[
            "SECTOR_RET_1D"
        ]
    )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    missing_columns = [
        column
        for column in required_features
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            "Live X2 features are missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    ready = frame.dropna(
        subset=required_features
    )

    if ready.empty:
        raise ValueError(
            "No latest row contains all required X2 features."
        )

    latest_date = pd.Timestamp(
        ready.index[-1]
    )

    row = ready.loc[
        [ready.index[-1]],
        required_features,
    ].copy()

    current_close = float(
        ready.loc[
            ready.index[-1],
            "BASE_CLOSE",
        ]
    )

    current_atr = float(
        ready.loc[
            ready.index[-1],
            "BASE_ATR14_RS",
        ]
    )

    return (
        row,
        current_close,
        current_atr,
        latest_date,
    )


def _range_is_empirically_healthy(
    metadata: dict[str, Any],
) -> bool:
    holdout = (
        metadata.get(
            "holdout",
            {}
        )
        .get(
            "range",
            {}
        )
    )

    walk = (
        metadata.get(
            "walk_forward",
            {}
        )
        .get(
            "range",
            {}
        )
    )

    try:
        holdout_coverage = float(
            holdout.get(
                "coverage_percent"
            )
        )

        walk_coverage = float(
            walk.get(
                "coverage_percent"
            )
        )

    except (TypeError, ValueError):
        return False

    return (
        72.0
        <= holdout_coverage
        <= 92.0
        and
        70.0
        <= walk_coverage
        <= 92.0
    )


def predict_x2_hybrid(
    symbol: str,
) -> dict[str, Any]:
    symbol = normalize_symbol(symbol)

    metadata, models = _load_x2(
        symbol
    )

    features = metadata[
        "features"
    ]

    (
        X,
        current_close,
        current_atr,
        latest_date,
    ) = _build_live_feature_row(
        symbol,
        features,
    )

    components = point_components(
        models,
        X,
    )

    point_calibration = metadata[
        "point_calibration"
    ]

    x2_point_atr = float(
        apply_point_calibration(
            components,
            point_calibration,
        )[0]
    )

    experimental_move_rs = (
        x2_point_atr
        * current_atr
    )

    experimental_point_price = (
        current_close
        + experimental_move_rs
    )

    raw_lower, raw_upper = raw_quantiles(
        models,
        X,
    )

    lower_atr, upper_atr = apply_interval_calibration(
        raw_lower,
        raw_upper,
        metadata[
            "range_calibration"
        ],
    )

    lower_price = float(
        current_close
        + lower_atr[0]
        * current_atr
    )

    upper_price = float(
        current_close
        + upper_atr[0]
        * current_atr
    )

    expected_lower = min(
        lower_price,
        upper_price,
    )

    expected_upper = max(
        lower_price,
        upper_price,
    )

    training_status = str(
        metadata.get(
            "production_status",
            "EXPERIMENTAL",
        )
    )

    point_is_accepted = (
        training_status
        == "ACCEPTED_POINT_AND_RANGE"
    )

    range_is_healthy = (
        _range_is_empirically_healthy(
            metadata
        )
    )

    if point_is_accepted:
        predicted_price = (
            experimental_point_price
        )

        prediction_mode = (
            "X2_ACCEPTED_POINT_AND_RANGE"
        )

        central_source = (
            "X2 validated point forecast"
        )

    else:
        # Previous-close/current-close baseline remains the safest central
        # point because X2's exact-price model did not beat it in evaluation.
        predicted_price = (
            current_close
        )

        prediction_mode = (
            "BASELINE_SAFE_WITH_X2_RANGE"
            if range_is_healthy
            else "BASELINE_SAFE"
        )

        central_source = (
            "Previous-close baseline"
        )

    predicted_move_rs = (
        predicted_price
        - current_close
    )

    predicted_return_pct = (
        (
            predicted_move_rs
            / current_close
        )
        * 100.0
        if current_close
        else 0.0
    )

    experimental_return_pct = (
        (
            experimental_move_rs
            / current_close
        )
        * 100.0
        if current_close
        else 0.0
    )

    trend_signal = (
        "BULLISH"
        if experimental_return_pct
        > 0.10
        else "BEARISH"
        if experimental_return_pct
        < -0.10
        else "NEUTRAL"
    )

    holdout = metadata.get(
        "holdout",
        {}
    )

    holdout_point = holdout.get(
        "point_model",
        {}
    )

    walk = metadata.get(
        "walk_forward",
        {}
    )

    return {
        "symbol": symbol,
        "latest_market_date": str(
            latest_date.date()
        ),

        # Backward-compatible central prediction fields.
        "current_close": round(
            current_close,
            4,
        ),
        "predicted_price": round(
            predicted_price,
            4,
        ),
        "predicted_return_percent": round(
            predicted_return_pct,
            6,
        ),
        "trend_signal": trend_signal,
        "model": (
            "StockVision Hybrid: "
            "Baseline-Safe Central + X2 Probabilistic Range"
            if not point_is_accepted
            else metadata.get(
                "model_name",
                "Precision X2",
            )
        ),

        # Clear production honesty.
        "prediction_mode": prediction_mode,
        "central_source": central_source,
        "x2_training_status": training_status,
        "x2_point_validation_status": (
            "ACCEPTED"
            if point_is_accepted
            else "EXPERIMENTAL"
        ),
        "x2_range_validation_status": (
            "EMPIRICALLY_HEALTHY"
            if range_is_healthy
            else "EXPERIMENTAL"
        ),

        # X2 point remains visible for research/live validation even if it
        # is not used as the production central price.
        "experimental_x2_point": {
            "price": round(
                experimental_point_price,
                4,
            ),
            "move_rupees": round(
                experimental_move_rs,
                4,
            ),
            "move_percent": round(
                experimental_return_pct,
                6,
            ),
            "move_atr": round(
                x2_point_atr,
                6,
            ),
        },

        "expected_range": {
            "lower": round(
                expected_lower,
                4,
            ),
            "upper": round(
                expected_upper,
                4,
            ),
            "nominal_coverage_percent": 80.0,
            "holdout_observed_coverage_percent": (
                holdout.get(
                    "range",
                    {}
                ).get(
                    "coverage_percent"
                )
            ),
            "walk_forward_observed_coverage_percent": (
                walk.get(
                    "range",
                    {}
                ).get(
                    "coverage_percent"
                )
            ),
            "basis": (
                "X2 q10/q90 range with validation calibration"
            ),
        },

        "historical_error_profile": {
            "mae_rupees": holdout_point.get(
                "mae"
            ),
            "median_error_rupees": holdout_point.get(
                "median_absolute_error"
            ),
            "within_10_percent": holdout_point.get(
                "within_10_rupees_percent"
            ),
            "within_15_percent": holdout_point.get(
                "within_15_rupees_percent"
            ),
            "within_20_percent": holdout_point.get(
                "within_20_rupees_percent"
            ),
            "over_30_percent": holdout_point.get(
                "over_30_rupees_percent"
            ),
            "over_40_percent": holdout_point.get(
                "over_40_rupees_percent"
            ),
            "p90_error_rupees": holdout_point.get(
                "p90_absolute_error_rupees"
            ),
            "p95_error_rupees": holdout_point.get(
                "p95_absolute_error_rupees"
            ),
        },

        "raw_x2_components_atr": {
            "hist": round(
                float(
                    components["hist"][0]
                ),
                6,
            ),
            "median": round(
                float(
                    components["median"][0]
                ),
                6,
            ),
            "ridge": round(
                float(
                    components["ridge"][0]
                ),
                6,
            ),
        },

        "precision_note": (
            "The exact X2 point model did not beat the previous-close "
            "baseline in historical evaluation, so StockVision keeps a "
            "baseline-safe central estimate while exposing the X2 empirical "
            "80% range and the experimental X2 point separately."
            if not point_is_accepted
            else
            "X2 point and range passed the configured acceptance rules."
        ),
    }
