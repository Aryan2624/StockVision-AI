# =========================================================
# STOCKVISION V9 MODEL ANALYTICS
# =========================================================

V9_ANALYTICS = {

    "model": {
        "name": "StockVision V9",
        "engine": "Relative Strength Intelligence",
        "target": "5-Day Performance vs NIFTY 50",
        "classes": [
            "UNDERPERFORM",
            "NEUTRAL",
            "OUTPERFORM",
        ],
        "stocks": 15,
        "features": 29,
        "evaluation": "Purged Walk-Forward",
        "test_samples": 17086,
    },

    # =====================================================
    # OVERALL PERFORMANCE
    # =====================================================

    "overall": {
        "model_accuracy": 36.37,
        "majority_accuracy": 32.51,
        "momentum_accuracy": 33.48,

        "macro_f1": 35.27,
        "balanced_accuracy": 35.90,

        "improvement_vs_majority": 3.86,
        "improvement_vs_momentum": 2.89,
    },

    # =====================================================
    # YEARLY WALK-FORWARD PERFORMANCE
    # =====================================================

    "yearly": [
        {
            "year": "2022",
            "model": 36.48,
            "majority": 32.85,
            "momentum": 32.42,
        },
        {
            "year": "2023",
            "model": 38.75,
            "majority": 30.80,
            "momentum": 33.14,
        },
        {
            "year": "2024",
            "model": 34.50,
            "majority": 33.17,
            "momentum": 33.58,
        },
        {
            "year": "2025",
            "model": 36.66,
            "majority": 30.85,
            "momentum": 33.70,
        },
        {
            "year": "2026",
            "model": 34.96,
            "majority": 36.32,
            "momentum": 35.26,
        },
    ],

    # =====================================================
    # PER-STOCK ROBUSTNESS
    # =====================================================

    "per_stock": [
        {
            "symbol": "ICICIBANK",
            "model": 40.91,
            "majority": 25.99,
            "momentum": 33.89,
            "macro_f1": 36.30,
        },
        {
            "symbol": "HDFCBANK",
            "model": 41.62,
            "majority": 28.09,
            "momentum": 35.12,
            "macro_f1": 34.98,
        },
        {
            "symbol": "BHARTIARTL",
            "model": 38.02,
            "majority": 26.16,
            "momentum": 32.31,
            "macro_f1": 36.82,
        },
        {
            "symbol": "LT",
            "model": 38.81,
            "majority": 30.20,
            "momentum": 32.48,
            "macro_f1": 38.32,
        },
        {
            "symbol": "RELIANCE",
            "model": 39.86,
            "majority": 33.10,
            "momentum": 33.45,
            "macro_f1": 35.61,
        },
        {
            "symbol": "AXISBANK",
            "model": 37.14,
            "majority": 31.61,
            "momentum": 32.57,
            "macro_f1": 36.62,
        },
        {
            "symbol": "SBIN",
            "model": 35.82,
            "majority": 30.82,
            "momentum": 34.68,
            "macro_f1": 35.08,
        },
        {
            "symbol": "SUNPHARMA",
            "model": 33.98,
            "majority": 29.94,
            "momentum": 32.92,
            "macro_f1": 33.32,
        },
        {
            "symbol": "KOTAKBANK",
            "model": 36.70,
            "majority": 33.27,
            "momentum": 31.34,
            "macro_f1": 35.52,
        },
        {
            "symbol": "MARUTI",
            "model": 35.03,
            "majority": 33.89,
            "momentum": 32.31,
            "macro_f1": 34.54,
        },
        {
            "symbol": "ITC",
            "model": 33.51,
            "majority": 33.07,
            "momentum": 33.42,
            "macro_f1": 31.26,
        },
        {
            "symbol": "TCS",
            "model": 35.29,
            "majority": 39.07,
            "momentum": 32.92,
            "macro_f1": 34.12,
        },
        {
            "symbol": "HCLTECH",
            "model": 31.26,
            "majority": 35.21,
            "momentum": 36.08,
            "macro_f1": 31.24,
        },
        {
            "symbol": "INFY",
            "model": 34.86,
            "majority": 38.81,
            "momentum": 32.13,
            "macro_f1": 34.75,
        },
        {
            "symbol": "WIPRO",
            "model": 32.84,
            "majority": 38.37,
            "momentum": 36.61,
            "macro_f1": 32.73,
        },
    ],

    # =====================================================
    # NON-OVERLAPPING 5-DAY TEST
    # =====================================================

    "non_overlapping": [
        {
            "offset": 0,
            "samples": 3450,
            "model": 35.97,
            "majority": 32.49,
            "momentum": 32.58,
            "macro_f1": 34.75,
            "balanced": 35.50,
        },
        {
            "offset": 1,
            "samples": 3435,
            "model": 37.41,
            "majority": 32.26,
            "momentum": 33.19,
            "macro_f1": 36.24,
            "balanced": 36.97,
        },
        {
            "offset": 2,
            "samples": 3420,
            "model": 35.96,
            "majority": 32.57,
            "momentum": 34.04,
            "macro_f1": 34.96,
            "balanced": 35.60,
        },
        {
            "offset": 3,
            "samples": 3391,
            "model": 36.80,
            "majority": 32.70,
            "momentum": 34.24,
            "macro_f1": 35.83,
            "balanced": 36.34,
        },
        {
            "offset": 4,
            "samples": 3390,
            "model": 35.72,
            "majority": 32.51,
            "momentum": 33.39,
            "macro_f1": 34.54,
            "balanced": 35.10,
        },
    ],

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    "feature_importance": [
        {"feature": "ATR14_PERCENT", "importance": 0.00953},
        {"feature": "VOLATILITY_20", "importance": 0.00800},
        {"feature": "VOLATILITY_10", "importance": 0.00649},
        {"feature": "ROLLING_BETA_20", "importance": 0.00576},
        {"feature": "SMA50_DISTANCE", "importance": 0.00465},
        {"feature": "TREND_20", "importance": 0.00463},
        {"feature": "MACD_HIST_PERCENT", "importance": 0.00441},
        {"feature": "RELATIVE_RETURN_20", "importance": 0.00389},
        {"feature": "NIFTY_VOLATILITY_20", "importance": 0.00386},
        {"feature": "RETURN_20", "importance": 0.00385},
        {"feature": "NIFTY_RETURN_1", "importance": 0.00360},
        {"feature": "EMA20_DISTANCE", "importance": 0.00338},
        {"feature": "NIFTY_RETURN_20", "importance": 0.00327},
        {"feature": "VOLUME_ZSCORE_20", "importance": 0.00252},
        {"feature": "RETURN_5", "importance": 0.00236},
    ],

    # =====================================================
    # ROBUSTNESS SUMMARY
    # =====================================================

    "robustness": {
        "stocks_beating_majority": 11,
        "stocks_total": 15,
        "stocks_beating_majority_3pp": 9,
        "stocks_beating_momentum": 13,

        "non_overlap_model_accuracy": 36.37,
        "non_overlap_majority_accuracy": 32.51,
        "non_overlap_momentum_accuracy": 33.49,

        "non_overlap_macro_f1": 35.26,
        "non_overlap_balanced_accuracy": 35.90,
    },

    # =====================================================
    # LIMITATIONS
    # =====================================================

    "limitations": [
        "36.37% is three-class walk-forward accuracy, not guaranteed stock-direction accuracy.",
        "V9 predicts relative performance versus NIFTY 50 over five trading days.",
        "The model has a modest statistical edge rather than deterministic predictive power.",
        "Performance varies across stocks and market periods.",
        "The original research universe contains 15 NSE stocks.",
        "Raw prediction probabilities are not calibrated certainty scores.",
        "Predictions should not be interpreted as investment advice.",
    ],
}


# =========================================================
# PUBLIC API FUNCTION
# =========================================================

def get_model_analytics():

    return V9_ANALYTICS