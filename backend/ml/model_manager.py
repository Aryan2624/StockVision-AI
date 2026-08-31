"""
StockVision Precision V3 Universal model manager.

Keeps FastAPI compatible:
    from ml.model_manager import predict_any_stock
"""

from __future__ import annotations

from typing import Any

from .universal_precision_v3 import (
    normalize_symbol,
    predict_universal_v3,
    train_universal_v3,
)


def predict_any_stock(symbol: str) -> dict[str, Any]:
    symbol = normalize_symbol(
        symbol
    )

    return predict_universal_v3(
        symbol,
        auto_train=True,
    )


def train_any_stock(symbol: str | None = None) -> dict[str, Any]:
    # V3 trains once across the complete configured universe.
    return train_universal_v3()
