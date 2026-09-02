"""
CLI trainer for StockVision Precision Model.

Examples:
    python -m ml.train_precision RELIANCE.NS
    python -m ml.train_precision TCS.NS
"""

from __future__ import annotations

import sys

from .precision_model import (
    normalize_symbol,
    train_precision_model,
)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m ml.train_precision RELIANCE.NS"
        )
        raise SystemExit(1)

    symbol = normalize_symbol(
        sys.argv[1]
    )

    train_precision_model(
        symbol,
        verbose=1,
    )


if __name__ == "__main__":
    main()
