"""
Train StockVision Precision V2.1.

Examples:
    python -m ml.train_precision_v2_1 RELIANCE.NS
    python -m ml.train_precision_v2_1 TCS.NS
"""

from __future__ import annotations

import sys

from .precision_v2_1 import (
    normalize_symbol,
    train_precision_v2_1,
)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m ml.train_precision_v2_1 RELIANCE.NS"
        )
        raise SystemExit(1)

    symbol = normalize_symbol(
        sys.argv[1]
    )

    train_precision_v2_1(
        symbol
    )


if __name__ == "__main__":
    main()
