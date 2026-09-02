"""
Train StockVision Precision X1.

Run:
    python -m ml.train_x1 RELIANCE.NS
"""

from __future__ import annotations

import sys

from .x1_model import (
    normalize_symbol,
    train_x1,
)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m ml.train_x1 RELIANCE.NS"
        )
        raise SystemExit(1)

    train_x1(
        normalize_symbol(
            sys.argv[1]
        )
    )


if __name__ == "__main__":
    main()
