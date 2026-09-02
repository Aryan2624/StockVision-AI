"""
Train StockVision Precision X2.

Run:
    python -m ml.train_x2 RELIANCE.NS
"""

from __future__ import annotations

import sys

from .x2_probabilistic import (
    normalize_symbol,
    train_x2,
)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m ml.train_x2 RELIANCE.NS"
        )
        raise SystemExit(1)

    train_x2(
        normalize_symbol(
            sys.argv[1]
        )
    )


if __name__ == "__main__":
    main()
