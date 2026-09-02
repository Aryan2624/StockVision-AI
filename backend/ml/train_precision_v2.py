
from __future__ import annotations

import sys

from .precision_v2 import (
    normalize_symbol,
    train_precision_v2,
)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m ml.train_precision_v2 RELIANCE.NS"
        )
        raise SystemExit(1)

    symbol = normalize_symbol(
        sys.argv[1]
    )

    train_precision_v2(symbol)


if __name__ == "__main__":
    main()
