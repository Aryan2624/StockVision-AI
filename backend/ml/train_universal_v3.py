"""
Train StockVision Precision V3 Universal.

Run:
    python -m ml.train_universal_v3
"""

from __future__ import annotations

from .universal_precision_v3 import train_universal_v3


def main() -> None:
    train_universal_v3()


if __name__ == "__main__":
    main()
