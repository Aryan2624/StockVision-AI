"""
StockVision Precision X1 — Context Data Collector
=================================================

This is Step 1 of the new forecasting branch.

It builds a leakage-aware AFTER-MARKET dataset for next-NSE-trading-day
forecasting.

Important timing assumption
---------------------------
A forecast row for Indian trading date D is generated AFTER the NSE close on D.

Therefore:
- current-day stock/NIFTY/sector/India-VIX values from D are allowed;
- US indices, Brent and USD/INR are shifted so same-calendar-day future closes
  cannot leak into the Indian row.

The script saves a CSV containing:
- stock OHLC-derived features
- NIFTY context
- sector context
- India VIX context (when Yahoo returns it)
- lagged USD/INR
- lagged Brent
- lagged S&P 500
- lagged Nasdaq
- next-day targets for later model training

Run:
    python -m ml.x1_data_collector RELIANCE.NS
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

PERIOD = "8y"

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_CONTEXT = {
    "USDINR": "INR=X",
    "BRENT": "BZ=F",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
}

BANK_STOCKS = {
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "AXISBANK",
    "KOTAKBANK",
    "INDUSINDBK",
    "BANKBARODA",
    "PNB",
}

IT_STOCKS = {
    "TCS",
    "INFY",
    "HCLTECH",
    "WIPRO",
    "TECHM",
    "LTIM",
    "MPHASIS",
    "COFORGE",
}

ENERGY_STOCKS = {
    "RELIANCE",
    "ONGC",
    "NTPC",
    "POWERGRID",
    "BPCL",
    "IOC",
    "GAIL",
    "COALINDIA",
}

# Used only if Yahoo does not provide enough history for the NSE energy index.
# RELIANCE itself is intentionally excluded so the sector proxy is not just a
# disguised copy of the target stock.
ENERGY_PEER_BASKET = [
    "ONGC.NS",
    "BPCL.NS",
    "IOC.NS",
    "HINDPETRO.NS",
    "GAIL.NS",
    "PETRONET.NS",
]


# ============================================================
# SYMBOL HELPERS
# ============================================================

def normalize_symbol(symbol: str) -> str:
    clean = str(symbol or "").strip().upper().replace(" ", "")

    if not clean:
        raise ValueError("Stock symbol is required.")

    if clean.startswith("^") or "." in clean or "=" in clean:
        return clean

    return f"{clean}.NS"


def short_symbol(symbol: str) -> str:
    return normalize_symbol(symbol).replace(".NS", "")


def safe_name(symbol: str) -> str:
    return (
        normalize_symbol(symbol)
        .replace("^", "IDX_")
        .replace("=", "_EQ_")
        .replace(".", "_")
        .replace("-", "_")
    )


def sector_proxy_for(symbol: str) -> str:
    short = short_symbol(symbol)

    if short in BANK_STOCKS:
        return "^NSEBANK"

    if short in IT_STOCKS:
        return "^CNXIT"

    if short in ENERGY_STOCKS:
        return "^CNXENERGY"

    return "^NSEI"


# ============================================================
# DOWNLOAD / CLEAN
# ============================================================

def clean_yfinance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    frame = frame.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        first_level = set(frame.columns.get_level_values(0))

        if "Close" in first_level:
            frame.columns = frame.columns.get_level_values(0)
        else:
            frame.columns = frame.columns.get_level_values(-1)

    wanted = ["Open", "High", "Low", "Close", "Volume"]

    for column in wanted:
        if column not in frame.columns:
            if column == "Volume":
                frame[column] = 0.0
            else:
                raise ValueError(
                    f"Downloaded series is missing {column}."
                )

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame[wanted].copy()

    frame = frame.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    frame = frame[
        (frame["Open"] > 0)
        & (frame["High"] > 0)
        & (frame["Low"] > 0)
        & (frame["Close"] > 0)
    ]

    frame = frame[
        ~frame.index.duplicated(keep="last")
    ]

    # Normalize timezone-aware timestamps to timezone-naive calendar dates.
    index = pd.to_datetime(frame.index)

    try:
        index = index.tz_localize(None)
    except TypeError:
        pass

    frame.index = index.normalize()

    return frame.sort_index()


def download_daily(
    symbol: str,
    period: str = PERIOD,
    minimum_rows: int = 100,
) -> pd.DataFrame:
    frame = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    frame = clean_yfinance_frame(frame)

    if len(frame) < minimum_rows:
        raise ValueError(
            f"{symbol}: only {len(frame)} usable daily rows."
        )

    return frame


def optional_download(
    symbol: str,
    label: str,
    period: str = PERIOD,
) -> pd.DataFrame | None:
    try:
        frame = download_daily(
            symbol,
            period=period,
            minimum_rows=50,
        )

        print(
            f"  OK   {label:<14} {symbol:<12} rows={len(frame)}"
        )

        return frame

    except Exception as exc:
        print(
            f"  WARN {label:<14} {symbol:<12} unavailable: {exc}"
        )

        return None



def build_equal_weight_peer_index(
    symbols: list[str],
    period: str = PERIOD,
    minimum_members: int = 3,
) -> pd.DataFrame | None:
    """
    Build an equal-weight synthetic sector index from adjusted Yahoo closes.

    Each peer is rebased to 100 at its first valid point. The index is the
    cross-sectional mean of available rebased peers. This gives RELIANCE a
    real oil/gas peer context when Yahoo's NIFTY Energy history is unavailable.
    """
    rebased = []

    for symbol in symbols:
        frame = optional_download(
            symbol,
            f"PEER_{short_symbol(symbol)}",
            period,
        )

        if frame is None or frame.empty:
            continue

        close = frame["Close"].astype(float).copy()

        first_valid = close.dropna()

        if first_valid.empty:
            continue

        base = float(first_valid.iloc[0])

        if base <= 0:
            continue

        series = (
            close / base
        ) * 100.0

        series.name = symbol

        rebased.append(series)

    if len(rebased) < minimum_members:
        return None

    panel = pd.concat(
        rebased,
        axis=1,
    ).sort_index()

    synthetic_close = panel.mean(
        axis=1,
        skipna=True,
    )

    synthetic = pd.DataFrame(
        {
            "Open": synthetic_close,
            "High": synthetic_close,
            "Low": synthetic_close,
            "Close": synthetic_close,
            "Volume": 0.0,
        }
    )

    synthetic = synthetic.dropna(
        subset=["Close"]
    )

    return synthetic


# ============================================================
# INDICATORS
# ============================================================

def compute_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100.0 - (
        100.0
        / (1.0 + rs)
    )


def compute_atr(
    frame: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    close = frame["Close"].astype(float)

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def rolling_zscore(
    series: pd.Series,
    window: int = 20,
) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()

    return (
        (series - mean)
        / std.replace(0, np.nan)
    )


# ============================================================
# TIME-SAFE CONTEXT ALIGNMENT
# ============================================================

def same_day_context(
    close: pd.Series,
    india_dates: pd.Index,
) -> pd.Series:
    """
    For NSE/Nifty/sector/India-VIX series whose D close is available when an
    AFTER-MARKET D forecast is created.
    """
    close = close.copy()
    close.index = pd.to_datetime(close.index).normalize()

    return close.reindex(
        india_dates,
    ).ffill()


def lagged_foreign_context(
    close: pd.Series,
    india_dates: pd.Index,
) -> pd.Series:
    """
    Prevent same-calendar-day future-close leakage.

    Example:
      Indian row: 2026-08-27 after NSE close
      US 2026-08-27 close happens later, so it must NOT be used.
      US 2026-08-26 close is allowed.

    We shift each foreign daily observation's availability date +1 calendar day
    before aligning it to Indian trading dates.
    """
    shifted = close.copy()

    shifted.index = (
        pd.to_datetime(
            shifted.index
        ).normalize()
        + pd.Timedelta(days=1)
    )

    shifted = shifted[
        ~shifted.index.duplicated(keep="last")
    ].sort_index()

    return shifted.reindex(
        india_dates,
    ).ffill()


# ============================================================
# STOCK FEATURES
# ============================================================

def make_stock_features(
    stock: pd.DataFrame,
) -> pd.DataFrame:
    close = stock["Close"].astype(float)
    open_ = stock["Open"].astype(float)
    high = stock["High"].astype(float)
    low = stock["Low"].astype(float)
    volume = stock["Volume"].fillna(0).astype(float)

    features = pd.DataFrame(
        index=stock.index
    )

    # Price action
    for window in (1, 2, 3, 5, 10, 20):
        features[
            f"STOCK_RET_{window}D"
        ] = (
            close.pct_change(window)
            * 100.0
        )

    features["STOCK_GAP_PCT"] = (
        (
            open_
            / close.shift(1).replace(0, np.nan)
        )
        - 1.0
    ) * 100.0

    features["STOCK_INTRADAY_PCT"] = (
        (
            close
            / open_.replace(0, np.nan)
        )
        - 1.0
    ) * 100.0

    features["STOCK_RANGE_PCT"] = (
        (high - low)
        / close.replace(0, np.nan)
    ) * 100.0

    features["STOCK_CLOSE_LOCATION"] = (
        (close - low)
        / (high - low).replace(0, np.nan)
    )

    # Trend
    sma = {}
    ema = {}

    for window in (5, 10, 20, 50):
        sma[window] = close.rolling(window).mean()

        ema[window] = close.ewm(
            span=window,
            adjust=False,
        ).mean()

        features[
            f"STOCK_SMA{window}_DIST"
        ] = (
            (
                close
                / sma[window].replace(0, np.nan)
            )
            - 1.0
        ) * 100.0

        features[
            f"STOCK_EMA{window}_DIST"
        ] = (
            (
                close
                / ema[window].replace(0, np.nan)
            )
            - 1.0
        ) * 100.0

    features[
        "STOCK_EMA5_20_SPREAD"
    ] = (
        (
            ema[5]
            / ema[20].replace(0, np.nan)
        )
        - 1.0
    ) * 100.0

    # Momentum
    features["STOCK_RSI14"] = (
        compute_rsi(close)
    )

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    macd = ema12 - ema26

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
    ).mean()

    features["STOCK_MACD_PCT"] = (
        macd
        / close.replace(0, np.nan)
    ) * 100.0

    features[
        "STOCK_MACD_SIGNAL_PCT"
    ] = (
        macd_signal
        / close.replace(0, np.nan)
    ) * 100.0

    features["STOCK_MACD_HIST_PCT"] = (
        (macd - macd_signal)
        / close.replace(0, np.nan)
    ) * 100.0

    # Volatility
    atr14 = compute_atr(
        stock,
        14,
    )

    features["STOCK_ATR14_RS"] = (
        atr14
    )

    features["STOCK_ATR14_PCT"] = (
        atr14
        / close.replace(0, np.nan)
        * 100.0
    )

    one_day_return = (
        close.pct_change()
        * 100.0
    )

    for window in (5, 10, 20):
        features[
            f"STOCK_VOL_{window}"
        ] = (
            one_day_return
            .rolling(window)
            .std()
        )

    # Volume
    log_volume = np.log1p(
        volume
    )

    features[
        "STOCK_VOLUME_LOG_CHANGE"
    ] = (
        log_volume.diff()
    )

    features[
        "STOCK_VOLUME_Z20"
    ] = (
        rolling_zscore(
            volume,
            20,
        )
    )

    features[
        "STOCK_VOLUME_RATIO5_20"
    ] = (
        volume.rolling(5).mean()
        / volume.rolling(20).mean().replace(0, np.nan)
    )

    # Raw values needed for target/interpretation.
    features["BASE_CLOSE"] = close
    features["BASE_ATR14_RS"] = atr14

    return features


# ============================================================
# CONTEXT FEATURES
# ============================================================

def add_return_features(
    output: pd.DataFrame,
    name: str,
    aligned_close: pd.Series,
    windows: tuple[int, ...] = (1, 3, 5),
) -> None:
    for window in windows:
        output[
            f"{name}_RET_{window}D"
        ] = (
            aligned_close.pct_change(window)
            * 100.0
        )


def make_context_features(
    india_dates: pd.Index,
    nifty: pd.DataFrame,
    sector: pd.DataFrame,
    india_vix: pd.DataFrame | None,
    foreign: dict[str, pd.DataFrame | None],
) -> pd.DataFrame:
    output = pd.DataFrame(
        index=india_dates
    )

    # NSE / sector: same-day is valid after Indian market close.
    nifty_close = same_day_context(
        nifty["Close"],
        india_dates,
    )

    sector_close = same_day_context(
        sector["Close"],
        india_dates,
    )

    add_return_features(
        output,
        "NIFTY",
        nifty_close,
    )

    output[
        "NIFTY_VOL_10"
    ] = (
        output[
            "NIFTY_RET_1D"
        ]
        .rolling(10)
        .std()
    )

    add_return_features(
        output,
        "SECTOR",
        sector_close,
    )

    # India VIX: same-day close is okay for an AFTER-MARKET forecast.
    if india_vix is not None:
        vix_close = same_day_context(
            india_vix["Close"],
            india_dates,
        )

        output[
            "INDIA_VIX_LEVEL"
        ] = vix_close

        output[
            "INDIA_VIX_CHANGE_1D"
        ] = (
            vix_close.pct_change()
            * 100.0
        )

        output[
            "INDIA_VIX_Z20"
        ] = rolling_zscore(
            vix_close,
            20,
        )

    else:
        output[
            "INDIA_VIX_LEVEL"
        ] = np.nan

        output[
            "INDIA_VIX_CHANGE_1D"
        ] = np.nan

        output[
            "INDIA_VIX_Z20"
        ] = np.nan

    # Foreign / 24-hour global markets: shifted by one calendar day for safety.
    for label, frame in foreign.items():
        if frame is None:
            output[
                f"{label}_RET_1D_LAG"
            ] = np.nan

            output[
                f"{label}_RET_5D_LAG"
            ] = np.nan

            continue

        aligned = lagged_foreign_context(
            frame["Close"],
            india_dates,
        )

        output[
            f"{label}_RET_1D_LAG"
        ] = (
            aligned.pct_change()
            * 100.0
        )

        output[
            f"{label}_RET_5D_LAG"
        ] = (
            aligned.pct_change(5)
            * 100.0
        )

    return output


# ============================================================
# FINAL DATASET
# ============================================================

def build_x1_dataset(
    symbol: str,
    period: str = PERIOD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    symbol = normalize_symbol(
        symbol
    )

    print(
        "\n"
        + "=" * 76
    )
    print(
        " STOCKVISION PRECISION X1 — DATA COLLECTOR"
    )
    print(
        " Context-Enriched After-Market Next-Day Dataset"
    )
    print(
        "=" * 76
    )

    print(
        f"Target stock: {symbol}"
    )

    print(
        "\nDownloading primary Indian series..."
    )

    stock = download_daily(
        symbol,
        period=period,
        minimum_rows=500,
    )

    print(
        f"  OK   STOCK          {symbol:<12} rows={len(stock)}"
    )

    nifty = download_daily(
        "^NSEI",
        period=period,
        minimum_rows=500,
    )

    print(
        f"  OK   NIFTY          {'^NSEI':<12} rows={len(nifty)}"
    )

    sector_symbol = sector_proxy_for(
        symbol
    )

    if sector_symbol == "^NSEI":
        sector = nifty
        print(
            f"  INFO SECTOR         using NIFTY fallback for {symbol}"
        )
    else:
        sector = optional_download(
            sector_symbol,
            "SECTOR",
            period,
        )

        if sector is None:
            # RELIANCE / energy fallback: use a real peer basket before falling
            # back to the broad NIFTY index.
            if short_symbol(symbol) in ENERGY_STOCKS:
                print(
                    "  INFO SECTOR         trying equal-weight oil/gas peer basket..."
                )

                peer_sector = build_equal_weight_peer_index(
                    ENERGY_PEER_BASKET,
                    period=period,
                    minimum_members=3,
                )

                if peer_sector is not None:
                    sector = peer_sector
                    sector_symbol = "ENERGY_PEER_BASKET"

                    print(
                        f"  OK   SECTOR         ENERGY_PEER_BASKET rows={len(sector)}"
                    )

                else:
                    print(
                        "  INFO SECTOR         peer basket unavailable -> ^NSEI"
                    )
                    sector = nifty
                    sector_symbol = "^NSEI"

            else:
                print(
                    "  INFO SECTOR         fallback -> ^NSEI"
                )
                sector = nifty
                sector_symbol = "^NSEI"

    india_vix = optional_download(
        "^INDIAVIX",
        "INDIA_VIX",
        period,
    )

    print(
        "\nDownloading lagged global context..."
    )

    foreign: dict[
        str,
        pd.DataFrame | None
    ] = {}

    for label, ticker in GLOBAL_CONTEXT.items():
        foreign[label] = optional_download(
            ticker,
            label,
            period,
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

    dataset = pd.concat(
        [
            stock_features,
            context_features,
        ],
        axis=1,
    )

    # Cross-market relative features.
    dataset[
        "STOCK_MINUS_NIFTY_RET1"
    ] = (
        dataset[
            "STOCK_RET_1D"
        ]
        - dataset[
            "NIFTY_RET_1D"
        ]
    )

    dataset[
        "STOCK_MINUS_NIFTY_RET5"
    ] = (
        dataset[
            "STOCK_RET_5D"
        ]
        - dataset[
            "NIFTY_RET_5D"
        ]
    )

    dataset[
        "STOCK_MINUS_SECTOR_RET1"
    ] = (
        dataset[
            "STOCK_RET_1D"
        ]
        - dataset[
            "SECTOR_RET_1D"
        ]
    )

    # Targets: next NSE trading-day close.
    next_close = (
        dataset[
            "BASE_CLOSE"
        ].shift(-1)
    )

    dataset[
        "TARGET_NEXT_CLOSE"
    ] = (
        next_close
    )

    dataset[
        "TARGET_RESIDUAL_RS"
    ] = (
        next_close
        - dataset[
            "BASE_CLOSE"
        ]
    )

    dataset[
        "TARGET_RETURN_PCT"
    ] = (
        dataset[
            "TARGET_RESIDUAL_RS"
        ]
        / dataset[
            "BASE_CLOSE"
        ].replace(0, np.nan)
        * 100.0
    )

    dataset[
        "TARGET_MOVE_ATR"
    ] = (
        dataset[
            "TARGET_RESIDUAL_RS"
        ]
        / dataset[
            "BASE_ATR14_RS"
        ].replace(0, np.nan)
    )

    dataset.insert(
        0,
        "DATE",
        dataset.index,
    )

    dataset.insert(
        1,
        "SYMBOL",
        symbol,
    )

    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Do not silently fill unavailable context. We want to see missingness.
    missing_percent = (
        dataset.isna().mean()
        * 100.0
    ).sort_values(
        ascending=False
    )

    target_columns = {
        "TARGET_NEXT_CLOSE",
        "TARGET_RESIDUAL_RS",
        "TARGET_RETURN_PCT",
        "TARGET_MOVE_ATR",
    }

    feature_columns = [
        column
        for column in dataset.columns
        if column
        not in {
            "DATE",
            "SYMBOL",
            *target_columns,
        }
    ]

    # Training-ready rows require all currently available model features and
    # targets. If India VIX is completely unavailable, exclude its three
    # columns from the required set instead of deleting the whole dataset.
    all_nan_columns = [
        column
        for column in feature_columns
        if dataset[
            column
        ].isna().all()
    ]

    required_features = [
        column
        for column in feature_columns
        if column
        not in all_nan_columns
    ]

    training_ready = dataset.dropna(
        subset=(
            required_features
            + list(
                target_columns
            )
        )
    ).copy()

    report = {
        "symbol": symbol,
        "period": period,
        "sector_symbol": sector_symbol,
        "raw_stock_rows": int(
            len(
                stock
            )
        ),
        "dataset_rows_before_drop": int(
            len(
                dataset
            )
        ),
        "training_ready_rows": int(
            len(
                training_ready
            )
        ),
        "feature_count_available": int(
            len(
                required_features
            )
        ),
        "all_nan_optional_columns": (
            all_nan_columns
        ),
        "date_start": (
            str(
                training_ready[
                    "DATE"
                ].min().date()
            )
            if not training_ready.empty
            else None
        ),
        "date_end": (
            str(
                training_ready[
                    "DATE"
                ].max().date()
            )
            if not training_ready.empty
            else None
        ),
        "missing_percent_top_15": {
            key: round(
                float(
                    value
                ),
                3,
            )
            for key, value
            in missing_percent.head(
                15
            ).items()
        },
        "timing_assumption": (
            "Forecast is generated after NSE close. Foreign/24h context is "
            "shifted one calendar day before alignment to avoid same-day "
            "future-close leakage."
        ),
    }

    return (
        training_ready,
        report,
    )


# ============================================================
# SAVE / CLI
# ============================================================

def save_x1_dataset(
    symbol: str,
) -> dict[str, Any]:
    dataset, report = build_x1_dataset(
        symbol
    )

    name = safe_name(
        symbol
    )

    csv_path = (
        DATA_DIR
        / f"x1_{name}.csv"
    )

    report_path = (
        DATA_DIR
        / f"x1_{name}_report.json"
    )

    dataset.to_csv(
        csv_path,
        index=False,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 76
    )
    print(
        " DATA QUALITY SUMMARY"
    )
    print(
        "=" * 76
    )

    print(
        f"Training-ready rows:    {report['training_ready_rows']}"
    )

    print(
        f"Available features:     {report['feature_count_available']}"
    )

    print(
        f"Date range:             {report['date_start']} -> {report['date_end']}"
    )

    print(
        f"Sector context:         {report['sector_symbol']}"
    )

    if report[
        "all_nan_optional_columns"
    ]:
        print(
            "Unavailable optional columns:"
        )

        for column in report[
            "all_nan_optional_columns"
        ]:
            print(
                f"  - {column}"
            )

    print(
        "\nHighest missing percentages before training-row cleanup:"
    )

    for column, value in report[
        "missing_percent_top_15"
    ].items():
        print(
            f"  {column:<30} {value:>7.3f}%"
        )

    print(
        "\nSaved CSV:"
    )
    print(
        f"  {csv_path}"
    )

    print(
        "Saved report:"
    )
    print(
        f"  {report_path}"
    )

    print(
        "\nIMPORTANT:"
    )
    print(
        "This is only the data-validation step. Do not train X1 until "
        "the context columns and row count look healthy."
    )

    return report


def main() -> None:
    if len(
        sys.argv
    ) < 2:
        print(
            "Usage: python -m ml.x1_data_collector RELIANCE.NS"
        )

        raise SystemExit(
            1
        )

    save_x1_dataset(
        sys.argv[
            1
        ]
    )


if __name__ == "__main__":
    main()
