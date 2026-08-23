import math

import pandas as pd
import yfinance as yf

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)


# =========================================================
# STOCKVISION ML ENGINES
# =========================================================

from ml.model_manager import (
    predict_any_stock,
)

from ml.multi_model_manager import (
    predict_any_stock_multi,
)

from ml.training_status import (
    start_training,
    get_training_status,
)

from ml.v9_model_manager import (
    predict_relative_strength,
    get_v9_status,
)

from ml.analytics_manager import (
    get_model_analytics,
)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="StockVision AI API",

    description=(
        "Live stock market analytics, "
        "BiLSTM forecasting, V9 relative-strength intelligence "
        "and model analytics."
    ),

    version="2.4.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",

        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],

    allow_credentials=True,

    allow_methods=[
        "*",
    ],

    allow_headers=[
        "*",
    ],
)


# =========================================================
# SYMBOL NORMALIZATION
# =========================================================

def normalize_symbol(
    symbol: str,
):

    symbol = (
        str(symbol)
        .strip()
        .upper()
    )

    if not symbol:
        raise ValueError(
            "Stock symbol cannot be empty."
        )

    # Index symbol
    if symbol.startswith("^"):
        return symbol

    # Already has exchange suffix
    if "." in symbol:
        return symbol

    # Default market = NSE
    return f"{symbol}.NS"


# =========================================================
# SAFE FLOAT
# =========================================================

def safe_float(
    value,
    default=None,
):

    try:

        value = float(
            value
        )

        if (
            math.isnan(value)
            or
            math.isinf(value)
        ):
            return default

        return value

    except (
        TypeError,
        ValueError,
    ):
        return default


# =========================================================
# FLATTEN YFINANCE COLUMNS
# =========================================================

def flatten_columns(
    dataframe,
):

    df = dataframe.copy()

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):

        df.columns = [
            column[0]
            if isinstance(
                column,
                tuple,
            )
            else column

            for column in df.columns
        ]

    return df


# =========================================================
# RANGE CONFIGURATION
# =========================================================

def get_range_config(
    range_name,
):

    configs = {

        "1d": {
            "period": "5d",
            "interval": "5m",
        },

        "5d": {
            "period": "5d",
            "interval": "15m",
        },

        "1mo": {
            "period": "1mo",
            "interval": "60m",
        },

        "6mo": {
            "period": "6mo",
            "interval": "1d",
        },

        "ytd": {
            "period": "ytd",
            "interval": "1d",
        },

        "1y": {
            "period": "1y",
            "interval": "1d",
        },

        "5y": {
            "period": "5y",
            "interval": "1d",
        },
    }

    return configs.get(
        range_name,
        configs["1d"],
    )


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(
    daily_data,
):

    df = flatten_columns(
        daily_data
    )

    if (
        df.empty
        or
        "Close" not in df.columns
    ):

        return {
            "sma20": None,
            "ema20": None,
            "rsi14": None,
            "macd": None,
            "macd_signal": None,
            "daily_return": None,
        }

    close = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    # =====================================================
    # SMA 20
    # =====================================================

    sma20 = (
        close
        .rolling(
            20
        )
        .mean()
    )

    # =====================================================
    # EMA 20
    # =====================================================

    ema20 = (
        close
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    # =====================================================
    # RSI 14
    # =====================================================

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = (
        -delta.clip(
            upper=0
        )
    )

    average_gain = (
        gain
        .rolling(
            14
        )
        .mean()
    )

    average_loss = (
        loss
        .rolling(
            14
        )
        .mean()
    )

    rs = (
        average_gain
        /
        average_loss.replace(
            0,
            float("nan"),
        )
    )

    rsi14 = (
        100
        -
        (
            100
            /
            (
                1
                +
                rs
            )
        )
    )

    # =====================================================
    # MACD
    # =====================================================

    ema12 = (
        close
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        close
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    macd = (
        ema12
        -
        ema26
    )

    macd_signal = (
        macd
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    # =====================================================
    # DAILY RETURN
    # =====================================================

    daily_return = (
        close
        .pct_change()
        *
        100
    )

    return {

        "sma20":
            safe_float(
                sma20.iloc[-1]
            ),

        "ema20":
            safe_float(
                ema20.iloc[-1]
            ),

        "rsi14":
            safe_float(
                rsi14.iloc[-1]
            ),

        "macd":
            safe_float(
                macd.iloc[-1]
            ),

        "macd_signal":
            safe_float(
                macd_signal.iloc[-1]
            ),

        "daily_return":
            safe_float(
                daily_return.iloc[-1]
            ),
    }


# =========================================================
# BUILD CHART DATA
# =========================================================

def build_chart_data(
    dataframe,
    range_name,
):

    df = flatten_columns(
        dataframe
    )

    if (
        df.empty
        or
        "Close" not in df.columns
    ):
        return []

    df = df.copy()

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    if "Volume" in df.columns:

        df["Volume"] = pd.to_numeric(
            df["Volume"],
            errors="coerce",
        )

    df.dropna(
        subset=[
            "Close",
        ],
        inplace=True,
    )

    # =====================================================
    # LATEST SESSION FOR 1D
    # =====================================================

    if (
        range_name == "1d"
        and
        not df.empty
    ):

        latest_date = (
            df.index[-1]
            .date()
        )

        df = df[
            df.index.date
            ==
            latest_date
        ]

    chart = []

    for index, row in df.iterrows():

        price = safe_float(
            row[
                "Close"
            ]
        )

        if price is None:
            continue

        volume = safe_float(
            row.get(
                "Volume",
                0,
            ),
            0,
        )

        if range_name == "1d":

            label = (
                index.strftime(
                    "%H:%M"
                )
            )

        elif range_name in [
            "5d",
            "1mo",
        ]:

            label = (
                index.strftime(
                    "%d %b %H:%M"
                )
            )

        else:

            label = (
                index.strftime(
                    "%d %b %Y"
                )
            )

        chart.append(
            {
                "time":
                    label,

                "date":
                    index.strftime(
                        "%Y-%m-%d"
                    ),

                "price":
                    round(
                        price,
                        2,
                    ),

                "close":
                    round(
                        price,
                        2,
                    ),

                "volume":
                    int(
                        volume or 0
                    ),
            }
        )

    return chart


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {

        "message":
            "StockVision API is running",

        "version":
            "2.4.0",

        "engines": {

            "live_market":
                "ready",

            "next_day_prediction":
                "available",

            "v3_multi_horizon":
                "available",

            "v9_relative_strength":
                "available",

            "model_analytics":
                "available",
        },
    }


# =========================================================
# LIVE STOCK DATA
# =========================================================

@app.get(
    "/stock/{symbol}"
)
def get_stock(

    symbol: str,

    range: str = Query(
        default="1d",
    ),
):

    try:

        symbol = normalize_symbol(
            symbol
        )

        range = (
            str(range)
            .strip()
            .lower()
        )

        allowed_ranges = [
            "1d",
            "5d",
            "1mo",
            "6mo",
            "ytd",
            "1y",
            "5y",
        ]

        if range not in allowed_ranges:
            range = "1d"

        ticker = yf.Ticker(
            symbol
        )

        # =================================================
        # CHART HISTORY
        # =================================================

        config = get_range_config(
            range
        )

        history = ticker.history(
            period=
                config[
                    "period"
                ],

            interval=
                config[
                    "interval"
                ],

            auto_adjust=False,
        )

        history = flatten_columns(
            history
        )

        if history.empty:

            raise ValueError(
                f"No market data found for {symbol}."
            )

        # =================================================
        # LATEST SESSION
        # =================================================

        if range == "1d":

            latest_session_date = (
                history.index[-1]
                .date()
            )

            current_session = history[
                history.index.date
                ==
                latest_session_date
            ]

        else:

            current_session = history

        # =================================================
        # CURRENT PRICE
        # =================================================

        price = safe_float(
            current_session[
                "Close"
            ]
            .iloc[-1]
        )

        # =================================================
        # DAILY HISTORY
        # =================================================

        daily = ticker.history(
            period="6mo",
            interval="1d",
            auto_adjust=False,
        )

        daily = flatten_columns(
            daily
        )

        if daily.empty:

            raise ValueError(
                f"Could not retrieve daily history for {symbol}."
            )

        # =================================================
        # PREVIOUS CLOSE
        # =================================================

        previous_close = None

        try:

            fast_info = ticker.fast_info

            previous_close = safe_float(
                fast_info.get(
                    "previousClose"
                )
            )

        except Exception:

            previous_close = None

        if (
            previous_close is None
            and
            len(daily) >= 2
        ):

            previous_close = safe_float(
                daily[
                    "Close"
                ]
                .iloc[-2]
            )

        # =================================================
        # DAILY OHLCV
        # =================================================

        latest_daily = (
            daily.iloc[-1]
        )

        open_price = safe_float(
            latest_daily.get(
                "Open"
            )
        )

        high_price = safe_float(
            latest_daily.get(
                "High"
            )
        )

        low_price = safe_float(
            latest_daily.get(
                "Low"
            )
        )

        volume = safe_float(
            latest_daily.get(
                "Volume"
            ),
            0,
        )

        if volume is not None:

            volume = int(
                volume
            )

        # =================================================
        # CHANGE
        # =================================================

        change = None
        change_percent = None

        if (
            price is not None
            and
            previous_close not in [
                None,
                0,
            ]
        ):

            change = (
                price
                -
                previous_close
            )

            change_percent = (
                change
                /
                previous_close
                *
                100
            )

        # =================================================
        # INDICATORS
        # =================================================

        indicators = (
            calculate_indicators(
                daily
            )
        )

        # =================================================
        # COMPANY INFORMATION
        # =================================================

        company_name = symbol

        exchange = "NSE"

        currency = "INR"

        try:

            fast_info = ticker.fast_info

            currency_value = (
                fast_info.get(
                    "currency"
                )
            )

            if currency_value:

                currency = str(
                    currency_value
                )

        except Exception:

            pass

        try:

            info = ticker.info

            name_value = (
                info.get(
                    "longName"
                )
                or
                info.get(
                    "shortName"
                )
            )

            if name_value:

                company_name = (
                    name_value
                )

            exchange_value = (
                info.get(
                    "exchange"
                )
            )

            if exchange_value:

                exchange = (
                    exchange_value
                )

        except Exception:

            pass

        # =================================================
        # CHART
        # =================================================

        chart = build_chart_data(
            history,
            range,
        )

        # =================================================
        # RESPONSE
        # =================================================

        return {

            "symbol":
                symbol,

            "name":
                company_name,

            "exchange":
                exchange,

            "currency":
                currency,

            "range":
                range,

            "price":
                round(
                    price,
                    2,
                )
                if price is not None
                else None,

            "current_price":
                round(
                    price,
                    2,
                )
                if price is not None
                else None,

            "previous_close":
                round(
                    previous_close,
                    2,
                )
                if previous_close is not None
                else None,

            "change":
                round(
                    change,
                    2,
                )
                if change is not None
                else None,

            "change_percent":
                round(
                    change_percent,
                    2,
                )
                if change_percent is not None
                else None,

            "open":
                round(
                    open_price,
                    2,
                )
                if open_price is not None
                else None,

            "high":
                round(
                    high_price,
                    2,
                )
                if high_price is not None
                else None,

            "low":
                round(
                    low_price,
                    2,
                )
                if low_price is not None
                else None,

            "volume":
                volume,

            # =================================================
            # TOP-LEVEL INDICATORS
            # =================================================

            "sma20":
                indicators[
                    "sma20"
                ],

            "ema20":
                indicators[
                    "ema20"
                ],

            "rsi14":
                indicators[
                    "rsi14"
                ],

            "macd":
                indicators[
                    "macd"
                ],

            "macd_signal":
                indicators[
                    "macd_signal"
                ],

            "daily_return":
                indicators[
                    "daily_return"
                ],

            # =================================================
            # NESTED INDICATORS
            # =================================================

            "indicators": {

                # New naming
                "sma20":
                    indicators[
                        "sma20"
                    ],

                "ema20":
                    indicators[
                        "ema20"
                    ],

                "rsi14":
                    indicators[
                        "rsi14"
                    ],

                "macd":
                    indicators[
                        "macd"
                    ],

                "macd_signal":
                    indicators[
                        "macd_signal"
                    ],

                "daily_return":
                    indicators[
                        "daily_return"
                    ],

                # Compatibility naming
                "sma_20":
                    indicators[
                        "sma20"
                    ],

                "ema_20":
                    indicators[
                        "ema20"
                    ],

                "rsi":
                    indicators[
                        "rsi14"
                    ],
            },

            "chart":
                chart,

            "history":
                chart,
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        )


# =========================================================
# NEXT-DAY AI PREDICTION
# =========================================================

@app.get(
    "/predict/{symbol}"
)
def predict_stock(
    symbol: str,
):

    try:

        symbol = normalize_symbol(
            symbol
        )

        return predict_any_stock(
            symbol
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        )


# =========================================================
# V3 MULTI-HORIZON FORECAST
# =========================================================

@app.get(
    "/future-predict/{symbol}"
)
def future_predict(
    symbol: str,
):

    try:

        symbol = normalize_symbol(
            symbol
        )

        return predict_any_stock_multi(
            symbol
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        )


# =========================================================
# START V3 MODEL TRAINING
# =========================================================

@app.post(
    "/future-train/{symbol}"
)
def future_train(
    symbol: str,
):

    try:

        symbol = normalize_symbol(
            symbol
        )

        return start_training(
            symbol
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        )


# =========================================================
# V3 TRAINING STATUS
# =========================================================

@app.get(
    "/future-status/{symbol}"
)
def future_status(
    symbol: str,
):

    try:

        symbol = normalize_symbol(
            symbol
        )

        return get_training_status(
            symbol
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        )


# =========================================================
# V9 RELATIVE-STRENGTH PREDICTION
# =========================================================

@app.get(
    "/relative-predict/{symbol}"
)
def relative_predict(
    symbol: str,
):

    try:

        symbol = normalize_symbol(
            symbol
        )

        return predict_relative_strength(
            symbol
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        )


# =========================================================
# V9 MODEL STATUS
# =========================================================

@app.get(
    "/relative-status"
)
def relative_status():

    try:

        return get_v9_status()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(
                error
            ),
        )


# =========================================================
# V9 MODEL ANALYTICS
# =========================================================

@app.get(
    "/model-analytics"
)
def model_analytics():

    try:

        return get_model_analytics()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(
                error
            ),
        )