
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

MODEL_VERSION = "precision-v2"
RANDOM_SEED = 42
DIRECTION_THRESHOLD = 0.15
MIN_VALIDATION_IMPROVEMENT = 0.50
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "RET1", "RET3", "RET5", "RET10",
    "GAP", "INTRADAY", "RANGE", "CLOSE_LOCATION",
    "SMA5_DIST", "SMA20_DIST", "SMA50_DIST",
    "EMA5_DIST", "EMA20_DIST", "EMA50_DIST",
    "RSI14", "MACD_PCT", "MACD_HIST_PCT",
    "ATR14_PCT", "VOL5", "VOL10", "VOL20",
    "VOLUME_Z20", "VOLUME_RATIO5_20",
    "NIFTY_RET1", "NIFTY_RET5", "NIFTY_VOL10",
    "REL_RET1", "REL_RET5",
    "SECTOR_RET1", "SECTOR_RET5", "REL_SECTOR_RET1",
    "ABOVE_SMA20", "ABOVE_SMA50",
]

BANKS = {
    "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB",
}
IT = {
    "TCS", "INFY", "HCLTECH", "WIPRO",
    "TECHM", "LTIM", "MPHASIS", "COFORGE",
}


def normalize_symbol(symbol: str) -> str:
    symbol = str(symbol or "").strip().upper().replace(" ", "")
    if not symbol:
        raise ValueError("Symbol is required.")
    if symbol.startswith("^") or "." in symbol:
        return symbol
    return f"{symbol}.NS"


def safe_name(symbol: str) -> str:
    return (
        normalize_symbol(symbol)
        .replace("^", "IDX_")
        .replace(".", "_")
        .replace("-", "_")
    )


def paths(symbol: str) -> dict[str, Path]:
    name = safe_name(symbol)
    return {
        "direction": MODEL_DIR / f"{name}_{MODEL_VERSION}_direction.pkl",
        "magnitude": MODEL_DIR / f"{name}_{MODEL_VERSION}_magnitude.pkl",
        "direct": MODEL_DIR / f"{name}_{MODEL_VERSION}_direct.pkl",
        "metadata": MODEL_DIR / f"{name}_{MODEL_VERSION}_metadata.json",
    }


def artifacts_exist(symbol: str) -> bool:
    p = paths(symbol)
    return all(p[k].exists() for k in p)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    frame = frame.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        if "Close" in set(frame.columns.get_level_values(0)):
            frame.columns = frame.columns.get_level_values(0)
        else:
            frame.columns = frame.columns.get_level_values(-1)

    needed = ["Open", "High", "Low", "Close", "Volume"]

    for col in needed:
        if col not in frame.columns:
            raise ValueError(f"Missing market column: {col}")
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame[needed].dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    frame = frame[
        (frame["Open"] > 0)
        & (frame["High"] > 0)
        & (frame["Low"] > 0)
        & (frame["Close"] > 0)
    ]

    return (
        frame[~frame.index.duplicated(keep="last")]
        .sort_index()
    )


def download(symbol: str, period: str = "8y") -> pd.DataFrame:
    frame = yf.download(
        normalize_symbol(symbol),
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    frame = clean(frame)

    if len(frame) < 500:
        raise ValueError(
            f"Not enough daily history for {symbol}: {len(frame)} rows."
        )

    return frame


def context_close(
    symbol: str,
    index: pd.Index,
    period: str,
) -> pd.Series:
    try:
        frame = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        frame = clean(frame)
        return frame["Close"].astype(float).reindex(index).ffill()
    except Exception:
        return pd.Series(np.nan, index=index, dtype=float)


def sector_proxy(symbol: str) -> str:
    short = normalize_symbol(symbol).replace(".NS", "")
    if short in BANKS:
        return "^NSEBANK"
    if short in IT:
        return "^CNXIT"
    return "^NSEI"


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
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
    return 100 - (100 / (1 + rs))


def atr_percent(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    h = frame["High"].astype(float)
    l = frame["Low"].astype(float)
    c = frame["Close"].astype(float)
    pc = c.shift(1)

    tr = pd.concat(
        [h - l, (h - pc).abs(), (l - pc).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return (atr / c.replace(0, np.nan)) * 100


def feature_frame(
    symbol: str,
    stock: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    c = stock["Close"].astype(float)
    o = stock["Open"].astype(float)
    h = stock["High"].astype(float)
    l = stock["Low"].astype(float)
    v = stock["Volume"].fillna(0).astype(float)

    f = pd.DataFrame(index=stock.index)

    f["RET1"] = c.pct_change(1) * 100
    f["RET3"] = c.pct_change(3) * 100
    f["RET5"] = c.pct_change(5) * 100
    f["RET10"] = c.pct_change(10) * 100

    f["GAP"] = ((o / c.shift(1).replace(0, np.nan)) - 1) * 100
    f["INTRADAY"] = ((c / o.replace(0, np.nan)) - 1) * 100
    f["RANGE"] = ((h / l.replace(0, np.nan)) - 1) * 100
    f["CLOSE_LOCATION"] = (
        (c - l) / (h - l).replace(0, np.nan)
    )

    sma5 = c.rolling(5).mean()
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()

    ema5 = c.ewm(span=5, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    f["SMA5_DIST"] = ((c / sma5) - 1) * 100
    f["SMA20_DIST"] = ((c / sma20) - 1) * 100
    f["SMA50_DIST"] = ((c / sma50) - 1) * 100
    f["EMA5_DIST"] = ((c / ema5) - 1) * 100
    f["EMA20_DIST"] = ((c / ema20) - 1) * 100
    f["EMA50_DIST"] = ((c / ema50) - 1) * 100

    f["RSI14"] = rsi(c)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    f["MACD_PCT"] = (macd / c) * 100
    f["MACD_HIST_PCT"] = ((macd - signal) / c) * 100

    ret1 = c.pct_change() * 100

    f["ATR14_PCT"] = atr_percent(stock)
    f["VOL5"] = ret1.rolling(5).std()
    f["VOL10"] = ret1.rolling(10).std()
    f["VOL20"] = ret1.rolling(20).std()

    mean20 = v.rolling(20).mean()
    std20 = v.rolling(20).std()
    f["VOLUME_Z20"] = (v - mean20) / std20.replace(0, np.nan)
    f["VOLUME_RATIO5_20"] = (
        v.rolling(5).mean() / mean20.replace(0, np.nan)
    )

    nifty = context_close("^NSEI", stock.index, period)
    nifty_ret1 = nifty.pct_change() * 100

    f["NIFTY_RET1"] = nifty_ret1
    f["NIFTY_RET5"] = nifty.pct_change(5) * 100
    f["NIFTY_VOL10"] = nifty_ret1.rolling(10).std()
    f["REL_RET1"] = f["RET1"] - f["NIFTY_RET1"]
    f["REL_RET5"] = f["RET5"] - f["NIFTY_RET5"]

    sector_symbol = sector_proxy(symbol)

    if sector_symbol == "^NSEI":
        sector = nifty
    else:
        sector = context_close(sector_symbol, stock.index, period)
        if sector.notna().sum() < 100:
            sector = nifty

    f["SECTOR_RET1"] = sector.pct_change() * 100
    f["SECTOR_RET5"] = sector.pct_change(5) * 100
    f["REL_SECTOR_RET1"] = f["RET1"] - f["SECTOR_RET1"]

    f["ABOVE_SMA20"] = (c > sma20).astype(float)
    f["ABOVE_SMA50"] = (c > sma50).astype(float)

    # Robust clipping of extreme market-data glitches.
    for col in f.columns:
        if col in {"ABOVE_SMA20", "ABOVE_SMA50"}:
            continue

        if col == "RSI14":
            f[col] = f[col].clip(0, 100)
        elif col == "CLOSE_LOCATION":
            f[col] = f[col].clip(0, 1)
        elif "VOLUME_Z" in col:
            f[col] = f[col].clip(-8, 8)
        elif "VOLUME_RATIO" in col:
            f[col] = f[col].clip(0, 8)
        else:
            f[col] = f[col].clip(-60, 60)

    return f[FEATURES]


def direction_class(values: np.ndarray | pd.Series) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return np.where(
        x > DIRECTION_THRESHOLD,
        2,
        np.where(x < -DIRECTION_THRESHOLD, 0, 1),
    ).astype(int)


def dataset_for(symbol: str, period: str = "8y") -> pd.DataFrame:
    stock = download(symbol, period)
    f = feature_frame(symbol, stock, period)
    c = stock["Close"].astype(float)

    next_ret = ((c.shift(-1) / c) - 1) * 100

    data = f.copy()
    data["BASE_CLOSE"] = c
    data["TARGET_RETURN"] = next_ret
    data["TARGET_MAGNITUDE"] = next_ret.abs()
    data["TARGET_DIRECTION"] = direction_class(next_ret)

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 600:
        raise ValueError(
            f"Only {len(data)} usable rows after feature engineering."
        )

    return data


def direction_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=1.5,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
        random_state=RANDOM_SEED,
    )


def regression_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=1.5,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
        random_state=RANDOM_SEED,
    )


def recency_weights(n: int) -> np.ndarray:
    return np.linspace(0.75, 1.25, n, dtype=float)


def class_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    w = recency_weights(len(y))
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)

    for cls, count in zip(classes, counts):
        w[y == cls] *= total / (len(classes) * count)

    return w


def fit_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    X = frame[FEATURES]
    y_ret = frame["TARGET_RETURN"].to_numpy(float)
    y_mag = frame["TARGET_MAGNITUDE"].to_numpy(float)
    y_dir = frame["TARGET_DIRECTION"].to_numpy(int)

    cap = max(2.0, float(np.percentile(np.abs(y_ret), 99)))

    d = direction_model()
    m = regression_model()
    r = regression_model()

    d.fit(X, y_dir, sample_weight=class_weights(y_dir))

    weights = recency_weights(len(frame))

    m.fit(
        X,
        np.clip(y_mag, 0, cap),
        sample_weight=weights,
    )

    r.fit(
        X,
        np.clip(y_ret, -cap, cap),
        sample_weight=weights,
    )

    return {
        "direction": d,
        "magnitude": m,
        "direct": r,
    }


def components(bundle: dict[str, Any], X: pd.DataFrame) -> dict[str, np.ndarray]:
    d = bundle["direction"]
    probs = d.predict_proba(X)

    p = {
        0: np.zeros(len(X)),
        1: np.zeros(len(X)),
        2: np.zeros(len(X)),
    }

    for idx, cls in enumerate(d.classes_):
        p[int(cls)] = probs[:, idx]

    mag = np.maximum(
        0.0,
        bundle["magnitude"].predict(X),
    )

    direct = bundle["direct"].predict(X)

    prob_mag = (p[2] - p[0]) * mag

    return {
        "direct": np.asarray(direct, float),
        "prob_mag": np.asarray(prob_mag, float),
        "magnitude": np.asarray(mag, float),
        "p_down": p[0],
        "p_flat": p[1],
        "p_up": p[2],
    }


def prices(base: np.ndarray, returns: np.ndarray) -> np.ndarray:
    return np.asarray(base, float) * (
        1 + np.asarray(returns, float) / 100
    )


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base: np.ndarray,
) -> dict[str, float]:
    actual = prices(base, y_true)
    pred = prices(base, y_pred)
    err = np.abs(actual - pred)

    return {
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "mape_percent": float(np.mean(err / actual) * 100),
        "direction_accuracy_percent": float(
            np.mean(
                direction_class(y_true) == direction_class(y_pred)
            ) * 100
        ),
        "within_2_rupees_percent": float(np.mean(err <= 2) * 100),
        "within_5_rupees_percent": float(np.mean(err <= 5) * 100),
        "within_10_rupees_percent": float(np.mean(err <= 10) * 100),
    }


def apply_calibration(
    comp: dict[str, np.ndarray],
    calibration: dict[str, Any],
) -> np.ndarray:
    pred = (
        calibration["direct_weight"] * comp["direct"]
        + calibration["prob_mag_weight"] * comp["prob_mag"]
        + calibration["bias_percent"]
    )

    cap = calibration.get("cap_percent")

    if cap is not None:
        pred = np.clip(pred, -cap, cap)

    return np.asarray(pred, float)


def optimize_calibration(
    train_returns: np.ndarray,
    val: pd.DataFrame,
    comp: dict[str, np.ndarray],
) -> dict[str, Any]:
    y = val["TARGET_RETURN"].to_numpy(float)
    base = val["BASE_CLOSE"].to_numpy(float)

    baseline = metrics(y, np.zeros_like(y), base)["mae"]

    abs_train = np.abs(train_returns)

    caps = [
        None,
        float(np.percentile(abs_train, 60)),
        float(np.percentile(abs_train, 70)),
        float(np.percentile(abs_train, 80)),
        float(np.percentile(abs_train, 90)),
    ]

    best = {
        "direct_weight": 0.0,
        "prob_mag_weight": 0.0,
        "baseline_weight": 1.0,
        "bias_percent": 0.0,
        "cap_percent": None,
        "validation_mae": baseline,
        "baseline_validation_mae": baseline,
        "validation_improvement_percent": 0.0,
        "accepted": False,
    }

    grid = np.arange(0, 1.0001, 0.05)

    for wd in grid:
        for wp in grid:
            if wd + wp > 1.0001:
                continue

            raw = wd * comp["direct"] + wp * comp["prob_mag"]

            bias = float(
                np.clip(
                    np.median(y - raw),
                    -0.25,
                    0.25,
                )
            )

            for cap in caps:
                pred = raw + bias

                if cap is not None:
                    pred = np.clip(pred, -cap, cap)

                mae = metrics(y, pred, base)["mae"]

                if mae < best["validation_mae"]:
                    improvement = (
                        (baseline - mae) / baseline * 100
                        if baseline > 0
                        else 0.0
                    )

                    best = {
                        "direct_weight": float(wd),
                        "prob_mag_weight": float(wp),
                        "baseline_weight": float(max(0, 1 - wd - wp)),
                        "bias_percent": bias,
                        "cap_percent": (
                            float(cap) if cap is not None else None
                        ),
                        "validation_mae": float(mae),
                        "baseline_validation_mae": float(baseline),
                        "validation_improvement_percent": float(improvement),
                        "accepted": True,
                    }

    if (
        best["validation_improvement_percent"]
        < MIN_VALIDATION_IMPROVEMENT
    ):
        best.update(
            {
                "direct_weight": 0.0,
                "prob_mag_weight": 0.0,
                "baseline_weight": 1.0,
                "bias_percent": 0.0,
                "cap_percent": None,
                "validation_mae": baseline,
                "validation_improvement_percent": 0.0,
                "accepted": False,
            }
        )

    return best


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(data)
    a = int(n * 0.70)
    b = int(n * 0.85)

    train = data.iloc[:a].copy()
    val = data.iloc[a:b].copy()
    test = data.iloc[b:].copy()

    if min(len(train), len(val), len(test)) < 100:
        raise ValueError("Not enough data for chronological split.")

    return train, val, test


def walk_forward(data: pd.DataFrame, folds: int = 4) -> dict[str, Any]:
    n = len(data)
    initial = int(n * 0.55)
    fold_size = max(80, (n - initial) // folds)

    all_true = []
    all_pred = []
    all_base = []
    fold_rows = []

    for i in range(folds):
        test_start = initial + i * fold_size
        test_end = n if i == folds - 1 else min(n, test_start + fold_size)

        if test_end - test_start < 50:
            continue

        dev = data.iloc[:test_start].copy()
        test = data.iloc[test_start:test_end].copy()

        val_size = max(80, int(len(dev) * 0.15))
        train = dev.iloc[:-val_size].copy()
        val = dev.iloc[-val_size:].copy()

        if len(train) < 300:
            continue

        bundle = fit_bundle(train)
        cal = optimize_calibration(
            train["TARGET_RETURN"].to_numpy(float),
            val,
            components(bundle, val[FEATURES]),
        )

        pred = apply_calibration(
            components(bundle, test[FEATURES]),
            cal,
        )

        y = test["TARGET_RETURN"].to_numpy(float)
        base = test["BASE_CLOSE"].to_numpy(float)

        model_m = metrics(y, pred, base)
        base_m = metrics(y, np.zeros_like(y), base)

        fold_rows.append(
            {
                "fold": i + 1,
                "test_start": str(test.index[0].date()),
                "test_end": str(test.index[-1].date()),
                "samples": len(test),
                "model_mae": model_m["mae"],
                "baseline_mae": base_m["mae"],
                "beats_baseline": model_m["mae"] < base_m["mae"],
            }
        )

        all_true.append(y)
        all_pred.append(pred)
        all_base.append(base)

    if not fold_rows:
        raise ValueError("No valid walk-forward folds.")

    y = np.concatenate(all_true)
    pred = np.concatenate(all_pred)
    base = np.concatenate(all_base)

    model_m = metrics(y, pred, base)
    base_m = metrics(y, np.zeros_like(y), base)

    improvement = (
        (base_m["mae"] - model_m["mae"])
        / base_m["mae"]
        * 100
        if base_m["mae"] > 0
        else 0.0
    )

    return {
        "fold_count": len(fold_rows),
        "samples": len(y),
        "folds_beating_baseline": sum(
            1 for row in fold_rows if row["beats_baseline"]
        ),
        "model": model_m,
        "baseline": base_m,
        "mae_improvement_vs_baseline_percent": float(improvement),
        "folds": fold_rows,
    }


def train_precision_v2(symbol: str) -> dict[str, Any]:
    symbol = normalize_symbol(symbol)

    print("\n" + "=" * 68)
    print(" STOCKVISION PRECISION V2")
    print(" Direction + Magnitude + Direct Return + Baseline Gate")
    print("=" * 68)
    print(f"Symbol: {symbol}")

    data = dataset_for(symbol, "8y")

    print(f"Rows: {len(data)} | Features: {len(FEATURES)}")
    print("\nRunning walk-forward evaluation...")

    wf = walk_forward(data)

    train, val, test = split_data(data)

    bundle = fit_bundle(train)

    cal = optimize_calibration(
        train["TARGET_RETURN"].to_numpy(float),
        val,
        components(bundle, val[FEATURES]),
    )

    test_comp = components(bundle, test[FEATURES])
    pred = apply_calibration(test_comp, cal)

    y_test = test["TARGET_RETURN"].to_numpy(float)
    base_test = test["BASE_CLOSE"].to_numpy(float)

    model_m = metrics(y_test, pred, base_test)
    baseline_m = metrics(y_test, np.zeros_like(y_test), base_test)

    holdout_improvement = (
        (baseline_m["mae"] - model_m["mae"])
        / baseline_m["mae"]
        * 100
        if baseline_m["mae"] > 0
        else 0.0
    )

    majority_folds = (
        wf["folds_beating_baseline"]
        >= max(1, (wf["fold_count"] + 1) // 2)
    )

    accepted = (
        cal["accepted"]
        and model_m["mae"] < baseline_m["mae"]
        and wf["model"]["mae"] < wf["baseline"]["mae"]
        and majority_folds
    )

    status = "ACCEPTED" if accepted else "EXPERIMENTAL"

    # Production learners are retrained on train + validation after evaluation.
    development = pd.concat([train, val])
    prod = fit_bundle(development)

    p = paths(symbol)

    joblib.dump(prod["direction"], p["direction"])
    joblib.dump(prod["magnitude"], p["magnitude"])
    joblib.dump(prod["direct"], p["direct"])

    metadata = {
        "symbol": symbol,
        "model_version": MODEL_VERSION,
        "model_name": "Precision V2 Direction + Magnitude Hybrid",
        "production_status": status,
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "direction_threshold_percent": DIRECTION_THRESHOLD,
        "sector_proxy": sector_proxy(symbol),
        "calibration": cal,
        "holdout_test": {
            "model": model_m,
            "baseline": baseline_m,
            "mae_improvement_vs_baseline_percent": float(
                holdout_improvement
            ),
            "beats_baseline": model_m["mae"] < baseline_m["mae"],
        },
        "walk_forward": wf,
        "acceptance": {
            "validation_accepted": bool(cal["accepted"]),
            "holdout_beats_baseline": bool(
                model_m["mae"] < baseline_m["mae"]
            ),
            "walk_forward_beats_baseline": bool(
                wf["model"]["mae"] < wf["baseline"]["mae"]
            ),
            "majority_folds_beat_baseline": bool(majority_folds),
        },
    }

    p["metadata"].write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nHOLDOUT TEST")
    print(f"Precision V2 MAE:   Rs {model_m['mae']:.2f}")
    print(f"Baseline MAE:       Rs {baseline_m['mae']:.2f}")
    print(f"Improvement:        {holdout_improvement:+.2f}%")
    print(f"RMSE:               Rs {model_m['rmse']:.2f}")
    print(f"MAPE:               {model_m['mape_percent']:.3f}%")
    print(
        f"Direction accuracy: {model_m['direction_accuracy_percent']:.2f}%"
    )
    print(f"Within Rs 2:        {model_m['within_2_rupees_percent']:.2f}%")
    print(f"Within Rs 5:        {model_m['within_5_rupees_percent']:.2f}%")
    print(f"Within Rs 10:       {model_m['within_10_rupees_percent']:.2f}%")

    print("\nWALK-FORWARD")
    print(
        f"Folds beating baseline: "
        f"{wf['folds_beating_baseline']}/{wf['fold_count']}"
    )
    print(f"Model MAE:          Rs {wf['model']['mae']:.2f}")
    print(f"Baseline MAE:       Rs {wf['baseline']['mae']:.2f}")
    print(
        f"Improvement:        "
        f"{wf['mae_improvement_vs_baseline_percent']:+.2f}%"
    )

    print("\n" + "=" * 68)
    print(f"PRODUCTION STATUS: {status}")
    print("=" * 68)

    return metadata


def load_metadata(symbol: str) -> dict[str, Any]:
    return json.loads(
        paths(symbol)["metadata"].read_text(encoding="utf-8")
    )


def load_bundle(symbol: str) -> dict[str, Any]:
    p = paths(symbol)
    return {
        "direction": joblib.load(p["direction"]),
        "magnitude": joblib.load(p["magnitude"]),
        "direct": joblib.load(p["direct"]),
    }


def predict_precision_v2(
    symbol: str,
    auto_train: bool = True,
) -> dict[str, Any]:
    symbol = normalize_symbol(symbol)

    if not artifacts_exist(symbol):
        if not auto_train:
            raise FileNotFoundError(
                f"Precision V2 is not trained for {symbol}."
            )
        train_precision_v2(symbol)

    metadata = load_metadata(symbol)
    bundle = load_bundle(symbol)

    stock = download(symbol, "2y")

    f = (
        feature_frame(symbol, stock, "2y")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if f.empty:
        raise ValueError("No valid latest feature row.")

    latest_date = f.index[-1]
    X = f[FEATURES].iloc[[-1]]

    comp = components(bundle, X)

    final_return = float(
        apply_calibration(
            comp,
            metadata["calibration"],
        )[0]
    )

    current_close = float(
        stock.loc[
            stock.index[stock.index <= latest_date][-1],
            "Close",
        ]
    )

    predicted_close = current_close * (
        1 + final_return / 100
    )

    signal = (
        "BULLISH"
        if final_return > DIRECTION_THRESHOLD
        else "BEARISH"
        if final_return < -DIRECTION_THRESHOLD
        else "NEUTRAL"
    )

    wf = metadata.get("walk_forward", {})
    holdout = metadata.get("holdout_test", {})

    return {
        "symbol": symbol,
        "latest_market_date": str(pd.Timestamp(latest_date).date()),
        "current_close": round(current_close, 4),
        "predicted_price": round(predicted_close, 4),
        "predicted_return_percent": round(final_return, 6),
        "trend_signal": signal,
        "model": metadata["model_name"],
        "model_version": MODEL_VERSION,
        "production_status": metadata["production_status"],
        "direction_probabilities": {
            "down": round(float(comp["p_down"][0]), 6),
            "flat": round(float(comp["p_flat"][0]), 6),
            "up": round(float(comp["p_up"][0]), 6),
        },
        "raw_components": {
            "direct_return_percent": round(
                float(comp["direct"][0]), 6
            ),
            "probability_magnitude_return_percent": round(
                float(comp["prob_mag"][0]), 6
            ),
            "predicted_absolute_magnitude_percent": round(
                float(comp["magnitude"][0]), 6
            ),
            "previous_close_return_percent": 0.0,
        },
        "blend": metadata["calibration"],
        "evaluation": {
            "holdout_test": holdout,
            "walk_forward_summary": {
                "fold_count": wf.get("fold_count"),
                "samples": wf.get("samples"),
                "folds_beating_baseline": wf.get(
                    "folds_beating_baseline"
                ),
                "model": wf.get("model"),
                "baseline": wf.get("baseline"),
                "mae_improvement_vs_baseline_percent": wf.get(
                    "mae_improvement_vs_baseline_percent"
                ),
            },
        },
        "precision_note": (
            "Precision V2 is accepted only when chronological validation, "
            "holdout testing and walk-forward evaluation support an edge "
            "over the previous-close baseline. Rs 1-2 error is measured, "
            "not guaranteed."
        ),
    }
