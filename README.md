# 📈 StockVision AI

### AI-Powered Stock Forecasting & Market Intelligence Platform

StockVision AI is a full-stack stock market analytics and forecasting platform built using **React, FastAPI, Machine Learning, Deep Learning and real market data**.

The project combines live stock analytics, technical indicators, BiLSTM forecasting, relative-strength intelligence, prediction-history validation, uncertainty estimation, news sentiment, alerts and walk-forward model evaluation inside a modern interactive dashboard.

> ⚠️ **StockVision AI is an educational and research project. It is not financial or investment advice.**

---

## 🚀 Project Overview

StockVision AI was developed as a complete market intelligence platform instead of relying on one prediction model.

The platform combines:

- Live stock market data
- Searchable NSE stock universe
- Interactive candlestick and price charts
- Technical indicators
- Next-day prediction
- Multi-horizon BiLSTM forecasting
- V9 Relative Strength Intelligence
- NIFTY 50 benchmarking
- Walk-forward backtesting
- Prediction uncertainty ranges
- Automatic prediction-history capture
- Predicted vs Actual validation
- Direction Accuracy
- MAE / Avg Error %
- Range Coverage
- Previous-close baseline comparison
- LIVE vs REPLAY evaluation
- News & Sentiment
- Watchlist and Stock Comparison
- Alerts
- Full-screen prediction-history chart
- Modern React dashboard

---

# 🧠 AI Architecture

```text
                         STOCKVISION AI

                        Live Market Data
                               │
                               ▼
                        Yahoo Finance
                               │
                               ▼
                       FastAPI Backend
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
 Technical Engine       Price Forecasting      V9 Relative AI
          │                    │                    │
 RSI / MACD             BiLSTM / Hybrid      Performance vs NIFTY
 SMA / EMA              1D / 3D / 5D / 10D       │
 Volatility             Uncertainty Range     ┌────┼────┐
                                                ▼    ▼    ▼
                                             UNDER NEUTRAL OUT
          │                    │                    │
          └────────────────────┴────────────────────┘
                               │
                               ▼
                    Prediction History Engine
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
          LIVE Capture     Historical      Validation
                           REPLAY          Metrics
               │               │               │
               └───────────────┴───────────────┘
                               │
                               ▼
                      Market Intelligence
                               │
                               ▼
                        React Dashboard
```

---

# ✨ Main Features

## 📊 Live Market Dashboard
- Current price
- Open / High / Low
- Volume
- Intraday candlestick data
- Technical indicators
- Real-time style market dashboard

## 🔎 NSE Stock Search
Dynamic NSE stock universe with searchable stock selection.

## 📈 Technical Indicators
- SMA
- EMA
- RSI
- MACD
- MACD Signal
- Daily Return
- Volatility features

---

# 🤖 Forecasting Engine

## BiLSTM Forecasting
The BiLSTM pipeline uses historical prices and technical indicators.

Main features:
- Open
- High
- Low
- Close
- Volume
- SMA20
- EMA20
- RSI14
- MACD
- MACD Signal
- Daily Return

Multi-horizon forecasts:
- 1 Day
- 3 Days
- 5 Days
- 10 Days

---

# 🧠 V9 Relative Strength Intelligence

V9 predicts stock performance relative to NIFTY 50.

Classes:

```text
0 → UNDERPERFORM
1 → NEUTRAL
2 → OUTPERFORM
```

Walk-forward evaluation:

- Future test samples: **17,086**
- Majority baseline: **32.51%**
- Relative-momentum baseline: **33.48%**
- V9 accuracy: **36.37%**
- Macro F1: **35.27%**
- Balanced accuracy: **35.90%**

> Training accuracy is not treated as real-world evaluation.

---

# 🕒 Automatic Prediction History

Each saved prediction can include:

- Prediction Date
- Target Date
- Predicted Price
- Actual Price
- Difference
- Error %
- Expected Range
- Direction
- Range Result
- Status

Example:

```text
Prediction Date | Predicted | Actual | Difference | Status
-----------------------------------------------------------
28 Aug          | ₹223.50   | ₹221.80 | -₹1.70   | RESOLVED
31 Aug          | ₹224.10   | --      | --        | PENDING
```

---

# 🌐 Full NSE Daily Capture

The full-universe batch can:

- Process the current NSE universe
- Save one immutable prediction per stock/date
- Resolve older pending rows
- Avoid duplicates
- Track processed/captured/failed counts
- Resume after restart
- Rate-limit market-data requests

Some very new, suspended or data-poor securities may remain unavailable. StockVision does not fabricate values when data is missing.

---

# 🗂 LIVE vs REPLAY History

## 🟢 LIVE
A genuine forecast actually captured by StockVision at that time.

## 🔵 REPLAY
Historical model replay using the frozen model and information available on that historical base date.

Replay is useful for backtesting, but is clearly labelled and is **not presented as an original live forecast**.

---

# 📊 Model Evaluation

StockVision does not use one vague “accuracy” number.

## Direction Accuracy
```text
Correct Directions / Total Direction Samples × 100
```

## MAE
```text
Average(|Actual Price - Predicted Price|)
```

## Avg Error %
```text
|Actual - Predicted| / Actual × 100
```

## Range Coverage
```text
Actual Inside Expected Range / Resolved Samples × 100
```

## Baseline Comparison
```text
Baseline Prediction = Current / Previous Close
```

## Vs Baseline
```text
(Baseline MAE - Model MAE) / Baseline MAE × 100
```

Positive = model MAE is lower.  
Negative = previous-close baseline is better.

LIVE and REPLAY metrics are displayed separately.

---

# 📉 Prediction History Chart

The chart shows:

- Predicted Value
- Actual Value
- Difference
- Final / Resolved Value
- Expected Range

Filters:
- 7D
- 30D
- 3M
- 6M
- 1Y
- All Dates

Also supports **Full Screen mode**.

---

# 📰 News & Sentiment

- Recent stock/company news
- Relevance filtering
- Positive / Neutral / Negative sentiment
- Sentiment trend
- Topics and sources

---

# 🔔 Alerts

Supports:
- Price above threshold
- Price below threshold
- Model/signal alerts

---

# ⭐ Watchlist
Save stocks for quick monitoring.

# ⚖️ Stock Comparison
Compare selected stocks using market and technical information.

---

# 🛠 Tech Stack

## Frontend
- React
- Vite
- Tailwind CSS
- Recharts
- ApexCharts
- Lucide React

## Backend
- FastAPI
- Uvicorn
- Python
- Pandas
- NumPy
- yfinance

## ML / DL
- TensorFlow
- Keras
- Scikit-learn
- BiLSTM
- Gradient Boosting
- Ridge Regression
- Histogram Gradient Boosting
- Walk-Forward Validation

---

# 📂 Project Structure

```text
StockVision-AI/
├── backend/
│   ├── main.py
│   ├── data/
│   └── ml/
│       ├── model_manager.py
│       ├── universal_precision_v3.py
│       ├── train_universal_v3.py
│       ├── train_x1.py
│       ├── train_x2.py
│       ├── x1_model.py
│       ├── x2_live_predictor.py
│       ├── x2_probabilistic.py
│       └── models/
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       ├── CandlestickStockChart.jsx
│       └── components/
│           └── PredictionHistoryChart.jsx
├── .gitignore
├── LICENSE
└── README.md
```

---

# ⚙️ Installation

## Backend
```cmd
cd /d E:\StockVision-AI\backend
ml_venv\Scripts\activate
python -m uvicorn main:app
```

Backend:
```text
http://127.0.0.1:8000
```

## Frontend
```cmd
cd /d E:\StockVision-AI\frontend
npm install
npm run dev
```

Frontend:
```text
http://localhost:5173
```

---

# 🔌 Important API Endpoints

```text
GET  /
GET  /stocks
GET  /stock/{symbol}
GET  /stock-candles/{symbol}
GET  /prediction-history/{symbol}
GET  /prediction-history-overview
GET  /prediction-full-universe/status
POST /prediction-full-universe/start
POST /prediction-full-universe/restart
```

---

# 🖥 Dashboard Modules

```text
Dashboard
Markets
Watchlist
Stock Comparison
AI Prediction
Future Forecast
Model Analytics
News & Sentiment
Alerts
Settings
```

---

# ✅ Current Project Status

```text
Live Market Dashboard         ✅
NSE Stock Search              ✅
Technical Indicators          ✅
BiLSTM Forecasting            ✅
Multi-Horizon Forecasting     ✅
V9 Relative Strength AI       ✅
Walk-Forward Evaluation       ✅
Prediction History            ✅
Full NSE History Capture      ✅
LIVE / REPLAY History         ✅
Model Performance Metrics     ✅
Baseline Comparison           ✅
Prediction History Chart      ✅
Full-Screen Chart             ✅
News & Sentiment              ✅
Watchlist                     ✅
Stock Comparison              ✅
Alerts                        ✅
GitHub Repository             ✅
```

---

# 🎯 Project Goal

The goal is not to claim perfect stock-market prediction.

StockVision AI demonstrates an end-to-end AI engineering workflow combining:

- Data engineering
- Machine learning
- Deep learning
- Model validation
- Backend APIs
- Frontend development
- Real-world market data
- Honest evaluation
- Production-style UI design

---

# ⚠️ Limitations

Stock predictions are affected by:
- News
- Macroeconomic events
- Sentiment
- Liquidity
- Corporate actions
- Unexpected shocks
- Third-party data availability

Some price models may not consistently outperform a simple previous-close baseline.

Historical REPLAY results are backtesting evidence, not original live predictions.

---

# 🔮 Future Improvements

- Cloud deployment
- Persistent server-side alert scheduler
- More market-data providers
- Fundamental analysis
- Portfolio analytics
- Better uncertainty calibration
- Model monitoring
- Experiment tracking
- Improved live validation

---

# 👨‍💻 Author

**Aryan Dubey**

B.Tech Artificial Intelligence & Machine Learning Student

---

# 📜 License

Licensed under the **MIT License**.

---

# ⭐ Support

If you find StockVision AI useful or interesting, consider giving the repository a ⭐ on GitHub.

> **StockVision AI — AI-powered stock forecasting and market intelligence with real market data, model validation and a proper interactive UI.**
