# 📈 StockVision AI

### AI-Powered Stock Forecasting & Market Intelligence Platform

StockVision AI is a full-stack stock market analytics and forecasting platform built using **React, FastAPI, Machine Learning, Deep Learning and real market data**.

The project combines live stock analytics, technical indicators, BiLSTM forecasting, relative-strength intelligence and walk-forward model evaluation inside a modern interactive dashboard.

> ⚠️ StockVision AI is an educational and research project. It is not financial or investment advice.

---

## 🚀 Project Overview

Stock markets are noisy, dynamic and difficult to predict using only historical prices.

StockVision AI was developed to provide a more complete market intelligence system instead of relying on only one prediction model.

The platform combines:

- Live stock market data
- Interactive price charts
- Technical indicators
- Next-day BiLSTM prediction
- Multi-horizon BiLSTM forecasting
- V9 Relative Strength Intelligence
- NIFTY 50 benchmarking
- Walk-forward backtesting
- Model robustness analysis
- Model performance analytics
- Searchable NSE stocks
- Modern React dashboard

---

# 🧠 AI Architecture

StockVision uses multiple AI engines instead of depending on a single model.

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
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
 Technical Engine   BiLSTM Engine   V9 Relative AI
          │              │              │
 RSI / MACD       Price Forecast    Performance
 SMA / EMA        1D / 3D / 5D     vs NIFTY 50
 Volatility            / 10D            │
          │              │        ┌─────┼─────┐
          │              │        ▼     ▼     ▼
          │              │      UNDER  NEUTRAL  OUT
          │              │
          └──────────────┴──────────────┐
                                       ▼
                             Market Intelligence
                                       │
                                       ▼
                              React Dashboard