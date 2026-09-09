# 📈 StockVision AI

### AI-Powered Stock Forecasting & Market Intelligence Platform

**StockVision AI** is a full-stack stock market analytics and forecasting platform built using **React, FastAPI, Machine Learning, Deep Learning, and real market data**.

The project combines live stock analytics, technical indicators, **BiLSTM forecasting, relative-strength intelligence, prediction-history validation, uncertainty estimation, news sentiment, alerts, and walk-forward model evaluation** inside a modern interactive dashboard.

> ⚠️ **StockVision AI is an educational and research project. It is not financial or investment advice.**

---

## 🚀 Project Overview

StockVision AI was developed as a complete **market intelligence platform** instead of relying on only one prediction model.

The platform combines:

- Live stock market data
- Searchable NSE stock universe
- AI-based stock forecasting
- Bidirectional LSTM model
- Technical indicator analysis
- Relative strength analysis
- Prediction history tracking
- Forecast validation
- Model uncertainty estimation
- News sentiment analysis
- Market alerts
- Walk-forward evaluation
- Interactive financial charts
- Modern responsive dashboard

The goal is to provide users with a single platform where they can **search, analyze, compare, and study stocks using both financial indicators and AI-powered forecasting**.

---

# ✨ Main Features

## 📊 Live Stock Market Data

StockVision AI fetches real stock market information using **Yahoo Finance (`yfinance`)**.

The application can retrieve:

- Open Price
- High Price
- Low Price
- Close Price
- Adjusted Close
- Trading Volume
- Historical Price Data
- Daily Price Movement

Example supported ticker symbols:

```text
RELIANCE.NS
TCS.NS
INFY.NS
HDFCBANK.NS
ICICIBANK.NS
SBIN.NS
ITC.NS
AAPL
MSFT
GOOGL
TSLA
```

---

## 🔎 Searchable Stock Universe

Users can search for different stocks directly from the StockVision dashboard.

Instead of being limited to one stock, the architecture is designed to support analysis for **multiple stocks dynamically**.

Example:

```text
Search Stock
      ↓
RELIANCE.NS
      ↓
Fetch Market Data
      ↓
Calculate Indicators
      ↓
Load / Train Model
      ↓
Generate Prediction
      ↓
Display Dashboard
```

---

# 🤖 AI Stock Forecasting

The main forecasting model used in StockVision AI is a:

## Bidirectional Long Short-Term Memory Network

```text
Bi-LSTM
```

A traditional LSTM learns patterns from sequential financial data.

A **Bidirectional LSTM** uses forward and backward sequence-processing layers during training, allowing the network to learn richer relationships within the historical input window.

The model analyzes historical stock patterns and generates the predicted stock price.

---

# 🧠 Model Input

The current AI model uses:

```text
Sequence Length : 60 Trading Days
Features        : 11
Model           : Bidirectional LSTM
```

The main input features are:

```text
Open
High
Low
Close
Volume
SMA20
EMA20
RSI14
MACD
SIGNAL
DAILY_RETURN
```

---

# ⚙️ Prediction Pipeline

```text
Yahoo Finance
      ↓
Historical Stock Data
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Technical Indicators
      ↓
Feature Scaling
      ↓
60-Day Sequence
      ↓
Bi-LSTM Model
      ↓
Price Forecast
      ↓
Inverse Scaling
      ↓
FastAPI Backend
      ↓
React Dashboard
```

---

# 📈 Technical Indicators

StockVision AI uses multiple technical indicators to provide additional market context to the prediction system.

---

## 📉 SMA — Simple Moving Average

The **Simple Moving Average** calculates the average closing price over a selected number of trading days.

Current indicator:

```text
SMA20
```

Formula:

```text
SMA = Sum of Closing Prices / Number of Periods
```

SMA helps identify the overall market trend.

---

## 📈 EMA — Exponential Moving Average

The **Exponential Moving Average** gives greater importance to recent price movements.

Current indicator:

```text
EMA20
```

EMA reacts faster to recent price changes compared to SMA.

---

## ⚡ RSI — Relative Strength Index

StockVision uses:

```text
RSI14
```

RSI measures the strength and momentum of recent price movements.

Common interpretation:

```text
RSI > 70
Potentially Overbought

RSI < 30
Potentially Oversold
```

---

## 📊 MACD

MACD is used to understand momentum and trend changes.

StockVision calculates:

```text
MACD
SIGNAL
```

The relationship between these values can help identify possible bullish or bearish momentum.

---

## 📈 Daily Return

Daily return measures the percentage movement of the stock compared with the previous trading session.

```text
Daily Return =
(Current Close - Previous Close)
/
Previous Close
```

---

# 🧭 Relative Strength Intelligence

StockVision AI also includes **relative-strength based analysis**.

Instead of analyzing only the predicted price, the system can compare stock momentum and market behavior to provide additional context.

This helps reduce dependence on a single AI forecast.

The intelligence layer can combine information such as:

```text
Price Movement
+
Technical Indicators
+
Momentum
+
Relative Strength
+
AI Forecast
+
Market Sentiment
=
Market Intelligence
```

---

# 📰 News Sentiment Analysis

Stock prices can be affected by market news and investor sentiment.

StockVision AI is designed to include a **news sentiment layer** where financial headlines can be analyzed to identify:

```text
Positive Sentiment
Neutral Sentiment
Negative Sentiment
```

This information can be displayed along with technical and forecasting results.

---

# 🎯 Prediction History Validation

One major challenge with stock forecasting projects is that predictions are usually shown without checking later whether they were correct.

StockVision AI addresses this using **prediction history tracking**.

The system can store:

```text
Prediction Date
Stock Symbol
Current Price
Predicted Price
Actual Future Price
Prediction Error
Direction Prediction
Direction Result
```

This allows historical AI predictions to be validated against real market outcomes.

---

# 📊 Model Evaluation

The current **RELIANCE.NS** model produced the following experimental test results:

| Metric | Result |
|---|---:|
| MAE | ₹79.60 |
| RMSE | ₹89.24 |
| R² Score | 0.0106 |
| MAPE | 5.60% |
| Direction Accuracy | 49.05% |

These results are intentionally shown transparently because StockVision AI is an **experimental AI forecasting system**.

The objective of the project is not to claim perfect stock prediction, but to build a platform where model performance can be **measured, validated, monitored, and improved**.

---

# 🎲 Prediction Uncertainty

Stock market predictions contain uncertainty.

Instead of presenting the AI forecast as a guaranteed value, StockVision AI is designed to provide prediction context such as:

```text
Predicted Price
Prediction Range
Model Confidence
Historical Error
Market Trend
Risk Context
```

This creates a more realistic representation of AI forecasting.

---

# 🔄 Walk-Forward Model Evaluation

StockVision AI is designed to support **walk-forward evaluation**.

Unlike random train-test splitting, walk-forward evaluation respects the chronological nature of financial data.

Example:

```text
Train
2018 ─────────── 2022

Test
2023

        ↓

Train
2018 ─────────────── 2023

Test
2024

        ↓

Train
2018 ─────────────────── 2024

Test
2025
```

This approach provides a more realistic evaluation of how a forecasting model may behave when exposed to future unseen data.

---

# 🔔 Stock Alerts

The platform is designed to support alerts based on conditions such as:

```text
Stock Price Target
RSI Level
Large Price Movement
AI Forecast Change
Technical Indicator Signal
Market Risk
```

This makes StockVision AI more than a simple prediction application.

---

# 📊 Interactive Dashboard

The StockVision frontend provides a modern trading-dashboard style interface.

The dashboard can include:

- Stock Search
- Current Price
- Price Change
- Historical Chart
- AI Forecast
- Technical Indicators
- RSI
- MACD
- SMA
- EMA
- Watchlist
- Prediction History
- News Sentiment
- Market Intelligence
- Alerts

---

# 📉 Interactive Charts

The frontend uses:

```text
Recharts
```

for responsive stock market visualizations.

Charts can represent:

```text
Historical Closing Price
Predicted Price
Price Trends
Technical Indicators
Market Movement
Prediction History
```

---

# ⭐ Watchlist

StockVision includes a watchlist-style interface where important stocks can be monitored.

Future versions can allow users to create persistent personalized watchlists.

Example:

```text
My Watchlist

RELIANCE.NS
TCS.NS
INFY.NS
HDFCBANK.NS
SBIN.NS
```

---

# 🏗️ System Architecture

```text
                 ┌─────────────────────────┐
                 │      Yahoo Finance      │
                 │        yfinance         │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Historical Market Data  │
                 │      OHLCV Prices       │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Feature Engineering   │
                 │                         │
                 │ SMA / EMA / RSI / MACD  │
                 │      Daily Return       │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    Data Preprocessing   │
                 │        Scaling          │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      Bi-LSTM Model      │
                 │    Price Forecasting    │
                 └────────────┬────────────┘
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │  Prediction  │ │  Technical   │ │    Market    │
      │   History    │ │ Intelligence │ │  Sentiment   │
      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     FastAPI Backend     │
                 │     REST API Layer      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      React + Vite       │
                 │   StockVision Dashboard │
                 └─────────────────────────┘
```

---

# 🛠️ Technology Stack

## 💻 Frontend

| Technology | Purpose |
|---|---|
| React | Frontend UI |
| Vite | Development & Build Tool |
| Tailwind CSS | Styling |
| Recharts | Financial Charts |
| Lucide React | Icons |
| JavaScript | Frontend Logic |

---

## ⚡ Backend

| Technology | Purpose |
|---|---|
| Python | Backend & AI |
| FastAPI | REST API |
| Uvicorn | ASGI Server |
| yfinance | Market Data |
| Pandas | Data Processing |
| NumPy | Numerical Computing |

---

## 🤖 Machine Learning

| Technology | Purpose |
|---|---|
| TensorFlow | Deep Learning |
| Keras | Neural Network Development |
| Bi-LSTM | Time-Series Forecasting |
| Scikit-learn | Data Preprocessing |
| Joblib | Scaler / Model Utilities |

---

# 📂 Project Structure

```text
StockVision-AI/
│
├── backend/
│   │
│   ├── main.py
│   │
│   ├── requirements.txt
│   │
│   ├── models/
│   │   ├── RELIANCE_NS.keras
│   │   └── ...
│   │
│   ├── scalers/
│   │   └── ...
│   │
│   ├── services/
│   │   └── ...
│   │
│   └── utils/
│       └── ...
│
├── frontend/
│   │
│   ├── src/
│   │   ├── components/
│   │   ├── assets/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   │
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── models/
│   └── ...
│
├── .gitignore
│
├── README.md
│
└── LICENSE
```

> Project structure may change as StockVision AI continues to develop.

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/Aryan2624/StockVision-AI.git
```

Enter the project:

```bash
cd StockVision-AI
```

---

# 🐍 Backend Setup

Go to the backend directory:

```bash
cd backend
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If installing manually:

```bash
pip install fastapi uvicorn yfinance pandas numpy scikit-learn tensorflow joblib
```

Start the FastAPI backend:

```bash
uvicorn main:app --reload
```

---

# ⚛️ Frontend Setup

Open another terminal and move into the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the React application:

```bash
npm run dev
```

The frontend normally runs at:

```text
http://localhost:5173
```

---

# 🚀 Run Complete StockVision AI

Use two terminals.

## Terminal 1 — Backend

```bash
cd backend
venv\Scripts\activate
uvicorn main:app --reload
```

## Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

Then open:

```text
http://localhost:5173
```

---

# 🔄 Application Workflow

```text
User Opens StockVision
          ↓
Searches Stock Symbol
          ↓
React Sends API Request
          ↓
FastAPI Receives Symbol
          ↓
Yahoo Finance Data Downloaded
          ↓
Data Cleaning
          ↓
Technical Indicators Generated
          ↓
Features Scaled
          ↓
Last 60 Trading Days Selected
          ↓
Bi-LSTM Model Loaded
          ↓
Stock Price Predicted
          ↓
Prediction Inverse Scaled
          ↓
Technical Analysis Generated
          ↓
Market Intelligence Generated
          ↓
API Returns Results
          ↓
React Dashboard Updated
```

---

# 🎯 Project Objectives

StockVision AI was built to:

- Develop an end-to-end AI application
- Combine AI with full-stack development
- Work with real financial market data
- Implement time-series forecasting
- Apply Deep Learning to financial data
- Develop a Bi-LSTM model
- Perform technical stock analysis
- Track prediction performance
- Measure forecasting uncertainty
- Build a modern interactive UI
- Support multiple stocks
- Create a scalable market-intelligence platform
- Improve practical AI engineering skills

---

# 🧪 Model Development

The model development process includes:

```text
Collect Historical Data
        ↓
Clean Dataset
        ↓
Generate Technical Indicators
        ↓
Select Features
        ↓
Scale Features
        ↓
Create 60-Day Sequences
        ↓
Split Data Chronologically
        ↓
Train Bi-LSTM
        ↓
Fine Tune Model
        ↓
Evaluate Model
        ↓
Save Model
        ↓
Save Scaler
        ↓
Serve Through FastAPI
```

---

# 📊 Evaluation Metrics

StockVision uses multiple metrics because relying only on accuracy is not suitable for stock price forecasting.

## MAE

```text
Mean Absolute Error
```

Measures the average absolute difference between the predicted price and actual price.

---

## RMSE

```text
Root Mean Squared Error
```

Penalizes larger forecasting errors more heavily.

---

## MAPE

```text
Mean Absolute Percentage Error
```

Measures forecasting error as a percentage.

---

## R² Score

Measures how well the model explains variation in stock prices.

---

## Direction Accuracy

Direction accuracy checks whether the model correctly predicts whether the stock moves:

```text
UP
or
DOWN
```

This is particularly useful for evaluating financial forecasting models.

---

# 💾 Model Storage

Trained models are stored in `.keras` format.

Example:

```text
models/
└── RELIANCE_NS.keras
```

Associated scalers and model metadata can also be stored for each stock.

Generated per-stock metadata files are excluded from Git tracking where required using:

```text
.gitignore
```

This keeps the repository clean and prevents unnecessary generated files from being committed.

---

# 🔮 Future Improvements

- [ ] Improve directional prediction accuracy
- [ ] Improve model generalization
- [ ] Train models for more NSE stocks
- [ ] Automatic model training for new stocks
- [ ] Automatic model retraining
- [ ] Add GRU model comparison
- [ ] Add Transformer forecasting model
- [ ] Add ensemble forecasting
- [ ] Add candlestick charts
- [ ] Add Bollinger Bands
- [ ] Add ATR
- [ ] Add ADX
- [ ] Add volume indicators
- [ ] Improve relative-strength engine
- [ ] Improve prediction confidence
- [ ] Add forecast confidence intervals
- [ ] Expand prediction-history validation
- [ ] Improve walk-forward evaluation
- [ ] Add financial news APIs
- [ ] Improve sentiment analysis
- [ ] Add company fundamentals
- [ ] Add sector comparison
- [ ] Add market benchmark comparison
- [ ] Add stock screener
- [ ] Add portfolio tracker
- [ ] Add portfolio analytics
- [ ] Add persistent watchlist
- [ ] Add user authentication
- [ ] Add database integration
- [ ] Add notification system
- [ ] Add price alerts
- [ ] Add AI-generated market summaries
- [ ] Add explainable AI
- [ ] Deploy frontend
- [ ] Deploy backend
- [ ] Improve mobile experience

---

# ⚠️ Limitations

Financial markets are affected by many factors including:

```text
Economic Events
Company Performance
Interest Rates
Market Sentiment
Global Events
Government Policies
Institutional Activity
Liquidity
Unexpected News
Investor Psychology
```

Historical stock data alone cannot perfectly predict future prices.

The current Bi-LSTM model is experimental and its forecasts should not be treated as guaranteed market outcomes.

---

# ⚠️ Disclaimer

StockVision AI is created for:

```text
Education
Research
Machine Learning Experimentation
Portfolio Development
```

It is **not a financial advisory system**.

The predictions, indicators, sentiment results, or market intelligence generated by StockVision AI should not be considered investment recommendations.

Always perform your own research before making financial decisions.

---

# 💡 What I Learned

Building StockVision AI provided practical experience with:

```text
Python
Machine Learning
Deep Learning
TensorFlow
Keras
LSTM
Bidirectional LSTM
Time-Series Forecasting
Feature Engineering
Technical Analysis
Model Evaluation
Walk-Forward Testing
FastAPI
REST APIs
React
Vite
Tailwind CSS
Recharts
Yahoo Finance
Git
GitHub
Full-Stack AI Development
```

---

# 👨‍💻 Author

## Aryan Dubey

**B.Tech Artificial Intelligence & Machine Learning Student**

### Aspiring AI Engineer | Building Real-World AI Solutions

Interested in:

```text
Artificial Intelligence
Machine Learning
Deep Learning
Computer Vision
Natural Language Processing
Large Language Models
Time-Series Forecasting
Generative AI
AI Engineering
Full-Stack AI Applications
```

---

# 🔗 GitHub

```text
https://github.com/Aryan2624
```

---

# 📂 StockVision AI Repository

```text
https://github.com/Aryan2624/StockVision-AI
```

---

# ⭐ Support StockVision AI

If you find this project interesting or useful, consider giving the repository a **⭐ Star**.

It supports the project and motivates future improvements.

---

<div align="center">

# 📈 StockVision AI

### AI-Powered Stock Forecasting & Market Intelligence

**React • FastAPI • Python • TensorFlow • Bi-LSTM • Tailwind CSS**

### Turning Real Market Data Into AI-Powered Insights

⭐ **Star the repository if you like StockVision AI**

</div>
