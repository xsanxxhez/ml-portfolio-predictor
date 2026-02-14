
#  ML Portfolio Predictor

> **End-to-end machine learning system for algorithmic stock trading with portfolio optimization**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)]()

![Banner](results/efficient_frontier.png)

---

## **Results at a Glance**

| Metric | Value |
|--------|-------|
| **Total Return** | **+300.4%** ($10,000 → $40,043) |
| **Sharpe Ratio** | **4.1** (institutional grade) |
| **Max Drawdown** | **-14.7%** (controlled risk) |
| **Win Rate** | **70-89%** (consistent profitability) |
| **Backtest Period** | 9 years (2015-2023) |
| **Outperformance vs Buy & Hold** | **+220% to +421%** per stock |

---

## **What Does This Project Do?**

This system combines **machine learning**, **quantitative finance**, and **portfolio optimization** to predict stock price movements and generate trading signals.

### **Key Features:**
- **Automated data pipeline** (Yahoo Finance API)
- **34 technical indicators** (RSI, MACD, Bollinger Bands, momentum, volatility)
- **3 ML models** (Random Forest, XGBoost, LSTM neural networks)
- **Modern Portfolio Theory** optimization (Sharpe ratio maximization)
- **Realistic backtesting** (transaction costs, walk-forward validation)
- **Professional visualizations** (dashboards, charts, heatmaps)
- **One-command deployment** (`./run.sh`)

---

## **Quick Start**

### **1. Clone Repository**
```bash
git clone https://github.com/xsanxxhez/ml-portfolio-predictor.git
cd ml-portfolio-predictor
