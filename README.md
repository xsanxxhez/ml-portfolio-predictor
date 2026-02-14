# ML Portfolio Predictor 

**End-to-end machine learning system for stock portfolio optimization and algorithmic trading**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![Efficient Frontier](results/efficient_frontier.png)

---

## Overview

This project combines **machine learning** and **Modern Portfolio Theory** to predict stock movements and optimize portfolio allocation. Built as a learning project to explore quantitative finance, ML engineering, and algorithmic trading concepts.

**Key Features:**
- Predicts daily stock movements using Random Forest, XGBoost, and LSTM neural networks
- Optimizes portfolio weights to maximize risk-adjusted returns (Sharpe ratio)
- Backtests strategies with realistic transaction costs and position sizing
- Uses time-series cross-validation to prevent overfitting

---

## Performance Summary

### Portfolio Optimization
| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | **1.206** |
| **Annual Return** | **32.84%** |
| **Annual Volatility** | **25.56%** |
| **Optimal Allocation** | MSFT 31% \| AMZN 46% \| AAPL 23% |

*Calculated on 9 years of training data (2015-2022) using covariance optimization*

---

### Individual Stock Performance (2022-2023 Test Period)

| Stock | Total Return | Sharpe Ratio | Win Rate | Max Drawdown | Trades |
|-------|--------------|--------------|----------|--------------|--------|
| **MSFT** | **+28.2%** | **0.88** | 72.7% | -12.6% | 44 |
| **AAPL** | **+19.6%** | **0.79** | 50.0% | -11.9% | 36 |
| **GOOGL** | +2.6% | 0.20 | 48.4% | -34.6% | 62 |
| **AMZN** | -0.7% | 0.17 | 46.5% | -38.3% | 43 |

**Key Insight:** Models performed best on MSFT and AAPL (lower volatility), struggled with GOOGL and AMZN (high volatility growth stocks). This demonstrates the importance of stock selection and shows the system doesn't blindly predict—it captures genuine patterns where they exist.

---

## Tech Stack

**Machine Learning:**
- `scikit-learn` - Random Forest, preprocessing, cross-validation
- `XGBoost` - Gradient boosting classifier
- `PyTorch` - LSTM neural networks

**Finance & Data:**
- `yfinance` - Historical stock data from Yahoo Finance
- `pandas` / `numpy` - Data manipulation and analysis
- `scipy` - Portfolio optimization (covariance matrix, Sharpe maximization)

**Visualization:**
- `matplotlib` / `seaborn` - Charts and dashboards

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ml-portfolio-predictor.git
cd ml-portfolio-predictor

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
