#!/bin/bash

# ML Portfolio Predictor - Complete Pipeline Runner
# Author: Your Name
# Date: 2026-02-14

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="/Users/xsan/Downloads/ml-portfolio-predictor"
cd "$PROJECT_DIR"

# Activate virtual environment
source .venv/bin/activate

echo ""
echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                                                            ║${NC}"
echo -e "${MAGENTA}║          ML PORTFOLIO PREDICTOR - FULL PIPELINE            ║${NC}"
echo -e "${MAGENTA}║                                                            ║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print section header
print_section() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Function to print step
print_step() {
    echo -e "${BLUE}▶ $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Step 1: Data Collection
print_section "STEP 1: DATA COLLECTION"
print_step "Downloading stock data from Yahoo Finance..."
python src/1_data_collection.py
if [ $? -eq 0 ]; then
    print_success "Data collection complete!"
else
    print_error "Data collection failed!"
    exit 1
fi

# Step 2: Feature Engineering
print_section "STEP 2: FEATURE ENGINEERING"
print_step "Calculating technical indicators (RSI, MACD, Bollinger Bands, etc.)..."
python src/2_feature_engineering.py
if [ $? -eq 0 ]; then
    print_success "Feature engineering complete!"
    echo -e "${YELLOW}  → Generated 34 features per stock${NC}"
else
    print_error "Feature engineering failed!"
    exit 1
fi

# Step 3: Model Training
print_section "STEP 3: ML MODEL TRAINING"
print_step "Training Random Forest and XGBoost models for all stocks..."
python src/3_train_models.py
if [ $? -eq 0 ]; then
    print_success "Model training complete!"
    echo -e "${YELLOW}  → Trained 2 models per stock (RF + XGBoost)${NC}"
else
    print_error "Model training failed!"
    exit 1
fi

# Step 4: LSTM Training (Optional - takes longer)
read -p "$(echo -e ${YELLOW}Train LSTM neural networks? [y/N]: ${NC})" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_section "STEP 4: LSTM NEURAL NETWORK TRAINING"
    print_step "Training LSTM models (this may take a few minutes)..."
    python src/4_lstm_model.py
    if [ $? -eq 0 ]; then
        print_success "LSTM training complete!"
    else
        print_error "LSTM training failed (continuing anyway)..."
    fi
else
    echo -e "${YELLOW}Skipping LSTM training${NC}"
fi

# Step 5: Portfolio Optimization
print_section "STEP 5: PORTFOLIO OPTIMIZATION"
print_step "Optimizing portfolio using Modern Portfolio Theory..."
python src/5_portfolio_optimization.py
if [ $? -eq 0 ]; then
    print_success "Portfolio optimization complete!"
    echo -e "${YELLOW}  → Maximized Sharpe ratio using covariance matrix${NC}"
else
    print_error "Portfolio optimization failed!"
    exit 1
fi

# Step 6: Backtesting
print_section "STEP 6: STRATEGY BACKTESTING"
print_step "Backtesting ML trading strategy on all stocks..."
python src/6_backtesting.py
if [ $? -eq 0 ]; then
    print_success "Backtesting complete!"
else
    print_error "Backtesting failed!"
    exit 1
fi

# Step 7: Dashboard Creation
print_section "STEP 7: DASHBOARD GENERATION"
print_step "Creating visualizations and dashboards..."
python src/7_dashboard.py
if [ $? -eq 0 ]; then
    print_success "Dashboard creation complete!"
else
    print_error "Dashboard creation failed!"
    exit 1
fi

# Summary
print_section "PIPELINE COMPLETE! 🎉"

echo ""
echo -e "${GREEN}All steps completed successfully!${NC}"
echo ""
echo -e "${CYAN}Generated Files:${NC}"
echo -e "  ${YELLOW}Data:${NC}"
echo "    • data/raw/*.csv (5 stock CSVs)"
echo "    • data/processed/*_features.csv (34 features per stock)"
echo ""
echo -e "  ${YELLOW}Models:${NC}"
echo "    • models/*_random_forest.pkl"
echo "    • models/*_xgboost.pkl"
if [ -f "models/AAPL_lstm.pt" ]; then
    echo "    • models/*_lstm.pt"
fi
echo ""
echo -e "  ${YELLOW}Results:${NC}"
echo "    • results/efficient_frontier.png"
echo "    • results/optimal_weights.png"
echo "    • results/correlation_heatmap.png"
echo "    • results/*_backtest.png (4 stocks)"
echo "    • results/*_dashboard.png (4 stocks)"
echo "    • results/portfolio_results.csv"
echo ""

# Display key results
if [ -f "results/portfolio_results.csv" ]; then
    echo -e "${CYAN}Portfolio Summary:${NC}"
    echo ""
    cat results/portfolio_results.csv | column -t -s,
    echo ""
fi

# Open results folder
echo -e "${BLUE}Opening results folder...${NC}"
open results/ 2>/dev/null || xdg-open results/ 2>/dev/null || echo "Results saved in: results/"

echo ""
echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                   NEXT STEPS                               ║${NC}"
echo -e "${MAGENTA}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${MAGENTA}║  1. Review results in results/ folder                      ║${NC}"
echo -e "${MAGENTA}║  2. Open dashboards and charts                             ║${NC}"
echo -e "${MAGENTA}║  3. Analyze portfolio_results.csv                          ║${NC}"
echo -e "${MAGENTA}║  4. Prepare interview presentation                         ║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
