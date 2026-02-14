#!/bin/bash

# ML Portfolio Predictor - Results Viewer
# Quick view of all results and metrics

PROJECT_DIR="/Users/xsan/Downloads/ml-portfolio-predictor"
cd "$PROJECT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

clear

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          ML PORTFOLIO PREDICTOR - RESULTS SUMMARY          ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Portfolio Optimization Results
if [ -f "results/portfolio_results.csv" ]; then
    echo -e "${YELLOW}═══ OPTIMAL PORTFOLIO ALLOCATION ═══${NC}"
    echo ""
    cat results/portfolio_results.csv | column -t -s, | head -6
    echo ""
fi

# Backtest Summary
echo -e "${YELLOW}═══ BACKTESTING RESULTS ═══${NC}"
echo ""

# Extract backtest summary from last run
if [ -f "backtest_log.txt" ]; then
    tail -20 backtest_log.txt | grep -A 10 "BACKTEST SUMMARY"
else
    echo "Run backtesting first: python src/6_backtesting.py"
fi

echo ""
echo -e "${YELLOW}═══ GENERATED FILES ═══${NC}"
echo ""
echo -e "${GREEN}Visualizations:${NC}"
ls -lh results/*.png 2>/dev/null | awk '{print "  • " $9 " (" $5 ")"}'

echo ""
echo -e "${GREEN}Data Files:${NC}"
ls -lh results/*.csv 2>/dev/null | awk '{print "  • " $9 " (" $5 ")"}'

echo ""
echo -e "${GREEN}Models:${NC}"
ls -lh models/*.pkl models/*.pt 2>/dev/null | awk '{print "  • " $9 " (" $5 ")"}'

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Commands:${NC}"
echo "  • View portfolio: open results/efficient_frontier.png"
echo "  • View backtests: open results/*_backtest.png"
echo "  • View dashboards: open results/*_dashboard.png"
echo "  • Open all results: open results/"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Ask what to open
read -p "Open results folder? [Y/n]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    open results/
fi
