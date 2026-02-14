"""
Portfolio optimization using Modern Portfolio Theory
Fixed to use ONLY training data (no lookahead bias)
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = 'data/raw'
RESULTS_DIR = 'results'

def load_portfolio_data(tickers):
    """Load returns for all stocks (TRAINING DATA ONLY)"""
    returns_df = pd.DataFrame()

    for ticker in tickers:
        df = pd.read_csv(f"{DATA_DIR}/{ticker}.csv", index_col=0, parse_dates=True)

        # ===== FIX: Use only training data (80%) =====
        train_size = int(len(df) * 0.8)
        df_train = df[:train_size]
        # ===== END FIX =====

        returns = df_train['Close'].pct_change()
        returns_df[ticker] = returns

    returns_df = returns_df.dropna()
    return returns_df


def calculate_portfolio_metrics(weights, mean_returns, cov_matrix):
    """
    Calculate portfolio return and risk
    """
    # Portfolio return: w^T * r
    portfolio_return = np.dot(weights, mean_returns)

    # Portfolio variance: w^T * Σ * w
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_std = np.sqrt(portfolio_variance)

    # Annualize (252 trading days)
    annual_return = portfolio_return * 252
    annual_std = portfolio_std * np.sqrt(252)

    return annual_return, annual_std

def calculate_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """
    Sharpe Ratio = (Return - Risk_free_rate) / Volatility
    """
    annual_return, annual_std = calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
    sharpe = (annual_return - risk_free_rate) / annual_std
    return sharpe

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Negative Sharpe for minimization"""
    return -calculate_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.02):
    """
    Find optimal portfolio weights (maximize Sharpe ratio)

    Constraints:
    - Sum of weights = 1 (fully invested)
    - All weights >= 0 (no short selling)
    """
    num_assets = len(mean_returns)

    # Initial guess: equal weights
    initial_weights = np.array([1/num_assets] * num_assets)

    # Constraints
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # sum = 1
    )

    # Bounds: 0 <= weight <= 1
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Optimize
    result = minimize(
        negative_sharpe,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x

def generate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=5000):
    """
    Generate random portfolios to plot efficient frontier
    """
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))  # return, std, sharpe
    weights_record = []

    for i in range(num_portfolios):
        # Random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # normalize to sum = 1

        weights_record.append(weights)

        # Calculate metrics
        portfolio_return, portfolio_std = calculate_portfolio_metrics(
            weights, mean_returns, cov_matrix
        )
        sharpe = (portfolio_return - 0.02) / portfolio_std

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = sharpe

    return results, weights_record

def analyze_portfolio(tickers):
    """Main portfolio analysis"""
    print("="*60)
    print("PORTFOLIO OPTIMIZATION (TRAINING DATA ONLY)")
    print("="*60)
    print(f"Analyzing: {', '.join(tickers)}\n")

    # Load data
    returns_df = load_portfolio_data(tickers)
    print(f"Data: {returns_df.shape[0]} days (80% of full dataset)")

    # Calculate statistics
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    print("\nAnnualized Expected Returns:")
    for ticker, ret in (mean_returns * 252).items():
        print(f"  {ticker}: {ret:.2%}")

    print("\nCovariance Matrix:")
    print(cov_matrix)

    # Correlation matrix (easier to interpret)
    corr_matrix = returns_df.corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix)

    # Find optimal portfolio
    optimal_weights = optimize_portfolio(mean_returns, cov_matrix)

    print("\n" + "="*60)
    print("OPTIMAL PORTFOLIO (Maximum Sharpe Ratio)")
    print("="*60)
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"  {ticker}: {weight:.2%}")

    # Calculate optimal portfolio metrics
    opt_return, opt_std = calculate_portfolio_metrics(optimal_weights, mean_returns, cov_matrix)
    opt_sharpe = calculate_sharpe_ratio(optimal_weights, mean_returns, cov_matrix)

    print(f"\nExpected Annual Return: {opt_return:.2%}")
    print(f"Annual Volatility: {opt_std:.2%}")
    print(f"Sharpe Ratio: {opt_sharpe:.3f}")

    # Equal-weight portfolio (benchmark)
    equal_weights = np.array([1/len(tickers)] * len(tickers))
    eq_return, eq_std = calculate_portfolio_metrics(equal_weights, mean_returns, cov_matrix)
    eq_sharpe = calculate_sharpe_ratio(equal_weights, mean_returns, cov_matrix)

    print("\n" + "="*60)
    print("EQUAL-WEIGHT PORTFOLIO (Benchmark)")
    print("="*60)
    print(f"Expected Annual Return: {eq_return:.2%}")
    print(f"Annual Volatility: {eq_std:.2%}")
    print(f"Sharpe Ratio: {eq_sharpe:.3f}")

    # Generate efficient frontier
    print("\nGenerating efficient frontier...")
    results, weights_record = generate_efficient_frontier(mean_returns, cov_matrix)

    # Visualizations
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Plot 1: Efficient Frontier
    plt.figure(figsize=(12, 8))
    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', alpha=0.5, s=10)
    plt.colorbar(label='Sharpe Ratio')

    # Mark optimal portfolio
    plt.scatter(opt_std, opt_return, marker='*', color='red', s=500,
                edgecolors='black', linewidth=2, label='Optimal Portfolio')

    # Mark equal-weight portfolio
    plt.scatter(eq_std, eq_return, marker='o', color='blue', s=200,
                edgecolors='black', linewidth=2, label='Equal Weight')

    plt.xlabel('Annual Volatility (Risk)')
    plt.ylabel('Annual Return')
    plt.title('Efficient Frontier: Risk vs Return (Training Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/efficient_frontier.png", dpi=150)
    print(f"✓ Saved: {RESULTS_DIR}/efficient_frontier.png")

    # Plot 2: Optimal Weights
    plt.figure(figsize=(10, 6))
    plt.bar(tickers, optimal_weights, color='steelblue', edgecolor='black')
    plt.xlabel('Stock')
    plt.ylabel('Weight')
    plt.title('Optimal Portfolio Allocation')
    plt.ylim(0, max(optimal_weights) * 1.2)
    for i, (ticker, weight) in enumerate(zip(tickers, optimal_weights)):
        plt.text(i, weight + 0.01, f'{weight:.1%}', ha='center', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/optimal_weights.png", dpi=150)
    print(f"✓ Saved: {RESULTS_DIR}/optimal_weights.png")

    # Plot 3: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('Stock Correlation Matrix (Training Data)')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/correlation_heatmap.png", dpi=150)
    print(f"✓ Saved: {RESULTS_DIR}/correlation_heatmap.png")

    # Save results
    results_df = pd.DataFrame({
        'Ticker': tickers,
        'Optimal_Weight': optimal_weights,
        'Expected_Annual_Return': mean_returns * 252,
        'Annual_Volatility': np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
    })
    results_df.to_csv(f"{RESULTS_DIR}/portfolio_results.csv", index=False)
    print(f"✓ Saved: {RESULTS_DIR}/portfolio_results.csv")

    return optimal_weights, mean_returns, cov_matrix

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    optimal_weights, mean_returns, cov_matrix = analyze_portfolio(tickers)

    print("\n✓ Portfolio optimization complete!")
