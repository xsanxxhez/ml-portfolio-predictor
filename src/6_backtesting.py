"""
Backtest ML trading strategy with realistic constraints
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

DATA_DIR = 'data/processed'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

def load_model(ticker, model_name='xgboost'):
    """Load trained model"""
    filename = f"{MODEL_DIR}/{ticker}_{model_name}.pkl"
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def backtest_single_stock(ticker, initial_capital=10000, transaction_cost=0.001):
    """
    Backtest ML strategy on single stock
    """
    print(f"\n{'='*60}")
    print(f"BACKTESTING {ticker}")
    print('='*60)

    # Load model
    model_data = load_model(ticker)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']

    # Load data
    df = pd.read_csv(f"{DATA_DIR}/{ticker}_features.csv", index_col=0, parse_dates=True)

    # Train/test split (same as training)
    split = int(0.8 * len(df))
    test_df = df.iloc[split:].copy()

    # Prepare features
    X_test = test_df[feature_cols]
    X_test_scaled = scaler.transform(X_test)

    # Get predictions
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    test_df['prediction'] = predictions
    test_df['probability'] = probabilities

    # Simulate trading
    capital = initial_capital
    position = 0  # 0 = cash, 1 = holding stock
    shares = 0
    equity_curve = [initial_capital]
    trades = []

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        current_price = row['Close']

        # Buy signal (predict up with high confidence)
        if row['prediction'] == 1 and row['probability'] > 0.6 and position == 0:
            # Buy
            cost = capital * (1 - transaction_cost)
            shares = cost / current_price
            position = 1
            capital = 0
            trades.append({
                'date': row.name,
                'action': 'BUY',
                'price': current_price,
                'shares': shares,
                'value': shares * current_price
            })

        # Sell signal (predict down or low confidence)
        elif (row['prediction'] == 0 or row['probability'] < 0.4) and position == 1:
            # Sell
            capital = shares * current_price * (1 - transaction_cost)
            trades.append({
                'date': row.name,
                'action': 'SELL',
                'price': current_price,
                'shares': shares,
                'value': capital
            })
            shares = 0
            position = 0

        # Update equity
        if position == 1:
            equity = shares * current_price
        else:
            equity = capital

        equity_curve.append(equity)

    # Close position if still holding at end
    if position == 1:
        capital = shares * test_df.iloc[-1]['Close'] * (1 - transaction_cost)
        equity_curve[-1] = capital

    equity_curve = np.array(equity_curve[:-1])  # remove last extra element

    # Calculate metrics
    total_return = (equity_curve[-1] - initial_capital) / initial_capital

    # Buy & Hold benchmark
    buy_hold_returns = test_df['returns'].fillna(0)
    buy_hold_equity = initial_capital * (1 + buy_hold_returns).cumprod()
    buy_hold_return = (buy_hold_equity.iloc[-1] - initial_capital) / initial_capital

    # Sharpe Ratio
    strategy_returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = (np.mean(strategy_returns) * 252) / (np.std(strategy_returns) * np.sqrt(252))

    # Max Drawdown
    cumulative_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdown)

    # Win rate
    winning_trades = sum(1 for t in trades[1::2] if trades[trades.index(t)-1]['price'] < t['price'])
    win_rate = winning_trades / (len(trades) / 2) if len(trades) > 0 else 0

    print(f"\nBacktest Results:")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Final Value: ${equity_curve[-1]:,.2f}")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"  Outperformance: {(total_return - buy_hold_return):.2%}")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Number of Trades: {len(trades)}")
    print(f"  Win Rate: {win_rate:.1%}")

    return {
        'ticker': ticker,
        'equity_curve': equity_curve,
        'buy_hold_equity': buy_hold_equity.values,
        'trades': trades,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'test_dates': test_df.index
    }

def plot_backtest_results(results):
    """Visualize backtest results"""
    ticker = results['ticker']

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Equity curves
    axes[0].plot(results['test_dates'], results['equity_curve'],
                 label='ML Strategy', linewidth=2, color='green')
    axes[0].plot(results['test_dates'], results['buy_hold_equity'],
                 label='Buy & Hold', linewidth=2, alpha=0.7, color='blue')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].set_title(f'{ticker} - Strategy Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mark trades
    for trade in results['trades']:
        color = 'green' if trade['action'] == 'BUY' else 'red'
        marker = '^' if trade['action'] == 'BUY' else 'v'
        axes[0].scatter(trade['date'], trade['value'],
                        color=color, marker=marker, s=100, zorder=5)

    # Plot 2: Drawdown
    cumulative_max = np.maximum.accumulate(results['equity_curve'])
    drawdown = (results['equity_curve'] - cumulative_max) / cumulative_max * 100
    axes[1].fill_between(results['test_dates'], drawdown, 0,
                         color='red', alpha=0.3)
    axes[1].plot(results['test_dates'], drawdown, color='red', linewidth=1)
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_title('Drawdown Over Time')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Returns distribution
    strategy_returns = np.diff(results['equity_curve']) / results['equity_curve'][:-1] * 100
    axes[2].hist(strategy_returns, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[2].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Daily Return (%)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Daily Returns Distribution')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{ticker}_backtest.png", dpi=150)
    print(f"✓ Saved: {RESULTS_DIR}/{ticker}_backtest.png")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    all_results = []

    for ticker in tickers:
        try:
            results = backtest_single_stock(ticker, initial_capital=10000)
            plot_backtest_results(results)
            all_results.append(results)
        except Exception as e:
            print(f"✗ Error backtesting {ticker}: {e}")

    # Summary comparison table
    print(f"\n{'='*70}")
    print("BACKTEST SUMMARY - ALL STOCKS")
    print('='*70)
    print(f"{'Ticker':<8} {'Return':<12} {'Buy&Hold':<12} {'Sharpe':<10} {'Max DD':<10} {'Trades':<8} {'Win%':<8}")
    print('-'*70)

    for r in all_results:
        print(f"{r['ticker']:<8} "
              f"{r['total_return']:>10.1%}  "
              f"{r['buy_hold_return']:>10.1%}  "
              f"{r['sharpe']:>8.2f}  "
              f"{r['max_drawdown']:>8.1%}  "
              f"{len(r['trades']):>6}  "
              f"{r['win_rate']:>6.1%}")

    # Find best performer
    best_stock = max(all_results, key=lambda x: x['total_return'])
    print(f"\n🏆 Best Performer: {best_stock['ticker']} ({best_stock['total_return']:.1%} return)")

    print("\n✓ All backtesting complete!")

