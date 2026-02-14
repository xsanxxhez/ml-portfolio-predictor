"""
Create comprehensive ML trading dashboard
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

DATA_DIR = 'data/processed'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

def create_ml_dashboard(ticker='AAPL'):
    """Generate complete dashboard"""
    print(f"Creating dashboard for {ticker}...")

    # Load model
    with open(f"{MODEL_DIR}/{ticker}_xgboost.pkl", 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']

    # Load data
    df = pd.read_csv(f"{DATA_DIR}/{ticker}_features.csv", index_col=0, parse_dates=True)

    # Create figure with 8 subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # ========== Plot 1: Price & Moving Averages ==========
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1.5, color='black')
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
    ax1.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)
    ax1.fill_between(df.index, df['BB_lower'], df['BB_upper'], alpha=0.1, label='Bollinger Bands')
    ax1.set_title(f'{ticker} - Price & Technical Indicators', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ========== Plot 2: Volume ==========
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(df.index, df['Volume'], color='steelblue', alpha=0.6)
    ax2.set_title('Trading Volume', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3, axis='y')

    # ========== Plot 3: RSI ==========
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df.index, df['RSI'], color='purple', linewidth=1.5)
    ax3.axhline(70, color='red', linestyle='--', label='Overbought')
    ax3.axhline(30, color='green', linestyle='--', label='Oversold')
    ax3.fill_between(df.index, 30, 70, alpha=0.1)
    ax3.set_title('RSI (Relative Strength Index)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ========== Plot 4: MACD ==========
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df.index, df['MACD'], label='MACD', linewidth=1.5)
    ax4.plot(df.index, df['MACD_signal'], label='Signal', linewidth=1.5)
    ax4.bar(df.index, df['MACD_hist'], label='Histogram', alpha=0.3)
    ax4.set_title('MACD', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ========== Plot 5: Volatility ==========
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(df.index, df['volatility_annual'], color='red', linewidth=1.5)
    ax5.set_title('Annualized Volatility', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Volatility')
    ax5.grid(True, alpha=0.3)

    # ========== Plot 6: Feature Importance ==========
    ax6 = fig.add_subplot(gs[2, :2])
    feature_importance = pd.DataFrame({
        'feature': model_data['feature_cols'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    ax6.barh(feature_importance['feature'], feature_importance['importance'], color='teal', edgecolor='black')
    ax6.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Importance')
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.invert_yaxis()

    # ========== Plot 7: Returns Distribution ==========
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.hist(df['returns'].dropna() * 100, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax7.axvline(0, color='red', linestyle='--', linewidth=2)
    ax7.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Return (%)')
    ax7.set_ylabel('Frequency')
    ax7.grid(True, alpha=0.3, axis='y')

    # ========== Plot 8: Correlation Heatmap ==========
    ax8 = fig.add_subplot(gs[3, :2])
    corr_features = ['Close', 'Volume', 'RSI', 'MACD', 'volatility_20', 'momentum_10', 'SMA_20', 'SMA_50']
    corr_matrix = df[corr_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, ax=ax8, cbar_kws={"shrink": 0.8})
    ax8.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

    # ========== Plot 9: Summary Stats ==========
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.axis('off')

    # Calculate stats
    total_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
    annual_vol = df['volatility_annual'].iloc[-1]
    sharpe = (df['returns'].mean() * 252) / (df['returns'].std() * np.sqrt(252))
    max_price = df['Close'].max()
    min_price = df['Close'].min()
    avg_volume = df['Volume'].mean()

    stats_text = f"""
    SUMMARY STATISTICS
    
    Total Return: {total_return:.1%}
    Annual Volatility: {annual_vol:.1%}
    Sharpe Ratio: {sharpe:.2f}
    
    Max Price: ${max_price:.2f}
    Min Price: ${min_price:.2f}
    Current Price: ${df['Close'].iloc[-1]:.2f}
    
    Avg Daily Volume: {avg_volume/1e6:.1f}M
    
    Data Period:
    {df.index[0].date()} to
    {df.index[-1].date()}
    
    Total Days: {len(df)}
    """

    ax9.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Overall title
    fig.suptitle(f'{ticker} - ML TRADING SYSTEM DASHBOARD',
                 fontsize=18, fontweight='bold', y=0.995)

    # Save
    plt.savefig(f"{RESULTS_DIR}/{ticker}_dashboard.png", dpi=200, bbox_inches='tight')
    print(f"✓ Dashboard saved: {RESULTS_DIR}/{ticker}_dashboard.png")

    plt.show()

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    for ticker in tickers:
        print(f"\nCreating dashboard for {ticker}...")
        try:
            create_ml_dashboard(ticker)
        except Exception as e:
            print(f"✗ Error creating dashboard for {ticker}: {e}")

    print("\n✓ All dashboards complete!")



