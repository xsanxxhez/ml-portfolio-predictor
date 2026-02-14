"""
Calculate technical indicators and features for each stock
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = 'data/raw'
OUTPUT_DIR = 'data/processed'

def calculate_returns(df):
    """Calculate returns"""
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    return df

def calculate_technical_indicators(df):
    """Add technical indicators"""
    close = df['Close']

    # Moving averages
    df['SMA_10'] = close.rolling(window=10).mean()
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = close.ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_middle'] = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (2 * bb_std)
    df['BB_lower'] = df['BB_middle'] - (2 * bb_std)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

    # Volatility
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_annual'] = df['volatility_20'] * np.sqrt(252)

    # Volume features
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

    # Price momentum
    df['momentum_5'] = close / close.shift(5) - 1
    df['momentum_10'] = close / close.shift(10) - 1
    df['momentum_20'] = close / close.shift(20) - 1

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)

    return df

def create_target(df, horizon=1):
    """Create target variable: 1 if price goes up, 0 if down"""
    df['target'] = (df['returns'].shift(-horizon) > 0).astype(int)
    return df

def process_stock(ticker):
    """Process single stock"""
    print(f"Processing {ticker}...")

    # Load
    df = pd.read_csv(f"{DATA_DIR}/{ticker}.csv", index_col=0, parse_dates=True)

    # ===== FIX: Convert price columns to numeric =====
    price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows with NaN in price columns (from conversion)
    df = df.dropna(subset=['Close'])
    # ===== END FIX =====

    # Calculate features
    df = calculate_returns(df)
    df = calculate_technical_indicators(df)
    df = create_target(df)

    # Drop NaN
    df = df.dropna()

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(f"{OUTPUT_DIR}/{ticker}_features.csv")

    print(f"✓ {ticker}: {df.shape[0]} rows, {df.shape[1]} features")
    return df


if __name__ == "__main__":
    print("=" * 50)
    print("FEATURE ENGINEERING")
    print("=" * 50)

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

    for ticker in tickers:
        process_stock(ticker)

    print("\n✓ Feature engineering complete!")
