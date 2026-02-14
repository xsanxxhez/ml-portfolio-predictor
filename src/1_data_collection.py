"""
Download historical stock data for portfolio
Fixed to save clean single-level CSV files
"""
import yfinance as yf
import pandas as pd
import os

# Configuration
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
START_DATE = '2015-01-01'
END_DATE = '2024-01-01'
DATA_DIR = 'data/raw'

def download_stock_data(tickers, start, end):
    """Download data for multiple stocks"""
    os.makedirs(DATA_DIR, exist_ok=True)

    all_data = {}

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        # Use Ticker object instead of download() to avoid multi-index
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)

        if df.empty:
            print(f"Warning: No data for {ticker}")
            continue

        # Keep only OHLCV columns (drops Dividends, Stock Splits)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Save with Date as index (clean single-level CSV)
        df.to_csv(f"{DATA_DIR}/{ticker}.csv")
        all_data[ticker] = df

        print(f"✓ {ticker}: {len(df)} days, from {df.index[0].date()} to {df.index[-1].date()}")

    return all_data

def create_combined_dataset(all_data):
    """Combine close prices into single DataFrame"""
    close_prices = pd.DataFrame()

    for ticker, df in all_data.items():
        close_prices[ticker] = df['Close']

    # Save combined
    close_prices.to_csv(f"{DATA_DIR}/combined_prices.csv")
    print(f"\n✓ Combined dataset saved: {close_prices.shape}")

    return close_prices

if __name__ == "__main__":
    print("=" * 50)
    print("STOCK DATA COLLECTION")
    print("=" * 50)

    # Download
    all_data = download_stock_data(TICKERS, START_DATE, END_DATE)

    # Combine
    combined = create_combined_dataset(all_data)

    # Quick stats
    print("\nPrice Statistics:")
    print(combined.describe())

    print("\nMissing values:")
    print(combined.isna().sum())

    print("\n✓ Data collection complete!")
