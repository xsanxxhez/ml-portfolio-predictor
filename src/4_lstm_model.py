"""
Train LSTM neural network for sequence prediction (PyTorch)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

DATA_DIR = 'data/processed'
MODEL_DIR = 'models'

class StockDataset(Dataset):
    """Custom dataset for LSTM"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """LSTM Neural Network"""
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(25, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Take output from last time step
        last_output = lstm_out[:, -1, :]

        # Dense layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out

def create_sequences(data, seq_length=60):
    """Create sequences for LSTM"""
    X, y = [], []

    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :-1])  # all features except target
        y.append(data[i, -1])  # target

    return np.array(X), np.array(y)

def prepare_lstm_data(ticker, seq_length=60):
    """Prepare data for LSTM"""
    print(f"Loading data for {ticker}...")

    df = pd.read_csv(f"{DATA_DIR}/{ticker}_features.csv", index_col=0, parse_dates=True)

    # Select features
    feature_cols = ['Close', 'Volume', 'RSI', 'MACD', 'volatility_20', 'momentum_10']
    df_features = df[feature_cols + ['target']].dropna()

    print(f"Dataset shape: {df_features.shape}")

    # Scale
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features)

    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)

    print(f"Created {len(X)} sequences")

    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler

def train_lstm(ticker, epochs=30, batch_size=32, learning_rate=0.001):
    """Train LSTM model"""
    print(f"\n{'='*60}")
    print(f"TRAINING LSTM FOR {ticker}")
    print('='*60)

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(ticker)

    # Create datasets
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_size = X_train.shape[2]  # number of features
    model = LSTMModel(input_size=input_size).to(device)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Accuracy
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total

        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)

                test_loss += loss.item()

                predicted = (outputs > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        test_loss /= len(test_loader)
        test_acc = correct / total

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.3f}")

    print(f"\n✓ Training complete!")
    print(f"Final Test Accuracy: {test_acc:.3f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{MODEL_DIR}/{ticker}_lstm.pt")
    print(f"✓ Model saved: {MODEL_DIR}/{ticker}_lstm.pt")

    # Plot training history
    os.makedirs('results', exist_ok=True)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Accuracy', linewidth=2)
    plt.plot(test_accs, label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/{ticker}_lstm_training.png", dpi=150)
    print(f"✓ Training plot saved: results/{ticker}_lstm_training.png")

    return model, train_accs, test_accs

if __name__ == "__main__":
    ticker = 'AAPL'
    model, train_accs, test_accs = train_lstm(ticker, epochs=30, batch_size=32)

    print("\n" + "="*60)
    print("LSTM TRAINING COMPLETE")
    print("="*60)
