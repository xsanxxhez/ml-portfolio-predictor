"""
Train traditional ML models (Random Forest, XGBoost) with TIME SERIES CV
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle
import os

DATA_DIR = 'data/processed'
MODEL_DIR = 'models'

def load_stock_data(ticker):
    """Load processed data"""
    df = pd.read_csv(f"{DATA_DIR}/{ticker}_features.csv", index_col=0, parse_dates=True)
    return df

def prepare_ml_data(df):
    """Prepare features and target"""
    exclude_cols = ['target', 'Close', 'Open', 'High', 'Low', 'Adj Close', 'returns', 'log_returns', 'Volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df['target']

    # Time series split (no random shuffle!)
    split_point = int(len(df) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler, X_train, y_train

def cross_validate_model(model, X, y, n_splits=5):
    """
    Time Series Cross-Validation
    Walk-forward validation: always train on past, test on future

    X: DataFrame (raw features before scaling)
    y: Series (target)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    print(f"  Performing {n_splits}-fold time series cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        # ===== FIX: Use .iloc for DataFrame indexing =====
        X_train_cv = X.iloc[train_idx]
        X_val_cv = X.iloc[val_idx]
        y_train_cv = y.iloc[train_idx]
        y_val_cv = y.iloc[val_idx]
        # ===== END FIX =====

        # Scale (important: fit on train, transform on val)
        scaler = StandardScaler()
        X_train_cv_scaled = scaler.fit_transform(X_train_cv)
        X_val_cv_scaled = scaler.transform(X_val_cv)

        # Train and evaluate
        model.fit(X_train_cv_scaled, y_train_cv)
        val_acc = model.score(X_val_cv_scaled, y_val_cv)
        cv_scores.append(val_acc)

        print(f"    Fold {fold}: Accuracy = {val_acc:.3f}")

    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)

    print(f"  Mean CV Score: {mean_score:.3f} (+/- {std_score:.3f})")

    return cv_scores, mean_score, std_score

def train_random_forest(X_train, y_train, X_test, y_test, X_train_raw, y_train_raw):
    """Train Random Forest with CV"""
    print("\nTraining Random Forest...")

    # Cross-validation BEFORE final training
    rf_cv = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    cv_scores, mean_cv, std_cv = cross_validate_model(rf_cv, X_train_raw, y_train_raw, n_splits=5)

    # Final model on full training set (with regularization)
    print("  Training final model on full training set...")
    rf = RandomForestClassifier(
        n_estimators=50,      # Fewer trees (was 100)
        max_depth=5,          # Shallower (was 10)
        min_samples_split=50, # More data per split (was 20)
        min_samples_leaf=20,  # Require more samples in leaf
        max_features='sqrt',  # Use fewer features per tree
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Evaluate
    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)

    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    print(f"  Overfitting Gap: {(train_acc - test_acc):.3f}")

    return rf, cv_scores

def train_xgboost(X_train, y_train, X_test, y_test, X_train_raw, y_train_raw):
    """Train XGBoost with CV"""
    print("\nTraining XGBoost...")

    # Cross-validation
    xgb_cv = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    cv_scores, mean_cv, std_cv = cross_validate_model(xgb_cv, X_train_raw, y_train_raw, n_splits=5)

    # Final model (with regularization)
    print("  Training final model on full training set...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=50,      # Fewer trees (was 100)
        max_depth=3,          # Shallower (was 5)
        learning_rate=0.05,   # Slower (was 0.1)
        subsample=0.8,        # Use 80% of data per tree
        colsample_bytree=0.8, # Use 80% of features per tree
        reg_alpha=0.1,        # L1 regularization
        reg_lambda=1.0,       # L2 regularization
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    # Evaluate
    train_acc = xgb_model.score(X_train, y_train)
    test_acc = xgb_model.score(X_test, y_test)

    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    print(f"  Overfitting Gap: {(train_acc - test_acc):.3f}")

    return xgb_model, cv_scores

def save_model(model, scaler, feature_cols, ticker, model_name, cv_scores=None):
    """Save trained model"""
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'cv_scores': cv_scores
    }

    filename = f"{MODEL_DIR}/{ticker}_{model_name}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Model saved: {filename}")

if __name__ == "__main__":
    print("=" * 50)
    print("MODEL TRAINING WITH CROSS-VALIDATION")
    print("=" * 50)

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

    all_cv_results = {}

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"TRAINING MODELS FOR {ticker}")
        print('='*50)

        try:
            # Load data
            df = load_stock_data(ticker)
            print(f"Loaded {ticker}: {df.shape}")

            # Prepare
            X_train, X_test, y_train, y_test, feature_cols, scaler, X_train_raw, y_train_raw = prepare_ml_data(df)
            print(f"Train: {X_train.shape}, Test: {X_test.shape}")

            # Train models with CV
            rf_model, rf_cv_scores = train_random_forest(X_train, y_train, X_test, y_test, X_train_raw, y_train_raw)
            xgb_model, xgb_cv_scores = train_xgboost(X_train, y_train, X_test, y_test, X_train_raw, y_train_raw)

            # Save
            save_model(rf_model, scaler, feature_cols, ticker, 'random_forest', rf_cv_scores)
            save_model(xgb_model, scaler, feature_cols, ticker, 'xgboost', xgb_cv_scores)

            # Store CV results
            all_cv_results[ticker] = {
                'rf_cv': rf_cv_scores,
                'xgb_cv': xgb_cv_scores
            }

        except Exception as e:
            print(f"✗ Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print('='*70)
    print(f"{'Ticker':<8} {'RF CV Mean':<12} {'RF CV Std':<12} {'XGB CV Mean':<12} {'XGB CV Std':<12}")
    print('-'*70)

    for ticker, results in all_cv_results.items():
        rf_mean = np.mean(results['rf_cv'])
        rf_std = np.std(results['rf_cv'])
        xgb_mean = np.mean(results['xgb_cv'])
        xgb_std = np.std(results['xgb_cv'])

        print(f"{ticker:<8} {rf_mean:>10.3f}  {rf_std:>10.3f}  {xgb_mean:>10.3f}  {xgb_std:>10.3f}")

    print("\n✓ All training complete!")
