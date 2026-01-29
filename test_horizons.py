import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

df = pd.read_parquet('cleaned_data/featured_AAPL.parquet')

feature_cols = [
    'spread_bps', 'mid_price_usd', 'spread_pct',
    'imbalance_0', 'imbalance_1', 'imbalance_2', 'cumulative_imbalance',
    'momentum_10', 'momentum_20', 'momentum_50',
    'volatility_10', 'volatility_20', 'volatility_50',
    'total_volume_top'
]

X = df[feature_cols].values
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]

# Scale once
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=== Testing All Horizons ===\n")

for horizon in ['return_50ms', 'return_100ms', 'return_200ms', 'return_500ms']:
    y = df[horizon].values
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    train_r2 = model.score(X_train_scaled, y_train)
    test_r2 = model.score(X_test_scaled, y_test)
    
    print(f"{horizon:15s}: Train R²={train_r2*100:.4f}%  Test R²={test_r2*100:.4f}%")