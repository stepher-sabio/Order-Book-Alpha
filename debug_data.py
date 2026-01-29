import pandas as pd
import numpy as np

df = pd.read_parquet('cleaned_data/featured_AAPL.parquet')

print("=== DATA SANITY CHECK ===\n")

# Check a few samples
print("Sample of data:")
print(df[['imbalance_0', 'momentum_10', 'return_50ms']].head(20))

print("\n=== Feature Statistics ===")
print(df[['imbalance_0', 'momentum_10', 'volatility_10', 'return_50ms']].describe())

print("\n=== Correlations with Target ===")
feature_cols = [
    'spread_bps', 'mid_price_usd', 'spread_pct',
    'imbalance_0', 'imbalance_1', 'imbalance_2', 'cumulative_imbalance',
    'momentum_10', 'momentum_20', 'momentum_50',
    'volatility_10', 'volatility_20', 'volatility_50',
    'total_volume_top'
]

correlations = df[feature_cols + ['return_50ms']].corr()['return_50ms'].drop('return_50ms').sort_values(ascending=False)
print(correlations)

print("\n=== Check for NaN or Inf ===")
for col in feature_cols + ['return_50ms']:
    n_nan = df[col].isna().sum()
    n_inf = np.isinf(df[col]).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"⚠️  {col}: {n_nan} NaN, {n_inf} Inf")

print("\n=== Check Feature Ranges ===")
print("Do features have reasonable ranges?")
print(df[['imbalance_0', 'cumulative_imbalance', 'momentum_10', 'spread_bps']].describe())
