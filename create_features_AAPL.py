import pandas as pd
import numpy as np

print("=== Loading sampled & labeled data ===")
df = pd.read_parquet('cleaned_data/sampled_labeled_AAPL.parquet')
print(f"Loaded {len(df):,} samples")

# Convert prices to USD for interpretation
price_scale = 1e8
df['mid_price_usd'] = df['mid_price'] / price_scale
df['bid_px_0_usd'] = df['bid_px_0'] / price_scale
df['ask_px_0_usd'] = df['ask_px_0'] / price_scale

print("\n=== Creating 14 core features ===")

# ============================================
# Category 1: Price & Spread (3 features)
# ============================================
print("1. Price & spread features...")

df['spread_usd'] = df['ask_px_0_usd'] - df['bid_px_0_usd']
df['spread_bps'] = (df['spread_usd'] / df['mid_price_usd']) * 10000  # basis points
df['spread_pct'] = df['spread_usd'] / df['mid_price_usd']

# ============================================
# Category 2: Depth Imbalance (4 features)
# ============================================
print("2. Depth imbalance features...")

# CRITICAL FIX: Convert uint32 to float to prevent underflow
df['imbalance_0'] = (df['bid_sz_0'].astype(float) - df['ask_sz_0'].astype(float)) / \
                    (df['bid_sz_0'].astype(float) + df['ask_sz_0'].astype(float))

df['imbalance_1'] = (df['bid_sz_1'].astype(float) - df['ask_sz_1'].astype(float)) / \
                    (df['bid_sz_1'].astype(float) + df['ask_sz_1'].astype(float))

df['imbalance_2'] = (df['bid_sz_2'].astype(float) - df['ask_sz_2'].astype(float)) / \
                    (df['bid_sz_2'].astype(float) + df['ask_sz_2'].astype(float))

# Cumulative imbalance across all levels
total_bid = df['bid_sz_0'].astype(float) + df['bid_sz_1'].astype(float) + df['bid_sz_2'].astype(float)
total_ask = df['ask_sz_0'].astype(float) + df['ask_sz_1'].astype(float) + df['ask_sz_2'].astype(float)
df['cumulative_imbalance'] = (total_bid - total_ask) / (total_bid + total_ask)

# ============================================
# Category 3: Momentum (3 features)
# ============================================
print("3. Momentum features...")

df['momentum_10'] = df['log_mid'] - df['log_mid'].shift(10)   # 500ms
df['momentum_20'] = df['log_mid'] - df['log_mid'].shift(20)   # 1s
df['momentum_50'] = df['log_mid'] - df['log_mid'].shift(50)   # 2.5s

# ============================================
# Category 4: Volatility (3 features)
# ============================================
print("4. Volatility features...")

returns = df['log_mid'].diff()

df['volatility_10'] = returns.rolling(window=10).apply(lambda x: np.sum(x**2), raw=True)
df['volatility_20'] = returns.rolling(window=20).apply(lambda x: np.sum(x**2), raw=True)
df['volatility_50'] = returns.rolling(window=50).apply(lambda x: np.sum(x**2), raw=True)

# ============================================
# Category 5: Liquidity (1 feature)
# ============================================
print("5. Liquidity features...")

df['total_volume_top'] = df['bid_sz_0'] + df['ask_sz_0']

# ============================================
# Clean up and save
# ============================================
print("\n=== Cleaning up NaN values ===")
print(f"Before: {len(df):,} samples")

# Remove rows with NaN (from rolling windows and shifts)
df = df.dropna().reset_index(drop=True)

print(f"After removing NaN: {len(df):,} samples")

# ============================================
# Define feature list
# ============================================
feature_cols = [
    # Price & Spread (3)
    'spread_bps', 'mid_price_usd', 'spread_pct',
    # Depth Imbalance (4)
    'imbalance_0', 'imbalance_1', 'imbalance_2', 'cumulative_imbalance',
    # Momentum (3)
    'momentum_10', 'momentum_20', 'momentum_50',
    # Volatility (3)
    'volatility_10', 'volatility_20', 'volatility_50',
    # Liquidity (1)
    'total_volume_top'
]

# ============================================
# Feature quality checks
# ============================================
print("\n=== Feature Summary ===")
print(f"Total features: {len(feature_cols)}")

print("\nFeature statistics:")
print(df[feature_cols].describe().round(6))

print("\n=== Checking for problematic values ===")
problem_found = False
for col in feature_cols:
    n_inf = np.isinf(df[col]).sum()
    n_nan = df[col].isna().sum()
    if n_inf > 0 or n_nan > 0:
        print(f"⚠️  {col}: {n_inf} infinite, {n_nan} NaN")
        problem_found = True

if not problem_found:
    print("✅ No infinite or NaN values found!")

# ============================================
# Check feature correlations
# ============================================
print("\n=== Correlation Analysis ===")
print("Checking momentum correlations:")
mom_corr = df[['momentum_10', 'momentum_20', 'momentum_50']].corr()
print(mom_corr.round(3))

print("\nChecking volatility correlations:")
vol_corr = df[['volatility_10', 'volatility_20', 'volatility_50']].corr()
print(vol_corr.round(3))

# ============================================
# Label summary
# ============================================
print("\n=== Label Summary ===")
horizons = [50, 100, 200, 500]
for h in horizons:
    label_col = f'return_{h}ms'
    ret_bps = df[label_col] * 10000
    print(f"{h}ms: mean={ret_bps.mean():+.3f}bps, std={ret_bps.std():.3f}bps, "
          f"samples={df[label_col].notna().sum():,}")

# ============================================
# Save featured dataset
# ============================================
output_file = 'featured_AAPL.parquet'
df.to_parquet(output_file, index=False)

print(f"\n✅ Saved featured dataset to: {output_file}")
print(f"   Shape: {df.shape}")
print(f"   Features: {len(feature_cols)}")
print(f"   Samples: {len(df):,}")

print("\n" + "="*50)
print("FEATURE LIST:")
for i, feat in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {feat}")
print("="*50)

print("\n🎉 Ready for modeling!")
print("\nNext step: Train baseline model (Lasso Regression)")