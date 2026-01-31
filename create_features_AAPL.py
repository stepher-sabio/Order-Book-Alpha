"""
create_features_AAPL.py - Enhanced Feature Engineering with Research-Backed Features
30 features total: 14 CORE + 7 ELECTIVE + 9 RESEARCH-BACKED

ORGANIZATION:
- CORE (14): Simple, essential, always keep
- ELECTIVE (7): Advanced, experimental, can swap
- RESEARCH (9): Proven in academic literature

RESEARCH CITATIONS:
1. Cont et al. (2014) - Depth asymmetry, OFI
2. Hasbrouck (2009) - Effective spread
3. Zhang et al. (2019) - Price acceleration
4. Cartea & Jaimungal (2015) - Volume dynamics

REMOVED FEATURES (2):
- microprice (Stoikov 2018) - redundant with mid_price_usd
- depth_asymmetry (Cont 2014) - redundant with cumulative_imbalance
"""

import pandas as pd
import numpy as np

print("="*70)
print("ENHANCED FEATURE ENGINEERING (30 FEATURES)")
print("14 CORE + 7 ELECTIVE + 9 RESEARCH-BACKED")
print("="*70)

print("\n=== Loading sampled & labeled data ===")
df = pd.read_parquet('cleaned_data/sampled_labeled_AAPL.parquet')
print(f"Loaded {len(df):,} samples")

# Convert prices to USD
price_scale = 1e8
df['mid_price_usd'] = df['mid_price'] / price_scale

for level in [0, 1, 2]:
    df[f'bid_px_{level}_usd'] = df[f'bid_px_{level}'] / price_scale
    df[f'ask_px_{level}_usd'] = df[f'ask_px_{level}'] / price_scale

# Calculate log returns for momentum/volatility
returns = df['log_mid'].diff().shift(1)

# ============================================
# CORE FEATURES (14) - Simple & Essential
# ============================================
print("\n" + "="*70)
print("CORE FEATURES (14)")
print("="*70)

print("\n1. Price & Spread (2 features)")
df['spread_bps'] = ((df['ask_px_0_usd'] - df['bid_px_0_usd']) / df['mid_price_usd']) * 10000
# mid_price_usd already exists

print("   ✓ spread_bps")
print("   ✓ mid_price_usd")

print("\n2. Top-of-Book Imbalance (2 features)")
df['imbalance_0'] = (
    (df['bid_sz_0'].astype(float) - df['ask_sz_0'].astype(float)) /
    (df['bid_sz_0'].astype(float) + df['ask_sz_0'].astype(float) + 1e-10)
)

total_bid = sum(df[f'bid_sz_{level}'].astype(float) for level in range(3))
total_ask = sum(df[f'ask_sz_{level}'].astype(float) for level in range(3))
df['cumulative_imbalance'] = (total_bid - total_ask) / (total_bid + total_ask + 1e-10)

print("   ✓ imbalance_0")
print("   ✓ cumulative_imbalance")

print("\n3. Short-term Momentum (3 features)")
df['momentum_5'] = df['log_mid'] - df['log_mid'].shift(5)
df['momentum_10'] = df['log_mid'] - df['log_mid'].shift(10)
df['momentum_20'] = df['log_mid'] - df['log_mid'].shift(20)

print("   ✓ momentum_5 (250ms)")
print("   ✓ momentum_10 (500ms)")
print("   ✓ momentum_20 (1s)")

print("\n4. Realized Volatility (2 features)")
df['volatility_10'] = returns.rolling(10).std()
df['volatility_20'] = returns.rolling(20).std()

print("   ✓ volatility_10 (500ms)")
print("   ✓ volatility_20 (1s)")

print("\n5. Order Flow (2 features)")
df['microprice_mom'] = df['mid_price_usd'].diff(5).shift(1)
df['flow_0'] = (
    df['bid_sz_0'].astype(float).diff().shift(1) - 
    df['ask_sz_0'].astype(float).diff().shift(1)
)

print("   ✓ microprice_mom")
print("   ✓ flow_0")

print("\n6. Book Shape (3 features)")
df['bid_slope'] = (df['bid_px_0_usd'] - df['bid_px_2_usd']) / df['mid_price_usd']
df['ask_slope'] = (df['ask_px_2_usd'] - df['ask_px_0_usd']) / df['mid_price_usd']
df['total_depth'] = total_bid + total_ask

print("   ✓ bid_slope")
print("   ✓ ask_slope")
print("   ✓ total_depth")

# ============================================
# ELECTIVE FEATURES (7) - Advanced
# ============================================
print("\n" + "="*70)
print("ELECTIVE FEATURES (7)")
print("="*70)

print("\n7. Advanced Imbalance (2 features)")
df['imbalance_change'] = df['imbalance_0'].diff(5).shift(1)
df['persistent_imbalance'] = df['imbalance_0'].shift(1).rolling(20).mean()

print("   ✓ imbalance_change")
print("   ✓ persistent_imbalance")

print("\n8. Spread Dynamics (2 features)")
df['spread_change'] = df['spread_bps'].diff(5).shift(1)
df['spread_vol_ratio'] = df['spread_bps'] / (df['volatility_20'] * 10000 + 1e-10)

print("   ✓ spread_change")
print("   ✓ spread_vol_ratio")

print("\n9. Multi-scale Patterns (1 feature)")
df['vol_ratio'] = df['volatility_10'] / (df['volatility_20'] + 1e-10)

print("   ✓ vol_ratio")

print("\n10. Key Interactions (2 features)")
df['imb_vol_interaction'] = df['imbalance_0'] * df['volatility_10']
df['spread_mom_interaction'] = df['spread_bps'] * np.abs(df['momentum_10'])

print("   ✓ imb_vol_interaction")
print("   ✓ spread_mom_interaction")

# ============================================
# RESEARCH-BACKED FEATURES (9) - Proven in Literature
# ============================================
print("\n" + "="*70)
print("RESEARCH-BACKED FEATURES (9)")
print("="*70)

print("\n11. Microprice (Stoikov 2018) - REMOVED")
# REMOVED: Redundant with existing price features
print("   ✗ microprice (removed - redundant)")

print("\n12. Multi-Level Imbalance (Cont 2014) ⭐⭐⭐")
df['imbalance_1'] = (
    (df['bid_sz_1'].astype(float) - df['ask_sz_1'].astype(float)) /
    (df['bid_sz_1'].astype(float) + df['ask_sz_1'].astype(float) + 1e-10)
)
df['imbalance_2'] = (
    (df['bid_sz_2'].astype(float) - df['ask_sz_2'].astype(float)) /
    (df['bid_sz_2'].astype(float) + df['ask_sz_2'].astype(float) + 1e-10)
)

print("   ✓ imbalance_1 (level 1 pressure)")
print("   ✓ imbalance_2 (level 2 pressure)")

print("\n13. Depth Asymmetry (Cont 2014) - REMOVED")
# REMOVED: Redundant with cumulative_imbalance
print("   ✗ depth_asymmetry (removed - redundant)")

print("\n14. Order Flow Imbalance (Cont 2014) ⭐⭐")
# Net change in bid/ask volumes (flow pressure)
df['ofi_simple'] = (
    df['bid_sz_0'].astype(float).diff().shift(1) - 
    df['ask_sz_0'].astype(float).diff().shift(1)
) / (df['total_depth'] + 1e-10)

print("   ✓ ofi_simple (order flow imbalance)")

print("\n15. Spread Volatility & Quality ⭐⭐⭐")
# How volatile is the spread? (regime detection)
df['spread_volatility'] = df['spread_bps'].shift(1).rolling(20).std()

# Spread relative to depth (liquidity quality)
df['spread_depth_ratio'] = df['spread_bps'] / (df['total_depth'] + 1e-10)

print("   ✓ spread_volatility")
print("   ✓ spread_depth_ratio")

print("\n16. Effective Spread (Hasbrouck 2009) ⭐⭐⭐")
# Spread relative to volatility (trading cost in vol units)
df['effective_spread'] = (
    df['spread_bps'] / 
    (df['volatility_10'] * 10000 + 1e-10)
)

print("   ✓ effective_spread")

print("\n17. Price Acceleration (Zhang 2019) ⭐⭐")
# Second derivative of price (momentum of momentum)
df['price_acceleration'] = (
    df['momentum_10'] - df['momentum_10'].shift(10).shift(1)
)

print("   ✓ price_acceleration")

print("\n18. Volume Acceleration (Cartea 2015) ⭐⭐")
# Change in total depth (liquidity draining/arriving)
df['volume_acceleration'] = (
    df['total_depth'] - df['total_depth'].shift(10).shift(1)
) / (df['total_depth'].shift(10).shift(1) + 1e-10)

print("   ✓ volume_acceleration")

print("\n19. Momentum Alignment ⭐")
# Are all time scales agreeing? (trend strength)
df['momentum_alignment'] = (
    np.sign(df['momentum_5']) + 
    np.sign(df['momentum_10']) + 
    np.sign(df['momentum_20'])
) / 3  # Ranges from -1 (all down) to +1 (all up)

print("   ✓ momentum_alignment")

# ============================================
# Clean up NaN values
# ============================================
print("\n" + "="*70)
print("DATA CLEANING")
print("="*70)

initial_count = len(df)
df = df.dropna().reset_index(drop=True)
final_count = len(df)

print(f"Initial samples:    {initial_count:,}")
print(f"After cleaning:     {final_count:,}")
print(f"Rows removed:       {initial_count - final_count:,} ({(initial_count-final_count)/initial_count*100:.1f}%)")

# ============================================
# Define feature lists
# ============================================
CORE_FEATURES = [
    # Price & Spread (2)
    'spread_bps', 'mid_price_usd',
    # Imbalance (2)
    'imbalance_0', 'cumulative_imbalance',
    # Momentum (3)
    'momentum_5', 'momentum_10', 'momentum_20',
    # Volatility (2)
    'volatility_10', 'volatility_20',
    # Order Flow (2)
    'microprice_mom', 'flow_0',
    # Book Shape (3)
    'bid_slope', 'ask_slope', 'total_depth'
]

ELECTIVE_FEATURES = [
    # Advanced Imbalance (2)
    'imbalance_change', 'persistent_imbalance',
    # Spread Dynamics (2)
    'spread_change', 'spread_vol_ratio',
    # Multi-scale (1)
    'vol_ratio',
    # Interactions (2)
    'imb_vol_interaction', 'spread_mom_interaction'
]

RESEARCH_FEATURES = [
    # Cont et al. (2014)
    'imbalance_1', 'imbalance_2', 'ofi_simple',
    # Spread quality
    'spread_volatility', 'spread_depth_ratio',
    # Hasbrouck (2009)
    'effective_spread',
    # Dynamics
    'price_acceleration', 'volume_acceleration',
    # Alignment
    'momentum_alignment'
]

ALL_FEATURES = CORE_FEATURES + ELECTIVE_FEATURES + RESEARCH_FEATURES

# ============================================
# Feature Summary
# ============================================
print("\n" + "="*70)
print("FEATURE SUMMARY")
print("="*70)

print(f"\n✅ CORE Features:              {len(CORE_FEATURES)}")
print(f"🔄 ELECTIVE Features:          {len(ELECTIVE_FEATURES)}")
print(f"⭐ RESEARCH Features:          {len(RESEARCH_FEATURES)}")
print(f"📊 TOTAL Features:             {len(ALL_FEATURES)}")

print("\n" + "="*70)
print("ALL FEATURES (30)")
print("="*70)
print("\nCORE (14):")
for i, feat in enumerate(CORE_FEATURES, 1):
    print(f"  {i:2d}. {feat}")

print("\nELECTIVE (7):")
for i, feat in enumerate(ELECTIVE_FEATURES, 1):
    print(f"  {i:2d}. {feat}")

print("\nRESEARCH (9):")
for i, feat in enumerate(RESEARCH_FEATURES, 1):
    print(f"  {i:2d}. {feat}")

# ============================================
# Quality Checks
# ============================================
print("\n" + "="*70)
print("QUALITY CHECKS")
print("="*70)

print("\n1. Checking for NaN/Inf values...")
problem_found = False
for col in ALL_FEATURES:
    n_inf = np.isinf(df[col]).sum()
    n_nan = df[col].isna().sum()
    if n_inf > 0 or n_nan > 0:
        print(f"   ⚠️  {col}: {n_inf} infinite, {n_nan} NaN")
        problem_found = True

if not problem_found:
    print("   ✅ No NaN or infinite values!")

print("\n2. Checking for high correlation (>0.90)...")
corr_matrix = df[ALL_FEATURES].corr()
high_corr_pairs = []

for i in range(len(ALL_FEATURES)):
    for j in range(i+1, len(ALL_FEATURES)):
        if abs(corr_matrix.iloc[i, j]) > 0.90:
            high_corr_pairs.append((
                ALL_FEATURES[i], 
                ALL_FEATURES[j], 
                corr_matrix.iloc[i, j]
            ))

if high_corr_pairs:
    print(f"   ⚠️  Found {len(high_corr_pairs)} highly correlated pairs:")
    for f1, f2, corr in high_corr_pairs[:5]:
        print(f"      {f1} <-> {f2}: {corr:.3f}")
    print("   Note: Consider removing one from each pair if causing issues")
else:
    print("   ✅ No extreme multicollinearity!")

# ============================================
# Feature-Target Correlations
# ============================================
print("\n" + "="*70)
print("TOP 20 FEATURES BY CORRELATION WITH return_200ms")
print("="*70)

target = 'return_200ms'
all_corrs = []
for feat in ALL_FEATURES:
    corr = df[feat].corr(df[target])
    all_corrs.append((feat, corr))

all_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

for i, (feat, corr) in enumerate(all_corrs[:20], 1):
    # Mark feature type
    if feat in CORE_FEATURES:
        marker = "[CORE]"
    elif feat in ELECTIVE_FEATURES:
        marker = "[ELECT]"
    else:
        marker = "[RESEARCH]"
    print(f"  {i:2d}. {feat:30s} {marker:12s} {corr:+.4f}")

# ============================================
# Save Dataset
# ============================================
print("\n" + "="*70)
print("SAVING DATASET")
print("="*70)

from pathlib import Path
Path('cleaned_data').mkdir(exist_ok=True)

output_file = 'cleaned_data/featured_AAPL.parquet'
df.to_parquet(output_file, index=False)
print(f"✅ Saved to: {output_file}")
print(f"   Shape: {df.shape}")
print(f"   Samples: {len(df):,}")

# Save feature configuration
import json

feature_config = {
    'core_features': CORE_FEATURES,
    'elective_features': ELECTIVE_FEATURES,
    'research_features': RESEARCH_FEATURES,
    'all_features': ALL_FEATURES,
    'total_count': len(ALL_FEATURES),
    'description': {
        'core': '14 essential features - simple, interpretable',
        'elective': '7 advanced features - experimental, swappable',
        'research': '9 research-backed features - proven in literature'
    },
    'removed_features': {
        'microprice': 'Removed - redundant with mid_price_usd',
        'depth_asymmetry': 'Removed - redundant with cumulative_imbalance'
    },
    'citations': {
        'depth_asymmetry_ofi': 'Cont et al. (2014) - Journal of Financial Econometrics',
        'ofi': 'Cont et al. (2014)',
        'effective_spread': 'Hasbrouck (2009) - Journal of Finance',
        'price_acceleration': 'Zhang et al. (2019) - IEEE Transactions',
        'volume_acceleration': 'Cartea & Jaimungal (2015) - Cambridge University Press'
    }
}

with open('cleaned_data/feature_config.json', 'w') as f:
    json.dump(feature_config, f, indent=2)
print(f"✅ Feature config saved to: cleaned_data/feature_config.json")

print("\n" + "="*70)
print("✅ FEATURE ENGINEERING COMPLETE!")
print(f"   30 features created (14 core + 7 elective + 9 research)")
print(f"   {len(df):,} samples ready for modeling")
print("="*70)