import pandas as pd
import numpy as np

df = pd.read_parquet('cleaned_data/featured_AAPL.parquet')

print("=== LABEL ANALYSIS ===\n")

# Check return distribution
print("Return Statistics (50ms):")
returns_bps = df['return_50ms'] * 10000
print(returns_bps.describe())

print("\n=== Sign Distribution ===")
positive = (df['return_50ms'] > 0).sum()
negative = (df['return_50ms'] < 0).sum()
zero = (df['return_50ms'] == 0).sum()

total = len(df)
print(f"Positive returns: {positive:,} ({positive/total*100:.2f}%)")
print(f"Negative returns: {negative:,} ({negative/total*100:.2f}%)")
print(f"Zero returns:     {zero:,} ({zero/total*100:.2f}%)")

print("\n=== Check Imbalance vs Return Relationship ===")
print("Expected: Positive imbalance → Positive return (buying pressure → price up)")

sample = df.sample(n=100000, random_state=42)

# When imbalance > 0 (more bids), what's the average return?
pos_imb = sample[sample['imbalance_0'] > 0.1]['return_50ms'].mean()
neg_imb = sample[sample['imbalance_0'] < -0.1]['return_50ms'].mean()

print(f"\nWhen imbalance_0 > 0.1  (buy pressure):  avg return = {pos_imb*10000:+.4f} bps")
print(f"When imbalance_0 < -0.1 (sell pressure): avg return = {neg_imb*10000:+.4f} bps")

if pos_imb > neg_imb:
    print("✅ Correct: Buy pressure → Price up")
else:
    print("❌ INVERTED: Buy pressure → Price DOWN! Labels may be backwards")

print("\n=== Correlation Check ===")
corr = sample[['imbalance_0', 'return_50ms']].corr()
print(corr)

print("\n=== Sample Rows ===")
print("First 20 rows:")
print(df[['imbalance_0', 'cumulative_imbalance', 'momentum_10', 'return_50ms']].head(20))