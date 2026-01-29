import pandas as pd
import numpy as np

df = pd.read_parquet('cleaned_data/sampled_labeled_AAPL.parquet')

print("=== RAW DATA CHECK ===\n")

print("First 20 rows of size data:")
print(df[['bid_sz_0', 'ask_sz_0', 'bid_sz_1', 'ask_sz_1']].head(20))

print("\n=== Size Statistics ===")
print(df[['bid_sz_0', 'ask_sz_0', 'bid_sz_1', 'ask_sz_1']].describe())

print("\n=== Check for zeros ===")
print(f"bid_sz_0 zeros: {(df['bid_sz_0'] == 0).sum():,}")
print(f"ask_sz_0 zeros: {(df['ask_sz_0'] == 0).sum():,}")

print("\n=== Manual imbalance calculation on first 20 rows ===")
sample = df.head(20).copy()
sample['manual_imb'] = (sample['bid_sz_0'] - sample['ask_sz_0']) / (sample['bid_sz_0'] + sample['ask_sz_0'])
print(sample[['bid_sz_0', 'ask_sz_0', 'manual_imb']])

print("\n=== Check if sizes might be in wrong units ===")
print("Are bid_sz values extremely large?")
print(f"Max bid_sz_0: {df['bid_sz_0'].max():,.0f}")
print(f"Max ask_sz_0: {df['ask_sz_0'].max():,.0f}")