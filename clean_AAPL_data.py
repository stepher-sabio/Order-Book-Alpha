import pandas as pd
import numpy as np
import pyarrow.parquet as pq

print("Processing AAPL data in batches...")

# First pass: determine spread thresholds from a sample
print("\n=== STEP 1: Analyzing sample to find thresholds ===")
sample = pd.read_parquet('cleaned_data/apple_data.parquet', columns=['bid_px_0', 'ask_px_0', 'spread'])
sample = sample.sample(n=min(1_000_000, len(sample)), random_state=42)
sample = sample[sample['bid_px_0'] < sample['ask_px_0']]

spread_percentiles = sample['spread'].quantile([0.001, 0.01, 0.5, 0.99, 0.999])
print("Spread percentiles:")
print(spread_percentiles)

upper_spread = spread_percentiles[0.999]
lower_spread = spread_percentiles[0.001]
print(f"\nThresholds: {lower_spread:.0f} to {upper_spread:.0f}")

# Second pass: filter and save in batches
print("\n=== STEP 2: Filtering data in batches ===")
parquet_file = pq.ParquetFile('cleaned_data/apple_data.parquet')
output_file = 'cleaned_AAPL.parquet'

total_kept = 0
chunks_to_write = []

for i, batch in enumerate(parquet_file.iter_batches(batch_size=5_000_000)):
    print(f"Processing batch {i+1}...", end=' ')
    
    # Convert to pandas
    chunk = batch.to_pandas()
    
    # Filter
    chunk = chunk[chunk['bid_px_0'] < chunk['ask_px_0']].copy()
    chunk = chunk[(chunk['spread'] >= lower_spread) & (chunk['spread'] <= upper_spread)].copy()
    chunk['mid_price'] = (chunk['bid_px_0'] + chunk['ask_px_0']) / 2
    
    chunks_to_write.append(chunk)
    total_kept += len(chunk)
    print(f"kept {len(chunk):,} events (total: {total_kept:,})")

# Combine all chunks and save once
print("\n=== STEP 3: Combining and saving ===")
df_final = pd.concat(chunks_to_write, ignore_index=True)
df_final.to_parquet(output_file, index=False)

print(f"\n✅ Done! Saved to {output_file}")
print(f"\n=== FINAL SUMMARY ===")
print(f"Total events: {len(df_final):,}")
print(f"Spread range: {df_final['spread'].min():.0f} to {df_final['spread'].max():.0f}")
print(f"Mid-price range: {df_final['mid_price'].min():.0f} to {df_final['mid_price'].max():.0f}")