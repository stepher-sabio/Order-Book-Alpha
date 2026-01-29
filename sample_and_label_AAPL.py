import pandas as pd
import numpy as np
import pyarrow.parquet as pq

print("=== STEP 1: Reading file info ===")
parquet_file = pq.ParquetFile('cleaned_data/cleaned_AAPL.parquet')
print(f"Total rows: {parquet_file.metadata.num_rows:,}")

# Columns we need
columns_to_read = [
    'timestamp', 'mid_price', 
    'bid_px_0', 'ask_px_0', 'bid_sz_0', 'ask_sz_0',
    'bid_px_1', 'ask_px_1', 'bid_sz_1', 'ask_sz_1',
    'bid_px_2', 'ask_px_2', 'bid_sz_2', 'ask_sz_2',
    'spread'
]

print("\n=== STEP 2: Determining time range ===")
# Read just timestamps to create sampling grid
timestamps_df = pd.read_parquet('cleaned_data/cleaned_AAPL.parquet', columns=['timestamp'])
start_time = timestamps_df['timestamp'].min()
end_time = timestamps_df['timestamp'].max()
duration_sec = (end_time - start_time) / 1e9

print(f"Start: {pd.to_datetime(start_time, unit='ns')}")
print(f"End: {pd.to_datetime(end_time, unit='ns')}")
print(f"Duration: {duration_sec/3600:.1f} hours")

# Create sampling grid
sampling_interval_ms = 50
sampling_interval_ns = sampling_interval_ms * 1_000_000

time_grid = np.arange(start_time, end_time, sampling_interval_ns)
print(f"\nSampling every {sampling_interval_ms}ms")
print(f"Time grid size: {len(time_grid):,}")

# Find sample indices
print("\n=== STEP 3: Finding sample indices ===")
sample_indices = np.searchsorted(timestamps_df['timestamp'].values, time_grid, side='right') - 1
sample_indices = np.clip(sample_indices, 0, len(timestamps_df) - 1)
sample_indices_set = set(sample_indices)

print(f"Unique sample indices: {len(sample_indices_set):,}")

del timestamps_df  # Free memory

print("\n=== STEP 4: Extracting samples in chunks ===")
all_samples = []
chunk_size = 5_000_000

for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size, columns=columns_to_read)):
    print(f"Processing chunk {i+1}...", end=' ')
    
    chunk = batch.to_pandas()
    start_idx = i * chunk_size
    end_idx = start_idx + len(chunk)
    
    # Find which sample indices fall in this chunk
    chunk_sample_indices = [idx - start_idx for idx in sample_indices_set 
                           if start_idx <= idx < end_idx]
    
    if chunk_sample_indices:
        chunk_sample_indices = sorted(chunk_sample_indices)
        samples = chunk.iloc[chunk_sample_indices].copy()
        all_samples.append(samples)
        print(f"extracted {len(samples):,} samples")
    else:
        print("no samples")
    
    # Free memory
    del chunk

print("\n=== STEP 5: Combining samples ===")
print("Concatenating all samples...")
sampled = pd.concat(all_samples, ignore_index=True)
sampled = sampled.sort_values('timestamp').reset_index(drop=True)

print(f"Total samples: {len(sampled):,}")

print("\n=== STEP 6: Creating labels ===")
price_scale = 1e8
sampled['mid_price_usd'] = sampled['mid_price'] / price_scale
sampled['log_mid'] = np.log(sampled['mid_price'])

horizons_ms = [50, 100, 200, 500]

for h_ms in horizons_ms:
    print(f"Creating {h_ms}ms labels...", end=' ')
    steps_ahead = h_ms // sampling_interval_ms
    sampled[f'return_{h_ms}ms'] = sampled['log_mid'].shift(-steps_ahead) - sampled['log_mid']
    valid = sampled[f'return_{h_ms}ms'].notna().sum()
    print(f"{valid:,} valid")

# Clean up
label_cols = [f'return_{h}ms' for h in horizons_ms]
sampled_clean = sampled.dropna(subset=label_cols).reset_index(drop=True)

print(f"\nFinal dataset: {len(sampled_clean):,} samples")

print("\n=== STEP 7: Verifying columns ===")
level_status = {}
for level in [0, 1, 2]:
    cols_needed = [f'bid_px_{level}', f'ask_px_{level}', 
                   f'bid_sz_{level}', f'ask_sz_{level}']
    has_all = all(col in sampled_clean.columns for col in cols_needed)
    level_status[level] = has_all
    status = "✅" if has_all else "❌"
    print(f"Level {level}: {status}")

# Save
output_file = 'sampled_labeled_AAPL.parquet'
sampled_clean.to_parquet(output_file, index=False)

print(f"\n✅ Saved to: {output_file}")
print(f"   Shape: {sampled_clean.shape}")
print(f"   Columns: {len(sampled_clean.columns)}")

print("\n=== LABEL STATISTICS ===")
for h_ms in horizons_ms:
    ret_bps = sampled_clean[f'return_{h_ms}ms'] * 10000
    print(f"{h_ms}ms: mean={ret_bps.mean():+.3f}bps, std={ret_bps.std():.3f}bps")

print("\n🎉 Ready for feature engineering!")