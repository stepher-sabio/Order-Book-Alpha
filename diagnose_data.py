import pandas as pd
import numpy as np

# Load your data
df = pd.read_parquet('apple_data.parquet')  

print("=== DATA DIAGNOSTICS ===\n")

# 1. Check timestamp monotonicity
print("1. Timestamp Check:")
is_monotonic = df['timestamp'].is_monotonic_increasing
print(f"   Timestamps monotonic: {is_monotonic}")
if not is_monotonic:
    print("   ⚠️  WARNING: Non-monotonic timestamps detected!")

# 2. Check for book integrity (bid < ask)
print("\n2. Book Integrity Check:")
invalid_books = (df['bid_px_0'] >= df['ask_px_0']).sum()
print(f"   Invalid books (bid >= ask): {invalid_books}")
if invalid_books > 0:
    print("   ⚠️  WARNING: Crossed book detected!")

# 3. Check data coverage
print("\n3. Time Coverage:")
start_time = pd.to_datetime(df['timestamp'].min(), unit='ns')
end_time = pd.to_datetime(df['timestamp'].max(), unit='ns')
duration = (df['timestamp'].max() - df['timestamp'].min()) / 1e9  # seconds
print(f"   Start: {start_time}")
print(f"   End: {end_time}")
print(f"   Duration: {duration:.1f} seconds ({duration/3600:.2f} hours)")
print(f"   Total events: {len(df):,}")

# 4. Check for missing values
print("\n4. Missing Values:")
missing = df[['bid_px_0', 'ask_px_0', 'bid_sz_0', 'ask_sz_0']].isnull().sum()
print(missing)

# 5. Quick spread statistics
print("\n5. Spread Statistics (in price units):")
print(df['spread'].describe())

print("\n✅ Diagnostics complete!")