import pandas as pd
import numpy as np

print("Loading cleaned AAPL data...")
df = pd.read_parquet('cleaned_data/cleaned_AAPL.parquet')
print(f"Total events: {len(df):,}")

# Convert prices to dollars for easier interpretation
print("\nConverting prices to dollars...")
price_scale = 1e8  # 100 million
df['mid_price_usd'] = df['mid_price'] / price_scale
df['spread_usd'] = df['spread'] / price_scale

print(f"Mid-price range: ${df['mid_price_usd'].min():.2f} to ${df['mid_price_usd'].max():.2f}")
print(f"Spread range: ${df['spread_usd'].min():.4f} to ${df['spread_usd'].max():.4f}")

# Create log mid-price for returns calculation
df['log_mid'] = np.log(df['mid_price'])

# Define horizons (in nanoseconds)
horizons_ms = [50, 100, 200, 500]
horizons_ns = [h * 1_000_000 for h in horizons_ms]  # convert ms to ns

print(f"\n=== Creating labels for horizons: {horizons_ms} ms ===")

for h_ms, h_ns in zip(horizons_ms, horizons_ns):
    print(f"\nProcessing {h_ms}ms horizon...")
    
    # Find the future price at time t + horizon
    # We'll use a simple approach: for each row, find the first event >= t + h_ns
    df[f'future_time_{h_ms}ms'] = df['timestamp'] + h_ns
    
    # This is computationally intensive, so let's do it on a sample first
    # to verify it works, then we can optimize
    print(f"  (This will take a moment for {len(df):,} events...)")

print("\n⚠️  WAIT - this approach would be too slow for 93M events.")
print("Let me show you a faster method using time-based sampling first.")
print("\nShould we:")
print("  A) Sample the data at fixed intervals (e.g., every 10ms) - FAST")
print("  B) Create labels for all 93M events - SLOW but complete")
print("\nRecommendation: Start with option A for rapid iteration.")