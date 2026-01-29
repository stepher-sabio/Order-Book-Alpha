import pandas as pd

df = pd.read_parquet('cleaned_data/featured_AAPL.parquet')

print("=== ZERO RETURNS BY HORIZON ===\n")

for horizon in ['return_50ms', 'return_100ms', 'return_200ms', 'return_500ms']:
    total = len(df)
    zeros = (df[horizon] == 0).sum()
    positive = (df[horizon] > 0).sum()
    negative = (df[horizon] < 0).sum()
    
    pct_zero = zeros / total * 100
    pct_pos = positive / total * 100
    pct_neg = negative / total * 100
    
    # Among non-zeros, what's the balance?
    if positive + negative > 0:
        pos_of_nonzero = positive / (positive + negative) * 100
    else:
        pos_of_nonzero = 0
    
    print(f"{horizon}:")
    print(f"  Zeros:    {zeros:,} ({pct_zero:.2f}%)")
    print(f"  Positive: {positive:,} ({pct_pos:.2f}%)")
    print(f"  Negative: {negative:,} ({pct_neg:.2f}%)")
    print(f"  Among non-zeros: {pos_of_nonzero:.2f}% positive")
    print()