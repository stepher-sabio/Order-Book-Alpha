import pandas as pd

df = pd.read_parquet('cleaned_data/sampled_labeled_AAPL.parquet')
print("Available columns:")
print(df.columns.tolist())
print(f"\nTotal columns: {len(df.columns)}")