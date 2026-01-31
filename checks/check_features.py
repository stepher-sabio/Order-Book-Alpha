import pandas as pd

df = pd.read_parquet('cleaned_data/featured_AAPL.parquet')
print("Columns in file:")
print(df.columns.tolist())
print(f"\nTotal columns: {len(df.columns)}")