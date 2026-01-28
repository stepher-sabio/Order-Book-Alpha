import pandas as pd
import pyarrow.parquet as pq

# Read the parquet file
df = pd.read_parquet('microsoft_data.parquet')

# Or with pyarrow for more control
table = pq.read_table('microsoft_data.parquet')
df = table.to_pandas()

print(df.head())
print(f"Total records: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")