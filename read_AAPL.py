import pyarrow.parquet as pq

# Read in batches
parquet_file = pq.ParquetFile('apple_data.parquet')

for i in range(parquet_file.num_row_groups):
    # Read one row group at a time
    table = parquet_file.read_row_group(i)
    df_chunk = table.to_pandas()
    
    print(f"Chunk {i}: {len(df_chunk):,} rows")
    # Process this chunk
    
    if i == 0:  # Just look at first chunk for now
        print(df_chunk.head())
        break