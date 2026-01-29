import pandas as pd

df = pd.read_parquet('cleaned_data/sampled_labeled_AAPL.parquet')

print("=== DATA TYPES ===")
print(df[['bid_sz_0', 'ask_sz_0', 'bid_sz_1', 'ask_sz_1']].dtypes)

print("\n=== Test calculation ===")
test = df.head(1).copy()
print(f"bid_sz_0: {test['bid_sz_0'].values[0]}")
print(f"ask_sz_0: {test['ask_sz_0'].values[0]}")
print(f"Numerator (bid - ask): {test['bid_sz_0'].values[0] - test['ask_sz_0'].values[0]}")
print(f"Denominator (bid + ask): {test['bid_sz_0'].values[0] + test['ask_sz_0'].values[0]}")
print(f"Division: {(test['bid_sz_0'].values[0] - test['ask_sz_0'].values[0]) / (test['bid_sz_0'].values[0] + test['ask_sz_0'].values[0])}")

# Now try with pandas operation
test['imb'] = (test['bid_sz_0'] - test['ask_sz_0']) / (test['bid_sz_0'] + test['ask_sz_0'])
print(f"Pandas result: {test['imb'].values[0]}")