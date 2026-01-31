"""
Data Quality Check - Enhanced Version
Shows comprehensive statistics for all columns in table format
"""

import pandas as pd
import numpy as np

# Load data
print("="*80)
print("DATA QUALITY CHECK")
print("="*80)

df = pd.read_parquet('cleaned_data/featured_AAPL.parquet')

print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================
# Create comprehensive statistics table
# ============================================
print("\n" + "="*80)
print("COLUMN STATISTICS")
print("="*80)

stats_data = []

for col in df.columns:
    col_data = df[col]
    
    stats = {
        'Column': col,
        'Type': str(col_data.dtype),
        'Non-Null': f"{col_data.notna().sum():,}",
        'Null': f"{col_data.isna().sum():,}",
        'Null %': f"{100 * col_data.isna().sum() / len(col_data):.2f}%",
        'Zeros': f"{(col_data == 0).sum():,}",
        'Zero %': f"{100 * (col_data == 0).sum() / len(col_data):.2f}%",
        'Unique': f"{col_data.nunique():,}",
        'Min': f"{col_data.min():.6f}" if pd.api.types.is_numeric_dtype(col_data) else "N/A",
        'Max': f"{col_data.max():.6f}" if pd.api.types.is_numeric_dtype(col_data) else "N/A",
        'Mean': f"{col_data.mean():.6f}" if pd.api.types.is_numeric_dtype(col_data) else "N/A",
        'Std': f"{col_data.std():.6f}" if pd.api.types.is_numeric_dtype(col_data) else "N/A",
    }
    
    stats_data.append(stats)

# Create DataFrame
stats_df = pd.DataFrame(stats_data)

# Display full table
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\n" + stats_df.to_string(index=False))

# ============================================
# Highlight potential issues
# ============================================
print("\n" + "="*80)
print("POTENTIAL ISSUES")
print("="*80)

issues_found = False

# Check for columns with NaN values
null_cols = stats_df[stats_df['Null'].str.replace(',', '').astype(int) > 0]
if len(null_cols) > 0:
    print("\n⚠️  Columns with NULL values:")
    for _, row in null_cols.iterrows():
        print(f"   {row['Column']:30s} - {row['Null']:>10s} nulls ({row['Null %']})")
    issues_found = True
else:
    print("\n✅ No NULL values found")

# Check for columns with >90% zeros
high_zero_cols = stats_df[stats_df['Zero %'].str.rstrip('%').astype(float) > 90]
if len(high_zero_cols) > 0:
    print("\n⚠️  Columns with >90% zeros:")
    for _, row in high_zero_cols.iterrows():
        print(f"   {row['Column']:30s} - {row['Zero %']}")
    issues_found = True
else:
    print("\n✅ No columns with excessive zeros (>90%)")

# Check for constant columns (only 1 unique value)
constant_cols = stats_df[stats_df['Unique'].str.replace(',', '').astype(int) == 1]
if len(constant_cols) > 0:
    print("\n⚠️  Constant columns (only 1 unique value):")
    for _, row in constant_cols.iterrows():
        print(f"   {row['Column']:30s}")
    issues_found = True
else:
    print("\n✅ No constant columns")

# Check for columns with infinite values
print("\n" + "-"*80)
print("Checking for infinite values...")
inf_cols = []
for col in df.select_dtypes(include=[np.number]).columns:
    n_inf = np.isinf(df[col]).sum()
    if n_inf > 0:
        inf_cols.append((col, n_inf))

if inf_cols:
    print("\n⚠️  Columns with infinite values:")
    for col, n_inf in inf_cols:
        print(f"   {col:30s} - {n_inf:,} infinite values")
    issues_found = True
else:
    print("✅ No infinite values found")

# ============================================
# Summary
# ============================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if not issues_found:
    print("\n✅ Data quality looks good! No major issues detected.")
else:
    print("\n⚠️  Some issues detected. Review the warnings above.")

print(f"\nTotal rows: {df.shape[0]:,}")
print(f"Total columns: {df.shape[1]}")
print(f"Date range: {df.index.min()} to {df.index.max()}" if isinstance(df.index, pd.DatetimeIndex) else "")

# ============================================
# Optional: Show sample of data
# ============================================
print("\n" + "="*80)
print("SAMPLE DATA (First 5 rows)")
print("="*80)
print(df.head().to_string())

print("\n" + "="*80)
print("SAMPLE DATA (Last 5 rows)")
print("="*80)
print(df.tail().to_string())

print("\n" + "="*80)
print("Data quality check complete!")
print("="*80)