"""
validate_models.py - Comprehensive Model Validation
Beyond just R² and directional accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# Configuration - CHANGE THIS TO VALIDATE DIFFERENT MODELS
# ============================================
MODEL_PATH = 'models/phase4_xgb_fast.pkl'  # <-- Change this path to validate different models

PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)

TARGET_HORIZON = 'return_200ms'

# ============================================
# 1. Load Model & Data
# ============================================
print("="*70)
print("COMPREHENSIVE MODEL VALIDATION")
print("="*70)

print(f"\nLoading model: {MODEL_PATH}")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print("Loading test data...")
df = pd.read_parquet('cleaned_data/featured_AAPL.parquet')

from utils import FEATURE_COLS

X = df[FEATURE_COLS].values
y = df[TARGET_HORIZON].values

# Use last 20% as test (same as training)
split_idx = int(len(X) * 0.8)
X_test = X[split_idx:]
y_test = y[split_idx:]

print(f"Test samples: {len(y_test):,}")

# Predictions
y_pred = model.predict(X_test)

# ============================================
# 2. Basic Metrics (Sanity Check)
# ============================================
print("\n" + "="*70)
print("1. BASIC METRICS")
print("="*70)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
correlation = np.corrcoef(y_test, y_pred)[0, 1]

print(f"R²:          {r2*100:.4f}%")
print(f"MAE:         {mae*10000:.3f} bps")
print(f"RMSE:        {rmse*10000:.3f} bps")
print(f"Correlation: {correlation:.4f}")

# ============================================
# 3. Residual Analysis
# ============================================
print("\n" + "="*70)
print("2. RESIDUAL ANALYSIS")
print("="*70)

residuals = y_test - y_pred

print(f"Mean residual:   {residuals.mean()*10000:+.4f} bps (should be ~0)")
print(f"Std residual:    {residuals.std()*10000:.3f} bps")
print(f"Min residual:    {residuals.min()*10000:+.2f} bps")
print(f"Max residual:    {residuals.max()*10000:+.2f} bps")

# Check for bias
if abs(residuals.mean()) > 0.0001:
    print("⚠️  Model has bias (mean residual != 0)")
else:
    print("✅ No systematic bias")

# Plot residuals
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals histogram
axes[0, 0].hist(residuals * 10000, bins=100, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Residual (bps)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Residual Distribution')
axes[0, 0].grid(True, alpha=0.3)

# Predicted vs Actual
axes[0, 1].scatter(y_test * 10000, y_pred * 10000, alpha=0.1, s=1)
axes[0, 1].plot([y_test.min()*10000, y_test.max()*10000], 
                [y_test.min()*10000, y_test.max()*10000], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Return (bps)')
axes[0, 1].set_ylabel('Predicted Return (bps)')
axes[0, 1].set_title('Predicted vs Actual')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Residuals vs Predicted
axes[1, 0].scatter(y_pred * 10000, residuals * 10000, alpha=0.1, s=1)
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Return (bps)')
axes[1, 0].set_ylabel('Residual (bps)')
axes[1, 0].set_title('Residuals vs Predicted (check for heteroscedasticity)')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot (normality check)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (normality of residuals)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'validation_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Residual plots saved to {PLOTS_DIR}/validation_residuals.png")

# ============================================
# 4. Prediction Quality by Return Magnitude
# ============================================
print("\n" + "="*70)
print("3. PREDICTION QUALITY BY RETURN MAGNITUDE")
print("="*70)

# Bin by actual return size
bins = [-np.inf, -0.001, -0.0005, 0, 0.0005, 0.001, np.inf]
labels = ['Large Down', 'Small Down', 'Tiny Down', 'Tiny Up', 'Small Up', 'Large Up']
df_test = pd.DataFrame({'actual': y_test, 'pred': y_pred})
df_test['bin'] = pd.cut(df_test['actual'], bins=bins, labels=labels)

print("\nR² by return magnitude:")
for label in labels:
    subset = df_test[df_test['bin'] == label]
    if len(subset) > 100:
        r2_subset = r2_score(subset['actual'], subset['pred'])
        mae_subset = mean_absolute_error(subset['actual'], subset['pred']) * 10000
        print(f"  {label:12s}: R²={r2_subset*100:6.3f}%, MAE={mae_subset:5.2f} bps, n={len(subset):,}")

# ============================================
# 5. Directional Accuracy Deep Dive
# ============================================
print("\n" + "="*70)
print("4. DIRECTIONAL ACCURACY ANALYSIS")
print("="*70)

# Overall
dir_acc_all = np.mean(np.sign(y_test) == np.sign(y_pred))

# Non-zero only
nonzero_mask = y_test != 0
dir_acc_nonzero = np.mean(np.sign(y_test[nonzero_mask]) == np.sign(y_pred[nonzero_mask]))

# By magnitude
strong_move_mask = np.abs(y_test) > 0.0005  # >0.5 bps
dir_acc_strong = np.mean(np.sign(y_test[strong_move_mask]) == np.sign(y_pred[strong_move_mask]))

print(f"Direction Accuracy (all):         {dir_acc_all*100:.2f}%")
print(f"Direction Accuracy (non-zero):    {dir_acc_nonzero*100:.2f}%")
print(f"Direction Accuracy (|ret|>0.5bp): {dir_acc_strong*100:.2f}%")

# Confusion matrix for non-zeros
y_sign = np.sign(y_test[nonzero_mask])
pred_sign = np.sign(y_pred[nonzero_mask])

tp = np.sum((y_sign == 1) & (pred_sign == 1))  # Predicted up, was up
tn = np.sum((y_sign == -1) & (pred_sign == -1))  # Predicted down, was down
fp = np.sum((y_sign == -1) & (pred_sign == 1))  # Predicted up, was down
fn = np.sum((y_sign == 1) & (pred_sign == -1))  # Predicted down, was up

print(f"\nConfusion Matrix (non-zero returns):")
print(f"  True Positives (↑→↑):   {tp:,}")
print(f"  True Negatives (↓→↓):   {tn:,}")
print(f"  False Positives (↓→↑):  {fp:,}")
print(f"  False Negatives (↑→↓):  {fn:,}")

precision_up = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_up = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nWhen predicting UP:")
print(f"  Precision: {precision_up*100:.2f}% (how often is it correct?)")
print(f"  Recall:    {recall_up*100:.2f}% (% of actual ups captured)")

# ============================================
# 6. Temporal Stability (Walk-Forward)
# ============================================
print("\n" + "="*70)
print("5. TEMPORAL STABILITY (Out-of-Sample Decay)")
print("="*70)

# Split test set into 10 chunks (chronological)
n_chunks = 10
chunk_size = len(y_test) // n_chunks

print(f"Splitting test set into {n_chunks} chronological chunks...")

chunk_r2s = []
chunk_maes = []

for i in range(n_chunks):
    start = i * chunk_size
    end = start + chunk_size if i < n_chunks - 1 else len(y_test)
    
    y_chunk = y_test[start:end]
    pred_chunk = y_pred[start:end]
    
    if len(y_chunk) > 0:
        r2_chunk = r2_score(y_chunk, pred_chunk)
        mae_chunk = mean_absolute_error(y_chunk, pred_chunk) * 10000
        
        chunk_r2s.append(r2_chunk)
        chunk_maes.append(mae_chunk)
        
        print(f"  Chunk {i+1:2d}: R²={r2_chunk*100:6.3f}%, MAE={mae_chunk:5.2f} bps")

# Plot temporal stability
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(range(1, n_chunks+1), [r*100 for r in chunk_r2s], marker='o', linewidth=2)
axes[0].axhline(r2*100, color='red', linestyle='--', label=f'Overall R²={r2*100:.2f}%')
axes[0].set_xlabel('Test Set Chunk (chronological)')
axes[0].set_ylabel('R² (%)')
axes[0].set_title('Model Stability Over Time (R²)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, n_chunks+1), chunk_maes, marker='o', linewidth=2, color='orange')
axes[1].axhline(mae*10000, color='red', linestyle='--', label=f'Overall MAE={mae*10000:.2f} bps')
axes[1].set_xlabel('Test Set Chunk (chronological)')
axes[1].set_ylabel('MAE (bps)')
axes[1].set_title('Model Stability Over Time (MAE)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'validation_temporal_stability.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Temporal stability plot saved to {PLOTS_DIR}/validation_temporal_stability.png")

# Check for decay
r2_trend = np.polyfit(range(n_chunks), chunk_r2s, 1)[0]
if r2_trend < -0.0005:
    print(f"\n⚠️  Performance degrading over time (R² slope = {r2_trend*100:.4f}%)")
    print("    Model may not generalize well to future data")
elif r2_trend > 0.0005:
    print(f"\n✅ Performance improving over time (R² slope = {r2_trend*100:.4f}%)")
    print("    Model generalizes well")
else:
    print(f"\n✅ Performance stable over time (R² slope ≈ 0)")

# ============================================
# 7. Economic Validation (Trading Simulation)
# ============================================
print("\n" + "="*70)
print("6. ECONOMIC VALIDATION (Trading Simulation)")
print("="*70)

# Test MULTIPLE thresholds
thresholds = [0.0005, 0.001, 0.002, 0.005]  # 0.5, 1.0, 2.0, 5.0 bps

print(f"\nTesting multiple trading thresholds:\n")
print(f"{'Threshold':>12s} {'Trades':>10s} {'Win%':>8s} {'AvgWin':>10s} {'AvgLoss':>10s} {'Total':>10s} {'Sharpe':>10s}")
print("-" * 90)

best_sharpe = 0
best_threshold = 0  # Initialize to avoid NoneType error
sharpe = 0  # Initialize to avoid NameError
win_rate = 0  # Initialize to avoid NameError

for threshold in thresholds:
    positions = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
    pnl = positions * y_test
    
    n_trades = np.sum(positions != 0)
    if n_trades == 0:
        continue
    
    # Calculate trading metrics
    win_rate = np.sum(pnl > 0) / n_trades
    avg_win = np.mean(pnl[pnl > 0]) * 10000 if np.sum(pnl > 0) > 0 else 0
    avg_loss = np.mean(pnl[pnl < 0]) * 10000 if np.sum(pnl < 0) > 0 else 0
    total_return = np.sum(pnl) * 10000
    
    # CORRECTED SHARPE CALCULATION
    # Create full PnL time series
    full_pnl = np.zeros(len(y_test))
    trade_mask = positions != 0
    full_pnl[trade_mask] = pnl[trade_mask]
    
    # Calculate Sharpe on time series (not per-trade)
    if np.std(full_pnl) > 0:
        # Annualization: 200ms samples, 117,000 samples/day
        samples_per_day = 6.5 * 3600 / 0.2  # 117,000
        sharpe = (np.mean(full_pnl) / np.std(full_pnl)) * np.sqrt(samples_per_day)
    else:
        sharpe = 0
    
    print(f"{threshold*10000:10.2f} bps {n_trades:10,} {win_rate*100:7.2f}% {avg_win:9.2f} {avg_loss:10.2f} {total_return:10.2f} {sharpe:10.3f}")
    
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_threshold = threshold
        
# ============================================
# 8. Final Summary
# ============================================
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

print(f"\n✅ Statistical Metrics:")
print(f"   R² = {r2*100:.3f}%")
print(f"   Direction Accuracy (non-zero) = {dir_acc_nonzero*100:.1f}%")
print(f"   Correlation = {correlation:.3f}")

print(f"\n✅ Residual Analysis:")
print(f"   Mean bias = {residuals.mean()*10000:.4f} bps")
print(f"   {'No systematic errors' if abs(residuals.mean()) < 0.0001 else '⚠️ Systematic bias detected'}")

print(f"\n✅ Temporal Stability:")
print(f"   R² trend slope = {r2_trend*100:.4f}%")

print(f"\n✅ Economic Viability:")
print(f"   Best Sharpe Ratio = {best_sharpe:.2f} (at {best_threshold*10000:.1f} bps threshold)")
print(f"   Win Rate = {win_rate*100:.1f}%")

print(f"\n📊 Validation plots saved to: {PLOTS_DIR}/")
print("="*70)