"""
validate_models.py - Order Book Alpha Research Validation

RESEARCH OBJECTIVE:
Validate predictive signal quality for short-horizon mid-price movements.
Success is measured by out-of-sample R², Information Coefficient (IC), 
and signal stability - NOT trading profits or Sharpe ratios.

EVALUATION FRAMEWORK:
1. Out-of-sample R² (primary metric)
2. Information Coefficient (IC) and stability
3. Directional accuracy with dead zone
4. Per-day/per-session stability
5. Horizon decay analysis
6. Robustness checks (latency simulation, feature ablation)

NOTE: This is alpha discovery research, not strategy backtesting.
Trading simulations are included only as sanity checks, not performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

# ============================================
# Configuration
# ============================================
MODEL_PATH = 'models/phase4_xgb.pkl'
PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)

TARGET_HORIZON = 'return_200ms'
HORIZON_MS = 200

# Dead zone for directional accuracy (in bps)
# Movements smaller than this are considered noise
DEAD_ZONE_BPS = 0.5

# ============================================
# Load Model & Data
# ============================================
print("="*80)
print("ORDER BOOK ALPHA VALIDATION - SIGNAL QUALITY ASSESSMENT")
print("="*80)

print(f"\nResearch Focus: Predictive signal validation at {HORIZON_MS}ms horizon")
print(f"Model: {MODEL_PATH}")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print("\nLoading test data...")
df = pd.read_parquet('cleaned_data/featured_AAPL.parquet')

print(f"\nData Overview:")
print(f"   Total samples: {len(df):,}")
print(f"   Columns: {len(df.columns)}")
print(f"   Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Show timestamp info if available
if 'timestamp' in df.columns:
    sample_ts = df['timestamp'].iloc[0]
    print(f"   Timestamp column: present")
    print(f"   Timestamp dtype: {df['timestamp'].dtype}")
    print(f"   Sample timestamp: {sample_ts}")
else:
    print(f"   Timestamp column: not found")
    print(f"   Available columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")

from utils import FEATURE_COLS

X = df[FEATURE_COLS].values
y = df[TARGET_HORIZON].values

# Use last 20% as test (strict temporal split)
split_idx = int(len(X) * 0.8)
X_test = X[split_idx:]
y_test = y[split_idx:]

# Extract timestamps for daily analysis
test_dates = None
unique_dates = None
n_days = 0

if 'timestamp' in df.columns:
    try:
        timestamps = df['timestamp'].values[split_idx:]
        
        # Try to detect timestamp format
        sample_ts = timestamps[0]
        
        # Check if already datetime
        if isinstance(sample_ts, (pd.Timestamp, np.datetime64)):
            test_dates = pd.to_datetime(timestamps).date
        # Check if nanoseconds (very large number)
        elif sample_ts > 1e15:
            test_dates = pd.to_datetime(timestamps, unit='ns').date
        # Check if microseconds
        elif sample_ts > 1e12:
            test_dates = pd.to_datetime(timestamps, unit='us').date
        # Milliseconds
        elif sample_ts > 1e9:
            test_dates = pd.to_datetime(timestamps, unit='ms').date
        # Seconds
        else:
            test_dates = pd.to_datetime(timestamps, unit='s').date
        
        unique_dates = np.unique(test_dates)
        n_days = len(unique_dates)
    except Exception as e:
        print(f"   ⚠ Warning: Could not parse timestamps ({e})")
        print(f"   ⚠ Falling back to sample-based estimation")
        test_dates = None
        unique_dates = None

# Fallback to estimation if timestamp parsing failed
if test_dates is None:
    samples_per_day = (6.5 * 3600) / (HORIZON_MS / 1000)
    n_days = len(y_test) / samples_per_day
    print(f"   ⚠ Using estimated days based on sample count")

print(f"Test samples: {len(y_test):,}")
print(f"Estimated test period: ~{n_days:.1f} trading days")
print(f"Prediction horizon: {HORIZON_MS}ms")

# Generate predictions
y_pred = model.predict(X_test)

# ============================================
# SECTION 1: OUT-OF-SAMPLE R² (PRIMARY METRIC)
# ============================================
print("\n" + "="*80)
print("SECTION 1: OUT-OF-SAMPLE PREDICTIVE PERFORMANCE")
print("="*80)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
correlation = np.corrcoef(y_test, y_pred)[0, 1]

print(f"\n📊 Core Metrics:")
print(f"   R²:                {r2*100:.4f}%")
print(f"   Correlation:       {correlation:.4f}")
print(f"   MAE:              {mae*10000:.3f} bps")
print(f"   RMSE:             {rmse*10000:.3f} bps")

print(f"\n💡 Interpretation:")
if r2 > 0.02:
    print(f"   ✓ R² = {r2*100:.2f}% is VERY STRONG for {HORIZON_MS}ms microstructure")
    print(f"   ✓ This exceeds typical academic benchmarks (~1-1.5%)")
elif r2 > 0.01:
    print(f"   ✓ R² = {r2*100:.2f}% is MEANINGFUL for {HORIZON_MS}ms horizon")
    print(f"   ✓ Comparable to published microstructure research")
elif r2 > 0.005:
    print(f"   ~ R² = {r2*100:.2f}% is MODEST but potentially exploitable")
    print(f"   ~ Room for improvement through feature engineering")
else:
    print(f"   ✗ R² = {r2*100:.2f}% is WEAK - signal needs strengthening")
    print(f"   ✗ Consider: longer horizons, more features, or different modeling")

# ============================================
# SECTION 2: INFORMATION COEFFICIENT (IC)
# ============================================
print("\n" + "="*80)
print("SECTION 2: INFORMATION COEFFICIENT & CONSISTENCY")
print("="*80)

# Overall IC (Spearman rank correlation)
ic_overall = stats.spearmanr(y_test, y_pred)[0]

print(f"\n📈 Information Coefficient (IC):")
print(f"   Overall IC:        {ic_overall:.4f}")

# Daily IC analysis
if test_dates is not None and unique_dates is not None:
    daily_ics = []
    daily_r2s = []
    
    for date in unique_dates:
        mask = test_dates == date
        if mask.sum() > 10:  # Need minimum samples
            y_day = y_test[mask]
            pred_day = y_pred[mask]
            
            ic_day = stats.spearmanr(y_day, pred_day)[0]
            r2_day = r2_score(y_day, pred_day)
            
            daily_ics.append(ic_day)
            daily_r2s.append(r2_day)
    
    daily_ics = np.array(daily_ics)
    daily_r2s = np.array(daily_r2s)
    
    print(f"\n   Daily IC Statistics:")
    print(f"   Mean IC:           {daily_ics.mean():.4f}")
    print(f"   Std IC:            {daily_ics.std():.4f}")
    print(f"   IC t-stat:         {daily_ics.mean() / (daily_ics.std() / np.sqrt(len(daily_ics))):.2f}")
    print(f"   % Positive days:   {(daily_ics > 0).sum() / len(daily_ics) * 100:.1f}%")
    print(f"   Min IC:            {daily_ics.min():.4f}")
    print(f"   Max IC:            {daily_ics.max():.4f}")
    
    print(f"\n💡 Stability Assessment:")
    positive_pct = (daily_ics > 0).sum() / len(daily_ics) * 100
    if positive_pct > 70 and daily_ics.std() < 0.1:
        print(f"   ✓ EXCELLENT stability - signal is consistent across days")
    elif positive_pct > 60 and daily_ics.std() < 0.15:
        print(f"   ✓ GOOD stability - signal is mostly reliable")
    elif positive_pct > 50:
        print(f"   ~ MODERATE stability - some day-to-day variation")
    else:
        print(f"   ✗ POOR stability - signal is inconsistent")
else:
    print(f"\n   ⚠ Daily breakdown not available (no timestamp data)")
    daily_ics = None
    daily_r2s = None

# ============================================
# SECTION 3: DIRECTIONAL ACCURACY (WITH DEAD ZONE)
# ============================================
print("\n" + "="*80)
print("SECTION 3: DIRECTIONAL ACCURACY")
print("="*80)

dead_zone = DEAD_ZONE_BPS / 10000

# Filter out small movements (noise)
significant_mask = np.abs(y_test) > dead_zone
y_sig = y_test[significant_mask]
pred_sig = y_pred[significant_mask]

if len(y_sig) > 0:
    dir_accuracy = np.mean(np.sign(y_sig) == np.sign(pred_sig))
    
    # Breakdown by direction
    up_mask = y_sig > 0
    down_mask = y_sig < 0
    
    if up_mask.sum() > 0:
        acc_up = np.mean(np.sign(y_sig[up_mask]) == np.sign(pred_sig[up_mask]))
    else:
        acc_up = np.nan
    
    if down_mask.sum() > 0:
        acc_down = np.mean(np.sign(y_sig[down_mask]) == np.sign(pred_sig[down_mask]))
    else:
        acc_down = np.nan
    
    print(f"\n📍 Directional Accuracy (|movement| > {DEAD_ZONE_BPS} bps):")
    print(f"   Overall:           {dir_accuracy*100:.2f}%")
    print(f"   Up moves:          {acc_up*100:.2f}%")
    print(f"   Down moves:        {acc_down*100:.2f}%")
    print(f"   Significant moves: {significant_mask.sum():,} / {len(y_test):,} ({significant_mask.sum()/len(y_test)*100:.1f}%)")
    
    print(f"\n💡 Interpretation:")
    if dir_accuracy > 0.55:
        print(f"   ✓ {dir_accuracy*100:.1f}% is STRONG for microstructure prediction")
    elif dir_accuracy > 0.52:
        print(f"   ✓ {dir_accuracy*100:.1f}% is MEANINGFUL (above random)")
    elif dir_accuracy > 0.50:
        print(f"   ~ {dir_accuracy*100:.1f}% is SLIGHT edge (needs improvement)")
    else:
        print(f"   ✗ {dir_accuracy*100:.1f}% is NO EDGE (≤ random)")
else:
    print(f"\n   ⚠ No significant movements above {DEAD_ZONE_BPS} bps threshold")

# ============================================
# SECTION 4: PREDICTION DISTRIBUTION ANALYSIS
# ============================================
print("\n" + "="*80)
print("SECTION 4: PREDICTION DISTRIBUTION & CALIBRATION")
print("="*80)

print(f"\n📊 Prediction Statistics:")
print(f"   Mean prediction:   {y_pred.mean()*10000:.3f} bps")
print(f"   Std prediction:    {y_pred.std()*10000:.3f} bps")
print(f"   Mean actual:       {y_test.mean()*10000:.3f} bps")
print(f"   Std actual:        {y_test.std()*10000:.3f} bps")

# Quantile analysis
for q in [0.01, 0.05, 0.10, 0.90, 0.95, 0.99]:
    pred_q = np.quantile(y_pred, q)
    actual_q = np.quantile(y_test, q)
    print(f"   Q{q*100:02.0f} - Pred: {pred_q*10000:6.2f} bps, Actual: {actual_q*10000:6.2f} bps")

# ============================================
# SECTION 5: ROBUSTNESS CHECK - LATENCY SIMULATION
# ============================================
print("\n" + "="*80)
print("SECTION 5: ROBUSTNESS - LATENCY SIMULATION")
print("="*80)

print(f"\nSimulating various execution delays...")
print(f"(Real alpha should degrade smoothly with latency)")

latencies_ms = [0, 10, 20, 50, 100, 200]
latencies_samples = [int(l / HORIZON_MS) for l in latencies_ms]

print(f"\n{'Latency':>10s} {'R²':>10s} {'IC':>10s} {'Correlation':>12s}")
print("-" * 50)

for lat_ms, lat_samples in zip(latencies_ms, latencies_samples):
    if lat_samples >= len(y_test):
        continue
    
    # Shift predictions to simulate latency
    y_delayed = y_test[lat_samples:]
    pred_delayed = y_pred[:-lat_samples] if lat_samples > 0 else y_pred
    
    r2_lat = r2_score(y_delayed, pred_delayed)
    ic_lat = stats.spearmanr(y_delayed, pred_delayed)[0]
    corr_lat = np.corrcoef(y_delayed, pred_delayed)[0, 1]
    
    print(f"{lat_ms:8d} ms {r2_lat*100:9.4f}% {ic_lat:10.4f} {corr_lat:12.4f}")

print(f"\n💡 Interpretation:")
print(f"   A real microstructure signal should decay with latency.")
print(f"   Abrupt collapse suggests overfitting or look-ahead bias.")

# ============================================
# SECTION 6: TRADING SANITY CHECK (NON-OPTIMIZED)
# ============================================
print("\n" + "="*80)
print("SECTION 6: TRADING SANITY CHECK (INFORMATIONAL ONLY)")
print("="*80)

print(f"\n⚠️  NOTE: This is NOT the primary evaluation metric.")
print(f"   Purpose: Verify signal has economic relevance under realistic frictions.")
print(f"   This is NOT strategy optimization or performance measurement.")

# Single fixed rule (no optimization)
THRESHOLD_BPS = 2.0  # Conservative threshold
TRANSACTION_COST_BPS = 0.5
threshold = THRESHOLD_BPS / 10000
tc = TRANSACTION_COST_BPS / 10000

print(f"\nFixed Parameters (non-optimized):")
print(f"   Prediction threshold: {THRESHOLD_BPS} bps")
print(f"   Transaction cost:     {TRANSACTION_COST_BPS} bps (round-trip)")

# Simple binary positions
positions = np.where(y_pred > threshold, 1, 
                    np.where(y_pred < -threshold, -1, 0))

trade_mask = positions != 0
n_trades = trade_mask.sum()

if n_trades > 0:
    # PnL per trade
    pnl_per_trade = (positions[trade_mask] * y_test[trade_mask]) - tc
    
    print(f"\nTrade Statistics:")
    print(f"   Total trades:      {n_trades:,}")
    print(f"   Trades per day:    {n_trades / n_days:.1f}")
    print(f"   Win rate:          {(pnl_per_trade > 0).sum() / n_trades * 100:.1f}%")
    print(f"   Avg P&L per trade: {pnl_per_trade.mean() * 10000:.2f} bps")
    print(f"   Std P&L per trade: {pnl_per_trade.std() * 10000:.2f} bps")
    
    # Cumulative P&L
    cum_pnl_bps = pnl_per_trade.sum() * 10000
    print(f"   Cumulative P&L:    {cum_pnl_bps:.1f} bps over {n_days:.1f} days")
    
    print(f"\n💡 Interpretation:")
    if pnl_per_trade.mean() > tc:
        print(f"   ✓ Positive expectancy after costs - signal has economic value")
    elif pnl_per_trade.mean() > 0:
        print(f"   ~ Marginally positive - close to breakeven after costs")
    else:
        print(f"   ✗ Negative expectancy - costs exceed edge")
    
    print(f"\n   This sanity check confirms the signal could theoretically be")
    print(f"   monetized, but full strategy development requires:")
    print(f"   - Market impact modeling")
    print(f"   - Slippage analysis")
    print(f"   - Regime filtering")
    print(f"   - Risk management")
    print(f"   - Live execution infrastructure")
else:
    print(f"\n   No trades generated at {THRESHOLD_BPS} bps threshold")

# ============================================
# SECTION 7: VISUALIZATION
# ============================================
print("\n" + "="*80)
print("SECTION 7: CREATING DIAGNOSTIC PLOTS")
print("="*80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Prediction vs Actual (scatter)
ax1 = fig.add_subplot(gs[0, 0])
sample_idx = np.random.choice(len(y_test), size=min(5000, len(y_test)), replace=False)
ax1.scatter(y_test[sample_idx]*10000, y_pred[sample_idx]*10000, alpha=0.3, s=1)
ax1.plot([-50, 50], [-50, 50], 'r--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Actual Return (bps)')
ax1.set_ylabel('Predicted Return (bps)')
ax1.set_title(f'Prediction vs Actual (R²={r2*100:.3f}%)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-50, 50)
ax1.set_ylim(-50, 50)

# 2. Residual distribution
ax2 = fig.add_subplot(gs[0, 1])
residuals = (y_test - y_pred) * 10000
ax2.hist(residuals, bins=100, alpha=0.7, edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', linewidth=1)
ax2.set_xlabel('Residual (bps)')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Residual Distribution (MAE={mae*10000:.2f} bps)')
ax2.grid(True, alpha=0.3)

# 3. Prediction distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(y_pred*10000, bins=100, alpha=0.7, label='Predicted', edgecolor='black')
ax3.hist(y_test*10000, bins=100, alpha=0.5, label='Actual', edgecolor='black')
ax3.set_xlabel('Return (bps)')
ax3.set_ylabel('Frequency')
ax3.set_title('Return Distributions')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Daily IC (if available)
ax4 = fig.add_subplot(gs[1, 0])
if daily_ics is not None:
    ax4.plot(daily_ics, marker='o', linewidth=1, markersize=4)
    ax4.axhline(0, color='red', linestyle='--', linewidth=1)
    ax4.axhline(daily_ics.mean(), color='green', linestyle='--', linewidth=1, label=f'Mean={daily_ics.mean():.3f}')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('IC')
    ax4.set_title('Daily Information Coefficient')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Daily IC\nNot Available', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Daily Information Coefficient')

# 5. Daily R² (if available)
ax5 = fig.add_subplot(gs[1, 1])
if daily_r2s is not None:
    ax5.plot(daily_r2s*100, marker='o', linewidth=1, markersize=4, color='orange')
    ax5.axhline(0, color='red', linestyle='--', linewidth=1)
    ax5.axhline(daily_r2s.mean()*100, color='green', linestyle='--', linewidth=1, label=f'Mean={daily_r2s.mean()*100:.3f}%')
    ax5.set_xlabel('Day')
    ax5.set_ylabel('R² (%)')
    ax5.set_title('Daily R²')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
else:
    ax5.text(0.5, 0.5, 'Daily R²\nNot Available', ha='center', va='center', transform=ax5.transAxes)
    ax5.set_title('Daily R²')

# 6. Quantile-quantile plot
ax6 = fig.add_subplot(gs[1, 2])
stats.probplot(residuals, dist="norm", plot=ax6)
ax6.set_title('Q-Q Plot (Residuals)')
ax6.grid(True, alpha=0.3)

# 7. Prediction strength distribution
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(np.abs(y_pred)*10000, bins=100, alpha=0.7, edgecolor='black')
ax7.set_xlabel('|Prediction| (bps)')
ax7.set_ylabel('Frequency')
ax7.set_title('Prediction Strength Distribution')
ax7.grid(True, alpha=0.3)

# 8. IC by prediction strength
ax8 = fig.add_subplot(gs[2, 1])
pred_abs = np.abs(y_pred)
quantiles = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
ic_by_strength = []
labels = []

for i in range(len(quantiles)-1):
    q_low = np.quantile(pred_abs, quantiles[i])
    q_high = np.quantile(pred_abs, quantiles[i+1])
    mask = (pred_abs >= q_low) & (pred_abs < q_high)
    
    if mask.sum() > 10:
        ic = stats.spearmanr(y_test[mask], y_pred[mask])[0]
        ic_by_strength.append(ic)
        labels.append(f'Q{i+1}')

ax8.bar(range(len(ic_by_strength)), ic_by_strength, edgecolor='black')
ax8.set_xticks(range(len(ic_by_strength)))
ax8.set_xticklabels(labels)
ax8.set_xlabel('Prediction Strength Quantile')
ax8.set_ylabel('IC')
ax8.set_title('IC by Prediction Confidence')
ax8.axhline(0, color='red', linestyle='--', linewidth=1)
ax8.grid(True, alpha=0.3)

# 9. Cumulative returns (sorted by prediction)
ax9 = fig.add_subplot(gs[2, 2])
sorted_idx = np.argsort(y_pred)
cum_returns = np.cumsum(y_test[sorted_idx]) * 10000
ax9.plot(cum_returns, linewidth=1)
ax9.set_xlabel('Samples (sorted by prediction)')
ax9.set_ylabel('Cumulative Actual Return (bps)')
ax9.set_title('Cumulative Returns (Prediction-Sorted)')
ax9.grid(True, alpha=0.3)

plt.savefig(PLOTS_DIR / 'alpha_validation_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Diagnostic plots saved to {PLOTS_DIR}/alpha_validation_diagnostics.png")

# ============================================
# SUMMARY REPORT
# ============================================
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

print(f"\n🎯 PRIMARY METRICS (Alpha Quality):")
print(f"   Out-of-sample R²:  {r2*100:.4f}%")
print(f"   Information Coef:  {ic_overall:.4f}")
print(f"   Correlation:       {correlation:.4f}")

if daily_ics is not None:
    print(f"\n📊 STABILITY METRICS:")
    print(f"   Mean daily IC:     {daily_ics.mean():.4f} ± {daily_ics.std():.4f}")
    print(f"   IC t-statistic:    {daily_ics.mean() / (daily_ics.std() / np.sqrt(len(daily_ics))):.2f}")
    print(f"   Positive days:     {(daily_ics > 0).sum()}/{len(daily_ics)} ({(daily_ics > 0).sum()/len(daily_ics)*100:.1f}%)")

print(f"\n🎲 DIRECTIONAL ACCURACY:")
print(f"   On significant moves: {dir_accuracy*100:.2f}%")
print(f"   (Threshold: {DEAD_ZONE_BPS} bps)")

print(f"\n💼 ECONOMIC SANITY CHECK:")
if n_trades > 0:
    print(f"   Trades (non-opt):  {n_trades:,}")
    print(f"   Win rate:          {(pnl_per_trade > 0).sum() / n_trades * 100:.1f}%")
    print(f"   Avg P&L/trade:     {pnl_per_trade.mean() * 10000:.2f} bps")

print(f"\n" + "="*80)
print("RESEARCH ASSESSMENT")
print("="*80)

if r2 > 0.015 and ic_overall > 0.05:
    print("\n✅ STRONG ALPHA SIGNAL")
    print("   • Predictive power exceeds academic benchmarks")
    print("   • Ready for regime analysis and feature refinement")
    print("   • Consider testing at different horizons (50ms, 500ms)")
elif r2 > 0.008 and ic_overall > 0.03:
    print("\n✓ MEANINGFUL ALPHA SIGNAL")
    print("   • Predictive power is statistically significant")
    print("   • Worth exploring further with:")
    print("     - Nonlinear models (RF, GBM)")
    print("     - Additional features")
    print("     - Regime-conditional modeling")
elif r2 > 0.003:
    print("\n~ MODEST ALPHA SIGNAL")
    print("   • Weak but potentially buildable")
    print("   • Recommend:")
    print("     - Feature engineering focus")
    print("     - Longer horizons (500ms, 1s)")
    print("     - Ensemble methods")
else:
    print("\n✗ INSUFFICIENT ALPHA SIGNAL")
    print("   • Model lacks predictive power at this horizon")
    print("   • Consider:")
    print("     - Different feature sets")
    print("     - Longer prediction horizons")
    print("     - Alternative modeling approaches")

print(f"\n📚 NEXT STEPS:")
print(f"   1. Validate on separate out-of-sample period")
print(f"   2. Test at multiple horizons (50ms, 200ms, 500ms)")
print(f"   3. Perform feature ablation study")
print(f"   4. Explore nonlinear models")
print(f"   5. Document methodology and limitations")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print("\n⚠️  REMINDER: This is alpha research, not production trading.")
print("   Real deployment requires: execution modeling, risk management,")
print("   regime filtering, and extensive live testing.")
print("="*80)