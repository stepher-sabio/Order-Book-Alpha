"""
phase3_gradientboosting.py - Gradient Boosting Models
OPTIMIZED FOR SPEED with HistGradientBoostingRegressor

MAJOR SPEED IMPROVEMENTS:
1. HistGradientBoostingRegressor - 10-100x faster than standard GradientBoostingRegressor
2. Native histogram binning (like XGBoost)
3. Built-in categorical feature support
4. Better parallelization
5. More memory efficient

EXPECTED PERFORMANCE ON 11.7M SAMPLES (with HistGradientBoosting):
- Fast config: 2-5 minutes (vs 15-30 min with standard GB)
- Balanced config: 5-10 minutes (vs 30-60 min)
- Accurate config: 10-20 minutes (vs 1-2 hours)

SPEED COMPARISON:
Standard GB:     SLOW (hours)
HistGradientGB:  FAST (minutes) ← Using this!
XGBoost:         VERY FAST (minutes)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor  # FAST version!
import time
from pathlib import Path
import matplotlib.pyplot as plt
import gc
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Import our utilities
from utils import (
    load_data, 
    time_based_split,
    evaluate_model,
    print_metrics,
    save_model,
    save_results,
    format_duration,
    print_section,
    FEATURE_COLS,
    TARGET_HORIZONS
)

# ============================================
# Configuration
# ============================================
RESULTS_DIR = Path('results')
MODELS_DIR = Path('models')
PLOTS_DIR = Path('plots')
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Start with sample for testing
SAMPLE_SIZE = None  # Set to 1_000_000 for quick testing

# Target
TARGET_HORIZON = 'return_200ms'

# Validation for early stopping
USE_VALIDATION_SET = True
VALIDATION_SIZE = 0.1

# Memory optimization
ENABLE_MEMORY_OPTIMIZATION = True

# ============================================
# Gradient Boosting Configurations (FAST VERSION)
# ============================================
def get_gradient_boosting_configs():
    """
    Optimized HistGradientBoosting configs for large datasets
    
    HistGradientBoostingRegressor is MUCH faster because:
    - Native histogram binning (like XGBoost 'hist' method)
    - Better parallelization across trees
    - Efficient memory usage
    - No need to sort data at each split
    - Can handle missing values natively
    
    This is scikit-learn's answer to XGBoost speed!
    """
    
    configs = {
        # ULTRA FAST: 2-5 minutes on 11.7M samples
        'GB_Fast': {
            'max_iter': 100,           # Similar to n_estimators
            'learning_rate': 0.1,
            'max_depth': 4,            # Can go deeper than standard GB
            'min_samples_leaf': 20,
            'max_bins': 255,           # Histogram bins (255 is max)
            'l2_regularization': 0.0,
            'early_stopping': True,
            'n_iter_no_change': 10,
            'validation_fraction': 0.1,
            'random_state': 42,
            'verbose': 1
        },
        
        # BALANCED: 5-10 minutes on 11.7M samples
        'GB_Balanced': {
            'max_iter': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_samples_leaf': 10,
            'max_bins': 255,
            'l2_regularization': 0.0,
            'early_stopping': True,
            'n_iter_no_change': 15,
            'validation_fraction': 0.1,
            'random_state': 42,
            'verbose': 1
        },
        
        # ACCURATE: 10-20 minutes on 11.7M samples (still fast!)
        'GB_Accurate': {
            'max_iter': 300,
            'learning_rate': 0.05,
            'max_depth': 8,
            'min_samples_leaf': 5,
            'max_bins': 255,
            'l2_regularization': 0.1,  # Some regularization
            'early_stopping': True,
            'n_iter_no_change': 20,
            'validation_fraction': 0.1,
            'random_state': 42,
            'verbose': 1
        }
    }
    
    return configs

# ============================================
# Memory-Efficient Data Loading
# ============================================
def load_data_efficient(file_path, sample_size=None, random_state=42):
    """Load data with memory optimizations"""
    print(f"Loading data from {file_path}...")
    
    df = pd.read_parquet(file_path)
    print(f"Original size: {len(df):,} rows")
    
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size:,} rows...")
        df = df.sample(n=sample_size, random_state=random_state)
    
    if ENABLE_MEMORY_OPTIMIZATION:
        print("Optimizing memory usage (float64 → float32)...")
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df

# ============================================
# Main Training Function
# ============================================
def train_gradient_boosting():
    """Train and evaluate Gradient Boosting models"""
    
    print_section("PHASE 3: GRADIENT BOOSTING")
    
    # ============================================
    # Load Data
    # ============================================
    print_section("Loading Data")
    df = load_data_efficient('cleaned_data/featured_AAPL.parquet', sample_size=SAMPLE_SIZE)
    
    X = df[FEATURE_COLS].values
    y = df[TARGET_HORIZON].values
    
    print(f"\nFeatures: {len(FEATURE_COLS)}")
    print(f"Target: {TARGET_HORIZON}")
    print(f"X shape: {X.shape}")
    
    # Convert to float32 for speed and memory
    if ENABLE_MEMORY_OPTIMIZATION:
        X = X.astype('float32')
        y = y.astype('float32')
        print("Data converted to float32")
    
    del df
    gc.collect()
    
    # ============================================
    # Train/Validation/Test Split
    # ============================================
    print_section("Train/Validation/Test Split")
    
    # 70% train, 10% val, 20% test
    X_train, X_temp, y_train, y_temp = time_based_split(X, y, test_size=0.3)
    X_val, X_test, y_val, y_test = time_based_split(
        X_temp, y_temp, test_size=(0.2/0.3)
    )
    
    print(f"Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    del X, y, X_temp, y_temp
    gc.collect()
    
    # ============================================
    # Load Phase 1 Baseline
    # ============================================
    print_section("Phase 1 Baseline")
    phase1_results = pd.read_csv('results/phase1_linear_results.csv')
    phase1_best = phase1_results[phase1_results['horizon'] == TARGET_HORIZON].iloc[0]
    baseline_r2 = phase1_best['test_r2']
    
    print(f"Phase 1 Best (Linear): R² = {baseline_r2*100:.4f}%")
    print(f"Goal: Beat this with Gradient Boosting!")
    
    # ============================================
    # Gradient Boosting Configurations
    # ============================================
    print_section("Gradient Boosting Configurations")
    
    configs = get_gradient_boosting_configs()
    
    for name, params in configs.items():
        print(f"  • {name}:")
        print(f"      Iterations: {params['max_iter']}, "
              f"Depth: {params['max_depth']}, "
              f"LR: {params['learning_rate']}, "
              f"Early stop: {params['n_iter_no_change']} iters")
    
    # ============================================
    # Train Models
    # ============================================
    all_results = []
    
    for model_name, params in configs.items():
        print_section(f"Training: {model_name}")
        
        print(f"Configuration:")
        print(f"  Max iterations: {params['max_iter']}")
        print(f"  Max depth: {params['max_depth']}")
        print(f"  Learning rate: {params['learning_rate']}")
        print(f"  Max bins: {params['max_bins']}")
        print(f"  Early stopping: {params['early_stopping']} ({params['n_iter_no_change']} iters)")
        
        start_time = time.time()
        
        # Create FAST HistGradientBoosting model
        model = HistGradientBoostingRegressor(**params)
        
        print(f"\nTraining HistGradientBoosting (FAST version)...")
        print(f"(Expected time: {['2-5', '5-10', '10-20'][list(configs.keys()).index(model_name)]} minutes on full data)")
        
        # Train - HistGradientBoosting handles validation internally
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Check iterations used
        actual_iterations = model.n_iter_
        print(f"\n✅ Training completed in {format_duration(train_time)}")
        print(f"   Requested iterations: {params['max_iter']}")
        print(f"   Actual iterations used: {actual_iterations}")
        
        if actual_iterations < params['max_iter']:
            print(f"   🎯 Early stopping triggered! Saved {params['max_iter'] - actual_iterations} iterations")
        
        # ============================================
        # Evaluate
        # ============================================
        print("\n--- Evaluation ---")
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        train_metrics = evaluate_model(y_train, y_train_pred, 'train')
        val_metrics = evaluate_model(y_val, y_val_pred, 'val')
        test_metrics = evaluate_model(y_test, y_test_pred, 'test')
        
        # Calculate directional accuracy
        train_dir_acc = np.mean(np.sign(y_train) == np.sign(y_train_pred)) * 100
        val_dir_acc = np.mean(np.sign(y_val) == np.sign(y_val_pred)) * 100
        test_dir_acc = np.mean(np.sign(y_test) == np.sign(y_test_pred)) * 100
        
        train_metrics['train_directional_accuracy'] = train_dir_acc
        val_metrics['val_directional_accuracy'] = val_dir_acc
        test_metrics['test_directional_accuracy'] = test_dir_acc
        
        print_metrics(train_metrics, 'Train')
        print_metrics(val_metrics, 'Val')
        print_metrics(test_metrics, 'Test')
        
        print(f"\n--- Directional Accuracy ---")
        print(f"Train: {train_dir_acc:.2f}%")
        print(f"Val:   {val_dir_acc:.2f}%")
        print(f"Test:  {test_dir_acc:.2f}%")
        
        # ============================================
        # Compare to Phase 1 Baseline
        # ============================================
        improvement = test_metrics['test_r2'] - baseline_r2
        pct_improvement = (improvement / baseline_r2) * 100
        
        print(f"\n--- vs Phase 1 Linear Baseline ---")
        print(f"Phase 1 R²:  {baseline_r2*100:.4f}%")
        print(f"Phase 3 R²:  {test_metrics['test_r2']*100:.4f}%")
        print(f"Improvement: {improvement*100:+.4f}% ({pct_improvement:+.1f}%)")
        
        if test_metrics['test_r2'] > baseline_r2:
            print("✅ Gradient Boosting beats linear baseline!")
        else:
            print("⚠️  Gradient Boosting didn't improve over linear")
        
        # ============================================
        # Feature Importance
        # ============================================
        print("\n--- Top 10 Feature Importance ---")
        
        # HistGradientBoosting uses different method for feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': FEATURE_COLS,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(importance_df.head(10).to_string(index=False))
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['feature'][:10], 
                     importance_df['importance'][:10])
            plt.xlabel('Importance')
            plt.title(f'{model_name} - Feature Importance')
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f'phase3_{model_name.lower()}_feature_importance.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Feature importance plot saved")
        else:
            print("⚠️  Feature importance not available for this model")
        
        # ============================================
        # Generalization Check
        # ============================================
        train_r2 = train_metrics['train_r2']
        test_r2 = test_metrics['test_r2']
        r2_gap = train_r2 - test_r2
        
        print(f"\n--- Generalization Check ---")
        print(f"Train R²: {train_r2*100:.4f}%")
        print(f"Test R²:  {test_r2*100:.4f}%")
        print(f"Gap:      {r2_gap*100:.4f}%")
        
        if r2_gap < 0.2:
            print("✅ Excellent generalization")
        elif r2_gap < 0.5:
            print("✅ Good generalization")
        elif r2_gap < 1.0:
            print("⚠️  Mild overfitting")
        else:
            print("⚠️  Significant overfitting")
            print("    Consider: reduce max_depth, increase min_samples_leaf, lower learning_rate")
        
        # ============================================
        # Save Results
        # ============================================
        results = {
            'phase': 'phase3_gradientboosting',
            'model': model_name,
            'horizon': TARGET_HORIZON,
            'n_features': len(FEATURE_COLS),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_time_sec': train_time,
            'max_iter_requested': params['max_iter'],
            'n_iter_actual': actual_iterations,
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'max_bins': params['max_bins'],
            'baseline_r2': baseline_r2,
            'improvement_over_baseline': improvement,
            **train_metrics,
            **val_metrics,
            **test_metrics
        }
        
        all_results.append(results)
        
        # Save model
        model_path = MODELS_DIR / f'phase3_{model_name.lower()}.pkl'
        save_model(model, model_path)
        
        # Clean up
        del y_train_pred, y_val_pred, y_test_pred
        gc.collect()
        
        print("\n" + "-"*60)
    
    # ============================================
    # Summary
    # ============================================
    print_section("PHASE 3 SUMMARY")
    
    results_df = pd.DataFrame(all_results)
    
    print("\nModel Comparison:")
    comparison = results_df[['model', 'test_r2', 'test_directional_accuracy',
                            'improvement_over_baseline', 'train_time_sec', 'n_iter_actual']].copy()
    comparison['test_r2'] = comparison['test_r2'] * 100
    comparison['improvement_over_baseline'] = comparison['improvement_over_baseline'] * 100
    comparison = comparison.sort_values('test_r2', ascending=False)
    
    print(comparison.to_string(index=False))
    
    # Best model
    best_idx = results_df['test_r2'].idxmax()
    best_model = results_df.loc[best_idx, 'model']
    best_r2 = results_df.loc[best_idx, 'test_r2']
    best_time = results_df.loc[best_idx, 'train_time_sec']
    best_dir_acc = results_df.loc[best_idx, 'test_directional_accuracy']
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Test R²: {best_r2*100:.4f}%")
    print(f"   Directional Accuracy: {best_dir_acc:.2f}%")
    print(f"   Train time: {format_duration(best_time)}")
    print(f"   Improvement over baseline: {(best_r2-baseline_r2)*100:+.4f}%")
    
    # Target check
    target_r2 = 0.014  # 1.4%
    print(f"\n📊 Target: {target_r2*100:.2f}%")
    print(f"   Current: {best_r2*100:.4f}%")
    if best_r2 >= target_r2:
        print("   ✅ TARGET ACHIEVED!")
    else:
        print(f"   📈 Gap: {(target_r2-best_r2)*100:.4f}%")
    
    # Save results
    results_path = RESULTS_DIR / 'phase3_gradientboosting_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to {results_path}")
    
    # ============================================
    # Cross-Phase Comparison
    # ============================================
    print_section("Cross-Phase Comparison")
    
    # Extract best R² from each phase
    print("\n--- Performance Across Phases ---")
    print(f"Phase 1 (Linear):            {baseline_r2*100:.4f}%")
    
    # Phase 2
    try:
        phase2_results = pd.read_csv('results/phase2_randomforest_optimized_results.csv')
        phase2_best = phase2_results.loc[phase2_results['test_r2'].idxmax()]
        print(f"Phase 2 (Random Forest):     {phase2_best['test_r2']*100:.4f}%")
        
        # Phase 3 (current)
        print(f"Phase 3 (Gradient Boosting): {best_r2*100:.4f}%")
        
        # Summary
        print("\n--- Summary ---")
        if best_r2 > phase2_best['test_r2']:
            improvement_over_rf = (best_r2 - phase2_best['test_r2']) * 100
            print(f"✅ Phase 3 beats Phase 2 by {improvement_over_rf:+.4f}%")
        else:
            print(f"⚠️  Phase 2 still ahead by {(phase2_best['test_r2'] - best_r2)*100:.4f}%")
            
        if best_r2 > baseline_r2:
            improvement_over_baseline = (best_r2 - baseline_r2) * 100
            print(f"✅ Phase 3 beats Phase 1 by {improvement_over_baseline:+.4f}%")
            
    except FileNotFoundError:
        print(f"Phase 2 (Random Forest):     Not found")
        print(f"Phase 3 (Gradient Boosting): {best_r2*100:.4f}%")
        print("\n⚠️  Phase 2 results not found - run Phase 2 first for comparison")
    
    # ============================================
    # Next Steps
    # ============================================
    print_section("Next Steps")
    
    if SAMPLE_SIZE is None:
        print("1. Training completed on full 11.7M dataset ✅")
        print("2. Try Phase 4 (XGBoost) for potentially faster training")
        print("3. Consider ensemble methods (combine GB + RF)")
        print("4. Feature engineering based on importance plots")
    else:
        print(f"1. Currently using {SAMPLE_SIZE:,} samples")
        print("2. Set SAMPLE_SIZE=None to train on full 11.7M")
        print("3. Expected training time will increase proportionally")
        print("4. Consider trying XGBoost (Phase 4) for faster full-data training")
    
    print("\n--- Speed Comparison Guide ---")
    print("Phase 2 (Random Forest): Slowest but highly parallel")
    print("Phase 3 (Gradient Boosting): Medium speed, sequential learning")
    print("Phase 4 (XGBoost): Fastest, optimized GB implementation")
    
    print_section("PHASE 3 COMPLETE")
    
    return results_df

# ============================================
# Run
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("HISTGRADIENT BOOSTING TRAINING (FAST VERSION)")
    print("=" * 60)
    sample_display = "All samples (11.7M)" if SAMPLE_SIZE is None else f"{SAMPLE_SIZE:,}"
    print(f"Sample size: {sample_display}")
    print(f"Memory optimization: {ENABLE_MEMORY_OPTIMIZATION}")
    print(f"Early stopping: Enabled")
    print(f"Algorithm: HistGradientBoostingRegressor (10-100x faster!)")
    print("=" * 60)
    print()
    
    print("ℹ️  HistGradientBoosting Info (FAST VERSION):")
    print("  • 10-100x faster than standard GradientBoostingRegressor")
    print("  • Uses histogram binning like XGBoost")
    print("  • Sequential learning (learns from previous errors)")
    print("  • Built-in early stopping")
    print("  • Native scikit-learn implementation")
    print("=" * 60)
    print()
    
    results = train_gradient_boosting()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nHistGradientBoosting vs Other Methods:")
    print("  ✓ 10-100x faster than standard GradientBoosting")
    print("  ✓ Similar speed to XGBoost")
    print("  ✓ More accurate than Random Forest (usually)")
    print("  ✓ Native scikit-learn (no extra dependencies)")
    print("  ✓ Handles missing values automatically")
    print("=" * 60)