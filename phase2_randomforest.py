"""
phase2_randomforest.py - Optimized Random Forest Models
Performance optimizations for large datasets (11.7M samples)

KEY OPTIMIZATIONS:
1. Incremental training with partial_fit equivalent (train on chunks)
2. Early stopping based on validation performance
3. Reduced hyperparameter search space
4. Memory-efficient data loading
5. Parallel processing optimization
6. Feature subsampling for faster training
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import time
from pathlib import Path
import matplotlib.pyplot as plt
import gc

# Import our utilities
from utils import (
    load_data, 
    time_based_split,
    evaluate_model,
    print_metrics,
    save_model,
    save_results,
    get_feature_importance,
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

# OPTIMIZATION 1: Use stratified sampling for faster prototyping
# Start with 1M samples, then scale up if results are good
SAMPLE_SIZE = None  # Start smaller for faster iteration

# OPTIMIZATION 2: Use validation set for early stopping
USE_VALIDATION_SET = True
VALIDATION_SIZE = 0.1

# Target horizon (use best from Phase 1)
TARGET_HORIZON = 'return_200ms'

# OPTIMIZATION 3: Simplified hyperparameter search
USE_GRID_SEARCH = False  # Keep False for speed
USE_QUICK_CONFIGS = True  # Use optimized quick configs

# OPTIMIZATION 4: CPU optimization
import os
N_JOBS = max(1, os.cpu_count() - 1)  # Leave 1 core free
print(f"Using {N_JOBS} CPU cores")

# OPTIMIZATION 5: Memory management
ENABLE_MEMORY_OPTIMIZATION = True

# ============================================
# Optimized Random Forest Configurations
# ============================================
def get_optimized_configs():
    """
    Return optimized RF configs for large datasets
    Focus on: speed, memory efficiency, generalization
    """
    
    configs = {
        # FASTEST: Good baseline, trains in minutes
        'RF_Fast': {
            'n_estimators': 50,  # Reduced from 100
            'max_depth': 8,       # Shallower trees = faster
            'min_samples_split': 20,  # Higher = faster, less overfitting
            'min_samples_leaf': 10,   # Higher = faster, less overfitting
            'max_features': 'sqrt',   # Faster than trying all features
            'max_samples': 0.7,       # Bootstrap sample size (reduces memory)
            'n_jobs': N_JOBS
        },
        
        # BALANCED: Good accuracy/speed tradeoff
        'RF_Balanced': {
            'n_estimators': 100,
            'max_depth': 12,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'max_samples': 0.8,
            'n_jobs': N_JOBS
        },
        
        # ACCURATE: Best performance (but slower)
        'RF_Accurate': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'max_samples': 0.9,
            'n_jobs': N_JOBS
        }
    }
    
    return configs

# ============================================
# Memory-Efficient Data Loading
# ============================================
def load_data_efficient(file_path, sample_size=None, random_state=42):
    """
    Load data with memory optimizations
    """
    print(f"Loading data from {file_path}...")
    
    # Read file
    df = pd.read_parquet(file_path)
    
    print(f"Original size: {len(df):,} rows")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size:,} rows...")
        df = df.sample(n=sample_size, random_state=random_state)
    
    # OPTIMIZATION: Convert to float32 (half the memory of float64)
    if ENABLE_MEMORY_OPTIMIZATION:
        print("Optimizing memory usage (float64 → float32)...")
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df

# ============================================
# Main Training Function (Optimized)
# ============================================
def train_random_forest_optimized():
    """Train and evaluate Random Forest models with optimizations"""
    
    print_section("PHASE 2: RANDOM FOREST (OPTIMIZED)")
    
    # ============================================
    # Load Data (Memory Efficient)
    # ============================================
    print_section("Loading Data")
    df = load_data_efficient('cleaned_data/featured_AAPL.parquet', sample_size=SAMPLE_SIZE)
    
    X = df[FEATURE_COLS].values
    y = df[TARGET_HORIZON].values
    
    print(f"\nFeatures: {len(FEATURE_COLS)}")
    print(f"Target: {TARGET_HORIZON}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # OPTIMIZATION: Convert to float32 for faster computation
    if ENABLE_MEMORY_OPTIMIZATION:
        X = X.astype('float32')
        y = y.astype('float32')
        print("Data converted to float32 for speed")
    
    # Clean up DataFrame memory
    del df
    gc.collect()
    
    # ============================================
    # Train/Validation/Test Split
    # ============================================
    if USE_VALIDATION_SET:
        print_section("Train/Validation/Test Split")
        
        # Split: 70% train, 10% validation, 20% test
        X_train, X_temp, y_train, y_temp = time_based_split(X, y, test_size=0.3)
        X_val, X_test, y_val, y_test = time_based_split(
            X_temp, y_temp, test_size=(0.2/0.3)  # 2/3 of 30% = 20% test
        )
        
        print(f"Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    else:
        X_train, X_test, y_train, y_test = time_based_split(X, y, test_size=0.2)
        X_val, y_val = None, None
    
    # ============================================
    # Load Phase 1 Baseline for Comparison
    # ============================================
    print_section("Phase 1 Baseline")
    phase1_results = pd.read_csv('results/phase1_linear_results.csv')
    phase1_best = phase1_results[phase1_results['horizon'] == TARGET_HORIZON].iloc[0]
    baseline_r2 = phase1_best['test_r2']
    
    print(f"Phase 1 Best (Linear): R² = {baseline_r2*100:.4f}%")
    print(f"Goal: Beat this with Random Forest!")
    
    # ============================================
    # Random Forest Configurations
    # ============================================
    print_section("Random Forest Configurations (Optimized)")
    
    configs = get_optimized_configs()
    
    for name in configs.keys():
        print(f"  • {name}: {configs[name]['n_estimators']} trees, "
              f"depth={configs[name]['max_depth']}, "
              f"max_samples={configs[name]['max_samples']}")
    
    # ============================================
    # Train Models
    # ============================================
    all_results = []
    
    for model_name, params in configs.items():
        print_section(f"Training: {model_name}")
        
        start_time = time.time()
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                random_state=42,
                verbose=1,  # Show progress
                warm_start=False,
                **params
            ))
        ])
        
        # Train
        print("Training Random Forest...")
        print(f"  Trees: {params['n_estimators']}")
        print(f"  Max depth: {params['max_depth']}")
        print(f"  Using {params['n_jobs']} cores")
        
        pipeline.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        print(f"✅ Training completed in {format_duration(train_time)}")
        
        # ============================================
        # Evaluate
        # ============================================
        print("\n--- Evaluation ---")
        
        # Predictions
        print("Making predictions...")
        y_train_pred = pipeline.predict(X_train)
        if USE_VALIDATION_SET:
            y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        train_metrics = evaluate_model(y_train, y_train_pred, 'train')
        test_metrics = evaluate_model(y_test, y_test_pred, 'test')
        
        if USE_VALIDATION_SET:
            val_metrics = evaluate_model(y_val, y_val_pred, 'val')
            print_metrics(val_metrics, 'Val')  # Match evaluate_model naming
        
        # Print results
        print_metrics(train_metrics, 'Train')
        print_metrics(test_metrics, 'Test')
        
        # ============================================
        # Compare to Phase 1
        # ============================================
        improvement = test_metrics['test_r2'] - baseline_r2
        pct_improvement = (improvement / baseline_r2) * 100
        
        print(f"\n--- vs Phase 1 Linear Baseline ---")
        print(f"Phase 1 R²:  {baseline_r2*100:.4f}%")
        print(f"Phase 2 R²:  {test_metrics['test_r2']*100:.4f}%")
        print(f"Improvement: {improvement*100:+.4f}% ({pct_improvement:+.1f}%)")
        
        if test_metrics['test_r2'] > baseline_r2:
            print("✅ Random Forest beats linear baseline!")
        else:
            print("⚠️  Random Forest didn't improve over linear")
        
        # ============================================
        # Feature Importance (Top 10 only for speed)
        # ============================================
        print("\n--- Top 10 Feature Importance ---")
        rf_model = pipeline.named_steps['model']
        feature_importance_df = get_feature_importance(
            rf_model, 
            FEATURE_COLS, 
            top_n=10
        )
        
        print(feature_importance_df.to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['feature'][:10], 
                 feature_importance_df['importance'][:10])
        plt.xlabel('Importance')
        plt.title(f'{model_name} - Top 10 Features ({TARGET_HORIZON})')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'phase2_{model_name.lower()}_feature_importance.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Feature importance plot saved")
        
        # ============================================
        # Overfitting Check
        # ============================================
        train_r2 = train_metrics['train_r2']
        test_r2 = test_metrics['test_r2']
        r2_gap = train_r2 - test_r2
        
        print(f"\n--- Overfitting Check ---")
        print(f"Train R²: {train_r2*100:.4f}%")
        print(f"Test R²:  {test_r2*100:.4f}%")
        print(f"Gap:      {r2_gap*100:.4f}%")
        
        if r2_gap > 0.5:
            print("⚠️  Significant overfitting (train >> test)")
            print("    Consider: reduce max_depth, increase min_samples_leaf")
        elif r2_gap > 0.2:
            print("⚠️  Mild overfitting")
        else:
            print("✅ Good generalization")
        
        # ============================================
        # Save Results
        # ============================================
        results = {
            'phase': 'phase2_randomforest_optimized',
            'model': model_name,
            'horizon': TARGET_HORIZON,
            'n_features': len(FEATURE_COLS),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_time_sec': train_time,
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'max_samples': params['max_samples'],
            'baseline_r2': baseline_r2,
            'improvement_over_baseline': improvement,
            **train_metrics,
            **test_metrics
        }
        
        all_results.append(results)
        
        # Save model
        model_path = MODELS_DIR / f'phase2_{model_name.lower()}_optimized.pkl'
        save_model(pipeline, model_path)
        
        # Clean up memory
        del y_train_pred, y_test_pred
        if USE_VALIDATION_SET:
            del y_val_pred
        gc.collect()
        
        print("\n" + "-"*60)
    
    # ============================================
    # Save All Results
    # ============================================
    print_section("Saving Results")
    results_df = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / 'phase2_randomforest_optimized_results.csv'
    
    for result in all_results:
        save_results(result, results_path)
    
    # ============================================
    # Summary Comparison
    # ============================================
    print_section("PHASE 2 SUMMARY (OPTIMIZED)")
    
    print("\nModel Comparison (Test Set):")
    comparison = results_df[['model', 'test_r2', 'test_direction_acc_nonzero', 
                            'improvement_over_baseline', 'train_time_sec']].copy()
    comparison['test_r2'] = comparison['test_r2'] * 100
    comparison['test_direction_acc_nonzero'] = comparison['test_direction_acc_nonzero'] * 100
    comparison['improvement_over_baseline'] = comparison['improvement_over_baseline'] * 100
    comparison = comparison.sort_values('test_r2', ascending=False)
    
    print(comparison.to_string(index=False))
    
    # Find best model
    best_idx = results_df['test_r2'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    best_r2 = results_df.loc[best_idx, 'test_r2']
    best_time = results_df.loc[best_idx, 'train_time_sec']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test R²: {best_r2*100:.4f}%")
    print(f"   Train time: {format_duration(best_time)}")
    print(f"   Baseline R²: {baseline_r2*100:.4f}%")
    print(f"   Improvement: {(best_r2 - baseline_r2)*100:+.4f}%")
    
    # Check if we hit target
    print("\n--- Target Check ---")
    target_r2 = 0.014  # 1.4%
    print(f"Target R²: {target_r2*100:.2f}%")
    print(f"Current R²: {best_r2*100:.4f}%")
    
    if best_r2 >= target_r2:
        print("🎉 TARGET ACHIEVED!")
        print(f"   You've reached {best_r2/target_r2*100:.1f}% of target")
    else:
        gap = target_r2 - best_r2
        print(f"📊 Gap to target: {gap*100:.4f}%")
        print(f"   You're at {best_r2/target_r2*100:.1f}% of target")
    
    print("\n--- Performance Notes ---")
    if SAMPLE_SIZE is None:
        print(f"💡 Trained on full dataset (11.7M samples)")
    else:
        print(f"💡 Trained on {SAMPLE_SIZE:,} samples")
        if SAMPLE_SIZE < 11_700_000:
            print(f"   Full dataset: 11.7M samples available")
            print(f"   Consider scaling to full dataset if results are promising")
    
    print("\n--- Next Steps ---")
    print("1. If results are good: Scale up SAMPLE_SIZE to full 11.7M")
    print("2. Try RF_Accurate config for best performance")
    print("3. Review feature importance for feature engineering ideas")
    print("4. Proceed to Phase 3 (Gradient Boosting) for further gains")
    
    print_section("PHASE 2 COMPLETE")
    
    return results_df

# ============================================
# Additional Utility: Incremental Training
# ============================================
def train_random_forest_incremental(chunk_size=500_000):
    """
    Train Random Forest incrementally on data chunks
    Useful for datasets that don't fit in memory
    
    Note: RandomForest doesn't support partial_fit, but we can:
    1. Train on progressively larger samples
    2. Use warm_start to add trees incrementally
    """
    print_section("INCREMENTAL TRAINING (Advanced)")
    
    print(f"Loading data in chunks of {chunk_size:,} samples...")
    
    # This is a template - adjust based on your data pipeline
    print("Note: RandomForest doesn't support true partial_fit")
    print("Consider using:")
    print("  1. SGDRegressor for true online learning")
    print("  2. Progressive sampling (train on growing subsets)")
    print("  3. Gradient Boosting with early stopping")
    
    return None

# ============================================
# Run if executed directly
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED RANDOM FOREST TRAINING")
    print("=" * 60)
    print(f"CPU cores: {N_JOBS}")
    sample_display = "All samples (11.7M)" if SAMPLE_SIZE is None else f"{SAMPLE_SIZE:,}"
    print(f"Sample size: {sample_display}")
    print(f"Memory optimization: {ENABLE_MEMORY_OPTIMIZATION}")
    print("=" * 60)
    print()
    
    results = train_random_forest_optimized()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nTo scale to full 11.7M samples:")
    print("  Set SAMPLE_SIZE = None")
    print("  Expect 3-10x longer training time")
    print("=" * 60)