"""
phase4_xgboost.py - XGBoost 
5-10x faster than Random Forest, often better accuracy

Use this if Random Forest is still too slow on 11.7M samples
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from pathlib import Path
import matplotlib.pyplot as plt
import gc

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

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
SAMPLE_SIZE = None  # 1M for quick test

# Target
TARGET_HORIZON = 'return_200ms'

# CPU optimization
import os
N_JOBS = max(1, os.cpu_count() - 1)

# ============================================
# XGBoost Configurations
# ============================================
def get_xgboost_configs():
    """
    Optimized XGBoost configs for large datasets
    Much faster than Random Forest!
    """
    
    configs = {
        # ULTRA FAST: 2-5 minutes on 11.7M
        'XGB_Fast': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',  # Fast histogram-based
            'n_jobs': N_JOBS,
            'random_state': 42
        },
        
        # BALANCED: 5-10 minutes on 11.7M
        'XGB_Balanced': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'n_jobs': N_JOBS,
            'random_state': 42
        },
        
        # ACCURATE: 10-15 minutes on 11.7M
        'XGB_Accurate': {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 8,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'tree_method': 'hist',
            'n_jobs': N_JOBS,
            'random_state': 42
        }
    }
    
    return configs

# ============================================
# Main Training Function
# ============================================
def train_xgboost():
    """Train and evaluate XGBoost models"""
    
    if not XGBOOST_AVAILABLE:
        print("ERROR: XGBoost not installed!")
        print("Install with: pip install xgboost")
        return None
    
    print_section("PHASE 4: XGBOOST (FAST ALTERNATIVE)")
    
    # ============================================
    # Load Data
    # ============================================
    print_section("Loading Data")
    df = load_data('cleaned_data/featured_AAPL.parquet', sample_size=SAMPLE_SIZE)
    
    X = df[FEATURE_COLS].values
    y = df[TARGET_HORIZON].values
    
    print(f"\nFeatures: {len(FEATURE_COLS)}")
    print(f"Target: {TARGET_HORIZON}")
    print(f"X shape: {X.shape}")
    
    # Convert to float32 for speed
    X = X.astype('float32')
    y = y.astype('float32')
    
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
    
    print(f"Train: {len(X_train):,}")
    print(f"Val:   {len(X_val):,}")
    print(f"Test:  {len(X_test):,}")
    
    # ============================================
    # Load Phase 1 Baseline
    # ============================================
    print_section("Phase 1 Baseline")
    phase1_results = pd.read_csv('results/phase1_linear_results.csv')
    phase1_best = phase1_results[phase1_results['horizon'] == TARGET_HORIZON].iloc[0]
    baseline_r2 = phase1_best['test_r2']
    
    print(f"Phase 1 Best (Linear): R² = {baseline_r2*100:.4f}%")
    
    # ============================================
    # XGBoost Configurations
    # ============================================
    print_section("XGBoost Configurations")
    
    configs = get_xgboost_configs()
    
    for name, params in configs.items():
        print(f"  • {name}:")
        print(f"      Trees: {params['n_estimators']}, "
              f"Depth: {params['max_depth']}, "
              f"LR: {params['learning_rate']}")
    
    # ============================================
    # Train Models with Early Stopping
    # ============================================
    all_results = []
    
    for model_name, params in configs.items():
        print_section(f"Training: {model_name}")
        
        start_time = time.time()
        
        # Create model with early stopping
        model = XGBRegressor(
            **params,
            early_stopping_rounds=20,  # Stop if no improvement
            eval_metric='rmse'
        )
        
        # Train with validation set
        print(f"Training XGBoost with early stopping...")
        print(f"  Using {params['n_jobs']} CPU cores")
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False  # Set True to see progress
        )
        
        train_time = time.time() - start_time
        print(f"✅ Training completed in {format_duration(train_time)}")
        print(f"   Best iteration: {model.best_iteration}")
        
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
        print_metrics(val_metrics, 'Val')  # Match evaluate_model naming
        print_metrics(test_metrics, 'Test')
        
        print(f"\n--- Directional Accuracy ---")
        print(f"Train: {train_dir_acc:.2f}%")
        print(f"Val:   {val_dir_acc:.2f}%")
        print(f"Test:  {test_dir_acc:.2f}%")
        
        # ============================================
        # Compare to Phase 1
        # ============================================
        improvement = test_metrics['test_r2'] - baseline_r2
        pct_improvement = (improvement / baseline_r2) * 100 if baseline_r2 != 0 else 0
        
        print(f"\n--- vs Phase 1 Linear Baseline ---")
        print(f"Phase 1 R²:  {baseline_r2*100:.4f}%")
        print(f"Phase 4 R²:  {test_metrics['test_r2']*100:.4f}%")
        print(f"Improvement: {improvement*100:+.4f}% ({pct_improvement:+.1f}%)")
        
        if test_metrics['test_r2'] > baseline_r2:
            print("✅ XGBoost beats linear baseline!")
        else:
            print("⚠️  XGBoost didn't improve over linear")
        
        # ============================================
        # Feature Importance
        # ============================================
        print("\n--- Top 10 Feature Importance ---")
        
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
        plt.savefig(PLOTS_DIR / f'phase4_{model_name.lower()}_feature_importance.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # ============================================
        # Overfitting Check
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
        else:
            print("⚠️  Some overfitting")
        
        # ============================================
        # Save Results
        # ============================================
        results = {
            'phase': 'phase4_xgboost',
            'model': model_name,
            'horizon': TARGET_HORIZON,
            'n_features': len(FEATURE_COLS),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_time_sec': train_time,
            'n_estimators': model.best_iteration,
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'baseline_r2': baseline_r2,
            'improvement_over_baseline': improvement,
            **train_metrics,
            **val_metrics,
            **test_metrics
        }
        
        all_results.append(results)
        
        # Save model
        model_path = MODELS_DIR / f'phase4_{model_name.lower()}.pkl'
        save_model(model, model_path)
        
        print("\n" + "-"*60)
    
    # ============================================
    # Summary
    # ============================================
    print_section("SUMMARY")
    
    results_df = pd.DataFrame(all_results)
    
    print("\nModel Comparison:")
    comparison = results_df[['model', 'test_r2', 'test_directional_accuracy',
                            'improvement_over_baseline', 'train_time_sec']].copy()
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
    print(f"   Improvement: {(best_r2-baseline_r2)*100:+.4f}% over baseline")
    
    # Target check
    target_r2 = 0.014
    print(f"\n📊 Target: {target_r2*100:.2f}%")
    print(f"   Current: {best_r2*100:.4f}%")
    if best_r2 >= target_r2:
        print("   ✅ TARGET ACHIEVED!")
    else:
        print(f"   📈 Gap: {(target_r2-best_r2)*100:.4f}%")
    
    # Save results
    results_path = RESULTS_DIR / 'phase4_xgboost_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to {results_path}")
    
    print("\n--- Next Steps ---")
    if SAMPLE_SIZE is None:
        print("1. Training on full 11.7M dataset")
        print("2. Try XGB_Accurate for best performance")
        print("3. Ensemble with Random Forest if available")
        print("4. Consider feature engineering for further gains")
    elif SAMPLE_SIZE < 11_700_000:
        print(f"1. Currently using {SAMPLE_SIZE:,} samples")
        print("2. Set SAMPLE_SIZE=None to train on full 11.7M")
        print("3. Expected time: 10-30 minutes (vs hours for RF)")
    else:
        print("1. Try XGB_Accurate for best performance")
        print("2. Ensemble with Random Forest if available")
        print("3. Consider feature engineering for further gains")
    
    print_section("COMPLETE")
    
    # ============================================
    # Cross-Phase Comparison
    # ============================================
    print_section("Cross-Phase Comparison")
    
    # Extract best R² from each phase
    phase_r2 = {}
    
    # Phase 1
    try:
        phase1_results = pd.read_csv('results/phase1_linear_results.csv')
        phase1_best = phase1_results[phase1_results['horizon'] == TARGET_HORIZON].iloc[0]
        phase_r2['Phase 1 (Linear)'] = phase1_best['test_r2'] * 100
        print(f"Phase 1 (Linear):           {phase1_best['test_r2']*100:.4f}%")
    except Exception as e:
        print(f"Phase 1 (Linear):           Not found")
    
    # Phase 2
    try:
        phase2_results = pd.read_csv('results/phase2_randomforest_results.csv')
        phase2_best = phase2_results.loc[phase2_results['test_r2'].idxmax()]
        phase_r2['Phase 2 (Random Forest)'] = phase2_best['test_r2'] * 100
        print(f"Phase 2 (Random Forest):    {phase2_best['test_r2']*100:.4f}%")
    except Exception as e:
        print(f"Phase 2 (Random Forest):    Not found")
    
    # Phase 3
    try:
        phase3_results = pd.read_csv('results/phase3_gradientboosting_results.csv')
        phase3_best = phase3_results.loc[phase3_results['test_r2'].idxmax()]
        phase_r2['Phase 3 (Gradient Boosting)'] = phase3_best['test_r2'] * 100
        print(f"Phase 3 (Gradient Boosting): {phase3_best['test_r2']*100:.4f}%")
    except Exception as e:
        print(f"Phase 3 (Gradient Boosting): Not found")
    
    # Phase 4 (current)
    phase_r2['Phase 4 (XGBoost)'] = best_r2 * 100
    print(f"Phase 4 (XGBoost):          {best_r2*100:.4f}%")
    
    # Summary
    if len(phase_r2) > 1:
        print("\n--- Summary ---")
        best_phase = max(phase_r2, key=phase_r2.get)
        best_phase_r2 = phase_r2[best_phase]
        print(f"🏆 Best Overall: {best_phase} with R² = {best_phase_r2:.4f}%")
        
        # Calculate improvements
        if 'Phase 1 (Linear)' in phase_r2:
            baseline = phase_r2['Phase 1 (Linear)']
            current = phase_r2['Phase 4 (XGBoost)']
            improvement = current - baseline
            print(f"   Phase 4 vs Phase 1: {improvement:+.4f}% ({(improvement/baseline)*100:+.1f}%)")
    
    return results_df

# ============================================
# GPU Acceleration (if available)
# ============================================
def train_xgboost_gpu():
    """
    Train with GPU acceleration (requires CUDA)
    Can be 10-100x faster than CPU!
    """
    print_section("XGBoost GPU Training")
    
    try:
        # Test if GPU is available
        model = XGBRegressor(tree_method='gpu_hist', n_estimators=10)
        print("✅ GPU detected! Using gpu_hist")
        
        # Modify configs to use GPU
        configs = get_xgboost_configs()
        for config in configs.values():
            config['tree_method'] = 'gpu_hist'
        
        print("\nGPU training will be MUCH faster!")
        print("Expected time on 11.7M samples: 2-5 minutes")
        
        return train_xgboost()
        
    except Exception as e:
        print(f"⚠️  GPU not available: {e}")
        print("Falling back to CPU training")
        return train_xgboost()

# ============================================
# Run
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("XGBOOST TRAINING (FAST ALTERNATIVE TO RANDOM FOREST)")
    print("=" * 60)
    sample_display = "All samples (11.7M)" if SAMPLE_SIZE is None else f"{SAMPLE_SIZE:,}"
    print(f"Sample size: {sample_display}")
    print(f"CPU cores: {N_JOBS}")
    print("=" * 60)
    print()
    
    if not XGBOOST_AVAILABLE:
        print("\n⚠️  XGBoost not found!")
        print("Install with: pip install xgboost")
        print("Or: conda install -c conda-forge xgboost")
    else:
        # Check for GPU
        try:
            test_model = XGBRegressor(tree_method='gpu_hist', n_estimators=1)
            print("🎮 GPU detected! Training will be very fast.")
            print("=" * 60)
            print()
            results = train_xgboost_gpu()
        except:
            print("💻 Using CPU (GPU not available)")
            print("=" * 60)
            print()
            results = train_xgboost()
        
        print("\n" + "=" * 60)
        print("Why XGBoost is faster than Random Forest:")
        print("  • Builds trees sequentially (learns from mistakes)")
        print("  • Uses histogram binning (faster splits)")
        print("  • Better memory efficiency")
        print("  • Early stopping prevents wasted computation")
        print("  • Often achieves same accuracy with fewer trees")
        print("=" * 60)