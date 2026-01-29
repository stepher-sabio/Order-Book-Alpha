"""
phase1_linear.py - Baseline Linear Models
Compares: LinearRegression, Ridge, Lasso
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import time
from pathlib import Path

# Import our utilities
from utils import (
    load_data, 
    time_based_split,
    evaluate_model,
    print_metrics,
    time_series_cv,
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
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# For quick testing, set to a small number (e.g., 100000)
# For full training, set to None
SAMPLE_SIZE = None  # Use all data

# Cross-validation settings
USE_CV = False  # Set to True to enable CV (slower but more robust)
CV_SPLITS = 3

# Target horizon to focus on
TARGET_HORIZON = 'return_50ms'  # Start with shortest (hardest) horizon

# ============================================
# Main Training Function
# ============================================
def train_linear_models():
    """Train and compare linear models"""
    
    print_section("PHASE 1: LINEAR MODELS")
    
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
    print(f"y shape: {y.shape}")
    
    # ============================================
    # Train/Test Split
    # ============================================
    X_train, X_test, y_train, y_test = time_based_split(X, y, test_size=0.2)
    
    # ============================================
    # Define Models
    # ============================================
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=1e-6, max_iter=5000, random_state=42)
    }
    
    print_section("Models to Train")
    for name in models.keys():
        print(f"  • {name}")
    
    # ============================================
    # Train Each Model
    # ============================================
    all_results = []
    
    for model_name, model in models.items():
        print_section(f"Training: {model_name}")
        
        start_time = time.time()
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Optional: Cross-validation
        if USE_CV:
            cv_scores = time_series_cv(pipeline, X_train, y_train, n_splits=CV_SPLITS)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
        else:
            cv_mean, cv_std = None, None
        
        # Train on full training set
        print("\nTraining on full training set...")
        pipeline.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        print(f"Training completed in {format_duration(train_time)}")
        
        # ============================================
        # Evaluate
        # ============================================
        print("\n--- Evaluation ---")
        
        # Predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        train_metrics = evaluate_model(y_train, y_train_pred, 'train')
        test_metrics = evaluate_model(y_test, y_test_pred, 'test')
        
        # Print results
        print_metrics(train_metrics, 'Train')
        print_metrics(test_metrics, 'Test')
        
        # ============================================
        # Feature Importance
        # ============================================
        print("\n--- Feature Importance ---")
        feature_importance = get_feature_importance(
            pipeline.named_steps['model'], 
            FEATURE_COLS, 
            top_n=10
        )
        
        # ============================================
        # Check for overfitting
        # ============================================
        train_r2 = train_metrics['train_r2']
        test_r2 = test_metrics['test_r2']
        r2_gap = train_r2 - test_r2
        
        print(f"\n--- Overfitting Check ---")
        print(f"Train R²: {train_r2*100:.4f}%")
        print(f"Test R²:  {test_r2*100:.4f}%")
        print(f"Gap:      {r2_gap*100:.4f}%")
        
        if r2_gap > 0.005:  # 0.5% gap
            print("⚠️  Potential overfitting (train >> test)")
        else:
            print("✅ Good generalization")
        
        # ============================================
        # Save Results
        # ============================================
        results = {
            'phase': 'phase1_linear',
            'model': model_name,
            'horizon': TARGET_HORIZON,
            'n_features': len(FEATURE_COLS),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_time_sec': train_time,
            **train_metrics,
            **test_metrics
        }
        
        if USE_CV:
            results['cv_mean_r2'] = cv_mean
            results['cv_std_r2'] = cv_std
        
        all_results.append(results)
        
        # Save individual model
        model_path = MODELS_DIR / f'phase1_{model_name.lower()}.pkl'
        save_model(pipeline, model_path)
        
        print("\n" + "-"*60)
    
    # ============================================
    # Save All Results
    # ============================================
    print_section("Saving Results")
    results_df = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / 'phase1_linear_results.csv'
    save_results(all_results[0], results_path)  # Save first result
    
    for result in all_results[1:]:  # Append rest
        save_results(result, results_path)
    
    # ============================================
    # Summary Comparison
    # ============================================
    print_section("PHASE 1 SUMMARY")
    
    print("\nModel Comparison (Test Set):")
    comparison = results_df[['model', 'test_r2', 'test_mae_bps', 'test_direction_acc', 'train_time_sec']].copy()
    comparison['test_r2'] = comparison['test_r2'] * 100  # Convert to percentage
    comparison['test_direction_acc'] = comparison['test_direction_acc'] * 100
    comparison = comparison.sort_values('test_r2', ascending=False)
    
    print(comparison.to_string(index=False))
    
    # Find best model
    best_idx = results_df['test_r2'].idxmax()
    best_model = results_df.loc[best_idx, 'model']
    best_r2 = results_df.loc[best_idx, 'test_r2']
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Test R²: {best_r2*100:.4f}%")
    
    # Interpretation
    print("\n--- Interpretation ---")
    if best_r2 > 0.014:  # 1.4%
        print("✅ EXCELLENT! Exceeded 1.4% R² target with linear model!")
        print("   This suggests strong linear relationships in features.")
    elif best_r2 > 0.010:  # 1.0%
        print("✅ GOOD! Above 1% R². Nonlinear models should push higher.")
    elif best_r2 > 0.005:  # 0.5%
        print("⚠️  MODERATE. Baseline established. Expect improvement with trees.")
    else:
        print("⚠️  LOW. May need feature engineering or data quality check.")
    
    print("\n--- Next Steps ---")
    print("1. Review feature importance from best model")
    print("2. Consider dropping low-importance features")
    print("3. Proceed to Phase 2: Random Forest (nonlinear)")
    
    print_section("PHASE 1 COMPLETE")
    
    return results_df

# ============================================
# Run if executed directly
# ============================================
if __name__ == "__main__":
    results = train_linear_models()