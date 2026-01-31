"""
utils.py - Shared utilities for order book modeling
UPDATED: 32 features (14 core + 7 elective + 11 research-backed)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
from pathlib import Path

# ============================================
# Feature Configuration (32 features total)
# ============================================

CORE_FEATURES = [
    # Price & Spread (2)
    'spread_bps', 'mid_price_usd',
    # Imbalance (2)
    'imbalance_0', 'cumulative_imbalance',
    # Momentum (3)
    'momentum_5', 'momentum_10', 'momentum_20',
    # Volatility (2)
    'volatility_10', 'volatility_20',
    # Order Flow (2)
    'microprice_mom', 'flow_0',
    # Book Shape (3)
    'bid_slope', 'ask_slope', 'total_depth'
]

ELECTIVE_FEATURES = [
    # Advanced Imbalance (2)
    'imbalance_change', 'persistent_imbalance',
    # Spread Dynamics (2)
    'spread_change', 'spread_vol_ratio',
    # Multi-scale (1)
    'vol_ratio',
    # Interactions (2)
    'imb_vol_interaction', 'spread_mom_interaction'
]

RESEARCH_FEATURES = [
    # Cont et al. (2014) - Multi-level & OFI
    'imbalance_1', 'imbalance_2', 'ofi_simple',
    # Spread quality
    'spread_volatility', 'spread_depth_ratio',
    # Hasbrouck (2009) - Effective spread
    'effective_spread',
    # Dynamics
    'price_acceleration', 'volume_acceleration',
    # Alignment
    'momentum_alignment'
]

# Use all features by default
FEATURE_COLS = CORE_FEATURES + ELECTIVE_FEATURES + RESEARCH_FEATURES

TARGET_HORIZONS = ['return_50ms', 'return_100ms', 'return_200ms', 'return_500ms']

# ============================================
# Data Loading
# ============================================
def load_data(filepath='cleaned_data/featured_AAPL.parquet', sample_size=None):
    """
    Load featured dataset
    
    Args:
        filepath: Path to parquet file
        sample_size: If provided, randomly sample this many rows (for quick testing)
    
    Returns:
        DataFrame
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_parquet(filepath)
    
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size:,} rows for quick testing...")
        df = df.sample(n=sample_size, random_state=42).sort_index()
    
    print(f"Loaded {len(df):,} samples")
    return df

# ============================================
# Train/Test Split
# ============================================
def time_based_split(X, y, test_size=0.2):
    """
    Split data by time (no shuffling!)
    
    Args:
        X: Features
        y: Target
        test_size: Fraction for test set (most recent data)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n=== Time-Based Split ===")
    print(f"Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

# ============================================
# Evaluation Metrics
# ============================================
def evaluate_model(y_true, y_pred, set_name='test'):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        set_name: Name for display (e.g., 'train', 'test')
    
    Returns:
        Dictionary of metrics
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Convert to basis points for interpretability
    mae_bps = mae * 10000
    rmse_bps = rmse * 10000
    
    # Direction accuracy - ALL samples
    direction_acc_all = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    # Direction accuracy - NON-ZERO samples only
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        direction_acc_nonzero = np.mean(
            np.sign(y_true[nonzero_mask]) == np.sign(y_pred[nonzero_mask])
        )
    else:
        direction_acc_nonzero = np.nan
    
    # Correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Percentage of zeros in actual
    pct_zeros = (y_true == 0).sum() / len(y_true)
    
    metrics = {
        f'{set_name}_r2': r2,
        f'{set_name}_mae_bps': mae_bps,
        f'{set_name}_rmse_bps': rmse_bps,
        f'{set_name}_direction_acc_all': direction_acc_all,
        f'{set_name}_direction_acc_nonzero': direction_acc_nonzero,
        f'{set_name}_correlation': correlation,
        f'{set_name}_pct_zeros': pct_zeros
    }
    
    return metrics

def print_metrics(metrics, set_name='Test'):
    """Pretty print metrics"""
    # Normalize set_name to lowercase for key lookup
    set_name_lower = set_name.lower()
    
    print(f"\n{set_name} Metrics:")
    print(f"  R²:                    {metrics[f'{set_name_lower}_r2']*100:.4f}%")
    print(f"  MAE:                   {metrics[f'{set_name_lower}_mae_bps']:.3f} bps")
    print(f"  RMSE:                  {metrics[f'{set_name_lower}_rmse_bps']:.3f} bps")
    print(f"  Direction Acc (all):   {metrics[f'{set_name_lower}_direction_acc_all']*100:.2f}%")
    print(f"  Direction Acc (≠0):    {metrics[f'{set_name_lower}_direction_acc_nonzero']*100:.2f}%")
    print(f"  Correlation:           {metrics[f'{set_name_lower}_correlation']:.4f}")
    print(f"  % Zeros:               {metrics[f'{set_name_lower}_pct_zeros']*100:.2f}%")

# ============================================
# Cross-Validation
# ============================================
def time_series_cv(pipeline, X, y, n_splits=3):
    """
    Perform time series cross-validation
    
    Args:
        pipeline: Sklearn pipeline
        X: Features
        y: Target
        n_splits: Number of CV folds
    
    Returns:
        List of R² scores for each fold
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    print(f"\n=== Time Series Cross-Validation ({n_splits} folds) ===")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        pipeline.fit(X_train_cv, y_train_cv)
        y_pred_cv = pipeline.predict(X_val_cv)
        
        r2 = r2_score(y_val_cv, y_pred_cv)
        cv_scores.append(r2)
        
        print(f"Fold {fold}: R² = {r2*100:.4f}%")
    
    mean_r2 = np.mean(cv_scores)
    std_r2 = np.std(cv_scores)
    
    print(f"\nCross-Validation R²: {mean_r2*100:.4f}% ± {std_r2*100:.4f}%")
    
    return cv_scores

# ============================================
# Model Persistence
# ============================================
def save_model(model, filepath):
    """Save model to disk"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to: {filepath}")

def load_model(filepath):
    """Load model from disk"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from: {filepath}")
    return model

# ============================================
# Results Management
# ============================================
def save_results(results_dict, filepath):
    """Save results to CSV"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([results_dict])
    
    # Append if file exists, otherwise create new
    if Path(filepath).exists():
        existing = pd.read_csv(filepath)
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(filepath, index=False)
    print(f"✅ Results saved to: {filepath}")

def load_results(filepath):
    """Load results from CSV"""
    if Path(filepath).exists():
        return pd.read_csv(filepath)
    else:
        print(f"⚠️  Results file not found: {filepath}")
        return None

# ============================================
# Feature Importance
# ============================================
def get_feature_importance(model, feature_names, top_n=10):
    """
    Extract feature importance from model
    
    Works for:
    - Linear models (coefficients)
    - Tree-based models (feature_importances_)
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'coef_'):
        # Linear model
        importance = np.abs(model.coef_)
        importance_type = 'abs_coefficient'
    elif hasattr(model, 'feature_importances_'):
        # Tree-based model
        importance = model.feature_importances_
        importance_type = 'importance'
    else:
        print("⚠️  Model does not have feature importance")
        return None
    
    df = pd.DataFrame({
        'feature': feature_names,
        importance_type: importance
    }).sort_values(importance_type, ascending=False)
    
    print(f"\nTop {top_n} Features:")
    print(df.head(top_n).to_string(index=False))
    
    return df

# ============================================
# Helpers
# ============================================
def format_duration(seconds):
    """Format seconds into readable duration"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.2f}h"

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(title.center(60))
    print("="*60)