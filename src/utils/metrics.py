"""
Evaluation metrics for climate model bias correction.

This module provides functions to evaluate model performance:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Bias reduction percentage
- Pattern correlation
"""

import numpy as np
from scipy.stats import pearsonr

def compute_mae(y_true, y_pred, mask=None):
    """
    Compute Mean Absolute Error, optionally with a mask.
    
    Args:
        y_true (numpy.ndarray): Ground truth values
        y_pred (numpy.ndarray): Predicted values
        mask (numpy.ndarray, optional): Binary mask (1=valid, 0=invalid)
        
    Returns:
        float: MAE value
    """
    if mask is not None:
        valid_indices = mask > 0
        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]
    
    return np.mean(np.abs(y_true - y_pred))

def compute_rmse(y_true, y_pred, mask=None):
    """
    Compute Root Mean Square Error, optionally with a mask.
    
    Args:
        y_true (numpy.ndarray): Ground truth values
        y_pred (numpy.ndarray): Predicted values
        mask (numpy.ndarray, optional): Binary mask (1=valid, 0=invalid)
        
    Returns:
        float: RMSE value
    """
    if mask is not None:
        valid_indices = mask > 0
        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]
    
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def compute_bias_reduction(original_bias, corrected_bias):
    """
    Compute bias reduction percentage.
    
    Args:
        original_bias (float): Original bias (e.g., MAE or RMSE)
        corrected_bias (float): Corrected bias after model application
        
    Returns:
        float: Bias reduction percentage
    """
    return (1 - corrected_bias / original_bias) * 100

def compute_pattern_correlation(y_true, y_pred, mask=None):
    """
    Compute pattern correlation, optionally with a mask.
    
    Args:
        y_true (numpy.ndarray): Ground truth values
        y_pred (numpy.ndarray): Predicted values
        mask (numpy.ndarray, optional): Binary mask (1=valid, 0=invalid)
        
    Returns:
        float: Pattern correlation coefficient
    """
    if mask is not None:
        valid_indices = mask > 0
        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]
    
    corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return corr

def evaluate_model_performance(cmip6_data, oras5_data, corrected_data, mask=None):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        cmip6_data (numpy.ndarray): Original CMIP6 model data
        oras5_data (numpy.ndarray): ORAS5 observation data (ground truth)
        corrected_data (numpy.ndarray): Bias-corrected model data
        mask (numpy.ndarray, optional): Binary mask (1=valid, 0=invalid)
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Original bias metrics
    original_mae = compute_mae(oras5_data, cmip6_data, mask)
    original_rmse = compute_rmse(oras5_data, cmip6_data, mask)
    original_corr = compute_pattern_correlation(oras5_data, cmip6_data, mask)
    
    # Corrected bias metrics
    corrected_mae = compute_mae(oras5_data, corrected_data, mask)
    corrected_rmse = compute_rmse(oras5_data, corrected_data, mask)
    corrected_corr = compute_pattern_correlation(oras5_data, corrected_data, mask)
    
    # Bias reduction
    mae_reduction = compute_bias_reduction(original_mae, corrected_mae)
    rmse_reduction = compute_bias_reduction(original_rmse, corrected_rmse)
    
    # Results dictionary
    results = {
        'original_mae': original_mae,
        'original_rmse': original_rmse,
        'original_corr': original_corr,
        'corrected_mae': corrected_mae,
        'corrected_rmse': corrected_rmse,
        'corrected_corr': corrected_corr,
        'mae_reduction': mae_reduction,
        'rmse_reduction': rmse_reduction
    }
    
    return results
