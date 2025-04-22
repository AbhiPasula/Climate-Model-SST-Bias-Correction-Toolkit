"""
Training history visualization utilities.

This module provides functions to visualize model training history:
- Loss curves
- Metric evolution
- Learning rate schedules
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_history(history, output_path=None, skip_epochs=10, figsize=(10, 10)):
    """
    Plot training and validation loss history.
    
    Args:
        history: Keras history object or path to history CSV file
        output_path (str, optional): Path to save the plot
        skip_epochs (int): Number of initial epochs to skip in plot (stabilizes y-axis)
        figsize (tuple): Figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Load history if it's a file path
    if isinstance(history, str) and os.path.exists(history):
        if history.endswith('.csv'):
            hist_df = pd.read_csv(history)
        elif history.endswith('.json'):
            hist_df = pd.read_json(history)
        else:
            raise ValueError("History file must be either CSV or JSON")
        
        history_dict = hist_df.to_dict('list')
    else:
        # Assume it's a keras history object
        history_dict = history.history
    
    # Ensure output directory exists
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot loss
    if 'loss' in history_dict:
        plt.plot(history_dict['loss'][skip_epochs:], linewidth=2)
    if 'val_loss' in history_dict:
        plt.plot(history_dict['val_loss'][skip_epochs:], linewidth=2)
        
    # Set title and labels based on model type
    model_type = "Model"
    if 'bilstm' in str(output_path).lower():
        model_type = "BiLSTM"
    elif 'unet' in str(output_path).lower():
        model_type = "UNet"
    elif 'convlstm' in str(output_path).lower():
        model_type = "ConvLSTM"
        
    plt.title(f'{model_type} Training Loss', size=28)
    plt.ylabel('Loss', size=28)
    plt.xlabel('Epoch', size=28)
    plt.xticks(size=20)
    plt.yticks(size=20)
    
    # Add legend based on available data
    legend_items = []
    if 'loss' in history_dict:
        legend_items.append('Training loss')
    if 'val_loss' in history_dict:
        legend_items.append('Validation loss')
    plt.legend(legend_items, loc='upper right', fontsize=16)
    
    # Save the plot if an output path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Saved training history plot to {output_path}")
    
    return plt.gcf()

def plot_metrics_comparison(history_dict, metric_name='mae', output_path=None, figsize=(10, 6)):
    """
    Plot comparison of metrics across different models.
    
    Args:
        history_dict (dict): Dictionary mapping model names to history objects/files
        metric_name (str): Name of metric to plot
        output_path (str, optional): Path to save the plot
        figsize (tuple): Figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=figsize)
    
    # Process each model's history
    for model_name, history in history_dict.items():
        # Load history if it's a file path
        if isinstance(history, str) and os.path.exists(history):
            if history.endswith('.csv'):
                hist_df = pd.read_csv(history)
            elif history.endswith('.json'):
                hist_df = pd.read_json(history)
            else:
                raise ValueError("History file must be either CSV or JSON")
            
            history_data = hist_df.to_dict('list')
        else:
            # Assume it's a keras history object
            history_data = history.history
        
        # Plot the specified metric
        if metric_name in history_data:
            plt.plot(history_data[metric_name], linewidth=2, label=f"{model_name}")
    
    plt.title(f'Model Comparison - {metric_name.upper()}', size=20)
    plt.ylabel(metric_name.upper(), size=16)
    plt.xlabel('Epoch', size=16)
    plt.legend(loc='best', fontsize=14)
    
    # Save the plot if an output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved metrics comparison plot to {output_path}")
    
    return plt.gcf()
