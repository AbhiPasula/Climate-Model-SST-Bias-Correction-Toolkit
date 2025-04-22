"""
Data preprocessing utilities for climate model bias correction.

This module provides functions to preprocess climate data:
- Normalization
- Sequence creation
- Dimension adjustment
- Train/test splitting
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def data_minus_mean(data, mean_data):
    """
    Subtract climatological mean from data.
    
    Args:
        data (numpy.ndarray): Input climate data
        mean_data (numpy.ndarray): Climatological mean data
        
    Returns:
        numpy.ndarray: Normalized data
    """
    num = np.size(np.squeeze(data[:,1,1]))
    num = num/12  # Assuming monthly data, divide by 12 to get years
    
    # Repeat mean data for each year
    mean_repeated = np.repeat(mean_data, num, 0)
    
    # Subtract mean
    return data - mean_repeated

def data_plus_mean(data, mean_data):
    """
    Add climatological mean to data.
    
    Args:
        data (numpy.ndarray): Input climate data (anomalies)
        mean_data (numpy.ndarray): Climatological mean data
        
    Returns:
        numpy.ndarray: Data with mean added back
    """
    num = np.size(np.squeeze(data[:,1,1]))
    num = num/12  # Assuming monthly data, divide by 12 to get years
    
    # Repeat mean data for each year
    mean_repeated = np.repeat(mean_data, num, 0)
    
    # Add mean
    return data + mean_repeated

def resize_data(data, target_size):
    """
    Resize data to target dimensions.
    
    Args:
        data (numpy.ndarray): Input data
        target_size (int): Target size (square)
        
    Returns:
        tensorflow.Tensor: Resized data
    """
    return tf.image.resize(data, [target_size, target_size])

def prepare_unet_data(cmip6_data, oras5_data, mean_data, target_size=128, train_size=0.8):
    """
    Prepare data for UNet model.
    
    Args:
        cmip6_data (numpy.ndarray): CMIP6 model data
        oras5_data (numpy.ndarray): ORAS5 observation data
        mean_data (numpy.ndarray): Climatological mean data
        target_size (int): Size to resize images to
        train_size (float): Fraction of data to use for training
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Remove mean
    cmip6_normalized = data_minus_mean(cmip6_data, mean_data)
    oras5_normalized = data_minus_mean(oras5_data, mean_data)
    
    # Add channel dimension
    cmip6_normalized = np.expand_dims(cmip6_normalized, axis=-1)
    oras5_normalized = np.expand_dims(oras5_normalized, axis=-1)
    
    # Convert to float32
    cmip6_normalized = cmip6_normalized.astype(np.float32)
    oras5_normalized = oras5_normalized.astype(np.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        cmip6_normalized, oras5_normalized, train_size=train_size, shuffle=True
    )
    
    # Resize data
    X_train = resize_data(X_train, target_size)
    y_train = resize_data(y_train, target_size)
    X_test = resize_data(X_test, target_size)
    y_test = resize_data(y_test, target_size)
    
    return X_train, X_test, y_train, y_test

def prepare_bilstm_data(cmip6_data, cmip6_data_t, oras5_data, train_size=0.8, shuffle=False):
    """
    Prepare data for BiLSTM model.
    
    Args:
        cmip6_data (numpy.ndarray): CMIP6 model data
        cmip6_data_t (numpy.ndarray): CMIP6 transposed model data
        oras5_data (numpy.ndarray): ORAS5 observation data
        train_size (float): Fraction of data to use for training
        shuffle (bool): Whether to shuffle data for train/test split
    
    Returns:
        tuple: X_train1, X_train2, y_train, X_test1, X_test2, y_test
    """
    # Get indices for train/test split
    indices = np.arange(len(cmip6_data))
    train_idx, val_idx = train_test_split(indices, train_size=train_size, shuffle=shuffle)
    
    # Create training sets
    X_train1 = cmip6_data[train_idx]
    X_train2 = cmip6_data_t[train_idx]
    y_train = oras5_data[train_idx]
    
    # Create testing sets
    X_test1 = cmip6_data[val_idx]
    X_test2 = cmip6_data_t[val_idx]
    y_test = oras5_data[val_idx]
    
    return X_train1, X_train2, y_train, X_test1, X_test2, y_test

def prepare_convlstm_data(cmip6_data, oras5_data, sequence_length=4, train_size=0.85):
    """
    Prepare data for ConvLSTM model.
    
    Args:
        cmip6_data (numpy.ndarray): CMIP6 model data
        oras5_data (numpy.ndarray): ORAS5 observation data
        sequence_length (int): Length of input sequences
        train_size (float): Fraction of data to use for training
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Create input sequences
    idx_slice = np.array([
        range(i, i + sequence_length) 
        for i in range(cmip6_data.shape[0] - (sequence_length - 1))
    ])
    
    # Create sequences for both CMIP6 and ORAS5 data
    cmip6_sequences = cmip6_data[idx_slice]
    oras5_sequences = oras5_data[idx_slice]
    
    # Add channel dimension if needed
    cmip6_sequences = np.expand_dims(cmip6_sequences, axis=-1)
    oras5_sequences = np.expand_dims(oras5_sequences, axis=-1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        cmip6_sequences, oras5_sequences, train_size=train_size, shuffle=True
    )
    
    return X_train, X_test, y_train, y_test