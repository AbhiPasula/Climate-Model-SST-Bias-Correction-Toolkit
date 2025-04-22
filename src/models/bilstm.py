"""
Bidirectional LSTM model for temporal bias correction in climate data.

This module provides functions to create a BiLSTM architecture with:
- Dual input streams for original and transposed data
- Bidirectional LSTM layers with 512 units
- Concatenation of both streams for integrated temporal learning
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional, LSTM

def create_bilstm_model():
    """Create BiLSTM model architecture.
    
    The model accepts two inputs:
    1. Original CMIP6 time series data
    2. Transposed CMIP6 time series data
    
    Returns:
        tf.keras.Model: Compiled BiLSTM model
    """
    # Define input shapes for original and transposed data
    inp1 = layers.Input(shape=(5215, 1))
    inp2 = layers.Input(shape=(5215, 1))
    
    # BiLSTM layers for first input (original data)
    x1 = Bidirectional(LSTM(512, return_sequences=True), 
                      batch_input_shape=(5215, 1), 
                      merge_mode='concat')(inp1)
    
    # BiLSTM layers for second input (transposed data)
    x2 = Bidirectional(LSTM(512, return_sequences=True), 
                      batch_input_shape=(5215, 1), 
                      merge_mode='concat')(inp2)
    
    # Concatenate outputs from both inputs
    x3 = tf.keras.layers.Concatenate()([x1, x2])
    
    # Output layer
    l3 = tf.keras.layers.Dense(1, activation='relu')(x3)
    
    # Create model
    model = tf.keras.models.Model([inp1, inp2], l3)
    
    return model

def compile_bilstm_model(model, learning_rate=1e-4):
    """Compile BiLSTM model with appropriate optimizer and loss.
    
    Args:
        model (tf.keras.Model): BiLSTM model to compile
        learning_rate (float): Learning rate for Adam optimizer
        
    Returns:
        tf.keras.Model: Compiled model
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="mae",  # Mean Absolute Error for more robust training
        metrics=['mse']  # Track Mean Squared Error as additional metric
    )
    
    return model

def prepare_bilstm_inputs(data, transposed_data):
    """Prepare inputs for BiLSTM model.
    
    Args:
        data (numpy.ndarray): Original climate data
        transposed_data (numpy.ndarray): Transposed climate data
        
    Returns:
        tuple: Processed input data for BiLSTM model
    """
    # Reshape data if needed
    if len(data.shape) > 2:
        # Flatten spatial dimensions for BiLSTM
        orig_shape = data.shape
        data = data.reshape(orig_shape[0], -1)
        transposed_data = transposed_data.reshape(orig_shape[0], -1)
    
    # Add channel dimension if needed
    if len(data.shape) == 2:
        data = data[..., np.newaxis]
        transposed_data = transposed_data[..., np.newaxis]
    
    return data, transposed_data
