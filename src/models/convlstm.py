"""
Convolutional LSTM model for spatiotemporal bias correction in climate data.

This module provides functions to create a ConvLSTM architecture with:
- Stacked ConvLSTM2D layers
- Encoder-decoder structure with varying filter sizes
- BatchNormalization between layers
- 3D convolutional output layer
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

def create_convlstm_model(sequence_length=4, height=85, width=85):
    """Create ConvLSTM model architecture.
    
    Args:
        sequence_length (int): Number of time steps in input sequences
        height (int): Height of input images
        width (int): Width of input images
        
    Returns:
        keras.Model: ConvLSTM model
    """
    # Input layer with specified frame size
    inp = layers.Input(shape=(sequence_length, height, width, 1))
    
    # First ConvLSTM layer
    x = layers.ConvLSTM2D(
        filters=8,
        kernel_size=(7, 7),
        padding="same",
        return_sequences=True
    )(inp)
    x = layers.BatchNormalization()(x)
    
    # Second ConvLSTM layer
    x = layers.ConvLSTM2D(
        filters=16,
        kernel_size=(7, 7),
        padding="same",
        return_sequences=True
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Third ConvLSTM layer
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Fourth ConvLSTM layer
    x = layers.ConvLSTM2D(
        filters=48,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Fifth ConvLSTM layer
    x = layers.ConvLSTM2D(
        filters=48,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Sixth ConvLSTM layer
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Seventh ConvLSTM layer
    x = layers.ConvLSTM2D(
        filters=16,
        kernel_size=(7, 7),
        padding="same",
        return_sequences=True
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Eighth ConvLSTM layer
    x = layers.ConvLSTM2D(
        filters=8,
        kernel_size=(7, 7),
        padding="same",
        return_sequences=True
    )(x)
    
    # Output layer - Conv3D for spatiotemporal integration
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), padding="same"
    )(x)
    
    # Create model
    model = keras.models.Model(inp, x)
    
    return model

def compile_model(model, learning_rate=1e-4):
    """Compile ConvLSTM model with appropriate optimizer and loss.
    
    Args:
        model (keras.Model): ConvLSTM model to compile
        learning_rate (float): Learning rate for Adam optimizer
        
    Returns:
        keras.Model: Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=["mae"]
    )
    
    return model

def create_input_sequences(data, sequence_length=4):
    """Create input sequences for ConvLSTM model.
    
    This function converts a time series of spatial data into
    overlapping sequences for ConvLSTM training.
    
    Args:
        data (numpy.ndarray): Input data with shape [time, height, width]
        sequence_length (int): Number of time steps per sequence
        
    Returns:
        numpy.ndarray: Sequences with shape [samples, sequence_length, height, width]
    """
    import numpy as np
    
    # Create indices for sequence extraction
    idx_slice = np.array([range(i, i + sequence_length) 
                          for i in range(data.shape[0] - (sequence_length - 1))])
    
    # Extract sequences
    data_sequences = data[idx_slice]
    
    # Add channel dimension if needed
    if len(data_sequences.shape) == 3:
        data_sequences = np.expand_dims(data_sequences, axis=-1)
    
    return data_sequences
