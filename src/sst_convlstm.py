#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Masking, add, Activation, BatchNormalization, 
    Conv2DTranspose, UpSampling2D, ConvLSTM2D, Conv3D
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow import image
from tensorflow import float64
from tensorflow.keras import Input
from tensorflow import keras
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import time
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(89)

def configure_gpu():
    """Configure GPU for training."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Use GPU if available
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.get_memory_usage('GPU:0')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

def load_cmip6_historical_data(base_path='../data/'):
    """Load CMIP6 historical data (1958-2014)."""
    cmip6_historical = np.array(scipy.io.loadmat(f'{base_path}thetao/cmip6_thetao_1958_2014_fill_diststen.mat')['cmip6_ad_sten'])
    print(f"CMIP6 historical data shape: {cmip6_historical.shape}")
    return cmip6_historical

def load_cmip6_ssp_data(scenario, base_path='../data/'):
    """Load CMIP6 SSP scenario data for a specific period."""
    period = "2015_2022"
    cmip6_ssp = np.array(scipy.io.loadmat(f'{base_path}thetao/cmip6_thetao_{scenario}_{period}_fill_diststen.mat')['cmip6_ad_sten'])
    print(f"CMIP6 {scenario} {period} data shape: {cmip6_ssp.shape}")
    
    # Trim to first 72 timesteps
    cmip6_ssp = cmip6_ssp[0:72, :, :]
    print(f"After trimming: {cmip6_ssp.shape}")
    
    return cmip6_ssp

def load_oras5_historical_data(base_path='../data/'):
    """Load ORAS5 historical data (1958-2014)."""
    oras5_historical = np.array(scipy.io.loadmat(f'{base_path}thetao/oras5_sst_1958_2014_fill_diststen.mat')['oras5_ad_sten'])
    print(f"ORAS5 historical data shape: {oras5_historical.shape}")
    return oras5_historical

def load_oras5_future_data(base_path='../data/'):
    """Load ORAS5 future data (2015-2022)."""
    oras5_future = np.array(scipy.io.loadmat(f'{base_path}thetao/oras5_sst_2015_2022_fill_diststen.mat')['oras5_ad_sten'])
    
    # Trim to first 72 timesteps
    oras5_future = oras5_future[0:72, :, :]
    print(f"ORAS5 future data shape after trimming: {oras5_future.shape}")
    
    return oras5_future

def combine_data(cmip6_historical, cmip6_ssp_list, oras5_historical, oras5_future):
    """Combine historical and SSP data for all scenarios."""
    # Concatenate CMIP6 data
    cmip6_combined = np.concatenate([cmip6_historical] + cmip6_ssp_list, axis=0)
    
    # Repeat ORAS5 future data for each SSP scenario
    oras5_combined = np.concatenate([oras5_historical] + [oras5_future] * len(cmip6_ssp_list), axis=0)
    
    print(f"Combined CMIP6 data shape: {cmip6_combined.shape}")
    print(f"Combined ORAS5 data shape: {oras5_combined.shape}")
    
    return cmip6_combined, oras5_combined

def create_input_sequences(data, sequence_length=4):
    """Create input sequences for ConvLSTM model."""
    idx_slice = np.array([range(i, i + sequence_length) for i in range(data.shape[0] - (sequence_length - 1))])
    
    # Create sequences
    data_sequences = data[idx_slice]
    
    return data_sequences

def create_convlstm_model():
    """Create ConvLSTM model architecture."""
    # Input layer with no definite frame size
    inp = layers.Input(shape=(4, 85, 85, 1))
    
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
    
    # Output layer
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), padding="same"
    )(x)
    
    # Create model
    model = keras.models.Model(inp, x)
    
    return model

def compile_model(model, learning_rate=1e-4):
    """Compile the ConvLSTM model."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=["mae"]
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, batch_size=64, epochs=2500):
    """Train the ConvLSTM model."""
    # Create output directories
    os.makedirs('../output', exist_ok=True)
    os.makedirs('../output/models', exist_ok=True)
    
    # Create ModelCheckpoint callback
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath='../output/models/convlstm_model_best.keras',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    start = time.time()
    
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        shuffle=False,
        callbacks=[model_checkpoint_callback]
    )
    
    end = time.time()
    print('Time spent:', end - start)
    
    # Save the final model
    model.save('../output/models/convlstm_model.keras')
    
    return history

def plot_training_history(history):
    """Plot and save training history."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'][10:-1])
    plt.plot(history.history['val_loss'][10:-1])
    plt.title('ConvLSTM Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('../output/convlstm_train_mse_loss.jpg')
    
    # Save history to files
    hist_df = pd.DataFrame(history.history)
    hist_df.to_json('../output/convlstm_history.json')
    hist_df.to_csv('../output/convlstm_history.csv')

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    results = model.evaluate(X_test, y_test)
    print(f"Test loss: {results[0]}, Test MAE: {results[1]}")
    
    # Predict on test data
    y_hat = model.predict(X_test)
    y_hat = np.squeeze(y_hat)
    print(f"Predictions shape: {y_hat.shape}")
    
    # Save predictions
    os.makedirs('../output', exist_ok=True)
    np.save('../output/convlstm_thetao_test.npy', y_hat)
    
    return results

def predict_future_scenario(model, scenario, period="2023_2100", sequence_length=4, base_path='../data/'):
    """Generate predictions for a future scenario."""
    print(f"Processing {scenario} scenario for period {period}")
    
    # Load scenario data
    cmip6_data = np.array(scipy.io.loadmat(f'{base_path}thetao/cmip6_thetao_{scenario}_{period}_fill_diststen.mat')['cmip6_ad_sten'])
    print(f"CMIP6 {scenario} {period} data shape: {cmip6_data.shape}")
    
    # Create input sequences
    idx_slice = np.array([range(i, i + sequence_length) for i in range(cmip6_data.shape[0] - (sequence_length - 1))])
    cmip6_sequences = cmip6_data[idx_slice]
    
    # Add channel dimension
    cmip6_sequences = np.expand_dims(cmip6_sequences, axis=-1)
    print(f"Input sequences shape: {cmip6_sequences.shape}")
    
    # Generate predictions
    y_hat = model.predict(cmip6_sequences)
    y_hat = np.squeeze(y_hat)
    print(f"Predictions shape: {y_hat.shape}")
    
    # Save predictions
    os.makedirs('../output', exist_ok=True)
    np.save(f'../output/convlstm_thetao_{scenario}_{period}.npy', y_hat)
    
    return y_hat

def main():
    """Main function to run the ConvLSTM model."""
    # Configure GPU
    configure_gpu()
    
    # Load historical data
    cmip6_historical = load_cmip6_historical_data()
    oras5_historical = load_oras5_historical_data()
    
    # Load SSP scenario data for 2015-2022
    scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    cmip6_ssp_list = []
    
    for scenario in scenarios:
        cmip6_ssp = load_cmip6_ssp_data(scenario)
        cmip6_ssp_list.append(cmip6_ssp)
    
    # Load ORAS5 future data
    oras5_future = load_oras5_future_data()
    
    # Combine data
    cmip6_combined, oras5_combined = combine_data(
        cmip6_historical, cmip6_ssp_list, oras5_historical, oras5_future
    )
    
    # Create input sequences
    sequence_length = by if value for key, value in locals().items() else 4
    cmip6_sequences = create_input_sequences(cmip6_combined, sequence_length)
    oras5_sequences = create_input_sequences(oras5_combined, sequence_length)
    
    # Add channel dimension
    cmip6_sequences = np.expand_dims(cmip6_sequences, axis=-1)
    oras5_sequences = np.expand_dims(oras5_sequences, axis=-1)
    
    print(f"CMIP6 sequences shape: {cmip6_sequences.shape}")
    print(f"ORAS5 sequences shape: {oras5_sequences.shape}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        cmip6_sequences, oras5_sequences, train_size=0.85, random_state=42, shuffle=True
    )
    
    print(f"Training data shapes: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shapes: {X_test.shape}, {y_test.shape}")
    
    # Save the data
    os.makedirs('../output/data', exist_ok=True)
    np.save('../output/data/cmip6_train_convlstm.npy', X_train)
    np.save('../output/data/oras5_train_convlstm.npy', y_train)
    np.save('../output/data/cmip6_test_convlstm.npy', X_test)
    np.save('../output/data/oras5_test_convlstm.npy', y_test)
    
    # Create and compile model
    model = create_convlstm_model()
    model = compile_model(model)
    model.summary()
    
    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Generate predictions for future scenarios
    for scenario in scenarios:
        predict_future_scenario(model, scenario, period="2023_2100")

if __name__ == "__main__":
    main()
