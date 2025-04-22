#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(89)

def configure_gpu():
    """Configure GPU for training."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.get_memory_usage('GPU:0')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

def load_historical_data(base_path='../data/'):
    """Load historical CMIP6 and ORAS5 data (1958-2014)."""
    # Load CMIP6 historical data for both inputs
    cmip6_hist_l1 = np.load(f'{base_path}thetao/cmip6_thetao_1958_2014_2d.npy')
    cmip6_hist_l1 = cmip6_hist_l1[:,0:-2]
    print(f"CMIP6 historical data (input 1) shape: {cmip6_hist_l1.shape}")
    
    cmip6_hist_l2 = np.load(f'{base_path}thetao/cmip6_thetao_transpose_ts_1958_2014_2d.npy')
    cmip6_hist_l2 = cmip6_hist_l2[:,0:-2]
    print(f"CMIP6 historical data (input 2) shape: {cmip6_hist_l2.shape}")
    
    # Load ORAS5 historical data
    oras5_sst_2014 = np.load(f'{base_path}thetao/oras5_temp_1958_2014_2d.npy')
    print(f"ORAS5 historical data shape: {oras5_sst_2014.shape}")
    
    return cmip6_hist_l1, cmip6_hist_l2, oras5_sst_2014

def load_ssp_data(scenario, base_path='../data/'):
    """Load SSP scenario data for a specific scenario."""
    # Load CMIP6 SSP data for both inputs
    cmip6_ssp_l1 = np.load(f'{base_path}thetao/cmip6_thetao_{scenario}_2015_2022_2d.npy')
    cmip6_ssp_l2 = np.load(f'{base_path}thetao/cmip6_thetao_trans_{scenario}_2015_2022_2d.npy')
    
    print(f"CMIP6 {scenario} data (input 1) shape: {cmip6_ssp_l1.shape}")
    print(f"CMIP6 {scenario} data (input 2) shape: {cmip6_ssp_l2.shape}")
    
    # Adjust dimensions
    cmip6_ssp_l1 = cmip6_ssp_l1[:,0:-2]
    cmip6_ssp_l2 = cmip6_ssp_l2[:,0:-2]
    
    return cmip6_ssp_l1, cmip6_ssp_l2

def load_future_oras5_data(base_path='../data/'):
    """Load ORAS5 future data (2015-2022)."""
    oras5_ssp = np.load(f'{base_path}thetao/oras5_temp_2015_2022_2d.npy')
    print(f"ORAS5 future data shape: {oras5_ssp.shape}")
    return oras5_ssp

def combine_data(historical_data, ssp_data_list):
    """Combine historical and SSP data for all scenarios."""
    cmip6_hist_l1, cmip6_hist_l2, oras5_sst_2014 = historical_data
    
    # Unpack SSP data
    ssp_l1_list = [data[0] for data in ssp_data_list]
    ssp_l2_list = [data[1] for data in ssp_data_list]
    oras5_ssp = ssp_data_list[0][2]  # Same ORAS5 data for all SSPs
    
    # Concatenate CMIP6 data for both inputs
    cmip6_sst_ssp = np.concatenate([cmip6_hist_l1] + ssp_l1_list, axis=0)
    cmip6_sst_ssp_t = np.concatenate([cmip6_hist_l2] + ssp_l2_list, axis=0)
    
    # Concatenate ORAS5 data (repeating future data for each SSP)
    oras5_sst_ssp = np.concatenate([oras5_sst_2014] + [oras5_ssp] * len(ssp_data_list), axis=0)
    
    print(f"Combined CMIP6 data (input 1) shape: {cmip6_sst_ssp.shape}")
    print(f"Combined CMIP6 data (input 2) shape: {cmip6_sst_ssp_t.shape}")
    print(f"Combined ORAS5 data shape: {oras5_sst_ssp.shape}")
    
    return cmip6_sst_ssp, cmip6_sst_ssp_t, oras5_sst_ssp

def split_train_test(cmip6_sst_ssp, cmip6_sst_ssp_t, oras5_sst_ssp, train_size=0.8, shuffle=False):
    """Split data into training and testing sets."""
    indices = np.arange(len(cmip6_sst_ssp))
    train_idx, val_idx = train_test_split(indices, train_size=train_size, shuffle=shuffle)
    
    # Create training sets
    X_train1 = cmip6_sst_ssp[train_idx]
    X_train2 = cmip6_sst_ssp_t[train_idx]
    y_train = oras5_sst_ssp[train_idx]
    
    # Create testing sets
    X_test1 = cmip6_sst_ssp[val_idx]
    X_test2 = cmip6_sst_ssp_t[val_idx]
    y_test = oras5_sst_ssp[val_idx]
    
    print(f"Training data shapes: {X_train1.shape}, {X_train2.shape}, {y_train.shape}")
    print(f"Testing data shapes: {X_test1.shape}, {X_test2.shape}, {y_test.shape}")
    
    # Save the data
    os.makedirs('../output/data', exist_ok=True)
    np.save('../output/data/cmip6_train1.npy', X_train1)
    np.save('../output/data/cmip6_train2.npy', X_train2)
    np.save('../output/data/oras5_train.npy', y_train)
    np.save('../output/data/cmip6_test1.npy', X_test1)
    np.save('../output/data/cmip6_test2.npy', X_test2)
    np.save('../output/data/oras5_test.npy', y_test)
    
    return X_train1, X_train2, y_train, X_test1, X_test2, y_test

def create_bilstm_model():
    """Create BiLSTM model architecture."""
    inp1 = layers.Input(shape=(5215, 1))
    inp2 = layers.Input(shape=(5215, 1))
    
    # BiLSTM layers for first input
    x1 = Bidirectional(LSTM(512, return_sequences=True), 
                      batch_input_shape=(5215, 1), 
                      merge_mode='concat')(inp1)
    
    # BiLSTM layers for second input
    x2 = Bidirectional(LSTM(512, return_sequences=True), 
                      batch_input_shape=(5215, 1), 
                      merge_mode='concat')(inp2)
    
    # Concatenate outputs from both inputs
    x3 = keras.layers.Concatenate()([x1, x2])
    
    # Output layer
    l3 = tf.keras.layers.Dense(1, activation='relu')(x3)
    
    # Create model
    model = tf.keras.models.Model([inp1, inp2], l3)
    
    return model

def load_model(model_path):
    """Load a pre-trained model."""
    model_path = os.path.join('../output/models', model_path) if not model_path.startswith('../') else model_path
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="mae", metrics='mse')
    return model

def train_model(model, X_train1, X_train2, y_train, X_test1, X_test2, y_test, 
                batch_size=32, epochs=500, shuffle=False):
    """Train the BiLSTM model."""
    start = datetime.datetime.now()
    
    # Create a ModelCheckpoint callback
    os.makedirs('../output/models', exist_ok=True)
    checkpoint_path = '../output/models/bilstm_model_best.h5'
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        x=[X_train1, X_train2],
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X_test1, X_test2], y_test),
        shuffle=shuffle,
        callbacks=[checkpoint_callback]
    )
    
    end = datetime.datetime.now()
    elapsed = end - start
    print('Time spent:', elapsed)
    
    # Save final model
    os.makedirs('../output/models', exist_ok=True)
    model.save(f'../output/models/bilstm_sst_hist_ssp_{end.strftime("%Y_%m_%d_%H_%M_%S")}.h5')
    
    return history

def evaluate_model(model, X_test1, X_test2, y_test):
    """Evaluate the model on test data."""
    results = model.evaluate([X_test1, X_test2], y_test)
    y_hat = model.predict([X_test1, X_test2])
    y_hat = np.squeeze(y_hat)
    print(f"Predicted output shape: {y_hat.shape}")
    
    # Create output directory if it doesn't exist
    os.makedirs('../output', exist_ok=True)
    np.save('../output/bilstm_thetao_test.npy', y_hat)
    
    return results, y_hat

def plot_history(history):
    """Plot training history."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'][50:-1])
    plt.plot(history.history['val_loss'][50:-1])
    plt.title('BiLSTM Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Create output directory if it doesn't exist
    os.makedirs('../output', exist_ok=True)
    
    plt.savefig('../output/bilstm_train_mse_loss.jpg')
    
    # Save history to file
    hist_df = pd.DataFrame(history.history)
    hist_df.to_json('../output/bilstm_history.json')
    hist_df.to_csv('../output/bilstm_history.csv')

def predict_future_scenario(model, scenario, base_path='../data/'):
    """Generate predictions for a future scenario (2023-2100)."""
    print(f"Processing future {scenario} scenario (2023-2100)")
    
    # Load future scenario data
    cmip6_ssp_l1 = np.load(f'{base_path}thetao/cmip6_thetao_{scenario}_2023_2100_2d.npy')
    cmip6_ssp_l2 = np.load(f'{base_path}thetao/cmip6_thetao_trans_{scenario}_2023_2100_2d.npy')
    
    print(f"CMIP6 future {scenario} data shapes: {cmip6_ssp_l1.shape}, {cmip6_ssp_l2.shape}")
    
    # Adjust dimensions
    cmip6_ssp_l1 = cmip6_ssp_l1[:,0:-2]
    cmip6_ssp_l2 = cmip6_ssp_l2[:,0:-2]
    
    # Generate predictions
    y_hat = model.predict([cmip6_ssp_l1, cmip6_ssp_l2])
    y_hat = np.squeeze(y_hat)
    print(f"Predictions shape: {y_hat.shape}")
    
    # Save predictions to output directory
    os.makedirs('../output', exist_ok=True)
    np.save(f'../output/bilstm_thetao_{scenario}_2023_2100.npy', y_hat)
    
    # Reshape predictions to 3D
    dataNoMissing = np.load(f'{base_path}thetao/datanomissing.npy')
    dataNoMissing = dataNoMissing[0:-2,]
    
    data = np.eye(936, 7225)
    data[:] = np.nan
    data[:, dataNoMissing] = y_hat
    
    ytest_hat = data.reshape([936, 85, 85])
    print(f"Reshaped predictions: {ytest_hat.shape}")
    
    np.save(f'../output/bilstm_thetao_{scenario}_2023_2100_3d.npy', ytest_hat)

def main():
    """Main function to run the BiLSTM model."""
    # Configure GPU
    configure_gpu()
    
    # Load historical data
    historical_data = load_historical_data()
    
    # Load SSP scenario data for 2015-2022
    scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    oras5_ssp = load_future_oras5_data()
    
    ssp_data_list = []
    for scenario in scenarios:
        cmip6_ssp_l1, cmip6_ssp_l2 = load_ssp_data(scenario)
        ssp_data_list.append((cmip6_ssp_l1, cmip6_ssp_l2, oras5_ssp))
    
    # Combine data
    cmip6_sst_ssp, cmip6_sst_ssp_t, oras5_sst_ssp = combine_data(historical_data, ssp_data_list)
    
    # Split data into training and testing sets
    X_train1, X_train2, y_train, X_test1, X_test2, y_test = split_train_test(
        cmip6_sst_ssp, cmip6_sst_ssp_t, oras5_sst_ssp
    )
    
    # Create a new model
    model = create_bilstm_model()
    model.compile(optimizer="adam", loss="mae", metrics='mse')
    model.summary()
    
    # Train model
    history = train_model(model, X_train1, X_train2, y_train, X_test1, X_test2, y_test)
    
    # Evaluate model
    results, y_hat = evaluate_model(model, X_test1, X_test2, y_test)
    
    # Plot training history
    plot_history(history)
    
    # Generate predictions for future scenarios (2023-2100)
    for scenario in scenarios:
        predict_future_scenario(model, scenario)

if __name__ == "__main__":
    main()
