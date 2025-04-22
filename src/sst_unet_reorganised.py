# %%
import os
import scipy.io
import sys
import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU
import time
import datetime
import pandas as pd
from scipy.interpolate import interp2d
keras.utils.set_random_seed(89)
#%%
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.get_memory_usage('GPU:0')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e) 

def load_cmip6_sst_data(base_path='../data/sst/'):
    """
    Load and process CMIP6 sst (sea surface height) data from multiple scenarios.
    
    Parameters:
    -----------
    base_path : str
        Base directory path where the .mat files are stored
        
    Returns:
    --------
    numpy.ndarray
        Concatenated array of historical and SSP scenario data
    """
    
    # Load historical data (1958-2014)
    historical_file = f'{base_path}cmip6_sst_1958_2014_fill_diststen.mat'
    cmip6_historical = np.array(scipy.io.loadmat(historical_file)['cmip6_ad_sten'])
    
    # Dictionary of SSP scenarios
    ssp_scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    ssp_data = {}
    
    # Load and process each SSP scenario
    for scenario in ssp_scenarios:
        # Load scenario data
        scenario_file = f'{base_path}cmip6_sst_{scenario}_2015_2022_fill_diststen.mat'
        scenario_data = np.array(scipy.io.loadmat(scenario_file)['cmip6_ad_sten'])
        
        # Trim to first 72 timesteps
        ssp_data[scenario] = scenario_data[0:72, :, :]
        
        print(f'Loaded {scenario}: Shape = {ssp_data[scenario].shape}')
    
    # Concatenate all data
    concatenated_data = np.concatenate(
        [cmip6_historical] + [ssp_data[scenario] for scenario in ssp_scenarios],
        axis=0
    )
    
    print(f'Final concatenated shape: {concatenated_data.shape}')
    
    return concatenated_data
import numpy as np
import scipy.io

def load_oras5_sst_data(base_path='../data/sst/'):
    """
    Load and process ORAS5 sst (sea surface height) data, including historical data
    and future projections with repetition.
    
    Parameters:
    -----------
    base_path : str
        Base directory path where the .mat files are stored
        
    Returns:
    --------
    numpy.ndarray
        Concatenated array of historical and repeated future data
    """
    
    # Load historical data (1958-2014)
    historical_file = f'{base_path}oras5_sst_1958_2014_fill_diststen.mat'
    oras5_historical = np.array(scipy.io.loadmat(historical_file)['oras5_ad_sten'])
    
    # Load future data (2015-2022)
    future_file = f'{base_path}oras5_sst_2015_2022_fill_diststen.mat'
    oras5_future = np.array(scipy.io.loadmat(future_file)['oras5_ad_sten'])
    
    # Trim future data to first 72 timesteps
    oras5_future_trimmed = oras5_future[0:72, :, :]
    print(f'Future data shape after trimming: {oras5_future_trimmed.shape}')
    
    # Concatenate historical data with 4 repetitions of future data
    # This matches the structure of having data for 4 different SSP scenarios
    concatenated_data = np.concatenate(
        [oras5_historical] + [oras5_future_trimmed] * 4,
        axis=0
    )
    
    print(f'Final concatenated shape: {concatenated_data.shape}')
    
    return concatenated_data
def data_minus_mean(data):
    oras5_mean1=np.array(scipy.io.loadmat('../data/sst/oras5_historical_sst_1958_2020_mean.mat')['oras5_mclim'])
    num=np.size(np.squeeze(data[:,1,1]))
    num=num/12
    print(num)
    oras5_mean=np.repeat(oras5_mean1,num,0)
    data=data-oras5_mean
    return data

def data_plus_mean(data):
    oras5_mean1=np.array(scipy.io.loadmat('../data/sst/oras5_historical_sst_1958_2020_mean.mat')['oras5_mclim'])
    num=np.size(np.squeeze(data[:,1,1]))
    num=num/12
    print(num)
    oras5_mean=np.repeat(oras5_mean1,num,0)
    data=data+oras5_mean
    return data
def resize_data(data, target_size):
    """Resize data to target dimensions."""
    return tf.image.resize(data, [target_size, target_size])

def create_unet_model(input_size=128):
    """Create U-Net model architecture."""
    def double_conv_block(x, n_filters):
        x = layers.Conv2D(n_filters, 3, padding="same", activation='tanh', kernel_initializer="he_normal")(x)
        x = layers.Conv2D(n_filters, 3, padding="same", activation='tanh', kernel_initializer="he_normal")(x)
        return x

    def downsample_block(x, n_filters):
        f = double_conv_block(x, n_filters)
        p = layers.MaxPool2D(2)(f)
        p = layers.Dropout(0.2)(p)
        return f, p

    def upsample_block(x, conv_features, n_filters):
        x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        x = layers.concatenate([x, conv_features])
        x = layers.Dropout(0.2)(x)
        x = double_conv_block(x, n_filters)
        return x

    inputs = layers.Input(shape=(input_size, input_size, 1))
    
    # Encoder
    f0, p0 = downsample_block(inputs, 32)
    f1, p1 = downsample_block(p0, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)
    
    # Bottleneck
    bottleneck = double_conv_block(p4, 1024)
    
    # Decoder
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)
    u10 = upsample_block(u9, f0, 32)
    
    outputs = layers.Conv2D(1, 1, padding="same")(u10)
    
    return keras.Model(inputs, outputs, name="U-Net")
def custom_mse_loss(mask):
    """Create custom MSE loss function with mask."""
    mask1=tf.convert_to_tensor(mask.astype(np.float32))
    mask = tf.image.resize(np.expand_dims(mask1, axis = -1), [128,128]) 
    def mse_loss(y_true, y_pred):
        y_true = tf.multiply(y_true, mask)
        y_pred = tf.multiply(y_pred, mask)
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    return mse_loss
def process_ssp_scenario_2023(scenario, model, input_path_template, output_path_template, pix_size, final_size):
    input_path = input_path_template.format(scenario=scenario)
    output_path = output_path_template.format(scenario=scenario)
    target_size=128
    # Load and preprocess data
    cmip6_data = np.array(scipy.io.loadmat(input_path)['cmip6_ad_sten'])
    oras5_mean1=np.array(scipy.io.loadmat('../data/sst/oras5_historical_sst_1958_2020_mean.mat')['oras5_mclim'])
    num=np.size(np.squeeze(cmip6_data[:,1,1]))
    num=num/12
    oras5_mean=np.repeat(oras5_mean1,num,0)
    cmip6_data_processed=cmip6_data-oras5_mean
    cmip6_data_expanded = np.expand_dims(cmip6_data_processed, axis=-1)
    cmip6_data_resized=tf.image.resize(cmip6_data_expanded, [target_size, target_size])
    
    print(f"Input shape for scenario {scenario}: {cmip6_data_resized.shape}")
    
    # Predict using the model
    x_hat = model.predict(cmip6_data_resized)
    x_hat_resized=tf.image.resize(x_hat, [85, 85])
    print(x_hat_resized.shape)
     # Postprocess and save results
    unet_out = np.squeeze(x_hat_resized)+oras5_mean
    print(f"Output shape for scenario {scenario}: {x_hat_resized.shape}")
    np.save(output_path, unet_out)
    print(f"Saved output for {scenario} to {output_path}")

def main():
    import os
    import scipy.io
    import sys
    import numpy as np
    import scipy
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import time
    import datetime
    import pandas as pd
    from scipy.interpolate import interp2d
    import matplotlib.pyplot as plt

    tf.keras.utils.set_random_seed(89)
    
    # Configure GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPU, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
    # load cmip6 data
    cmip6_data = load_cmip6_sst_data()
    # load ORAS5 data
    oras5_data = load_oras5_sst_data()
    # remove mean
    cmip6_data1=data_minus_mean(cmip6_data)
    print(cmip6_data1.shape)
    oras5_data1=data_minus_mean(oras5_data)
    print(oras5_data1.shape)

    # preprocess data
    cmip6_sst=np.expand_dims(cmip6_data1, axis = -1)
    oras5_data1=np.expand_dims(oras5_data1, axis = -1)

    print(cmip6_sst.shape)

    # %%
    cmip6_sst=cmip6_sst.astype(np.float32)
    oras5_sst_ssp=oras5_data1.astype(np.float32)
    print(cmip6_sst.shape)
    #%%
    # mask=np.load('../mask_cmip6.npy')
    # print(mask.shape)
    # mask=torch.from_numpy(mask)
    # %%
    from sklearn.model_selection import train_test_split
    # np.random
    X_train1, X_test1, y_train, y_test = train_test_split(cmip6_sst, oras5_sst_ssp, train_size=0.87654321, shuffle=True)
    print(X_train1.shape, X_test1.shape)
    print(y_train.shape, y_test.shape)

    print(X_train1.shape, X_test1.shape)
    print(y_train.shape, y_test.shape)
    np.save('cmip6_train.npy',X_train1)
    np.save('oras5_train.npy',y_train)

    np.save('cmip6_test.npy',X_test1)

    np.save('oras5_test.npy',y_test)

    # %%
    pix_size=128
    X_train1 = resize_data(X_train1, pix_size) 
    
    y_train =resize_data(y_train, pix_size)

    X_test1 = resize_data(X_test1, pix_size)
    y_test = resize_data(X_test1, pix_size)
    print(X_train1.shape, X_test1.shape)
    print(y_train.shape, y_test.shape)

    print(X_train1.shape, X_test1.shape)
    print(y_train.shape, y_test.shape)

     # Create and compile model
    model = create_unet_model()
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    mask=np.array(scipy.io.loadmat('../data/oras5_mask.mat')['mask1'])
    model.compile(
        loss=custom_mse_loss(mask),
        optimizer=optimizer,
        metrics=["mae"]
    )
    # train model
    history = model.fit(
    x               = X_train1,
    y               = y_train,
    batch_size      = 64,
    epochs          = 1000,
    validation_data=(X_test1,y_test),
    shuffle         = False)

    # Save training history
    hist_df = pd.DataFrame(history.history)
    # hist_df.to_json('history.json')
    hist_df.to_csv('../output/sst_history.csv')
    
    # Plot training history

    plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'],linewidth=2)
    plt.plot(history.history['val_loss'],linewidth=2)
    plt.title('UNet SST loss',size=28)
    plt.ylabel('loss',size=28)
    plt.xlabel('epoch',size=28)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(['SST Train loss','SST Validation loss'], loc='upper right')
    plt.savefig('../output/sst_train_mse_loss.jpg')
    plt.close()


    scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    input_path_template = '../data/sst/cmip6_sst_{scenario}_2023_2100_fill_diststen.mat'
    output_path_template = '../output/unet_sst_{scenario}_2023_2100.npy'
    pix_size = 128  # Replace with the actual pixel size for resizing
    final_size = 85  # Replace with the final size after resizing
    for scenario in scenarios:
        process_ssp_scenario_2023(scenario, model, input_path_template, output_path_template, pix_size, final_size)

    scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    input_path_template = '../data/sst/cmip6_sst_{scenario}_2015_2022_fill_diststen.mat'
    output_path_template = '../output/unet_sst_{scenario}_2015_2022.npy'
    pix_size = 128  # Replace with the actual pixel size for resizing
    final_size = 85  # Replace with the final size after resizing
    for scenario in scenarios:
        process_ssp_scenario_2023(scenario, model, input_path_template, output_path_template, pix_size, final_size)

if __name__ == "__main__":
    main()

