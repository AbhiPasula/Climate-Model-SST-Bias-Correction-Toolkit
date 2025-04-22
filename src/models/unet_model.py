"""
U-Net model architecture for spatial bias correction in climate data.

This module provides functions to create a U-Net architecture with:
- Symmetric encoder-decoder structure with skip connections
- Tanh activation for normalized ocean data
- Dropout regularization
- Custom MSE loss with ocean mask
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_unet_model(input_size=128):
    """Create U-Net model architecture.
    
    Args:
        input_size (int): Size of input images (square)
        
    Returns:
        keras.Model: Compiled U-Net model
    """
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

    # Input layer
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
    
    # Output layer
    outputs = layers.Conv2D(1, 1, padding="same")(u10)
    
    return keras.Model(inputs, outputs, name="U-Net")

def custom_mse_loss(mask):
    """Create custom MSE loss function with ocean mask.
    
    This function applies an ocean mask to focus the loss calculation
    only on ocean regions, ignoring land areas.
    
    Args:
        mask (numpy.ndarray): Binary ocean mask (1=ocean, 0=land)
        
    Returns:
        function: Custom loss function
    """
    mask1 = tf.convert_to_tensor(mask.astype(np.float32))
    mask = tf.image.resize(np.expand_dims(mask1, axis=-1), [128, 128]) 
    
    def mse_loss(y_true, y_pred):
        y_true = tf.multiply(y_true, mask)
        y_pred = tf.multiply(y_pred, mask)
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    
    return mse_loss

def resize_data(data, target_size):
    """Resize data to target dimensions.
    
    Args:
        data (numpy.ndarray): Input data
        target_size (int): Target size (square)
        
    Returns:
        tensorflow.Tensor: Resized data
    """
    return tf.image.resize(data, [target_size, target_size])
