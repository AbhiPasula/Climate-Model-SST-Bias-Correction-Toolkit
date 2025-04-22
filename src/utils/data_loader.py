"""
Data loading utilities for climate model bias correction.

This module provides functions to load and process various climate data formats:
- CMIP6 model outputs
- ORAS5 observation data
- SSP scenario projections
"""

import os
import numpy as np
import scipy.io

def load_cmip6_sst_data(base_path='../data/sst/'):
    """
    Load and process CMIP6 SST data from multiple scenarios.
    
    Args:
        base_path (str): Base directory path where the .mat files are stored
        
    Returns:
        numpy.ndarray: Concatenated array of historical and SSP scenario data
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

def load_oras5_sst_data(base_path='../data/sst/'):
    """
    Load and process ORAS5 SST data, including historical data
    and future projections with repetition.
    
    Args:
        base_path (str): Base directory path where the .mat files are stored
        
    Returns:
        numpy.ndarray: Concatenated array of historical and repeated future data
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

def load_bilstm_data(base_path='../data/thetao/'):
    """
    Load data for BiLSTM model (original and transposed).
    
    Args:
        base_path (str): Base directory path
        
    Returns:
        tuple: Historical data, list of SSP data tuples, ORAS5 data
    """
    # Load CMIP6 historical data for both inputs
    cmip6_hist_l1 = np.load(f'{base_path}cmip6_thetao_1958_2014_2d.npy')
    cmip6_hist_l1 = cmip6_hist_l1[:,0:-2]
    
    cmip6_hist_l2 = np.load(f'{base_path}cmip6_thetao_transpose_ts_1958_2014_2d.npy')
    cmip6_hist_l2 = cmip6_hist_l2[:,0:-2]
    
    # Load ORAS5 historical data
    oras5_sst_2014 = np.load(f'{base_path}oras5_temp_1958_2014_2d.npy')
    
    # Define historical data tuple
    historical_data = (cmip6_hist_l1, cmip6_hist_l2, oras5_sst_2014)
    
    # Load SSP scenario data for 2015-2022
    scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    oras5_ssp = np.load(f'{base_path}oras5_temp_2015_2022_2d.npy')
    
    ssp_data_list = []
    for scenario in scenarios:
        cmip6_ssp_l1 = np.load(f'{base_path}cmip6_thetao_{scenario}_2015_2022_2d.npy')
        cmip6_ssp_l2 = np.load(f'{base_path}cmip6_thetao_trans_{scenario}_2015_2022_2d.npy')
        
        # Adjust dimensions
        cmip6_ssp_l1 = cmip6_ssp_l1[:,0:-2]
        cmip6_ssp_l2 = cmip6_ssp_l2[:,0:-2]
        
        ssp_data_list.append((cmip6_ssp_l1, cmip6_ssp_l2, oras5_ssp))
    
    return historical_data, ssp_data_list, oras5_ssp

def load_convlstm_data(base_path='../data/thetao/'):
    """
    Load data for ConvLSTM model.
    
    Args:
        base_path (str): Base directory path
        
    Returns:
        tuple: CMIP6 historical data, ORAS5 historical data, list of CMIP6 SSP data, ORAS5 future data
    """
    # Load historical data
    cmip6_historical = np.array(scipy.io.loadmat(f'{base_path}cmip6_thetao_1958_2014_fill_diststen.mat')['cmip6_ad_sten'])
    oras5_historical = np.array(scipy.io.loadmat(f'{base_path}oras5_sst_1958_2014_fill_diststen.mat')['oras5_ad_sten'])
    
    # Load SSP scenario data for 2015-2022
    scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    cmip6_ssp_list = []
    
    for scenario in scenarios:
        cmip6_ssp = np.array(scipy.io.loadmat(f'{base_path}cmip6_thetao_{scenario}_2015_2022_fill_diststen.mat')['cmip6_ad_sten'])
        cmip6_ssp = cmip6_ssp[0:72, :, :]  # Trim to first 72 timesteps
        cmip6_ssp_list.append(cmip6_ssp)
    
    # Load ORAS5 future data
    oras5_future = np.array(scipy.io.loadmat(f'{base_path}oras5_sst_2015_2022_fill_diststen.mat')['oras5_ad_sten'])
    oras5_future = oras5_future[0:72, :, :]  # Trim to first 72 timesteps
    
    return cmip6_historical, oras5_historical, cmip6_ssp_list, oras5_future

def load_ocean_mask(mask_path='../data/oras5_mask.mat'):
    """
    Load ocean mask for custom loss functions.
    
    Args:
        mask_path (str): Path to mask file
        
    Returns:
        numpy.ndarray: Ocean mask (1=ocean, 0=land)
    """
    return np.array(scipy.io.loadmat(mask_path)['mask1'])

def load_climatology_mean(base_path='../data/sst/'):
    """
    Load climatology mean for normalization.
    
    Args:
        base_path (str): Base directory path
        
    Returns:
        numpy.ndarray: Climatology mean values
    """
    mean_file = f'{base_path}oras5_historical_sst_1958_2020_mean.mat'
    return np.array(scipy.io.loadmat(mean_file)['oras5_mclim'])
