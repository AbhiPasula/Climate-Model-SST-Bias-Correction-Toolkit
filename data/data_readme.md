# Climate Model Bias Correction Data

This directory should contain the climate data needed to train and test the bias correction models. Due to the large size of climate datasets, these files are not included in the repository and must be downloaded separately.

## Required Data Structure

```
data/
├── sst/                          # Sea Surface Temperature data for UNet
│   ├── cmip6_sst_1958_2014_fill_diststen.mat
│   ├── cmip6_sst_ssp126_2015_2022_fill_diststen.mat
│   ├── cmip6_sst_ssp245_2015_2022_fill_diststen.mat
│   ├── cmip6_sst_ssp370_2015_2022_fill_diststen.mat
│   ├── cmip6_sst_ssp585_2015_2022_fill_diststen.mat
│   ├── cmip6_sst_ssp126_2023_2100_fill_diststen.mat
│   ├── cmip6_sst_ssp245_2023_2100_fill_diststen.mat
│   ├── cmip6_sst_ssp370_2023_2100_fill_diststen.mat
│   ├── cmip6_sst_ssp585_2023_2100_fill_diststen.mat
│   ├── oras5_sst_1958_2014_fill_diststen.mat
│   ├── oras5_sst_2015_2022_fill_diststen.mat
│   └── oras5_historical_sst_1958_2020_mean.mat
│   ├── cmip6_thetao_1958_2014_2d.npy
│   ├── cmip6_thetao_transpose_ts_1958_2014_2d.npy
│   ├── cmip6_thetao_ssp126_2015_2022_2d.npy
│   ├── cmip6_thetao_ssp245_2015_2022_2d.npy
│   ├── cmip6_thetao_ssp370_2015_2022_2d.npy
│   ├── cmip6_thetao_ssp585_2015_2022_2d.npy
│   ├── cmip6_thetao_trans_ssp126_2015_2022_2d.npy
│   ├── cmip6_thetao_trans_ssp245_2015_2022_2d.npy
│   ├── cmip6_thetao_trans_ssp370_2015_2022_2d.npy
│   ├── cmip6_thetao_trans_ssp585_2015_2022_2d.npy
│   ├── cmip6_thetao_ssp126_2023_2100_2d.npy
│   ├── cmip6_thetao_ssp245_2023_2100_2d.npy
│   ├── cmip6_thetao_ssp370_2023_2100_2d.npy
│   ├── cmip6_thetao_ssp585_2023_2100_2d.npy
│   ├── cmip6_thetao_trans_ssp126_2023_2100_2d.npy
│   ├── cmip6_thetao_trans_ssp245_2023_2100_2d.npy
│   ├── cmip6_thetao_trans_ssp370_2023_2100_2d.npy
│   ├── cmip6_thetao_trans_ssp585_2023_2100_2d.npy
│   ├── oras5_temp_1958_2014_2d.npy
│   ├── oras5_temp_2015_2022_2d.npy
│   ├── datanomissing.npy
│   ├── cmip6_thetao_1958_2014_fill_diststen.mat
│   ├── cmip6_thetao_ssp126_2015_2022_fill_diststen.mat
│   ├── cmip6_thetao_ssp245_2015_2022_fill_diststen.mat
│   ├── cmip6_thetao_ssp370_2015_2022_fill_diststen.mat
│   ├── cmip6_thetao_ssp585_2015_2022_fill_diststen.mat
│   ├── cmip6_thetao_ssp126_2023_2100_fill_diststen.mat
│   ├── cmip6_thetao_ssp245_2023_2100_fill_diststen.mat
│   ├── cmip6_thetao_ssp370_2023_2100_fill_diststen.mat
│   ├── cmip6_thetao_ssp585_2023_2100_fill_diststen.mat
│   ├── oras5_sst_1958_2014_fill_diststen.mat
│   └── oras5_sst_2015_2022_fill_diststen.mat
│
└── oras5_mask.mat                # Ocean mask for custom loss function
```

## Data Sources

### CMIP6 Data
CMIP6 (Coupled Model Intercomparison Project Phase 6) data can be downloaded from the Earth System Grid Federation (ESGF):
- [ESGF Portal](https://esgf-node.llnl.gov/projects/cmip6/)

CMIP6 projections are downloaded from CNRM-CM6 historical and four SSP projections for the Bay of Bengal region.

The specific CMIP6 model used in this project is CNRM-CM6-1 from Centre National de Recherches Météorologiques (CNRM).

### ORAS5 Data
ORAS5 (Ocean ReAnalysis System 5) data is available from the Copernicus Marine Service:
- [Copernicus Marine Service](https://marine.copernicus.eu/)

ORAS5 reanalysis data is downloaded for the Bay of Bengal region.

## Data Preprocessing

The raw data has been preprocessed to:
1. Remap to a common grid (85 x 85)
2. Fill missing values in coastal regions
3. Calculate and store monthly climatological means from ORAS5
4. Create the transposed forms needed for BiLSTM model

### File Format Details

- `.mat` files: MATLAB matrices containing:
  - `cmip6_ad_sten`: CMIP6 model data (dimensions: time x lat x lon)
  - `oras5_ad_sten`: ORAS5 reanalysis data (dimensions: time x lat x lon)
  - `oras5_mclim`: ORAS5 climatological monthly means (dimensions: 12 x lat x lon)
  - `mask1`: Ocean mask (1=ocean, 0=land)

- `.npy` files: NumPy arrays:
  - 2D files: Flattened spatial patterns (dimensions: time x spatial_points)
  - Transposed files: Time-space transposed for bidirectional learning
  - `datanomissing.npy`: Indices of valid ocean points


## Creating Your Own Dataset

If you want to use different climate models or variables, you'll need to preprocess the data into the same format. See the `preprocessing/` directory for scripts to help with this task.


