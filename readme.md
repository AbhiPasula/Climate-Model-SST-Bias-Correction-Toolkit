# Global Climate Model Error Correction Toolkit

This repository contains a set of deep learning models for global climate model error correction, with a focus on sea surface temperature (SST). The toolkit includes three different neural network architectures (UNet, BiLSTM, and ConvLSTM) that can be applied to correct systematic biases in CMIP6 CNRM-CM6 climate model outputs.

## Overview

Climate models like those in CMIP6 have systematic biases when compared to reanalysis data like ORAS5. This toolkit applies deep learning techniques to learn and correct these biases, producing more accurate climate projections for future scenarios (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5).

![Model Overview](./docs/model_overview.png)

## Model Architectures

### 1. UNet 

The UNet model focuses on spatial patterns and relationships:

- **Architecture**: Symmetric encoder-decoder with skip connections
- **Features**: 
  - 5 downsampling/upsampling layers with filter sizes from 32 to 1024
  - Tanh activation functions for handling normalized ocean data
  - Dropout (0.2) between blocks to prevent overfitting
  - Custom MSE loss function with ocean mask to focus on relevant regions

### 2. BiLSTM 

The BiLSTM model captures temporal dependencies in climate data:

- **Architecture**: Dual input bidirectional LSTM network
- **Features**:
  - Parallel BiLSTM layers (512 units) for original and transposed inputs
  - Concatenation of bidirectional features to capture multi-directional patterns
  - Dense output layer with ReLU activation
  - MAE loss function for robust training

### 3. ConvLSTM 

The ConvLSTM model integrates both spatial and temporal patterns:

- **Architecture**: Stacked ConvLSTM2D layers with encoder-decoder structure
- **Features**:
  - 8 ConvLSTM2D layers with varying filter counts (8→48→8)
  - Batch normalization between layers
  - Varied kernel sizes (7×7 → 3×3 → 7×7)
  - Final Conv3D layer for spatiotemporal integration
  - MSE loss function

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/climate-bias-correction.git
cd climate-bias-correction

# Create a conda environment (recommended)
conda create -n climate-bias python=3.9
conda activate climate-bias

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The toolkit provides a unified command-line interface for all models:

```bash
# General usage
python run_sst_correction.py --method [unet|bilstm|convlstm] [options]

# Examples:
# Run UNet with custom parameters
python run_sst_correction.py --method unet --epochs 1000 --batch_size 64

# Load a pre-trained BiLSTM model
python run_sst_correction.py --method bilstm --load_model

# Run ablation study with ConvLSTM
python run_sst_correction.py --method convlstm --ablation --epochs 100
```

### Interactive Mode

For easier use, run the script without arguments for an interactive prompt:

```bash
python run_sst_correction.py
```

This will guide you through:
1. Choosing a correction method
2. Loading existing models or training new ones
3. Setting hyperparameters
4. Running ablation studies

### Output Structure

Results are saved in the following structure:

```
output/
├── models/                       # Saved model files
│   ├── unet_sst_model.h5
│   ├── bilstm_model_best.h5
│   └── convlstm_model.keras
├── data/                         # Processed input/output data
│   ├── cmip6_train.npy
│   └── ...
├── unet_sst_ssp126_2023_2100.npy # Bias-corrected predictions
├── bilstm_thetao_ssp245_2023_2100.npy
├── convlstm_thetao_ssp370_2023_2100.npy
└── ...
```

## Model Training Process

All models follow a similar training process:

1. **Data Loading**: Historical (1958-2014) and near-future (2015-2020) data from CMIP6 and ORAS5
2. **Preprocessing**: Normalization, mean removal, dimension adjustment, and sequence creation
3. **Training**: Cross-validation with 80/20 split, Adam optimizer (lr=1e-4), and early stopping
4. **Validation**: Performance evaluation on validation data using MAE/MSE metrics
5. **Prediction**: Generation of bias-corrected projections for future scenarios (2023-2100)

## Hyperparameters

Initial weights are sampled from the He normal distribution. The Adam optimizer, with a learning rate of 0.0001, is utilized for training. We performed hyperparameter tuning through cross-validation, sweeping batch sizes from 32 to 128 and testing various learning rates, finally finding that a batch size of 64 is optimal. The validation set, comprising 20% of data from 1958 to 2020, includes historical simulations and projections. Hyperparameters are tuned through cross-validation on this validation set, selecting the model with the best performance for final use.

The architecture uses tanh activation functions in the convolutional layers to better handle the normalized ocean data values, which can include both positive and negative anomalies. Dropout layers (0.2) are incorporated between encoder and decoder blocks to prevent overfitting, especially important given the spatial and temporal correlations in climate data. For the UNet architecture, we implemented a symmetric design with five downsampling and upsampling operations, using filter sizes ranging from 32 to 1024 through the network depth. The model was trained for 1000 epochs with early stopping based on validation loss to prevent overfitting, and the best-performing model weights were saved for future prediction. A custom MSE loss function with an ocean mask was implemented to ensure the model only focuses on relevant ocean regions and ignores land areas in the calculation of errors.

## Results Visualization

The toolkit includes visualization tools for model evaluation:

```python
# Plot training history
python plot_training_history.py --method unet
``

## Citation

This manuscript for this work is under review in the IOP Science Machine Learning Earth journal.

```
@article{pasula2025global,
  title={Global Climate Model Error Correction using Data Driven Deep Learning},
  author={Abhishek, Pasula. and Deepak, Subramani.},
  journal={IOP Science Machine Learning Earth},
  year={2025},
  volume={1},
  pages={1--20}
}
```
## Acknowledgements

We acknowledge the World Climate Research Programme's Working Group on Coupled Modelling, which is responsible for CMIP, and we thank the climate modeling groups for producing and making available their model output. We also thank the EU Copernicus Marine Service for providing the ORAS5 observational dataset.
