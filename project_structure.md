# Project Structure

This document describes the organization of the climate model bias correction toolkit project.

```
climate-bias-correction/
│
├── data/                          # Data directory (not included in repo)
│   ├── sst/                       # Sea Surface Temperature data
│   ├── thetao/                    # Ocean temperature data
│   └── README.md                  # Instructions for obtaining data
│
├── src/                           # Source code
│   ├── models/                    # Model implementations
│   │   ├── __init__.py
│   │   ├── unet.py                # UNet model architecture
│   │   ├── bilstm.py              # BiLSTM model architecture
│   │   └── convlstm.py            # ConvLSTM model architecture
│   │
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── data_loader.py         # Data loading functions
│   │   ├── preprocessing.py       # Data preprocessing functions
│   │   └── metrics.py             # Evaluation metrics
│   │
│   ├── visualization/             # Visualization tools
│   │   ├── __init__.py
│   │   ├── plot_training.py       # Training history plotting
│   │   └── plot_results.py        # Results visualization
│   │
│   ├── sst_unet_reorganised.py    # UNet SST bias correction script
│   ├── sst_bilstm.py              # BiLSTM SST bias correction script
│   ├── sst_convlstm.py            # ConvLSTM SST bias correction script
│   └── run_sst_correction.py      # Main execution script
│
├── output/                        # Output directory (created by scripts)
│   ├── models/                    # Saved models
│   ├── data/                      # Processed data
│   └── figures/                   # Generated figures
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── exploratory_analysis.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── docs/                          # Documentation
│   ├── model_overview.png
│   ├── training_process.md
│   └── evaluation_metrics.md
│
├── tests/                         # Test cases
│   ├── test_models.py
│   ├── test_data_loader.py
│   └── test_preprocessing.py
│
├── requirements.txt               # Project dependencies
├── setup.py                       # Package setup script
├── README.md                      # Project README
├── LICENSE                        # Project license
└── .gitignore                     # Git ignore file
```

## Key Components

### Models

The model implementations (UNet, BiLSTM, ConvLSTM) are located in the `src/models/` directory, but the main application scripts are in the root of the `src/` directory:

- **sst_unet_reorganised.py**: Handles UNet model training and prediction
- **sst_bilstm.py**: Handles BiLSTM model training and prediction
- **sst_convlstm.py**: Handles ConvLSTM model training and prediction

### Execution Script

The main script for running the SST correction is `run_sst_correction.py`, which provides a unified interface for all models.

### Data

The required data should be placed in the `data/` directory following the structure outlined in the main README. This directory is not included in the repository due to size constraints.

### Output

All outputs (models, predictions, figures) are saved to the `output/` directory, which is automatically created by the scripts.

## Getting Started

See the main README.md file for installation and usage instructions.
