"""
Visualization tools for climate model bias correction.

This package contains visualization functions for:
- Training history
- Bias maps
- Time series
- Scenario comparisons
"""

from .plot_training import plot_training_history
from .plot_maps import plot_bias_map, plot_correction_comparison
from .plot_timeseries import plot_timeseries, plot_scenario_comparison

__all__ = [
    'plot_training_history',
    'plot_bias_map',
    'plot_correction_comparison',
    'plot_timeseries',
    'plot_scenario_comparison'
]
