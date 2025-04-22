from setuptools import setup, find_packages

setup(
    name="climate-bias-correction",
    version="0.1.0",
    author="Climate Model Bias Correction Toolkit Contributors",
    author_email="your.email@example.com",
    description="Deep learning toolkit for climate model bias correction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/climate-bias-correction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "numpy>=1.23.5",
        "scipy>=1.10.1",
        "pandas>=1.5.3",
        "matplotlib>=3.7.1",
        "scikit-learn>=1.2.2",
        "colorama>=0.4.6",
        "h5py>=3.8.0",
        "netCDF4>=1.6.3",
        "cartopy>=0.21.1"
    ],
    entry_points={
        "console_scripts": [
            "run-sst-correction=src.run_sst_correction:main",
        ],
    },
)
