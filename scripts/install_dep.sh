#!/usr/bin/env bash
set -e

# Install system dependencies for TA-Lib
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing TA-Lib dependencies for macOS..."
    brew install ta-lib
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing TA-Lib dependencies for Linux..."
    sudo apt-get update
    sudo apt-get install -y ta-lib
fi

echo "Creating virtual environment..."
#conda create -n ml_env python=3.11
#conda activate ml_env

echo "Installing project dependencies..."
pip install --upgrade pip

# Core data science & ML
pip install numpy==1.24.0 pandas scikit-learn

# Finance data & Technical Analysis
pip install yfinance
# For Python 3.11, TA-Lib may fail to build with pip.
conda install -c conda-forge ta-lib


pip install torch
pip install tensorboard
pip install optuna 
pip install psutil
# Computer vision & AI
pip install opencv-python opencv-contrib-python
pip install tensorflow tensorflow-io tensorflow-addons

# Natural language processing
pip install blis
pip install nltk spacy textblob
python -m spacy download en_core_web_sm

# Web scraping & APIs
pip install requests beautifulsoup4
pip install newsapi-python praw finnhub-python

# Visualization & UI
pip install matplotlib seaborn rich

# Error handling & utilities
pip install retrying tqdm pyyaml coloredlogs

# Testing
pip install pytest pytest-cov pytest-xdist

# Documentation
pip install sphinx sphinx_rtd_theme

echo "All dependencies installed successfully!"