# HTS-Oracle
A deep learning, multi-modal ensemble framework for molecular activity prediction in high throughput screening (HTS) campaigns.
# Overview
Despite rapid advances in in silico drug discovery approaches, high throughput screening (HTS) assays remain the key methodology for obtaining initial hit molecules, especially for novel targets that have been considered undruggable. However, traditional HTS campaigns typically yield low hit rates of just 1–2%, making them costly and time-consuming endeavors.
HTS-Oracle addresses these challenges by integrating transformer-based chemical language models (ChemBERTa) with traditional cheminformatics features using an advanced ensemble learning strategy. This approach significantly improves hit rates while maximizing the value of negative HTS results that are typically discarded.

# Repository Structure
CD28-project/

├── HTS.py                 # Main training script for the ensemble model

├── HTSOracle.py          # Streamlit web application for predictions

├── libraries             # Required libraries for model training

├── cd28_new.yml         # Conda environment configuration

└── README.md            # This file

# Installation

# Prerequisites
Python 3.8 or higher
Conda package manager
CUDA-compatible GPU (recommended for transformer models)

# Environment Setup
1. Create and activate the conda environment:
bashconda env create -f cd28_new.yml
conda activate cd28_new

2. Install additional training libraries:
bash# Install libraries from the libraries file
pip install -r libraries

# Usage

# Training the Model

To train the HTS-Oracle ensemble model:

python HTS.py

This script will:
  1. Load and preprocess molecular data (SMILES, activity data)
  2. Generate ChemBERTa embeddings and cheminformatics features
  3. Train the multi-modal ensemble model
  4. Evaluate model performance and save trained weights

# Running the Web Application
To launch the Streamlit prediction interface:

  streamlit run HTSOracle.py

The application will be available at http://localhost:8501 and provides:

  - Molecular activity prediction from SMILES input
  - Batch prediction capabilities
  - Model confidence scores
  - Visualization of chemical space and predictions


# Hardware Requirements

# Minimum Requirements
RAM: 8GB minimum (16GB recommended)
Storage: 5GB free space for models and data
CPU: Multi-core processor (4+ cores recommended)

# Recommended Requirements
RAM: 16GB or more
GPU: NVIDIA GPU with 8GB+ VRAM (for CUDA acceleration)
Storage: 10GB+ free space
CPU: 8+ cores for parallel processing

# Input Data Requirements

# Required CSV Files
  library.csv
    Columns: ID, Smiles
      Contains molecular library with SMILES strings
  positives.csv
    Columns: ID
      Contains IDs of positive samples for binary classification
# Data Format
SMILES: Valid chemical SMILES notation
IDs: Unique identifiers matching between files
Encoding: UTF-8 recommended

# Compatibility Notes
OS: Cross-platform (Windows, macOS, Linux)
Python: Tested on 3.7, 3.8, 3.9, 3.10
CUDA: Optional, versions 11.0+ recommended
Memory: Scalable based on available resources
