# HTS-Oracle
A deep learning, multi-modal ensemble framework for molecular activity prediction in high throughput screening (HTS) campaigns.
# Overview
Despite rapid advances in in silico drug discovery approaches, high throughput screening (HTS) assays remain the key methodology for obtaining initial hit molecules, especially for novel targets that have been considered undruggable. However, traditional HTS campaigns typically yield low hit rates of just 1–2%, making them costly and time-consuming endeavors.
HTS-Oracle addresses these challenges by integrating transformer-based chemical language models (ChemBERTa) with traditional cheminformatics features using an advanced ensemble learning strategy. This approach significantly improves hit rates while maximizing the value of negative HTS results that are typically discarded.

# Repository Structure
HTS-Oracle/
├── HTS.py                 # Main training script for the ensemble model
├── HTSOracle.py          # Streamlit web application for predictions
├── libraries             # Required libraries for model training
├── cd28_new.yml         # Conda environment configuration
└── README.md            # This file
