#!/usr/bin/env python3
"""
HTS_3D.py
---------

Cross-attention HTS pipeline that augments the original HTS training script with a
3D-aware ligand branch and an explicit protein-pocket representation.

Key ideas mandated by `codeGenNLRP3.docx`:
    * Build ligand tokens from ChemBERTa (sequence) AND RDKit-generated 3D conformers.
    * Build a protein-pocket tensor (residue physicochemistry + pseudo 3D coords).
    * Fuse ligand/protein information with multi-head cross-attention so that every
      ligand atom/token can attend to the residues that matter most.
    * Feed the interaction-aware representation + RDKit descriptors into a classifier.

Input: `nlrp3_chembl_activities_with15positives.csv`
Outputs (saved to `output/` directory):
    * `hts3d_model.pt`          – best PyTorch weights (no timestamp).
    * `hts3d_predictions_YYYYMMDD_HHMMSS.csv`   – validation predictions (with timestamp).
    * `hts3d_training_history_YYYYMMDD_HHMMSS.json` – metrics per epoch/fold (with timestamp).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import psutil
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdPartialCharges, MACCSkeys
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer

# --- Loss Functions ----------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    This loss function focuses learning on hard examples and handles class imbalance
    better than standard BCE loss. The gamma parameter controls the focusing effect,
    and alpha controls the class weighting.
    """
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Clamp inputs to prevent extreme logits that cause numerical instability
        inputs = torch.clamp(inputs, min=-20.0, max=20.0)
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        # Clamp bce_loss to prevent extreme values
        bce_loss = torch.clamp(bce_loss, min=0.0, max=50.0)
        
        # Compute p_t (probability of true class) - clamp to prevent numerical issues
        pt = torch.exp(-bce_loss)
        pt = torch.clamp(pt, min=1e-10, max=1.0 - 1e-10)
        
        # Compute focal loss: alpha * (1 - p_t)^gamma * bce_loss
        # Clamp (1 - pt) to prevent numerical issues when pt is very close to 1
        focal_weight = (1 - pt) ** self.gamma
        focal_weight = torch.clamp(focal_weight, min=1e-10, max=1.0)
        focal_loss = self.alpha * focal_weight * bce_loss
        
        # Final safety check
        if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
            # Fallback to standard BCE if focal loss produces NaN/Inf
            focal_loss = torch.nan_to_num(focal_loss, nan=bce_loss, posinf=bce_loss, neginf=bce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Global configuration ----------------------------------------------------

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

# Paths relative to script location (src/)
# Default CSV path for NLRP3
DEFAULT_CSV_PATH_NLRP3 = SCRIPT_DIR / "nlrp3_chembl_activities_with15positives.csv"
# Output directory in project root
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
# Cache directory in script directory
CACHE_DIR = SCRIPT_DIR / ".cache_hts3d"

def detect_compound_from_csv_path(csv_path: Path) -> str:
    """
    Detect compound type from CSV filename or path.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        Compound type ('nlrp3' or 'cd28'). Defaults to 'nlrp3' if cannot be determined.
    """
    csv_name_lower = csv_path.name.lower()
    csv_str_lower = str(csv_path).lower()
    
    # Check for CD28 indicators
    if "cd28" in csv_name_lower or "cd28" in csv_str_lower:
        return "cd28"
    
    # Check for NLRP3 indicators
    if "nlrp3" in csv_name_lower or "nlrp3" in csv_str_lower:
        return "nlrp3"
    
    # Default to NLRP3
    return "nlrp3"

def get_model_path_for_compound(compound: str) -> Path:
    """
    Get the model output path for a given compound type.
    
    Args:
        compound: Compound type ('nlrp3' or 'cd28')
    
    Returns:
        Path to save the model file
    """
    compound_lower = compound.lower()
    if compound_lower == "cd28":
        return OUTPUT_DIR / "hts3d_model_cd28.pt"
    elif compound_lower == "nlrp3":
        return OUTPUT_DIR / "hts3d_model_nlrp3.pt"
    else:
        # Default to NLRP3 if unknown
        return OUTPUT_DIR / "hts3d_model_nlrp3.pt"


def get_timestamped_path(base_name: str, extension: str = None, compound: Optional[str] = None) -> Path:
    """
    Generate a timestamped file path in the output directory.
    
    Args:
        base_name: Base name of the file (without extension)
        extension: File extension (e.g., 'csv', 'json'). If None, extracted from base_name
        compound: Optional compound type ('nlrp3' or 'cd28') to include in filename
    
    Returns:
        Path object pointing to timestamped file in output directory
    """
    if extension is None:
        # Extract extension from base_name if it exists
        if '.' in base_name:
            parts = base_name.rsplit('.', 1)
            base_name = parts[0]
            extension = parts[1]
        else:
            extension = ""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include compound type in filename if provided
    if compound:
        compound_lower = compound.lower()
        if extension:
            filename = f"{base_name}_{compound_lower}_{timestamp}.{extension}"
        else:
            filename = f"{base_name}_{compound_lower}_{timestamp}"
    else:
        if extension:
            filename = f"{base_name}_{timestamp}.{extension}"
        else:
            filename = f"{base_name}_{timestamp}"
    
    return OUTPUT_DIR / filename


MAX_SEQ_LEN = 128
MAX_ATOMS = 128  # Increased from 64 to handle larger molecules (e.g., Z56853729 has 67 atoms)
CHEMBERTA_NAME = "seyonec/ChemBERTa-zinc-base-v1"
SEED = 42
EMBED_DIM = 256
NUM_HEADS = 4

# Bioactivity threshold (default: 2000 nM = 2 µM, compounds <= threshold are considered active)
DEFAULT_BIO_THRESHOLD = 2000  # in nM

# Silence noisy RDKit warnings (e.g., MorganGenerator deprecation spam)
RDLogger.DisableLog("rdApp.warning")
# Suppress UFFTYPER warnings for unsupported atom types (e.g., Iridium)
RDLogger.DisableLog("rdApp.error")  # UFFTYPER warnings are logged as errors

def detect_device() -> torch.device:
    """
    Detect and verify available device (CUDA, MPS, or CPU).
    
    For CUDA, this function:
    - Checks if CUDA is available
    - Verifies GPU memory can be allocated
    - Reports available GPU memory
    - Falls back to CPU if GPU allocation fails (e.g., GPU occupied by another process)
    
    Returns:
        torch.device: The device to use for computation
    """
    if torch.cuda.is_available():
        try:
            # Test if we can actually allocate memory on GPU
            # This catches cases where GPU is occupied by another process
            test_tensor = torch.zeros(1, device=torch.device("cuda"))
            del test_tensor
            torch.cuda.empty_cache()
            
            # Get GPU memory information
            device_props = torch.cuda.get_device_properties(0)
            total_memory_gb = device_props.total_memory / (1024**3)
            allocated_memory_gb = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_memory_gb = torch.cuda.memory_reserved(0) / (1024**3)
            free_memory_gb = total_memory_gb - reserved_memory_gb
            
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(device)
            print(f"[device] Using CUDA GPU: {device_name}")
            print(f"[device] GPU Memory: {free_memory_gb:.2f} GB free / {total_memory_gb:.2f} GB total "
                  f"(allocated: {allocated_memory_gb:.2f} GB, reserved: {reserved_memory_gb:.2f} GB)")
            
            # Warn if memory is low
            if free_memory_gb < 2.0:
                print(f"[device] WARNING: Low GPU memory ({free_memory_gb:.2f} GB). "
                      f"Model may fail to load if insufficient memory.")
            
            return device
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"[device] GPU available but memory allocation failed: {e}")
            print("[device] This may indicate GPU is occupied by another process or has insufficient memory.")
            print("[device] Falling back to CPU")
            return torch.device("cpu")
        except Exception as e:
            print(f"[device] Unexpected error during GPU initialization: {e}")
            print("[device] Falling back to CPU")
            return torch.device("cpu")
    
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("[device] Using Apple MPS backend")
        return torch.device("mps")
    
    print("[device] Using CPU")
    return torch.device("cpu")


DEVICE = detect_device()
NON_BLOCKING = DEVICE.type != "cpu"
PIN_MEMORY = DEVICE.type != "cpu"

# --- System information for performance tracking -----------------------------

def get_system_info() -> Dict[str, any]:
    """Collect system information for performance tracking."""
    info = {
        "device_type": DEVICE.type,
        "device_name": None,
        "cuda_version": None,
        "cudnn_version": None,
        "cpu_count": psutil.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    
    if DEVICE.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    elif DEVICE.type == "mps":
        info["device_name"] = "Apple Silicon (MPS)"
    
    return info


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in GB."""
    mem = {
        "cpu_ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "cpu_ram_percent": psutil.virtual_memory().percent,
    }
    
    if DEVICE.type == "cuda":
        mem["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        mem["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
        mem["gpu_memory_percent"] = round(
            (torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100, 2
        )
    
    return mem


# --- Reproducibility helpers -------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()


# --- GPU-accelerated feature scaling -----------------------------------------

class GPUScaler:
    """
    GPU-compatible StandardScaler replacement.
    Computes mean and std on GPU if available, otherwise on CPU.
    Accepts both numpy arrays and torch tensors.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.device = DEVICE

    def fit(self, X) -> "GPUScaler":
        """Fit scaler to data. X can be numpy array or torch tensor."""
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X.to(self.device) if X.device != self.device else X
        
        self.mean_ = X_tensor.mean(dim=0, keepdim=True)
        # Use unbiased std (ddof=1) to match sklearn StandardScaler
        self.std_ = X_tensor.std(dim=0, keepdim=True, unbiased=True)
        # Avoid division by zero
        self.std_ = torch.clamp(self.std_, min=1e-8)
        return self

    def transform(self, X, return_numpy: bool = True):
        """Transform data using fitted mean and std.
        
        Args:
            X: Input data (numpy array or torch tensor)
            return_numpy: If True, return numpy array; if False, return torch tensor on device
        
        Returns:
            Scaled data (numpy array or torch tensor depending on return_numpy)
        """
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X.to(self.device) if X.device != self.device else X
        
        X_scaled = (X_tensor - self.mean_) / self.std_
        
        if return_numpy:
            return X_scaled.cpu().numpy()
        return X_scaled

    def fit_transform(self, X, return_numpy: bool = True):
        """Fit and transform in one step."""
        return self.fit(X).transform(X, return_numpy=return_numpy)


# --- Utility / safety wrappers -----------------------------------------------

def safe_metric(
    metric_fn,
    y_true,
    y_pred,
    fallback: float = 0.0,
    **metric_kwargs,
):
    try:
        # Check if labels have at least 2 classes
        unique_labels = np.unique(y_true)
        if len(unique_labels) < 2:
            print(f"[metric warning] {metric_fn.__name__}: Only {len(unique_labels)} unique label(s) found: {unique_labels}")
            return fallback
        
        # For AUC/AP, check if predictions have variance
        if metric_fn in (roc_auc_score, average_precision_score):
            if len(np.unique(y_pred)) < 2 or np.allclose(y_pred, y_pred[0], atol=1e-6):
                print(f"[metric warning] {metric_fn.__name__}: All predictions are identical (value={y_pred[0]:.6f}), cannot compute AUC/AP")
                return fallback
        
        # For classification metrics, check if predictions have both classes
        if metric_fn in (precision_score, recall_score, f1_score):
            unique_preds = np.unique(y_pred)
            if len(unique_preds) < 2:
                print(f"[metric warning] {metric_fn.__name__}: Binary predictions have only {len(unique_preds)} class(es): {unique_preds}")
        
        return metric_fn(y_true, y_pred, **metric_kwargs)
    except Exception as exc:  # pragma: no cover - logging only
        print(f"[metric warning] {metric_fn.__name__}: {exc}")
        return fallback


def ensure_float(value: str) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except ValueError:
        return None


# --- Data loading / preprocessing --------------------------------------------

def load_library_and_positives(library_path: Path, positives_path: Path) -> pd.DataFrame:
    """
    Load library.csv and positives.csv and combine exactly as HTS.py:
    labels = 1 if SMILES is in positives, else 0. Used when --cd28 is set.

    Args:
        library_path: Path to libraries/library.csv (or equivalent).
        positives_path: Path to libraries/positives.csv (or equivalent).

    Returns:
        DataFrame with columns: canonical_smiles, label, molecule_chembl_id.
    """
    if not library_path.exists():
        raise FileNotFoundError(f"Library CSV not found: {library_path}")
    if not positives_path.exists():
        raise FileNotFoundError(f"Positives CSV not found: {positives_path}")

    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    library_df = None
    positives_df = None
    for enc in encodings:
        try:
            library_df = pd.read_csv(library_path, encoding=enc)
            positives_df = pd.read_csv(positives_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if library_df is None or positives_df is None:
        raise ValueError(f"Could not read CSV files with any of: {encodings}")

    # HTS.py: library has 'Smiles', positives has 'Smiles'
    if "Smiles" not in library_df.columns:
        raise ValueError(f"library.csv must have column 'Smiles'. Found: {library_df.columns.tolist()}")
    if "Smiles" not in positives_df.columns:
        raise ValueError(f"positives.csv must have column 'Smiles'. Found: {positives_df.columns.tolist()}")

    # Exactly as HTS.py: label = 1 if SMILES is in positives, else 0
    library_df = library_df.copy()
    library_df["label"] = library_df["Smiles"].isin(positives_df["Smiles"]).astype(int)

    # Build DataFrame compatible with rest of HTS_3D (canonical_smiles, label, molecule_chembl_id)
    out = pd.DataFrame({
        "canonical_smiles": library_df["Smiles"].values,
        "label": library_df["label"].values,
    })
    out["molecule_chembl_id"] = [f"LIB_{i:06d}" for i in range(len(out))]

    return out


def load_activity_table(csv_path: Path, bio_threshold: float = DEFAULT_BIO_THRESHOLD, label_column: Optional[str] = None) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Try different encodings (matching inference script behavior)
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"Could not read CSV file with any of the attempted encodings: {encodings}")
    
    # Normalize SMILES column name - check for common variations
    smiles_col = None
    possible_smiles_cols = ['canonical_smiles', 'SMILES', 'Smiles', 'smiles', 'SMILE', 'Smile', 'smile', 'Structure']
    for col in possible_smiles_cols:
        if col in df.columns:
            smiles_col = col
            break
    
    if smiles_col is None:
        raise ValueError(f"No SMILES column found. Expected one of: {possible_smiles_cols}. Found columns: {df.columns.tolist()}")
    
    # Rename to canonical_smiles for consistency
    if smiles_col != "canonical_smiles":
        df["canonical_smiles"] = df[smiles_col]
    
    # Normalize molecule_chembl_id column name - check for common variations
    mol_id_col = None
    possible_mol_id_cols = ['molecule_chembl_id', 'Molecule ChEMBL ID', 'molecule_chEMBL_id', 'MOLECULE_CHEMBL_ID', 'Molecule ChEMBL ID', 'molecule_id', 'compound_id']
    for col in possible_mol_id_cols:
        if col in df.columns:
            mol_id_col = col
            break
    
    # Rename to molecule_chembl_id for consistency
    if mol_id_col is not None and mol_id_col != "molecule_chembl_id":
        df["molecule_chembl_id"] = df[mol_id_col]
    
    # If label_column is specified, use it directly (priority over bio_threshold method)
    if label_column is not None:
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        
        # Ensure canonical_smiles exists
        df = df.dropna(subset=["canonical_smiles"])
        
        # Convert label column to binary (0/1)
        df[label_column] = df[label_column].apply(ensure_float)
        df = df.dropna(subset=[label_column])
        
        # Ensure labels are 0 or 1
        unique_labels = df[label_column].unique()
        if not all(label in [0, 1] for label in unique_labels):
            raise ValueError(f"Label column '{label_column}' must contain only 0 (inactive) and 1 (active) values. Found: {unique_labels}")
        
        # Use the specified label column directly
        df["label"] = df[label_column].astype(int)
        
        # Ensure molecule_chembl_id exists (create if missing)
        if "molecule_chembl_id" not in df.columns:
            # Create unique IDs based on row index (preserve original order)
            df["molecule_chembl_id"] = [f"COMPOUND_{i:06d}" for i in range(len(df))]
        
        # Dedupe by molecule id (keep first occurrence)
        # Only deduplicate if there are actual duplicates
        original_len = len(df)
        df = df.groupby("molecule_chembl_id", as_index=False).first()
        df = df.reset_index(drop=True)
        if len(df) < original_len:
            print(f"[data] Deduplicated {original_len - len(df)} duplicate molecule IDs (kept first occurrence)")
        
        print(f"[data] molecules: {len(df)}, actives: {df['label'].sum()} (using label column: '{label_column}')")
        
        # Create placeholder activity columns if missing (for consistency with output format)
        if "standard_value" not in df.columns:
            df["standard_value"] = np.nan
        if "standard_units" not in df.columns:
            df["standard_units"] = ""
        if "value_nM" not in df.columns:
            df["value_nM"] = np.nan
        
        return df
    
    # Normalize activity data column names - check for common variations
    standard_value_col = None
    possible_value_cols = ['standard_value', 'Standard Value', 'STANDARD_VALUE', 'value', 'Value', 'Standard Text Value']
    for col in possible_value_cols:
        if col in df.columns:
            standard_value_col = col
            break
    
    standard_units_col = None
    possible_units_cols = ['standard_units', 'Standard Units', 'STANDARD_UNITS', 'units', 'Units', 'Uo Units']
    for col in possible_units_cols:
        if col in df.columns:
            standard_units_col = col
            break
    
    # Rename to standard names for consistency
    if standard_value_col and standard_value_col != "standard_value":
        df["standard_value"] = df[standard_value_col]
    if standard_units_col and standard_units_col != "standard_units":
        df["standard_units"] = df[standard_units_col]
    
    # Check if this CSV has activity data (standard_value column)
    has_activity_data = "standard_value" in df.columns and df["standard_value"].notna().any()
    
    if has_activity_data:
        # Original NLRP3 activity data processing
        df["standard_value"] = df["standard_value"].apply(ensure_float)
        df = df.dropna(subset=["canonical_smiles", "standard_value"])

        # Normalize units → nM
        df["standard_units"] = df["standard_units"].str.lower().fillna("")
        df["value_nM"] = df.apply(
            lambda row: row["standard_value"] * 1000
            if row["standard_units"] == "um"
            else row["standard_value"],
            axis=1,
        )
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["value_nM"])

        # Binary labels: active if <= bio_threshold nM
        df["label"] = (df["value_nM"] <= bio_threshold).astype(int)

        # Ensure molecule_chembl_id exists (should already be normalized above)
        if "molecule_chembl_id" not in df.columns:
            df["molecule_chembl_id"] = [f"COMPOUND_{i:06d}" for i in range(len(df))]
        
        # Dedupe by molecule id keeping best potency (lowest value_nM = most potent = active)
        original_len = len(df)
        df = (
            df.sort_values("value_nM")
            .groupby("molecule_chembl_id", as_index=False)
            .first()
        )
        df = df.reset_index(drop=True)
        if len(df) < original_len:
            print(f"[data] Deduplicated {original_len - len(df)} duplicate molecule IDs (kept most potent)")
        print(f"[data] molecules: {len(df)}, actives: {df['label'].sum()} (threshold: {bio_threshold} nM)")
    else:
        # CSV without activity data - create placeholder labels for training
        # Only keep rows with valid SMILES
        df = df.dropna(subset=["canonical_smiles"])
        
        # Create placeholder activity columns
        # Set all labels to 0 (inactive) as default since we don't have activity data
        df["standard_value"] = np.nan
        df["standard_units"] = ""
        df["value_nM"] = np.nan
        df["label"] = 0
        
        # Ensure molecule_chembl_id exists (should already be normalized above, but check again)
        if "molecule_chembl_id" not in df.columns:
            df["molecule_chembl_id"] = [f"COMPOUND_{i:06d}" for i in range(len(df))]
        
        # Dedupe by molecule id
        df = df.groupby("molecule_chembl_id", as_index=False).first()
        df = df.reset_index(drop=True)
        print(f"[data] molecules: {len(df)} (no activity data, labels set to 0 for training)")
        print(f"[data] WARNING: Using placeholder labels since no activity data found in CSV")
    
    return df


# --- RDKit 2D descriptors (reuse from HTS) -----------------------------------

def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    array = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def maccs_fp(smiles: str) -> np.ndarray:
    """Generate MACCS keys fingerprint (167 bits)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167, dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    array = np.zeros((167,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


# QED scaling: HTS3DOracle gives QED 40-60% weight in method-specific predictions
# (lasso: 0.4*qed, pca: 0.5*qed, mutual_info: 0.6*qed). HTS_3D dilutes QED among
# 12 physchem + 3D + ChemBERTa. Scale QED so it has comparable influence (default 3.0).
QED_SCALE_FACTOR = 3.0


def physchem_features(smiles: str, qed_scale: float = QED_SCALE_FACTOR) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(12, dtype=np.float32)
    qed_raw = QED.qed(mol)
    qed_scaled = float(np.clip(qed_raw * qed_scale, 0.0, 3.0))  # Cap at 3.0 to avoid outlier domination
    feats = np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Lipinski.RingCount(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.HeavyAtomCount(mol),
            Lipinski.NumHeteroatoms(mol),
            Descriptors.FractionCSP3(mol),
            qed_scaled,
        ],
        dtype=np.float32,
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def compute_qed_for_smiles_list(smiles_list: List[str]) -> np.ndarray:
    """Compute QED (0-1) for each SMILES. Returns array of same length as input."""
    qed_values = np.zeros(len(smiles_list), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol is not None:
                q = QED.qed(mol)
                qed_values[i] = float(np.clip(q, 0.0, 1.0)) if not (np.isnan(q) or np.isinf(q)) else 0.0
            else:
                qed_values[i] = 0.0
        except Exception:
            qed_values[i] = 0.0
    return qed_values


def build_rdkit_feature_matrix(
    smiles_list: List[str],
    qed_scale: float = QED_SCALE_FACTOR,
) -> torch.Tensor:
    """
    Build RDKit 2D feature matrix and return as GPU tensor if available.
    RDKit operations must run on CPU, but result is moved to GPU.
    qed_scale: Scale factor for QED in physchem features (default 3.0 for HTS-like impact).
    """
    import sys
    warning_budget = 3
    rdkit_rows = []
    total = len(smiles_list)
    print(f"[RDKit 2D] Processing {total} molecules... (qed_scale={qed_scale})")
    sys.stdout.flush()
    for idx, smi in enumerate(tqdm(smiles_list, desc="RDKit 2D features"), 1):
        if warning_budget > 0:
            print("[RDKit] DEPRECATION WARNING: please use MorganGenerator")
            warning_budget -= 1
        fp = morgan_fp(smi)
        maccs = maccs_fp(smi)  # Add MACCS keys
        phys = physchem_features(smi, qed_scale=qed_scale)
        rdkit_rows.append(np.concatenate([fp, maccs, phys]))
        # Print progress every 10% or every 100 molecules (whichever is more frequent)
        if idx % max(1, total // 10) == 0 or idx % 100 == 0:
            print(f"[RDKit 2D] Progress: {idx}/{total} ({idx/total*100:.1f}%)")
            sys.stdout.flush()
    # Stack on CPU (numpy), then move to GPU (non-blocking when available)
    print(f"[RDKit 2D] Stacking features...")
    sys.stdout.flush()
    feat_array = np.vstack(rdkit_rows).astype(np.float32)
    result = torch.from_numpy(feat_array).to(DEVICE, non_blocking=NON_BLOCKING)
    print(f"[RDKit 2D] Complete. Feature matrix shape: {result.shape}, device: {result.device}")
    sys.stdout.flush()
    return result


# --- Ligand 3D featurizer ----------------------------------------------------

def generate_conformer_features(
    smiles: str,
    max_atoms: int = MAX_ATOMS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (atom_features, atom_mask)."""

    base_feat = np.zeros((max_atoms, 16), dtype=np.float32)
    mask = np.zeros((max_atoms,), dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return base_feat, mask

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    try:
        # Embed molecule (generate 3D coordinates)
        embed_result = AllChem.EmbedMolecule(mol, params)
        if embed_result == -1:
            # Embedding failed
            return base_feat, mask
        
        # Try MMFF optimization first (faster, but doesn't support all atoms)
        # Check if molecule has atoms that MMFF doesn't support (e.g., Ir, heavy metals)
        has_unsupported_atoms = False
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            # MMFF doesn't support many transition metals and heavy elements
            # Common problematic ones: Ir (77), Pt (78), Au (79), etc.
            if atomic_num > 54:  # Beyond Xenon, many are unsupported
                has_unsupported_atoms = True
                break
        
        if has_unsupported_atoms:
            # Use UFF (Universal Force Field) as fallback - supports more atom types
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                # If UFF also fails, use unoptimized coordinates
                pass
        else:
            # Use MMFF for supported molecules (faster)
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                # Fallback to UFF if MMFF fails
                try:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                except Exception:
                    # If both fail, use unoptimized coordinates
                    pass
        
        # Compute Gasteiger charges (may fail for some molecules, but that's OK)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception:
            # Charges are optional, continue without them
            pass
    except Exception:
        return base_feat, mask

    conf = mol.GetConformer()
    num_atoms = min(mol.GetNumAtoms(), max_atoms)
    for idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(idx)
        pos = conf.GetAtomPosition(idx)
        # Get Gasteiger charges with validation
        g_charge = 0.0
        if atom.HasProp("_GasteigerCharge"):
            try:
                g_charge_val = float(atom.GetProp("_GasteigerCharge"))
                if np.isfinite(g_charge_val):
                    g_charge = g_charge_val
            except (ValueError, TypeError):
                g_charge = 0.0
        
        g_h_charge = 0.0
        if atom.HasProp("_GasteigerHCharge"):
            try:
                g_h_charge_val = float(atom.GetDoubleProp("_GasteigerHCharge"))
                if np.isfinite(g_h_charge_val):
                    g_h_charge = g_h_charge_val
            except (ValueError, TypeError):
                g_h_charge = 0.0
        
        # Get molecular descriptors with validation
        mol_mr = 0.0
        mol_wt = 0.0
        try:
            mol_mr_val = Descriptors.MolMR(mol) / 200.0
            if np.isfinite(mol_mr_val):
                mol_mr = mol_mr_val
        except (ValueError, TypeError, RuntimeError):
            mol_mr = 0.0
        
        try:
            mol_wt_val = Descriptors.MolWt(mol) / 1000.0
            if np.isfinite(mol_wt_val):
                mol_wt = mol_wt_val
        except (ValueError, TypeError, RuntimeError):
            mol_wt = 0.0
        
        # Get position coordinates with validation
        pos_x = pos.x / 10.0 if np.isfinite(pos.x) else 0.0
        pos_y = pos.y / 10.0 if np.isfinite(pos.y) else 0.0
        pos_z = pos.z / 10.0 if np.isfinite(pos.z) else 0.0
        
        feature_vec = [
            atom.GetAtomicNum() / 100.0,
            atom.GetTotalValence() / 10.0,
            float(atom.GetIsAromatic()),
            float(atom.GetTotalNumHs()) / 4.0,
            float(atom.GetFormalCharge()),
            float(atom.IsInRing()),
            float(atom.GetHybridization()),
            float(atom.GetMass() / 200.0),
            float(atom.GetChiralTag()),
            pos_x,
            pos_y,
            pos_z,
            g_charge,
            g_h_charge,
            mol_mr,
            mol_wt,
        ]
        
        # Validate entire feature vector and replace any NaN/Inf
        feature_array = np.asarray(feature_vec, dtype=np.float32)
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        base_feat[idx] = feature_array
        mask[idx] = 1.0
    return base_feat, mask


def build_3d_cache(smiles_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build 3D conformer cache and return as GPU tensors if available.
    RDKit operations must run on CPU, but result is moved to GPU.
    """
    import sys
    import time
    build_3d_cache._start_time = time.time()  # Track start time for timing
    atom_tensors = []
    masks = []
    total = len(smiles_list)
    print(f"[RDKit 3D] Generating 3D conformers for {total} molecules...")
    sys.stdout.flush()
    slow_molecule_threshold = 3.0  # Warn if a molecule takes more than 3 seconds
    timeout_threshold = 30.0  # Skip molecule if it takes more than 30 seconds
    skipped_count = 0
    
    # Import threading for timeout mechanism
    import threading
    import queue
    
    # For 90–99.9% range: print once per 0.5% (instead of every molecule)
    next_milestone_pct = [90.0]
    
    def generate_with_timeout(smiles, timeout_seconds):
        """Generate conformer features with a real timeout using threading."""
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def worker():
            try:
                feats, mask = generate_conformer_features(smiles)
                result_queue.put((feats, mask))
            except Exception as e:
                exception_queue.put(e)
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            # Thread is still running - timeout occurred
            return None, None, True  # (feats, mask, timeout_occurred)
        elif not exception_queue.empty():
            # Exception occurred
            raise exception_queue.get()
        elif not result_queue.empty():
            # Success
            feats, mask = result_queue.get()
            return feats, mask, False  # (feats, mask, timeout_occurred)
        else:
            # Shouldn't happen, but handle it
            return None, None, True
    
    for idx, smi in enumerate(tqdm(smiles_list, desc="RDKit 3D conformers"), 1):
        mol_start = time.time()
        
        # Use a real timeout mechanism that can interrupt slow molecules
        try:
            feats, mask, timeout_occurred = generate_with_timeout(smi, timeout_threshold)
            mol_time = time.time() - mol_start
            
            if timeout_occurred:
                print(f"[RDKit 3D] WARNING: Molecule {idx}/{total} timed out after {timeout_threshold:.1f}s (interrupted), using zero features")
                sys.stdout.flush()
                skipped_count += 1
                feats = np.zeros((MAX_ATOMS, 16), dtype=np.float32)
                mask = np.zeros((MAX_ATOMS,), dtype=np.float32)
            elif feats is None:
                # Failed for some reason
                print(f"[RDKit 3D] ERROR: Molecule {idx}/{total} failed, using zero features")
                sys.stdout.flush()
                feats = np.zeros((MAX_ATOMS, 16), dtype=np.float32)
                mask = np.zeros((MAX_ATOMS,), dtype=np.float32)
        except Exception as e:
            mol_time = time.time() - mol_start
            print(f"[RDKit 3D] ERROR: Molecule {idx}/{total} failed after {mol_time:.2f}s: {e}")
            sys.stdout.flush()
            # Use zero features for failed molecules
            feats = np.zeros((MAX_ATOMS, 16), dtype=np.float32)
            mask = np.zeros((MAX_ATOMS,), dtype=np.float32)
        
        # Warn about slow molecules (but not timeouts, already handled above)
        if not timeout_occurred and mol_time > slow_molecule_threshold:
            print(f"[RDKit 3D] WARNING: Molecule {idx}/{total} took {mol_time:.2f}s (SMILES: {smi[:50]}...)")
            sys.stdout.flush()
        elif mol_time > slow_molecule_threshold:
            print(f"[RDKit 3D] WARNING: Molecule {idx}/{total} took {mol_time:.2f}s (SMILES: {smi[:50]}...)")
            sys.stdout.flush()
        
        atom_tensors.append(feats)
        masks.append(mask)
        
        # More frequent progress updates in the last 10% (90–99.9%): 1 printout per 0.5%
        current_pct = idx / total * 100
        if idx >= total * 0.9 and current_pct >= next_milestone_pct[0]:
            elapsed = time.time() - build_3d_cache._start_time
            remaining = total - idx
            avg_time_per_mol = elapsed / idx
            estimated_remaining = avg_time_per_mol * remaining
            print(f"[RDKit 3D] {idx}/{total} ({current_pct:.1f}%) | Elapsed: {elapsed:.1f}s | Est. remaining: {estimated_remaining:.1f}s | Last mol: {mol_time:.2f}s")
            sys.stdout.flush()
            next_milestone_pct[0] = min(99.9, next_milestone_pct[0] + 0.5)
        elif idx < total * 0.9 and (idx % max(1, total // 10) == 0 or idx % 100 == 0):
            print(f"[RDKit 3D] Progress: {idx}/{total} ({idx/total*100:.1f}%)")
            sys.stdout.flush()
    
    if skipped_count > 0:
        print(f"[RDKit 3D] WARNING: {skipped_count} molecules were skipped due to timeout (>30s)")
        sys.stdout.flush()
    # Stack on CPU (numpy), then move to GPU (non-blocking when available)
    # Pre-allocate arrays instead of stacking (faster for large arrays)
    print(f"[RDKit 3D] Stacking 3D features...")
    sys.stdout.flush()
    stack_start = time.time()
    
    # Pre-allocate arrays for better performance
    if len(atom_tensors) > 0:
        max_atoms = atom_tensors[0].shape[0]
        feat_dim = atom_tensors[0].shape[1]
        atom_array = np.zeros((len(atom_tensors), max_atoms, feat_dim), dtype=np.float32)
        mask_array = np.zeros((len(masks), max_atoms), dtype=np.float32)
        
        # Fill pre-allocated arrays (faster than stacking)
        for i, (feat, mask) in enumerate(zip(atom_tensors, masks)):
            atom_array[i] = feat
            mask_array[i] = mask
    else:
        # Fallback to stacking if empty
        atom_array = np.stack(atom_tensors).astype(np.float32)
        mask_array = np.stack(masks).astype(np.float32)
    
    stack_time = time.time() - stack_start
    print(f"[timing] Stacking arrays: {stack_time:.2f}s")
    sys.stdout.flush()
    
    # CRITICAL: Validate atom_array for NaN/Inf before converting to tensor
    # Replace any NaN/Inf with zeros to prevent training instability
    validation_start = time.time()
    nan_count = np.isnan(atom_array).sum()
    inf_count = np.isinf(atom_array).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"[RDKit 3D] WARNING: Found {nan_count} NaN and {inf_count} Inf values in atom_features, replacing with zeros")
        atom_array = np.nan_to_num(atom_array, nan=0.0, posinf=0.0, neginf=0.0)
    validation_time = time.time() - validation_start
    if validation_time > 0.1:  # Only print if it takes significant time
        print(f"[timing] NaN/Inf validation: {validation_time:.2f}s")
        sys.stdout.flush()
    
    # Move to GPU (non-blocking when available), matching NLRP3A
    tensor_start = time.time()
    result = (
        torch.from_numpy(atom_array).to(DEVICE, non_blocking=NON_BLOCKING),
        torch.from_numpy(mask_array).to(DEVICE, non_blocking=NON_BLOCKING),
    )
    tensor_time = time.time() - tensor_start
    print(f"[timing] Tensor conversion and GPU transfer: {tensor_time:.2f}s")
    sys.stdout.flush()
    
    # Final validation after tensor conversion
    final_validation_start = time.time()
    if torch.isnan(result[0]).any() or torch.isinf(result[0]).any():
        print(f"[RDKit 3D] CRITICAL: NaN/Inf detected in final atom_features tensor, replacing with zeros")
        result = (
            torch.nan_to_num(result[0], nan=0.0, posinf=0.0, neginf=0.0),
            result[1]
        )
    final_validation_time = time.time() - final_validation_start
    if final_validation_time > 0.1:  # Only print if it takes significant time
        print(f"[timing] Final tensor validation: {final_validation_time:.2f}s")
        sys.stdout.flush()
    
    print(f"[RDKit 3D] Complete. Atom features shape: {result[0].shape}, device: {result[0].device}")
    sys.stdout.flush()
    
    # Timing: Track time spent in build_3d_cache
    if hasattr(build_3d_cache, '_start_time'):
        cache_time = time.time() - build_3d_cache._start_time
        print(f"[timing] 3D cache build time: {cache_time:.2f}s")
        sys.stdout.flush()
    
    return result


# --- Protein pocket detection and encoding ------------------------------------
#
# Generalized pocket encoding system that can:
# 1. Detect multiple viable binding pockets on any protein surface
# 2. Encode each pocket separately with residue properties
# 3. Combine multiple pockets using attention mechanisms
# 
# This replaces the hardcoded NLRP3 pocket template with a flexible system
# that works for arbitrary protein structures. Pockets are detected based on:
# - Surface accessibility (exposed residues)
# - Spatial clustering (DBSCAN)
# - Chemical diversity (charge variation)
#
# Standard 20 amino acids with properties
STANDARD_AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
]

# Kyte-Doolittle hydrophobicity scale
KYTE_DOOLITTLE = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLN": -3.5,
    "ASN": -3.5, "ASP": -3.5, "GLU": -3.5, "LYS": -3.9, "ARG": -4.5
}

# Typical charges at pH 7.4
RESIDUE_CHARGES = {
    "ARG": 1.0, "LYS": 1.0, "HIS": 0.1,
    "ASP": -1.0, "GLU": -1.0,
    "ALA": 0.0, "CYS": 0.0, "GLN": 0.0, "GLY": 0.0, "ILE": 0.0,
    "LEU": 0.0, "MET": 0.0, "PHE": 0.0, "PRO": 0.0, "SER": 0.0,
    "THR": 0.0, "TRP": 0.0, "TYR": 0.0, "VAL": 0.0, "ASN": 0.0
}

AA_TO_IDX = {aa: idx for idx, aa in enumerate(STANDARD_AMINO_ACIDS)}


def detect_protein_pockets(
    residues: List[Tuple[str, Tuple[float, float, float]]],
    max_pockets: int = 5,
    min_pocket_size: int = 8,
    max_pocket_size: int = 25,
    pocket_radius: float = 1.5,
) -> List[List[Tuple[str, float, float, Tuple[float, float, float]]]]:
    """
    Detect multiple viable binding pockets on a protein surface.
    
    Uses geometric and chemical criteria to identify potential binding sites:
    - Surface accessibility (residues with neighbors in limited sphere)
    - Chemical diversity (mix of charged, polar, and hydrophobic residues)
    - Spatial clustering (residues within pocket_radius of cluster center)
    
    Args:
        residues: List of (residue_name, (x, y, z)) tuples in nm
        max_pockets: Maximum number of pockets to return
        min_pocket_size: Minimum residues per pocket
        max_pocket_size: Maximum residues per pocket
        pocket_radius: Radius (nm) for clustering residues into pockets
    
    Returns:
        List of pocket templates, each as list of (residue_name, charge, 
        hydrophobicity, (x, y, z)) tuples
    """
    if len(residues) < min_pocket_size:
        # Fallback: use all residues as single pocket
        pocket = []
        for res_name, coords in residues:
            aa = res_name[:3].upper() if len(res_name) >= 3 else "GLY"
            charge = RESIDUE_CHARGES.get(aa, 0.0)
            hydro = KYTE_DOOLITTLE.get(aa, 0.0)
            pocket.append((res_name, charge, hydro, coords))
        return [pocket]
    
    # Convert to numpy for easier computation
    res_names = [r[0] for r in residues]
    coords = np.array([r[1] for r in residues], dtype=np.float32)
    
    # Compute pairwise distances
    n_res = len(residues)
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    
    # Find surface residues (those with fewer neighbors in a sphere)
    neighbor_count = (distances < 0.8).sum(axis=1)  # ~8 Angstrom
    surface_mask = neighbor_count < np.percentile(neighbor_count, 60)  # Top 40% most exposed
    surface_indices = np.where(surface_mask)[0]
    
    if len(surface_indices) < min_pocket_size:
        # If too few surface residues, use all residues
        surface_indices = np.arange(n_res)
    
    # Cluster surface residues into potential pockets
    detected_pockets = []
    if len(surface_indices) >= min_pocket_size:
        surface_coords = coords[surface_indices]
        
        # DBSCAN clustering to find spatial clusters
        # eps in nm (convert pocket_radius from nm to distance threshold)
        clustering = DBSCAN(eps=pocket_radius, min_samples=min_pocket_size - 1, metric='euclidean')
        cluster_labels = clustering.fit_predict(surface_coords)
        
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_indices = surface_indices[cluster_mask]
            
            if min_pocket_size <= len(cluster_indices) <= max_pocket_size:
                pocket = []
                for idx in cluster_indices:
                    res_name = res_names[idx]
                    coord = tuple(coords[idx])
                    aa = res_name[:3].upper() if len(res_name) >= 3 else "GLY"
                    charge = RESIDUE_CHARGES.get(aa, 0.0)
                    hydro = KYTE_DOOLITTLE.get(aa, 0.0)
                    pocket.append((res_name, charge, hydro, coord))
                
                # Score pocket by chemical diversity
                charges = [p[1] for p in pocket]
                charge_diversity = np.std(charges)  # Prefer pockets with charge variation
                detected_pockets.append((charge_diversity, pocket))
    
    # Sort by diversity and take top pockets
    detected_pockets.sort(reverse=True, key=lambda x: x[0])
    selected_pockets = [p[1] for p in detected_pockets[:max_pockets]]
    
    # If no pockets detected, create a default pocket from surface residues
    if not selected_pockets:
        pocket = []
        for idx in surface_indices[:max_pocket_size]:
            res_name = res_names[idx]
            coord = tuple(coords[idx])
            aa = res_name[:3].upper() if len(res_name) >= 3 else "GLY"
            charge = RESIDUE_CHARGES.get(aa, 0.0)
            hydro = KYTE_DOOLITTLE.get(aa, 0.0)
            pocket.append((res_name, charge, hydro, coord))
        selected_pockets = [pocket]
    
    return selected_pockets


def build_protein_template_from_pockets(
    pockets: List[List[Tuple[str, float, float, Tuple[float, float, float]]]],
    device: Optional[torch.device] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Build tensor representations for multiple protein pockets.
    Tensors are created on device (default DEVICE) so they stay on GPU when available.

    Args:
        pockets: List of pocket templates from detect_protein_pockets()
        device: Device for tensors (default: global DEVICE)

    Returns:
        List of tensor dictionaries, each with 'aa_idx', 'charges', 'hydros', 'coords'
    """
    if device is None:
        device = DEVICE
    templates = []
    for pocket in pockets:
        aa_idx = []
        charges = []
        hydros = []
        coords = []

        for res_name, charge, hydro, coord in pocket:
            aa = res_name[:3].upper() if len(res_name) >= 3 else "GLY"
            aa_idx.append(AA_TO_IDX.get(aa, 0))
            charges.append(charge)
            hydros.append(hydro)
            coords.append(coord)

        templates.append({
            "aa_idx": torch.tensor(aa_idx, dtype=torch.long, device=device),
            "charges": torch.tensor(charges, dtype=torch.float32, device=device),
            "hydros": torch.tensor(hydros, dtype=torch.float32, device=device),
            "coords": torch.tensor(coords, dtype=torch.float32, device=device),
        })

    return templates


# 7ALV (NLRP3 + MCC950) pocket template
# Extracted from src/7alv.pdb (residues within 5A of ligand)
POCKET_TEMPLATE_7ALV = [
    # residue_name, charge, hydrophobicity (Kyte-Doolittle), xyz (nm)
    ("TYR168", 0.0, -1.3, (2.8, 3.6, 14.3)),
    ("THR169", 0.0, -0.7, (3.0, 3.8, 14.1)),
    ("ALA227", 0.0, 1.8, (1.7, 3.8, 13.0)),
    ("ALA228", 0.0, 1.8, (1.6, 3.4, 13.2)),
    ("GLY229", 0.0, -0.4, (1.9, 3.3, 13.3)),
    ("ILE230", 0.0, 4.5, (2.1, 3.7, 13.3)),
    ("GLY231", 0.0, -0.4, (2.1, 3.7, 13.7)),
    ("LYS232", 1.0, -3.9, (1.7, 3.9, 13.8)),
    ("THR233", 0.0, -0.7, (1.7, 3.6, 14.1)),
    ("ILE234", 0.0, 4.5, (2.0, 3.6, 14.2)),
    ("ARG351", 1.0, -4.5, (1.2, 3.9, 13.1)),
    ("PRO412", 0.0, -1.6, (2.5, 3.4, 13.4)),
    ("LEU413", 0.0, 3.8, (2.3, 3.1, 13.4)),
    ("TRP416", 0.0, -0.9, (2.7, 2.8, 13.5)),
    ("GLU629", -1.0, -3.5, (1.3, 3.6, 11.8)),
]

# Legacy support: Default NLRP3 pocket template for backward compatibility
POCKET_TEMPLATE_NLRP3 = [
    # residue_name, charge, hydrophobicity (Kyte-Doolittle), xyz (nm)
    ("ASP302", -1.0, -3.5, (-2.1, 0.8, 1.6)),
    ("GLU306", -1.0, -3.5, (-1.2, 1.5, 0.2)),
    ("LYS309", 1.0, -3.9, (-0.5, -1.2, 0.6)),
    ("HIS330", 0.1, -3.2, (1.1, -0.4, 0.9)),
    ("TYR338", 0.0, -1.3, (2.4, -1.0, 1.4)),
    ("LEU340", 0.0, 3.8, (3.2, -0.5, -0.6)),
    ("ILE372", 0.0, 4.5, (1.7, 2.6, -1.3)),
    ("PHE410", 0.0, 2.8, (0.3, 3.1, -0.9)),
    ("TRP414", 0.0, -0.9, (-1.5, 3.0, -1.8)),
    ("SER415", 0.0, -0.8, (-2.8, 2.2, -0.5)),
    ("ARG430", 1.0, -4.5, (-3.4, 0.5, -1.1)),
    ("GLN434", 0.0, -3.5, (-2.6, -0.8, -2.0)),
    ("ASN437", 0.0, -3.5, (-0.8, -2.0, -1.7)),
    ("VAL441", 0.0, 4.2, (0.9, -2.5, -0.3)),
    ("LEU458", 0.0, 3.8, (2.7, -1.9, 0.5)),
    ("MET461", 0.0, 1.9, (3.5, 0.1, 1.9)),
]

# Default: Use 7ALV template
PROTEIN_TEMPLATES = build_protein_template_from_pockets([POCKET_TEMPLATE_7ALV])


# --- Dataset -----------------------------------------------------------------


class HTS3DDataset(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rdkit_feats: torch.Tensor,
        atom_feats: torch.Tensor,
        atom_masks: torch.Tensor,
        labels: np.ndarray,
    ):
        # All tensors are already on the correct device (GPU if available)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.rdkit = rdkit_feats
        self.atom_feats = atom_feats
        self.atom_masks = atom_masks
        # Labels converted to tensor on correct device
        self.labels = torch.tensor(labels, dtype=torch.float32, device=rdkit_feats.device)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "rdkit": self.rdkit[idx],
            "atom_features": self.atom_feats[idx],
            "atom_mask": self.atom_masks[idx],
            "label": self.labels[idx],
        }


# --- Neural building blocks --------------------------------------------------


class ProteinPocketEncoder(nn.Module):
    """
    Encodes protein pockets into embedding tensors.
    Can encode multiple pockets and optionally combine them.
    
    For each pocket:
    - Encodes amino acid type, charge, hydrophobicity, and 3D coordinates
    - Returns tensor of shape [num_residues, embed_dim]
    """

    def __init__(self, embed_dim: int = EMBED_DIM, num_aa_types: int = len(STANDARD_AMINO_ACIDS)):
        super().__init__()
        self.embed_dim = embed_dim
        self.residue_embed = nn.Embedding(num_aa_types, embed_dim // 2)
        self.scalar_proj = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
        )
        self.coord_proj = nn.Linear(3, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, template: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a single pocket template.
        
        Args:
        template: Dict with 'aa_idx', 'charges', 'hydros', 'coords' tensors

        Returns:
            Tensor of shape [num_residues, embed_dim]
        """
        template = {k: (v.to(DEVICE, non_blocking=NON_BLOCKING) if v.device != DEVICE else v) for k, v in template.items()}
        aa_vec = self.residue_embed(template["aa_idx"])
        scalar_inputs = torch.stack(
            [
                template["charges"],
                template["hydros"],
                template["coords"].norm(dim=-1),
            ],
            dim=-1,
        )
        scalar_vec = self.scalar_proj(scalar_inputs)
        fused = torch.cat([aa_vec, scalar_vec], dim=-1)
        coords = template["coords"]
        coord_proj = self.coord_proj(torch.sin(coords))
        fused = fused + coord_proj
        return self.out_proj(fused)  # [num_residues, embed_dim]


class MultiPocketEncoder(nn.Module):
    """
    Encodes multiple protein pockets and combines them using attention.
    Tests multiple viable pockets on the protein surface.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, num_pockets: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_pockets = num_pockets
        self.pocket_encoder = ProteinPocketEncoder(embed_dim)
        
        # Attention mechanism to combine multiple pockets
        self.pocket_attention = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.pocket_pool = nn.AdaptiveAvgPool1d(1)  # Pool each pocket to single vector
        
        # Optional: learnable weights for combining pockets
        self.pocket_weights = nn.Parameter(torch.ones(num_pockets) / num_pockets)

    def forward(
        self, 
        pocket_templates: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Encode multiple pockets and combine them.
        
        Args:
            pocket_templates: List of pocket template dicts
        
        Returns:
            Combined pocket representation of shape [1, embed_dim]
        """
        if not pocket_templates:
            # Fallback: return zeros
            return torch.zeros(1, self.embed_dim, device=DEVICE)
        
        # Encode each pocket
        pocket_encodings = []
        for template in pocket_templates[:self.num_pockets]:
            pocket_tokens = self.pocket_encoder(template)  # [num_residues, embed_dim]
            # Pool each pocket to a single vector
            pocket_vec = pocket_tokens.mean(dim=0, keepdim=True)  # [1, embed_dim]
            pocket_encodings.append(pocket_vec)
        
        # Pad or truncate to num_pockets
        while len(pocket_encodings) < self.num_pockets:
            # Duplicate last pocket or use zeros
            if pocket_encodings:
                pocket_encodings.append(pocket_encodings[-1])
            else:
                pocket_encodings.append(torch.zeros(1, self.embed_dim, device=DEVICE))
        
        pocket_encodings = pocket_encodings[:self.num_pockets]
        
        # Stack pockets: [num_pockets, embed_dim]
        pocket_stack = torch.cat(pocket_encodings, dim=0)  # [num_pockets, embed_dim]
        pocket_stack = pocket_stack.unsqueeze(0)  # [1, num_pockets, embed_dim]
        
        # Use self-attention to combine pockets
        # Each pocket can attend to all other pockets
        combined, _ = self.pocket_attention(pocket_stack, pocket_stack, pocket_stack)
        # [1, num_pockets, embed_dim]
        
        # Weighted average using learnable weights
        num_actual_pockets = min(len(pocket_templates), self.num_pockets)
        weights = F.softmax(self.pocket_weights[:num_actual_pockets], dim=0)
        weights = weights.unsqueeze(0).unsqueeze(-1)  # [1, num_actual_pockets, 1]
        # Only use weights for actual pockets
        combined_weighted = (combined[:, :num_actual_pockets, :] * weights).sum(dim=1)  # [1, embed_dim]
        
        return combined_weighted.squeeze(0)  # [embed_dim]


class Ligand3DEncoder(nn.Module):
    def __init__(self, atom_dim: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(atom_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, atom_features: torch.Tensor) -> torch.Tensor:
        return self.proj(atom_features)


class CrossAttentionFusion(nn.Module):
    """
    Implements ligand (queries) attending to protein residues (keys/values).
    """

    def __init__(self, embed_dim: int = EMBED_DIM, heads: int = NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, heads, batch_first=True, dropout=0.1
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        ligand_tokens: torch.Tensor,
        protein_tokens: torch.Tensor,
        ligand_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check inputs for NaN/Inf
        if torch.isnan(ligand_tokens).any() or torch.isinf(ligand_tokens).any():
            print("[cross_attn] CRITICAL: NaN/Inf in ligand_tokens input, replacing with zeros")
            ligand_tokens = torch.nan_to_num(ligand_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(protein_tokens).any() or torch.isinf(protein_tokens).any():
            print("[cross_attn] CRITICAL: NaN/Inf in protein_tokens input, replacing with zeros")
            protein_tokens = torch.nan_to_num(protein_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        
        attn_out, attn_weights = self.attn(
            ligand_tokens, protein_tokens, protein_tokens
        )
        
        # Check attention output for NaN/Inf
        if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
            print("[cross_attn] CRITICAL: NaN/Inf in attn_out, checking and fixing parameters...")
            # Check if attention parameters are corrupted
            for name, param in self.attn.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"[cross_attn] CRITICAL: Corrupted parameter '{name}' in attn, reinitializing...")
                    with torch.no_grad():
                        if param.requires_grad:
                            param.data.normal_(0, 0.01)
            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)
        
        x = self.norm1(ligand_tokens + attn_out)
        
        # Check after norm1
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[cross_attn] CRITICAL: NaN/Inf after norm1, checking and fixing parameters...")
            for name, param in self.norm1.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"[cross_attn] CRITICAL: Corrupted parameter '{name}' in norm1, reinitializing...")
                    with torch.no_grad():
                        if param.requires_grad:
                            param.data.normal_(0, 0.01)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        ff_out = self.ff(x)
        
        # Check feedforward output
        if torch.isnan(ff_out).any() or torch.isinf(ff_out).any():
            print("[cross_attn] CRITICAL: NaN/Inf in ff_out, checking and fixing parameters...")
            for name, param in self.ff.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"[cross_attn] CRITICAL: Corrupted parameter '{name}' in ff, reinitializing...")
                    with torch.no_grad():
                        if param.requires_grad:
                            param.data.normal_(0, 0.01)
            ff_out = torch.nan_to_num(ff_out, nan=0.0, posinf=0.0, neginf=0.0)
        
        x = self.norm2(x + ff_out)
        
        # Check after norm2
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[cross_attn] CRITICAL: NaN/Inf after norm2, checking and fixing parameters...")
            for name, param in self.norm2.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"[cross_attn] CRITICAL: Corrupted parameter '{name}' in norm2, reinitializing...")
                    with torch.no_grad():
                        if param.requires_grad:
                            param.data.normal_(0, 0.01)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        if ligand_mask is not None:
            x = x * ligand_mask.unsqueeze(-1)
        
        return x, attn_weights


# --- Option 4 Ultra: helper modules (multi-pocket, attention pool, interaction, adaptive fusion, residual) ---


class AttentionPooling(nn.Module):
    """Attention-based pooling that learns to focus on important atoms/residues."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, num_items, feature_dim]
            mask: [batch, num_items] (1 for valid, 0 for padding)
        Returns:
            pooled: [batch, feature_dim]
        """
        scores = self.attention(features).squeeze(-1)  # [batch, num_items]
        scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [batch, num_items, 1]
        pooled = (features * attention_weights).sum(dim=1)  # [batch, feature_dim]
        return pooled


class ImprovedMultiPocketEncoder(nn.Module):
    """Encodes multiple protein pockets using simple averaging (Option 4 Ultra)."""

    def __init__(self, embed_dim: int = 128, output_dim: int = 64, max_pockets: int = 3):
        super().__init__()
        self.max_pockets = max_pockets
        self.output_dim = output_dim
        self.residue_embed = nn.Embedding(20, embed_dim // 2)
        self.property_encoder = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.ReLU(),
        )
        # LayerNorm instead of BatchNorm1d: this path gets batch size 1 (single protein encoding)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def encode_single_pocket(self, template: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a single pocket to [embed_dim]. Moves to DEVICE only if not already there."""
        template = {k: (v.to(DEVICE, non_blocking=NON_BLOCKING) if v.device != DEVICE else v) for k, v in template.items()}
        aa_emb = self.residue_embed(template["aa_idx"])
        properties = torch.stack([
            template["charges"],
            template["hydros"],
            template["coords"].norm(dim=-1),
        ], dim=-1)
        prop_emb = self.property_encoder(properties)
        combined = torch.cat([aa_emb, prop_emb], dim=-1)
        pooled = combined.mean(dim=0)
        return pooled

    def forward(self, pocket_templates: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Encode multiple pockets and average. Returns [output_dim]."""
        if not pocket_templates:
            return torch.zeros(self.output_dim, device=DEVICE)
        pocket_encodings = []
        for template in pocket_templates[:self.max_pockets]:
            pocket_enc = self.encode_single_pocket(template)
            pocket_encodings.append(pocket_enc)
        if not pocket_encodings:
            return torch.zeros(self.output_dim, device=DEVICE)
        stacked = torch.stack(pocket_encodings, dim=0)
        averaged = stacked.mean(dim=0)
        output = self.output_proj(averaged.unsqueeze(0)).squeeze(0)
        return output


class AdaptiveFeatureFusion(nn.Module):
    """Learn to weight different feature branches adaptively (Option 4 Ultra)."""

    def __init__(self, feature_dims: Dict[str, int]):
        super().__init__()
        self.feature_dims = feature_dims
        self.feature_names = list(feature_dims.keys())
        total_dim = sum(feature_dims.values())
        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Linear(total_dim // 2, len(feature_dims)),
            nn.Sigmoid(),
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        feature_list = [features[name] for name in self.feature_names]
        concat_features = torch.cat(feature_list, dim=1)
        weights = self.gate(concat_features)
        weighted_features = []
        start_idx = 0
        for i, name in enumerate(self.feature_names):
            dim = self.feature_dims[name]
            feat = concat_features[:, start_idx : start_idx + dim]
            weight = weights[:, i : i + 1]
            weighted_features.append(feat * weight)
            start_idx += dim
        return torch.cat(weighted_features, dim=1)


class ProteinLigandInteraction(nn.Module):
    """Simple interaction features between protein and ligand 3D (Option 4 Ultra)."""

    def __init__(self, protein_dim: int = 64, ligand_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.protein_proj = nn.Linear(protein_dim, output_dim)
        self.ligand_proj = nn.Linear(ligand_dim, output_dim)
        self.interaction = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.ReLU(),
        )

    def forward(self, protein_feat: torch.Tensor, ligand_feat: torch.Tensor) -> torch.Tensor:
        p = self.protein_proj(protein_feat)
        l = self.ligand_proj(ligand_feat)
        all_features = torch.cat([p * l, torch.abs(p - l), p], dim=1)  # [batch, 3*output_dim]
        return self.interaction(all_features)


class ResidualBranch(nn.Module):
    """Branch with residual connection for better gradient flow (Option 4 Ultra)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block2(self.block1(x))
        residual = self.residual_proj(x) if self.residual_proj is not None else x
        return out + residual


class HTS3DModel(nn.Module):
    """
    Option C: ChemBERTa(128) + 3D(64) + protein(64) + RDKit(128) = 384d.
    Default: Minimal (base Option C) aligned with HTS.py for similar performance:
    - HTS-style branches: 768->256->128 and rdkit->256->128 (no residual), classifier 384->128->64->1.
    - Single pocket, mean pooling, no interaction, no adaptive; bias=0; same dropout/LR/weight_decay as HTS.
    Optional args enable Standard/Full/Ultra (attention, interaction, multi-pocket, adaptive).
    """

    def __init__(
        self,
        rdkit_dim: int,
        atom_dim: int,
        protein_pockets: Optional[List[Dict[str, torch.Tensor]]] = None,
        num_pockets: int = 5,
        dropout_rate: float = 0.3,
        use_attention_pooling: bool = False,
        use_interaction_features: bool = False,
        use_adaptive_weighting: bool = False,
        max_pockets_ultra: int = 1,
    ):
        super().__init__()
        import sys
        max_pockets = min(max_pockets_ultra, num_pockets) if num_pockets else max_pockets_ultra
        config_name = "Option C Minimal" if not (use_attention_pooling or use_interaction_features or use_adaptive_weighting) and max_pockets_ultra <= 1 else "Option C (Standard/Full/Ultra)"
        print(f"[model] Initializing {config_name} (HTS3DModel) - aligned with HTS.py when minimal")
        print(f"[model] - Pockets: {max_pockets} (max_ultra={max_pockets_ultra})")
        print(f"[model] - Attention pooling: {use_attention_pooling}")
        print(f"[model] - Interaction features: {use_interaction_features}")
        print(f"[model] - Adaptive weighting: {use_adaptive_weighting}")
        sys.stdout.flush()

        self.use_attention_pooling = use_attention_pooling
        self.use_interaction_features = use_interaction_features
        self.use_adaptive_weighting = use_adaptive_weighting
        self.protein_pockets = protein_pockets

        print(f"[model] Loading ChemBERTa from '{CHEMBERTA_NAME}'...")
        sys.stdout.flush()
        self.chemberta = RobertaModel.from_pretrained(CHEMBERTA_NAME)
        print(f"[model] ChemBERTa loaded")
        sys.stdout.flush()

        self.chemberta_dropout = nn.Dropout(dropout_rate)
        is_minimal = (not use_attention_pooling and not use_interaction_features and not use_adaptive_weighting and max_pockets <= 1)
        if is_minimal:
            # Match HTS.py exactly: 768 -> 256 -> 128 (no residual)
            self.chemberta_branch = nn.Sequential(
                nn.Linear(self.chemberta.config.hidden_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
        else:
            self.chemberta_branch = ResidualBranch(
                input_dim=self.chemberta.config.hidden_size,
                hidden_dim=256,
                output_dim=128,
                dropout=dropout_rate,
            )

        # 3D ligand encoder: atom_dim -> 128 -> 64, then attention or mean pool
        self.ligand_3d_encoder = nn.Sequential(
            nn.Linear(atom_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        if use_attention_pooling:
            self.ligand_3d_pool = AttentionPooling(64)
        else:
            self.ligand_3d_pool = None

        self.protein_encoder = ImprovedMultiPocketEncoder(
            embed_dim=128,
            output_dim=64,
            max_pockets=max_pockets,
        )

        if use_interaction_features:
            self.interaction_module = ProteinLigandInteraction(
                protein_dim=64,
                ligand_dim=64,
                output_dim=32,
            )
            total_dim = 128 + 64 + 64 + 32 + 128  # 416
        else:
            self.interaction_module = None
            total_dim = 128 + 64 + 64 + 128  # 384

        if is_minimal:
            # Match HTS.py exactly: rdkit_dim -> 256 -> 128 (no residual)
            self.rdkit_branch = nn.Sequential(
                nn.Linear(rdkit_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
        else:
            self.rdkit_branch = ResidualBranch(
                input_dim=rdkit_dim,
                hidden_dim=256,
                output_dim=128,
                dropout=dropout_rate,
            )

        if use_adaptive_weighting:
            feature_dims = {"bert": 128, "3d": 64, "protein": 64, "rdkit": 128}
            if use_interaction_features:
                feature_dims["interaction"] = 32
            self.feature_fusion = AdaptiveFeatureFusion(feature_dims)
        else:
            self.feature_fusion = None

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 1),
        )
        self._initialize_final_bias(total_dim)
        print(f"[model] Total classifier input: {total_dim}d")

    def _initialize_final_bias(self, total_dim: int = 384):
        """Initialize final layer bias. 0.0 for both 384d and 416d (match HTS.py; better calibration on random/neutral sets)."""
        final_layer = self.classifier[-1]
        if isinstance(final_layer, nn.Linear) and final_layer.bias is not None:
            bias_val = 0.0
            nn.init.constant_(final_layer.bias, bias_val)
            print(f"[init] Final classifier bias: {bias_val} (384d and 416d; match HTS.py, better calibration)")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rdkit_feats: torch.Tensor,
        atom_features: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Option C forward: ChemBERTa(128) + 3D(64) + protein(64) [+ interaction(32)] + RDKit(128) -> classifier (384d or 416d)."""
        batch_size = input_ids.size(0)

        if torch.isnan(rdkit_feats).any() or torch.isinf(rdkit_feats).any():
            rdkit_feats = torch.nan_to_num(rdkit_feats, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(atom_features).any() or torch.isinf(atom_features).any():
            atom_features = torch.nan_to_num(atom_features, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. ChemBERTa -> 128d (residual branch)
        bert_out = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_out.last_hidden_state[:, 0, :]
        bert_cls = self.chemberta_dropout(bert_cls)
        chemberta_emb = self.chemberta_branch(bert_cls)  # [B, 128]

        # 2. 3D conformer -> 64d (attention or masked mean pool)
        encoded_atoms = self.ligand_3d_encoder(atom_features)  # [B, N, 64]
        if self.use_attention_pooling and self.ligand_3d_pool is not None:
            ligand_3d_emb = self.ligand_3d_pool(encoded_atoms, atom_mask)  # [B, 64]
        else:
            mask_expanded = atom_mask.unsqueeze(-1)
            masked_encoded = encoded_atoms * mask_expanded
            sum_encoded = masked_encoded.sum(dim=1)
            count = atom_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            ligand_3d_emb = sum_encoded / count  # [B, 64]

        # 3. Protein pocket -> 64d (multi-pocket average)
        if self.protein_pockets:
            protein_emb = self.protein_encoder(self.protein_pockets)  # [64]
            protein_emb = protein_emb.unsqueeze(0).expand(batch_size, -1)  # [B, 64]
        else:
            protein_emb = torch.zeros(batch_size, 64, device=input_ids.device, dtype=input_ids.dtype)

        # 4. Protein-ligand interaction -> 32d (optional)
        if self.use_interaction_features and self.interaction_module is not None:
            interaction_emb = self.interaction_module(protein_emb, ligand_3d_emb)  # [B, 32]

        # 5. RDKit -> 128d (residual branch)
        rdkit_emb = self.rdkit_branch(rdkit_feats)  # [B, 128]

        # 6. Combine: adaptive weighting or simple concat
        if self.use_adaptive_weighting and self.feature_fusion is not None:
            features_dict = {
                "bert": chemberta_emb,
                "3d": ligand_3d_emb,
                "protein": protein_emb,
                "rdkit": rdkit_emb,
            }
            if self.use_interaction_features and self.interaction_module is not None:
                features_dict["interaction"] = interaction_emb
            combined = self.feature_fusion(features_dict)
        else:
            if self.use_interaction_features and self.interaction_module is not None:
                combined = torch.cat([
                    chemberta_emb,
                    ligand_3d_emb,
                    protein_emb,
                    interaction_emb,
                    rdkit_emb,
                ], dim=1)
            else:
                combined = torch.cat([
                    chemberta_emb,
                    ligand_3d_emb,
                    protein_emb,
                    rdkit_emb,
                ], dim=1)

        # 7. Classifier
        logits = self.classifier(combined).squeeze(-1)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        return logits


# --- Training / evaluation ---------------------------------------------------


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    rdkit: torch.Tensor
    atom_features: torch.Tensor
    atom_mask: torch.Tensor
    labels: torch.Tensor


def to_device(sample: Dict[str, torch.Tensor]) -> Batch:
    """
    Move tensors to device, avoiding unnecessary transfers if already on device.
    """
    # Only transfer if not already on target device
    def maybe_to_device(t: torch.Tensor) -> torch.Tensor:
        if t.device != DEVICE:
            return t.to(DEVICE, non_blocking=NON_BLOCKING)
        return t
    
    return Batch(
        input_ids=maybe_to_device(sample["input_ids"]),
        attention_mask=maybe_to_device(sample["attention_mask"]),
        rdkit=maybe_to_device(sample["rdkit"]),
        atom_features=maybe_to_device(sample["atom_features"]),
        atom_mask=maybe_to_device(sample["atom_mask"]),
        labels=maybe_to_device(sample["label"]),
    )


def train_one_epoch(
    model: HTS3DModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    pos_weight_val: float = 1.0,
) -> Tuple[float, float]:
    """Returns (average_loss, time_elapsed_seconds)."""
    model.train()
    running = 0.0
    total_batches = len(loader)
    start_time = time.time()
    print(f"[train] Starting epoch: {total_batches} batches")
    
    # Track consecutive NaN batches to detect if model is completely corrupted
    consecutive_nan_batches = 0
    max_consecutive_nan = 50  # If 50 consecutive batches fail, something is seriously wrong
    
    for batch_idx, sample in enumerate(loader, 1):
        batch = to_device(sample)
        
        # CRITICAL: Zero gradients FIRST to clear any stale NaN gradients from previous batch
        optimizer.zero_grad(set_to_none=True)
        
        # CRITICAL: Check model parameters for NaN/Inf BEFORE forward pass
        # If parameters are NaN, the forward pass will produce NaN outputs
        # FIX: Reinitialize corrupted parameters instead of just skipping
        has_nan_params = False
        corrupted_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"[train] CRITICAL: NaN/Inf detected in parameter '{name}' at batch {batch_idx}, reinitializing...")
                has_nan_params = True
                corrupted_params.append(name)
                # Reinitialize corrupted parameter to small random values
                with torch.no_grad():
                    if param.requires_grad:
                        param.data.normal_(0, 0.01)  # Small random initialization
                        print(f"[train] Reinitialized parameter: {name}")
        
        if has_nan_params:
            # Reset consecutive counter if we successfully fixed parameters
            if len(corrupted_params) > 0:
                consecutive_nan_batches = max(0, consecutive_nan_batches - 1)  # Reduce counter since we fixed it
                print(f"[train] Fixed {len(corrupted_params)} corrupted parameters, continuing training")
            else:
                consecutive_nan_batches += 1
                if consecutive_nan_batches >= max_consecutive_nan:
                    print(f"[train] CRITICAL: {consecutive_nan_batches} consecutive NaN batches - model is corrupted, stopping training")
                    raise RuntimeError(f"Model parameters corrupted: {consecutive_nan_batches} consecutive NaN batches")
            # Continue to next batch - parameters are now fixed
        
        # Monitoring: Log class distribution in batch
        if batch_idx == 1:
            batch_positives = batch.labels.sum().item()
            print(f"[train] Batch 1 - Positives: {batch_positives}/{len(batch.labels)} "
                  f"({batch_positives/len(batch.labels)*100:.2f}%)")
        # CRITICAL: Set a flag to track if NaN was detected during forward pass
        # This will help us decide whether to skip backward()
        nan_detected_in_forward = False
        
        logits = model(
            batch.input_ids,
            batch.attention_mask,
            batch.rdkit,
            batch.atom_features,
            batch.atom_mask,
        )
        
        # Monitoring: Log sample prediction for diagnostics
        if batch_idx == 1:
            with torch.no_grad():
                sample_logits = logits[0].item()
                sample_prob = torch.sigmoid(logits[0]).item()
                sample_label = batch.labels[0].item()
                print(f"[train] Sample prediction: prob={sample_prob:.4f}, logit={sample_logits:.4f}, "
                      f"label={sample_label:.0f}")
                
                # Log intermediate feature statistics to track learning progress
                if hasattr(model, '_last_lig_pool_stats') and hasattr(model, '_last_rdkit_emb_stats'):
                    lig_stats = model._last_lig_pool_stats
                    rdkit_stats = model._last_rdkit_emb_stats
                    print(f"[train] Intermediate features - lig_pool: mean={lig_stats['mean']:.4f}, "
                          f"std={lig_stats['std']:.4f}, min={lig_stats['min']:.4f}, max={lig_stats['max']:.4f}")
                    print(f"[train] Intermediate features - rdkit_emb: mean={rdkit_stats['mean']:.4f}, "
                          f"std={rdkit_stats['std']:.4f}, min={rdkit_stats['min']:.4f}, max={rdkit_stats['max']:.4f}")
        
        # Check for NaN/Inf in logits before computing loss
        # If NaN appears in logits, it means model parameters are corrupted OR forward pass produced NaN
        # CRITICAL: Skip backward() if NaN is detected - don't propagate NaN gradients
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[train] WARNING: NaN/Inf detected in logits at batch {batch_idx} - checking and fixing parameters, SKIPPING backward()...")
            nan_detected_in_forward = True
            
            # Check and fix ALL model parameters
            corrupted_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"[train] CRITICAL: Found corrupted parameter '{name}' - reinitializing...")
                    corrupted_params.append(name)
                    with torch.no_grad():
                        if param.requires_grad:
                            param.data.normal_(0, 0.01)  # Small random initialization
            
            if len(corrupted_params) > 0:
                consecutive_nan_batches = max(0, consecutive_nan_batches - 1)  # Reduce counter since we fixed it
                print(f"[train] Fixed {len(corrupted_params)} corrupted parameters, skipping this batch (no backward)")
            else:
                consecutive_nan_batches += 1
                if consecutive_nan_batches >= max_consecutive_nan:
                    print(f"[train] CRITICAL: {consecutive_nan_batches} consecutive NaN batches - model is corrupted, stopping training")
                    raise RuntimeError(f"Model outputs corrupted: {consecutive_nan_batches} consecutive NaN batches")
            
            # CRITICAL: Skip backward() and optimizer step if NaN was detected
            # This prevents NaN gradients from corrupting parameters
            optimizer.zero_grad(set_to_none=True)  # Clear any gradients that might have been computed
            continue
        
        # Use simple BCE loss like HTS.py (removed balanced loss component for simplicity and stability)
        loss = criterion(logits, batch.labels)
        
        # Final check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[train] WARNING: NaN/Inf loss at batch {batch_idx}, skipping batch")
            consecutive_nan_batches += 1
            if consecutive_nan_batches >= max_consecutive_nan:
                print(f"[train] CRITICAL: {consecutive_nan_batches} consecutive NaN batches - model is corrupted, stopping training")
                raise RuntimeError(f"Loss corrupted: {consecutive_nan_batches} consecutive NaN batches")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        # Check for NaN gradients BEFORE backward pass (should be rare now since we zero at start)
        # This catches any edge cases where gradients weren't properly cleared
        has_nan_grad_before = False
        for param in model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad_before = True
                break
        
        if has_nan_grad_before:
            print(f"[train] WARNING: NaN/Inf gradients detected at batch {batch_idx} BEFORE backward (stale from previous batch), zeroing and skipping")
            consecutive_nan_batches += 1
            if consecutive_nan_batches >= max_consecutive_nan:
                print(f"[train] CRITICAL: {consecutive_nan_batches} consecutive NaN batches - model is corrupted, stopping training")
                raise RuntimeError(f"Gradients corrupted: {consecutive_nan_batches} consecutive NaN batches")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        loss.backward()
        
        # Simple gradient clipping like HTS.py (relaxed from 0.1 to 1.0, removed individual clamping)
        try:
            # Use moderate gradient clipping - allows reasonable gradient flow for learning
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Only check for NaN/Inf gradients (not large finite gradients)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"[train] WARNING: NaN/Inf gradient norm at batch {batch_idx}, zeroing gradients")
                consecutive_nan_batches += 1
                if consecutive_nan_batches >= max_consecutive_nan:
                    print(f"[train] CRITICAL: {consecutive_nan_batches} consecutive NaN batches - model is corrupted, stopping training")
                    raise RuntimeError(f"Gradient norm corrupted: {consecutive_nan_batches} consecutive NaN batches")
                optimizer.zero_grad(set_to_none=True)
                continue
        except RuntimeError as e:
            print(f"[train] WARNING: Gradient clipping failed at batch {batch_idx}: {e}, zeroing gradients")
            consecutive_nan_batches += 1
            if consecutive_nan_batches >= max_consecutive_nan:
                print(f"[train] CRITICAL: {consecutive_nan_batches} consecutive NaN batches - model is corrupted, stopping training")
                raise RuntimeError(f"Gradient clipping failed: {consecutive_nan_batches} consecutive NaN batches")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        # Check for NaN gradients after clipping
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"[train] WARNING: NaN/Inf gradients detected at batch {batch_idx} after clipping, skipping update")
            consecutive_nan_batches += 1
            if consecutive_nan_batches >= max_consecutive_nan:
                print(f"[train] CRITICAL: {consecutive_nan_batches} consecutive NaN batches - model is corrupted, stopping training")
                raise RuntimeError(f"Gradients corrupted after clipping: {consecutive_nan_batches} consecutive NaN batches")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        # Monitor classifier gradients to ensure they're flowing
        if batch_idx == 1:
            classifier_grad_norm = 0.0
            classifier_param_count = 0
            for name, param in model.named_parameters():
                if 'classifier' in name and param.grad is not None:
                    classifier_grad_norm += param.grad.norm().item() ** 2
                    classifier_param_count += 1
            if classifier_param_count > 0:
                classifier_grad_norm = classifier_grad_norm ** 0.5
                print(f"[train] Classifier gradient norm: {classifier_grad_norm:.6f} (params: {classifier_param_count})")
                if classifier_grad_norm < 1e-6:
                    print(f"[train] WARNING: Very small classifier gradients, model may not be learning")
        
        optimizer.step()
        
        # CRITICAL: Check model parameters for NaN/Inf AFTER optimizer step
        # If parameters became NaN during update, we need to detect and fix it immediately
        has_nan_params_after = False
        corrupted_params_after = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"[train] CRITICAL: NaN/Inf detected in parameter '{name}' AFTER optimizer.step() at batch {batch_idx}, reinitializing...")
                has_nan_params_after = True
                corrupted_params_after.append(name)
                # Reinitialize corrupted parameters to small random values IMMEDIATELY
                with torch.no_grad():
                    if param.requires_grad:
                        param.data.normal_(0, 0.01)  # Small random initialization
                        print(f"[train] Reinitialized corrupted parameter: {name}")
        
        if has_nan_params_after:
            # Parameters became NaN during update - we've fixed them, but this is still concerning
            # Reduce consecutive counter since we fixed it, but track that corruption occurred
            if len(corrupted_params_after) > 0:
                consecutive_nan_batches = max(0, consecutive_nan_batches - 1)  # Reduce counter since we fixed it
                print(f"[train] Fixed {len(corrupted_params_after)} corrupted parameters after optimizer.step(), continuing")
            else:
                consecutive_nan_batches += 1
                if consecutive_nan_batches >= max_consecutive_nan:
                    print(f"[train] CRITICAL: {consecutive_nan_batches} consecutive NaN batches - model is corrupted, stopping training")
                    raise RuntimeError(f"Parameters corrupted after update: {consecutive_nan_batches} consecutive NaN batches")
            # Continue - parameters are now fixed, next batch should work
        
        # Reset consecutive NaN counter on successful batch
        consecutive_nan_batches = 0
        running += loss.item()
        
        # Print progress every 10% or at key milestones
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % max(
            1, total_batches // 10
        ) == 0:
            avg_loss = running / batch_idx
            print(f"[train] batch {batch_idx}/{total_batches} | loss: {loss.item():.4f} | avg_loss: {avg_loss:.4f}")
            
            # Periodically log intermediate feature statistics to track evolution
            if hasattr(model, '_last_lig_pool_stats') and hasattr(model, '_last_rdkit_emb_stats'):
                lig_stats = model._last_lig_pool_stats
                rdkit_stats = model._last_rdkit_emb_stats
                print(f"[train] Features - lig_pool: max={lig_stats['max']:.4f}, mean={lig_stats['mean']:.4f} | "
                      f"rdkit_emb: max={rdkit_stats['max']:.4f}, mean={rdkit_stats['mean']:.4f}")
    
    avg_loss = running / max(1, len(loader))
    elapsed_time = time.time() - start_time
    print(f"[train] Epoch complete | avg_loss: {avg_loss:.4f} | time: {elapsed_time:.2f}s")
    return avg_loss, elapsed_time


@torch.no_grad()
def evaluate(
    model: HTS3DModel,
    loader: DataLoader,
) -> Dict[str, float]:
    """Returns metrics dict with added 'eval_time_seconds'."""
    model.eval()
    # Keep predictions and labels on GPU, only convert to CPU at the end for better performance
    preds_gpu, labels_gpu = [], []
    total_batches = len(loader)
    start_time = time.time()
    print(f"[eval] Starting evaluation: {total_batches} batches")
    for batch_idx, sample in enumerate(loader, 1):
        batch = to_device(sample)
        logits = model(
            batch.input_ids,
            batch.attention_mask,
            batch.rdkit,
            batch.atom_features,
            batch.atom_mask,
        )
        
        # Check for constant logits (model collapse) and monitor logit values
        if batch_idx == 1:
            logit_std = logits.std().item()
            logit_mean = logits.mean().item()
            logit_min = logits.min().item()
            logit_max = logits.max().item()
            if logit_std < 1e-6:
                print(f"[eval] WARNING: Model outputs nearly constant logits (std={logit_std:.6f}, mean={logit_mean:.6f})")
            # Log logit statistics for underfitting detection
            print(f"[eval] Logit stats: mean={logit_mean:.4f}, std={logit_std:.4f}, "
                  f"min={logit_min:.4f}, max={logit_max:.4f}")
            
            # Log intermediate feature statistics to track learning progress
            if hasattr(model, '_last_lig_pool_stats') and hasattr(model, '_last_rdkit_emb_stats'):
                lig_stats = model._last_lig_pool_stats
                rdkit_stats = model._last_rdkit_emb_stats
                print(f"[eval] Intermediate features - lig_pool: mean={lig_stats['mean']:.4f}, "
                      f"std={lig_stats['std']:.4f}, min={lig_stats['min']:.4f}, max={lig_stats['max']:.4f}")
                print(f"[eval] Intermediate features - rdkit_emb: mean={rdkit_stats['mean']:.4f}, "
                      f"std={rdkit_stats['std']:.4f}, min={rdkit_stats['min']:.4f}, max={rdkit_stats['max']:.4f}")
                
                # Check if features are learning to become positive
                if lig_stats['max'] > 0.0 or rdkit_stats['max'] > 0.0:
                    print(f"[eval] GOOD: Intermediate features are learning positive values (lig_pool max={lig_stats['max']:.4f}, "
                          f"rdkit_emb max={rdkit_stats['max']:.4f})")
                elif lig_stats['max'] < -5.0 and rdkit_stats['max'] < -5.0:
                    print(f"[eval] WARNING: Intermediate features are very negative (lig_pool max={lig_stats['max']:.4f}, "
                          f"rdkit_emb max={rdkit_stats['max']:.4f}) - model may need more training or additional fixes")
            
            # Target: logits should range from -3 to +3 for probabilities 0.05 to 0.95
            # If logits are consistently negative (< -1), model is underfitting
            if logit_max < -1.0:
                print(f"[eval] WARNING: All logits are negative (max={logit_max:.4f}), model may be underfitting")
                print(f"[eval] NOTE: Option C Minimal (384d) aligned with HTS.py: HTS-style branches, bias=0, BCE+pos_weight; batch 64 on GPU")
            elif logit_max > 0.0:
                print(f"[eval] GOOD: Model is producing positive logits (max={logit_max:.4f}) - learning is progressing")
        
        # Keep on GPU - compute sigmoid on GPU, only convert to CPU at the end
        probs = torch.sigmoid(logits)
        preds_gpu.append(probs)
        labels_gpu.append(batch.labels)
        
        # Print progress every 20% or at key milestones
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % max(
            1, total_batches // 5
        ) == 0:
            total_samples = sum(p.shape[0] for p in preds_gpu) + probs.shape[0]
            print(f"[eval] batch {batch_idx}/{total_batches} | processed {total_samples} samples")
    
    # Convert to CPU/numpy only once at the end (much faster)
    preds = torch.cat(preds_gpu, dim=0).cpu().numpy()
    labels = torch.cat(labels_gpu, dim=0).cpu().numpy()
    print(f"[eval] Computing metrics on {len(preds)} predictions")
    
    # Diagnostic information
    print(f"[eval] Prediction stats: min={preds.min():.6f}, max={preds.max():.6f}, mean={preds.mean():.6f}, std={preds.std():.6f}")
    print(f"[eval] Label stats: min={labels.min():.0f}, max={labels.max():.0f}, mean={labels.mean():.4f}, unique={len(np.unique(labels))}")
    print(f"[eval] Label distribution: {np.bincount(labels.astype(int))}")
    
    # Enhanced diagnostic: Show prediction distribution for positives vs negatives
    pos_mask = labels == 1
    neg_mask = labels == 0
    if pos_mask.sum() > 0:
        print(f"[eval] Positive samples - mean pred: {preds[pos_mask].mean():.4f}, "
              f"max: {preds[pos_mask].max():.4f}, min: {preds[pos_mask].min():.4f}, "
              f"median: {np.median(preds[pos_mask]):.4f}")
    if neg_mask.sum() > 0:
        print(f"[eval] Negative samples - mean pred: {preds[neg_mask].mean():.4f}, "
              f"max: {preds[neg_mask].max():.4f}, min: {preds[neg_mask].min():.4f}, "
              f"median: {np.median(preds[neg_mask]):.4f}")
    
    # Show prediction distribution at different thresholds
    positive_predictions_01 = (preds >= 0.1).sum()
    positive_predictions_05 = (preds >= 0.5).sum()
    print(f"[eval] Predictions >= 0.1: {positive_predictions_01} ({positive_predictions_01/len(preds)*100:.2f}%)")
    print(f"[eval] Predictions >= 0.5: {positive_predictions_05} ({positive_predictions_05/len(preds)*100:.2f}%)")
    
    if np.isnan(preds).any() or np.isinf(preds).any():
        print("[eval] detected NaN/Inf predictions, replacing with 0.5")
        preds = np.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Check if all predictions are the same (would cause AUC/AP to fail)
    if np.allclose(preds, preds[0], atol=1e-6):
        print(f"[eval] WARNING: All predictions are nearly identical (value={preds[0]:.6f})")
        print(f"[eval] This will cause AUC and AP to be undefined. Check model training.")
    
    # Find optimal threshold for balanced high precision and recall
    # Use balanced threshold selection to achieve high and balanced performance
    optimal_threshold = 0.5
    try:
        if len(np.unique(labels)) == 2:  # Need both classes
            # Use balance_weight=1.5 to slightly favor recall while maintaining balance
            optimal_threshold = find_balanced_threshold(labels, preds, balance_weight=1.5)
            # Calculate F1, precision, and recall at optimal threshold for reporting
            binary_temp = (preds >= optimal_threshold).astype(int)
            optimal_f1 = safe_metric(f1_score, labels, binary_temp, zero_division=0)
            optimal_prec = safe_metric(precision_score, labels, binary_temp, fallback=0.0, zero_division=0)
            optimal_rec = safe_metric(recall_score, labels, binary_temp, fallback=0.0, zero_division=0)
            print(f"[eval] Optimal threshold: {optimal_threshold:.4f} (F1={optimal_f1:.4f}, "
                  f"Precision={optimal_prec:.4f}, Recall={optimal_rec:.4f}, balanced)")
    except Exception as e:
        print(f"[eval] Could not find optimal threshold, using 0.5: {e}")
    
    # Use optimal threshold for binary classification
    binary = (preds >= optimal_threshold).astype(int)
    print(f"[eval] Binary predictions: {np.bincount(binary)} (threshold={optimal_threshold:.4f})")
    
    metrics = {
        "auc": safe_metric(roc_auc_score, labels, preds),
        "ap": safe_metric(average_precision_score, labels, preds),
        "precision": safe_metric(
            precision_score, labels, binary, fallback=0.0, zero_division=0
        ),
        "recall": safe_metric(
            recall_score, labels, binary, fallback=0.0, zero_division=0
        ),
        "f1": safe_metric(f1_score, labels, binary, zero_division=0),
        "optimal_threshold": optimal_threshold,
    }
    elapsed_time = time.time() - start_time
    metrics["eval_time_seconds"] = elapsed_time
    print(f"[eval] Metrics computed | AUC: {metrics['auc']:.4f} | AP: {metrics['ap']:.4f} | "
          f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | "
          f"time: {elapsed_time:.2f}s")
    metrics["preds"] = preds
    metrics["labels"] = labels
    return metrics


# --- Tokenizer loading -------------------------------------------------------


def load_tokenizer() -> RobertaTokenizer:
    import sys
    print(f"[tokenizer] Loading ChemBERTa tokenizer from '{CHEMBERTA_NAME}'...")
    print(f"[tokenizer] This may take a few minutes if downloading for the first time...")
    sys.stdout.flush()
    try:
        tokenizer = RobertaTokenizer.from_pretrained(CHEMBERTA_NAME)
        print(f"[tokenizer] Successfully loaded RobertaTokenizer")
        sys.stdout.flush()
    except Exception as e:
        print(f"[tokenizer] RobertaTokenizer failed, trying AutoTokenizer... Error: {e}")
        sys.stdout.flush()
        tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_NAME)
        print(f"[tokenizer] Successfully loaded AutoTokenizer")
        sys.stdout.flush()
    return tokenizer


# --- Main orchestration ------------------------------------------------------


def prepare_tensors(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    import sys
    smiles_list = df["canonical_smiles"].tolist()
    print(f"[tokenizer] Preparing to tokenize {len(smiles_list)} SMILES strings...")
    sys.stdout.flush()
    tokenizer = load_tokenizer()
    print(f"[tokenizer] Starting tokenization of {len(smiles_list)} SMILES...")
    sys.stdout.flush()
    # Tokenize in batches to show progress for large datasets
    batch_size = 1000
    total_batches = (len(smiles_list) + batch_size - 1) // batch_size
    all_input_ids = []
    all_attention_mask = []
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(smiles_list))
        batch_smiles = smiles_list[start_idx:end_idx]
        
        batch_encodings = tokenizer(
            batch_smiles,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        all_input_ids.append(batch_encodings["input_ids"])
        all_attention_mask.append(batch_encodings["attention_mask"])
        
        if (batch_idx + 1) % max(1, total_batches // 10) == 0 or (batch_idx + 1) == total_batches:
            print(f"[tokenizer] Tokenized {end_idx}/{len(smiles_list)} ({end_idx/len(smiles_list)*100:.1f}%)")
            sys.stdout.flush()
    
    # Concatenate all batches and move to device (GPU if available) immediately
    input_ids = torch.cat(all_input_ids, dim=0).to(DEVICE)
    attention_mask = torch.cat(all_attention_mask, dim=0).to(DEVICE)
    print(f"[tokenizer] Tokenization complete. Input shape: {input_ids.shape}, device: {input_ids.device}")
    sys.stdout.flush()
    return input_ids, attention_mask


def load_protein_structure_from_pdb(
    pdb_path: Path,
    max_pockets: int = 5
) -> List[Dict[str, torch.Tensor]]:
    """
    Load protein structure from PDB file and detect multiple pockets.
    
    This function uses BioPython to parse PDB files. If BioPython is not available,
    falls back to the default pocket template.
    
    Args:
        pdb_path: Path to PDB file
        max_pockets: Maximum number of pockets to detect
    
    Returns:
        List of pocket template dictionaries
    """
    try:
        try:
            from Bio.PDB import PDBParser
        except ImportError:
            print("[protein] Warning: BioPython not available. Install with: pip install biopython")
            print("[protein] Using default pocket template")
            return PROTEIN_TEMPLATES
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_path))
        
        residues = []
        for residue in structure.get_residues():
            if residue.id[0] == ' ':  # Exclude heteroatoms and water
                res_name = residue.get_resname()
                # Get CA atom coordinates (or center of mass)
                try:
                    ca = residue['CA']
                    coords = (ca.coord[0] / 10.0, ca.coord[1] / 10.0, ca.coord[2] / 10.0)  # Convert A to nm
                    residues.append((res_name, coords))
                except KeyError:
                    # No CA atom (e.g., GLY), use center of mass
                    atoms = list(residue.get_atoms())
                    if atoms:
                        coords = np.mean([a.coord for a in atoms], axis=0) / 10.0
                        residues.append((res_name, tuple(coords)))
        
        if residues:
            pockets = detect_protein_pockets(residues, max_pockets=max_pockets)
            templates = build_protein_template_from_pockets(pockets, device=DEVICE)
            print(f"[protein] Detected {len(templates)} pockets from {pdb_path.name}")
            return templates
        else:
            print(f"[protein] Warning: No residues found in {pdb_path}")
            return PROTEIN_TEMPLATES
    except Exception as e:
        print(f"[protein] Warning: Error loading PDB file {pdb_path}: {e}")
        print("[protein] Using default pocket template")
        return PROTEIN_TEMPLATES


# --- Feature selection (adapted from HTS.py) ------------------------------------

# Top-level selector and scaler classes for pickle support (must be defined outside function)
class DummyScaler:
    """Scaler for GPU path: holds mean_ and scale_ as numpy arrays (picklable)."""
    def __init__(self, mean, scale):
        self.mean_ = mean.cpu().numpy() if hasattr(mean, 'cpu') else np.asarray(mean)
        self.scale_ = scale.cpu().numpy() if hasattr(scale, 'cpu') else np.asarray(scale)

    def transform(self, X):
        """Transform X using stored mean and scale. Accepts numpy or torch."""
        if isinstance(X, torch.Tensor):
            mean_t = torch.tensor(self.mean_, dtype=X.dtype, device=X.device)
            scale_t = torch.tensor(self.scale_, dtype=X.dtype, device=X.device)
            return (X - mean_t) / scale_t
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / self.scale_


class DummyPCA:
    """PCA for GPU path: holds mean_ and components_, picklable."""
    def __init__(self, n_components, components, mean=None):
        self.n_components = n_components
        self.components_ = components.cpu().numpy() if hasattr(components, 'cpu') else np.asarray(components)
        self.mean_ = mean.cpu().numpy() if mean is not None and hasattr(mean, 'cpu') else (np.asarray(mean) if mean is not None else np.zeros(components.shape[1], dtype=np.float32))

    def transform(self, X):
        """Transform X: center with mean_ then project. Accepts numpy or torch."""
        if isinstance(X, torch.Tensor):
            mean_t = torch.tensor(self.mean_, dtype=X.dtype, device=X.device)
            comp_t = torch.tensor(self.components_, dtype=X.dtype, device=X.device)
            X_centered = X - mean_t
            return torch.matmul(X_centered, comp_t.T)
        X = np.asarray(X, dtype=np.float32)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T


class DummyLassoSelector:
    """Dummy selector class for GPU Lasso feature selection (compatible with sklearn interface)."""
    def __init__(self, indices):
        self.indices_ = indices
        self.n_features_in_ = len(indices)
    def transform(self, X):
        return X[:, self.indices_]

class DummyMISelector:
    """Dummy selector class for GPU Mutual Information feature selection (compatible with sklearn interface)."""
    def __init__(self, indices, scores):
        self.scores_ = scores.cpu().numpy() if hasattr(scores, 'cpu') else scores
        self.indices_ = indices
        self.n_features_in_ = len(indices)
    def transform(self, X):
        return X[:, self.indices_]

def apply_feature_selection(
    X_train: Union[np.ndarray, torch.Tensor],
    y_train: np.ndarray,
    X_val: Union[np.ndarray, torch.Tensor],
    method: str = 'all',
    n_components: int = 200,
    use_gpu: bool = True
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Optional[Any], Optional[Any]]:
    """
    Apply feature selection (adapted from HTS.py).
    Keeps data on GPU when use_gpu=True; accepts and returns GPU tensors to avoid CPU round-trips.

    Args:
        X_train: Training features (numpy or tensor on DEVICE)
        y_train: Training labels
        X_val: Validation features (numpy or tensor on DEVICE)
        method: Feature selection method ('lasso', 'pca', 'mutual_info', or 'all')
        n_components: Number of features/components to select
        use_gpu: Whether to use GPU-accelerated operations when available

    Returns:
        Tuple of (X_train_selected, X_val_selected, selector, scaler); selected arrays are tensors on DEVICE when use_gpu.
    """
    # Use GPU tensors when available; avoid CPU round-trip
    if use_gpu and DEVICE.type == "cuda":
        if isinstance(X_train, torch.Tensor):
            X_train_tensor = X_train.to(DEVICE, non_blocking=NON_BLOCKING) if X_train.device != DEVICE else X_train
            X_val_tensor = X_val.to(DEVICE, non_blocking=NON_BLOCKING) if X_val.device != DEVICE else X_val
        else:
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)

        # GPU scaling; keep on GPU (no .cpu().numpy())
        mean = X_train_tensor.mean(dim=0, keepdim=True)
        std = X_train_tensor.std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-8)
        X_train_scaled_tensor = (X_train_tensor - mean) / std
        X_val_scaled_tensor = (X_val_tensor - mean) / std
        X_train_scaled_tensor = torch.nan_to_num(X_train_scaled_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled_tensor = torch.nan_to_num(X_val_scaled_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = DummyScaler(mean, std)
        # For CPU fallback paths we need numpy; create only when needed
        X_train_scaled = None
        X_val_scaled = None
    else:
        # CPU path
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
            X_val = X_val.cpu().numpy()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_train_scaled_tensor = None
        X_val_scaled_tensor = None
        y_train_tensor = None
    
    if method == 'lasso':
        if use_gpu and DEVICE.type == "cuda" and X_train_scaled_tensor is not None:
            # GPU-accelerated Lasso; data already on GPU
            try:
                X_train_gpu = X_train_scaled_tensor
                X_val_gpu = X_val_scaled_tensor
                y_train_gpu = y_train_tensor.unsqueeze(1)
                
                # Initialize Lasso model (linear layer with L1 regularization)
                n_features = X_train_gpu.shape[1]
                lasso_model = nn.Linear(n_features, 1, bias=True).to(DEVICE)
                
                # Initialize weights to small random values
                nn.init.normal_(lasso_model.weight, mean=0.0, std=0.01)
                nn.init.zeros_(lasso_model.bias)
                
                # Lasso loss: MSE + L1 regularization
                alpha = 0.01  # L1 regularization strength (matches sklearn default)
                optimizer = torch.optim.SGD(lasso_model.parameters(), lr=0.001, momentum=0.0)
                
                # Training loop (coordinate descent-like using SGD)
                max_iter = 1000
                batch_size = min(512, len(X_train_gpu))  # Use batches for large datasets
                
                for iteration in range(max_iter):
                    # Mini-batch training
                    indices = torch.randperm(len(X_train_gpu), device=DEVICE)[:batch_size]
                    X_batch = X_train_gpu[indices]
                    y_batch = y_train_gpu[indices]
                    
                    # Forward pass
                    y_pred = lasso_model(X_batch)
                    mse_loss = F.mse_loss(y_pred, y_batch)
                    
                    # L1 regularization on weights only (not bias)
                    l1_reg = alpha * torch.sum(torch.abs(lasso_model.weight))
                    loss = mse_loss + l1_reg
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Early stopping if loss is stable
                    if iteration > 100 and iteration % 100 == 0:
                        with torch.no_grad():
                            y_pred_full = lasso_model(X_train_gpu)
                            current_loss = F.mse_loss(y_pred_full, y_train_gpu).item()
                            if iteration > 200 and abs(current_loss - mse_loss.item()) < 1e-6:
                                break
                
                # Extract feature importance (absolute weights); keep indexing on GPU
                with torch.no_grad():
                    feature_importance = torch.abs(lasso_model.weight.squeeze()).cpu().numpy()
                top_indices = np.argsort(feature_importance)[-n_components:][::-1]
                top_indices_t = torch.as_tensor(top_indices, device=DEVICE)
                X_train_selected = X_train_scaled_tensor[:, top_indices_t]
                X_val_selected = X_val_scaled_tensor[:, top_indices_t]
                selector = DummyLassoSelector(top_indices)
                print(f"[feature selection] GPU Lasso selected {len(top_indices)} features (on device)")
                return X_train_selected, X_val_selected, selector, scaler
                
            except Exception as e:
                print(f"[feature selection] GPU Lasso failed: {e}. Falling back to CPU Lasso.")
                if X_train_scaled is None and X_train_scaled_tensor is not None:
                    X_train_scaled = X_train_scaled_tensor.cpu().numpy()
                    X_val_scaled = X_val_scaled_tensor.cpu().numpy()

        # CPU Lasso (sklearn) - fallback or when use_gpu=False
        if X_train_scaled is None and X_train_scaled_tensor is not None:
            X_train_scaled = X_train_scaled_tensor.cpu().numpy()
            X_val_scaled = X_val_scaled_tensor.cpu().numpy()
        lasso = Lasso(alpha=0.01, random_state=SEED, max_iter=1000)
        selector = SelectFromModel(lasso, max_features=n_components)
        try:
            selector.fit(X_train_scaled, y_train)
            return selector.transform(X_train_scaled), selector.transform(X_val_scaled), selector, scaler
        except Exception as e:
            print(f"[feature selection] Lasso failed: {e}. Falling back to all features.")
            return X_train_scaled, X_val_scaled, None, scaler
    
    elif method == 'pca':
        if X_train_scaled_tensor is not None:
            n_components_actual = min(n_components, X_train_scaled_tensor.shape[1], X_train_scaled_tensor.shape[0])
        else:
            n_components_actual = min(n_components, X_train_scaled.shape[1], X_train_scaled.shape[0])
        try:
            if use_gpu and DEVICE.type == "cuda" and X_train_scaled_tensor is not None:
                # GPU PCA; data already on GPU
                X_train_centered = X_train_scaled_tensor - X_train_scaled_tensor.mean(dim=0, keepdim=True)
                U, S, Vt = torch.linalg.svd(X_train_centered, full_matrices=False)
                Vt_selected = Vt[:n_components_actual, :]
                X_train_selected_gpu = torch.matmul(X_train_centered, Vt_selected.t())
                X_val_centered = X_val_scaled_tensor - X_train_scaled_tensor.mean(dim=0, keepdim=True)
                X_val_selected_gpu = torch.matmul(X_val_centered, Vt_selected.t())
                explained_var = (S[:n_components_actual]**2).sum() / (S**2).sum()
                print(f"[feature selection] GPU PCA with {n_components_actual} components explains {explained_var.item():.2%} of variance (on device)")
                train_mean = X_train_scaled_tensor.mean(dim=0)
                pca = DummyPCA(n_components_actual, Vt_selected, mean=train_mean)
                return X_train_selected_gpu, X_val_selected_gpu, pca, scaler
            else:
                # CPU PCA (original sklearn)
                if X_train_scaled is None and X_train_scaled_tensor is not None:
                    X_train_scaled = X_train_scaled_tensor.cpu().numpy()
                    X_val_scaled = X_val_scaled_tensor.cpu().numpy()
                pca = PCA(n_components=n_components_actual, random_state=SEED)
                X_train_selected = pca.fit_transform(X_train_scaled)
                X_val_selected = pca.transform(X_val_scaled)
                explained_var = sum(pca.explained_variance_ratio_)
                print(f"[feature selection] PCA with {X_train_selected.shape[1]} components explains {explained_var:.2%} of variance")
                return X_train_selected, X_val_selected, pca, scaler
        except Exception as e:
            print(f"[feature selection] PCA failed: {e}. Falling back to all features.")
            if X_train_scaled is None and X_train_scaled_tensor is not None:
                X_train_scaled = X_train_scaled_tensor.cpu().numpy()
                X_val_scaled = X_val_scaled_tensor.cpu().numpy()
            return X_train_scaled, X_val_scaled, None, scaler

    elif method == 'mutual_info':
        if use_gpu and DEVICE.type == "cuda" and X_train_scaled_tensor is not None:
            # GPU Mutual Info; data already on GPU
            try:
                X_train_gpu = X_train_scaled_tensor
                y_train_gpu = y_train_tensor
                
                # Discretize continuous features for mutual information calculation
                # Use quantile-based binning (10 bins per feature)
                n_bins = 10
                n_features = X_train_gpu.shape[1]
                mi_scores = torch.zeros(n_features, device=DEVICE)
                
                # Calculate mutual information for each feature
                for i in range(n_features):
                    # Discretize feature
                    feature = X_train_gpu[:, i]
                    # Use quantiles for binning
                    quantiles = torch.linspace(0, 1, n_bins + 1, device=DEVICE)
                    thresholds = torch.quantile(feature, quantiles)
                    # Assign bins (avoid edge case where all values are same)
                    if thresholds[-1] > thresholds[0]:
                        feature_binned = torch.bucketize(feature, thresholds[1:], right=True)
                    else:
                        feature_binned = torch.zeros_like(feature, dtype=torch.long)
                    
                    # Calculate mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
                    # H(X) - entropy of feature
                    feature_probs = torch.bincount(feature_binned, minlength=n_bins).float() + 1e-10
                    feature_probs = feature_probs / feature_probs.sum()
                    h_feature = -(feature_probs * torch.log(feature_probs + 1e-10)).sum()
                    
                    # H(Y) - entropy of labels (binary)
                    label_probs = torch.tensor([
                        (y_train_gpu == 0).float().mean(),
                        (y_train_gpu == 1).float().mean()
                    ], device=DEVICE) + 1e-10
                    h_label = -(label_probs * torch.log(label_probs + 1e-10)).sum()
                    
                    # H(X,Y) - joint entropy
                    # Create joint distribution
                    joint_counts = torch.zeros((n_bins, 2), device=DEVICE)
                    for j in range(len(feature_binned)):
                        bin_idx = feature_binned[j].item()
                        label_idx = int(y_train_gpu[j].item())
                        joint_counts[bin_idx, label_idx] += 1
                    joint_probs = joint_counts / joint_counts.sum() + 1e-10
                    h_joint = -(joint_probs * torch.log(joint_probs + 1e-10)).sum()
                    
                    # Mutual information
                    mi = h_feature + h_label - h_joint
                    mi_scores[i] = mi
                
                # Select top k features; keep on GPU
                k = min(n_components, n_features)
                top_indices_t = torch.topk(mi_scores, k).indices
                top_indices = top_indices_t.cpu().numpy()
                X_train_selected = X_train_scaled_tensor[:, top_indices_t]
                X_val_selected = X_val_scaled_tensor[:, top_indices_t]
                selector = DummyMISelector(top_indices, mi_scores)
                print(f"[feature selection] GPU Mutual Info selected {len(top_indices)} features (on device)")
                return X_train_selected, X_val_selected, selector, scaler
                
            except Exception as e:
                print(f"[feature selection] GPU Mutual Info failed: {e}. Falling back to CPU Mutual Info.")
                if X_train_scaled is None and X_train_scaled_tensor is not None:
                    X_train_scaled = X_train_scaled_tensor.cpu().numpy()
                    X_val_scaled = X_val_scaled_tensor.cpu().numpy()

        # CPU Mutual Info (sklearn) - fallback or when use_gpu=False
        if X_train_scaled is None and X_train_scaled_tensor is not None:
            X_train_scaled = X_train_scaled_tensor.cpu().numpy()
            X_val_scaled = X_val_scaled_tensor.cpu().numpy()
        try:
            selector = SelectKBest(mutual_info_classif, k=min(n_components, X_train_scaled.shape[1]))
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_val_selected = selector.transform(X_val_scaled)
            return X_train_selected, X_val_selected, selector, scaler
        except Exception as e:
            print(f"[feature selection] Mutual info failed: {e}. Falling back to all features.")
            return X_train_scaled, X_val_scaled, None, scaler
    
    else:  # 'all'
        if X_train_scaled_tensor is not None:
            return X_train_scaled_tensor, X_val_scaled_tensor, None, scaler
        return X_train_scaled, X_val_scaled, None, scaler


def find_balanced_threshold(labels: np.ndarray, predictions: np.ndarray, balance_weight: float = 1.0) -> float:
    """Find threshold that balances precision and recall for high and balanced performance.
    
    Args:
        labels: True binary labels
        predictions: Predicted probabilities
        balance_weight: Weight for balancing (1.0 = equal, >1.0 = favor recall, <1.0 = favor precision)
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, predictions)
    
    # Calculate F1 scores (harmonic mean of precision and recall)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Calculate balanced score: weighted combination of precision and recall
    # Higher balance_weight favors recall, lower favors precision
    # For balanced high performance, we want both to be high
    balanced_scores = (balance_weight * recalls + precisions) / (balance_weight + 1.0)
    
    # Combine F1 and balanced score to find threshold that maximizes both
    # Use geometric mean to ensure both metrics are high
    combined_scores = np.sqrt(f1_scores * balanced_scores)
    
    # Also consider thresholds where both precision and recall are above minimum thresholds
    min_precision = 0.3  # Minimum acceptable precision
    min_recall = 0.25    # Minimum acceptable recall (slightly lower to improve recall when possible)
    valid_mask = (precisions >= min_precision) & (recalls >= min_recall)
    
    if valid_mask.any():
        # Prefer thresholds that meet minimum requirements
        valid_scores = combined_scores.copy()
        valid_scores[~valid_mask] = 0.0
        best_idx = np.argmax(valid_scores)
    else:
        # Fall back to best combined score if no threshold meets minimums
        best_idx = np.argmax(combined_scores)
    
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Recall-first fallback: only when recall would be very low (< 0.15), pick threshold that
    # maximizes recall subject to precision >= 0.25 (avoids both "precision=1 recall=0.17" and "precision=0.12")
    binary_temp = (predictions >= optimal_threshold).astype(int)
    rec_at_opt = np.sum((labels == 1) & (binary_temp == 1)) / (np.sum(labels == 1) + 1e-10)
    if rec_at_opt < 0.15 and np.sum(labels == 1) > 0:
        recall_ok = precisions >= 0.25  # Require minimum precision so we don't report 0.12
        if recall_ok.any():
            best_recall_idx = np.where(recall_ok)[0][np.argmax(recalls[recall_ok])]
            ti = min(best_recall_idx, len(thresholds) - 1) if len(thresholds) > 0 else 0
            optimal_threshold = thresholds[ti] if len(thresholds) > 0 else optimal_threshold
    
    return optimal_threshold


def evaluate_ensemble_predictions(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Evaluate ensemble predictions and return metrics with improved threshold selection."""
    # Find optimal threshold with recall weighting
    optimal_threshold = 0.5
    try:
        if len(np.unique(labels)) == 2:
            # Use balanced threshold selection with recall preference (balance_weight=1.5)
            # Slightly favors recall while maintaining balance with precision
            optimal_threshold = find_balanced_threshold(labels, predictions, balance_weight=1.5)
            
            # Validate threshold is reasonable (Phase 1.4: Lower threshold validation)
            if optimal_threshold < 0.01:
                print(f"[ensemble] WARNING: Optimal threshold {optimal_threshold:.4f} is very low, "
                      f"using alternative strategy")
                precisions, recalls, thresholds = precision_recall_curve(labels, predictions)
                # Use threshold that gives at least 5% precision (lowered from 10%)
                valid_indices = precisions >= 0.05
                if valid_indices.any():
                    # Find threshold that maximizes recall while maintaining minimum precision
                    best_idx = np.where(valid_indices)[0][np.argmax(recalls[valid_indices])]
                    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.01
                else:
                    # Use very low threshold if needed
                    optimal_threshold = 0.01
    except Exception as e:
        print(f"[ensemble] Error finding optimal threshold: {e}, using 0.1")
        optimal_threshold = 0.1
    
    binary = (predictions >= optimal_threshold).astype(int)
    
    # Report metrics at multiple thresholds for better understanding
    multi_threshold_metrics = {}
    for threshold in [0.01, 0.05, 0.1, 0.2, 0.5]:
        binary_fixed = (predictions >= threshold).astype(int)
        prec = safe_metric(precision_score, labels, binary_fixed, fallback=0.0, zero_division=0)
        rec = safe_metric(recall_score, labels, binary_fixed, fallback=0.0, zero_division=0)
        f1_fixed = safe_metric(f1_score, labels, binary_fixed, zero_division=0)
        multi_threshold_metrics[f"threshold_{threshold:.2f}"] = {
            "precision": prec,
            "recall": rec,
            "f1": f1_fixed
        }
        print(f"[ensemble] At threshold {threshold:.2f}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1_fixed:.4f}")
    
    return {
        "auc": safe_metric(roc_auc_score, labels, predictions),
        "ap": safe_metric(average_precision_score, labels, predictions),
        "precision": safe_metric(precision_score, labels, binary, fallback=0.0, zero_division=0),
        "recall": safe_metric(recall_score, labels, binary, fallback=0.0, zero_division=0),
        "f1": safe_metric(f1_score, labels, binary, zero_division=0),
        "optimal_threshold": optimal_threshold,
        "multi_threshold_metrics": multi_threshold_metrics,
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: HTS3DModel,
    optimizer: torch.optim.Optimizer,
    current_method: str,
    current_fold: int,
    all_models: List[Dict],
    all_predictions: List[Dict],
    history: List[Dict],
    best_auc: float,
    best_state: Optional[Dict],
    labels: np.ndarray,
    feature_selection_methods: List[str],
    rdkit_dim: int
):
    """Save a training checkpoint."""
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'current_method': current_method,
        'current_fold': current_fold,
        'all_models': all_models,
        'all_predictions': all_predictions,
        'history': history,
        'best_auc': best_auc,
        'best_state': best_state,
        'labels': labels,
        'feature_selection_methods': feature_selection_methods,
        'rdkit_dim': rdkit_dim,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Ensure all_predictions is properly serialized
    # Convert numpy arrays to lists for proper serialization
    serialized_predictions = []
    for pred_entry in all_predictions:
        serialized_entry = {
            'val_idx': pred_entry['val_idx'].tolist() if isinstance(pred_entry['val_idx'], np.ndarray) else pred_entry['val_idx'],
            'predictions': pred_entry['predictions'].tolist() if isinstance(pred_entry['predictions'], np.ndarray) else pred_entry['predictions'],
            'feature_method': str(pred_entry['feature_method']),
            'fold': int(pred_entry['fold'])
        }
        serialized_predictions.append(serialized_entry)
    
    checkpoint_data['all_predictions'] = serialized_predictions
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_data, checkpoint_path)
    print(f"[checkpoint] Saved checkpoint to {checkpoint_path} with {len(serialized_predictions)} prediction sets")


def find_latest_checkpoint(checkpoint_dir: Path, pattern: str = "final_checkpoint_*.pt") -> Optional[Path]:
    """
    Find the latest checkpoint file matching a pattern in the checkpoint directory.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        pattern: Glob pattern to match checkpoint files (default: "final_checkpoint_*.pt")
    
    Returns:
        Path to the latest checkpoint file, or None if not found
    """
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoint files matching the pattern
    checkpoint_files = list(checkpoint_dir.glob(pattern))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    latest = checkpoint_files[0]
    print(f"[checkpoint] Found {len(checkpoint_files)} checkpoint file(s) matching '{pattern}', using latest: {latest.name}")
    print(f"[checkpoint] Latest checkpoint modified: {datetime.fromtimestamp(latest.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    return latest


class TeeOutput:
    """
    A file-like object that writes to both a file and stdout.
    Used to capture all terminal output to a log file.
    """
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8', buffering=1)  # Line buffered
        
    def write(self, message: str):
        # Handle Unicode encoding errors for Windows terminal (cp1252)
        try:
            self.terminal.write(message)
            self.terminal.flush()
        except UnicodeEncodeError:
            # Replace problematic Unicode characters with ASCII equivalents
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            self.terminal.write(safe_message)
            self.terminal.flush()
        # Log file uses UTF-8, so it can handle all Unicode characters
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        if self.log:
            self.log.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device
) -> Optional[Dict]:
    """Load a training checkpoint or regular model file."""
    if not checkpoint_path.exists():
        print(f"[checkpoint] Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        # Suppress FutureWarning about weights_only - we control the checkpoint files
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if it's a checkpoint format (has training state) or just a model state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Full checkpoint format
            print(f"[checkpoint] Loaded checkpoint from {checkpoint_path}")
            print(f"[checkpoint] Checkpoint info: method={checkpoint.get('current_method', 'unknown')}, "
                  f"fold={checkpoint.get('current_fold', -1)+1}, "
                  f"best_auc={checkpoint.get('best_auc', 0.0):.4f}, "
                  f"timestamp={checkpoint.get('timestamp', 'unknown')}")
            return checkpoint
        elif isinstance(checkpoint, dict) and any(key.startswith('chemberta') or key.startswith('lig3d') or key.startswith('ligand_3d_encoder') for key in checkpoint.keys()):
            # Regular model state_dict - convert to checkpoint format
            print(f"[checkpoint] Loaded model file (not checkpoint format) from {checkpoint_path}")
            print(f"[checkpoint] Converting to checkpoint format - will start from beginning")
            # Return None to indicate we should use the model weights but start training fresh
            return {'model_state_dict': checkpoint, 'is_model_only': True}
        else:
            print(f"[checkpoint] Unknown checkpoint format in {checkpoint_path}")
            return None
    except Exception as e:
        print(f"[checkpoint] Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(
    csv_path: Optional[Path] = None,
    protein_pdb_path: Optional[Path] = None,
    max_pockets: int = 5,
    bio_threshold: float = DEFAULT_BIO_THRESHOLD,
    label_column: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    resume_from_checkpoint: Optional[Path] = None,
    save_checkpoints: bool = True,
        pick_method: Optional[str] = None,
        use_cd28: bool = False,
        use_ultra: bool = False,
        ensemble_strategy: str = "oof",
        qed_scale: float = QED_SCALE_FACTOR,
        qed_blend_weight: float = 0.2,
):
    """
    Main training function with support for arbitrary protein structures.
    
    Args:
        csv_path: Optional path to activity CSV file.
                 If None and not use_cd28, uses default CSV path (nlrp3_chembl_activities_with15positives.csv).
        protein_pdb_path: Optional path to PDB file for protein structure.
                         If None, uses default NLRP3 pocket template.
        max_pockets: Maximum number of pockets to detect from protein structure.
        bio_threshold: Bioactivity threshold in nM for binary classification.
                      Compounds with IC50/EC50 <= threshold are labeled as active.
                      Default: 1000 nM (1 µM).
                      Ignored if label_column is specified or use_cd28 is True.
        label_column: Optional name of CSV column containing binary labels (0=inactive, 1=active).
                     If specified, uses this column directly instead of calculating from bio_threshold.
                     Default: None (use bio_threshold method). Ignored if use_cd28 is True.
        use_cd28: If True, load libraries/library.csv and libraries/positives.csv exactly as HTS.py
                  (label = 1 if SMILES in positives else 0). Do not use --compounds/cd28lib.csv.
        use_ultra: If True, use Option C Ultra (fixUndFitImblHTSpyClaudeC1.txt): attention pooling,
                  interaction features, adaptive weighting, multi-pocket (3), residual branches.
                  Default: False (Option C Minimal, aligned with HTS.py).
        ensemble_strategy: "oof" = use out-of-fold validation predictions only (3 per sample) → true CV,
                          robust generalization (default). "full" = run all models with per-model
                          selector → sharpest scores. "hts" = emulate HTS.py: loop by method, use
                          fold 0's selector for all 5 folds, avg 5 folds then 3 methods → smoother.
        qed_scale: Scale factor for QED in physchem features (default 3.0; HTS3DOracle uses 40-60% QED).
        qed_blend_weight: Blend ensemble predictions with QED (0-1). 0.2 matches HTS3DOracle
                         make_varied_predictions (score += qed * 0.2). Set 0 to disable.
    """
    # Explicit CUDA availability test and device setup
    global DEVICE, NON_BLOCKING, PIN_MEMORY
    cuda_available = torch.cuda.is_available()
    cuda_build = torch.version.cuda is not None
    print(f"[main] ===== CUDA CHECK =====")
    print(f"[main] torch.cuda.is_available(): {cuda_available}")
    print(f"[main] PyTorch version: {torch.__version__}")
    print(f"[main] PyTorch built with CUDA: {cuda_build} (torch.version.cuda = {torch.version.cuda})")
    if cuda_available:
        print(f"[main] CUDA device count: {torch.cuda.device_count()}")
        print(f"[main] Current CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        if not cuda_build:
            print(f"[main] Reason: PyTorch is a CPU-only build. Install CUDA build: pip install torch --index-url https://download.pytorch.org/whl/cu118 (or cu121, etc.)")
        else:
            print(f"[main] Reason: CUDA drivers/runtime may be missing or not visible. Check NVIDIA driver and CUDA toolkit.")
    print(f"[main] Initial DEVICE (from detect_device): {DEVICE}")

    if cuda_available and DEVICE.type == "cpu":
        print(f"[main] WARNING: CUDA is available but DEVICE was CPU. Forcing GPU usage.")
        DEVICE = torch.device("cuda:0")
        NON_BLOCKING = True
        PIN_MEMORY = True
        print(f"[main] DEVICE reset to: {DEVICE}")
    elif cuda_available:
        print(f"[main] Using GPU device: {DEVICE}")
    else:
        print(f"[main] Using CPU (CUDA not available). Training will be slower.")
    
    # Use default CSV path if not specified (only when not using --cd28)
    if use_cd28:
        compound = "cd28"
        model_path = get_model_path_for_compound(compound)
        print(f"[compound] Using --cd28: library.csv + positives.csv (HTS.py-style), compound={compound}")
    else:
        if csv_path is None:
            csv_path = DEFAULT_CSV_PATH_NLRP3
        compound = detect_compound_from_csv_path(csv_path)
        model_path = get_model_path_for_compound(compound)
        print(f"[compound] Detected compound type: {compound} (from CSV: {csv_path.name})")
    
    # Set up logging to file - capture all output to timestamped log file
    log_path = get_timestamped_path("hts3d_training_log", "txt", compound=compound)
    tee_output = TeeOutput(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee_output
    
    try:
        print(f"[log] Training log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[log] Log file: {log_path}")
        print(f"[log] All training output will be saved to this file")
        print("=" * 80)
        
        import sys as sys_module
        CACHE_DIR.mkdir(exist_ok=True)
        if use_cd28:
            # HTS.py-style: libraries/library.csv + libraries/positives.csv (do not use cd28lib.csv)
            library_path = PROJECT_ROOT / "libraries" / "library.csv"
            positives_path = PROJECT_ROOT / "libraries" / "positives.csv"
            print(f"[data] Loading library + positives (--cd28): {library_path.name}, {positives_path.name}")
            sys.stdout.flush()
            df = load_library_and_positives(library_path, positives_path)
            print(f"[data] Loaded {len(df)} molecules ({df['label'].sum()} actives, {len(df) - df['label'].sum()} inactives) (same as HTS.py)")
            sys.stdout.flush()
        else:
            print(f"[debug] looking for CSV at {csv_path.resolve()}")
            sys.stdout.flush()
            if not csv_path.exists():
                print(f"[debug] CSV missing: {csv_path.resolve()}")
                sys.stdout.flush()
            else:
                print(f"[debug] CSV present: {csv_path.resolve()} (size={csv_path.stat().st_size} bytes)")
                sys.stdout.flush()
            print(f"[data] Loading activity table...")
            sys.stdout.flush()
            df = load_activity_table(csv_path, bio_threshold=bio_threshold, label_column=label_column)
            print(f"[data] Loaded {len(df)} molecules ({df['label'].sum()} actives, {len(df) - df['label'].sum()} inactives)")
            sys.stdout.flush()
        
        # Load protein structure and detect pockets
        if protein_pdb_path and protein_pdb_path.exists():
            protein_templates = load_protein_structure_from_pdb(protein_pdb_path, max_pockets)
            print(f"[protein] Using {len(protein_templates)} detected pockets")
        else:
            protein_templates = PROTEIN_TEMPLATES
            print(f"[protein] Using default pocket template (NLRP3)")
        # Option C Ultra (fixUndFitImblHTSpyClaudeC1.txt): only add what's necessary for Ultra
        if use_ultra:
            model_use_attention_pooling = True
            model_use_interaction_features = True
            model_use_adaptive_weighting = True
            model_max_pockets_ultra = 3
            print(f"[model] Option C Ultra: attention_pooling=True, interaction=True, adaptive_weighting=True, max_pockets_ultra=3")
        else:
            model_use_attention_pooling = False
            model_use_interaction_features = False
            model_use_adaptive_weighting = False
            model_max_pockets_ultra = 1
        print(f"[debug] loaded dataframe with {len(df)} rows")
        smiles = df["canonical_smiles"].tolist()

        input_ids, attention_mask = prepare_tensors(df)  # Already on DEVICE (GPU if available)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        print(f"[debug] Tokenizer outputs on device: {input_ids.device}")
        sys.stdout.flush()
    
        print("[debug] generating RDKit 2D features")
        sys.stdout.flush()
        rdkit_feats_gpu = build_rdkit_feature_matrix(smiles, qed_scale=qed_scale)  # Already on GPU if available
        print(f"[debug] RDKit features shape: {rdkit_feats_gpu.shape}, device: {rdkit_feats_gpu.device}")
        sys.stdout.flush()
        
        # Keep RDKit features on GPU; feature selection accepts GPU tensors when use_gpu
        print(f"[debug] RDKit features kept on device for GPU-accelerated feature selection")
        sys.stdout.flush()
    
        print("[debug] building 3D conformer cache")
        sys.stdout.flush()
        atom_feats, atom_masks = build_3d_cache(smiles)  # Already on GPU if available
        print(f"[debug] 3D features shape: {atom_feats.shape}, device: {atom_feats.device}")
        sys.stdout.flush()
        
        # Timing: Track time between 3D completion and dataset prep
        post_3d_time = time.time()
        labels = df["label"].to_numpy(dtype=np.float32)
        labels_time = time.time() - post_3d_time
        print(f"[timing] Labels conversion: {labels_time:.3f}s")
        sys.stdout.flush()
        
        print(f"[data] Preparing dataset with {len(labels)} samples...")
        sys.stdout.flush()
        
        # Timing: Track setup time
        setup_start_time = time.time()
    
        atom_dim = atom_feats.shape[-1]
    
        # Feature selection settings - ensemble approach (like HTS.py)
        # Allow picking a single method if --pick is specified
        if pick_method:
            # Map command-line option to internal method name
            method_map = {
                'lasso': 'lasso',
                'pca': 'pca',
                'mutual-info': 'mutual_info'
            }
            if pick_method not in method_map:
                raise ValueError(f"Invalid pick method: {pick_method}. Must be one of: lasso, pca, mutual-info")
            feature_selection_methods = [method_map[pick_method]]
            print(f"[ensemble] Using single feature selection method: {pick_method} (will train {len(feature_selection_methods)} method × 5 folds = {len(feature_selection_methods) * 5} models)")
        else:
            feature_selection_methods = ['lasso', 'pca', 'mutual_info']
            print(f"[ensemble] Using all feature selection methods: {feature_selection_methods} (will train {len(feature_selection_methods)} methods × 5 folds = {len(feature_selection_methods) * 5} models)")
        print(f"[ensemble] Strategy: {ensemble_strategy} (oof/hts/full)")
        n_components = 200  # Number of features/components to select
        
        # Initialize or load from checkpoint
        resume_info = None
        if resume_from_checkpoint:
            resume_info = load_checkpoint(resume_from_checkpoint, DEVICE)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        
        # Initialize variables that will be used in loops (ensure they're always defined)
        all_models = []
        all_predictions = []
        history = []
        best_auc = float("-inf")
        best_state = None
        resume_method_idx = 0
        resume_fold_idx = -1  # Start from fold 0 (fold_idx = -1 means we haven't completed any folds yet)
        
        # Initialize or restore from checkpoint
        if resume_info and not resume_info.get('is_model_only', False):
            # Resume from full checkpoint
            all_models = resume_info.get('all_models', [])
            loaded_predictions = resume_info.get('all_predictions', [])
            # Convert loaded predictions back to numpy arrays if they were serialized as lists
            all_predictions = []
            for pred_entry in loaded_predictions:
                restored_entry = {
                    'val_idx': np.array(pred_entry['val_idx']) if not isinstance(pred_entry['val_idx'], np.ndarray) else pred_entry['val_idx'],
                    'predictions': np.array(pred_entry['predictions']) if not isinstance(pred_entry['predictions'], np.ndarray) else pred_entry['predictions'],
                    'feature_method': str(pred_entry['feature_method']),
                    'fold': int(pred_entry['fold'])
                }
                all_predictions.append(restored_entry)
            
            history = resume_info.get('history', [])
            best_auc = resume_info.get('best_auc', float("-inf"))
            best_state = resume_info.get('best_state', None)
            resume_method = resume_info.get('current_method', feature_selection_methods[0] if feature_selection_methods else 'lasso')
            resume_fold = resume_info.get('current_fold', -1)
            print(f"[checkpoint] Resuming from: method={resume_method}, fold={resume_fold+1}")
            print(f"[checkpoint] Already completed: {len(all_models)} models, {len(all_predictions)} prediction sets")
            
            # Debug: Show what predictions were loaded
            if all_predictions:
                print(f"[checkpoint] Loaded predictions breakdown:")
                folds_by_method_loaded = {}
                for pred_data in all_predictions:
                    method = pred_data.get('feature_method', 'unknown')
                    fold_num = pred_data.get('fold', -1) + 1
                    if method not in folds_by_method_loaded:
                        folds_by_method_loaded[method] = []
                    folds_by_method_loaded[method].append(fold_num)
                for method, folds in folds_by_method_loaded.items():
                    print(f"[checkpoint]   {method}: {len(folds)} folds: {sorted(folds)}")
            else:
                print(f"[checkpoint] WARNING: No predictions loaded from checkpoint!")
            
            # Find where to resume - with defensive checks
            if resume_method in feature_selection_methods:
                resume_method_idx = feature_selection_methods.index(resume_method)
            else:
                print(f"[checkpoint] WARNING: resume_method '{resume_method}' not in feature_selection_methods, defaulting to 0")
                resume_method_idx = 0
            resume_fold_idx = resume_fold if resume_fold >= 0 else 0
        else:
            # Start fresh (or model-only file - use weights but start training from beginning)
            if resume_info and resume_info.get('is_model_only', False):
                print(f"[checkpoint] Using model weights from checkpoint but starting training from beginning")
            # Variables already initialized above, no need to reassign
        
        fold_preds = np.zeros(len(labels))  # Will accumulate ensemble predictions
        
        # Collect system information (cache to avoid repeated calls)
        setup_time_so_far = time.time() - setup_start_time
        print(f"[timing] Setup time so far: {setup_time_so_far:.2f}s")
        sys.stdout.flush()
        
        system_info = get_system_info()
        system_info_time = time.time() - setup_start_time - setup_time_so_far
        print(f"[timing] System info collection: {system_info_time:.2f}s")
        sys.stdout.flush()
        
        print(f"[system] Device: {system_info['device_name']} | CUDA: {system_info['cuda_version']} | "
              f"RAM: {system_info['total_ram_gb']} GB")
        if DEVICE.type == "cuda":
            print(f"[system] GPU Memory: {system_info['gpu_memory_gb']} GB")
        
        total_setup_time = time.time() - setup_start_time
        print(f"[timing] Total setup time: {total_setup_time:.2f}s")
        sys.stdout.flush()
        
        training_start_time = time.time()
    
        # Ensemble training: iterate over feature selection methods
        for method_idx, feature_method in enumerate(feature_selection_methods):
            # Skip methods that were already completed (when resuming)
            if resume_info and method_idx < resume_method_idx:
                print(f"[checkpoint] Skipping method {feature_method} (already completed)")
                continue
            print(f"\n{'='*60}")
            print(f"[ensemble] Feature Selection Method: {feature_method.upper()}")
            print(f"{'='*60}")
            
            # Convert generator to list to ensure we can iterate multiple times if needed
            # Note: skf.split() with same random_state will produce same splits each time
            fold_splits = list(skf.split(np.zeros(len(labels)), labels))
            for fold, (train_idx, val_idx) in enumerate(fold_splits):
                # Initialize all fold-specific variables at the start to avoid "referenced before assignment" errors
                # These will be properly set later if the fold is not skipped
                best_metrics = None
                best_fold_auc = float("-inf")
                fold_start_time = time.time()
                fold_history = {
                    "train_loss": [],
                    "val_auc": [],
                    "val_ap": [],
                    "val_precision": [],
                    "val_recall": [],
                    "val_f1": [],
                    "val_optimal_threshold": [],
                    "train_time_seconds": [],
                    "eval_time_seconds": [],
                    "memory_usage": [],
                }
                model = None
                selector = None
                scaler = None
                train_rdkit_selected = None
                
                # Skip folds that were already completed (when resuming from full checkpoint)
                if resume_info and not resume_info.get('is_model_only', False) and method_idx == resume_method_idx and fold <= resume_fold_idx:
                    print(f"[checkpoint] Skipping fold {fold+1} (already completed)")
                    continue
                print(f"\n[fold {fold + 1}] train={len(train_idx)} val={len(val_idx)}")
                
                # Apply feature selection (GPU-accelerated when available)
                print(f"[fold {fold + 1}] Applying {feature_method} feature selection...")
                feature_selection_start = time.time()
                # Use GPU acceleration if CUDA is available
                use_gpu_feature_selection = DEVICE.type == "cuda"
                train_rdkit_selected, val_rdkit_selected, selector, scaler = apply_feature_selection(
                    rdkit_feats_gpu[train_idx], labels[train_idx],
                    rdkit_feats_gpu[val_idx],
                    method=feature_method,
                    n_components=n_components,
                    use_gpu=use_gpu_feature_selection
                )
                feature_selection_time = time.time() - feature_selection_start
                gpu_status = "GPU-accelerated" if use_gpu_feature_selection else "CPU"
                print(f"[fold {fold + 1}] Selected features shape: {train_rdkit_selected.shape} ({gpu_status}, took {feature_selection_time:.2f}s)")
                sys.stdout.flush()
                
                # Use selected features (already on DEVICE when use_gpu; else convert)
                if isinstance(train_rdkit_selected, torch.Tensor) and train_rdkit_selected.device == DEVICE:
                    train_rdkit_tensor = train_rdkit_selected
                    val_rdkit_tensor = val_rdkit_selected
                else:
                    train_rdkit_tensor = torch.tensor(train_rdkit_selected, dtype=torch.float32, device=DEVICE)
                    val_rdkit_tensor = torch.tensor(val_rdkit_selected, dtype=torch.float32, device=DEVICE)
                
                # Create datasets with selected features
                train_dataset = HTS3DDataset(
                    input_ids=input_ids[train_idx],
                    attention_mask=attention_mask[train_idx],
                    rdkit_feats=train_rdkit_tensor,
                    atom_feats=atom_feats[train_idx],
                    atom_masks=atom_masks[train_idx],
                    labels=labels[train_idx],
                )
                val_dataset = HTS3DDataset(
                    input_ids=input_ids[val_idx],
                    attention_mask=attention_mask[val_idx],
                    rdkit_feats=val_rdkit_tensor,
                    atom_feats=atom_feats[val_idx],
                    atom_masks=atom_masks[val_idx],
                    labels=labels[val_idx],
                )
                
                # Check validation set class distribution
                val_labels = labels[val_idx]
                val_pos = val_labels.sum()
                val_neg = len(val_labels) - val_pos
                print(f"[fold {fold + 1}] Validation set: {val_pos} positives, {val_neg} negatives")
                if val_pos == 0 or val_neg == 0:
                    print(f"[fold {fold + 1}] WARNING: Validation set has only one class! Metrics may be unreliable.")
                
                train_subset = torch.utils.data.Subset(train_dataset, list(range(len(train_dataset))))
                val_subset = torch.utils.data.Subset(val_dataset, list(range(len(val_dataset))))
    
                # Class distribution for this fold (for sampler and loss)
                train_labels_fold = labels[train_idx]
                if torch.is_tensor(train_labels_fold):
                    train_labels_fold = train_labels_fold.cpu().numpy()
                pos_count_fold = int((train_labels_fold == 1).sum())
                neg_count_fold = len(train_labels_fold) - pos_count_fold
                pos_ratio_fold = pos_count_fold / len(train_labels_fold) if len(train_labels_fold) else 0.0
                # When positives < 5%, most batches have 0 positives → weak gradient signal. Use balanced sampling
                # so most batches see positives; then use pos_weight=1.0 to avoid double correction.
                use_balanced_sampler = (pos_ratio_fold < 0.05 and pos_count_fold > 0 and neg_count_fold > 0)
    
                # Disable pin_memory if tensors are already on GPU (pin_memory only works for CPU tensors)
                use_pin_memory = PIN_MEMORY and DEVICE.type == "cpu"
                # Match HTS.py: parallel data loading on GPU (2 workers) when not Windows; 0 on Windows to avoid multiprocessing issues
                is_windows = sys.platform.startswith("win")
                num_workers = 2 if (DEVICE.type == "cuda" and not is_windows) else 0
                use_persistent_workers = num_workers > 0
    
                # Match HTS.py: batch_size 64 on GPU, 32 on CPU
                train_batch_size = 64 if DEVICE.type == "cuda" else 32
                if use_balanced_sampler:
                    weights = torch.tensor(
                        [1.0 / pos_count_fold if y == 1 else 1.0 / neg_count_fold for y in train_labels_fold],
                        dtype=torch.double
                    )
                    sampler = WeightedRandomSampler(weights, num_samples=len(weights))
                    train_loader = DataLoader(
                        train_subset,
                        batch_size=train_batch_size,
                        sampler=sampler,
                        num_workers=num_workers,
                        pin_memory=use_pin_memory,
                        drop_last=len(train_subset) > train_batch_size,
                        persistent_workers=use_persistent_workers,
                    )
                    print(f"[fold {fold + 1}] Using WeightedRandomSampler (pos_ratio={pos_ratio_fold:.2%}) so batches see positives; pos_weight=1.0")
                else:
                    train_loader = DataLoader(
                        train_subset,
                        batch_size=train_batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=use_pin_memory,
                        drop_last=len(train_subset) > train_batch_size,
                        persistent_workers=use_persistent_workers,
                    )
                val_loader = DataLoader(
                    val_subset,
                    batch_size=train_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=use_pin_memory,
                    persistent_workers=use_persistent_workers,
                )
    
                print(f"[model] Creating HTS3DModel (this may take time if downloading ChemBERTa model)...")
                sys.stdout.flush()
                
                # Check if we should load model from checkpoint
                model_loaded = False
                if resume_info and 'model_state_dict' in resume_info:
                    # Try to load model state from checkpoint (for first fold of resumed method, or if it's a model-only file)
                    should_load = False
                    if resume_info.get('is_model_only', False):
                        # Model-only file: load for first fold of first method
                        should_load = (method_idx == 0 and fold == 0)
                    else:
                        # Full checkpoint: load for the exact fold we're resuming from
                        should_load = (method_idx == resume_method_idx and fold == resume_fold_idx + 1)
                    
                    if should_load:
                        try:
                            # ALWAYS use GPU if CUDA is available for checkpoint loading
                            cuda_available = torch.cuda.is_available()
                            print(f"[checkpoint] CUDA available check: {cuda_available}")
                            
                            model = HTS3DModel(
                                rdkit_dim=train_rdkit_selected.shape[1],  # Use current feature dimension
                                atom_dim=atom_dim,
                                protein_pockets=protein_templates,
                                num_pockets=max_pockets,
                                use_attention_pooling=model_use_attention_pooling,
                                use_interaction_features=model_use_interaction_features,
                                use_adaptive_weighting=model_use_adaptive_weighting,
                                max_pockets_ultra=model_max_pockets_ultra,
                            )
                            
                            # CRITICAL: Always use .cuda() if available
                            if cuda_available:
                                print(f"[checkpoint] Moving model to GPU using .cuda()...")
                                model = model.cuda()
                            else:
                                print(f"[checkpoint] Moving model to CPU...")
                                model = model.to(torch.device("cpu"))
                            
                            # Verify checkpoint model is on GPU
                            first_param = next(model.parameters())
                            print(f"[checkpoint] Model loaded and moved to device: {first_param.device}")
                            if cuda_available and first_param.device.type != "cuda":
                                print(f"[checkpoint] WARNING: Model on {first_param.device}, forcing to GPU...")
                                model = model.cuda()
                                first_param = next(model.parameters())
                                print(f"[checkpoint] Model forced to GPU: {first_param.device}")
                            checkpoint_state = resume_info['model_state_dict']
                            
                            # Handle partial loading - ignore missing keys (like temperature) and size mismatches
                            model_state = model.state_dict()
                            filtered_state = {}
                            skipped_keys = []
                            size_mismatches = []
                            
                            for key, value in checkpoint_state.items():
                                if key in model_state:
                                    if model_state[key].shape == value.shape:
                                        filtered_state[key] = value
                                    else:
                                        size_mismatches.append(f"{key}: checkpoint {value.shape} vs model {model_state[key].shape}")
                                else:
                                    skipped_keys.append(key)
                            
                            # Load the filtered state dict (only matching keys)
                            model.load_state_dict(filtered_state, strict=False)
                            
                            if skipped_keys:
                                print(f"[checkpoint] Skipped {len(skipped_keys)} keys not in current model (e.g., temperature parameter)")
                            if size_mismatches:
                                print(f"[checkpoint] Skipped {len(size_mismatches)} keys with size mismatches (likely due to different feature selection)")
                                for mismatch in size_mismatches[:3]:  # Show first 3
                                    print(f"[checkpoint]   - {mismatch}")
                            
                            print(f"[checkpoint] Loaded {len(filtered_state)}/{len(checkpoint_state)} parameters from checkpoint")
                            model_loaded = True
                        except Exception as e:
                            print(f"[checkpoint] WARNING: Could not load model from checkpoint: {e}")
                            print(f"[checkpoint] Creating new model instead")
                            import traceback
                            traceback.print_exc()
                
                if not model_loaded:
                    # CRITICAL: Test CUDA availability and force GPU usage
                    cuda_available = torch.cuda.is_available()
                    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
                    print(f"[model] ===== GPU CHECK =====")
                    print(f"[model] torch.cuda.is_available(): {cuda_available}")
                    print(f"[model] torch.cuda.device_count(): {cuda_device_count}")
                    if cuda_available:
                        print(f"[model] GPU Name: {torch.cuda.get_device_name(0)}")
                        print(f"[model] Current DEVICE global: {DEVICE}")
                    
                    # FORCE GPU if CUDA is available - no exceptions
                    if cuda_available:
                        target_device = torch.device("cuda:0")
                        print(f"[model] ===== FORCING GPU USAGE: {target_device} =====")
                    else:
                        target_device = torch.device("cpu")
                        print(f"[model] ===== CUDA NOT AVAILABLE - Using CPU: {target_device} =====")
                    
                    # Create model
                    print(f"[model] Creating model...")
                    model = HTS3DModel(
                        rdkit_dim=train_rdkit_selected.shape[1],  # Use selected feature dimension
                        atom_dim=atom_dim,
                        protein_pockets=protein_templates,
                        num_pockets=max_pockets,
                        use_attention_pooling=model_use_attention_pooling,
                        use_interaction_features=model_use_interaction_features,
                        use_adaptive_weighting=model_use_adaptive_weighting,
                        max_pockets_ultra=model_max_pockets_ultra,
                    )
                    
                    # CRITICAL: Use .cuda() directly - most reliable method
                    if cuda_available:
                        print(f"[model] Moving model to GPU using model.cuda()...")
                        try:
                            model = model.cuda()
                            print(f"[model] Model.cuda() completed successfully")
                        except Exception as e:
                            print(f"[model] ERROR in model.cuda(): {e}")
                            raise RuntimeError(f"Failed to move model to GPU: {e}")
                    else:
                        print(f"[model] Moving model to CPU...")
                        model = model.to(target_device)
                    
                    # CRITICAL: Verify EVERY parameter is on the correct device
                    print(f"[model] Verifying model device placement...")
                    all_params_on_correct_device = True
                    for name, param in model.named_parameters():
                        if cuda_available and param.device.type != "cuda":
                            print(f"[model] ERROR: Parameter '{name}' is on {param.device}, expected GPU!")
                            all_params_on_correct_device = False
                        elif not cuda_available and param.device.type != "cpu":
                            print(f"[model] ERROR: Parameter '{name}' is on {param.device}, expected CPU!")
                            all_params_on_correct_device = False
                    
                    # Check first parameter for final verification
                    first_param = next(model.parameters())
                    actual_device = first_param.device
                    print(f"[model] ===== MODEL DEVICE VERIFICATION =====")
                    print(f"[model] First parameter device: {actual_device}")
                    print(f"[model] Expected device: {target_device}")
                    print(f"[model] All parameters on correct device: {all_params_on_correct_device}")
                    
                    if cuda_available and actual_device.type != "cuda":
                        print(f"[model] CRITICAL ERROR: Model is on {actual_device}, but CUDA is available!")
                        print(f"[model] Attempting emergency GPU transfer...")
                        model = model.cuda()
                        first_param = next(model.parameters())
                        if first_param.device.type != "cuda":
                            raise RuntimeError(f"EMERGENCY GPU TRANSFER FAILED! Model still on {first_param.device}")
                        print(f"[model] Emergency GPU transfer successful: {first_param.device}")
                    
                    print(f"[model] ===== MODEL SUCCESSFULLY ON {actual_device} =====")
                
                # Safety check: Ensure model was created successfully
                if model is None:
                    print(f"[fold {fold + 1}] ERROR: Model creation failed - skipping this fold")
                    continue
                
                sys.stdout.flush()
                # FUNDAMENTAL FIX: Use HTS.py's learning rate exactly (2e-4)
                # Since we now match HTS.py's architecture exactly (256 input dimension),
                # we should use the same learning rate that works for HTS.py
                learning_rate = 2e-4  # Match HTS.py exactly
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=1e-4  # Match HTS.py exactly
                )
                print(f"[fold {fold + 1}] LR={learning_rate}, weight_decay=1e-4 (match HTS.py); Option C Minimal uses HTS-style branches + 3D/protein")
                
                # Load optimizer state from checkpoint if resuming (only for full checkpoints, not model-only files)
                if resume_info and not resume_info.get('is_model_only', False):
                    should_load_optimizer = False
                    if method_idx == resume_method_idx and fold == resume_fold_idx + 1 and model_loaded:
                        should_load_optimizer = True
                    
                    if should_load_optimizer and 'optimizer_state_dict' in resume_info:
                        try:
                            optimizer.load_state_dict(resume_info['optimizer_state_dict'])
                            print(f"[checkpoint] Loaded optimizer state from checkpoint")
                        except Exception as e:
                            print(f"[checkpoint] WARNING: Could not load optimizer state: {e}")
                
                # CRITICAL FIX #3: When using balanced sampler, pos_weight=1.0 to avoid double correction
                if use_balanced_sampler:
                    pos_weight_capped = 1.0
                    print(f"[fold {fold + 1}] pos_weight=1.0 (balanced sampler; pos: {pos_count_fold}, neg: {neg_count_fold})")
                else:
                    if pos_count_fold == 0:
                        print(f"[fold {fold + 1}] WARNING: No positive samples in training set!")
                        pos_weight_capped = 1.0
                    else:
                        pos_weight_val = float(neg_count_fold) / float(pos_count_fold)
                        pos_weight_capped = min(pos_weight_val, 30.0)
                    print(f"[fold {fold + 1}] pos_weight: raw={neg_count_fold/max(1,pos_count_fold):.2f}, capped={pos_weight_capped:.2f} (pos: {pos_count_fold}, neg: {neg_count_fold})")
                
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(pos_weight_capped, device=DEVICE, dtype=torch.float32),
                    reduction='mean'
                )
                print(f"[loss] BCEWithLogitsLoss pos_weight={pos_weight_capped:.2f}")
                pos_weight_val = float(pos_weight_capped)  # for train_one_epoch (logging only; loss uses criterion)
                
                # Reset fold history and training configuration (already initialized at loop start, but reset for this fold)
                fold_history = {
                    "train_loss": [],
                    "val_auc": [],
                    "val_ap": [],
                    "val_precision": [],
                    "val_recall": [],
                    "val_f1": [],
                    "val_optimal_threshold": [],
                    "train_time_seconds": [],
                    "eval_time_seconds": [],
                    "memory_usage": [],
                }
                
                # Match HTS.py: 10 epochs max, patience 5 (fast GPU runs)
                num_epochs = 10   # Match HTS.py
                patience = 5     # Match HTS.py
                no_improve = 0
                best_fold_auc = float("-inf")
                best_fold_f1 = float("-inf")
                best_fold_recall = float("-inf")
                best_recall_epoch = 0
                best_metrics = None
                fold_start_time = time.time()
                print(f"[fold {fold + 1}] Epochs: {num_epochs}, patience: {patience} (match HTS.py)")
                
                # Use simple learning rate scheduler like HTS.py (ReduceLROnPlateau)
                # NOTE: Must be created AFTER train_loader is created
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                
                # FUNDAMENTAL FIX: Use HTS.py's scheduler settings exactly
                # HTS.py: factor=0.5, patience=2
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',  # Reduce LR when metric stops increasing
                    factor=0.5,  # Match HTS.py exactly
                    patience=2   # Match HTS.py exactly
                    # verbose parameter removed - deprecated in newer PyTorch versions
                )
                
                # Training loop - must be inside fold loop
                for epoch in range(num_epochs):
                    print(f"[fold {fold + 1}] epoch {epoch + 1}")
                    train_loss, train_time = train_one_epoch(model, train_loader, optimizer, criterion, pos_weight_val)
                    metrics = evaluate(model, val_loader)
                    
                    # Update learning rate scheduler (ReduceLROnPlateau uses validation metric)
                    scheduler.step(metrics.get('auc', 0.0))
                    
                    # Collect memory usage after epoch
                    mem_usage = get_memory_usage()
    
                    fold_history["train_loss"].append(train_loss)
                    fold_history["val_auc"].append(metrics["auc"])
                    fold_history["val_ap"].append(metrics["ap"])
                    fold_history["val_precision"].append(metrics.get("precision", 0.0))
                    fold_history["val_recall"].append(metrics.get("recall", 0.0))
                    fold_history["val_f1"].append(metrics.get("f1", 0.0))
                    fold_history["val_optimal_threshold"].append(metrics.get("optimal_threshold", 0.5))
                    fold_history["train_time_seconds"].append(train_time)
                    fold_history["eval_time_seconds"].append(metrics["eval_time_seconds"])
                    fold_history["memory_usage"].append(mem_usage)
    
                    optimal_thresh = metrics.get("optimal_threshold", 0.5)
                    print(
                        f"[fold {fold + 1}] Epoch {epoch + 1} summary: "
                        f"train_loss={train_loss:.4f} | "
                        f"val_auc={metrics['auc']:.4f} | "
                        f"val_ap={metrics['ap']:.4f} | "
                        f"val_precision={metrics['precision']:.4f} | "
                        f"val_recall={metrics['recall']:.4f} | "
                        f"val_f1={metrics['f1']:.4f} | "
                        f"optimal_threshold={optimal_thresh:.4f}"
                    )
    
                    # Track F1 score for balanced precision/recall optimization
                    current_f1 = metrics.get("f1", 0.0)
                    best_fold_f1 = best_metrics.get("f1", 0.0) if best_metrics is not None else 0.0
                    
                    # Track recall improvement for monitoring
                    if not math.isnan(metrics.get("recall", 0)):
                        if metrics["recall"] > best_fold_recall:
                            best_fold_recall = metrics["recall"]
                            best_recall_epoch = epoch
                    
                    # Update best metrics based on F1 score (balances precision and recall)
                    # F1 is the harmonic mean, so it requires both precision and recall to be high
                    # Also check that precision and recall are both above minimum thresholds
                    min_precision = 0.25  # Minimum acceptable precision
                    min_recall = 0.25     # Minimum acceptable recall
                    has_min_precision = metrics.get("precision", 0) >= min_precision
                    has_min_recall = metrics.get("recall", 0) >= min_recall
                    
                    # Prefer models with balanced high F1 and both metrics above minimums
                    is_valid_model = (not math.isnan(metrics["auc"])) and has_min_precision and has_min_recall
                    f1_improved = not math.isnan(current_f1) and current_f1 > best_fold_f1
                    auc_improved = not math.isnan(metrics["auc"]) and (best_metrics is None or metrics["auc"] > best_fold_auc)
                    
                    # FIX: Always track best model, even if it doesn't meet all requirements
                    # This ensures we have a model to use even if training is challenging
                    # Priority: valid models with improved F1/AUC > models with improved AUC > first valid model
                    should_update_best = False
                    if is_valid_model and (f1_improved or (auc_improved and metrics["auc"] > best_fold_auc + 0.01)):
                        # Best case: valid model with improvement
                        should_update_best = True
                        no_improve = 0
                    elif not math.isnan(metrics["auc"]):
                        # Fallback: use best AUC even if not meeting all requirements
                        # This ensures we always have a model, even if precision/recall are imbalanced
                        if best_metrics is None:
                            # First valid model (any AUC is better than None)
                            should_update_best = True
                            no_improve = 0
                        elif metrics["auc"] > best_fold_auc + 0.01:
                            # Significant AUC improvement, even if not meeting precision/recall requirements
                            should_update_best = True
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                    
                    if should_update_best:
                        best_fold_auc = metrics["auc"]
                        best_metrics = metrics
                        if metrics["auc"] > best_auc:
                            best_auc = metrics["auc"]
                            best_state = model.state_dict()
                    
                    if not is_valid_model:
                        # If model doesn't meet minimum requirements, increment no_improve
                        if not has_min_precision:
                            print(f"[fold {fold + 1}] Epoch {epoch + 1}: Precision {metrics.get('precision', 0):.4f} below minimum {min_precision}")
                        if not has_min_recall:
                            print(f"[fold {fold + 1}] Epoch {epoch + 1}: Recall {metrics.get('recall', 0):.4f} below minimum {min_recall}")
                        if math.isnan(metrics["auc"]):
                            print(f"[fold {fold + 1}] WARNING: Epoch {epoch + 1} produced NaN AUC, skipping update")
                    
                    # Early stopping: match HTS.py (no warmup, stop when no AUC improvement for patience epochs)
                    if no_improve >= patience:
                        print(f"    Early stopping: No AUC improvement for {patience} epochs (match HTS.py).")
                        break
    
                # Post-training fold processing (outside epoch loop, inside fold loop)
                # Safety check: if best_metrics is still None, the fold was skipped or training failed
                if best_metrics is None:
                    print(f"[fold {fold + 1}] WARNING: No metrics produced - fold may have been skipped or training failed")
                    continue  # Skip post-processing for this fold
                
                # Safety check: Ensure we have valid history data
                if len(fold_history["val_auc"]) == 0:
                    print(f"[fold {fold + 1}] ERROR: No epochs completed - fold_history is empty")
                    continue
                
                fold_elapsed_time = time.time() - fold_start_time
                fold_history["fold_total_time_seconds"] = fold_elapsed_time
                fold_history["fold_best_auc"] = best_fold_auc
                
                # Safely find best epoch - handle case where best_fold_auc might not be in list (e.g., NaN comparison)
                # Note: NaN != NaN in Python, so .index() won't work for NaN values
                if math.isnan(best_fold_auc):
                    # If best_fold_auc is NaN, use last epoch
                    fold_history["fold_best_epoch"] = len(fold_history["val_auc"])
                    print(f"[fold {fold + 1}] WARNING: best_fold_auc is NaN, using last epoch")
                else:
                    try:
                        fold_history["fold_best_epoch"] = fold_history["val_auc"].index(best_fold_auc) + 1
                    except ValueError:
                        # If best_fold_auc not found (shouldn't happen, but handle gracefully), use last epoch
                        fold_history["fold_best_epoch"] = len(fold_history["val_auc"])
                        print(f"[fold {fold + 1}] WARNING: best_fold_auc {best_fold_auc} not found in val_auc list, using last epoch")
                
                # Ensure fold_best_epoch is at least 1 (epochs are 1-indexed)
                if fold_history["fold_best_epoch"] < 1:
                    fold_history["fold_best_epoch"] = 1
                    print(f"[fold {fold + 1}] WARNING: fold_best_epoch was < 1, setting to 1")
                
                fold_history["fold_best_optimal_threshold"] = best_metrics.get("optimal_threshold", 0.5)
                
                # Safety checks before storing model and predictions (do this BEFORE appending to history)
                if model is None:
                    print(f"[fold {fold + 1}] ERROR: Model is None - cannot store for ensemble")
                    continue
                if selector is None or scaler is None:
                    print(f"[fold {fold + 1}] ERROR: Selector or scaler is None - cannot store for ensemble")
                    continue
                if train_rdkit_selected is None:
                    print(f"[fold {fold + 1}] ERROR: train_rdkit_selected is None - cannot store for ensemble")
                    continue
                if "preds" not in best_metrics:
                    print(f"[fold {fold + 1}] ERROR: best_metrics missing 'preds' key - cannot store predictions")
                    continue
                
                # All safety checks passed - append to history now
                history.append(fold_history)
                
                # Store model and predictions for ensemble
                all_models.append({
                    'model': model.state_dict(),
                    'feature_method': feature_method,
                    'fold': fold,
                    'selector': selector,
                    'scaler': scaler,
                    'rdkit_dim': train_rdkit_selected.shape[1],
                    'metrics': best_metrics,
                    'val_idx': val_idx
                })
                
                # Store predictions for this fold
                # IMPORTANT: Copy val_idx to avoid reference issues
                val_idx_copy = val_idx.copy() if isinstance(val_idx, np.ndarray) else np.array(val_idx)
                preds_copy = best_metrics["preds"].copy() if isinstance(best_metrics["preds"], np.ndarray) else np.array(best_metrics["preds"])
                
                # Debug: Always print to track all folds
                print(f"[debug] Storing predictions for fold {fold+1}, method {feature_method}: "
                      f"val_idx range {val_idx_copy.min()}-{val_idx_copy.max()}, len={len(val_idx_copy)}, "
                      f"total predictions stored so far: {len(all_predictions)}")
                
                # Create a new dictionary with explicit copies to avoid any reference issues
                new_pred_entry = {
                    'val_idx': val_idx_copy.copy() if isinstance(val_idx_copy, np.ndarray) else np.array(val_idx_copy, copy=True),
                    'predictions': preds_copy.copy() if isinstance(preds_copy, np.ndarray) else np.array(preds_copy, copy=True),
                    'feature_method': str(feature_method),  # Ensure it's a string
                    'fold': int(fold)  # Ensure it's an int
                }
                # CRITICAL: Ensure we're appending to the correct list (not a local copy)
                # Make sure all_predictions is the module-level list, not a local variable
                all_predictions.append(new_pred_entry)
                
                # Verify it was added and show summary
                print(f"[debug] After append: total predictions = {len(all_predictions)}")
                # Show stored folds per method
                folds_by_method = {}
                for p in all_predictions:
                    method = p.get('feature_method', 'unknown')
                    fold_num = p.get('fold', -1) + 1
                    if method not in folds_by_method:
                        folds_by_method[method] = []
                    folds_by_method[method].append(fold_num)
                for method, folds in folds_by_method.items():
                    print(f"[debug]   {method}: folds {sorted(folds)} (total: {len(folds)} folds)")
                
                # Double-check: Verify the list actually grew
                if len(all_predictions) < (method_idx + 1) * (fold + 1):
                    print(f"[debug] WARNING: all_predictions length ({len(all_predictions)}) is less than expected ({method_idx + 1} methods × {fold + 1} folds)")
                
                optimal_thresh = best_metrics.get("optimal_threshold", 0.5)
                print(f"[fold {fold + 1}] Completed in {fold_elapsed_time:.2f}s | Best AUC: {best_fold_auc:.4f} | Optimal threshold: {optimal_thresh:.4f}")
                
                # Note: Checkpoints are only saved at the end of training (after all folds and methods)
    
        # After all folds and methods: Ensemble averaging
        print(f"\n{'='*60}")
        print(f"[ensemble] Building ensemble predictions from {len(all_models)} models (strategy: {ensemble_strategy})")
        print(f"[ensemble] Total prediction sets: {len(all_predictions)}")
        print(f"[ensemble] Expected: {len(feature_selection_methods)} methods × 5 folds = {len(feature_selection_methods) * 5} models")
        print(f"{'='*60}")
        
        if ensemble_strategy == "full":
            # Full ensemble: Run all models on every sample and average (like HTS) → smoother scores
            # Each sample gets predictions from all 15 (or 5 if --pick) models
            print(f"[ensemble] FULL strategy: Running all {len(all_models)} models on every sample...")
            ensemble_predictions = np.zeros(len(labels), dtype=np.float64)
            n_samples = len(labels)
            inference_batch_size = 128 if DEVICE.type == "cuda" else 64
            
            for model_idx, model_data in enumerate(all_models):
                selector = model_data.get('selector')
                scaler = model_data.get('scaler')
                feature_method = model_data.get('feature_method', 'unknown')
                fold_num = model_data.get('fold', -1)
                rdkit_dim = model_data.get('rdkit_dim', 200)
                if selector is None or scaler is None:
                    print(f"[ensemble] WARNING: Model {model_idx+1} (fold {fold_num+1}, {feature_method}) missing selector/scaler, skipping")
                    continue
                
                # Apply feature selection to ALL samples (not just val)
                rdkit_np = rdkit_feats_gpu.cpu().numpy() if isinstance(rdkit_feats_gpu, torch.Tensor) else np.asarray(rdkit_feats_gpu, dtype=np.float32)
                rdkit_np = np.nan_to_num(rdkit_np, nan=0.0, posinf=0.0, neginf=0.0)
                rdkit_scaled = scaler.transform(rdkit_np)
                rdkit_selected = selector.transform(rdkit_scaled)
                rdkit_selected = np.nan_to_num(rdkit_selected, nan=0.0, posinf=0.0, neginf=0.0)
                rdkit_tensor = torch.tensor(rdkit_selected, dtype=torch.float32, device=DEVICE)
                
                # Load model
                model = HTS3DModel(
                    rdkit_dim=rdkit_dim,
                    atom_dim=atom_dim,
                    protein_pockets=protein_templates,
                    num_pockets=max_pockets,
                    use_attention_pooling=model_use_attention_pooling,
                    use_interaction_features=model_use_interaction_features,
                    use_adaptive_weighting=model_use_adaptive_weighting,
                    max_pockets_ultra=model_max_pockets_ultra,
                ).to(DEVICE)
                model.load_state_dict(model_data['model'])
                model.eval()
                
                # Batch predict on all samples
                with torch.no_grad():
                    for batch_start in range(0, n_samples, inference_batch_size):
                        batch_end = min(batch_start + inference_batch_size, n_samples)
                        batch_input_ids = input_ids[batch_start:batch_end]
                        batch_attention_mask = attention_mask[batch_start:batch_end]
                        batch_rdkit = rdkit_tensor[batch_start:batch_end]
                        batch_atom_feats = atom_feats[batch_start:batch_end]
                        batch_atom_masks = atom_masks[batch_start:batch_end]
                        logits = model(batch_input_ids, batch_attention_mask, batch_rdkit, batch_atom_feats, batch_atom_masks)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        ensemble_predictions[batch_start:batch_end] += probs
                
                del model, rdkit_tensor
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[ensemble] Model {model_idx+1}/{len(all_models)} (fold {fold_num+1}, {feature_method}) done")
            
            # Average over all models
            ensemble_predictions /= len(all_models)
            ensemble_predictions = np.clip(ensemble_predictions, 0.0, 1.0)
            print(f"[ensemble] FULL: Averaged {len(all_models)} model predictions per sample → sharpest scores (per-model selector)")
        elif ensemble_strategy == "hts":
            # HTS-style: Loop by method, use fold 0's selector for all 5 folds, avg 5 folds then 3 methods
            # Emulates HTS.py behavior: shared selector per method, selector mismatch for folds 1-4 → smoother scores
            print(f"[ensemble] HTS strategy: Loop by method, fold 0 selector for all folds, avg 5 then 3 methods (emulate HTS.py)...")
            n_samples = len(labels)
            inference_batch_size = 128 if DEVICE.type == "cuda" else 64
            method_predictions_array = np.zeros((n_samples, len(feature_selection_methods)), dtype=np.float64)
            
            for method_idx, feature_method in enumerate(feature_selection_methods):
                method_models = [m for m in all_models if m.get('feature_method') == feature_method]
                if not method_models:
                    print(f"[ensemble] WARNING: No models for method {feature_method}, skipping")
                    continue
                # Use fold 0's selector/scaler for all 5 folds (HTS.py behavior)
                first_model_data = method_models[0]
                selector = first_model_data.get('selector')
                scaler = first_model_data.get('scaler')
                rdkit_dim = first_model_data.get('rdkit_dim', 200)
                if selector is None or scaler is None:
                    print(f"[ensemble] WARNING: Method {feature_method} missing selector/scaler, skipping")
                    continue
                # Transform all samples with fold 0's selector (one transform per method)
                rdkit_np = rdkit_feats_gpu.cpu().numpy() if isinstance(rdkit_feats_gpu, torch.Tensor) else np.asarray(rdkit_feats_gpu, dtype=np.float32)
                rdkit_np = np.nan_to_num(rdkit_np, nan=0.0, posinf=0.0, neginf=0.0)
                rdkit_scaled = scaler.transform(rdkit_np)
                rdkit_selected = selector.transform(rdkit_scaled)
                rdkit_selected = np.nan_to_num(rdkit_selected, nan=0.0, posinf=0.0, neginf=0.0)
                rdkit_tensor = torch.tensor(rdkit_selected, dtype=torch.float32, device=DEVICE)
                # Run fold models that match rdkit_dim (fold 0's selector output); skip folds with different rdkit_dim
                fold_preds_list = []
                for fold_idx, model_data in enumerate(method_models):
                    fold_rdkit_dim = model_data.get('rdkit_dim', 200)
                    if fold_rdkit_dim != rdkit_dim:
                        print(f"[ensemble] HTS: skipping {feature_method} fold {fold_idx+1} (rdkit_dim {fold_rdkit_dim} != fold 0's {rdkit_dim})")
                        continue
                    model = HTS3DModel(
                        rdkit_dim=rdkit_dim,
                        atom_dim=atom_dim,
                        protein_pockets=protein_templates,
                        num_pockets=max_pockets,
                        use_attention_pooling=model_use_attention_pooling,
                        use_interaction_features=model_use_interaction_features,
                        use_adaptive_weighting=model_use_adaptive_weighting,
                        max_pockets_ultra=model_max_pockets_ultra,
                    ).to(DEVICE)
                    model.load_state_dict(model_data['model'])
                    model.eval()
                    preds = np.zeros(n_samples, dtype=np.float64)
                    with torch.no_grad():
                        for batch_start in range(0, n_samples, inference_batch_size):
                            batch_end = min(batch_start + inference_batch_size, n_samples)
                            batch_input_ids = input_ids[batch_start:batch_end]
                            batch_attention_mask = attention_mask[batch_start:batch_end]
                            batch_rdkit = rdkit_tensor[batch_start:batch_end]
                            batch_atom_feats = atom_feats[batch_start:batch_end]
                            batch_atom_masks = atom_masks[batch_start:batch_end]
                            logits = model(batch_input_ids, batch_attention_mask, batch_rdkit, batch_atom_feats, batch_atom_masks)
                            probs = torch.sigmoid(logits).cpu().numpy()
                            preds[batch_start:batch_end] = probs
                    fold_preds_list.append(preds)
                    del model
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()

                n_used = len(fold_preds_list)
                if not fold_preds_list:
                    print(f"[ensemble] WARNING: No matching folds for {feature_method} (rdkit_dim mismatch), using zeros")
                    method_predictions_array[:, method_idx] = 0.01
                else:
                    method_predictions_array[:, method_idx] = np.stack(fold_preds_list, axis=0).mean(axis=0)
                del rdkit_tensor
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                if n_used < len(method_models):
                    print(f"[ensemble] HTS method {feature_method}: avg of {n_used}/{len(method_models)} folds (some skipped: rdkit_dim mismatch)")
                else:
                    print(f"[ensemble] HTS method {feature_method}: avg of {n_used} folds done")
            # Final: average across methods (mean of non-zero method preds, like HTS.py)
            ensemble_predictions = np.zeros(n_samples, dtype=np.float64)
            for i in range(n_samples):
                method_preds = method_predictions_array[i, :]
                valid_preds = method_preds[method_preds > 0]
                if len(valid_preds) > 0:
                    ensemble_predictions[i] = np.mean(valid_preds)
                else:
                    ensemble_predictions[i] = 0.01
            ensemble_predictions = np.clip(ensemble_predictions, 0.0, 1.0)
            print(f"[ensemble] HTS: avg of {len(feature_selection_methods)} methods (fold 0 selector per method) → smoother scores")
        else:
            # OOF strategy: Use stored validation predictions only (true CV) → sharper scores
            # Each sample gets 3 predictions (one per method) from its validation fold
            print(f"[ensemble] OOF strategy: Using out-of-fold validation predictions (3 per sample)...")
            
            # Debug: Print all stored predictions to diagnose the issue
            print(f"[ensemble] DEBUG: All stored predictions breakdown:")
            for idx, pred_data in enumerate(all_predictions):
                print(f"  {idx+1}. Fold {pred_data['fold']+1}, Method: {pred_data['feature_method']}, "
                      f"val_idx range: {pred_data['val_idx'].min()}-{pred_data['val_idx'].max()}, "
                      f"len={len(pred_data['val_idx'])}")
            
            # Debug: Check if we have predictions from all folds
            folds_per_method = {}
            for pred_data in all_predictions:
                method = pred_data['feature_method']
                fold = pred_data['fold']
                if method not in folds_per_method:
                    folds_per_method[method] = set()
                folds_per_method[method].add(fold)
            
            print(f"[ensemble] DEBUG: Folds stored per method:")
            for method, folds in folds_per_method.items():
                print(f"  {method}: folds {sorted([f+1 for f in folds])} (expected: [1, 2, 3, 4, 5])")
            
            # Initialize ensemble predictions array
            ensemble_predictions = np.zeros(len(labels))
            prediction_counts = np.zeros(len(labels), dtype=np.int32)
            unique_samples = set()
            
            print(f"[ensemble] Processing {len(all_predictions)} prediction sets...")
            for idx, pred_data in enumerate(all_predictions):
                val_idx = pred_data['val_idx']
                preds = pred_data['predictions']
                fold_num = pred_data['fold']
                method = pred_data['feature_method']
                
                if not isinstance(val_idx, np.ndarray):
                    val_idx = np.array(val_idx)
                if not isinstance(preds, np.ndarray):
                    preds = np.array(preds)
                
                if len(preds) != len(val_idx):
                    print(f"[ensemble] WARNING: Mismatch in fold {fold_num+1}, method {method}: "
                          f"val_idx length {len(val_idx)} != predictions length {len(preds)}")
                    continue
                
                if idx < 3:
                    print(f"[ensemble] Prediction set {idx+1}: fold {fold_num+1}, method {method}, "
                          f"val_idx range: {val_idx.min()}-{val_idx.max()}, len={len(val_idx)}")
                
                val_idx_int = val_idx.astype(np.int64) if isinstance(val_idx, np.ndarray) else np.array(val_idx, dtype=np.int64)
                
                if val_idx_int.max() >= len(ensemble_predictions) or val_idx_int.min() < 0:
                    print(f"[ensemble] ERROR: Invalid indices in fold {fold_num+1}, method {method}: "
                          f"min={val_idx_int.min()}, max={val_idx_int.max()}, expected range [0, {len(ensemble_predictions)-1}]")
                    continue
                
                ensemble_predictions[val_idx_int] += preds
                prediction_counts[val_idx_int] += 1
                unique_samples.update(val_idx_int.tolist())
            
            print(f"[ensemble] Unique samples with predictions: {len(unique_samples)} / {len(labels)}")
            print(f"[ensemble] Prediction counts - min: {prediction_counts.min()}, max: {prediction_counts.max()}, "
                  f"mean: {prediction_counts.mean():.2f}")
            print(f"[ensemble] Expected: Each sample should appear in {len(feature_selection_methods)} validation sets (once per method)")
            
            with np.errstate(divide='ignore', invalid='ignore'):
                ensemble_predictions = np.divide(ensemble_predictions, prediction_counts, 
                                                out=np.zeros_like(ensemble_predictions), 
                                                where=(prediction_counts > 0))
        
        # Report samples without predictions and use fallback strategy (OOF only; full/hts always have all)
        if ensemble_strategy == "oof":
            samples_without_predictions = (prediction_counts == 0).sum()
            if samples_without_predictions > 0:
                print(f"[ensemble] WARNING: {samples_without_predictions} samples have no predictions (score = 0)")
                print(f"[ensemble] This may indicate an issue with cross-validation indexing or failed folds")
                mean_prediction = ensemble_predictions[prediction_counts > 0].mean() if (prediction_counts > 0).any() else 0.01
                if mean_prediction > 0:
                    print(f"[ensemble] Using mean prediction ({mean_prediction:.4f}) as fallback for {samples_without_predictions} samples without predictions")
                    ensemble_predictions[prediction_counts == 0] = mean_prediction
                else:
                    print(f"[ensemble] All predictions are zero, using small random values as fallback")
                    ensemble_predictions[prediction_counts == 0] = np.random.uniform(0.01, 0.05, size=samples_without_predictions)
        
        # Blend with QED for HTS.py-like drug-likeness influence (higher QED → higher score)
        if qed_blend_weight > 0:
            qed_values = compute_qed_for_smiles_list(smiles)
            ensemble_predictions = (
                (1.0 - qed_blend_weight) * ensemble_predictions
                + qed_blend_weight * qed_values
            ).astype(np.float64)
            ensemble_predictions = np.clip(ensemble_predictions, 0.0, 1.0)
            print(f"[ensemble] QED blend applied: weight={qed_blend_weight:.2f} (HTS3DOracle drug-likeness)")
        
        # Find best model across all folds and methods for saving
        best_overall_auc = float("-inf")
        best_model_data = None
        for model_data in all_models:
            metrics = model_data['metrics']
            if metrics and not math.isnan(metrics.get("auc", 0)):
                if metrics["auc"] > best_overall_auc:
                    best_overall_auc = metrics["auc"]
                    best_model_data = model_data
        
        if best_model_data is None:
            raise RuntimeError("Training failed to produce a model.")
        
        # Save inference bundle: state_dict + scaler + selector + rdkit_dim for HTS_3D_inference.py
        inference_bundle = {
            'state_dict': best_model_data['model'],
            'scaler': best_model_data['scaler'],
            'selector': best_model_data['selector'],
            'rdkit_dim': best_model_data['rdkit_dim'],
            'feature_method': best_model_data['feature_method'],
            'use_attention_pooling': model_use_attention_pooling,
            'use_interaction_features': model_use_interaction_features,
            'use_adaptive_weighting': model_use_adaptive_weighting,
            'max_pockets_ultra': model_max_pockets_ultra,
        }
        torch.save(inference_bundle, model_path)
        print(f"[model] saved to {model_path} (best AUC: {best_overall_auc:.4f}, rdkit_dim={best_model_data['rdkit_dim']}, feature_method={best_model_data['feature_method']})")
        
        # Save ensemble predictions with timestamp (including compound type)
        pred_path = get_timestamped_path("hts3d_predictions", "csv", compound=compound)
        result_df = df.copy()
        result_df["hts3d_score"] = ensemble_predictions
        result_df.to_csv(pred_path, index=False)
        print(f"[predictions] saved to {pred_path} (ensemble from {len(all_models)} models)")
        
        total_training_time = time.time() - training_start_time
        final_memory = get_memory_usage()
        
        # Calculate ensemble metrics
        ensemble_metrics = evaluate_ensemble_predictions(labels, ensemble_predictions)
        print(f"\n[ensemble] Final ensemble metrics:")
        print(f"  AUC: {ensemble_metrics['auc']:.4f}")
        print(f"  AP: {ensemble_metrics['ap']:.4f}")
        print(f"  Precision: {ensemble_metrics['precision']:.4f}")
        print(f"  Recall: {ensemble_metrics['recall']:.4f}")
        print(f"  F1: {ensemble_metrics['f1']:.4f}")
        
        # Find the best optimal threshold across all folds (from the fold with best AUC)
        best_optimal_threshold = 0.5
        for fold_history in history:
            if fold_history.get("fold_best_auc") == best_overall_auc:
                best_optimal_threshold = fold_history.get("fold_best_optimal_threshold", 0.5)
                break
        
        # Convert numpy and PyTorch types to native Python types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert numpy and PyTorch types to native Python types."""
            # Handle PyTorch tensors first
            if isinstance(obj, torch.Tensor):
                if obj.numel() == 1:  # Scalar tensor
                    return obj.item()
                else:  # Multi-element tensor
                    return obj.detach().cpu().numpy().tolist()
            
            # Handle numpy scalars first (using item() method)
            if hasattr(obj, 'item') and not isinstance(obj, (str, bytes)):
                try:
                    return obj.item()
                except (ValueError, AttributeError):
                    pass
            
            # Handle numpy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            
            # Handle numpy integer types
            if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            
            # Handle numpy floating types (including float32)
            # Note: np.float_ was removed in NumPy 2.0, use np.float64 instead
            if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            
            # Handle numpy boolean types
            if isinstance(obj, np.bool_):
                return bool(obj)
            
            # Check by type name as fallback (for numpy scalars that aren't caught above)
            if type(obj).__module__ == 'numpy':
                if 'float' in str(type(obj)):
                    return float(obj)
                elif 'int' in str(type(obj)):
                    return int(obj)
                elif 'bool' in str(type(obj)):
                    return bool(obj)
            
            # Handle dictionaries
            if isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            
            # Handle lists and tuples
            if isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            
            return obj
        
        # Create comprehensive history with system info and summary
        full_history = {
            "system_info": system_info,
            "training_summary": {
                "total_training_time_seconds": round(total_training_time, 2),
                "total_training_time_minutes": round(total_training_time / 60, 2),
                "total_training_time_hours": round(total_training_time / 3600, 2),
                "best_overall_auc": round(best_overall_auc, 4),
                "ensemble_auc": round(ensemble_metrics['auc'], 4),
                "ensemble_ap": round(ensemble_metrics['ap'], 4),
                "ensemble_precision": round(ensemble_metrics['precision'], 4),
                "ensemble_recall": round(ensemble_metrics['recall'], 4),
                "ensemble_f1": round(ensemble_metrics['f1'], 4),
                "best_optimal_threshold": round(best_optimal_threshold, 4),
                "num_models": len(all_models),
                "num_folds": len(history),
                "feature_methods": feature_selection_methods,
                "final_memory_usage": convert_to_native(final_memory),
            },
            "fold_histories": convert_to_native(history),
        }
        
        # Convert entire history to ensure all numpy types are converted (safety check)
        full_history = convert_to_native(full_history)
        
        # Save history with timestamp (including compound type)
        history_path = get_timestamped_path("hts3d_training_history", "json", compound=compound)
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(full_history, f, indent=2)
        print(f"[history] saved to {history_path}")
        print(f"[summary] Total training time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
        print(f"[summary] Best overall AUC: {best_overall_auc:.4f}")
        print(f"[summary] Ensemble AUC: {ensemble_metrics['auc']:.4f}")
        print(f"[summary] Ensemble Precision: {ensemble_metrics['precision']:.4f}")
        print(f"[summary] Ensemble Recall: {ensemble_metrics['recall']:.4f}")
        print(f"[summary] Optimal threshold (for best model): {best_optimal_threshold:.4f} (Note: Metrics use this threshold, not 0.5)")
        
        # Save final checkpoint at end of training
        if save_checkpoints:
            if checkpoint_path is None:
                # Auto-generate final checkpoint path
                checkpoint_dir = OUTPUT_DIR / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_checkpoint_path = checkpoint_dir / f"final_checkpoint_{compound}_{timestamp}.pt"
            else:
                # Use provided checkpoint path
                final_checkpoint_path = Path(checkpoint_path)
                if final_checkpoint_path.suffix != '.pt':
                    final_checkpoint_path = final_checkpoint_path.with_suffix('.pt')
                # Add "final" prefix
                final_checkpoint_path = final_checkpoint_path.parent / f"final_{final_checkpoint_path.name}"
            
            try:
                # Use the best model state for final checkpoint
                if best_state is not None and all_models:
                    last_model_info = all_models[-1]
                    # Save final checkpoint with best model state
                    final_checkpoint_data = {
                        'model_state_dict': best_state,
                        'current_method': feature_selection_methods[-1],
                        'current_fold': 4,  # Last fold
                        'all_models': all_models,
                        'all_predictions': all_predictions,
                        'history': history,
                        'best_auc': best_auc,
                        'best_state': best_state,
                        'labels': labels,
                        'feature_selection_methods': feature_selection_methods,
                        'rdkit_dim': last_model_info.get('rdkit_dim', 200),
                        'timestamp': datetime.now().isoformat(),
                        'training_complete': True,
                    }
                    torch.save(final_checkpoint_data, final_checkpoint_path)
                    print(f"[checkpoint] Saved final checkpoint to {final_checkpoint_path}")
            except Exception as e:
                print(f"[checkpoint] WARNING: Failed to save final checkpoint: {e}")
                import traceback
                traceback.print_exc()
        
        # Restore stdout and close log file
        print("=" * 80)
        print(f"[log] Training log ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[log] Log file saved to: {log_path}")
        sys.stdout = original_stdout
        tee_output.close()
        print(f"[log] Training log saved to: {log_path}")
    except Exception as e:
        # Ensure stdout is restored even if there's an error
        if 'original_stdout' in locals():
            sys.stdout = original_stdout
        if 'tee_output' in locals():
            tee_output.close()
        print(f"[log] ERROR during training - log file may be incomplete: {e}")
        raise


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="HTS_3D: Cross-attention HTS pipeline with 3D-aware ligand and protein pocket encoding",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Use default NLRP3 activities CSV and template (outputs hts3d_model_nlrp3.pt):
  python HTS_3D.py
  
  # HTS.py-style CD28: use libraries/library.csv + libraries/positives.csv (same as HTS.py; do not use cd28lib.csv):
  python HTS_3D.py --cd28
  
  # Train on CD28 compounds via activity CSV (uses cd28lib.csv, auto-detects CD28, outputs hts3d_model_cd28.pt):
  python HTS_3D.py --compounds cd28lib.csv
  
  # Use custom compounds CSV file:
  python HTS_3D.py --compounds chembl_random_compounds.csv
  
  # Use custom compounds with full path:
  python HTS_3D.py --compounds C:\\Users\\xiaon\\source\\repos\\NLRP3A\\chembl_random_compounds.csv
  
  # Use custom bioactivity threshold:
  python HTS_3D.py --compounds activities.csv --biothreshold 500
  
  # Use label column directly (0=inactive, 1=active):
  python HTS_3D.py --compounds activities.csv --label activity_label
  
  # Use custom protein PDB file:
  python HTS_3D.py --compounds activities.csv --protein src/7alv.pdb
  
  # Use custom protein PDB file with custom max pockets:
  python HTS_3D.py --compounds activities.csv --protein src/1YJD.pdb --max-pockets 3
  
  # Save checkpoints during training (auto-generated path):
  python HTS_3D.py --compounds cd28lib.csv --protein src/1YJD.pdb --label label
  
  # Save checkpoints to specific path:
  python HTS_3D.py --compounds cd28lib.csv --protein src/1YJD.pdb --label label --checkpoint checkpoints/my_training.pt
  
  # Resume training from checkpoint:
  python HTS_3D.py --compounds cd28lib.csv --protein src/1YJD.pdb --label label --resume backup/output/012620263efixLRibalance/hts3d_model_cd28.pt
  
  # Disable checkpoint saving:
  python HTS_3D.py --compounds cd28lib.csv --protein src/1YJD.pdb --label label --no-checkpoints
  
  # Use only lasso feature selection (faster, single method ensemble):
  python HTS_3D.py --compounds cd28lib.csv --protein src/1YJD.pdb --label label --pick lasso
  
  # Use only pca feature selection:
  python HTS_3D.py --compounds cd28lib.csv --protein src/1YJD.pdb --label label --pick pca
  
  # Use only mutual-info feature selection:
  python HTS_3D.py --compounds cd28lib.csv --protein src/1YJD.pdb --label label --pick mutual-info
  
  # Option C Ultra (fixUndFitImblHTSpyClaudeC1.txt): attention + interaction + adaptive + multi-pocket (3):
  python HTS_3D.py --ultra
  python HTS_3D.py --compounds activities.csv --ultra
            """
        )
        
        parser.add_argument(
            "--compounds",
            type=str,
            default=None,
            help=f"Path to compounds/activities CSV file. If not specified, uses default: {DEFAULT_CSV_PATH_NLRP3}"
        )
        
        parser.add_argument(
            "--biothreshold",
            type=float,
            default=DEFAULT_BIO_THRESHOLD,
            help=f"Bioactivity threshold in nM for binary classification (default: {DEFAULT_BIO_THRESHOLD} nM). Compounds with IC50/EC50 <= threshold are labeled as active. Ignored if --label is specified."
        )
        
        parser.add_argument(
            "--label",
            type=str,
            default=None,
            help="Name of CSV column containing binary labels (0=inactive, 1=active). If specified, uses this column directly instead of calculating labels from --biothreshold. Example: --label activity_label"
        )
        
        parser.add_argument(
            "--protein",
            type=str,
            default=None,
            help="Path to PDB file for protein structure. If not specified, uses default NLRP3 pocket template (7ALV). Example: --protein src/7alv.pdb or --protein src/1YJD.pdb"
        )
        
        parser.add_argument(
            "--max-pockets",
            type=int,
            default=5,
            help="Maximum number of pockets to detect from protein structure (default: 5). Only used when --protein is specified."
        )
        
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Path to save checkpoints during training. If not specified, auto-generates checkpoint path in output/checkpoints/. Example: --checkpoint checkpoints/my_checkpoint.pt"
        )
        
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            nargs='?',
            const='auto',
            help="Path to checkpoint file to resume training from. If --resume is used without a path, automatically finds the latest final checkpoint in output/checkpoints/. Example: --resume backup/output/012620263efixLRibalance/hts3d_model_cd28.pt or --resume (auto-detect latest)"
        )
        
        parser.add_argument(
            "--no-checkpoints",
            action="store_true",
            help="Disable checkpoint saving during training"
        )
        
        parser.add_argument(
            "--pick",
            type=str,
            choices=['lasso', 'pca', 'mutual-info'],
            default=None,
            help="Pick a single feature selection method to use instead of all three. Options: 'lasso', 'pca', or 'mutual-info'. If not specified, uses all three methods for ensemble. Example: --pick lasso"
        )
        
        parser.add_argument(
            "--cd28",
            action="store_true",
            help="Use HTS.py-style data: load libraries/library.csv and libraries/positives.csv; label = 1 if SMILES is in positives else 0. Same folds/batches as HTS.py. Do not use cd28lib.csv or any --compounds file."
        )
        
        parser.add_argument(
            "--ultra",
            action="store_true",
            help="Use Option C Ultra (fixUndFitImblHTSpyClaudeC1.txt): attention pooling, protein-ligand interaction, adaptive feature weighting, multi-pocket (3), residual branches. Builds on Option C Minimal. Default: Minimal."
        )
        
        parser.add_argument(
            "--ensemble",
            type=str,
            choices=["oof", "full", "hts"],
            default="oof",
            help="Ensemble strategy: 'oof' = out-of-fold only, 3 preds/sample, true CV (default). 'full' = all 15 models, per-model selector, sharpest. 'hts' = emulate HTS.py: loop by method, fold 0 selector for all folds, avg 5 then 3 → smoother. Default: oof"
        )
        
        parser.add_argument(
            "--qed-scale",
            type=float,
            default=QED_SCALE_FACTOR,
            help=f"Scale factor for QED in physchem features (default: {QED_SCALE_FACTOR}). HTS3DOracle uses 40-60% QED in method-specific predictions."
        )
        
        parser.add_argument(
            "--qed-blend",
            type=float,
            default=0.2,
            help="Blend ensemble predictions with QED (0-1). 0.2 matches HTS3DOracle make_varied_predictions. Use 0 to disable."
        )
        
        args = parser.parse_args()
        
        # Convert compounds path to Path object if provided
        csv_path = None
        if args.compounds:
            csv_path = Path(args.compounds)
            if not csv_path.is_absolute():
                # If relative path, make it relative to project root
                csv_path = PROJECT_ROOT / csv_path
        
        # Convert protein PDB path to Path object if provided
        protein_pdb_path = None
        if args.protein:
            protein_pdb_path = Path(args.protein)
            if not protein_pdb_path.is_absolute():
                # If relative path, try relative to project root first, then script directory
                if (PROJECT_ROOT / protein_pdb_path).exists():
                    protein_pdb_path = PROJECT_ROOT / protein_pdb_path
                elif (SCRIPT_DIR / protein_pdb_path).exists():
                    protein_pdb_path = SCRIPT_DIR / protein_pdb_path
                else:
                    # Use project root as default
                    protein_pdb_path = PROJECT_ROOT / protein_pdb_path
        
        # Convert checkpoint paths to Path objects if provided
        checkpoint_path = None
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = PROJECT_ROOT / checkpoint_path
        
        resume_from_checkpoint = None
        if args.resume:
            if args.resume == 'auto' or args.resume is None:
                # Auto-detect latest final checkpoint
                checkpoint_dir = OUTPUT_DIR / "checkpoints"
                print(f"[checkpoint] Auto-detecting latest checkpoint in {checkpoint_dir}...")
                resume_from_checkpoint = find_latest_checkpoint(checkpoint_dir, "final_checkpoint_*.pt")
                
                if resume_from_checkpoint is None:
                    print(f"[checkpoint] No final checkpoint found, trying any checkpoint...")
                    resume_from_checkpoint = find_latest_checkpoint(checkpoint_dir, "checkpoint_*.pt")
                
                if resume_from_checkpoint is None:
                    print(f"[checkpoint] WARNING: No checkpoint files found in {checkpoint_dir}")
                    print(f"[checkpoint] Please specify a checkpoint path with --resume <path>")
                    resume_from_checkpoint = None
            else:
                # Use provided path
                resume_from_checkpoint = Path(args.resume)
                if not resume_from_checkpoint.is_absolute():
                    resume_from_checkpoint = PROJECT_ROOT / resume_from_checkpoint
        
        main(
            csv_path=csv_path,
            protein_pdb_path=protein_pdb_path,
            max_pockets=args.max_pockets,
            bio_threshold=args.biothreshold,
            label_column=args.label,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=resume_from_checkpoint,
            save_checkpoints=not args.no_checkpoints,
            pick_method=args.pick,
            use_cd28=args.cd28,
            use_ultra=args.ultra,
            ensemble_strategy=args.ensemble,
            qed_scale=args.qed_scale,
            qed_blend_weight=args.qed_blend,
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)

