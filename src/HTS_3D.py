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

Input: `nlrp3_chembl_activities.csv`
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdPartialCharges
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer

# --- Global configuration ----------------------------------------------------

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

# Paths relative to script location (src/)
CSV_PATH = SCRIPT_DIR / "nlrp3_chembl_activities.csv"
# Output directory in project root
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "hts3d_model.pt"
# Cache directory in script directory
CACHE_DIR = SCRIPT_DIR / ".cache_hts3d"


def get_timestamped_path(base_name: str, extension: str = None) -> Path:
    """
    Generate a timestamped file path in the output directory.
    
    Args:
        base_name: Base name of the file (without extension)
        extension: File extension (e.g., 'csv', 'json'). If None, extracted from base_name
    
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
    if extension:
        filename = f"{base_name}_{timestamp}.{extension}"
    else:
        filename = f"{base_name}_{timestamp}"
    
    return OUTPUT_DIR / filename


MAX_SEQ_LEN = 128
MAX_ATOMS = 64
CHEMBERTA_NAME = "seyonec/ChemBERTa-zinc-base-v1"
SEED = 42
EMBED_DIM = 256
NUM_HEADS = 4

# Silence noisy RDKit warnings (e.g., MorganGenerator deprecation spam)
RDLogger.DisableLog("rdApp.warning")

def detect_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] Using CUDA GPU: {torch.cuda.get_device_name(device)}")
        return device
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
        if len(np.unique(y_true)) < 2:
            return fallback
        if metric_fn in (roc_auc_score, average_precision_score):
            if np.allclose(y_pred, y_pred[0]):
                return fallback
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

def load_activity_table(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    
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
    
    # Check if this CSV has activity data (standard_value column)
    has_activity_data = "standard_value" in df.columns
    
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

        # Binary labels: active if <= 1000 nM (approx sub-micromolar)
        df["label"] = (df["value_nM"] <= 1_000).astype(int)

        # Dedupe by molecule id keeping best potency
        if "molecule_chembl_id" in df.columns:
            df = (
                df.sort_values("value_nM")
                .groupby("molecule_chembl_id", as_index=False)
                .first()
            )
        df = df.reset_index(drop=True)
        print(f"[data] molecules: {len(df)}, actives: {df['label'].sum()}")
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
        
        # Ensure molecule_chembl_id exists (create if missing)
        if "molecule_chembl_id" not in df.columns:
            df["molecule_chembl_id"] = [f"COMPOUND_{i}" for i in range(len(df))]
        
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


def physchem_features(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(12, dtype=np.float32)
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
            QED.qed(mol),
        ],
        dtype=np.float32,
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def build_rdkit_feature_matrix(smiles_list: List[str]) -> torch.Tensor:
    """
    Build RDKit 2D feature matrix and return as GPU tensor if available.
    RDKit operations must run on CPU, but result is moved to GPU.
    """
    warning_budget = 3
    rdkit_rows = []
    for smi in tqdm(smiles_list, desc="RDKit 2D features"):
        if warning_budget > 0:
            print("[RDKit] DEPRECATION WARNING: please use MorganGenerator")
            warning_budget -= 1
        fp = morgan_fp(smi)
        phys = physchem_features(smi)
        rdkit_rows.append(np.concatenate([fp, phys]))
    # Stack on CPU (numpy), then convert to torch and move to GPU
    feat_array = np.vstack(rdkit_rows).astype(np.float32)
    return torch.tensor(feat_array, dtype=torch.float32, device=DEVICE)


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
        AllChem.EmbedMolecule(mol, params)
        AllChem.MMFFOptimizeMolecule(mol)
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        return base_feat, mask

    conf = mol.GetConformer()
    num_atoms = min(mol.GetNumAtoms(), max_atoms)
    for idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(idx)
        pos = conf.GetAtomPosition(idx)
        g_charge = float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0
        g_h_charge = (
            float(atom.GetDoubleProp("_GasteigerHCharge"))
            if atom.HasProp("_GasteigerHCharge")
            else 0.0
        )
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
            pos.x / 10.0,
            pos.y / 10.0,
            pos.z / 10.0,
            g_charge,
            g_h_charge,
            Descriptors.MolMR(mol) / 200.0,
            Descriptors.MolWt(mol) / 1000.0,
        ]
        base_feat[idx] = np.asarray(feature_vec, dtype=np.float32)
        mask[idx] = 1.0
    return base_feat, mask


def build_3d_cache(smiles_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build 3D conformer cache and return as GPU tensors if available.
    RDKit operations must run on CPU, but result is moved to GPU.
    """
    atom_tensors = []
    masks = []
    for smi in tqdm(smiles_list, desc="RDKit 3D conformers"):
        feats, mask = generate_conformer_features(smi)
        atom_tensors.append(feats)
        masks.append(mask)
    # Stack on CPU (numpy), then convert to torch and move to GPU
    atom_array = np.stack(atom_tensors).astype(np.float32)
    mask_array = np.stack(masks).astype(np.float32)
    return (
        torch.tensor(atom_array, dtype=torch.float32, device=DEVICE),
        torch.tensor(mask_array, dtype=torch.float32, device=DEVICE),
    )


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
    pockets: List[List[Tuple[str, float, float, Tuple[float, float, float]]]]
) -> List[Dict[str, torch.Tensor]]:
    """
    Build tensor representations for multiple protein pockets.
    
    Args:
        pockets: List of pocket templates from detect_protein_pockets()
    
    Returns:
        List of tensor dictionaries, each with 'aa_idx', 'charges', 'hydros', 'coords'
    """
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
            "aa_idx": torch.tensor(aa_idx, dtype=torch.long),
            "charges": torch.tensor(charges, dtype=torch.float32),
            "hydros": torch.tensor(hydros, dtype=torch.float32),
            "coords": torch.tensor(coords, dtype=torch.float32),
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
        template = {k: v.to(DEVICE) for k, v in template.items()}
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
        attn_out, attn_weights = self.attn(
            ligand_tokens, protein_tokens, protein_tokens
        )
        x = self.norm1(ligand_tokens + attn_out)
        x = self.norm2(x + self.ff(x))
        if ligand_mask is not None:
            x = x * ligand_mask.unsqueeze(-1)
        return x, attn_weights


class HTS3DModel(nn.Module):
    def __init__(
        self, 
        rdkit_dim: int, 
        atom_dim: int,
        protein_pockets: Optional[List[Dict[str, torch.Tensor]]] = None,
        num_pockets: int = 5
    ):
        super().__init__()
        self.chemberta = RobertaModel.from_pretrained(CHEMBERTA_NAME)
        self.chemberta_proj = nn.Linear(self.chemberta.config.hidden_size, EMBED_DIM)

        self.lig3d = Ligand3DEncoder(atom_dim, EMBED_DIM)
        self.cross_attn = CrossAttentionFusion(EMBED_DIM, NUM_HEADS)
        
        # Use multi-pocket encoder to test multiple viable pockets
        if protein_pockets is None:
            protein_pockets = PROTEIN_TEMPLATES
        self.protein_pockets = protein_pockets
        self.protein_encoder = MultiPocketEncoder(EMBED_DIM, num_pockets=min(num_pockets, len(protein_pockets)))

        self.rdkit_branch = nn.Sequential(
            nn.Linear(rdkit_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM + 256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _initialize_weights(self):
        """Initialize model weights to prevent NaN from extreme initial values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier uniform initialization with gain scaled down
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rdkit_feats: torch.Tensor,
        atom_features: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chem_out = self.chemberta(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # [B, L, H]
        chem_tokens = self.chemberta_proj(chem_out)

        atom_tokens = self.lig3d(atom_features)
        ligand_tokens = torch.cat([chem_tokens, atom_tokens], dim=1)

        lig_mask = torch.cat([attention_mask, atom_mask], dim=1)
        lig_mask = (lig_mask > 0).float()

        # Encode multiple pockets and combine them
        combined_pocket = self.protein_encoder(self.protein_pockets)  # [embed_dim]
        # Expand to match ligand tokens for cross-attention
        # Use combined pocket as single "residue" for attention
        protein_tokens = combined_pocket.unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
        protein_tokens = protein_tokens.expand(ligand_tokens.size(0), -1, -1)  # [B, 1, embed_dim]

        fused, attn_weights = self.cross_attn(ligand_tokens, protein_tokens, lig_mask)

        lig_sum = fused.sum(dim=1)
        lig_den = lig_mask.sum(dim=1).clamp(min=1.0)
        lig_pool = lig_sum / lig_den.unsqueeze(-1)
        
        # Check for NaN/Inf in intermediate outputs
        if torch.isnan(lig_pool).any() or torch.isinf(lig_pool).any():
            print("[model] WARNING: NaN/Inf in lig_pool, replacing with zeros")
            lig_pool = torch.nan_to_num(lig_pool, nan=0.0, posinf=0.0, neginf=0.0)

        rdkit_emb = self.rdkit_branch(rdkit_feats)
        
        # Check for NaN/Inf in rdkit_emb
        if torch.isnan(rdkit_emb).any() or torch.isinf(rdkit_emb).any():
            print("[model] WARNING: NaN/Inf in rdkit_emb, replacing with zeros")
            rdkit_emb = torch.nan_to_num(rdkit_emb, nan=0.0, posinf=0.0, neginf=0.0)
        
        logits = self.classifier(torch.cat([lig_pool, rdkit_emb], dim=1)).squeeze(-1)
        
        # Final safety check on logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("[model] WARNING: NaN/Inf in final logits, replacing with zeros")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        
        return logits, attn_weights


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
) -> Tuple[float, float]:
    """Returns (average_loss, time_elapsed_seconds)."""
    model.train()
    running = 0.0
    total_batches = len(loader)
    start_time = time.time()
    print(f"[train] Starting epoch: {total_batches} batches")
    for batch_idx, sample in enumerate(loader, 1):
        batch = to_device(sample)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(
            batch.input_ids,
            batch.attention_mask,
            batch.rdkit,
            batch.atom_features,
            batch.atom_mask,
        )
        # Check for NaN/Inf in logits before computing loss
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[train] WARNING: NaN/Inf detected in logits at batch {batch_idx}, skipping batch")
            continue
        
        loss = criterion(logits, batch.labels)
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[train] WARNING: NaN/Inf loss at batch {batch_idx}, skipping batch")
            continue
        
        loss.backward()
        
        # Check for NaN gradients before clipping
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"[train] WARNING: NaN/Inf gradients detected at batch {batch_idx}, skipping update")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item()
        
        # Print progress every 10% or at key milestones
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % max(
            1, total_batches // 10
        ) == 0:
            avg_loss = running / batch_idx
            print(f"[train] batch {batch_idx}/{total_batches} | loss: {loss.item():.4f} | avg_loss: {avg_loss:.4f}")
    
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
    preds, labels = [], []
    total_batches = len(loader)
    start_time = time.time()
    print(f"[eval] Starting evaluation: {total_batches} batches")
    for batch_idx, sample in enumerate(loader, 1):
        batch = to_device(sample)
        logits, _ = model(
            batch.input_ids,
            batch.attention_mask,
            batch.rdkit,
            batch.atom_features,
            batch.atom_mask,
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)
        labels.append(batch.labels.cpu().numpy())
        
        # Print progress every 20% or at key milestones
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % max(
            1, total_batches // 5
        ) == 0:
            total_samples = sum(len(p) for p in preds) + len(probs)
            print(f"[eval] batch {batch_idx}/{total_batches} | processed {total_samples} samples")
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    print(f"[eval] Computing metrics on {len(preds)} predictions")
    if np.isnan(preds).any() or np.isinf(preds).any():
        print("[eval] detected NaN/Inf predictions, replacing with 0.5")
        preds = np.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)
    binary = (preds > 0.5).astype(int)
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
    try:
        tokenizer = RobertaTokenizer.from_pretrained(CHEMBERTA_NAME)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_NAME)
    return tokenizer


# --- Main orchestration ------------------------------------------------------


def prepare_tensors(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = load_tokenizer()
    encodings = tokenizer(
        df["canonical_smiles"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )
    return encodings["input_ids"], encodings["attention_mask"]


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
            templates = build_protein_template_from_pockets(pockets)
            print(f"[protein] Detected {len(templates)} pockets from {pdb_path.name}")
            return templates
        else:
            print(f"[protein] Warning: No residues found in {pdb_path}")
            return PROTEIN_TEMPLATES
    except Exception as e:
        print(f"[protein] Warning: Error loading PDB file {pdb_path}: {e}")
        print("[protein] Using default pocket template")
        return PROTEIN_TEMPLATES


def main(
    csv_path: Optional[Path] = None,
    protein_pdb_path: Optional[Path] = None,
    max_pockets: int = 5
):
    """
    Main training function with support for arbitrary protein structures.
    
    Args:
        csv_path: Optional path to activity CSV file.
                 If None, uses default CSV_PATH (nlrp3_chembl_activities.csv).
        protein_pdb_path: Optional path to PDB file for protein structure.
                         If None, uses default NLRP3 pocket template.
        max_pockets: Maximum number of pockets to detect from protein structure.
    """
    # Use default CSV path if not specified
    if csv_path is None:
        csv_path = CSV_PATH
    
    CACHE_DIR.mkdir(exist_ok=True)
    print(f"[debug] looking for CSV at {csv_path.resolve()}")
    if not csv_path.exists():
        print(f"[debug] CSV missing: {csv_path.resolve()}")
    else:
        print(f"[debug] CSV present: {csv_path.resolve()} (size={csv_path.stat().st_size} bytes)")
    df = load_activity_table(csv_path)
    
    # Load protein structure and detect pockets
    if protein_pdb_path and protein_pdb_path.exists():
        protein_templates = load_protein_structure_from_pdb(protein_pdb_path, max_pockets)
        print(f"[protein] Using {len(protein_templates)} detected pockets")
    else:
        protein_templates = PROTEIN_TEMPLATES
        print(f"[protein] Using default pocket template (NLRP3)")
    print(f"[debug] loaded dataframe with {len(df)} rows")
    smiles = df["canonical_smiles"].tolist()

    input_ids, attention_mask = prepare_tensors(df)
    # Move tokenizer outputs to GPU if available (they start on CPU)
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    print(f"[debug] Tokenizer outputs on device: {input_ids.device}")

    print("[debug] generating RDKit 2D features")
    rdkit_feats = build_rdkit_feature_matrix(smiles)  # Already on GPU if available
    print(f"[debug] RDKit features shape: {rdkit_feats.shape}, device: {rdkit_feats.device}")
    
    # GPU-accelerated scaling (all operations on GPU)
    scaler = GPUScaler()
    rdkit_feats = scaler.fit_transform(rdkit_feats, return_numpy=False)  # Keep on GPU
    print(f"[debug] Scaled RDKit features on device: {rdkit_feats.device}")

    print("[debug] building 3D conformer cache")
    atom_feats, atom_masks = build_3d_cache(smiles)  # Already on GPU if available
    print(f"[debug] 3D features shape: {atom_feats.shape}, device: {atom_feats.device}")
    labels = df["label"].to_numpy(dtype=np.float32)

    # Note: input_ids and attention_mask are already on GPU, but dataset will handle them correctly
    dataset = HTS3DDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        rdkit_feats=rdkit_feats,
        atom_feats=atom_feats,
        atom_masks=atom_masks,
        labels=labels,
    )

    rdkit_dim = rdkit_feats.shape[1]
    atom_dim = atom_feats.shape[-1]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    history = []
    best_auc = float("-inf")
    best_state = None
    fold_preds = np.zeros(len(dataset))
    
    # Collect system information
    system_info = get_system_info()
    print(f"[system] Device: {system_info['device_name']} | CUDA: {system_info['cuda_version']} | "
          f"RAM: {system_info['total_ram_gb']} GB")
    if DEVICE.type == "cuda":
        print(f"[system] GPU Memory: {system_info['gpu_memory_gb']} GB")
    
    training_start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n[fold {fold + 1}] train={len(train_idx)} val={len(val_idx)}")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Disable pin_memory if tensors are already on GPU (pin_memory only works for CPU tensors)
        use_pin_memory = PIN_MEMORY and DEVICE.type == "cpu"
        
        train_loader = DataLoader(
            train_subset,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            pin_memory=use_pin_memory,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=use_pin_memory,
        )

        model = HTS3DModel(
            rdkit_dim, 
            atom_dim, 
            protein_pockets=protein_templates,
            num_pockets=max_pockets
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
        
        # Calculate pos_weight with safety checks to prevent extreme values
        pos_count = (labels[train_idx] == 1).sum()
        neg_count = (labels[train_idx] == 0).sum()
        if pos_count == 0:
            print(f"[fold {fold + 1}] WARNING: No positive samples in training set!")
            pos_weight_val = 1.0
        else:
            pos_weight_val = float(neg_count) / float(pos_count)
            # Clamp pos_weight to reasonable range to avoid extreme values that cause NaN
            pos_weight_val = max(0.1, min(10.0, pos_weight_val))
        
        print(f"[fold {fold + 1}] pos_weight: {pos_weight_val:.4f} (pos: {pos_count}, neg: {neg_count})")
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_val, device=DEVICE)
        )

        fold_history = {
            "train_loss": [],
            "val_auc": [],
            "val_ap": [],
            "train_time_seconds": [],
            "eval_time_seconds": [],
            "memory_usage": [],
        }
        patience = 3
        no_improve = 0
        best_fold_auc = float("-inf")
        best_metrics = None
        fold_start_time = time.time()

        for epoch in range(6):
            print(f"[fold {fold + 1}] epoch {epoch + 1}")
            train_loss, train_time = train_one_epoch(model, train_loader, optimizer, criterion)
            metrics = evaluate(model, val_loader)
            
            # Collect memory usage after epoch
            mem_usage = get_memory_usage()

            fold_history["train_loss"].append(train_loss)
            fold_history["val_auc"].append(metrics["auc"])
            fold_history["val_ap"].append(metrics["ap"])
            fold_history["train_time_seconds"].append(train_time)
            fold_history["eval_time_seconds"].append(metrics["eval_time_seconds"])
            fold_history["memory_usage"].append(mem_usage)

            print(
                f"[fold {fold + 1}] Epoch {epoch + 1} summary: "
                f"train_loss={train_loss:.4f} | "
                f"val_auc={metrics['auc']:.4f} | "
                f"val_ap={metrics['ap']:.4f} | "
                f"val_precision={metrics['precision']:.4f} | "
                f"val_recall={metrics['recall']:.4f} | "
                f"val_f1={metrics['f1']:.4f}"
            )

            if best_metrics is None or (
                not math.isnan(metrics["auc"]) and metrics["auc"] > best_fold_auc
            ):
                best_fold_auc = metrics["auc"]
                best_metrics = metrics
                no_improve = 0
                if not math.isnan(metrics["auc"]) and metrics["auc"] > best_auc:
                    best_auc = metrics["auc"]
                    best_state = model.state_dict()
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("    early stopping.")
                    break

        if best_metrics is None:
            raise RuntimeError("No valid metrics were produced for this fold.")
        
        fold_elapsed_time = time.time() - fold_start_time
        fold_history["fold_total_time_seconds"] = fold_elapsed_time
        fold_history["fold_best_auc"] = best_fold_auc
        fold_history["fold_best_epoch"] = fold_history["val_auc"].index(best_fold_auc) + 1
        
        history.append(fold_history)
        fold_preds[val_idx] = best_metrics["preds"]
        print(f"[fold {fold + 1}] Completed in {fold_elapsed_time:.2f}s | Best AUC: {best_fold_auc:.4f}")

    if best_state is None:
        raise RuntimeError("Training failed to produce a model.")

    # Save model without timestamp
    torch.save(best_state, MODEL_PATH)
    print(f"[model] saved to {MODEL_PATH}")

    # Save predictions with timestamp
    pred_path = get_timestamped_path("hts3d_predictions", "csv")
    result_df = df.copy()
    result_df["hts3d_score"] = fold_preds
    result_df.to_csv(pred_path, index=False)
    print(f"[predictions] saved to {pred_path}")

    total_training_time = time.time() - training_start_time
    final_memory = get_memory_usage()
    
    # Create comprehensive history with system info and summary
    full_history = {
        "system_info": system_info,
        "training_summary": {
            "total_training_time_seconds": round(total_training_time, 2),
            "total_training_time_minutes": round(total_training_time / 60, 2),
            "total_training_time_hours": round(total_training_time / 3600, 2),
            "best_overall_auc": round(best_auc, 4),
            "num_folds": len(history),
            "final_memory_usage": final_memory,
        },
        "fold_histories": history,
    }
    
    # Save history with timestamp
    history_path = get_timestamped_path("hts3d_training_history", "json")
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(full_history, f, indent=2)
    print(f"[history] saved to {history_path}")
    print(f"[summary] Total training time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
    print(f"[summary] Best overall AUC: {best_auc:.4f}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="HTS_3D: Cross-attention HTS pipeline with 3D-aware ligand and protein pocket encoding",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Use default NLRP3 activities CSV and template:
  python HTS_3D.py
  
  # Use custom compounds CSV file:
  python HTS_3D.py --compounds chembl_random_compounds.csv
  
  # Use custom compounds with full path:
  python HTS_3D.py --compounds C:\\Users\\xiaon\\source\\repos\\NLRP3A\\chembl_random_compounds.csv
            """
        )
        
        parser.add_argument(
            "--compounds",
            type=str,
            default=None,
            help=f"Path to compounds/activities CSV file. If not specified, uses default: {CSV_PATH}"
        )
        
        args = parser.parse_args()
        
        # Convert compounds path to Path object if provided
        csv_path = None
        if args.compounds:
            csv_path = Path(args.compounds)
            if not csv_path.is_absolute():
                # If relative path, make it relative to project root
                csv_path = PROJECT_ROOT / csv_path
        
        # Can optionally pass protein_pdb_path for custom protein structures
        # Example: main(csv_path=Path("compounds.csv"), protein_pdb_path=Path("path/to/protein.pdb"), max_pockets=5)
        main(csv_path=csv_path)  # Defaults to NLRP3 template and CSV_PATH
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)

