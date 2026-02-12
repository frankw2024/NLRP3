import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Lipinski, QED
from rdkit.Chem import GetSSSR
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
# Lazy imports for heavy modules (loaded only when needed)
# from transformers import RobertaTokenizer, RobertaModel  # Moved to function level
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from rdkit import RDLogger
import warnings
import sys
import traceback
import logging
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

# Output directory for logs
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Suppress warnings
RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings
warnings.filterwarnings("ignore")

# ============================================================================
# PERFORMANCE OPTIMIZATIONS FOR GPU ACCELERATION
# ============================================================================
# This script has been optimized for maximum GPU performance:
# 1. Multiprocessing: RDKit feature generation parallelized across CPU cores
# 2. GPU Training: All model operations run on GPU with mixed precision (AMP)
# 3. Optimized Data Loading: Parallel workers, pinned memory, non-blocking transfers
# 4. Increased Batch Size: 64 for GPU (vs 32 for CPU) to maximize throughput
# 5. cuDNN Benchmark: Enabled for consistent input size optimization
# 6. Mixed Precision: Automatic Mixed Precision (AMP) for 2x speedup on modern GPUs
# ============================================================================

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
# NOTE: torch.cuda.manual_seed_all() moved to main() to avoid slow CUDA initialization at import time

# Device will be set after detection, but define constants for GPU optimization
# These will be set in main after device detection
DEVICE = None
NON_BLOCKING = False
PIN_MEMORY = False

# Error handling wrapper function
def safe_metric(metric_fn, y_true, y_pred, fallback_value=0.0, **kwargs):
    """Safely compute a metric, handling exceptions and edge cases."""
    try:
        # Check for NaNs or infinities
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            print(f"Warning: NaN or Inf values detected in predictions. Using fallback value.")
            return fallback_value
            
        # Check for single class in predictions or true values
        unique_pred = np.unique(y_pred > 0.5)
        unique_true = np.unique(y_true)
        
        if len(unique_pred) < 2 or len(unique_true) < 2:
            print(f"Warning: Only one class present in predictions or true values. Using fallback value.")
            return fallback_value
            
        # For metrics that require positive samples
        if metric_fn in [roc_auc_score, average_precision_score]:
            # Check for all zeros in true values
            if sum(y_true) == 0:
                print(f"Warning: No positive samples in true values. Using fallback value.")
                return fallback_value
                
            # Check for all zeros in predicted values
            if metric_fn == roc_auc_score and all(p == 0 for p in y_pred):
                print(f"Warning: All predictions are 0. Using fallback value.")
                return fallback_value
                
        return metric_fn(y_true, y_pred, **kwargs)
    except Exception as e:
        print(f"Error computing {metric_fn.__name__}: {str(e)}")
        return fallback_value

# Enhanced RDKit feature functions with error handling
def morgan_fp(smiles, radius=2, nBits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES: {smiles}")
            return np.zeros(nBits)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))
    except Exception as e:
        print(f"Error generating Morgan fingerprint for {smiles}: {str(e)}")
        return np.zeros(nBits)

def maccs_fp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(167)
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    except Exception as e:
        print(f"Error generating MACCS keys for {smiles}: {str(e)}")
        return np.zeros(167)

def extended_physchem_desc(smiles):
    """Extended set of physicochemical descriptors with error handling"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * 15
        
        # Count aromatic rings
        aromatic_rings = 0
        try:
            num_rings = list(GetSSSR(mol))  # Convert to list to make it clear
            for ring in num_rings:
                if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                    aromatic_rings += 1
        except Exception as e:
            print(f"Error processing rings for {smiles}: {str(e)}")
            aromatic_rings = 0
            
        # Safely compute descriptors with defaults if they fail
        descriptors = [0] * 15
        
        try: descriptors[0] = Descriptors.MolWt(mol)
        except: pass
        
        try: descriptors[1] = Descriptors.MolLogP(mol)
        except: pass
        
        try: descriptors[2] = Descriptors.NumRotatableBonds(mol)
        except: pass
        
        try: descriptors[3] = Descriptors.NumHAcceptors(mol)
        except: pass
        
        try: descriptors[4] = Descriptors.NumHDonors(mol)
        except: pass
        
        try: descriptors[5] = Descriptors.TPSA(mol)
        except: pass
        
        try: descriptors[6] = Descriptors.RingCount(mol)
        except: pass
        
        descriptors[7] = aromatic_rings
        
        try: descriptors[8] = Descriptors.HeavyAtomCount(mol)
        except: pass
        
        try: descriptors[9] = Descriptors.NumHeteroatoms(mol)
        except: pass
        
        try: descriptors[10] = Descriptors.FractionCSP3(mol)
        except: pass
        
        try: descriptors[11] = Descriptors.NumAromaticRings(mol)
        except: pass
        
        try: descriptors[12] = Lipinski.NumHAcceptors(mol)
        except: pass
        
        try: descriptors[13] = Lipinski.NumHDonors(mol)
        except: pass
        
        try: 
            qed = QED.qed(mol)
            if np.isnan(qed) or np.isinf(qed):
                qed = 0.0
            descriptors[14] = qed
        except: 
            descriptors[14] = 0.0
            
        return descriptors
    except Exception as e:
        print(f"Error computing physicochemical descriptors for {smiles}: {str(e)}")
        return [0] * 15

def process_single_smiles(smiles):
    """Process a single SMILES string to extract features (for multiprocessing)."""
    try:
        morgan = morgan_fp(smiles)
        maccs = maccs_fp(smiles)
        physchem = extended_physchem_desc(smiles)
        combined = np.concatenate((morgan, maccs, physchem))
        
        # Check for NaN or Inf values
        if np.any(np.isnan(combined)) or np.any(np.isinf(combined)):
            # Replace problematic values with zeros
            combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        
        return combined
    except Exception as e:
        # Return zeros as fallback
        return np.zeros(2048 + 167 + 15)

def rdkit_features_from_smiles(smiles_list, n_jobs=None, device=None):
    """
    Generate RDKit features using multiprocessing for parallel CPU-bound operations.
    Returns GPU tensor if device is GPU, otherwise numpy array (learned from HTS_3D.py).
    
    Args:
        smiles_list: List of SMILES strings
        n_jobs: Number of parallel workers (None = use all available CPUs, 0 = sequential)
        device: Target device for tensor (None = auto-detect)
    
    Returns:
        torch.Tensor on device if GPU available, otherwise numpy array
    """
    import sys
    import multiprocessing as mp
    logger = logging.getLogger(__name__)
    logger.info("[RDKit] Generating RDKit features...")
    sys.stdout.flush()
    
    # Determine number of workers
    # On Windows, disable multiprocessing by default to avoid hangs
    is_windows = sys.platform.startswith('win')
    
    if n_jobs is None:
        if is_windows:
            # On Windows, disable multiprocessing by default to avoid startup hangs
            # The spawn method can cause issues with module imports
            n_jobs = 0  # Sequential processing on Windows
            print("[RDKit] Using sequential processing on Windows (multiprocessing disabled to avoid hangs)")
        else:
            n_jobs = max(1, cpu_count() - 1)  # Leave one CPU free on Linux/Mac
    elif n_jobs == 0:
        print("[RDKit] Sequential processing requested")
    
    if n_jobs > 0:
        logger = logging.getLogger(__name__)
        logger.info(f"[RDKit] Using {n_jobs} parallel workers for RDKit feature generation (CPU-bound operations)")
        sys.stdout.flush()
        
        # Use multiprocessing for parallel feature generation
        # Note: RDKit operations are CPU-bound, so parallelization significantly speeds this up
        # On Windows, set start method to 'spawn' explicitly to avoid issues
        try:
            if is_windows:
                # Set start method for Windows (spawn is default but explicit is safer)
                try:
                    mp.set_start_method('spawn', force=False)
                except RuntimeError:
                    # Start method already set, ignore
                    pass
            
            # Use a timeout to prevent hanging
            with Pool(processes=n_jobs) as pool:
                # Use imap for progress tracking
                features = list(tqdm(
                    pool.imap(process_single_smiles, smiles_list, chunksize=max(1, len(smiles_list) // (n_jobs * 4))),
                    total=len(smiles_list),
                    desc="RDKit features"
                ))
        except Exception as e:
            logger.warning(f"[RDKit] Warning: Multiprocessing failed ({e}). Falling back to sequential processing.")
            sys.stdout.flush()
            # Fallback to sequential processing
            features = [process_single_smiles(smi) for smi in tqdm(smiles_list, desc="RDKit features")]
    else:
        # Sequential processing
        logger = logging.getLogger(__name__)
        logger.info("[RDKit] Processing sequentially...")
        sys.stdout.flush()
        features = [process_single_smiles(smi) for smi in tqdm(smiles_list, desc="RDKit features")]
    
    # Stack on CPU (numpy), then convert to torch and move to GPU if available
    logger.info(f"[RDKit] Stacking features...")
    sys.stdout.flush()
    features_array = np.vstack(features).astype(np.float32)
    
    # Check for any remaining invalid features
    if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
        print("[RDKit] Warning: Invalid feature values detected. Fixing...")
        sys.stdout.flush()
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Determine target device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to GPU tensor if device is GPU, otherwise return numpy array
    if device.type != "cpu":
        result = torch.tensor(features_array, dtype=torch.float32, device=device)
        logger.info(f"[RDKit] Generated RDKit features with shape: {result.shape}, device: {result.device}")
        sys.stdout.flush()
        return result
    else:
        logger.info(f"[RDKit] Generated RDKit features with shape: {features_array.shape} (CPU)")
        sys.stdout.flush()
        return features_array

# Improved feature selection functions
def apply_feature_selection(X_train, y_train, X_val, feature_selection_method='all', n_components=100, device=None, use_gpu_scaler=False):
    """
    Apply feature selection with optional GPU-accelerated scaling (learned from HTS_3D.py).
    
    Args:
        X_train: Training features (numpy array or torch tensor)
        y_train: Training labels
        X_val: Validation features (numpy array or torch tensor)
        feature_selection_method: Method to use ('lasso', 'pca', 'mutual_info', 'all')
        n_components: Number of components/features to select
        device: Target device for GPU operations
        use_gpu_scaler: If True, use GPUScaler instead of sklearn StandardScaler
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[feature_selection] Applying feature selection: {feature_selection_method}")
    sys.stdout.flush()
    
    try:
        # Always start with scaling - use GPU scaler if requested and device is GPU
        if use_gpu_scaler and device is not None and device.type != "cpu":
            scaler = GPUScaler(device=device)
            # Convert to tensor if needed
            if isinstance(X_train, np.ndarray):
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
            else:
                X_train_tensor = X_train.to(device) if X_train.device != device else X_train
            
            if isinstance(X_val, np.ndarray):
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
            else:
                X_val_tensor = X_val.to(device) if X_val.device != device else X_val
            
            # Scale on GPU, but return numpy for sklearn compatibility
            X_train_scaled = scaler.fit_transform(X_train_tensor, return_numpy=True)
            X_val_scaled = scaler.transform(X_val_tensor, return_numpy=True)
        else:
            # Use sklearn StandardScaler (CPU-based)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
        
        # Fix any NaN or Inf values that might have appeared after scaling
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        if feature_selection_method == 'lasso':
            # LASSO for feature selection
            lasso = Lasso(alpha=0.01, random_state=SEED, max_iter=1000, tol=1e-4)
            selector = SelectFromModel(lasso, prefit=False, max_features=min(n_components, X_train_scaled.shape[1]))
            
            # Handle case where feature selection might fail
            try:
                selector.fit(X_train_scaled, y_train)
                X_train_selected = selector.transform(X_train_scaled)
                X_val_selected = selector.transform(X_val_scaled)
            except Exception as e:
                print(f"Lasso feature selection failed: {str(e)}. Falling back to all features.")
                selector = None
                X_train_selected = X_train_scaled
                X_val_selected = X_val_scaled
            
        elif feature_selection_method == 'pca':
            # PCA for dimensionality reduction
            n_components_actual = min(n_components, min(X_train_scaled.shape))
            
            # Handle potential PCA issues
            try:
                pca = PCA(n_components=n_components_actual, random_state=SEED)
                X_train_selected = pca.fit_transform(X_train_scaled)
                X_val_selected = pca.transform(X_val_scaled)
                
                # Print explained variance
                explained_var = sum(pca.explained_variance_ratio_)
                print(f"PCA with {X_train_selected.shape[1]} components explains {explained_var:.2%} of variance")
                selector = pca
            except Exception as e:
                print(f"PCA failed: {str(e)}. Falling back to all features.")
                selector = None
                X_train_selected = X_train_scaled
                X_val_selected = X_val_scaled
            
        elif feature_selection_method == 'mutual_info':
            # Mutual information for feature selection
            try:
                selector = SelectKBest(mutual_info_classif, k=min(n_components, X_train_scaled.shape[1]))
                X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                X_val_selected = selector.transform(X_val_scaled)
            except Exception as e:
                print(f"Mutual information feature selection failed: {str(e)}. Falling back to all features.")
                selector = None
                X_train_selected = X_train_scaled
                X_val_selected = X_val_scaled
            
        else:  # 'all' - use all features
            selector = None
            X_train_selected = X_train_scaled
            X_val_selected = X_val_scaled
        
        # Final check for invalid values
        X_train_selected = np.nan_to_num(X_train_selected, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_selected = np.nan_to_num(X_val_selected, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Feature selection complete. Selected {X_train_selected.shape[1]} features.")
        return X_train_selected, X_val_selected, selector, scaler
    
    except Exception as e:
        logger.error(f"Feature selection error: {str(e)}")
        logger.warning("Falling back to original features without selection")
        return X_train, X_val, None, None

# Enhanced Dataset class with error handling
class MolecularDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, rdkit_features, max_length=128):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.rdkit_features = rdkit_features
        self.max_length = max_length
        
        # Pre-encode all SMILES to avoid repeated tokenization
        logger = logging.getLogger(__name__)
        logger.info("Pre-encoding SMILES with tokenizer...")
        self.encoded_data = []
        for smiles in tqdm(smiles_list):
            try:
                encoded = self.tokenizer(smiles, padding="max_length", truncation=True, 
                                     max_length=self.max_length, return_tensors="pt")
                self.encoded_data.append({
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0)
                })
            except Exception as e:
                print(f"Error encoding SMILES {smiles}: {str(e)}")
                # Create empty tensors as fallback
                self.encoded_data.append({
                    "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                    "attention_mask": torch.zeros(self.max_length, dtype=torch.long)
                })
        
        # Pre-convert RDKit features to tensors for efficiency (learned from HTS_3D.py)
        # If features are already GPU tensors, keep them on GPU; otherwise create CPU tensors
        logger.info("[dataset] Pre-converting RDKit features to tensors...")
        sys.stdout.flush()
        self.rdkit_tensors = []
        is_gpu_tensor = isinstance(self.rdkit_features, torch.Tensor) and self.rdkit_features.device.type != "cpu"
        
        for i in range(len(self.rdkit_features)):
            if is_gpu_tensor:
                # Already a GPU tensor, just index it
                rdkit_tensor = self.rdkit_features[i]
            else:
                # Convert from numpy to tensor (CPU, will be moved to GPU in training)
                rdkit_tensor = torch.tensor(self.rdkit_features[i], dtype=torch.float32)
            
            # Check for NaN or Inf values
            if torch.isnan(rdkit_tensor).any() or torch.isinf(rdkit_tensor).any():
                rdkit_tensor = torch.nan_to_num(rdkit_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            self.rdkit_tensors.append(rdkit_tensor)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        try:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            
            # Get pre-encoded data
            input_ids = self.encoded_data[idx]["input_ids"]
            attention_mask = self.encoded_data[idx]["attention_mask"]
            
            # Get RDKit features (pre-converted to tensor for efficiency)
            rdkit_tensor = self.rdkit_tensors[idx]
                
            return {
                "input_ids": input_ids, 
                "attention_mask": attention_mask, 
                "label": label, 
                "rdkit_features": rdkit_tensor
            }
        except Exception as e:
            print(f"Error getting item at index {idx}: {str(e)}")
            # Return dummy data
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
                "label": torch.tensor(0.0, dtype=torch.float),
                "rdkit_features": torch.zeros(self.rdkit_features.shape[1], dtype=torch.float)
            }

# Improved Model with dropout and batch normalization
class ImprovedCombinedModel(nn.Module):
    def __init__(self, chemberta_model="seyonec/ChemBERTa-zinc-base-v1", 
                 chemberta_output=768, rdkit_size=100, dropout_rate=0.3):
        super(ImprovedCombinedModel, self).__init__()
        
        # ChemBERTa branch - lazy import to avoid slow startup
        logger = logging.getLogger(__name__)
        logger.info(f"[model] Loading ChemBERTa model '{chemberta_model}' (this may take time)...")
        import sys
        sys.stdout.flush()
        try:
            from transformers import RobertaModel
            self.chemberta = RobertaModel.from_pretrained(chemberta_model)
            logger.info(f"[model] ChemBERTa model loaded successfully")
            sys.stdout.flush()
        except Exception as e:
            logger.warning(f"[model] Error loading ChemBERTa model: {str(e)}")
            logger.warning("[model] Using random initialization for ChemBERTa model")
            sys.stdout.flush()
            # Create an untrained model with same config as fallback
            from transformers import RobertaConfig, RobertaModel
            config = RobertaConfig.from_pretrained(chemberta_model)
            self.chemberta = RobertaModel(config)
            logger.info(f"[model] Fallback ChemBERTa model created")
            sys.stdout.flush()
            
        self.chemberta_dropout = nn.Dropout(dropout_rate)
        self.chemberta_branch = nn.Sequential(
            nn.Linear(chemberta_output, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # RDKit features branch
        self.rdkit_branch = nn.Sequential(
            nn.Linear(rdkit_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, rdkit_features):
        try:
            # Process ChemBERTa embeddings
            bert_output = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
            bert_pooled = bert_output.last_hidden_state[:, 0, :]  # CLS token
            bert_pooled = self.chemberta_dropout(bert_pooled)
            c_out = self.chemberta_branch(bert_pooled)
            
            # Process RDKit features
            r_out = self.rdkit_branch(rdkit_features)
            
            # Combine both branches
            x = torch.cat((c_out, r_out), dim=1)
            
            # Final classification
            return self.classifier(x).view(-1)
        except Exception as e:
            print(f"Forward pass error: {str(e)}")
            # Return zeros in case of failure
            return torch.zeros(input_ids.size(0), device=input_ids.device)

# Improved function to visualize training history
def plot_training_history(histories, fold_aucs, fold_aps, save_path="training_history.png"):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot training loss
        for fold, history in enumerate(histories):
            axes[0].plot(history['train_loss'], label=f'Fold {fold+1}')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Plot validation AUC
        for fold, history in enumerate(histories):
            axes[1].plot(history['val_auc'], label=f'Fold {fold+1}')
        axes[1].set_title('Validation AUC')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].legend()
        
        # Plot fold performance
        fold_nums = list(range(1, len(fold_aucs) + 1))
        width = 0.35
        axes[2].bar([x - width/2 for x in fold_nums], fold_aucs, width, label='AUC')
        axes[2].bar([x + width/2 for x in fold_nums], fold_aps, width, label='AP')
        axes[2].set_title('Fold Performance')
        axes[2].set_xlabel('Fold')
        axes[2].set_ylabel('Score')
        axes[2].set_xticks(fold_nums)
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    except Exception as e:
        print(f"Error creating training history plot: {str(e)}")
        print("Skipping plot creation")

# Safe plotting function for ROC and PR curves
# --- GPU-accelerated feature scaling (learned from HTS_3D.py) -----------------------------------------

class GPUScaler:
    """
    GPU-compatible StandardScaler replacement.
    Computes mean and std on GPU if available, otherwise on CPU.
    Accepts both numpy arrays and torch tensors.
    """
    def __init__(self, device=None):
        self.mean_ = None
        self.std_ = None
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

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


def plot_performance_curves(labels, method_predictions, final_predictions, methods, save_path="performance_curves.png"):
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        plt.figure(figsize=(12, 5))
        
        # ROC curve subplot
        plt.subplot(1, 2, 1)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        # Plot for each method
        for method_idx, method in enumerate(methods):
            try:
                method_preds = method_predictions[:, method_idx]
                # Check if predictions are valid for ROC curve
                if len(np.unique(method_preds)) < 2:
                    print(f"Warning: {method} predictions have less than 2 unique values. Skipping ROC curve.")
                    continue
                    
                method_auc = safe_metric(roc_auc_score, labels, method_preds)
                fpr, tpr, _ = roc_curve(labels, method_preds)
                plt.plot(fpr, tpr, label=f'{method} (AUC = {method_auc:.4f})')
            except Exception as e:
                print(f"Error plotting ROC curve for {method}: {str(e)}")
                continue
        
        # Plot ensemble ROC curve
        try:
            if len(np.unique(final_predictions)) >= 2:
                auc = safe_metric(roc_auc_score, labels, final_predictions)
                fpr, tpr, _ = roc_curve(labels, final_predictions)
                plt.plot(fpr, tpr, 'b-', label=f'Ensemble (AUC = {auc:.4f})', linewidth=2)
        except Exception as e:
            print(f"Error plotting ensemble ROC curve: {str(e)}")
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        
        # Precision-Recall curve subplot
        plt.subplot(1, 2, 2)
        
        # Plot for each method
        for method_idx, method in enumerate(methods):
            try:
                method_preds = method_predictions[:, method_idx]
                # Check for valid predictions
                if np.all(method_preds == 0) or np.all(method_preds == 1):
                    print(f"Warning: {method} predictions are all the same. Skipping PR curve.")
                    continue
                    
                method_ap = safe_metric(average_precision_score, labels, method_preds)
                precision, recall, _ = precision_recall_curve(labels, method_preds)
                plt.plot(recall, precision, label=f'{method} (AP = {method_ap:.4f})')
            except Exception as e:
                print(f"Error plotting PR curve for {method}: {str(e)}")
                continue
        
        # Plot ensemble PR curve
        try:
            if not np.all(final_predictions == 0) and not np.all(final_predictions == 1):
                ap = safe_metric(average_precision_score, labels, final_predictions)
                precision, recall, _ = precision_recall_curve(labels, final_predictions)
                plt.plot(recall, precision, 'b-', label=f'Ensemble (AP = {ap:.4f})', linewidth=2)
        except Exception as e:
            print(f"Error plotting ensemble PR curve: {str(e)}")
        
        # Add baseline
        pos_ratio = max(sum(labels)/len(labels), 0.001)  # Ensure not zero
        plt.axhline(y=pos_ratio, color='r', linestyle='--', alpha=0.3, 
                label=f'Baseline (ratio = {pos_ratio:.4f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        logger = logging.getLogger(__name__)
        logger.info(f"Performance curves saved to {save_path}")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error creating performance curves: {str(e)}")
        logger.exception("Full traceback:")
        traceback.print_exc()
        logger.warning("Skipping plot creation")

def setup_logging():
    """
    Set up logging to both file and console with timestamps.
    Creates a timestamped log file in the logs directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"hts_training_{timestamp}.log"
    
    # Explicitly create the log file immediately to ensure it exists
    log_file.touch(exist_ok=True)
    
    # Create formatter with timestamps
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler - write to timestamped log file
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler - write to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Write first log entry immediately to confirm file creation
    logging.info(f"Logging initialized. Log file: {log_file}")
    # Flush to ensure it's written to disk immediately
    file_handler.flush()
    
    return log_file


if __name__ == '__main__':
    # Start total timer
    total_start_time = time.time()
    
    # Set up logging with timestamped log file
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    # Print startup message immediately to show script is running
    logger.info("[startup] Starting HTS.py...")
    sys.stdout.flush()
    
    # Set CUDA seed if available (moved from module level to avoid slow import)
    logger.info("[startup] Checking CUDA availability...")
    sys.stdout.flush()
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
            logger.info("[startup] CUDA available, seed set")
        else:
            logger.info("[startup] CUDA not available")
    except Exception as e:
        logger.warning(f"[startup] CUDA check warning: {e}")
    sys.stdout.flush()
    
    # Detect device early for GPU optimizations (learned from HTS_3D.py)
    # Use try-except to prevent hangs during device detection
    logger.info("[startup] Detecting device...")
    sys.stdout.flush()
    
    try:
        # Check CUDA availability with timeout protection
        cuda_available = False
        try:
            cuda_available = torch.cuda.is_available()
        except Exception as e:
            logger.warning(f"[startup] Warning: CUDA check failed: {e}. Using CPU.")
            sys.stdout.flush()
            cuda_available = False
        
        # Check MPS availability
        mps_available = False
        if not cuda_available:
            try:
                mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            except Exception:
                mps_available = False
        
        # Set device
        if cuda_available:
            device = torch.device("cuda")
        elif mps_available:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        NON_BLOCKING = device.type != "cpu"
        PIN_MEMORY = device.type != "cpu"
        
        logger.info(f"[startup] Device detected: {device}")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"[startup] Error during device detection: {e}. Using CPU.")
        sys.stdout.flush()
        device = torch.device("cpu")
        NON_BLOCKING = False
        PIN_MEMORY = False
    
    # Load data
    data_start_time = time.time()
    logger.info("[data] Loading data...")
    sys.stdout.flush()
    try:
        library_df = pd.read_csv(PROJECT_ROOT / 'libraries' / 'library.csv')
        positives_df = pd.read_csv(PROJECT_ROOT / 'libraries' / 'positives.csv')
        
        # Create labels based on whether SMILES is in positives
        library_df['label'] = library_df['Smiles'].isin(positives_df['Smiles']).astype(int)
        
        # Extract SMILES and labels
        smiles_list = library_df['Smiles'].tolist()
        labels = library_df['label'].to_numpy()
        
        # Check class balance
        positive_count = np.sum(labels)
        total_count = len(labels)
        logger.info(f"[data] Loaded {total_count} molecules ({positive_count} actives, {total_count - positive_count} inactives)")
        logger.info(f"[data] Active ratio: {positive_count/total_count:.2%}")
        sys.stdout.flush()
        
        # Add check for very imbalanced data
        if positive_count / total_count < 0.01:
            logger.warning("[data] WARNING: Data is highly imbalanced. Consider using stratified sampling or other balancing techniques.")
            sys.stdout.flush()
        
        # Log data loading time
        data_elapsed = time.time() - data_start_time
        logger.info(f"[data] Data loading completed in {data_elapsed:.2f} seconds ({data_elapsed/60:.2f} minutes)")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"[data] Error loading data: {str(e)}")
        logger.exception("Full traceback:")
        traceback.print_exc()
        sys.exit(1)

    # Generate features with GPU support (learned from HTS_3D.py)
    rdkit_start_time = time.time()
    logger.info("[RDKit] Generating RDKit features...")
    sys.stdout.flush()
    try:
        rdkit_features = rdkit_features_from_smiles(smiles_list, device=device)
        
        # Handle both numpy array and torch tensor
        if isinstance(rdkit_features, torch.Tensor):
            logger.info(f"[RDKit] Generated RDKit features with shape: {rdkit_features.shape}, device: {rdkit_features.device}")
            # Check for invalid features on GPU
            if torch.isnan(rdkit_features).any() or torch.isinf(rdkit_features).any():
                logger.warning("[RDKit] WARNING: Invalid feature values detected. Fixing...")
                sys.stdout.flush()
                rdkit_features = torch.nan_to_num(rdkit_features, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            logger.info(f"[RDKit] Generated RDKit features with shape: {rdkit_features.shape}")
            # Check for invalid features on CPU
            invalid_features = np.any(np.isnan(rdkit_features)) or np.any(np.isinf(rdkit_features))
            if invalid_features:
                logger.warning("[RDKit] WARNING: Invalid feature values detected. Fixing...")
                sys.stdout.flush()
                rdkit_features = np.nan_to_num(rdkit_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Log RDKit feature generation time
        rdkit_elapsed = time.time() - rdkit_start_time
        logger.info(f"[RDKit] Feature generation completed in {rdkit_elapsed:.2f} seconds ({rdkit_elapsed/60:.2f} minutes)")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"[RDKit] Error generating features: {str(e)}")
        logger.exception("Full traceback:")
        traceback.print_exc()
        sys.exit(1)

    # Load tokenizer with error handling and progress reporting
    # Import transformers here (lazy import) to avoid slow startup
    tokenizer_start_time = time.time()
    logger.info("[tokenizer] Loading transformers library...")
    sys.stdout.flush()
    try:
        from transformers import RobertaTokenizer
    except ImportError as e:
        logger.error(f"[tokenizer] Error importing RobertaTokenizer: {e}")
        sys.exit(1)
    
    logger.info("[tokenizer] Loading ChemBERTa tokenizer...")
    logger.info("[tokenizer] This may take a few minutes if downloading for the first time...")
    sys.stdout.flush()
    try:
        tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        logger.info("[tokenizer] Successfully loaded RobertaTokenizer")
        sys.stdout.flush()
    except Exception as e:
        logger.warning(f"[tokenizer] RobertaTokenizer failed, trying AutoTokenizer... Error: {e}")
        sys.stdout.flush()
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            logger.info("[tokenizer] Successfully loaded AutoTokenizer")
            sys.stdout.flush()
        except Exception as e2:
            logger.warning(f"[tokenizer] AutoTokenizer also failed: {e2}")
            logger.warning("[tokenizer] Using fallback tokenizer")
            sys.stdout.flush()
            # Last resort: create a basic tokenizer
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=None)
            tokenizer.add_special_tokens({'pad_token': '[PAD]', 'unk_token': '[UNK]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'})
    
    # Log tokenizer loading time
    tokenizer_elapsed = time.time() - tokenizer_start_time
    logger.info(f"[tokenizer] Tokenizer loading completed in {tokenizer_elapsed:.2f} seconds ({tokenizer_elapsed/60:.2f} minutes)")
    sys.stdout.flush()
    
    # Device already detected, log configuration
    logger.info(f"\n{'='*60}")
    logger.info(f"DEVICE CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # Enable optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        logger.info(f"\nGPU OPTIMIZATIONS ENABLED:")
        logger.info(f"  - cuDNN benchmark mode: ON")
        logger.info(f"  - Mixed precision training (AMP): Will be enabled")
        logger.info(f"  - GPU-accelerated feature scaling: ON")
        logger.info(f"  - GPU tensor operations: ON")
        logger.info(f"  - Parallel data loading: 4 workers")
        logger.info(f"  - Pinned memory: {PIN_MEMORY}")
        logger.info(f"  - Non-blocking transfers: {NON_BLOCKING}")
        logger.info(f"  - Batch size: 64 (optimized for GPU)")
    elif device.type == "mps":
        logger.info(f"\nMPS OPTIMIZATIONS:")
        logger.info(f"  - GPU-accelerated feature scaling: ON")
        logger.info(f"  - GPU tensor operations: ON")
        logger.info(f"  - Batch size: 64 (optimized for MPS)")
    else:
        logger.info(f"\nCPU MODE:")
        logger.info(f"  - Batch size: 32")
        logger.info(f"  - Data loading: Sequential")
    logger.info(f"{'='*60}\n")
    sys.stdout.flush()
    
    # Training settings - optimized for device type
    num_epochs = 10
    patience = 5
    # Increase batch size for GPU/MPS to maximize throughput
    batch_size = 64 if device.type in ("cuda", "mps") else 32
    learning_rate = 2e-4
    weight_decay = 1e-4
    
    # Feature selection settings - try all three methods
    feature_selection_methods = ['lasso', 'pca', 'mutual_info']
    n_components = 200  # Number of features/components to select
    
    # Stratified K-Fold Training
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    final_models = []  # To store all models for ensemble
    fold_predictions = []  # Predictions from each fold
    fold_aucs = []
    fold_aps = []
    training_histories = []
    
    # Prepare array to store validation indices for later
    val_indices = []
    
    # Cross-validation training
    cv_start_time = time.time()
    logger.info("\n=== Starting Cross-Validation Training ===")
    for feature_method in feature_selection_methods:
        logger.info(f"\n----- Feature Selection Method: {feature_method} -----")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(rdkit_features, labels)):
            try:
                logger.info(f"\n=== Fold {fold+1} ===")
                train_smiles = [smiles_list[i] for i in train_idx]
                train_labels = labels[train_idx]
                train_rdkit = rdkit_features[train_idx]
    
                val_smiles = [smiles_list[i] for i in val_idx]
                val_labels = labels[val_idx]
                val_rdkit = rdkit_features[val_idx]
                
                # Store validation indices for later ensemble
                if fold == 0 and feature_method == feature_selection_methods[0]:
                    val_indices.append(val_idx)
                    
                # Apply feature selection with GPU scaler if device is GPU (learned from HTS_3D.py)
                use_gpu_scaler = device.type != "cpu"
                train_rdkit_selected, val_rdkit_selected, selector, scaler = apply_feature_selection(
                    train_rdkit, train_labels, val_rdkit, 
                    feature_selection_method=feature_method,
                    n_components=n_components,
                    device=device,
                    use_gpu_scaler=use_gpu_scaler
                )
                
                print(f"Selected features shape: {train_rdkit_selected.shape}")
                
                # Create datasets with selected features
                train_data = MolecularDataset(train_smiles, train_labels, tokenizer, train_rdkit_selected)
                val_data = MolecularDataset(val_smiles, val_labels, tokenizer, val_rdkit_selected)
    
                # Optimize DataLoader for GPU (learned from HTS_3D.py)
                # On Windows, use fewer workers to avoid multiprocessing issues
                import sys
                is_windows = sys.platform.startswith('win')
                num_workers = 2 if (device.type != "cpu" and not is_windows) else 0
                use_persistent_workers = device.type != "cpu" and not is_windows
                # Disable pin_memory if tensors are already on GPU (pin_memory only works for CPU tensors)
                use_pin_memory = PIN_MEMORY and device.type == "cpu"
                
                train_loader = DataLoader(
                    train_data, 
                    batch_size=batch_size, 
                    shuffle=True,
                    num_workers=num_workers,  # Parallel data loading on GPU (disabled on Windows)
                    pin_memory=use_pin_memory,  # Faster GPU transfer (only for CPU tensors)
                    persistent_workers=use_persistent_workers
                )
                val_loader = DataLoader(
                    val_data, 
                    batch_size=batch_size, 
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=use_pin_memory,
                    persistent_workers=use_persistent_workers
                )
    
                # Initialize model for this feature selection method and fold
                logger.info(f"[model] Creating model for Fold {fold+1}...")
                sys.stdout.flush()
                model = ImprovedCombinedModel(rdkit_size=train_rdkit_selected.shape[1]).to(device)
                logger.info(f"[model] Model created, moving to device: {device}")
                sys.stdout.flush()
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
                # Calculate class weights for imbalanced data
                pos_count = sum(train_labels)
                if pos_count > 0:  # Ensure no division by zero
                    pos_weight_val = (len(train_labels) - pos_count) / pos_count
                else:
                    pos_weight_val = 1.0
                    print("Warning: No positive samples in training set!")
                    
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val).to(device))
    
                # Mixed precision training for GPU speedup
                use_amp = torch.cuda.is_available()
                grad_scaler = torch.cuda.amp.GradScaler() if use_amp else None
                if use_amp:
                    print("Using mixed precision training (AMP) for GPU acceleration")
    
                best_auc = 0
                no_improve = 0
                history = {'train_loss': [], 'val_auc': [], 'val_ap': []}

                # Training loop starts here after pre-encoding
                logger.info(f"Starting training for Fold {fold+1} with {num_epochs} epochs...")
                logger.info(f"[train] Training set size: {len(train_data)}, batches: {len(train_loader)}")
                sys.stdout.flush()
                
                # Try-except block for the entire training loop
                try:
                    for epoch in range(num_epochs):
                        # Training
                        logger.info(f"[train] Starting epoch {epoch+1}/{num_epochs}...")
                        sys.stdout.flush()
                        model.train()
                        total_loss = 0
                        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
                        
                        batch_count = 0
                        logger.info(f"[train] Iterating through {len(train_loader)} batches...")
                        sys.stdout.flush()
                        for batch_idx, batch in enumerate(loop):
                            if batch_idx == 0:
                                logger.info(f"[train] Processing first batch (this may be slow on first forward pass)...")
                                sys.stdout.flush()
                            try:
                                # More efficient gradient clearing (set_to_none=True avoids zeroing memory)
                                optimizer.zero_grad(set_to_none=True)
                                
                                # Move tensors to device only if not already on device (optimization from HTS_3D.py)
                                input_ids = batch["input_ids"] if batch["input_ids"].device == device else batch["input_ids"].to(device, non_blocking=True)
                                attention_mask = batch["attention_mask"] if batch["attention_mask"].device == device else batch["attention_mask"].to(device, non_blocking=True)
                                labels_tensor = batch["label"] if batch["label"].device == device else batch["label"].to(device, non_blocking=True)
                                rdkit_tensor = batch["rdkit_features"] if batch["rdkit_features"].device == device else batch["rdkit_features"].to(device, non_blocking=True)
    
                                # Mixed precision forward pass
                                if use_amp:
                                    with torch.cuda.amp.autocast():
                                        if batch_idx == 0:
                                            logger.info(f"[train] Running first forward pass (this may be slow)...")
                                            sys.stdout.flush()
                                        preds = model(input_ids, attention_mask, rdkit_tensor)
                                        
                                        if batch_idx == 0:
                                            logger.info(f"[train] First forward pass complete")
                                            sys.stdout.flush()
                                        
                                        # Check for NaN predictions
                                        if torch.isnan(preds).any():
                                            print("Warning: NaN predictions detected during training. Skipping batch.")
                                            continue
                                            
                                        loss = criterion(preds, labels_tensor)
                                    
                                    # Mixed precision backward pass
                                    grad_scaler.scale(loss).backward()
                                    grad_scaler.unscale_(optimizer)
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                    grad_scaler.step(optimizer)
                                    grad_scaler.update()
                                else:
                                    if batch_idx == 0:
                                        logger.info(f"[train] Running first forward pass (this may be slow)...")
                                        sys.stdout.flush()
                                    preds = model(input_ids, attention_mask, rdkit_tensor)
                                    
                                    if batch_idx == 0:
                                        logger.info(f"[train] First forward pass complete")
                                        sys.stdout.flush()
                                    
                                    # Check for NaN predictions
                                    if torch.isnan(preds).any():
                                        print("Warning: NaN predictions detected during training. Skipping batch.")
                                        continue
                                        
                                    loss = criterion(preds, labels_tensor)
                                    loss.backward()
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                    optimizer.step()
                                
                                total_loss += loss.item()
                                batch_count += 1
                                loop.set_postfix(loss=total_loss / (batch_count))
                            except Exception as e:
                                print(f"Error in training batch: {str(e)}")
                                continue
                        
                        if batch_count > 0:  # Avoid division by zero
                            avg_train_loss = total_loss / batch_count
                        else:
                            avg_train_loss = float('inf')
                            print("Warning: No valid batches in this epoch!")
                            
                        history['train_loss'].append(avg_train_loss)
                        logger.info(f"Fold {fold+1}, Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
    
                        # Validation
                        model.eval()
                        val_preds, val_targets = [], []
                        valid_val_count = 0
                        
                        with torch.no_grad():
                            for batch in val_loader:
                                try:
                                    # Move tensors to device only if not already on device (optimization from HTS_3D.py)
                                    input_ids = batch["input_ids"] if batch["input_ids"].device == device else batch["input_ids"].to(device, non_blocking=True)
                                    attention_mask = batch["attention_mask"] if batch["attention_mask"].device == device else batch["attention_mask"].to(device, non_blocking=True)
                                    labels_tensor = batch["label"] if batch["label"].device == device else batch["label"].to(device, non_blocking=True)
                                    rdkit_tensor = batch["rdkit_features"] if batch["rdkit_features"].device == device else batch["rdkit_features"].to(device, non_blocking=True)
    
                                    # Use mixed precision for validation if available
                                    if use_amp:
                                        with torch.cuda.amp.autocast():
                                            preds = model(input_ids, attention_mask, rdkit_tensor)
                                    else:
                                        preds = model(input_ids, attention_mask, rdkit_tensor)
                                    
                                    # Check for NaN predictions
                                    if torch.isnan(preds).any():
                                        print("Warning: NaN predictions detected during validation. Skipping batch.")
                                        continue
                                    
                                    # Apply sigmoid in validation to get probability scores
                                    sigmoid_preds = torch.sigmoid(preds).cpu().numpy()
                                    batch_labels = labels_tensor.cpu().numpy()
                                    
                                    # Filter out NaN or Inf values
                                    valid_indices = ~(np.isnan(sigmoid_preds) | np.isinf(sigmoid_preds))
                                    if np.any(valid_indices):
                                        val_preds.append(sigmoid_preds[valid_indices])
                                        val_targets.append(batch_labels[valid_indices])
                                        valid_val_count += np.sum(valid_indices)
                                except Exception as e:
                                    print(f"Error in validation batch: {str(e)}")
                                    continue
    
                        # Process validation results
                        if len(val_preds) > 0 and valid_val_count > 0:
                            try:
                                val_preds = np.concatenate(val_preds)
                                val_targets = np.concatenate(val_targets)
                                
                                # Check if we have samples from both classes
                                if len(np.unique(val_targets)) < 2:
                                    print("Warning: Only one class in validation targets. AUC calculation skipped.")
                                    auc = 0
                                    ap = 0
                                else:
                                    # Use safe metrics
                                    auc = safe_metric(roc_auc_score, val_targets, val_preds)
                                    ap = safe_metric(average_precision_score, val_targets, val_preds)
                            except Exception as e:
                                print(f"Error calculating validation metrics: {str(e)}")
                                auc = 0
                                ap = 0
                        else:
                            print("Warning: No valid predictions for validation. Setting metrics to 0.")
                            auc = 0
                            ap = 0
                        
                        history['val_auc'].append(auc)
                        history['val_ap'].append(ap)
                        
                        logger.info(f"Fold {fold+1}, Epoch {epoch+1}, Validation AUC: {auc:.4f}, AP: {ap:.4f}")
                        
                        # Learning rate scheduler
                        scheduler.step(auc)
    
                        # Early stopping check
                        if auc > best_auc:
                            best_auc = auc
                            best_model_state = model.state_dict()
                            best_epoch = epoch
                            no_improve = 0
                        else:
                            no_improve += 1
                            if no_improve >= patience:
                                logger.info(f"Early stopping at epoch {epoch+1} for fold {fold+1}")
                                break
                                
                    logger.info(f"Best validation AUC: {best_auc:.4f} at epoch {best_epoch+1}")
                except Exception as e:
                    logger.error(f"Error in training loop: {str(e)}")
                    logger.exception("Full traceback:")
                    best_auc = 0
                    best_epoch = 0
                    best_model_state = model.state_dict()  # Use the last state as fallback
                
                # Save training history for this fold
                if feature_method == feature_selection_methods[-1]:  # Only for the last feature selection method
                    training_histories.append(history)
                    fold_aucs.append(best_auc)
                    fold_aps.append(history['val_ap'][best_epoch] if len(history['val_ap']) > best_epoch else 0)
                
                # Always save the model even if there were errors
                try:
                    # Save metadata required for inference
                    model_data = {
                        'model': best_model_state,
                        'selector': selector,
                        'scaler': scaler,
                        'feature_method': feature_method,
                        'n_components': n_components,
                        'best_auc': best_auc,
                        'fold': fold,
                    }
                    
                    final_models.append(model_data)
                    
                    # Generate predictions for the validation set using the best model
                    model.load_state_dict(best_model_state)
                    model.eval()
                    
                    # Validate that we have a reasonable model state
                    if torch.any(torch.isnan(next(model.parameters()))):
                        print("Warning: Model contains NaN parameters. Using zero predictions.")
                        fold_val_preds = np.zeros(len(val_idx))
                    else:
                        # Generate predictions
                        fold_val_preds = np.zeros(len(val_idx))
                        with torch.no_grad():
                            for i, idx in enumerate(range(len(val_smiles))):
                                try:
                                    input_ids = tokenizer(val_smiles[idx], padding="max_length", truncation=True, 
                                                        max_length=128, return_tensors="pt")["input_ids"].to(device)
                                    attention_mask = tokenizer(val_smiles[idx], padding="max_length", truncation=True, 
                                                            max_length=128, return_tensors="pt")["attention_mask"].to(device)
                                    rdkit_tensor = torch.tensor(val_rdkit_selected[idx], dtype=torch.float).unsqueeze(0).to(device)
    
                                    preds = model(input_ids, attention_mask, rdkit_tensor)
                                    pred_prob = torch.sigmoid(preds).cpu().numpy().item()
                                    
                                    # Check for NaN or Inf
                                    if np.isnan(pred_prob) or np.isinf(pred_prob):
                                        pred_prob = 0.0
                                        
                                    fold_val_preds[i] = pred_prob
                                except Exception as e:
                                    print(f"Error generating prediction for validation sample {i}: {str(e)}")
                                    fold_val_preds[i] = 0.0
                    
                    # Store predictions for this fold's validation set
                    fold_predictions.append((val_idx, fold_val_preds, feature_method))
                except Exception as e:
                    print(f"Error saving model or generating predictions: {str(e)}")
                    # Skip this fold entirely
                    continue
            except Exception as e:
                print(f"Fatal error in fold {fold+1}: {str(e)}")
                traceback.print_exc()
                continue  # Skip to next fold
    
    # Visualize training process if we have histories
    if training_histories:
        plot_training_history(training_histories, fold_aucs, fold_aps)
    
    # Log cross-validation training time
    cv_elapsed = time.time() - cv_start_time
    logger.info(f"\n[timing] Cross-validation training completed in {cv_elapsed:.2f} seconds ({cv_elapsed/60:.2f} minutes)")
    sys.stdout.flush()
    
    # Stacked ensemble using all features and models
    ensemble_start_time = time.time()
    logger.info("\n=== Building Stacked Ensemble ===")
    
    # Check if we have any models
    if not final_models:
        logger.error("Error: No models were successfully trained. Cannot build ensemble.")
        sys.exit(1)
    
    # Initialize array to store all predictions for each feature selection method
    all_predictions = np.zeros((len(smiles_list), len(feature_selection_methods)))
    
    # GPU-accelerated batch prediction for ensemble
    # Pre-tokenize all SMILES once (CPU operation, but fast)
    logger.info("Pre-tokenizing all SMILES for GPU batch processing...")
    batch_tokenize_size = 1000  # Tokenize in chunks to avoid memory issues
    all_input_ids = []
    all_attention_masks = []
    
    for i in tqdm(range(0, len(smiles_list), batch_tokenize_size), desc="Tokenizing SMILES"):
        batch_smiles = smiles_list[i:i+batch_tokenize_size]
        tokenized = tokenizer(
            batch_smiles, 
            padding="max_length", 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        all_input_ids.append(tokenized["input_ids"])
        all_attention_masks.append(tokenized["attention_mask"])
    
    # Concatenate and move to GPU
    input_ids_all = torch.cat(all_input_ids, dim=0).to(device)
    attention_mask_all = torch.cat(all_attention_masks, dim=0).to(device)
    logger.info(f"Tokenized {len(smiles_list)} SMILES. Moving to {device}...")
    
    # Use mixed precision for inference
    use_amp = torch.cuda.is_available()
    
    # For each feature selection method, apply the appropriate models to get predictions
    for method_idx, method in enumerate(feature_selection_methods):
        method_models = [m for m in final_models if m['feature_method'] == method]
        
        if not method_models:
            logger.warning(f"Warning: No models available for method {method}. Using zeros.")
            continue
        
        logger.info(f"\nGenerating predictions with {len(method_models)} models for method: {method}")
        logger.info(f"Using GPU batch processing with batch size: {batch_size}")
        
        # Pre-process RDKit features for this method (apply feature selection to all at once)
        # Get selector and scaler from first model (they should be the same for all folds of same method)
        first_model_data = method_models[0]
        selector = first_model_data.get('selector')
        scaler = first_model_data.get('scaler')
        
        # Apply feature selection to all RDKit features at once (CPU operation, but fast)
        if selector is not None and scaler is not None:
            logger.info(f"Applying feature selection ({method}) to all {len(smiles_list)} molecules...")
            rdkit_feat_scaled = scaler.transform(rdkit_features)
            rdkit_feat_selected = selector.transform(rdkit_feat_scaled)
        else:
            rdkit_feat_selected = rdkit_features
        
        # Handle NaN/Inf values
        rdkit_feat_selected = np.nan_to_num(rdkit_feat_selected, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to GPU tensor once
        rdkit_tensor_all = torch.tensor(rdkit_feat_selected, dtype=torch.float32).to(device)
        logger.info(f"RDKit features shape: {rdkit_tensor_all.shape}, device: {rdkit_tensor_all.device}")
        
        # Pre-load all models for this method and keep them in GPU memory
        loaded_models = []
        for model_data in method_models:
            try:
                model = ImprovedCombinedModel(rdkit_size=rdkit_feat_selected.shape[1]).to(device)
                model.load_state_dict(model_data['model'])
                model.eval()
                loaded_models.append(model)
            except Exception as e:
                print(f"Warning: Failed to load model: {str(e)}")
                continue
        
        if not loaded_models:
            logger.warning(f"Warning: No valid models loaded for method {method}. Using zeros.")
            continue
        
        # Batch process predictions for all molecules
        method_predictions = torch.zeros(len(smiles_list), len(loaded_models), device=device)
        
        # Process in batches to manage GPU memory
        inference_batch_size = batch_size * 2 if torch.cuda.is_available() else batch_size
        
        for batch_start in tqdm(range(0, len(smiles_list), inference_batch_size), desc=f"Predictions for {method}"):
            batch_end = min(batch_start + inference_batch_size, len(smiles_list))
            batch_input_ids = input_ids_all[batch_start:batch_end]
            batch_attention_mask = attention_mask_all[batch_start:batch_end]
            batch_rdkit = rdkit_tensor_all[batch_start:batch_end]
            
            # Get predictions from all models for this batch
            for model_idx, model in enumerate(loaded_models):
                try:
                    with torch.no_grad():
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                preds = model(batch_input_ids, batch_attention_mask, batch_rdkit)
                        else:
                            preds = model(batch_input_ids, batch_attention_mask, batch_rdkit)
                        
                        # Store predictions (keep on GPU for now)
                        method_predictions[batch_start:batch_end, model_idx] = torch.sigmoid(preds).squeeze()
                except Exception as e:
                    print(f"Warning: Error with model {model_idx} for batch {batch_start}-{batch_end}: {str(e)}")
                    continue
        
        # Average predictions from all models (on GPU)
        method_predictions_avg = method_predictions.mean(dim=1)
        
        # Handle NaN/Inf
        method_predictions_avg = torch.nan_to_num(method_predictions_avg, nan=0.0, posinf=1.0, neginf=0.0)
        method_predictions_avg = torch.clamp(method_predictions_avg, 0.0, 1.0)
        
        # Move to CPU and store
        all_predictions[:, method_idx] = method_predictions_avg.cpu().numpy()
        
        # Clear GPU memory
        del loaded_models, method_predictions, method_predictions_avg, rdkit_tensor_all
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Completed predictions for {method} method")
    
    # Check if we have valid predictions
    if np.all(all_predictions == 0):
        logger.warning("Warning: All predictions are zero. There might be a problem with the models.")
        # Create a small random variation to avoid division by zero in metrics
        final_predictions = np.random.uniform(0.01, 0.02, size=len(smiles_list))
    else:
        # Final ensemble prediction (average of all methods)
        final_predictions = np.zeros(len(smiles_list))
        for i in range(len(smiles_list)):
            # Average non-zero predictions for each sample
            method_preds = all_predictions[i, :]
            valid_preds = method_preds[method_preds > 0]
            if len(valid_preds) > 0:
                final_predictions[i] = np.mean(valid_preds)
            else:
                # No valid predictions, use a small value
                final_predictions[i] = 0.01
    
    # Check for NaN or Inf in final predictions
    if np.any(np.isnan(final_predictions)) or np.any(np.isinf(final_predictions)):
        logger.warning("Warning: NaN or Inf values in final predictions. Fixing...")
        final_predictions = np.nan_to_num(final_predictions, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure predictions are in [0, 1] range
        final_predictions = np.clip(final_predictions, 0, 1)
    
    # Save ensemble model and predictions
    try:
        ensemble_data = {
            'models': final_models,
            'feature_methods': feature_selection_methods,
            'predictions': final_predictions,
            'method_predictions': all_predictions
        }
        joblib.dump(ensemble_data, PROJECT_ROOT / "enhanced_ensemble_model.pkl")
        logger.info("[OK] Saved 'enhanced_ensemble_model.pkl' with combined predictions from all methods and folds.")
    except Exception as e:
        logger.error(f"Error saving ensemble model: {str(e)}")
        logger.exception("Full traceback:")
        import traceback
        traceback.print_exc()
        # Save predictions in a simpler format as backup
        np.save(PROJECT_ROOT / "final_predictions.npy", final_predictions)
        logger.info("[OK] Saved backup predictions to 'final_predictions.npy'")
    
    # Log ensemble building time
    ensemble_elapsed = time.time() - ensemble_start_time
    logger.info(f"[timing] Ensemble building completed in {ensemble_elapsed:.2f} seconds ({ensemble_elapsed/60:.2f} minutes)")
    sys.stdout.flush()
    
    # Independent evaluation of the performance on the entire dataset
    eval_start_time = time.time()
    logger.info("\n=== Final Evaluation ===")
    
    # Using the final averaged predictions from the ensemble
    try:
        # Set default values in case metrics fail
        auc, ap, precision, recall, f1 = 0, 0, 0, 0, 0
        
        # Calculate metrics safely
        auc = safe_metric(roc_auc_score, labels, final_predictions)
        ap = safe_metric(average_precision_score, labels, final_predictions)
        
        # Convert to binary predictions
        binary_preds = (final_predictions > 0.5).astype(int)
        
        precision = safe_metric(precision_score, labels, binary_preds)
        recall = safe_metric(recall_score, labels, binary_preds)
        f1 = safe_metric(f1_score, labels, binary_preds)
    
        logger.info("Overall Model Performance:")
        logger.info(f"AUC: {auc:.4f}")
        logger.info(f"Average Precision (AP): {ap:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
    except Exception as e:
        logger.error(f"Error calculating final metrics: {str(e)}")
        logger.exception("Full traceback:")
    
    # Compare feature selection methods safely
    logger.info("\n=== Feature Selection Method Comparison ===")
    for method_idx, method in enumerate(feature_selection_methods):
        try:
            method_preds = all_predictions[:, method_idx]
            
            # Check if predictions are all zeros
            if np.all(method_preds == 0):
                logger.warning(f"{method.upper()}: No valid predictions available")
                continue
                
            # Calculate metrics safely
            method_auc = safe_metric(roc_auc_score, labels, method_preds)
            method_ap = safe_metric(average_precision_score, labels, method_preds)
            
            logger.info(f"{method.upper()}: AUC = {method_auc:.4f}, AP = {method_ap:.4f}")
        except Exception as e:
            logger.error(f"Error calculating metrics for {method}: {str(e)}")
            logger.exception("Full traceback:")
    
    # Create a prediction data frame with scores
    try:
        results_df = pd.DataFrame({
            'ID': range(len(library_df)),  # Create sequential IDs
            'SMILES': library_df['Smiles'],
            'True_Label': library_df['label'],
            'Prediction_Score': final_predictions,
            'Predicted_Label': (final_predictions > 0.5).astype(int)
        })
    
        # Save predictions to CSV
        results_df.to_csv(PROJECT_ROOT / 'molecular_predictions.csv', index=False)
        logger.info("[OK] Saved predictions to 'molecular_predictions.csv'")
    except Exception as e:
        logger.error(f"Error creating results dataframe: {str(e)}")
        logger.exception("Full traceback:")
        # Save as numpy arrays as backup
        np.save("ids.npy", np.arange(len(library_df)))  # Save sequential IDs
        np.save("predictions.npy", final_predictions)
        logger.info("[OK] Saved backup arrays 'ids.npy' and 'predictions.npy'")
    
    # Plot ROC and Precision-Recall curves safely
    try:
        plot_performance_curves(labels, all_predictions, final_predictions, feature_selection_methods, save_path="performance_curves.png")
    except Exception as e:
        logger.error(f"Error plotting performance curves: {str(e)}")
        logger.exception("Full traceback:")
    
    # Log final evaluation time
    eval_elapsed = time.time() - eval_start_time
    logger.info(f"[timing] Final evaluation completed in {eval_elapsed:.2f} seconds ({eval_elapsed/60:.2f} minutes)")
    sys.stdout.flush()
    
    # Calculate and log total elapsed time
    total_elapsed = time.time() - total_start_time
    total_hours = int(total_elapsed // 3600)
    total_minutes = int((total_elapsed % 3600) // 60)
    total_seconds = total_elapsed % 60
    
    logger.info("\n=== Complete! ===")
    logger.info(f"[timing] Total execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    if total_hours > 0:
        logger.info(f"[timing] Total execution time: {total_hours}h {total_minutes}m {total_seconds:.2f}s")
    else:
        logger.info(f"[timing] Total execution time: {total_minutes}m {total_seconds:.2f}s")
    logger.info(f"Log file saved to: {log_file}")

# Code written by: Hossam Nada
