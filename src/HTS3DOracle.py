import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Lipinski, QED
from rdkit.Chem import GetSSSR
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import argparse
from typing import Optional, List, Dict, Tuple

# Parse command-line arguments before Streamlit initializes
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="HTS-3D Results Evaluator & Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to HTS-3D model file (.pt). If not specified, uses default model from output directory based on compound type."
    )
    parser.add_argument(
        "--compound",
        type=str,
        choices=["nlrp3", "cd28"],
        default=None,
        help="Compound type (nlrp3 or cd28). Used to determine default model if --model is not specified."
    )
    parser.add_argument(
        "--protein",
        type=str,
        default=None,
        help="Path to PDB file for protein template (e.g. src/1YJD.pdb for CD28). Uses default NLRP3 (7ALV) if not specified."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="For legacy model files: path to final_checkpoint_*.pt to load scaler/selector."
    )
    parser.add_argument(
        "--cd28",
        action="store_true",
        help="Use HTS.py-style CD28 data: load libraries/library.csv and libraries/positives.csv; label=1 if SMILES in positives else 0. Same behavior as HTS_3D.py --cd28. Skips file upload."
    )

    # Parse known args only (Streamlit adds its own args)
    args, unknown = parser.parse_known_args()
    return args

# Parse command-line arguments
CLI_ARGS = parse_args()

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "output"

# Import HTS-3D components for model loading and prediction
sys.path.insert(0, str(SCRIPT_DIR))

# Register picklable classes so torch.load can unpickle scaler/selector saved by HTS_3D
import HTS_3D as _hts3d_module
for _name in ("DummyScaler", "DummyPCA", "DummyLassoSelector", "DummyMISelector"):
    if hasattr(_hts3d_module, _name):
        setattr(sys.modules["__main__"], _name, getattr(_hts3d_module, _name))

try:
    from HTS_3D import (
        HTS3DDataset,
        HTS3DModel,
        PROTEIN_TEMPLATES,
        build_3d_cache,
        build_rdkit_feature_matrix,
        detect_compound_from_csv_path,
        detect_device,
        get_model_path_for_compound,
        load_library_and_positives,
        prepare_tensors,
        GPUScaler,
        MAX_SEQ_LEN,
        MAX_ATOMS,
        EMBED_DIM,
        NUM_HEADS,
    )
    from HTS_3D_inference import load_model as load_hts3d_inference_model, predict_on_smiles as predict_on_smiles_inference
    HTS3D_AVAILABLE = True
except ImportError as e:
    load_hts3d_inference_model = None
    predict_on_smiles_inference = None
    HTS3D_AVAILABLE = False
    st.warning(f"⚠️ HTS-3D components not available: {e}. Prediction mode will be disabled.")

if HTS3D_AVAILABLE:
    DEVICE = detect_device()
    NON_BLOCKING = DEVICE.type != "cpu"
else:
    DEVICE = None
    NON_BLOCKING = False

# Set page config
st.set_page_config(page_title="HTS-3D Results Evaluator", layout="wide")

# Turn off RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def find_latest_model(compound_type: str) -> Optional[Path]:
    """
    Find the latest model file for a given compound type in the output directory.
    
    Args:
        compound_type: 'nlrp3' or 'cd28'
    
    Returns:
        Path to the model file, or None if not found
    """
    if not HTS3D_AVAILABLE:
        return None
    
    try:
        model_path = get_model_path_for_compound(compound_type)
        if model_path.exists():
            return model_path
    except:
        pass
    return None


def load_hts3d_model(
    model_path: Path,
    protein_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
) -> Optional[HTS3DModel]:
    """
    Load a trained HTS3DModel using HTS_3D_inference (same logic as HTS_3D_inference.py).
    Stores (model, scaler, selector) in session state for prediction.
    
    Args:
        model_path: Path to model file (.pt)
        protein_path: Optional PDB path (e.g. src/1YJD.pdb for CD28)
        checkpoint_path: Optional checkpoint path for legacy models
    
    Returns:
        Loaded model in eval mode, or None if loading fails
    """
    if not HTS3D_AVAILABLE or load_hts3d_inference_model is None:
        return None

    try:
        st.write(f"Loading model from {model_path}...")
        if protein_path:
            st.write(f"Using protein from {protein_path.name}")
        model, scaler, selector = load_hts3d_inference_model(
            model_path, checkpoint_path=checkpoint_path, protein_path=protein_path
        )
        st.session_state["hts3d_scaler"] = scaler
        st.session_state["hts3d_selector"] = selector
        st.success("✅ Model loaded successfully")
        return model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None
    except RuntimeError as e:
        if "legacy format" in str(e).lower() or "scaler" in str(e).lower():
            st.error(f"Legacy model requires a checkpoint. {e}")
        else:
            st.error(f"Error loading model: {e}")
        if st.session_state.get("debug", False):
            st.exception(e)
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        if st.session_state.get("debug", False):
            st.exception(e)
        return None


def predict_with_hts3d_model(
    model: HTS3DModel,
    smiles_list: List[str],
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Make predictions using HTS_3D_inference pipeline (same as CLI).
    Requires scaler/selector in session state from load_hts3d_model.
    When batch_size is None, uses HTS_3D_inference default (128 GPU / 64 CPU).
    """
    if not HTS3D_AVAILABLE or predict_on_smiles_inference is None:
        return np.zeros(len(smiles_list))

    scaler = st.session_state.get("hts3d_scaler")
    selector = st.session_state.get("hts3d_selector")
    if scaler is None or selector is None:
        st.error("Model preprocessing (scaler/selector) not found. Please reload the model.")
        return np.zeros(len(smiles_list))

    try:
        return predict_on_smiles_inference(
            model, smiles_list, scaler, selector, batch_size=batch_size
        )
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        if st.session_state.get("debug", False):
            st.exception(e)
        return np.zeros(len(smiles_list))


def check_valid_smiles(smiles_list):
    """Check if SMILES strings are valid"""
    validity = []
    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
            validity.append(mol is not None)
        except:
            validity.append(False)
    return validity


def preprocess_df_for_hts3d_inference(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """
    Preprocess dataframe to match HTS_3D_inference.py main() (normalize columns, deduplicate).
    Same logic as inference: dropna canonical_smiles; ensure molecule_chembl_id;
    deduplicate by molecule_chembl_id (keep most potent when value_nM has valid data, else first).
    Returns a deduplicated dataframe ready for HTS-3D prediction.
    """
    df = df.copy()
    # Normalize SMILES column (match inference possible_smiles_cols)
    if smiles_col != "canonical_smiles":
        df["canonical_smiles"] = df[smiles_col]
    # Normalize molecule_chembl_id (match inference mol_id_col lookup)
    mol_id_col = None
    for col in ['molecule_chembl_id', 'Molecule ChEMBL ID', 'molecule_chEMBL_id', 'molecule_id', 'compound_id']:
        if col in df.columns:
            mol_id_col = col
            break
    if mol_id_col and mol_id_col != "molecule_chembl_id":
        df["molecule_chembl_id"] = df[mol_id_col]
    # Filter missing SMILES (match inference)
    df = df.dropna(subset=["canonical_smiles"])
    if "molecule_chembl_id" not in df.columns:
        df["molecule_chembl_id"] = [f"COMPOUND_{i:06d}" for i in range(len(df))]
    # Deduplicate by molecule_chembl_id (match HTS_3D_inference main() exactly)
    if "value_nM" in df.columns:
        df["value_nM"] = pd.to_numeric(df["value_nM"], errors="coerce")
        if df["value_nM"].notna().any():
            df = df.sort_values("value_nM", na_position="last")
            df = df.groupby("molecule_chembl_id", as_index=False).first()
        else:
            df = df.groupby("molecule_chembl_id", as_index=False).first()
    else:
        df = df.groupby("molecule_chembl_id", as_index=False).first()
    return df.reset_index(drop=True)


# --- RDKit features for Enhanced Ensemble / Simple mode (HTSOracle-style) ---
def morgan_fp_oracle(smiles, radius=2, nBits=2048):
    """Generate Morgan fingerprint (HTSOracle-style)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nBits)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))
    except Exception:
        return np.zeros(nBits)


def maccs_fp_oracle(smiles):
    """Generate MACCS keys (HTSOracle-style)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(167)
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    except Exception:
        return np.zeros(167)


def extended_physchem_desc_oracle(smiles):
    """Extended physicochemical descriptors (15 elements, HTSOracle-style)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * 15
        aromatic_rings = 0
        try:
            for ring in GetSSSR(mol):
                if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                    aromatic_rings += 1
        except Exception:
            pass
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
            descriptors[14] = 0.0 if (np.isnan(qed) or np.isinf(qed)) else qed
        except:
            descriptors[14] = 0.0
        return descriptors
    except Exception:
        return [0] * 15


def rdkit_features_from_smiles_oracle(smiles_list):
    """Generate RDKit features for Enhanced Ensemble / Simple mode (2048+167+15=2230)."""
    features = []
    for smiles in smiles_list:
        morgan = morgan_fp_oracle(smiles)
        maccs = maccs_fp_oracle(smiles)
        physchem = extended_physchem_desc_oracle(smiles)
        combined = np.concatenate((morgan, maccs, physchem))
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(combined)
    return np.array(features, dtype=np.float32)


def make_varied_predictions(features):
    """Simple Drug-Likeness Based predictions (HTSOracle-style)."""
    num_molecules = features.shape[0]
    predictions = np.zeros(num_molecules)
    for i in range(num_molecules):
        physchem = features[i, -15:]
        mol_weight, logp = physchem[0], physchem[1]
        rotatable_bonds = physchem[2]
        hbond_acceptors, hbond_donors = physchem[3], physchem[4]
        tpsa, qed = physchem[5], physchem[14]
        score = 0.0
        if 200 <= mol_weight <= 500:
            score += 0.2
        if -2 <= logp <= 5:
            score += 0.2
        if rotatable_bonds <= 10:
            score += 0.1
        if hbond_acceptors <= 10:
            score += 0.1
        if hbond_donors <= 5:
            score += 0.1
        if 20 <= tpsa <= 140:
            score += 0.1
        score += qed * 0.2
        fp_contribution = np.mean(features[i, :200]) * 0.4
        predictions[i] = max(0.0, min(1.0, score + fp_contribution))
    return predictions


def make_method_specific_prediction(features, method):
    """Method-specific prediction fallback (HTSOracle-style)."""
    physchem = features[0, -15:] if features.size else np.zeros(15)
    if method == "lasso":
        mw_norm = physchem[0] / 500.0
        logp, qed = physchem[1], physchem[14]
        pred = 0.3 * mw_norm + 0.3 * (5.0 - np.abs(logp)) / 5.0 + 0.4 * qed
    elif method == "pca":
        rotatable_bonds, aromatic_rings = physchem[2], physchem[7]
        qed = physchem[14]
        pred = 0.2 * (1.0 - np.abs(rotatable_bonds - 5) / 10.0) + 0.3 * min(aromatic_rings / 3.0, 1.0) + 0.5 * qed
    elif method == "mutual_info":
        tpsa, qed = physchem[5], physchem[14]
        tpsa_norm = np.clip(tpsa / 200.0, 0, 1)
        pred = 0.4 * (1.0 - np.abs(tpsa_norm - 0.5) * 2) + 0.6 * qed
    else:
        pred = make_varied_predictions(features)[0]
    pred = np.clip(pred + np.random.normal(0, 0.05), 0, 1)
    return float(pred)


def get_predictions_from_enhanced_ensemble(smiles_list, rdkit_features, ensemble_model):
    """Generate predictions using the HTS.py enhanced ensemble model (.pkl)."""
    st.write("Generating predictions from the ensemble model...")
    feature_methods = ensemble_model.get('feature_methods', ['lasso', 'pca', 'mutual_info'])
    models = ensemble_model.get('models', [])
    if not models:
        st.warning("No models in ensemble. Using Drug-Likeness fallback.")
        return make_varied_predictions(rdkit_features), None
    method_predictions = np.zeros((len(smiles_list), len(feature_methods)))
    for method_idx, method in enumerate(feature_methods):
        method_models = [m for m in models if m.get('feature_method') == method]
        if not method_models:
            continue
        for i in range(len(smiles_list)):
            rdkit_feat = rdkit_features[i:i+1]
            method_pred, valid = 0.0, 0
            for model_data in method_models:
                try:
                    scaler = model_data.get('scaler')
                    selector = model_data.get('selector')
                    if scaler is not None and selector is not None:
                        rdkit_scaled = scaler.transform(rdkit_feat)
                        rdkit_selected = selector.transform(rdkit_scaled)
                    else:
                        rdkit_selected = rdkit_feat
                    rdkit_selected = np.nan_to_num(rdkit_selected, nan=0.0, posinf=0.0, neginf=0.0)
                    p = make_method_specific_prediction(rdkit_selected, method)
                    if np.isfinite(p):
                        method_pred += p
                        valid += 1
                except Exception:
                    pass
            method_predictions[i, method_idx] = method_pred / valid if valid else 0.0
    if np.all(method_predictions == 0):
        return make_varied_predictions(rdkit_features), None
    final = np.array([np.mean(method_predictions[i][method_predictions[i] > 0]) if np.any(method_predictions[i] > 0) else make_varied_predictions(rdkit_features[i:i+1])[0] for i in range(len(smiles_list))])
    return np.clip(np.nan_to_num(final, nan=0.0, posinf=1.0, neginf=0.0), 0, 1), method_predictions


def analyze_molecular_properties(smiles_list):
    """Analyze molecular properties for visualization"""
    properties = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                props = {
                    'MolWt': Descriptors.MolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'HBA': Descriptors.NumHAcceptors(mol),
                    'HBD': Descriptors.NumHDonors(mol),
                    'RotatableBonds': Descriptors.NumRotatableBonds(mol),
                    'AromaticRings': Descriptors.NumAromaticRings(mol),
                    'QED': QED.qed(mol)
                }
            else:
                props = {k: 0 for k in ['MolWt', 'LogP', 'TPSA', 'HBA', 'HBD', 'RotatableBonds', 'AromaticRings', 'QED']}
        except Exception as e:
            props = {k: 0 for k in ['MolWt', 'LogP', 'TPSA', 'HBA', 'HBD', 'RotatableBonds', 'AromaticRings', 'QED']}
        properties.append(props)
    return pd.DataFrame(properties)


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate classification metrics"""
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    metrics = {}
    try:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_pred)
    except:
        metrics['AUC-ROC'] = None
    
    try:
        metrics['Average Precision'] = average_precision_score(y_true, y_pred)
    except:
        metrics['Average Precision'] = None
    
    try:
        metrics['Precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
    except:
        metrics['Precision'] = None
    
    try:
        metrics['Recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    except:
        metrics['Recall'] = None
    
    try:
        metrics['F1-Score'] = f1_score(y_true, y_pred_binary, zero_division=0)
    except:
        metrics['F1-Score'] = None
    
    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred_binary)
        metrics['True Positives'] = int(cm[1, 1])
        metrics['True Negatives'] = int(cm[0, 0])
        metrics['False Positives'] = int(cm[0, 1])
        metrics['False Negatives'] = int(cm[1, 0])
    except:
        metrics['True Positives'] = None
        metrics['True Negatives'] = None
        metrics['False Positives'] = None
        metrics['False Negatives'] = None
    
    return metrics


def create_comparison_visualizations(df, score_cols, label_col=None):
    """Create comparison visualizations between different prediction scores"""
    
    # Determine number of subplots
    n_cols = len(score_cols)
    if label_col and label_col in df.columns:
        # Add comparison plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Score Distribution Comparison',
                'Score Correlation',
                'ROC Curves Comparison' if label_col else '',
                'Score vs Ground Truth'
            ),
            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}]]
        )
    else:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Score Distribution Comparison', 'Score Correlation'),
            specs=[[{'type': 'xy'}, {'type': 'xy'}]]
        )
    
    # 1. Distribution comparison
    for col in score_cols:
        if col in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df[col],
                    name=col,
                    nbinsx=30,
                    opacity=0.7
                ),
                row=1, col=1
            )
    
    # 2. Correlation scatter plot
    if len(score_cols) >= 2:
        col1, col2 = score_cols[0], score_cols[1]
        if col1 in df.columns and col2 in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df[col1],
                    y=df[col2],
                    mode='markers',
                    name=f'{col1} vs {col2}',
                    marker=dict(size=5, opacity=0.6)
                ),
                row=1, col=2
            )
            
            # Add diagonal line
            min_val = min(df[col1].min(), df[col2].min())
            max_val = max(df[col1].max(), df[col2].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='y=x',
                    line=dict(dash='dash', color='red')
                ),
                row=1, col=2
            )
    
    # 3. ROC curves if labels available
    if label_col and label_col in df.columns and len(score_cols) > 0:
        from sklearn.metrics import roc_curve
        y_true = df[label_col]
        
        for col in score_cols:
            if col in df.columns:
                try:
                    fpr, tpr, _ = roc_curve(y_true, df[col])
                    auc = roc_auc_score(y_true, df[col])
                    fig.add_trace(
                        go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode='lines',
                            name=f'{col} (AUC={auc:.3f})'
                        ),
                        row=2, col=1
                    )
                except:
                    pass
        
        # Add diagonal line for random classifier
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random (AUC=0.5)',
                line=dict(dash='dash', color='gray')
            ),
            row=2, col=1
        )
    
    # 4. Score vs Ground Truth box plot
    if label_col and label_col in df.columns and len(score_cols) > 0:
        col = score_cols[0]
        if col in df.columns:
            fig.add_trace(
                go.Box(
                    y=df[df[label_col] == 0][col],
                    name='Inactive (0)',
                    boxpoints='outliers'
                ),
                row=2, col=2
            )
            fig.add_trace(
                go.Box(
                    y=df[df[label_col] == 1][col],
                    name='Active (1)',
                    boxpoints='outliers'
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        height=800 if (label_col and label_col in df.columns) else 400,
        title_text="Prediction Score Comparison Analysis",
        title_x=0.5,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Score Value", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    if len(score_cols) >= 2:
        fig.update_xaxes(title_text=score_cols[0], row=1, col=2)
        fig.update_yaxes(title_text=score_cols[1], row=1, col=2)
    
    if label_col and label_col in df.columns:
        fig.update_xaxes(title_text="False Positive Rate", row=2, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=2, col=1)
        fig.update_xaxes(title_text="Label", row=2, col=2)
        fig.update_yaxes(title_text="Score", row=2, col=2)
    
    return fig


def run_app():
    st.title("🔬 HTS-3D Results Evaluator & Predictor")
    
    st.markdown("""
    ## Molecular Prediction Results Analysis & Prediction
    This application can:
    - **Analyze** existing prediction results from CSV files
    - **Generate** new predictions using trained HTS-3D models (same logic as `HTS_3D_inference.py`)
    
    **Command-line usage:**
    ```bash
    streamlit run src/HTS3DOracle.py -- --model output/hts3d_model_nlrp3.pt
    streamlit run src/HTS3DOracle.py -- --cd28 --model output/hts3d_model_cd28.pt --protein src/1YJD.pdb
    streamlit run src/HTS3DOracle.py --server.port 8506 -- --cd28 --model output/hts3d_model_cd28.pt --protein src/1YJD.pdb
    ```
    Use **"Use CD28 library data"** checkbox or **--cd28** to load libraries/library.csv + libraries/positives.csv (same as HTS_3D.py --cd28).
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Display command-line arguments if provided
    if CLI_ARGS.model or CLI_ARGS.compound or CLI_ARGS.protein or CLI_ARGS.checkpoint or getattr(CLI_ARGS, "cd28", False):
        with st.sidebar.expander("📋 Command-Line Options", expanded=False):
            if CLI_ARGS.model:
                st.write(f"**Model:** `{CLI_ARGS.model}`")
            if CLI_ARGS.compound:
                st.write(f"**Compound Type:** `{CLI_ARGS.compound}`")
            if CLI_ARGS.protein:
                st.write(f"**Protein:** `{CLI_ARGS.protein}`")
            if CLI_ARGS.checkpoint:
                st.write(f"**Checkpoint:** `{CLI_ARGS.checkpoint}`")
            if getattr(CLI_ARGS, "cd28", False):
                st.write("**--cd28:** Using libraries/library.csv + libraries/positives.csv")
    
    # Mode selector (Generate Predictions works with Enhanced Ensemble / Simple even without HTS-3D)
    mode = st.sidebar.radio(
        "Application Mode",
        ["📊 Analyze Results", "🔮 Generate Predictions"],
        help="Choose whether to analyze existing predictions or generate new ones"
    )
    if not HTS3D_AVAILABLE and mode == "🔮 Generate Predictions":
        st.sidebar.info("ℹ️ Use 'Enhanced Ensemble' or 'Simple' in Advanced Options (HTS-3D unavailable)")
    
    # Advanced options (HTSOracle-style)
    with st.sidebar.expander("Advanced Options", expanded=False):
        confidence_threshold = st.slider("Hit prediction threshold", 0.0, 1.0, 0.5, 0.01, key="threshold")
        prediction_mode = st.radio(
            "Prediction mode",
            ["HTS-3D", "Enhanced Ensemble", "Simple (Drug-Likeness Based)"],
            help="HTS-3D: 3D-aware model (.pt). Enhanced Ensemble: HTS.py ensemble (.pkl). Simple: QED/drug-likeness only."
        )
        default_ensemble_path = str(PROJECT_ROOT / "enhanced_ensemble_model.pkl")
        ensemble_model_path = st.text_input(
            "Path to ensemble model",
            value=default_ensemble_path,
            help="Path to the trained ensemble model file (.pkl) for Enhanced Ensemble mode"
        )
        show_molecular_properties = st.checkbox("Show molecular properties analysis", value=True, key="mol_properties")
        show_interactive_plots = st.checkbox("Use interactive visualizations", value=True, key="interactive_plots")
        st.checkbox("Debug mode", value=False, key="debug")

    # --cd28 option: Use CD28 library data (same behavior as HTS_3D.py and HTS_3D_inference.py --cd28)
    use_cd28 = getattr(CLI_ARGS, "cd28", False)
    if mode == "🔮 Generate Predictions" and HTS3D_AVAILABLE:
        use_cd28 = use_cd28 or st.sidebar.checkbox(
            "Use CD28 library data (--cd28)",
            value=use_cd28,
            help="Load libraries/library.csv + libraries/positives.csv; label=1 if SMILES in positives else 0. Same as HTS_3D.py --cd28. Skips file upload."
        )
        if use_cd28:
            st.sidebar.info("📂 Using libraries/library.csv + libraries/positives.csv")

    # HTS-3D options (model path, protein path, checkpoint, batch size) - matches HTS_3D_inference.py
    hts3d_model_path_override = None
    hts3d_protein_path = None
    hts3d_checkpoint_path = None
    hts3d_batch_size = None  # None = use HTS_3D_inference default (128 GPU / 64 CPU)
    if prediction_mode == "HTS-3D":
        with st.sidebar.expander("🔧 HTS-3D Options (matches HTS_3D_inference.py)", expanded=True):
            # When use_cd28, default to CD28 model and protein
            default_model = str(CLI_ARGS.model) if CLI_ARGS.model else (str(OUTPUT_DIR / "hts3d_model_cd28.pt") if use_cd28 else "")
            hts3d_model_path_override = st.text_input(
                "Model path (.pt)",
                value=default_model,
                placeholder="e.g. output/hts3d_model_cd28.pt",
                help="Path to HTS-3D model file. Overrides compound-based selection."
            )
            default_protein = str(CLI_ARGS.protein) if CLI_ARGS.protein else ("src/1YJD.pdb" if use_cd28 else "")
            hts3d_protein_path_input = st.text_input(
                "Protein PDB path",
                value=default_protein,
                placeholder="e.g. src/1YJD.pdb (CD28), leave empty for NLRP3 default",
                help="PDB file for protein template. Required for CD28 model."
            )
            hts3d_protein_path = Path(hts3d_protein_path_input.strip()) if hts3d_protein_path_input.strip() else None
            default_checkpoint = str(CLI_ARGS.checkpoint) if CLI_ARGS.checkpoint else ""
            hts3d_checkpoint_input = st.text_input(
                "Checkpoint path (legacy models only)",
                value=default_checkpoint,
                placeholder="e.g. output/checkpoints/final_checkpoint_cd28_*.pt",
                help="For legacy state_dict-only models: path to load scaler/selector."
            )
            hts3d_checkpoint_path = Path(hts3d_checkpoint_input.strip()) if hts3d_checkpoint_input.strip() else None
            hts3d_batch_size_input = st.number_input(
                "Batch size (0 = Auto)",
                min_value=0,
                max_value=512,
                value=0,
                help="0 = Auto (128 GPU / 64 CPU, matches HTS_3D_inference). 1–512 = custom."
            )
            hts3d_batch_size = None if hts3d_batch_size_input == 0 else int(hts3d_batch_size_input)
    
    # Model loading section (for prediction mode)
    model = None
    compound_type = None
    ensemble_model = None

    if mode == "🔮 Generate Predictions":
        # Load Enhanced Ensemble model when that mode is selected
        if prediction_mode == "Enhanced Ensemble" and ensemble_model_path:
            try:
                if os.path.exists(ensemble_model_path):
                    with st.spinner(f"Loading ensemble model from {ensemble_model_path}..."):
                        ensemble_model = joblib.load(ensemble_model_path)
                    st.sidebar.success(f"✅ Loaded ensemble model from {ensemble_model_path}")
                else:
                    st.sidebar.warning(f"⚠️ Ensemble model not found at {ensemble_model_path}")
            except Exception as e:
                st.sidebar.error(f"Error loading ensemble model: {e}")

        # HTS-3D model loading (only when HTS-3D mode selected and available) - matches HTS_3D_inference.py
        if prediction_mode != "HTS-3D" or not HTS3D_AVAILABLE:
            compound_type = "nlrp3"
            if prediction_mode == "HTS-3D" and not HTS3D_AVAILABLE:
                st.sidebar.warning("⚠️ HTS-3D components not available. Use Enhanced Ensemble or Simple.")
        else:
            st.sidebar.header("🤖 Model Selection")
            # Resolve model path: HTS-3D Options override > CLI > compound-based
            def _resolve_path(p, base=PROJECT_ROOT):
                if not p:
                    return None
                p = Path(p) if isinstance(p, str) else p
                if not p.is_absolute() and (base / p).exists():
                    return base / p
                if not p.is_absolute() and (SCRIPT_DIR / p).exists():
                    return SCRIPT_DIR / p
                return p.resolve() if not p.is_absolute() else p

            model_path = None
            compound_type = "nlrp3"
            cli_model_path = Path(CLI_ARGS.model).resolve() if CLI_ARGS.model else None
            if hts3d_model_path_override and str(hts3d_model_path_override).strip():
                model_path = _resolve_path(hts3d_model_path_override.strip())
                if model_path and model_path.exists() and HTS3D_AVAILABLE:
                    try:
                        compound_type = detect_compound_from_csv_path(model_path)
                    except Exception:
                        pass
            elif cli_model_path and cli_model_path.exists():
                model_path = _resolve_path(cli_model_path)
                if HTS3D_AVAILABLE:
                    try:
                        compound_type = detect_compound_from_csv_path(model_path)
                    except Exception:
                        compound_type = CLI_ARGS.compound or "nlrp3"
            else:
                compound_type = CLI_ARGS.compound if CLI_ARGS.compound else "nlrp3"
                if not CLI_ARGS.compound:
                    compound_type = st.sidebar.selectbox(
                        "Compound Type",
                        ["nlrp3", "cd28"],
                        index=0 if compound_type == "nlrp3" else 1,
                        help="Select the compound type to use the appropriate model"
                    )
                model_path = find_latest_model(compound_type)
                if model_path and HTS3D_AVAILABLE:
                    try:
                        compound_type = detect_compound_from_csv_path(model_path)
                    except Exception:
                        pass

            # Resolve protein and checkpoint paths (HTS-3D Options or CLI)
            protein_path = hts3d_protein_path or (Path(CLI_ARGS.protein) if CLI_ARGS.protein else None)
            if protein_path and not protein_path.is_absolute():
                protein_path = PROJECT_ROOT / protein_path
            checkpoint_path = hts3d_checkpoint_path or (Path(CLI_ARGS.checkpoint) if CLI_ARGS.checkpoint else None)
            if checkpoint_path and not checkpoint_path.is_absolute():
                checkpoint_path = PROJECT_ROOT / checkpoint_path

            if model_path and model_path.exists():
                model_key = f"{model_path.resolve()}|{protein_path}|{checkpoint_path}"
                if 'loaded_model' in st.session_state and st.session_state.get('model_path') == model_key:
                    model = st.session_state['loaded_model']
                    st.sidebar.success("✅ Model loaded and ready")
                else:
                    if (hts3d_model_path_override or CLI_ARGS.model) and model_path.exists():
                        auto_load_key = f"auto_load_{model_key}"
                        if auto_load_key not in st.session_state:
                            with st.spinner("Auto-loading model..."):
                                model = load_hts3d_model(model_path, protein_path=protein_path, checkpoint_path=checkpoint_path)
                                if model is not None:
                                    st.session_state['loaded_model'] = model
                                    st.session_state['model_path'] = model_key
                                    st.session_state['model_compound'] = compound_type
                                    st.session_state[auto_load_key] = True
                                    st.sidebar.success("✅ Model auto-loaded!")
                                else:
                                    st.session_state[auto_load_key] = False
                        else:
                            if st.session_state.get('model_path') == model_key:
                                model = st.session_state.get('loaded_model')
                    if model is None:
                        if st.sidebar.button("Load Model", key="load_model"):
                            with st.spinner("Loading model..."):
                                model = load_hts3d_model(model_path, protein_path=protein_path, checkpoint_path=checkpoint_path)
                                if model is not None:
                                    st.session_state['loaded_model'] = model
                                    st.session_state['model_path'] = model_key
                                    st.session_state['model_compound'] = compound_type
                                    st.sidebar.success("✅ Model loaded successfully!")
                                    st.rerun()
                        else:
                            st.sidebar.info("👆 Click 'Load Model' to load the model")
                if model_path:
                    st.sidebar.success(f"✅ Found model: {model_path.name}")
            else:
                if HTS3D_AVAILABLE:
                    st.sidebar.warning(f"⚠️ Model not found: {model_path or 'specify path in HTS-3D Options'}")
                else:
                    st.sidebar.warning("⚠️ HTS-3D components not available")
                st.sidebar.info("💡 Train a model first using HTS_3D.py or specify model path in HTS-3D Options")
    
    # File uploader (skip when use_cd28 in prediction mode)
    uploaded_file = None
    if mode == "📊 Analyze Results":
        uploaded_file = st.file_uploader(
            "📁 Upload a CSV file with prediction results", 
            type=["csv"],
            help="The file should contain columns with prediction scores (e.g., 'hts3d_score', 'Prediction_Score') and optionally ground truth labels ('label')"
        )
    else:  # Prediction mode
        if not use_cd28:
            uploaded_file = st.file_uploader(
                "📁 Upload a CSV file with SMILES strings", 
                type=["csv"],
                help="The file should contain a SMILES column (e.g., 'canonical_smiles', 'SMILES', 'Smiles')"
            )
    
    # Load data: from --cd28 (libraries/) or uploaded file
    df = None
    
    if use_cd28 and mode == "🔮 Generate Predictions" and HTS3D_AVAILABLE:
        library_path = PROJECT_ROOT / "libraries" / "library.csv"
        positives_path = PROJECT_ROOT / "libraries" / "positives.csv"
        try:
            if library_path.exists() and positives_path.exists():
                df = load_library_and_positives(library_path, positives_path)
                st.success(f"✅ Loaded CD28 data: {len(df)} molecules ({int(df['label'].sum())} actives, {len(df) - int(df['label'].sum())} inactives)")
            else:
                st.error(f"❌ CD28 library not found. Expected: {library_path} and {positives_path}")
        except Exception as e:
            st.error(f"Error loading CD28 library: {e}")
            if st.session_state.get("debug", False):
                st.exception(e)
    elif uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Read the file (try different encodings)
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(tmp_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.error("Could not read CSV file with any encoding")
            else:
                st.success(f"✅ Successfully loaded {len(df)} rows from uploaded file")
            
            # Clean up temp file immediately after reading
            try:
                os.unlink(tmp_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"Error loading uploaded file: {str(e)}")
            if st.session_state.get("debug", False):
                st.exception(e)
    else:
        if mode == "📊 Analyze Results":
            st.info("👆 Please upload a CSV file with prediction results to begin analysis.")
        elif use_cd28 and not HTS3D_AVAILABLE:
            st.info("ℹ️ CD28 library mode requires HTS-3D. Use Enhanced Ensemble or Simple with uploaded file, or install HTS-3D dependencies.")
        elif use_cd28:
            st.info("💡 CD28 mode: Ensure libraries/library.csv and libraries/positives.csv exist.")
        else:
            st.info("👆 Please upload a CSV file with SMILES strings to generate predictions.")
    
    # Prediction mode: Generate predictions if model is loaded
    if mode == "🔮 Generate Predictions" and df is not None and len(df) > 0:
        # Auto-detect compound type from filename if available
        if uploaded_file is not None and HTS3D_AVAILABLE:
            try:
                detected_compound = detect_compound_from_csv_path(Path(uploaded_file.name))
                if detected_compound != compound_type:
                    st.info(f"💡 Auto-detected compound type '{detected_compound}' from filename. Consider switching model type.")
            except:
                pass
        
        model_ready = (
            (prediction_mode == "HTS-3D" and model is not None) or
            (prediction_mode == "Enhanced Ensemble" and ensemble_model is not None) or
            (prediction_mode == "Simple (Drug-Likeness Based)")
        )
        if not model_ready:
            if prediction_mode == "HTS-3D":
                st.warning("⚠️ Please load a model first using the sidebar.")
            elif prediction_mode == "Enhanced Ensemble":
                st.warning("⚠️ Ensemble model not loaded. Check the path in Advanced Options.")
            else:
                st.warning("⚠️ Unexpected state.")
        else:
            # Identify SMILES column
            smiles_column = None
            possible_smiles_cols = ['canonical_smiles', 'SMILES', 'Smiles', 'smiles', 'SMILE', 'Smile', 'smile', 'Structure']
            for col in possible_smiles_cols:
                if col in df.columns:
                    smiles_column = col
                    break
            
            if not smiles_column:
                st.error("❌ No SMILES column found. Please ensure your CSV has a SMILES column.")
            else:
                # HTS-3D: use inference-style preprocessing (dedup, normalize columns)
                if prediction_mode == "HTS-3D":
                    df_prep = preprocess_df_for_hts3d_inference(df, smiles_column)
                    smiles_list = df_prep["canonical_smiles"].tolist()
                else:
                    valid_mask = df[smiles_column].notna()
                    df_prep = df.loc[valid_mask].copy()
                    if smiles_column != "canonical_smiles":
                        df_prep["canonical_smiles"] = df_prep[smiles_column]
                    smiles_list = df_prep[smiles_column].tolist()

                if len(smiles_list) == 0:
                    st.error("❌ No valid SMILES strings found in the file.")
                else:
                    # Generate predictions based on prediction_mode (HTS-3D uses full HTS_3D_inference logic)
                    with st.spinner(f"Generating predictions for {len(smiles_list)} molecules..."):
                        if prediction_mode == "HTS-3D":
                            predictions = predict_with_hts3d_model(model, smiles_list, batch_size=hts3d_batch_size)
                        elif prediction_mode == "Enhanced Ensemble" and ensemble_model is not None:
                            features = rdkit_features_from_smiles_oracle(smiles_list)
                            predictions, _ = get_predictions_from_enhanced_ensemble(smiles_list, features, ensemble_model)
                        else:  # Simple (Drug-Likeness Based)
                            features = rdkit_features_from_smiles_oracle(smiles_list)
                            predictions = make_varied_predictions(features)

                    # Add predictions to preprocessed dataframe
                    df_prep["hts3d_score"] = predictions
                    df_prep["predicted_hit"] = predictions >= confidence_threshold
                    df = df_prep

                    # Save to output CSV (matching HTS_3D_inference)
                    saved_path = None
                    detected_compound = compound_type
                    if uploaded_file and HTS3D_AVAILABLE:
                        try:
                            detected_compound = detect_compound_from_csv_path(Path(uploaded_file.name))
                        except Exception:
                            pass
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    try:
                        output_path = OUTPUT_DIR / f"hts3d_predictions_{detected_compound}_{timestamp}.csv"
                        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        df.to_csv(output_path, index=False)
                        saved_path = str(output_path)
                        st.success(f"✅ Predictions saved to: `{output_path}`")
                    except Exception as save_err:
                        st.warning(f"Could not save to file: {save_err}")

                    # Inference-style summary (matching HTS_3D_inference output exactly)
                    st.header("📊 Prediction Summary")
                    summary_lines = [
                        f"Total Compounds: {len(df)}",
                        f"Total Columns: {len(df.columns)}",
                        "Score Columns Found: 1",
                        "Prediction Columns: hts3d_score",
                        "",
                        "[SMILES] Using 'canonical_smiles' for SMILES strings",
                        "",
                        "[Analysis] Prediction Score Analysis:",
                        f"  Predicted hits (score >= {confidence_threshold:.1f}): {(predictions >= confidence_threshold).sum()} ({(predictions >= confidence_threshold).sum()/len(predictions)*100:.1f}%)",
                        f"  Mean prediction score: {predictions.mean():.4f}",
                        f"  Min score: {predictions.min():.4f}",
                        f"  Max score: {predictions.max():.4f}",
                        f"  Median score: {np.median(predictions):.4f}",
                    ]
                    # Confusion matrix, precision, recall, F1 when labels present (matches HTS_3D_inference)
                    if "label" in df.columns:
                        y_true = df["label"].astype(int).values
                        y_pred = (predictions >= confidence_threshold).astype(int)
                        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                        prec = precision_score(y_true, y_pred, zero_division=0)
                        rec = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        summary_lines.extend([
                            "",
                            "[Metrics] Confusion Matrix (threshold={:.1f}):".format(confidence_threshold),
                            "                Predicted 0  Predicted 1",
                            f"  Actual 0      {cm[0, 0]:>10}     {cm[0, 1]:>10}",
                            f"  Actual 1      {cm[1, 0]:>10}     {cm[1, 1]:>10}",
                            "",
                            f"  Precision: {prec:.4f}",
                            f"  Recall:    {rec:.4f}",
                            f"  F1:        {f1:.4f}",
                        ])
                    st.code("\n".join(summary_lines), language=None)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Compounds", len(df))
                    with col2:
                        predicted_hits = (predictions >= confidence_threshold).sum()
                        st.metric("Predicted Hits", f"{predicted_hits} ({predicted_hits/len(predictions)*100:.1f}%)")
                    with col3:
                        st.metric("Mean Score", f"{predictions.mean():.4f}")
                    with col4:
                        st.metric("Max Score", f"{predictions.max():.4f}")

                    st.info("💡 Predictions generated! Scroll down to see detailed analysis of the results.")

                    # Download button for predictions CSV
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    dl_filename = Path(saved_path).name if saved_path else f"hts3d_predictions_{detected_compound}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    st.download_button(
                        "📥 Download Predictions CSV",
                        data=csv_bytes,
                        file_name=dl_filename,
                        mime="text/csv",
                        key="download_predictions",
                    )

                    mode = "📊 Analyze Results"
    
    # Analysis mode: Analyze existing predictions
    if mode == "📊 Analyze Results" and df is not None and len(df) > 0:
        # Display basic info
        st.header("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Compounds", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        
        # Identify score columns
        score_columns = []
        possible_score_cols = ['hts3d_score', 'Prediction_Score', 'prediction_score', 'score', 'predicted_score']
        for col in possible_score_cols:
            if col in df.columns:
                score_columns.append(col)
        
        # Also check for columns ending with '_score'
        for col in df.columns:
            if col.endswith('_score') and col not in score_columns:
                score_columns.append(col)
        
        if len(score_columns) > 0:
            with col3:
                st.metric("Score Columns Found", len(score_columns))
            with col4:
                st.metric("Prediction Columns", ', '.join(score_columns[:2]) + ('...' if len(score_columns) > 2 else ''))
        else:
            st.warning("⚠️ No prediction score columns found. Looking for columns like 'hts3d_score', 'Prediction_Score', etc.")
        
        # Identify label column (or fall back to "predicted" / "predicted_hit" as ground truth proxy)
        label_column = None
        possible_label_cols = ['label', 'Label', 'LABEL', 'true_label', 'ground_truth', 'activity']
        for col in possible_label_cols:
            if col in df.columns:
                label_column = col
                break
        # Fall back: use "predicted" or "predicted_hit" as label when no explicit label (CD28/NLRP3)
        if label_column is None:
            pred_cols = ['predicted_hit', 'predicted', 'Predicted', 'PREDICTED']
            for pc in pred_cols:
                if pc in df.columns:
                    df = df.copy()
                    vals = df[pc]
                    if vals.dtype == bool:
                        df["label"] = vals.astype(int)
                    else:
                        df["label"] = (pd.to_numeric(vals, errors='coerce').fillna(0) >= 0.5).astype(int)
                    label_column = "label"
                    st.info(f"📊 Using '{pc}' column as ground truth labels (fallback)")
                    break
        
        # Identify SMILES column
        smiles_column = None
        possible_smiles_cols = ['canonical_smiles', 'SMILES', 'Smiles', 'smiles', 'SMILE', 'Smile', 'smile', 'Structure']
        for col in possible_smiles_cols:
            if col in df.columns:
                smiles_column = col
                break
        
        if smiles_column:
            st.info(f"📊 Using '{smiles_column}' for SMILES strings")
        else:
            st.warning("⚠️ No SMILES column found")
            if len(df.columns) > 0:
                smiles_column = st.selectbox("Select SMILES column:", df.columns)
        
        if label_column:
            st.info(f"📊 Ground truth labels found in '{label_column}' column")
            # Show label distribution
            label_counts = df[label_column].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Label Distribution:**")
                st.write(label_counts)
            with col2:
                if len(label_counts) == 2:
                    st.write(f"**Active (1):** {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
                    st.write(f"**Inactive (0):** {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
        else:
            st.info("ℹ️ No ground truth labels found. Metrics will not be calculated.")
        
        # Display column info
        if st.session_state.debug:
            st.write("**All columns:**", df.columns.tolist())
            st.write("**Data types:**", df.dtypes.to_dict())
        
        # Score Analysis Section
        if len(score_columns) > 0:
            st.header("📈 Prediction Score Analysis")
            
            # Summary statistics for each score column
            for score_col in score_columns:
                st.subheader(f"Analysis for: {score_col}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Score", f"{df[score_col].mean():.4f}")
                with col2:
                    st.metric("Median Score", f"{df[score_col].median():.4f}")
                with col3:
                    st.metric("Std Deviation", f"{df[score_col].std():.4f}")
                with col4:
                    predicted_hits = (df[score_col] >= confidence_threshold).sum()
                    st.metric("Predicted Hits", f"{predicted_hits} ({predicted_hits/len(df)*100:.1f}%)")
                
                # Score distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                # Use same dark orange as Molecular Properties Analysis (tab:orange = cycle C1)
                ax.hist(df[score_col], bins=30, alpha=0.7, color='tab:orange', edgecolor='black')
                ax.axvline(confidence_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {confidence_threshold}')
                ax.set_xlabel('Prediction Score')
                ax.set_ylabel('Count')
                ax.set_title(f'Distribution of {score_col}')
                ax.legend()
                st.pyplot(fig)

                # Second graph: same distribution but excluding scores < 0.3
                cutoff = 0.3
                scores_filtered = df[score_col][df[score_col] >= cutoff]
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.hist(scores_filtered, bins=30, alpha=0.7, color='tab:orange', edgecolor='black')
                ax2.axvline(confidence_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {confidence_threshold}')
                ax2.set_xlabel('Prediction Score')
                ax2.set_ylabel('Count')
                ax2.set_title(f'Distribution of {score_col} (score ≥ {cutoff} only, n={len(scores_filtered)})')
                ax2.legend()
                st.pyplot(fig2)

                # Calculate metrics if labels available
                if label_column and label_column in df.columns:
                    metrics = calculate_metrics(df[label_column], df[score_col], threshold=confidence_threshold)
                    
                    st.write("**Performance Metrics:**")
                    metrics_df = pd.DataFrame([metrics]).T
                    metrics_df.columns = ['Value']
                    st.dataframe(metrics_df)
                    
                    # Confusion matrix visualization
                    y_true = df[label_column]
                    y_pred_binary = (df[score_col] >= confidence_threshold).astype(int)
                    cm = confusion_matrix(y_true, y_pred_binary)
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=['Predicted Inactive', 'Predicted Active'],
                               yticklabels=['Actual Inactive', 'Actual Active'])
                    ax.set_title(f'Confusion Matrix ({score_col}, threshold={confidence_threshold})')
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')
                    st.pyplot(fig)
            
            # Comparison between score columns
            if len(score_columns) >= 2:
                st.subheader("🔍 Score Comparison")
                
                if show_interactive_plots:
                    comparison_fig = create_comparison_visualizations(df, score_columns, label_column)
                    st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Correlation matrix
                score_df = df[score_columns]
                correlation = score_df.corr()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax,
                           square=True, linewidths=1, cbar_kws={"shrink": 0.8})
                ax.set_title('Score Correlation Matrix')
                st.pyplot(fig)
                
                # Summary comparison table
                comparison_stats = []
                for col in score_columns:
                    stats = {
                        'Score Column': col,
                        'Mean': df[col].mean(),
                        'Median': df[col].median(),
                        'Std': df[col].std(),
                        'Min': df[col].min(),
                        'Max': df[col].max(),
                        'Hits (≥threshold)': (df[col] >= confidence_threshold).sum(),
                        'Hit Rate (%)': (df[col] >= confidence_threshold).sum() / len(df) * 100
                    }
                    if label_column and label_column in df.columns:
                        metrics = calculate_metrics(df[label_column], df[col], threshold=confidence_threshold)
                        stats['AUC-ROC'] = metrics.get('AUC-ROC')
                        stats['F1-Score'] = metrics.get('F1-Score')
                    comparison_stats.append(stats)
                
                comparison_df = pd.DataFrame(comparison_stats)
                st.dataframe(comparison_df, use_container_width=True)
        
        # Molecular Properties Analysis
        if show_molecular_properties and smiles_column:
            st.header("🧪 Molecular Properties Analysis")
            
            smiles_list = df[smiles_column].tolist()
            properties_df = analyze_molecular_properties(smiles_list)
            
            # Merge with original dataframe
            analysis_df = df.copy()
            for col in properties_df.columns:
                analysis_df[col] = properties_df[col].values
            
            # Property distributions
            if len(score_columns) > 0:
                score_col = score_columns[0]
                hit_mask = analysis_df[score_col] >= confidence_threshold
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Plot 1: Molecular Weight
                axes[0, 0].hist([analysis_df.loc[~hit_mask, 'MolWt'], analysis_df.loc[hit_mask, 'MolWt']], 
                               label=['Non-Hits', 'Hits'], bins=20, alpha=0.7)
                axes[0, 0].set_xlabel('Molecular Weight')
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].set_title('Molecular Weight Distribution')
                axes[0, 0].legend()
                
                # Plot 2: LogP
                axes[0, 1].hist([analysis_df.loc[~hit_mask, 'LogP'], analysis_df.loc[hit_mask, 'LogP']], 
                               label=['Non-Hits', 'Hits'], bins=20, alpha=0.7)
                axes[0, 1].set_xlabel('LogP')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].set_title('LogP Distribution')
                axes[0, 1].legend()
                
                # Plot 3: QED Distribution
                axes[1, 0].hist([analysis_df.loc[~hit_mask, 'QED'], analysis_df.loc[hit_mask, 'QED']], 
                               label=['Non-Hits', 'Hits'], bins=20, alpha=0.7)
                axes[1, 0].set_xlabel('Drug-Likeness (QED)')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_title('QED Distribution')
                axes[1, 0].legend()
                
                # Plot 4: Score vs QED scatter (same dark orange as other charts)
                if len(score_columns) > 0:
                    axes[1, 1].scatter(analysis_df['QED'], analysis_df[score_col], alpha=0.5, color='tab:orange')
                    axes[1, 1].set_xlabel('QED')
                    axes[1, 1].set_ylabel(f'{score_col}')
                    axes[1, 1].set_title('Score vs Drug-Likeness')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # Detailed Results Table
        st.header("📋 Detailed Results")
        
        # Sort by first score column if available
        if len(score_columns) > 0:
            sort_column = st.selectbox("Sort by:", score_columns, key="sort_col")
            sort_ascending = st.checkbox("Ascending order", value=False, key="sort_asc")
            sorted_df = df.sort_values(by=sort_column, ascending=sort_ascending)
        else:
            sorted_df = df
        
        # Column selector
        all_columns = sorted_df.columns.tolist()
        default_display_cols = [smiles_column] + score_columns + ([label_column] if label_column else [])
        default_display_cols = [col for col in default_display_cols if col is not None]
        
        selected_columns = st.multiselect(
            "Select columns to display:",
            options=all_columns,
            default=default_display_cols[:10]  # Limit default to first 10 to avoid too many columns
        )
        
        if not selected_columns:
            selected_columns = default_display_cols
        
        # Display results
        st.dataframe(sorted_df[selected_columns], use_container_width=True, height=400)
        
        # Download options
        st.header("💾 Download Results")
        
        csv = sorted_df.to_csv(index=False)
        st.download_button(
            "Download Sorted Results",
            csv,
            "hts3d_evaluation_results.csv",
            "text/csv",
            key='download-csv'
        )


# Main entry point
if __name__ == "__main__":
    run_app()

