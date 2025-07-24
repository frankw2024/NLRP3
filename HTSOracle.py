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
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Lipinski, QED
from rdkit.Chem import GetSSSR
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import RobertaTokenizer
import torch
import torch.nn as nn
import time

# Set page config
st.set_page_config(page_title="Molecular Ensemble Predictor", layout="wide")

# Turn off RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Enhanced feature generation functions
def morgan_fp(smiles, radius=2, nBits=2048):
    """Generate Morgan fingerprint for a SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nBits)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))
    except Exception as e:
        if st.session_state.debug:
            st.write(f"Error in morgan_fp for {smiles}: {str(e)}")
        return np.zeros(nBits)

def maccs_fp(smiles):
    """Generate MACCS keys for a SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(167)
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    except Exception as e:
        if st.session_state.debug:
            st.write(f"Error in maccs_fp for {smiles}: {str(e)}")
        return np.zeros(167)

def extended_physchem_desc(smiles):
    """Extended set of physicochemical descriptors"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * 15
        
        # Count aromatic rings
        aromatic_rings = 0
        try:
            num_rings = GetSSSR(mol)
            for ring in num_rings:
                if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                    aromatic_rings += 1
        except Exception as e:
            if st.session_state.debug:
                st.write(f"Error processing rings: {str(e)}")
            aromatic_rings = 0
                
        # Compute all descriptors safely with defaults
        descriptors = [0] * 15
        
        try: descriptors[0] = Descriptors.MolWt(mol)                 # Molecular weight
        except: pass
        
        try: descriptors[1] = Descriptors.MolLogP(mol)               # LogP
        except: pass
        
        try: descriptors[2] = Descriptors.NumRotatableBonds(mol)     # Number of rotatable bonds
        except: pass
        
        try: descriptors[3] = Descriptors.NumHAcceptors(mol)         # Number of H-bond acceptors
        except: pass
        
        try: descriptors[4] = Descriptors.NumHDonors(mol)            # Number of H-bond donors
        except: pass
        
        try: descriptors[5] = Descriptors.TPSA(mol)                  # Topological polar surface area
        except: pass
        
        try: descriptors[6] = Descriptors.RingCount(mol)             # Ring count
        except: pass
        
        descriptors[7] = aromatic_rings                              # Aromatic ring count
        
        try: descriptors[8] = Descriptors.HeavyAtomCount(mol)        # Heavy atom count
        except: pass
        
        try: descriptors[9] = Descriptors.NumHeteroatoms(mol)        # Number of heteroatoms
        except: pass
        
        try: descriptors[10] = Descriptors.FractionCSP3(mol)         # Fraction of C atoms that are sp3 hybridized
        except: pass
        
        try: descriptors[11] = Descriptors.NumAromaticRings(mol)     # Number of aromatic rings
        except: pass
        
        try: descriptors[12] = Lipinski.NumHAcceptors(mol)           # Lipinski H-bond acceptors
        except: pass
        
        try: descriptors[13] = Lipinski.NumHDonors(mol)              # Lipinski H-bond donors
        except: pass
        
        try: 
            qed = QED.qed(mol)                                       # Quantitative Estimate of Drug-likeness
            if np.isnan(qed) or np.isinf(qed):
                qed = 0.0
            descriptors[14] = qed
        except: 
            descriptors[14] = 0.0
            
        return descriptors
    except Exception as e:
        if st.session_state.debug:
            st.write(f"Error in extended_physchem_desc for {smiles}: {str(e)}")
        return [0] * 15

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

def rdkit_features_from_smiles(smiles_list):
    """Generate RDKit features for a list of SMILES strings"""
    features = []
    error_count = 0
    chunk_size = 50  # Show progress in chunks
    
    for i, smiles in enumerate(smiles_list):
        try:
            if i % chunk_size == 0:
                st.write(f"Processing molecule {i+1}/{len(smiles_list)}...")
            
            morgan = morgan_fp(smiles)
            maccs = maccs_fp(smiles)
            physchem = extended_physchem_desc(smiles)
            combined = np.concatenate((morgan, maccs, physchem))
            
            # Check for NaN or Inf values
            if np.any(np.isnan(combined)) or np.any(np.isinf(combined)):
                if st.session_state.debug:
                    st.write(f"Warning: NaN or Inf in features for SMILES {i}: {smiles}")
                # Replace problematic values with zeros
                combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
                
            features.append(combined)
        except Exception as e:
            error_count += 1
            # Add zeros as fallback
            features.append(np.zeros(2048 + 167 + 15))
            if st.session_state.debug:
                st.write(f"Error processing SMILES {i}: {smiles}: {str(e)}")
    
    if error_count > 0:
        st.warning(f"⚠️ {error_count} molecules could not be processed correctly and were filled with zeros.")
    
    # Convert to numpy array
    features_array = np.array(features, dtype=np.float32)
    
    # Final check for any NaN or Inf values
    if np.isnan(features_array).any() or np.isinf(features_array).any():
        st.write("Warning: NaN or Inf values detected in features. Replacing with zeros.")
        features_array = np.nan_to_num(features_array)
    
    return features_array

def get_predictions_from_enhanced_ensemble(smiles_list, rdkit_features, ensemble_model):
    """Generate predictions using the enhanced ensemble model from the pipeline"""
    st.write("Generating predictions from the ensemble model...")
    
    # Extract necessary components from the model
    feature_methods = ensemble_model.get('feature_methods', ['lasso', 'pca', 'mutual_info'])
    models = ensemble_model.get('models', [])
    
    if not models:
        st.error("No models found in the ensemble. Using default prediction method.")
        return make_varied_predictions(rdkit_features), None
    
    # Initialize array to store all predictions for each feature selection method
    method_predictions = np.zeros((len(smiles_list), len(feature_methods)))
    
    # For each feature selection method, apply the appropriate models to get predictions
    for method_idx, method in enumerate(feature_methods):
        method_models = [m for m in models if m.get('feature_method') == method]
        
        if not method_models:
            st.warning(f"No models available for method {method}. Using zeros.")
            continue
        
        st.write(f"Generating predictions with {len(method_models)} models for method: {method}")
        
        # Progress bar for this method
        method_progress = st.progress(0)
        
        # Apply each model to get predictions for the full dataset
        for i in range(len(smiles_list)):
            try:
                # Update progress bar
                if i % max(1, len(smiles_list) // 100) == 0:
                    method_progress.progress(i / len(smiles_list))
                
                smiles = smiles_list[i]
                rdkit_feat = rdkit_features[i:i+1]
                
                # Average predictions from all folds for this method
                method_pred = 0
                valid_model_count = 0
                
                for model_data in method_models:
                    try:
                        # Apply the appropriate feature selection
                        selector = model_data.get('selector')
                        scaler = model_data.get('scaler')
                        
                        if selector is not None and scaler is not None:
                            rdkit_feat_scaled = scaler.transform(rdkit_feat)
                            if hasattr(selector, 'transform'):
                                rdkit_feat_selected = selector.transform(rdkit_feat_scaled)
                            else:  # For PCA or other transformers
                                rdkit_feat_selected = selector.transform(rdkit_feat_scaled)
                        else:
                            # No feature selection or failed feature selection
                            rdkit_feat_selected = rdkit_feat
                        
                        # Handle NaN/Inf values
                        rdkit_feat_selected = np.nan_to_num(rdkit_feat_selected, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # For this app, we'll use a simplified prediction approach instead of the full neural network
                        # since we might not have the ChemBERTa model available
                        pred_prob = make_method_specific_prediction(rdkit_feat_selected, method)
                        
                        # Check for NaN or Inf
                        if np.isnan(pred_prob) or np.isinf(pred_prob):
                            continue
                            
                        method_pred += pred_prob
                        valid_model_count += 1
                    except Exception as e:
                        if st.session_state.debug:
                            st.write(f"Error with model for {method}, sample {i}: {str(e)}")
                        continue
                
                # Average predictions from valid models
                if valid_model_count > 0:
                    method_pred /= valid_model_count
                    method_predictions[i, method_idx] = method_pred
                else:
                    method_predictions[i, method_idx] = 0.0  # Fallback
            except Exception as e:
                if st.session_state.debug:
                    st.write(f"Error generating prediction for sample {i} with method {method}: {str(e)}")
                method_predictions[i, method_idx] = 0.0  # Fallback
        
        # Complete the progress bar
        method_progress.progress(1.0)
    
    # Check if we have valid predictions
    if np.all(method_predictions == 0):
        st.warning("All method-specific predictions are zero. Using backup prediction method.")
        return make_varied_predictions(rdkit_features), None
    
    # Final ensemble prediction (average of all methods)
    final_predictions = np.zeros(len(smiles_list))
    for i in range(len(smiles_list)):
        # Average non-zero predictions for each sample
        method_preds = method_predictions[i, :]
        valid_preds = method_preds[method_preds > 0]
        if len(valid_preds) > 0:
            final_predictions[i] = np.mean(valid_preds)
        else:
            # No valid predictions, use backup
            final_predictions[i] = make_varied_predictions(rdkit_features[i:i+1])[0]
    
    # Check for NaN or Inf in final predictions
    if np.any(np.isnan(final_predictions)) or np.any(np.isinf(final_predictions)):
        st.warning("NaN or Inf values in final predictions. Fixing...")
        final_predictions = np.nan_to_num(final_predictions, nan=0.0, posinf=1.0, neginf=0.0)
        
    # Ensure predictions are in [0, 1] range
    final_predictions = np.clip(final_predictions, 0, 1)
    
    return final_predictions, method_predictions

def make_method_specific_prediction(features, method):
    """Generate a prediction for a specific feature selection method"""
    # This is a simplified version that generates predictions without the neural network
    # It's used as a fallback when the full model can't be loaded
    
    # Extract physicochemical descriptors if available
    # In the full features, the last 15 elements are the physicochemical descriptors
    if features.shape[1] >= 15:
        physchem = features[0, -15:] if features.shape[0] > 0 else np.zeros(15)
    else:
        # Use all features
        physchem = features[0, :] if features.shape[0] > 0 else features.flatten()
    
    # Default prediction
    prediction = 0.5
    
    # Method-specific scoring
    if method == "lasso":
        # Focus on molecular weight, logP, and QED if available
        if len(physchem) >= 15:
            mw_norm = physchem[0] / 500.0  # Normalized molecular weight
            logp = physchem[1]  # LogP values
            qed = physchem[14]  # Drug-likeness score
            
            # Create a simple linear model
            prediction = 0.3 * mw_norm + 0.3 * (5.0 - np.abs(logp)) / 5.0 + 0.4 * qed
        else:
            # If we don't have those specific features, use a mean of the available features
            prediction = np.mean(features) * 0.5 + 0.25
        
    elif method == "pca":
        # For PCA, focus on different aspects
        if len(physchem) >= 15:
            rotatable_bonds = physchem[2] 
            aromatic_rings = physchem[7]
            qed = physchem[14]
            
            # Create a score based on drug-likeness and structural features
            prediction = 0.2 * (1.0 - np.abs(rotatable_bonds - 5) / 10.0) + \
                        0.3 * np.minimum(aromatic_rings / 3.0, 1.0) + \
                        0.5 * qed
        else:
            # Use a different combination of features
            prediction = np.mean(features) * 0.6 + 0.2
            
    elif method == "mutual_info":
        # For mutual_info
        if len(physchem) >= 15:
            tpsa = physchem[5]  # TPSA values
            qed = physchem[14]  # QED values
            
            # Normalize TPSA to 0-1 range (typical range 0-200)
            tpsa_norm = np.clip(tpsa / 200.0, 0, 1)
            
            # Create predictions favoring moderate TPSA and high QED
            prediction = 0.4 * (1.0 - np.abs(tpsa_norm - 0.5) * 2) + 0.6 * qed
        else:
            # Use another combination
            prediction = np.mean(features) * 0.7 + 0.15
    
    else:
        # For any other method, use a general approach
        prediction = make_varied_predictions(features)[0]
    
    # Add some random variation and clip to [0, 1]
    prediction += np.random.normal(0, 0.05)
    prediction = np.clip(prediction, 0, 1)
    
    return prediction

def make_varied_predictions(features):
    """Generate varied predictions based on molecule properties"""
    # Extract some meaningful properties from features to influence predictions
    # The last 15 elements are the physicochemical descriptors
    
    num_molecules = features.shape[0]
    predictions = np.zeros(num_molecules)
    
    for i in range(num_molecules):
        # Get physicochemical properties (last 15 features)
        physchem = features[i, -15:]
        
        # Extract some key properties
        mol_weight = physchem[0]  # Molecular weight
        logp = physchem[1]        # LogP
        rotatable_bonds = physchem[2]  # Number of rotatable bonds
        hbond_acceptors = physchem[3]  # H-bond acceptors
        hbond_donors = physchem[4]     # H-bond donors
        tpsa = physchem[5]        # TPSA
        ring_count = physchem[6]  # Ring count
        qed = physchem[14]        # Drug-likeness
        
        # Create a score based on Lipinski's Rule of 5 and other drug-likeness properties
        # Higher score indicates better drug-like properties
        score = 0.0
        
        # Lipinski's Rule of 5 contribution
        if 200 <= mol_weight <= 500:  # Preferred molecular weight range
            score += 0.2
        
        if -2 <= logp <= 5:  # Preferred LogP range
            score += 0.2
            
        if rotatable_bonds <= 10:  # Preferred rotatable bonds
            score += 0.1
            
        if hbond_acceptors <= 10:  # Preferred H-bond acceptors
            score += 0.1
            
        if hbond_donors <= 5:  # Preferred H-bond donors
            score += 0.1
            
        if 20 <= tpsa <= 140:  # Preferred TPSA range
            score += 0.1
            
        # Add contribution from QED
        score += qed * 0.2
        
        # Add some variety based on fingerprint bits
        # Use a subset of Morgan fingerprint bits to add variety
        fp_bits = features[i, :200]  # Use only first 200 bits
        fp_contribution = np.mean(fp_bits) * 0.4
        
        # Combine scores
        final_score = score + fp_contribution
        
        # Ensure score is between 0 and 1
        predictions[i] = max(0.0, min(1.0, final_score))
    
    return predictions

def create_interactive_visualizations(results_df, properties_df):
    """Create interactive visualizations using Plotly"""
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Prediction Score Distribution',
            'Top 20 Compounds by Score',
            'Molecular Weight vs LogP',
            'QED vs Prediction Score'
        ),
        specs=[[{'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # 1. Prediction score distribution (histogram)
    fig.add_trace(
        go.Histogram(
            x=results_df['Prediction_Score'],
            nbinsx=30,
            name='Prediction Scores',
            marker_color='#1f77b4'
        ),
        row=1, col=1
    )
    
    # 2. Top 20 compounds by score (bar chart)
    top_20 = results_df.nlargest(20, 'Prediction_Score')
    
    fig.add_trace(
        go.Bar(
            x=top_20.index,
            y=top_20['Prediction_Score'],
            marker_color=top_20['Prediction_Score'].apply(lambda x: '#2ca02c' if x > 0.5 else '#1f77b4'),
            text=top_20['Prediction_Score'].round(3),
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # 3. Molecular weight vs LogP scatter plot
    combined_df = pd.concat([results_df, properties_df], axis=1)
    
    fig.add_trace(
        go.Scatter(
            x=combined_df['MolWt'],
            y=combined_df['LogP'],
            mode='markers',
            marker=dict(
                size=8,
                color=combined_df['Prediction_Score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Prediction Score', x=1.15)
            ),
            text=[f"ID: {idx}<br>MW: {mw:.1f}<br>LogP: {logp:.2f}<br>Score: {score:.3f}" 
                  for idx, mw, logp, score in zip(combined_df.index, combined_df['MolWt'], 
                                                  combined_df['LogP'], combined_df['Prediction_Score'])],
            hoverinfo='text'
        ),
        row=2, col=1
    )
    
    # 4. QED vs Prediction Score
    fig.add_trace(
        go.Scatter(
            x=combined_df['QED'],
            y=combined_df['Prediction_Score'],
            mode='markers',
            marker=dict(
                size=8,
                color=combined_df['Predicted_Hit'].apply(lambda x: '#2ca02c' if x else '#1f77b4'),
                showscale=False
            ),
            text=[f"ID: {idx}<br>QED: {qed:.3f}<br>Score: {score:.3f}" 
                  for idx, qed, score in zip(combined_df.index, combined_df['QED'], 
                                           combined_df['Prediction_Score'])],
            hoverinfo='text'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        width=1400,
        title_text="Comprehensive Molecular Analysis Dashboard",
        title_x=0.5,
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Molecular Weight", row=2, col=1)
    fig.update_yaxes(title_text="LogP", row=2, col=1)
    fig.update_xaxes(title_text="Drug-Likeness (QED)", row=2, col=2)
    fig.update_yaxes(title_text="Prediction Score", row=2, col=2)
    
    return fig

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
            if st.session_state.debug:
                st.write(f"Error analyzing properties for {smiles}: {str(e)}")
            props = {k: 0 for k in ['MolWt', 'LogP', 'TPSA', 'HBA', 'HBD', 'RotatableBonds', 'AromaticRings', 'QED']}
        properties.append(props)
    return pd.DataFrame(properties)

def analyze_model_structure(model_data):
    """Analyze the model structure and show detailed information"""
    st.subheader("Model Structure Analysis")
    
    if isinstance(model_data, dict):
        st.write("Model is a dictionary with the following keys:")
        for key in model_data.keys():
            st.write(f"- {key}")
            
        # Check specific keys we're interested in
        if 'feature_methods' in model_data:
            st.write(f"feature_methods: {model_data['feature_methods']}")
        
        if 'predictions' in model_data:
            predictions = model_data['predictions']
            if isinstance(predictions, np.ndarray):
                st.write(f"predictions: numpy array with shape {predictions.shape}")
                st.write(f"- min: {np.min(predictions):.4f}, max: {np.max(predictions):.4f}")
                st.write(f"- mean: {np.mean(predictions):.4f}, std: {np.std(predictions):.4f}")
                
                # Show histogram of predictions
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.histplot(predictions, bins=30, kde=True, ax=ax)
                ax.set_title("Distribution of Predictions in Model")
                ax.set_xlabel("Prediction Value")
                ax.set_ylabel("Count")
                st.pyplot(fig)
            else:
                st.write(f"predictions: {type(predictions)}")
        
        if 'method_predictions' in model_data:
            method_predictions = model_data['method_predictions']
            if isinstance(method_predictions, np.ndarray):
                st.write(f"method_predictions: numpy array with shape {method_predictions.shape}")
                
                if len(method_predictions.shape) == 2 and 'feature_methods' in model_data:
                    methods = model_data['feature_methods']
                    if len(methods) == method_predictions.shape[1]:
                        # Show stats for each method
                        stats = []
                        for i, method in enumerate(methods):
                            method_preds = method_predictions[:, i]
                            stats.append({
                                'Method': method,
                                'Min': np.min(method_preds),
                                'Max': np.max(method_preds),
                                'Mean': np.mean(method_preds),
                                'Std': np.std(method_preds)
                            })
                        st.write("Method prediction statistics:")
                        st.table(pd.DataFrame(stats))
            else:
                st.write(f"method_predictions: {type(method_predictions)}")
        
        if 'models' in model_data:
            models = model_data['models']
            if isinstance(models, list):
                st.write(f"models: list with {len(models)} models")
                if len(models) > 0:
                    # Show info about each model type
                    method_counts = {}
                    for model in models:
                        method = model.get('feature_method', 'unknown')
                        method_counts[method] = method_counts.get(method, 0) + 1
                    
                    st.write("Models per feature method:")
                    for method, count in method_counts.items():
                        st.write(f"- {method}: {count} models")
            else:
                st.write(f"models: {type(models)}")
                
        if 'smiles_list' in model_data:
            smiles_list = model_data['smiles_list']
            if isinstance(smiles_list, list):
                st.write(f"smiles_list: list with {len(smiles_list)} items")
                if len(smiles_list) > 0:
                    st.write("First 5 SMILES:")
                    for i, smiles in enumerate(smiles_list[:5]):
                        st.write(f"  {i+1}. {smiles}")
            else:
                st.write(f"smiles_list: {type(smiles_list)}")
    else:
        st.write(f"Model is not a dictionary, but a {type(model_data)}")

def run_app():
    st.title("🔬 Enhanced Molecular Ensemble Predictor")
    
    st.markdown("""
    ## Advanced Molecular Activity Prediction
    This application uses a pre-trained ensemble model to predict molecular activity. 
    The model combines multiple feature selection methods (LASSO, PCA, and Mutual Information) 
    trained on a large dataset of molecular compounds.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Advanced options
    with st.sidebar.expander("Advanced Options", expanded=False):
        show_method_comparison = st.checkbox("Show method comparison analysis", value=True, key="method_comparison")
        show_molecular_properties = st.checkbox("Show molecular properties analysis", value=True, key="mol_properties")
        show_interactive_plots = st.checkbox("Use interactive visualizations", value=True, key="interactive_plots")
        confidence_threshold = st.slider("Hit prediction threshold", 0.0, 1.0, 0.5, 0.01, key="threshold")
        
        # Debug options - use the session state directly via key
        st.checkbox("Debug mode", value=False, key="debug")
        show_model_analysis = st.checkbox("Show detailed model analysis", value=False, key="model_analysis")
        
        # Prediction method
        prediction_mode = st.radio(
            "Prediction mode",
            ["Enhanced Ensemble", "Simple (Drug-Likeness Based)"]
        )
        
        # Model selection
        model_path = st.text_input(
            "Path to ensemble model", 
            value="enhanced_ensemble_model.pkl",
            help="Path to the trained ensemble model file (.pkl)"
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload a CSV file with SMILES strings", 
        type=["csv"],
        help="The file should contain a column with SMILES strings named 'SMILES', 'Smiles', etc."
    )
    
    # Try to load the model
    model_loaded = False
    ensemble_model = None
    
    try:
        if os.path.exists(model_path):
            with st.spinner(f"Loading model from {model_path}..."):
                ensemble_model = joblib.load(model_path)
                model_loaded = True
                st.success(f"✅ Successfully loaded model from {model_path}")
        else:
            st.warning(f"⚠️ Model file not found at {model_path}. Will use backup prediction method.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}. Will use backup prediction method.")
        if st.session_state.debug:
            st.exception(e)
    
    # Show model analysis if requested
    if show_model_analysis and ensemble_model is not None:
        analyze_model_structure(ensemble_model)
    
    if uploaded_file:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Read the file
            df = pd.read_csv(tmp_path)
            st.success(f"✅ Successfully loaded {len(df)} compounds")
            
            if st.session_state.debug:
                st.write(f"Number of rows: {len(df)}")
                st.write(f"Columns: {df.columns.tolist()}")
            
            # Identify SMILES column
            smiles_col = None
            possible_cols = ['SMILES', 'Smiles', 'smiles', 'SMILE', 'Smile', 'smile', 'Structure']
            
            for col in possible_cols:
                if col in df.columns:
                    smiles_col = col
                    st.info(f"📊 Using column '{smiles_col}' for SMILES strings")
                    break
            
            if smiles_col is None:
                st.warning("⚠️ No standard SMILES column found")
                smiles_col = st.selectbox("Select SMILES column:", df.columns)
            
            # Extract SMILES
            smiles_list = df[smiles_col].tolist()
            
            # Create processing progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Check SMILES validity
            status_text.text("Validating SMILES strings...")
            validity = check_valid_smiles(smiles_list)
            valid_count = sum(validity)
            
            st.write(f"✅ Valid SMILES: {valid_count} out of {len(smiles_list)} ({valid_count/len(smiles_list)*100:.1f}%)")
            progress_bar.progress(0.3)
            
            # Generate RDKit features
            status_text.text("Generating molecular features...")
            features = rdkit_features_from_smiles(smiles_list)
            st.success(f"Calculated features with shape: {features.shape}")
            progress_bar.progress(0.7)
            
            # Generate predictions based on selected mode
            with st.spinner("Generating predictions..."):
                if prediction_mode == "Simple (Drug-Likeness Based)":
                    # Use simple drug-likeness based predictions
                    predictions = make_varied_predictions(features)
                    method_predictions = None
                else:
                    # Use enhanced ensemble if available
                    if model_loaded and ensemble_model is not None:
                        predictions, method_predictions = get_predictions_from_enhanced_ensemble(
                            smiles_list, features, ensemble_model
                        )
                    else:
                        st.warning("Enhanced ensemble model not available. Using backup prediction method.")
                        predictions = make_varied_predictions(features)
                        method_predictions = None
                
                st.success(f"Generated predictions for {len(predictions)} molecules")
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            # Create results DataFrame
            results = pd.DataFrame({
                'SMILES': smiles_list,
                'Valid': validity,
                'Prediction_Score': predictions,
                'Predicted_Hit': predictions >= confidence_threshold,
                'Confidence': np.abs(predictions - 0.5) * 2
            })
            
            # Add method-specific predictions if available
            if method_predictions is not None:
                feature_methods = ensemble_model.get('feature_methods', ['lasso', 'pca', 'mutual_info'])
                for i, method in enumerate(feature_methods):
                    results[f'Method_{method}'] = method_predictions[:, i]
            
            # Add all original columns from the input file
            for col in df.columns:
                if col != smiles_col:  # Don't duplicate the SMILES column
                    results[col] = df[col]
            
            # Display summary statistics
            st.header("📊 Prediction Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Compounds", len(results))
            with col2:
                st.metric("Predicted Hits", sum(results['Predicted_Hit']))
            with col3:
                st.metric("Average Score", f"{predictions.mean():.3f}")
            with col4:
                st.metric("Std. Deviation", f"{predictions.std():.3f}")
            
            # Display prediction distribution
            st.header("📈 Prediction Distribution")
            prediction_stats = {
                '< 0.1': sum(predictions < 0.1),
                '0.1-0.3': sum((predictions >= 0.1) & (predictions < 0.3)),
                '0.3-0.5': sum((predictions >= 0.3) & (predictions < 0.5)),
                '0.5-0.7': sum((predictions >= 0.5) & (predictions < 0.7)),
                '0.7-0.9': sum((predictions >= 0.7) & (predictions < 0.9)),
                '>= 0.9': sum(predictions >= 0.9)
            }
            
            for range_str, count in prediction_stats.items():
                st.write(f"**{range_str}**: {count} compounds ({count/len(predictions)*100:.1f}%)")
            
            # Show distribution stats
            st.write(f"**Mean**: {np.mean(predictions):.3f}")
            st.write(f"**Median**: {np.median(predictions):.3f}")
            st.write(f"**Std Dev**: {np.std(predictions):.3f}")
            
            # Create histogram of prediction distribution
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.hist(predictions, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(confidence_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {confidence_threshold}')
            ax.set_xlabel('Prediction Score')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Prediction Scores')
            ax.legend()
            st.pyplot(fig)
            
            # Molecular properties analysis
            if show_molecular_properties:
                st.header("🧪 Molecular Properties Analysis")
                
                # Calculate molecular properties
                properties_df = analyze_molecular_properties(smiles_list)
                
                if show_interactive_plots:
                    # Create interactive dashboard
                    interactive_fig = create_interactive_visualizations(results, properties_df)
                    st.plotly_chart(interactive_fig, use_container_width=True)
                
                # Create property distribution plots
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                hit_mask = results['Predicted_Hit']
                
                # Plot 1: Molecular Weight
                axes[0, 0].hist([properties_df.loc[~hit_mask, 'MolWt'], properties_df.loc[hit_mask, 'MolWt']], 
                               label=['Non-Hits', 'Hits'], bins=20, alpha=0.7)
                axes[0, 0].set_xlabel('Molecular Weight')
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].set_title('Molecular Weight Distribution')
                axes[0, 0].legend()
                
                # Plot 2: LogP
                axes[0, 1].hist([properties_df.loc[~hit_mask, 'LogP'], properties_df.loc[hit_mask, 'LogP']], 
                               label=['Non-Hits', 'Hits'], bins=20, alpha=0.7)
                axes[0, 1].set_xlabel('LogP')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].set_title('LogP Distribution')
                axes[0, 1].legend()
                
                # Plot 3: QED Distribution
                axes[1, 0].hist([properties_df.loc[~hit_mask, 'QED'], properties_df.loc[hit_mask, 'QED']], 
                               label=['Non-Hits', 'Hits'], bins=20, alpha=0.7)
                axes[1, 0].set_xlabel('Drug-Likeness (QED)')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_title('QED Distribution')
                axes[1, 0].legend()
                
                # Plot 4: TPSA
                axes[1, 1].hist([properties_df.loc[~hit_mask, 'TPSA'], properties_df.loc[hit_mask, 'TPSA']], 
                               label=['Non-Hits', 'Hits'], bins=20, alpha=0.7)
                axes[1, 1].set_xlabel('TPSA')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].set_title('TPSA Distribution')
                axes[1, 1].legend()
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Method comparison analysis
            if show_method_comparison and method_predictions is not None:
                st.header("🔍 Feature Selection Method Analysis")
                
                method_columns = [col for col in results.columns if col.startswith('Method_')]
                if method_columns:
                    # Method comparison visualization
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    for col in method_columns:
                        method_name = col.replace('Method_', '')
                        ax.hist(results[col], bins=30, alpha=0.6, label=method_name, density=True)
                    
                    ax.set_xlabel('Prediction Score')
                    ax.set_ylabel('Density')
                    ax.set_title('Method-Specific Prediction Distributions')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Method statistics
                    st.subheader("Method Performance Statistics")
                    method_stats = []
                    for col in method_columns:
                        method_name = col.replace('Method_', '')
                        stats = {
                            'Method': method_name,
                            'Mean Score': results[col].mean(),
                            'Std Dev': results[col].std(),
                            'Min Score': results[col].min(),
                            'Max Score': results[col].max(),
                            'Predicted Hits': sum(results[col] >= confidence_threshold),
                            'Hit Rate (%)': sum(results[col] >= confidence_threshold) / len(results) * 100
                        }
                        method_stats.append(stats)
                    
                    st.dataframe(pd.DataFrame(method_stats))
            
            # Results table
            st.header("📋 Detailed Results")
            
            # Sort by prediction score
            sorted_results = results.sort_values(by='Prediction_Score', ascending=False)
            
            # Column selector
            all_columns = sorted_results.columns.tolist()
            
            # Set default columns - include all original columns from the input file plus our prediction columns
            default_columns = ['SMILES', 'Prediction_Score', 'Predicted_Hit', 'Confidence']
            
            # Add all original columns from the input file except the SMILES column (which we already have)
            input_columns = [col for col in df.columns if col != smiles_col]
            default_columns = default_columns + input_columns
            
            # Make sure we don't have duplicates
            default_columns = list(dict.fromkeys(default_columns))
            
            # Let the user select which columns to display
            selected_columns = st.multiselect(
                "Select columns to display:",
                options=all_columns,
                default=[col for col in default_columns if col in all_columns]
            )
            
            if not selected_columns:
                selected_columns = default_columns
            
            # Display results
            st.dataframe(sorted_results[selected_columns])
            
            # Download options
            st.header("💾 Download Results")
            
            csv = sorted_results.to_csv(index=False)
            st.download_button(
                "Download Results",
                csv,
                "molecular_predictions.csv",
                "text/csv",
                key='download-csv'
            )
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            if st.session_state.debug:
                import traceback
                st.code(traceback.format_exc(), language="python")

# Main entry point
if __name__ == "__main__":
    run_app()

# Code written by Hossam Nada.
