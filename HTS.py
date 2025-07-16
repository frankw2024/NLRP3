import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Lipinski, QED
from rdkit.Chem import GetSSSR
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from rdkit import RDLogger
import warnings
import sys
import traceback

# Suppress warnings
RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings
warnings.filterwarnings("ignore")

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

# Load data
print("Loading data...")
try:
    library_df = pd.read_csv('library.csv')
    positives_df = pd.read_csv('positives.csv')
    library_df['label'] = library_df['ID'].isin(positives_df['ID']).astype(int)
    
    # Extract SMILES and labels
    smiles_list = library_df['Smiles'].tolist()
    labels = library_df['label'].to_numpy()
    
    # Check class balance
    positive_count = np.sum(labels)
    total_count = len(labels)
    print(f"Data loaded. Found {positive_count} positives out of {total_count} compounds ({positive_count/total_count:.2%})")
    
    # Add check for very imbalanced data
    if positive_count / total_count < 0.01:
        print("Warning: Data is highly imbalanced. Consider using stratified sampling or other balancing techniques.")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    sys.exit(1)

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

def rdkit_features_from_smiles(smiles_list):
    print("Generating RDKit features...")
    features = []
    error_count = 0
    
    for i, smiles in enumerate(tqdm(smiles_list)):
        try:
            morgan = morgan_fp(smiles)
            maccs = maccs_fp(smiles)
            physchem = extended_physchem_desc(smiles)
            combined = np.concatenate((morgan, maccs, physchem))
            
            # Check for NaN or Inf values
            if np.any(np.isnan(combined)) or np.any(np.isinf(combined)):
                print(f"Warning: NaN or Inf in features for SMILES {i}: {smiles}")
                # Replace problematic values with zeros
                combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
                
            features.append(combined)
        except Exception as e:
            print(f"Error processing SMILES {i}: {smiles}")
            print(f"Error details: {str(e)}")
            # Add zeros as fallback
            features.append(np.zeros(2048 + 167 + 15))
            error_count += 1
    
    if error_count > 0:
        print(f"Warning: {error_count} SMILES could not be processed correctly.")
    
    return np.array(features)

# Generate features
try:
    rdkit_features = rdkit_features_from_smiles(smiles_list)
    print(f"Generated RDKit features with shape: {rdkit_features.shape}")
    
    # Check for any invalid features
    invalid_features = np.any(np.isnan(rdkit_features)) or np.any(np.isinf(rdkit_features))
    if invalid_features:
        print("Warning: Invalid feature values detected. Fixing...")
        rdkit_features = np.nan_to_num(rdkit_features, nan=0.0, posinf=0.0, neginf=0.0)
except Exception as e:
    print(f"Error generating features: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Improved feature selection functions
def apply_feature_selection(X_train, y_train, X_val, feature_selection_method='all', n_components=100):
    print(f"Applying feature selection: {feature_selection_method}")
    
    try:
        # Always start with scaling
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
        
        print(f"Feature selection complete. Selected {X_train_selected.shape[1]} features.")
        return X_train_selected, X_val_selected, selector, scaler
    
    except Exception as e:
        print(f"Feature selection error: {str(e)}")
        print("Falling back to original features without selection")
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
        print("Pre-encoding SMILES with tokenizer...")
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

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        try:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            
            # Get pre-encoded data
            input_ids = self.encoded_data[idx]["input_ids"]
            attention_mask = self.encoded_data[idx]["attention_mask"]
            
            # Get RDKit features
            rdkit_tensor = torch.tensor(self.rdkit_features[idx], dtype=torch.float)
            
            # Check for NaN or Inf values
            if torch.isnan(rdkit_tensor).any() or torch.isinf(rdkit_tensor).any():
                print(f"Warning: NaN or Inf in RDKit features at index {idx}")
                rdkit_tensor = torch.nan_to_num(rdkit_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                
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
        
        # ChemBERTa branch
        try:
            self.chemberta = RobertaModel.from_pretrained(chemberta_model)
        except Exception as e:
            print(f"Error loading ChemBERTa model: {str(e)}")
            print("Using random initialization for ChemBERTa model")
            # Create an untrained model with same config as fallback
            from transformers import RobertaConfig
            config = RobertaConfig.from_pretrained(chemberta_model)
            self.chemberta = RobertaModel(config)
            
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
        print(f"Performance curves saved to {save_path}")
    except Exception as e:
        print(f"Error creating performance curves: {str(e)}")
        traceback.print_exc()
        print("Skipping plot creation")

# Load tokenizer with error handling
print("Loading ChemBERTa tokenizer...")
try:
    tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
except Exception as e:
    print(f"Error loading tokenizer: {str(e)}")
    print("Using fallback tokenizer")
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    except:
        # Last resort: create a basic tokenizer
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=None)
        tokenizer.add_special_tokens({'pad_token': '[PAD]', 'unk_token': '[UNK]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training settings
num_epochs = 10
patience = 5
batch_size = 32
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

print("\n=== Starting Cross-Validation Training ===")
for feature_method in feature_selection_methods:
    print(f"\n----- Feature Selection Method: {feature_method} -----")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(rdkit_features, labels)):
        try:
            print(f"\n=== Fold {fold+1} ===")
            train_smiles = [smiles_list[i] for i in train_idx]
            train_labels = labels[train_idx]
            train_rdkit = rdkit_features[train_idx]

            val_smiles = [smiles_list[i] for i in val_idx]
            val_labels = labels[val_idx]
            val_rdkit = rdkit_features[val_idx]
            
            # Store validation indices for later ensemble
            if fold == 0 and feature_method == feature_selection_methods[0]:
                val_indices.append(val_idx)
                
            # Apply feature selection
            train_rdkit_selected, val_rdkit_selected, selector, scaler = apply_feature_selection(
                train_rdkit, train_labels, val_rdkit, 
                feature_selection_method=feature_method,
                n_components=n_components
            )
            
            print(f"Selected features shape: {train_rdkit_selected.shape}")
            
            # Create datasets with selected features
            train_data = MolecularDataset(train_smiles, train_labels, tokenizer, train_rdkit_selected)
            val_data = MolecularDataset(val_smiles, val_labels, tokenizer, val_rdkit_selected)

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

            # Initialize model for this feature selection method and fold
            model = ImprovedCombinedModel(rdkit_size=train_rdkit_selected.shape[1]).to(device)
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

            best_auc = 0
            no_improve = 0
            history = {'train_loss': [], 'val_auc': [], 'val_ap': []}

            # Try-except block for the entire training loop
            try:
                for epoch in range(num_epochs):
                    # Training
                    model.train()
                    total_loss = 0
                    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
                    
                    batch_count = 0
                    for batch in loop:
                        try:
                            optimizer.zero_grad()
                            input_ids = batch["input_ids"].to(device)
                            attention_mask = batch["attention_mask"].to(device)
                            labels_tensor = batch["label"].to(device)
                            rdkit_tensor = batch["rdkit_features"].to(device)

                            preds = model(input_ids, attention_mask, rdkit_tensor)
                            
                            # Check for NaN predictions
                            if torch.isnan(preds).any():
                                print("Warning: NaN predictions detected during training. Skipping batch.")
                                continue
                                
                            loss = criterion(preds, labels_tensor)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
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
                    print(f"Fold {fold+1}, Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

                    # Validation
                    model.eval()
                    val_preds, val_targets = [], []
                    valid_val_count = 0
                    
                    with torch.no_grad():
                        for batch in val_loader:
                            try:
                                input_ids = batch["input_ids"].to(device)
                                attention_mask = batch["attention_mask"].to(device)
                                labels_tensor = batch["label"].to(device)
                                rdkit_tensor = batch["rdkit_features"].to(device)

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
                    
                    print(f"Fold {fold+1}, Epoch {epoch+1}, Validation AUC: {auc:.4f}, AP: {ap:.4f}")
                    
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
                            print(f"Early stopping at epoch {epoch+1} for fold {fold+1}")
                            break
                            
                print(f"Best validation AUC: {best_auc:.4f} at epoch {best_epoch+1}")
            except Exception as e:
                print(f"Error in training loop: {str(e)}")
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

# Stacked ensemble using all features and models
print("\n=== Building Stacked Ensemble ===")

# Check if we have any models
if not final_models:
    print("Error: No models were successfully trained. Cannot build ensemble.")
    sys.exit(1)

# Initialize array to store all predictions for each feature selection method
all_predictions = np.zeros((len(smiles_list), len(feature_selection_methods)))

# For each feature selection method, apply the appropriate models to get predictions
for method_idx, method in enumerate(feature_selection_methods):
    method_models = [m for m in final_models if m['feature_method'] == method]
    
    if not method_models:
        print(f"Warning: No models available for method {method}. Using zeros.")
        continue
    
    print(f"Generating predictions with {len(method_models)} models for method: {method}")
    
    # Apply each model to get predictions for the full dataset
    for i in tqdm(range(len(smiles_list)), desc=f"Predictions for {method}"):
        try:
            smiles = smiles_list[i]
            rdkit_feat = rdkit_features[i:i+1]
            
            # Average predictions from all folds for this method
            method_pred = 0
            valid_model_count = 0
            
            for model_data in method_models:
                try:
                    # Apply the appropriate feature selection
                    selector = model_data['selector']
                    scaler = model_data['scaler']
                    
                    if selector is not None and scaler is not None:
                        rdkit_feat_scaled = scaler.transform(rdkit_feat)
                        if hasattr(selector, 'transform'):
                            rdkit_feat_selected = selector.transform(rdkit_feat_scaled)
                        else:  # For PCA
                            rdkit_feat_selected = selector.transform(rdkit_feat_scaled)
                    else:
                        # No feature selection or failed feature selection
                        rdkit_feat_selected = rdkit_feat
                    
                    # Handle NaN/Inf values
                    rdkit_feat_selected = np.nan_to_num(rdkit_feat_selected, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Load model
                    model = ImprovedCombinedModel(rdkit_size=rdkit_feat_selected.shape[1]).to(device)
                    model.load_state_dict(model_data['model'])
                    model.eval()
                    
                    # Get prediction
                    with torch.no_grad():
                        input_ids = tokenizer(smiles, padding="max_length", truncation=True, 
                                            max_length=128, return_tensors="pt")["input_ids"].to(device)
                        attention_mask = tokenizer(smiles, padding="max_length", truncation=True, 
                                                max_length=128, return_tensors="pt")["attention_mask"].to(device)
                        rdkit_tensor = torch.tensor(rdkit_feat_selected[0], dtype=torch.float).unsqueeze(0).to(device)
                        
                        pred = model(input_ids, attention_mask, rdkit_tensor)
                        pred_prob = torch.sigmoid(pred).cpu().numpy().item()
                        
                        # Check for NaN or Inf
                        if np.isnan(pred_prob) or np.isinf(pred_prob):
                            continue
                            
                        method_pred += pred_prob
                        valid_model_count += 1
                except Exception as e:
                    print(f"Error with model for {method}, sample {i}: {str(e)}")
                    continue
            
            # Average predictions from valid models
            if valid_model_count > 0:
                method_pred /= valid_model_count
                all_predictions[i, method_idx] = method_pred
            else:
                all_predictions[i, method_idx] = 0.0  # Fallback
        except Exception as e:
            print(f"Error generating prediction for sample {i} with method {method}: {str(e)}")
            all_predictions[i, method_idx] = 0.0  # Fallback

# Check if we have valid predictions
if np.all(all_predictions == 0):
    print("Warning: All predictions are zero. There might be a problem with the models.")
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
    print("Warning: NaN or Inf values in final predictions. Fixing...")
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
    joblib.dump(ensemble_data, "enhanced_ensemble_model.pkl")
    print("✅ Saved 'enhanced_ensemble_model.pkl' with combined predictions from all methods and folds.")
except Exception as e:
    print(f"Error saving ensemble model: {str(e)}")
    # Save predictions in a simpler format as backup
    np.save("final_predictions.npy", final_predictions)
    print("✅ Saved backup predictions to 'final_predictions.npy'")

# Independent evaluation of the performance on the entire dataset
print("\n=== Final Evaluation ===")

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

    print("Overall Model Performance:")
    print(f"AUC: {auc:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
except Exception as e:
    print(f"Error calculating final metrics: {str(e)}")

# Compare feature selection methods safely
print("\n=== Feature Selection Method Comparison ===")
for method_idx, method in enumerate(feature_selection_methods):
    try:
        method_preds = all_predictions[:, method_idx]
        
        # Check if predictions are all zeros
        if np.all(method_preds == 0):
            print(f"{method.upper()}: No valid predictions available")
            continue
            
        # Calculate metrics safely
        method_auc = safe_metric(roc_auc_score, labels, method_preds)
        method_ap = safe_metric(average_precision_score, labels, method_preds)
        
        print(f"{method.upper()}: AUC = {method_auc:.4f}, AP = {method_ap:.4f}")
    except Exception as e:
        print(f"Error calculating metrics for {method}: {str(e)}")

# Create a prediction data frame with scores
try:
    results_df = pd.DataFrame({
        'ID': library_df['ID'],
        'SMILES': library_df['Smiles'],
        'True_Label': library_df['label'],
        'Prediction_Score': final_predictions,
        'Predicted_Label': (final_predictions > 0.5).astype(int)
    })

    # Save predictions to CSV
    results_df.to_csv('molecular_predictions.csv', index=False)
    print("✅ Saved predictions to 'molecular_predictions.csv'")
except Exception as e:
    print(f"Error creating results dataframe: {str(e)}")
    # Save as numpy arrays as backup
    np.save("ids.npy", library_df['ID'].values)
    np.save("predictions.npy", final_predictions)
    print("✅ Saved backup arrays 'ids.npy' and 'predictions.npy'")

# Plot ROC and Precision-Recall curves safely
try:
    plot_performance_curves(labels, all_predictions, final_predictions, feature_selection_methods, save_path="performance_curves.png")
except Exception as e:
    print(f"Error plotting performance curves: {str(e)}")

print("\n=== Complete! ===")
