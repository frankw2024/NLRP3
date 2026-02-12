#!/usr/bin/env python3
"""
HTS_3D_inference.py
-------------------

Inference script for HTS-3D model. Loads a trained model and makes predictions on new SMILES data.
Aligned with HTS_3D.py: same preprocessing (scaler, selector, nan_to_num), model architecture,
batch sizes (128 GPU / 64 CPU), protein/pocket handling, and deduplication (value_nM / molecule_chembl_id).

Usage:
    python HTS_3D_inference.py --model output/hts3d_model_nlrp3.pt --input confirmednlrp3_smiles.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import torch.nn as nn
from torch.utils.data import DataLoader

# Import necessary components from HTS_3D
# Use absolute import to avoid issues
import sys
sys.path.insert(0, str(Path(__file__).parent))

import HTS_3D as _hts3d_module
# Register picklable classes so torch.load can unpickle scaler/selector saved by HTS_3D
for _name in ("DummyScaler", "DummyPCA", "DummyLassoSelector", "DummyMISelector"):
    if hasattr(_hts3d_module, _name):
        setattr(sys.modules["__main__"], _name, getattr(_hts3d_module, _name))

from HTS_3D import (
    HTS3DDataset,
    HTS3DModel,
    PROTEIN_TEMPLATES,
    build_3d_cache,
    build_rdkit_feature_matrix,
    compute_qed_for_smiles_list,
    detect_compound_from_csv_path,
    detect_device,
    find_latest_checkpoint,
    get_model_path_for_compound,
    load_library_and_positives,
    load_protein_structure_from_pdb,
    load_tokenizer,
    prepare_tensors,
    MAX_SEQ_LEN,
    MAX_ATOMS,
    SCRIPT_DIR,
    PROJECT_ROOT,
    OUTPUT_DIR,
)

DEVICE = detect_device()
NON_BLOCKING = DEVICE.type != "cpu"

# Match HTS_3D.py inference batch sizes (ensemble prediction loop)
def _default_inference_batch_size() -> int:
    return 128 if DEVICE.type == "cuda" else 64


def _load_legacy_from_checkpoint(
    state_dict: dict,
    compound: str,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[Optional[object], Optional[object], int]:
    """
    Load scaler and selector from a checkpoint for legacy (state_dict-only) model files.
    If checkpoint_path is None, searches output/checkpoints/ for final_checkpoint_{compound}_*.pt.
    """
    ckpt_path = checkpoint_path
    if ckpt_path is None:
        checkpoint_dir = OUTPUT_DIR / "checkpoints"
        ckpt_path = find_latest_checkpoint(checkpoint_dir, f"final_checkpoint_{compound}_*.pt")
        if ckpt_path is None:
            ckpt_path = find_latest_checkpoint(checkpoint_dir, "final_checkpoint_*.pt")

    if ckpt_path is None or not ckpt_path.exists():
        return None, None, 0

    print(f"[inference] Loading scaler/selector from checkpoint: {ckpt_path}")
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    all_models = ckpt_data.get("all_models", [])
    if not all_models:
        return None, None, 0

    # Use first model with valid scaler/selector (rdkit_dim is same across folds for same method)
    for m in all_models:
        if m.get("scaler") is not None and m.get("selector") is not None:
            rdkit_dim = m.get("rdkit_dim", 200)
            return m["scaler"], m["selector"], rdkit_dim

    return None, None, 0


def load_model(
    model_path: Path,
    checkpoint_path: Optional[Path] = None,
    protein_path: Optional[Path] = None,
) -> Tuple[HTS3DModel, object, object]:
    """
    Load a trained HTS3DModel and preprocessing (scaler, selector).

    Supports:
    - New bundle format (state_dict + scaler + selector + rdkit_dim)
    - Legacy format: state_dict only; auto-searches output/checkpoints/ for a matching
      checkpoint to load scaler/selector, or use --checkpoint to specify one.

    Args:
        protein_path: Optional path to PDB file for protein template (e.g. src/1YJD.pdb for CD28).
                      If specified, uses protein structure from PDB; otherwise uses default (7ALV).

    Returns:
        (model, scaler, selector) - scaler and selector preprocess RDKit features.
    """
    print(f"[inference] Loading model from {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    compound = detect_compound_from_csv_path(model_path)
    print(f"[inference] Detected compound type: {compound}")

    loaded = torch.load(model_path, map_location=DEVICE, weights_only=False)

    state_dict = None
    scaler = None
    selector = None
    rdkit_dim = None

    # Architecture options (Minimal vs Ultra); inferred from state_dict if not in bundle
    use_attention_pooling = False
    use_interaction_features = False
    use_adaptive_weighting = False
    max_pockets_ultra = 1

    # New bundle format (saved by HTS_3D training)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        state_dict = loaded["state_dict"]
        scaler = loaded["scaler"]
        selector = loaded["selector"]
        rdkit_dim = loaded["rdkit_dim"]
        feature_method = loaded.get("feature_method", "unknown")
        # Read architecture from bundle if saved (future models)
        use_attention_pooling = loaded.get("use_attention_pooling", use_attention_pooling)
        use_interaction_features = loaded.get("use_interaction_features", use_interaction_features)
        use_adaptive_weighting = loaded.get("use_adaptive_weighting", use_adaptive_weighting)
        max_pockets_ultra = loaded.get("max_pockets_ultra", max_pockets_ultra)
        print(f"[inference] Bundle format: rdkit_dim={rdkit_dim}, feature_method={feature_method}")

    # Legacy format: raw state_dict
    elif isinstance(loaded, dict) and (
        "rdkit_branch.0.weight" in loaded or "rdkit_branch.block1.0.weight" in loaded
    ):
        state_dict = loaded
        if "rdkit_branch.block1.0.weight" in loaded:
            rdkit_dim = int(state_dict["rdkit_branch.block1.0.weight"].shape[1])
        else:
            rdkit_dim = int(state_dict["rdkit_branch.0.weight"].shape[1])
        print(f"[inference] Legacy format detected (state_dict only), rdkit_dim={rdkit_dim}")
        scaler, selector, ckpt_rdkit = _load_legacy_from_checkpoint(
            state_dict, compound, checkpoint_path
        )
        if rdkit_dim is None and ckpt_rdkit:
            rdkit_dim = ckpt_rdkit
        if scaler is None or selector is None:
            ckpt_dir = OUTPUT_DIR / "checkpoints"
            raise RuntimeError(
                "Model file is in legacy format (state_dict only). "
                "Scaler/selector not found in checkpoint. "
                f"Options: 1) Retrain with current HTS_3D.py. "
                f"2) Use --checkpoint <path> to point to a final_checkpoint_*.pt from the same run. "
                f"3) Check if a checkpoint exists in {ckpt_dir}"
            )

    else:
        raise RuntimeError(
            f"Unknown model format. Expected bundle (state_dict, scaler, selector) or legacy state_dict. "
            f"Keys found: {list(loaded.keys())[:10] if isinstance(loaded, dict) else 'not a dict'}"
        )

    # Infer architecture from state_dict when not in bundle (backward compat with Ultra models)
    sd_keys = set(state_dict.keys())
    if "ligand_3d_pool.attention.0.weight" in sd_keys:
        use_attention_pooling = True
    if "interaction_module.protein_proj.weight" in sd_keys:
        use_interaction_features = True
    if "feature_fusion.gate.0.weight" in sd_keys:
        use_adaptive_weighting = True
    if use_attention_pooling or use_interaction_features or use_adaptive_weighting:
        max_pockets_ultra = 3  # Ultra default

    atom_dim = 16
    max_pockets = 5  # Match HTS_3D main() default
    if protein_path and protein_path.exists():
        protein_pockets = load_protein_structure_from_pdb(protein_path, max_pockets=max_pockets)
        print(f"[inference] Using protein from {protein_path.name} ({len(protein_pockets)} pockets)")
    else:
        protein_pockets = PROTEIN_TEMPLATES
        if protein_path:
            print(f"[inference] Protein file not found: {protein_path}, using default (7ALV)")
    num_pockets = len(protein_pockets)  # Match HTS_3D: num_pockets from template list
    model = HTS3DModel(
        rdkit_dim=rdkit_dim,
        atom_dim=atom_dim,
        protein_pockets=protein_pockets,
        num_pockets=num_pockets,
        use_attention_pooling=use_attention_pooling,
        use_interaction_features=use_interaction_features,
        use_adaptive_weighting=use_adaptive_weighting,
        max_pockets_ultra=max_pockets_ultra,
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    print(f"[inference] Model loaded successfully")
    return model, scaler, selector


def predict_on_smiles(
    model: HTS3DModel,
    smiles_list: list[str],
    scaler: object,
    selector: object,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Make predictions on a list of SMILES strings.
    Preprocessing (scaler, selector, nan_to_num) matches HTS_3D.py training pipeline.

    Args:
        model: Trained HTS3DModel
        smiles_list: List of SMILES strings
        scaler: Fitted scaler from training (scale RDKit features)
        selector: Fitted feature selector from training (reduce to rdkit_dim)
        batch_size: Batch size for inference. If None, uses 128 (GPU) or 64 (CPU) to match HTS_3D.

    Returns:
        Array of prediction probabilities
    """
    if batch_size is None:
        batch_size = _default_inference_batch_size()
    print(f"[inference] Processing {len(smiles_list)} SMILES strings...")

    df = pd.DataFrame({"canonical_smiles": smiles_list})

    print("[inference] Tokenizing SMILES...")
    input_ids, attention_mask = prepare_tensors(df)
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    print("[inference] Generating RDKit 2D features...")
    rdkit_feats = build_rdkit_feature_matrix(smiles_list)

    # Apply same preprocessing as training: scale then feature selection
    rdkit_np = rdkit_feats.cpu().numpy() if rdkit_feats.device.type != "cpu" else rdkit_feats.numpy()
    rdkit_np = np.nan_to_num(rdkit_np, nan=0.0, posinf=0.0, neginf=0.0)
    rdkit_scaled = scaler.transform(rdkit_np)
    rdkit_selected = selector.transform(rdkit_scaled)
    rdkit_feats = torch.tensor(rdkit_selected, dtype=torch.float32, device=DEVICE)
    
    # Generate 3D conformer features
    print("[inference] Generating 3D conformer features...")
    atom_feats, atom_masks = build_3d_cache(smiles_list)
    
    # Create dataset
    labels = np.zeros(len(smiles_list), dtype=np.float32)  # Dummy labels for inference
    dataset = HTS3DDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        rdkit_feats=rdkit_feats,
        atom_feats=atom_feats,
        atom_masks=atom_masks,
        labels=labels,
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    # Make predictions
    print("[inference] Generating predictions...")
    predictions = []
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            # Move batch to device
            input_ids_batch = sample["input_ids"].to(DEVICE, non_blocking=NON_BLOCKING)
            attention_mask_batch = sample["attention_mask"].to(DEVICE, non_blocking=NON_BLOCKING)
            rdkit_batch = sample["rdkit"].to(DEVICE, non_blocking=NON_BLOCKING)
            atom_features_batch = sample["atom_features"].to(DEVICE, non_blocking=NON_BLOCKING)
            atom_mask_batch = sample["atom_mask"].to(DEVICE, non_blocking=NON_BLOCKING)
            
            # Forward pass
            logits = model(
                input_ids_batch,
                attention_mask_batch,
                rdkit_batch,
                atom_features_batch,
                atom_mask_batch,
            )
            
            # Convert to probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.append(probs)
            
            if (batch_idx + 1) % max(1, len(loader) // 10) == 0:
                print(f"[inference] Processed {batch_idx + 1}/{len(loader)} batches")
    
    predictions = np.concatenate(predictions)
    
    # Handle NaN/Inf
    if np.isnan(predictions).any() or np.isinf(predictions).any():
        print("[inference] Warning: NaN/Inf detected in predictions, replacing with 0.5")
        predictions = np.nan_to_num(predictions, nan=0.5, posinf=1.0, neginf=0.0)
    
    print(f"[inference] Generated predictions for {len(predictions)} molecules")
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="HTS-3D Inference: Make predictions on new SMILES data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model and input:
  python HTS_3D_inference.py --input confirmednlrp3_smiles.csv
  
  # Specify model and output:
  python HTS_3D_inference.py --model output/hts3d_model_nlrp3.pt --input confirmednlrp3_smiles.csv --output predictions.csv
  
  # Use CD28 model (with 1YJD protein template to match training):
  python HTS_3D_inference.py --model output/hts3d_model_cd28.pt --input cd28_smiles.csv --output cd28_predictions.csv --protein src/1YJD.pdb

  # CD28: load libraries/library.csv and libraries/positives.csv (same as HTS_3D --cd28):
  python HTS_3D_inference.py --cd28 --output cd28_predictions.csv

  # Legacy model with checkpoint (load scaler/selector from checkpoint):
  python HTS_3D_inference.py --model output/hts3d_model_nlrp3.pt --input data.csv --checkpoint output/checkpoints/final_checkpoint_nlrp3_20260128_120000.pt
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model file (.pt). If not specified, auto-detects based on input CSV."
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to input CSV file with SMILES strings. Not required when --cd28 is set."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file. If not specified, auto-generates based on input filename."
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference. Default: 128 (GPU) or 64 (CPU), matching HTS_3D.py."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for predicted hits and metrics (default: 0.5). Matches HTS_3D reporting."
    )

    parser.add_argument(
        "--qed-blend",
        type=float,
        default=0.2,
        help="Blend predictions with QED (0-1). 0.2 matches HTS_3D training (HTS3DOracle drug-likeness). Use 0 to disable."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="For legacy model files: path to final_checkpoint_*.pt to load scaler/selector. Auto-searches output/checkpoints/ if not specified."
    )

    parser.add_argument(
        "--protein",
        type=str,
        default=None,
        help="Path to PDB file for protein template (e.g. src/1YJD.pdb for CD28). If not specified, uses default NLRP3 template (7ALV)."
    )

    parser.add_argument(
        "--cd28",
        action="store_true",
        help="Use HTS.py-style CD28 data: load libraries/library.csv and libraries/positives.csv; label=1 if SMILES in positives else 0. Same behavior as HTS_3D.py --cd28. Overrides --input."
    )

    args = parser.parse_args()

    if not args.cd28 and not args.input:
        parser.error("--input is required when --cd28 is not set")

    # Determine compound type and model path
    compound = "cd28" if args.cd28 else None

    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
    else:
        compound = compound or ("cd28" if args.cd28 else "nlrp3")
        model_path = get_model_path_for_compound(compound)
        print(f"[inference] Using compound: {compound}, model: {model_path}")

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    if not compound:
        compound = detect_compound_from_csv_path(Path(args.input))

    # Determine output path
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        output_path = OUTPUT_DIR / f"hts3d_predictions_{compound}_{timestamp}.csv"

    # Load input data: --cd28 uses libraries/library.csv + libraries/positives.csv (same as HTS_3D --cd28)
    if args.cd28:
        library_path = PROJECT_ROOT / "libraries" / "library.csv"
        positives_path = PROJECT_ROOT / "libraries" / "positives.csv"
        print(f"[inference] Loading library + positives (--cd28): {library_path.name}, {positives_path.name}")
        df = load_library_and_positives(library_path, positives_path)
        print(f"[inference] Loaded {len(df)} molecules ({df['label'].sum()} actives, {len(df) - df['label'].sum()} inactives)")
    else:
        # Load input CSV (try different encodings)
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        print(f"[inference] Loading input CSV: {input_path}")
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(input_path, encoding=encoding)
                print(f"[inference] Successfully loaded CSV with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            print(f"Error: Could not read CSV file with any of the attempted encodings: {encodings}")
            sys.exit(1)
        # Normalize SMILES column name
        smiles_col = None
        possible_smiles_cols = ['canonical_smiles', 'SMILES', 'Smiles', 'smiles', 'SMILE', 'Smile', 'smile', 'Structure']
        for col in possible_smiles_cols:
            if col in df.columns:
                smiles_col = col
                break
        if smiles_col is None:
            print(f"Error: No SMILES column found. Expected one of: {possible_smiles_cols}")
            sys.exit(1)
        if smiles_col != "canonical_smiles":
            df["canonical_smiles"] = df[smiles_col]
        # Normalize molecule_chembl_id
        mol_id_col = None
        for col in ['molecule_chembl_id', 'Molecule ChEMBL ID', 'molecule_chEMBL_id', 'molecule_id', 'compound_id']:
            if col in df.columns:
                mol_id_col = col
                break
        if mol_id_col and mol_id_col != "molecule_chembl_id":
            df["molecule_chembl_id"] = df[mol_id_col]

    # Filter out rows with missing SMILES
    original_len = len(df)
    df = df.dropna(subset=["canonical_smiles"])
    if len(df) < original_len:
        print(f"[inference] Removed {original_len - len(df)} rows with missing SMILES")
    
    # Ensure molecule_chembl_id exists (create if missing)
    if "molecule_chembl_id" not in df.columns:
        # Create unique IDs based on row index (preserve original order)
        df["molecule_chembl_id"] = [f"COMPOUND_{i:06d}" for i in range(len(df))]
        print(f"[inference] Created molecule_chembl_id column")
    
    # Deduplicate by molecule_chembl_id (match HTS_3D.py data loading)
    # If activity data (value_nM) exists, keep most potent (lowest value_nM); otherwise keep first occurrence
    original_len = len(df)
    if "value_nM" in df.columns:
        # Check if value_nM has valid numeric data
        df["value_nM"] = pd.to_numeric(df["value_nM"], errors="coerce")
        if df["value_nM"].notna().any():
            # Sort by value_nM (ascending, so most potent = lowest value comes first)
            df = df.sort_values("value_nM", na_position="last")
            df = df.groupby("molecule_chembl_id", as_index=False).first()
            print(f"[inference] Deduplicated {original_len - len(df)} duplicate molecule IDs (kept most potent)")
        else:
            # No valid activity data, keep first occurrence
            df = df.groupby("molecule_chembl_id", as_index=False).first()
            print(f"[inference] Deduplicated {original_len - len(df)} duplicate molecule IDs (kept first occurrence)")
    else:
        # No activity data column, keep first occurrence
        df = df.groupby("molecule_chembl_id", as_index=False).first()
        print(f"[inference] Deduplicated {original_len - len(df)} duplicate molecule IDs (kept first occurrence)")
    
    df = df.reset_index(drop=True)
    
    print(f"[inference] Processing {len(df)} unique molecules")
    
    if len(df) == 0:
        print("Error: No valid SMILES strings found in input file")
        sys.exit(1)
    
    # Get SMILES list from deduplicated dataframe
    smiles_list = df["canonical_smiles"].tolist()
    
    # Resolve checkpoint path for legacy models
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path

    # Resolve protein PDB path
    protein_path = None
    if args.protein:
        protein_path = Path(args.protein)
        if not protein_path.is_absolute():
            protein_path = PROJECT_ROOT / protein_path

    # Load model and preprocessing (scaler, selector) from inference bundle
    model, scaler, selector = load_model(
        model_path, checkpoint_path=checkpoint_path, protein_path=protein_path
    )

    # Make predictions (batch_size: match HTS_3D inference_batch_size if not set)
    batch_size = args.batch_size if args.batch_size is not None else _default_inference_batch_size()
    predictions = predict_on_smiles(
        model, smiles_list, scaler=scaler, selector=selector, batch_size=batch_size
    )
    
    # Blend with QED for HTS.py-like drug-likeness impact (match HTS_3D training)
    if args.qed_blend > 0:
        qed_values = compute_qed_for_smiles_list(smiles_list)
        predictions = (1.0 - args.qed_blend) * predictions + args.qed_blend * qed_values
        predictions = np.clip(predictions, 0.0, 1.0)
        print(f"[inference] QED blend applied: weight={args.qed_blend:.2f}")
    
    # Create results dataframe (matching training script format)
    results_df = df.copy()
    results_df["hts3d_score"] = predictions
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"[inference] Predictions saved to: {output_path}")
    
    # Print summary (matching training output format)
    print(f"\n[inference] Summary:")
    print(f"Total Compounds: {len(results_df)}")
    print(f"Total Columns: {len(results_df.columns)}")
    score_cols = [col for col in results_df.columns if 'score' in col.lower()]
    print(f"Score Columns Found: {len(score_cols)}")
    if score_cols:
        print(f"Prediction Columns: {', '.join(score_cols)}")
    
    # Check for SMILES column
    if "canonical_smiles" in results_df.columns:
        print(f"\n[SMILES] Using 'canonical_smiles' for SMILES strings")
    
    threshold = args.threshold

    # Check for label column (if present)
    if "label" in results_df.columns:
        label_counts = results_df["label"].value_counts().sort_index()
        total_labels = label_counts.sum()
        if len(label_counts) == 2:
            active = label_counts.get(1, 0)
            inactive = label_counts.get(0, 0)
            print(f"\n[Labels] Ground truth labels found in 'label' column")
            print(f"\nLabel Distribution:")
            print(f"Active (1): {active} ({active/total_labels*100:.1f}%)")
            print(f"Inactive (0): {inactive} ({inactive/total_labels*100:.1f}%)")
        # Confusion matrix, precision, recall, F1 (use --threshold)
        y_true = results_df["label"].astype(int).values
        y_pred = (predictions >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        print(f"\n[Metrics] Confusion Matrix (threshold={threshold}):")
        print(f"                Predicted 0  Predicted 1")
        print(f"  Actual 0      {cm[0, 0]:>10}     {cm[0, 1]:>10}")
        print(f"  Actual 1      {cm[1, 0]:>10}     {cm[1, 1]:>10}")
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"\n  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")

    # Prediction score analysis (match HTS_3D summary format; threshold from --threshold)
    print(f"\n[Analysis] Prediction Score Analysis:")
    print(f"  Predicted hits (score >= {threshold}): {(predictions >= threshold).sum()} ({(predictions >= threshold).sum()/len(predictions)*100:.1f}%)")
    print(f"  Mean prediction score: {predictions.mean():.4f}")
    print(f"  Min score: {predictions.min():.4f}")
    print(f"  Max score: {predictions.max():.4f}")
    print(f"  Median score: {np.median(predictions):.4f}")


if __name__ == "__main__":
    main()
