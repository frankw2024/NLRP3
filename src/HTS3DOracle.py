import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

# Set page config
st.set_page_config(page_title="HTS-3D Results Evaluator", layout="wide")

# Turn off RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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
    st.title("🔬 HTS-3D Results Evaluator")
    
    st.markdown("""
    ## Molecular Prediction Results Analysis
    This application evaluates and analyzes prediction results from HTS-3D and HTS models.
    Upload a CSV file with prediction scores to visualize distributions, calculate metrics, and compare different models.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Advanced options
    with st.sidebar.expander("Advanced Options", expanded=False):
        confidence_threshold = st.slider("Hit prediction threshold", 0.0, 1.0, 0.5, 0.01, key="threshold")
        show_molecular_properties = st.checkbox("Show molecular properties analysis", value=True, key="mol_properties")
        show_interactive_plots = st.checkbox("Use interactive visualizations", value=True, key="interactive_plots")
        st.checkbox("Debug mode", value=False, key="debug")
    
    # File uploader - main input method (similar to HTSOracle.py)
    uploaded_file = st.file_uploader(
        "📁 Upload a CSV file with prediction results", 
        type=["csv"],
        help="The file should contain columns with prediction scores (e.g., 'hts3d_score', 'Prediction_Score') and optionally ground truth labels ('label')"
    )
    
    # Load data
    df = None
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Read the file
            df = pd.read_csv(tmp_path)
            st.success(f"✅ Successfully loaded {len(df)} rows from uploaded file")
            
            # Clean up temp file immediately after reading (we have the DataFrame now)
            try:
                os.unlink(tmp_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"Error loading uploaded file: {str(e)}")
            if st.session_state.debug:
                st.exception(e)
    else:
        st.info("👆 Please upload a CSV file with prediction results to begin analysis.")
    
    if df is not None and len(df) > 0:
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
        
        # Identify label column
        label_column = None
        possible_label_cols = ['label', 'Label', 'LABEL', 'true_label', 'ground_truth', 'activity']
        for col in possible_label_cols:
            if col in df.columns:
                label_column = col
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
                ax.hist(df[score_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(confidence_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {confidence_threshold}')
                ax.set_xlabel('Prediction Score')
                ax.set_ylabel('Count')
                ax.set_title(f'Distribution of {score_col}')
                ax.legend()
                st.pyplot(fig)
                
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
                
                # Plot 4: Score vs QED scatter
                if len(score_columns) > 0:
                    axes[1, 1].scatter(analysis_df['QED'], analysis_df[score_col], alpha=0.5)
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

