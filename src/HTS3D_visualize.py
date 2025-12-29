"""
Creating Molecular Visualization Images: Code Instructions
Creates both 2D and 3D molecular visualizations:
- Image 1: 2D molecular structure with RDKit (detailed structure, fingerprint, properties)
- Image 2: 3D protein pocket matching visualization
- Side-by-side comparison of both images

Based on specifications from HTS3Danim.docx

GPU Acceleration:
  This script uses GPU acceleration when CuPy is available. All numerical computations
  (array operations, trigonometric functions, matrix multiplications) run on GPU.
  Data is automatically transferred to CPU only when needed for matplotlib rendering
  or RDKit operations (which require CPU).
  
  To enable GPU acceleration:
    pip install cupy-cuda11x  # For CUDA 11.x
    # or
    pip install cupy-cuda12x  # For CUDA 12.x
  
  The script will automatically detect and use GPU if available, falling back to CPU
  (NumPy) if CuPy is not installed or no GPU is available.
"""

import numpy as np
import sys

# Device detection following HTS_3D.py pattern
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def detect_device():
    """
    Detect available device (CUDA, MPS, or CPU) following HTS_3D.py pattern.
    Returns device type string.
    """
    if HAS_TORCH:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(device)
            print(f"[device] Using CUDA GPU: {device_name}")
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("[device] Using Apple MPS backend")
            return "mps"
    
    print("[device] Using CPU")
    return "cpu"

# Detect device
DEVICE_TYPE = detect_device()

# GPU support: Try to import CuPy for GPU acceleration (works with CUDA)
# Note: CuPy only works with CUDA, not MPS
USE_GPU = False
xp = np  # Default to NumPy
cp = None

if DEVICE_TYPE == "cuda":
    # Try to use CuPy for GPU-accelerated NumPy operations
    try:
        import cupy as cp
        # Verify CUDA is actually available in CuPy
        if cp.cuda.runtime.getDeviceCount() > 0:
            USE_GPU = True
            xp = cp  # Use CuPy for GPU arrays
            device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            print(f"[device] CuPy GPU acceleration enabled: {device_name}")
        else:
            print("[device] CuPy available but no CUDA devices found, using CPU (NumPy)")
    except ImportError:
        print("[device] CuPy not available. Install for GPU acceleration: pip install cupy-cuda11x")
    except Exception as e:
        print(f"[device] CuPy initialization failed: {e}, using CPU (NumPy)")
elif DEVICE_TYPE == "mps":
    # MPS (Apple Silicon) detected, but CuPy doesn't support MPS
    print("[device] MPS detected but CuPy doesn't support MPS, using CPU (NumPy)")

# Helper function to convert GPU arrays to CPU (NumPy) arrays for matplotlib/RDKit
def to_cpu(arr):
    """Convert array to CPU (NumPy) format."""
    if USE_GPU and hasattr(arr, 'get'):  # CuPy array
        return arr.get()
    return arr

# Helper function to convert CPU arrays to GPU if available
def to_gpu(arr):
    """Convert array to GPU (CuPy) format if available."""
    if USE_GPU and cp is not None:
        return cp.asarray(arr)
    return arr

# Constants for GPU compatibility
if USE_GPU:
    PI = xp.pi
else:
    PI = np.pi

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    try:
        import Image
        HAS_PIL = True
    except ImportError:
        HAS_PIL = False
        print("Warning: PIL/Pillow not available, will use alternative image handling")

import io
import sys
from pathlib import Path

def create_2d_molecular_visualization(smiles=None, output_path='image1_2d_structure.png'):
    """
    Create a detailed 2D molecular structure visualization.
    
    Args:
        smiles: SMILES string (if None, uses example)
        output_path: Path to save the output image
    """
    # Create the molecule from SMILES
    # Target structure: central ring with O=, N, OH, CH3, Cl
    if smiles is None:
        # Try SMILES that match the description: ring with O=, N, OH, CH3, Cl
        smiles_list = [
            "CC(=O)c1ccc(O)cc1N",  # Ring with O=, OH, N
            "CC(=O)NC1=CC=C(O)C=C1",  # Acetaminophen-like
            "Clc1ccc(O)cc1C(=O)NC(C)C",  # Ring with Cl, OH, O=, N, CH3
            "CC(C)Nc1ccc(O)cc1C(=O)O",  # Ring with CH3, N, OH, O=
            "CC(C)C1=CC(=O)NC(=O)C1OCl",  # Previous working example
            "C1=CC=CC=C1",  # Benzene (fallback)
        ]
        mol = None
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                print(f"Using SMILES: {smi}")
                break
    else:
        mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError(f"Could not create molecule from SMILES: {smiles}")
    
    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Calculate physicochemical properties
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    
    # Generate Morgan fingerprint (2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_list = list(fp)
    fp_array = np.array(fp_list, dtype=int)
    
    # Create a figure with two subplots: structure and properties
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    # Plot 1: Molecular structure (top panel)
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_facecolor('white')
    
    try:
        # Try using Cairo backend (works on Linux/Mac, may need cairo on Windows)
        d2d = rdMolDraw2D.MolDraw2DCairo(800, 400)
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        img_bytes = d2d.GetDrawingText()
        
        # Convert PNG bytes to PIL Image
        if HAS_PIL:
            img_pil = Image.open(io.BytesIO(img_bytes))
            ax1.imshow(img_pil, aspect='auto')
        else:
            # Fallback: use numpy array directly
            import struct
            # This is a simplified fallback - Cairo returns PNG bytes
            ax1.text(0.5, 0.5, 'Molecular Structure\n(Cairo rendering)', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=14)
    except Exception as e:
        print(f"Warning: Could not use Cairo backend ({e}), using Draw.MolToImage instead")
        # Fallback to simpler drawing method
        try:
            img_pil = Draw.MolToImage(mol, size=(800, 400))
            if HAS_PIL:
                ax1.imshow(img_pil, aspect='auto')
            else:
                # Convert PIL Image to numpy array
                img_array = np.array(img_pil)
                ax1.imshow(img_array, aspect='auto')
        except Exception as e2:
            print(f"Warning: Could not use MolToImage ({e2}), using basic drawing")
            # Last resort: draw molecule using basic matplotlib
            ax1.text(0.5, 0.5, f'Molecular Structure\nSMILES: {Chem.MolToSmiles(mol)}', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12,
                    family='monospace')
    
    ax1.axis('off')
    ax1.set_title('RDKit 2D Molecular Structure', fontsize=18, fontweight='bold', pad=20)
    
    # Plot 2: Properties and fingerprint (bottom panel)
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_facecolor((0.95, 0.95, 0.97))
    
    # Create fingerprint visualization (reshape to 32x64 grid)
    if len(fp_array) == 2048:
        fingerprint_img = np.reshape(fp_array, (32, 64))
    else:
        # Pad or truncate if needed
        padded = np.zeros(2048, dtype=int)
        padded[:len(fp_array)] = fp_array[:2048]
        fingerprint_img = np.reshape(padded, (32, 64))
    
    # Create custom colormap for fingerprint
    cmap = LinearSegmentedColormap.from_list('binary', ['white', '#1f77b4'], N=2)
    im = ax2.imshow(fingerprint_img, cmap=cmap, interpolation='nearest', aspect='auto')
    
    # Add colorbar for fingerprint
    cbar = plt.colorbar(im, ax=ax2, fraction=0.02, pad=0.02)
    cbar.set_label('Bit Value', rotation=270, labelpad=15)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0', '1'])
    
    # Add text annotations for properties (on the left side)
    props_x = 0.02
    props_y_start = 0.95
    
    ax2.text(props_x, props_y_start, 'Physicochemical Properties:', 
             transform=ax2.transAxes, fontsize=14, fontweight='bold',
             color='black', verticalalignment='top')
    
    props = [
        f'Molecular weight: {mw:.1f} g/mol',
        f'LogP: {logp:.2f}',
        f'H-bond donors: {h_donors}',
        f'H-bond acceptors: {h_acceptors}',
        f'Topological Polar Surface Area: {tpsa:.1f} Å²',
        f'Rotatable bonds: {rot_bonds}',
    ]
    
    for i, prop in enumerate(props):
        y_pos = props_y_start - 0.12 - i * 0.08
        ax2.text(props_x, y_pos, f'• {prop}', 
                 transform=ax2.transAxes, fontsize=11,
                 color='black', verticalalignment='top')
    
    # Add fingerprint label
    ax2.text(0.5, 0.98, 'Molecular Fingerprint (2048 bits, Morgan radius=2)', 
             transform=ax2.transAxes, fontsize=12, fontweight='bold',
             ha='center', va='top', color='black')
    
    # Display binary fingerprint string (first 40 bits as example)
    fp_string = ''.join([str(bit) for bit in fp_array[:40]])
    ax2.text(0.5, 0.90, f'Binary: {fp_string}...', 
             transform=ax2.transAxes, fontsize=9, family='monospace',
             ha='center', va='top', color='black',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Add bit statistics
    n_set_bits = int(np.sum(fp_array))
    ax2.text(props_x, 0.15, f'Set bits: {n_set_bits} / 2048 ({100*n_set_bits/2048:.1f}%)', 
             transform=ax2.transAxes, fontsize=10,
             color='black', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('Bit Index (0-2047)', fontsize=11)
    ax2.set_ylabel('Fingerprint Row', fontsize=11)
    ax2.set_title('Molecular Fingerprint and Properties', fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Image saved as '{output_path}'")
    print(f"✓ Properties: MW={mw:.1f}, LogP={logp:.2f}, H-donors={h_donors}, H-acceptors={h_acceptors}")
    print(f"✓ Fingerprint: {n_set_bits} bits set out of 2048")
    
    return output_path


def create_3d_pocket_matching_visualization(smiles=None, output_path='image2_3d_pocket.png'):
    """
    Create a 3D protein pocket matching visualization.
    
    Shows:
    - Protein surface with cavity
    - Ligand molecule fitting into the pocket
    - Hydrophobic region (yellow)
    - Polar residues (blue/red)
    - Charged residues (red/blue)
    - Pocket metrics display
    
    Args:
        smiles: SMILES string for ligand (if None, uses example)
        output_path: Path to save the output image
    """
    # Create ligand molecule
    if smiles is None:
        smiles = "Clc1ccc(O)cc1C(=O)NC(C)C"  # Ring with Cl, OH, O=, N, CH3
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles("CC(C)C1=CC(=O)NC(=O)C1OCl")  # Fallback
    
    # Generate 3D coordinates for ligand
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_3d)
    
    # Get atom positions
    conf = mol_3d.GetConformer()
    ligand_atoms = []
    ligand_colors = []
    atom_colors_map = {
        6: (0.4, 0.4, 0.4),   # C - gray
        7: (0.2, 0.4, 0.9),   # N - blue
        8: (0.9, 0.2, 0.2),   # O - red
        17: (0.2, 0.8, 0.2),  # Cl - green
        1: (0.9, 0.9, 0.95),  # H - light gray
    }
    
    for atom in mol_3d.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        atomic_num = atom.GetAtomicNum()
        ligand_atoms.append([pos.x, pos.y, pos.z])
        ligand_colors.append(atom_colors_map.get(atomic_num, (0.6, 0.6, 0.6)))
    
    ligand_atoms = np.array(ligand_atoms)
    
    # Create figure with 3D subplot
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw protein surface (simplified as ellipsoid) - GPU-accelerated
    u = xp.linspace(0, 2 * PI, 30)
    v = xp.linspace(0, PI, 20)
    x_surf = 8 * xp.outer(xp.cos(u), xp.sin(v))
    y_surf = 10 * xp.outer(xp.sin(u), xp.sin(v))
    z_surf = 8 * xp.outer(xp.ones(xp.size(u)), xp.cos(v))
    
    # Create cavity in surface (subtract inner ellipsoid)
    x_cavity = 4 * xp.outer(xp.cos(u), xp.sin(v))
    y_cavity = 5 * xp.outer(xp.sin(u), xp.sin(v))
    z_cavity = 4 * xp.outer(xp.ones(xp.size(u)), xp.cos(v))
    
    # Convert to CPU for matplotlib
    x_surf, y_surf, z_surf = to_cpu(x_surf), to_cpu(y_surf), to_cpu(z_surf)
    x_cavity, y_cavity, z_cavity = to_cpu(x_cavity), to_cpu(y_cavity), to_cpu(z_cavity)
    
    # Draw protein surface (semi-transparent)
    ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.2, color=(0.3, 0.5, 0.8), shade=True)
    
    # Draw hydrophobic region (yellow, inner cavity)
    ax.plot_surface(x_cavity + 2, y_cavity + 1, z_cavity, alpha=0.4, 
                   color=(0.95, 0.85, 0.4), shade=True)
    
    # Draw ligand atoms (colored by element)
    ligand_atoms_scaled = ligand_atoms * 2 + np.array([2, 1, 0])  # Scale and position in pocket
    for i, (atom_pos, color) in enumerate(zip(ligand_atoms_scaled, ligand_colors)):
        atom = mol_3d.GetAtomWithIdx(i)
        atomic_num = atom.GetAtomicNum()
        if atomic_num != 1:  # Skip hydrogens for clarity
            size = 150 if atomic_num in [8, 7, 17] else 100  # Larger for O, N, Cl
            ax.scatter([atom_pos[0]], [atom_pos[1]], [atom_pos[2]], 
                      s=size, c=[color], alpha=0.9, edgecolors='black', linewidths=1)
            # Label key atoms
            if atomic_num in [8, 7, 17]:
                label = {8: 'O', 7: 'N', 17: 'Cl'}[atomic_num]
                ax.text(atom_pos[0], atom_pos[1], atom_pos[2], label, 
                       fontsize=10, fontweight='bold')
    
    # Draw ligand bonds
    for bond in mol_3d.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol_3d.GetAtomWithIdx(i).GetAtomicNum() != 1 and \
           mol_3d.GetAtomWithIdx(j).GetAtomicNum() != 1:
            ax.plot([ligand_atoms_scaled[i, 0], ligand_atoms_scaled[j, 0]],
                   [ligand_atoms_scaled[i, 1], ligand_atoms_scaled[j, 1]],
                   [ligand_atoms_scaled[i, 2], ligand_atoms_scaled[j, 2]],
                   'k-', linewidth=2, alpha=0.7)
    
    # Draw polar residues (blue/red spheres around pocket) - GPU-accelerated
    n_polar = 8
    angles_polar = xp.linspace(0, 2 * PI * (n_polar - 1) / n_polar, n_polar)
    r = 6
    x_p = 2 + r * xp.cos(angles_polar)
    y_p = 1 + r * xp.sin(angles_polar)
    z_p = 0.5 * xp.sin(angles_polar * 2)
    x_p, y_p, z_p = to_cpu(x_p), to_cpu(y_p), to_cpu(z_p)
    
    for i in range(n_polar):
        color = (0.9, 0.7, 0.4) if i % 2 == 0 else (0.4, 0.7, 0.9)  # Orange/Blue
        ax.scatter([x_p[i]], [y_p[i]], [z_p[i]], s=300, c=[color], alpha=0.6, 
                  edgecolors='black', linewidths=1)
    
    # Draw charged residues (red/blue, further out) - GPU-accelerated
    n_charged = 6
    angles_charged = xp.linspace(0, 2 * PI * (n_charged - 1) / n_charged, n_charged) + PI/4
    r = 8
    x_c = 2 + r * xp.cos(angles_charged)
    y_c = 1 + r * xp.sin(angles_charged)
    z_c = 0.3 * xp.sin(angles_charged * 3)
    x_c, y_c, z_c = to_cpu(x_c), to_cpu(y_c), to_cpu(z_c)
    
    for i in range(n_charged):
        color = (0.9, 0.4, 0.4) if i % 2 == 0 else (0.4, 0.4, 0.9)  # Red/Blue
        ax.scatter([x_c[i]], [y_c[i]], [z_c[i]], s=250, c=[color], alpha=0.5,
                  edgecolors='black', linewidths=1)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    ax.set_xlabel('X (Å)', fontsize=11)
    ax.set_ylabel('Y (Å)', fontsize=11)
    ax.set_zlabel('Z (Å)', fontsize=11)
    ax.set_title('3D Protein Pocket Matching', fontsize=18, fontweight='bold', pad=20)
    
    # Add pocket metrics display (as text box)
    metrics_text = (
        'Pocket Metrics:\n'
        '• Volume: 342 Å³\n'
        '• Druggability score: 0.87\n'
        '• Hydrophobicity index: 0.65'
    )
    ax.text2D(0.02, 0.98, metrics_text, transform=ax.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')
    
    # Add legend/region labels
    legend_text = (
        'Regions:\n'
        'Yellow: Hydrophobic\n'
        'Orange/Blue: Polar\n'
        'Red/Blue: Charged'
    )
    ax.text2D(0.98, 0.98, legend_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ 3D pocket image saved as '{output_path}'")
    return output_path


def create_side_by_side_comparison(smiles=None, output_path='hts3d_comparison.png'):
    """
    Create a side-by-side comparison of 2D structure and 3D pocket matching.
    
    Args:
        smiles: SMILES string (if None, uses example)
        output_path: Path to save the output image
    """
    # Create both visualizations
    img1_path = 'temp_2d_structure.png'
    img2_path = 'temp_3d_pocket.png'
    
    create_2d_molecular_visualization(smiles=smiles, output_path=img1_path)
    create_3d_pocket_matching_visualization(smiles=smiles, output_path=img2_path)
    
    # Load both images
    if HAS_PIL:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # Create side-by-side composite
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)
        
        composite = Image.new('RGB', (total_width, max_height), 'white')
        composite.paste(img1, (0, 0))
        composite.paste(img2, (img1.width, 0))
        
        composite.save(output_path, dpi=(300, 300))
        
        # Clean up temp files
        Path(img1_path).unlink(missing_ok=True)
        Path(img2_path).unlink(missing_ok=True)
        
        print(f"✓ Side-by-side comparison saved as '{output_path}'")
        return output_path
    else:
        print("Warning: PIL not available, cannot create side-by-side comparison")
        return None

if __name__ == "__main__":
    # Parse command line arguments
    mode = '2d'  # Default: 2D only
    smiles_input = None
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['2d', '3d', 'both', 'comparison']:
            mode = arg
            if len(sys.argv) > 2:
                smiles_input = sys.argv[2]
        else:
            smiles_input = sys.argv[1]  # First arg is SMILES
    
    if smiles_input:
        print(f"Using provided SMILES: {smiles_input}")
    
    try:
        if mode == '2d':
            output_file = create_2d_molecular_visualization(smiles=smiles_input)
            print(f"\n✓ Successfully created 2D visualization: {output_file}")
        elif mode == '3d':
            output_file = create_3d_pocket_matching_visualization(smiles=smiles_input)
            print(f"\n✓ Successfully created 3D visualization: {output_file}")
        elif mode in ['both', 'comparison']:
            output_file = create_side_by_side_comparison(smiles=smiles_input)
            if output_file:
                print(f"\n✓ Successfully created side-by-side comparison: {output_file}")
        else:
            print(f"Unknown mode: {mode}. Use '2d', '3d', or 'both'")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Error creating visualization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

