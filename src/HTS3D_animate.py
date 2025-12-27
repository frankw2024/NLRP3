"""
HTS-3D 35s explainer video (matplotlib) — frame-by-frame generator

Creates: hts3d_explainer.mp4 (35 seconds @ 10 fps by default)
Scenes (5s each):
  1) Inputs (SMILES -> molecule; protein silhouette)
  2) Ligand encoding (ChemBERTa embeddings + RDKit 3D)
  3) Pocket detection (protein mesh + highlighted pocket)
  4) Cross-attention (ligand nodes -> pocket residues with pulsing beams)
  5) 2D to 3D Transition (RDKit 2D structure -> 3D conformer -> pocket matching)
  6) Ensemble prediction (3 models -> ensemble score)
  7) Output ranking (bars sort; top hits highlight)

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

Install deps:
  pip install matplotlib numpy imageio imageio-ffmpeg
  For GPU acceleration: pip install cupy-cuda11x (or cupy-cuda12x depending on CUDA version)

Run:
  python hts3d_explainer.py
"""

import math
import numpy as np
import warnings
import os
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

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
from docx import Document
import re

# Suppress all warnings including RDKit deprecation warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Redirect stderr to suppress RDKit deprecation warnings
class SuppressRDKitWarnings:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.buffer = ''  # Buffer for partial writes
    
    def write(self, text):
        # Buffer text in case warnings are written in chunks
        self.buffer += text
        
        # Check if buffer contains complete lines
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            # Keep the last incomplete line in buffer
            self.buffer = lines[-1]
            # Process complete lines
            for line in lines[:-1]:
                # Suppress RDKit deprecation warnings about MorganGenerator
                if 'DEPRECATION WARNING' in line and 'MorganGenerator' in line:
                    continue  # Skip this line
                if 'please use MorganGenerator' in line:
                    continue  # Skip this line
                self.original_stderr.write(line + '\n')
        # If no newline, just buffer it (will be written on next write or flush)
    
    def flush(self):
        # Write any remaining buffered content (if it's not a warning)
        if self.buffer:
            if 'DEPRECATION WARNING' not in self.buffer and 'MorganGenerator' not in self.buffer:
                self.original_stderr.write(self.buffer)
            self.buffer = ''
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        # Forward any other attributes to the original stderr
        return getattr(self.original_stderr, name)

# Install the stderr filter BEFORE importing RDKit
_original_stderr = sys.stderr
sys.stderr = SuppressRDKitWarnings(_original_stderr)

# RDKit imports for real molecular data
try:
    # Suppress RDKit warnings before import
    import logging
    logging.getLogger('rdkit').setLevel(logging.ERROR)
    
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors
    from rdkit import RDLogger
    
    # Disable RDKit logging completely - this is the most effective way
    RDLogger.DisableLog('rdApp.*')
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not available, using simplified molecular representations")
except Exception as e:
    HAS_RDKIT = False
    print(f"Warning: RDKit not available: {e}")


# -----------------------------
# Config
# -----------------------------
FPS = 10
DURATION_S = 70  # Scene 3 (2D to 3D) is 18s, Scene 4 (Cross-Attention) is 7s, Scene 5 is 5s
W, H = 1280, 720
DPI = 100
FRAMES = 700  # Scenes 0-2: 100 frames each, Scene 3: 180 frames, Scene 4: 70 frames, Scene 5: 50 frames, Scene 6: 100 frames

SCENE_LEN_S = 10  # Doubled from 5 to match 2x slower animation
SCENE_FRAMES = FPS * SCENE_LEN_S  # 100 frames/scene (doubled from 50)

OUT_PATH = "hts3d_explainer.mp4"
SUBTITLE_PATH = "src/codeGen/aniVsubtitle_generated.docx"
#SUBTITLE_PATH = "src/codeGen/aniVsubtitle.docx"

BG = (0.05, 0.06, 0.08)  # dark background
FG = (0.92, 0.94, 0.97)  # near-white text

# Default SMILES for visualization
DEFAULT_SMILES = "CC(=O)c1ccc(O)cc1N"  # Ring with O=, OH, N


# -----------------------------
# Helpers
# -----------------------------
def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(t):
    """Smoothstep function with GPU support."""
    t_arr = to_gpu(xp.asarray(t)) if USE_GPU else xp.asarray(t)
    t_clipped = xp.clip(t_arr, 0, 1)
    result = t_clipped * t_clipped * (3 - 2 * t_clipped)
    return to_cpu(result) if USE_GPU and not isinstance(t, (list, tuple)) else float(result)


def scene_of_frame(f):
    """Determine which scene a frame belongs to."""
    # Scene 0-2: 100 frames each (0-299)
    # Scene 3 (Cross-Attention): 100 frames (300-399) - 10 seconds
    # Scene 4 (Ensemble): 50 frames (400-449) - 5 seconds
    # Scene 5 (Ranked): 100 frames (450-549) - 10 seconds
    # Scene 6 (2D to 3D): 150 frames (550-699) - 15 seconds
    if f < 300:
        return f // SCENE_FRAMES  # Scenes 0-2
    elif f < 400:  # Scene 3 (Cross-Attention): 100 frames (300-399)
        return 3
    elif f < 450:  # Scene 4 (Ensemble): 50 frames (400-449)
        return 4
    elif f < 550:  # Scene 5 (Ranked): 100 frames (450-549)
        return 5
    else:  # Scene 6 (2D to 3D): 150 frames (550-699)
        return 6


def local_t(f):
    """0..1 within scene."""
    s = scene_of_frame(f)
    if s < 3:
        # Scenes 0-2: normal duration (100 frames)
        return (f % SCENE_FRAMES) / (SCENE_FRAMES - 1)
    elif s == 3:
        # Scene 3 (Cross-Attention): 100 frames (300-399)
        scene3_start = 300
        scene3_frames = 100
        local_frame = f - scene3_start
        return local_frame / (scene3_frames - 1) if scene3_frames > 1 else 0
    elif s == 4:
        # Scene 4 (Ensemble): 50 frames (400-449)
        scene4_start = 400
        scene4_frames = 50
        local_frame = f - scene4_start
        return local_frame / (scene4_frames - 1) if scene4_frames > 1 else 0
    elif s == 5:
        # Scene 5 (Ranked): 100 frames (450-549)
        scene5_start = 450
        scene5_frames = 100
        local_frame = f - scene5_start
        return local_frame / (scene5_frames - 1) if scene5_frames > 1 else 0
    else:
        # Scene 6 (2D to 3D): 150 frames (550-699)
        scene6_start = 550
        scene6_frames = 150
        local_frame = f - scene6_start
        return local_frame / (scene6_frames - 1) if scene6_frames > 1 else 0


def setup_ax():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def draw_title(ax, title, subtitle=None, title_y=0.94, subtitle_y=0.875):
    # Title moved slightly higher (0.92 -> 0.94) for better separation from subtitle
    ax.text(0.05, title_y, title, color=FG, fontsize=28, fontweight="bold", va="top")
    if subtitle:
        ax.text(0.05, subtitle_y, subtitle, color=(0.75, 0.78, 0.85), fontsize=14, va="top")


def parse_srt_from_docx(docx_path):
    """Parse SRT-format subtitles from a Word document."""
    try:
        doc = Document(docx_path)
        subtitles = []
        current_sub = None
        lines = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip subtitle numbers (just digits)
            if line.isdigit():
                i += 1
                continue
            
            # Check if it's a timestamp line (format: HH:MM:SS,mmm --> HH:MM:SS,mmm)
            timestamp_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', line)
            if timestamp_match:
                # Parse start and end times
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, timestamp_match.groups())
                start_time = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
                end_time = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
                
                # Collect text lines until next number or timestamp
                text_parts = []
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.isdigit() or re.match(r'\d{2}:\d{2}:\d{2}', next_line):
                        break
                    if next_line:
                        text_parts.append(next_line)
                    i += 1
                
                if text_parts:
                    subtitle_text = ' '.join(text_parts)
                    subtitles.append({'start': start_time, 'end': end_time, 'text': subtitle_text})
            else:
                i += 1
        
        return subtitles
    except Exception as e:
        print(f"Warning: Could not load subtitles from {docx_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_subtitle_at_time(subtitles, time_seconds):
    """Get subtitle text for a given time in seconds."""
    for sub in subtitles:
        if sub['start'] <= time_seconds < sub['end']:
            return sub['text']
    return None


def wrap_text(text, max_chars_per_line=80):
    """Wrap text into multiple lines that fit within the video width.
    
    Args:
        text: The text to wrap
        max_chars_per_line: Maximum characters per line (default 80 for fontsize 15)
    
    Returns:
        List of text lines
    """
    if not text:
        return []
    
    # First split by newlines to preserve explicit line breaks
    paragraphs = text.split('\n')
    all_lines = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        
        words = paragraph.split()
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            # Check if adding this word would exceed the limit
            if current_length + word_length + 1 > max_chars_per_line and current_line:
                # Start a new line
                all_lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                # Add word to current line
                current_line.append(word)
                current_length += word_length + (1 if current_line else 0)
        
        # Add the last line of this paragraph
        if current_line:
            all_lines.append(' '.join(current_line))
    
    return all_lines


def draw_subtitle(ax, text, y_pos=0.08, fontsize=15):
    """Draw subtitle text at the bottom of the frame with automatic line wrapping.
    
    Increased line spacing to prevent overlap between multiple lines.
    """
    if not text:
        return
    
    # Wrap text into multiple lines
    lines = wrap_text(text, max_chars_per_line=80)
    
    if not lines:
        return
    
    # Line height in normalized coordinates - increased to prevent overlap
    # Original was 0.035, now 0.05 for better spacing between lines
    line_height = 0.05
    
    # Calculate total height needed
    total_height = (len(lines) - 1) * line_height
    start_y = y_pos + total_height / 2
    
    # Draw each line
    for i, line in enumerate(lines):
        line_y = start_y - i * line_height
        
        # Draw text with background
        ax.text(0.5, line_y, line, color=FG, fontsize=fontsize, fontweight="normal",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=(0, 0, 0, 0.75), 
                         edgecolor=(0.5, 0.5, 0.5, 0.5), linewidth=1))


def draw_arrow(ax, x1, y1, x2, y2, alpha=0.8, lw=2.5):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=(0.75, 0.8, 0.95), alpha=alpha),
    )


def render_frame(fig):
    fig.canvas.draw()
    # Use buffer_rgba() for Agg backend, then convert RGBA to RGB
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    # Convert RGBA to RGB
    img = img[:, :, :3]
    return img


# -----------------------------
# Enhanced 3D Ball-and-Stick Rendering Functions
# -----------------------------
def project_3d_to_2d(x, y, z, view_angle_x=0.5, view_angle_y=0.3):
    """Project 3D coordinates to 2D with perspective (GPU-accelerated)."""
    # Convert to GPU arrays if available
    if USE_GPU and (hasattr(x, '__len__') or hasattr(y, '__len__') or hasattr(z, '__len__')):
        x = to_gpu(xp.asarray(x))
        y = to_gpu(xp.asarray(y))
        z = to_gpu(xp.asarray(z))
        scale = 1.0 / (1.0 + z * 0.3)  # Perspective scaling
        x_proj = x * xp.cos(view_angle_y) - z * xp.sin(view_angle_y)
        y_proj = y * xp.cos(view_angle_x) + z * xp.sin(view_angle_x) * xp.cos(view_angle_y)
        return to_cpu(x_proj * scale), to_cpu(y_proj * scale)
    else:
        # Scalar or small array case - use NumPy for simplicity
        scale = 1.0 / (1.0 + z * 0.3)
        x_proj = x * np.cos(view_angle_y) - z * np.sin(view_angle_y)
        y_proj = y * np.cos(view_angle_x) + z * np.sin(view_angle_x) * np.cos(view_angle_y)
        return x_proj * scale, y_proj * scale


def draw_ball_and_stick_3d(ax, cx, cy, atoms_3d, bonds, atom_colors=None, 
                           bond_color=(0.7, 0.7, 0.7), view_angle_x=0.5, view_angle_y=0.3,
                           atom_scale=1.0, bond_width=2.0, alpha=1.0):
    """Draw realistic 3D ball-and-stick molecular structure.
    
    Args:
        ax: matplotlib axes
        cx, cy: center position in 2D
        atoms_3d: list of [x, y, z] coordinates for atoms
        bonds: list of (i, j) tuples for bonds
        atom_colors: list of RGB tuples for each atom (default: gray)
        bond_color: RGB tuple for bonds
        view_angle_x, view_angle_y: viewing angles
        atom_scale: scaling factor for atom sizes
        bond_width: line width for bonds
        alpha: transparency
    """
    if atom_colors is None:
        atom_colors = [(0.6, 0.6, 0.6)] * len(atoms_3d)
    
    # Project all atoms to 2D
    atoms_2d = []
    z_depths = []
    for atom in atoms_3d:
        x_proj, y_proj = project_3d_to_2d(atom[0], atom[1], atom[2], view_angle_x, view_angle_y)
        atoms_2d.append([cx + x_proj, cy + y_proj])
        z_depths.append(atom[2])
    
    atoms_2d = np.array(atoms_2d)
    
    # Sort bonds by depth (draw back bonds first)
    bond_depths = []
    for i, j in bonds:
        avg_z = (z_depths[i] + z_depths[j]) / 2
        bond_depths.append((avg_z, i, j))
    bond_depths.sort()
    
    # Draw bonds (back to front)
    for avg_z, i, j in bond_depths:
        z_factor = (avg_z - min(z_depths)) / (max(z_depths) - min(z_depths) + 1e-6)
        bond_alpha = 0.4 + 0.6 * z_factor  # Back bonds more transparent
        ax.plot([atoms_2d[i, 0], atoms_2d[j, 0]], 
               [atoms_2d[i, 1], atoms_2d[j, 1]],
               color=bond_color, lw=bond_width, alpha=bond_alpha * alpha, zorder=int(z_factor * 100))
    
    # Draw atoms (back to front)
    atom_order = sorted(range(len(z_depths)), key=lambda i: z_depths[i])
    for idx in atom_order:
        z_factor = (z_depths[idx] - min(z_depths)) / (max(z_depths) - min(z_depths) + 1e-6)
        atom_alpha = 0.6 + 0.4 * z_factor
        size = (80 + 60 * z_factor) * atom_scale
        ax.scatter([atoms_2d[idx, 0]], [atoms_2d[idx, 1]], 
                  s=size, color=atom_colors[idx], alpha=atom_alpha * alpha,
                  edgecolors=(0.9, 0.9, 0.9), linewidths=1.5, zorder=int(z_factor * 100) + 50)


def draw_detailed_rdkit_2d_molecule(ax, cx, cy, scale=1.0, alpha=1.0):
    """Draw detailed RDKit 2D molecular structure with atoms and bonds."""
    s = scale * 0.015  # Scale factor for coordinates
    
    # Define atom positions (relative to center)
    atoms = {
        'O1': (0, 4*s),      # Top O
        'N1': (-6*s, 2*s),   # Left N
        'C1': (-4*s, 2*s),   # Left C
        'C2': (-2*s, 0),     # Center-left C
        'C3': (0, 0),        # Center C
        'C4': (2*s, 0),      # Center-right C
        'C5': (4*s, 2*s),    # Right C
        'C6': (-6*s, 0),     # Bottom-left C
        'C7': (-4*s, -2*s),  # Left-bottom C
        'C8': (-2*s, -4*s),  # Bottom-left C
        'C9': (0, -4*s),     # Bottom-center C
        'C10': (2*s, -2*s),  # Right-bottom C
        'C11': (4*s, 0),     # Right C
        'C12': (6*s, 2*s),   # Far-right C (CH3)
        'Cl1': (4*s, -4*s),  # Bottom Cl
        'O2': (6*s, 0),      # OH oxygen
        'H1': (8*s, 2*s),    # CH3 hydrogen
    }
    
    # Define bonds: (atom1, atom2, bond_type)
    # bond_type: 1=single, 2=double, 3=triple
    bonds = [
        ('O1', 'C3', 2),  # Double bond
        ('N1', 'C1', 1),
        ('C1', 'C2', 2),  # Aromatic (double)
        ('C2', 'C3', 1),
        ('C3', 'C4', 1),
        ('C4', 'C5', 2),  # Aromatic (double)
        ('C5', 'O2', 1),
        ('C6', 'C7', 1),
        ('C7', 'C8', 2),  # Aromatic (double)
        ('C8', 'C9', 1),
        ('C9', 'C10', 1),
        ('C10', 'C11', 2),  # Aromatic (double)
        ('C11', 'C12', 1),
        ('C12', 'H1', 1),
        ('C11', 'Cl1', 1),
        ('C2', 'C6', 1),
        ('C4', 'C10', 1),
    ]
    
    # Atom colors by element
    atom_colors = {
        'C': (0.4, 0.4, 0.4),   # Carbon - gray
        'N': (0.2, 0.4, 0.9),   # Nitrogen - blue
        'O': (0.9, 0.2, 0.2),   # Oxygen - red
        'Cl': (0.2, 0.8, 0.2),  # Chlorine - green
        'H': (0.9, 0.9, 0.95),  # Hydrogen - light gray
    }
    
    # Draw bonds first (so atoms appear on top)
    for bond in bonds:
        atom1, atom2, bond_type = bond
        if atom1 in atoms and atom2 in atoms:
            x1, y1 = atoms[atom1]
            x2, y2 = atoms[atom2]
            x1, y1 = cx + x1, cy + y1
            x2, y2 = cx + x2, cy + y2
            
            if bond_type == 1:
                # Single bond
                ax.plot([x1, x2], [y1, y2], color=(0.6, 0.95, 0.85), 
                       lw=2.5*scale, alpha=alpha, zorder=1)
            elif bond_type == 2:
                # Double bond - draw two parallel lines
                perp = np.array([-(y2 - y1), x2 - x1])
                perp = perp / np.linalg.norm(perp) * 0.008 * scale
                ax.plot([x1 + perp[0], x2 + perp[0]], 
                       [y1 + perp[1], y2 + perp[1]], 
                       color=(0.6, 0.95, 0.85), lw=2.0*scale, alpha=alpha, zorder=1)
                ax.plot([x1 - perp[0], x2 - perp[0]], 
                       [y1 - perp[1], y2 - perp[1]], 
                       color=(0.6, 0.95, 0.85), lw=2.0*scale, alpha=alpha, zorder=1)
    
    # Draw atoms
    for atom_name, (x, y) in atoms.items():
        element = atom_name[0] if atom_name[0] != 'C' or len(atom_name) == 2 else 'C'
        if element == 'H' and len(atom_name) > 1:
            element = 'H'
        color = atom_colors.get(element, (0.6, 0.6, 0.6))
        size = 100 * scale if element != 'H' else 60 * scale
        
        ax.scatter([cx + x], [cy + y], s=size, color=color, alpha=alpha,
                  edgecolors=(0.9, 0.9, 0.9), linewidths=1.5*scale, zorder=2)
        
        # Label for non-carbon atoms
        if element != 'C' and element != 'H':
            ax.text(cx + x, cy + y, element, ha="center", va="center",
                   fontsize=8*scale, color=(1, 1, 1), fontweight="bold", zorder=3)


# -----------------------------
# Scene 1: Inputs
# -----------------------------
def draw_molecule_2d(ax, cx, cy, r, morph, smiles_pattern="benzene"):
    """Draw a more realistic 2D molecule structure."""
    if smiles_pattern == "benzene":
        # Benzene ring with substituents
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        xs = cx + r * np.cos(angles)
        ys = cy + r * np.sin(angles)
        
        # Aromatic ring (double bonds)
        for k in range(6):
            ax.plot([xs[k], xs[(k + 1) % 6]], [ys[k], ys[(k + 1) % 6]],
                    color=(0.6, 0.95, 0.85), lw=2.5, alpha=0.15 + 0.85 * morph)
            # Double bond indicator
            if k % 2 == 0:
                mid_x, mid_y = (xs[k] + xs[(k + 1) % 6]) / 2, (ys[k] + ys[(k + 1) % 6]) / 2
                perp = np.array([-(ys[(k + 1) % 6] - ys[k]), xs[(k + 1) % 6] - xs[k]])
                perp = perp / np.linalg.norm(perp) * 0.015
                ax.plot([mid_x - perp[0], mid_x + perp[0]], 
                       [mid_y - perp[1], mid_y + perp[1]],
                       color=(0.6, 0.95, 0.85), lw=1.5, alpha=0.15 + 0.85 * morph)
        
        # Carbon atoms
        ax.scatter(xs, ys, s=120, color=(0.4, 0.4, 0.4), alpha=0.15 + 0.85 * morph, 
                  edgecolors=(0.6, 0.95, 0.85), linewidths=1.5)
        
        # Substituent (OH group)
        sub_x, sub_y = xs[0], ys[0]
        ax.plot([sub_x, sub_x + 0.06], [sub_y, sub_y + 0.04],
                color=(0.6, 0.95, 0.85), lw=2.0, alpha=0.15 + 0.85 * morph)
        ax.scatter([sub_x + 0.06], [sub_y + 0.04], s=80, color=(0.9, 0.9, 0.95), 
                  alpha=0.15 + 0.85 * morph, edgecolors=(0.6, 0.95, 0.85), linewidths=1)


def draw_double_helix(ax, px, py, rot, alpha_val=0.3, helix_length=0.25):
    """Draw a DNA-like double helix structure (GPU-accelerated)."""
    # Parameters for the double helix
    n_turns = 2.5
    n_points = 100
    helix_radius = 0.08
    
    # Generate helix parameters on GPU if available
    t_helix = xp.linspace(0, n_turns * 2 * PI, n_points)
    
    # First strand (right-handed) - compute on GPU
    x1 = px + helix_radius * xp.cos(t_helix + rot)
    y1 = py + helix_radius * xp.sin(t_helix + rot) + (t_helix - n_turns * xp.pi) * helix_length / (n_turns * 2 * xp.pi)
    
    # Second strand (left-handed, offset by pi)
    x2 = px + helix_radius * xp.cos(t_helix + rot + PI)
    y2 = py + helix_radius * xp.sin(t_helix + rot + PI) + (t_helix - n_turns * PI) * helix_length / (n_turns * 2 * PI)
    
    # Convert to CPU for matplotlib
    x1, y1, x2, y2, t_helix = to_cpu(x1), to_cpu(y1), to_cpu(x2), to_cpu(y2), to_cpu(t_helix)
    
    # Draw the two strands
    ax.plot(x1, y1, color=(0.5, 0.7, 0.95), alpha=alpha_val, lw=3.0, zorder=2)
    ax.plot(x2, y2, color=(0.5, 0.7, 0.95), alpha=alpha_val, lw=3.0, zorder=2)
    
    # Draw base pairs (connecting lines between strands)
    n_base_pairs = 15
    base_indices = np.linspace(0, n_points - 1, n_base_pairs, dtype=int)
    for idx in base_indices:
        if 0 <= idx < len(x1) and 0 <= idx < len(x2):
            # Vary alpha slightly for depth effect
            bp_alpha = alpha_val * (0.6 + 0.4 * np.sin(t_helix[idx] * 2))
            ax.plot([x1[idx], x2[idx]], [y1[idx], y2[idx]], 
                   color=(0.7, 0.85, 1.0), alpha=bp_alpha, lw=1.5, zorder=1)
    
    # Add some backbone highlights
    for i in range(0, len(x1), 10):
        if 0 <= i < len(x1):
            ax.scatter([x1[i]], [y1[i]], s=20, color=(0.6, 0.8, 0.98), 
                      alpha=alpha_val * 0.8, zorder=3)
        if 0 <= i < len(x2):
            ax.scatter([x2[i]], [y2[i]], s=20, color=(0.6, 0.8, 0.98), 
                      alpha=alpha_val * 0.8, zorder=3)


def draw_protein_structure(ax, px, py, rot, alpha_val=0.3):
    """Draw a more realistic protein structure with double helix (GPU-accelerated)."""
    # Draw double helix as the main structure
    draw_double_helix(ax, px, py, rot, alpha_val=alpha_val, helix_length=0.25)
    
    # Add some surrounding protein context (simplified) - compute on GPU
    # Draw a subtle background shape to suggest protein environment
    blob_t = xp.linspace(0, 2 * PI, 200)
    blob_r = 0.15 + 0.02 * xp.sin(3 * blob_t) + 0.015 * xp.cos(5 * blob_t)
    bx = px + blob_r * xp.cos(blob_t + rot * 0.05)
    by = py + blob_r * xp.sin(blob_t + rot * 0.05)
    
    # Convert to CPU for matplotlib
    bx, by = to_cpu(bx), to_cpu(by)
    ax.fill(bx, by, color=(0.25, 0.5, 0.85), alpha=alpha_val * 0.2, zorder=0)


def scene1(ax, t):
    draw_title(ax, "High Throughput Screening 3D (HTS-3D)", "Inputs: Ligands (SMILES) + Protein Structure")

    # SMILES text flowing in (more realistic examples)
    smiles = [
        "CC(=O)NC1=CC=C(O)C=C1",  # Acetaminophen
        "C1=CC=C(C=C1)C(C)CC",    # Simple aromatic
        "CC(C)CC1=CC=C(C=C1)O",   # Phenol derivative
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine-like
    ]
    x_base = lerp(-0.2, 0.10, smoothstep(t))
    for i, s in enumerate(smiles):
        y = 0.72 - i * 0.06
        # Use monospace font for SMILES
        ax.text(x_base, y, s, color=(0.8, 0.85, 0.95), fontsize=12, 
               family='monospace', alpha=0.9)

    # Morphing: SMILES -> molecule diagram (more realistic)
    morph = smoothstep((t - 0.35) / 0.65)
    cx, cy = 0.28, 0.55
    draw_molecule_2d(ax, cx, cy, 0.08, morph)

    ax.text(0.10, 0.80, "Ligands (SMILES)", color=(0.75, 0.78, 0.85), fontsize=12)

    # More realistic protein structure
    px, py = 0.75, 0.55
    rot = 2 * PI * t * 0.3  # Slower rotation
    if USE_GPU:
        rot = float(to_cpu(xp.asarray(rot)))  # Convert to Python float
    draw_protein_structure(ax, px, py, rot, alpha_val=0.4)

    ax.text(0.67, 0.80, "Protein Structure", color=(0.75, 0.78, 0.85), fontsize=12)

    # Connecting arrow
    draw_arrow(ax, 0.40, 0.55, 0.58, 0.55, alpha=0.5)


# -----------------------------
# Scene 2: Ligand Encoding (2D + 3D)
# -----------------------------
def draw_3d_conformer(ax, cx, cy, theta, phi=0.3):
    """Draw a realistic 3D ball-and-stick molecular conformer (GPU-accelerated)."""
    # Create a more complex 3D structure (aromatic ring with substituents)
    ring_radius = 0.06
    n_ring = 6
    
    # Ring atoms in 3D - compute on GPU if available
    angles = xp.linspace(0, 2 * PI * (n_ring - 1) / n_ring, n_ring)
    x_3d = ring_radius * xp.cos(angles)
    y_3d = ring_radius * xp.sin(angles)
    z_3d = 0.02 * xp.sin(angles * 2)  # Slight puckering
    
    # Stack into array and convert to CPU for processing
    ring_atoms = xp.stack([x_3d, y_3d, z_3d], axis=1)
    
    # Add substituent atoms
    sub_atoms = xp.array([
        [ring_radius * 1.3, 0, 0.03],  # Right substituent
        [0, ring_radius * 1.2, -0.02],  # Top substituent
    ])
    
    atoms_3d = xp.vstack([ring_atoms, sub_atoms])
    atom_colors = [(0.5, 0.5, 0.5)] * n_ring + [(0.9, 0.9, 0.95), (0.8, 0.9, 0.95)]
    
    # Rotate around Y and Z axes - compute on GPU
    rot_y = xp.array([[xp.cos(theta), 0, xp.sin(theta)],
                      [0, 1, 0],
                      [-xp.sin(theta), 0, xp.cos(theta)]])
    rot_z = xp.array([[xp.cos(phi), -xp.sin(phi), 0],
                      [xp.sin(phi), xp.cos(phi), 0],
                      [0, 0, 1]])
    atoms_3d = atoms_3d @ rot_y.T @ rot_z.T
    
    # Convert to CPU for drawing (convert to numpy array, then to list)
    atoms_3d_cpu = to_cpu(atoms_3d)
    if isinstance(atoms_3d_cpu, np.ndarray):
        atoms_3d = atoms_3d_cpu.tolist()
    else:
        atoms_3d = atoms_3d_cpu
    
    # Define bonds
    bonds = []
    # Ring bonds
    for i in range(n_ring):
        bonds.append((i, (i + 1) % n_ring))
    # Substituent bonds
    bonds.append((0, n_ring))  # First substituent
    bonds.append((2, n_ring + 1))  # Second substituent
    
    # Draw using ball-and-stick style
    draw_ball_and_stick_3d(ax, cx, cy, atoms_3d, bonds, 
                          atom_colors=atom_colors,
                          bond_color=(0.7, 0.8, 0.9),
                          view_angle_x=0.4, view_angle_y=0.3,
                          atom_scale=1.2, bond_width=2.5, alpha=0.9)


def scene2(ax, t):
    # Move title and subtitle up to avoid overlap with middle diagram
    draw_title(ax, "Multi-Branch Encoding", "4 parallel branches: ChemBERTa, Ligand 3D, Protein Pocket, RDKit 2D", 
               title_y=0.97, subtitle_y=0.91)

    # Input molecule on left
    cx, cy = 0.12, 0.50
    draw_molecule_2d(ax, cx, cy, 0.05, 1.0)
    ax.text(0.12, 0.35, "SMILES", color=(0.75, 0.78, 0.85), fontsize=11, ha="center")

    # Four branches arranged in a grid - slightly larger spacing
    # Bottom row moved higher (0.30 -> 0.40) to make room for captions
    # Increased horizontal and vertical spacing slightly (0.36->0.37, 0.59->0.58, 0.68->0.69, 0.42->0.41)
    branch_positions = [
        (0.37, 0.69, "ChemBERTa\n(2D semantics)", "embedding"),
        (0.58, 0.69, "Ligand 3D\n(Conformers)", "3d"),
        (0.37, 0.41, "RDKit 2D\n(Descriptors)", "rdkit"),  # Swapped: RDKit 2D to left, moved higher
        (0.58, 0.41, "Protein Pocket\n(residues)", "pocket"),  # Swapped: Protein Pocket to right, moved higher
    ]
    
    # Draw arrows from input to branches
    for bx, by, _, _ in branch_positions:
        draw_arrow(ax, cx + 0.08, cy, bx - 0.08, by, alpha=0.6, lw=2.0)
    
    # Branch 1: ChemBERTa embeddings - slightly larger
    bx, by, label, branch_type = branch_positions[0]
    ax.text(bx, by + 0.11, label, color=FG, fontsize=11.5, fontweight="bold", ha="center")  # Slightly increased spacing and font
    bars_x0, bars_y0 = bx - 0.095, by - 0.075  # Slightly larger size
    n_bars = 24
    for i in range(n_bars):
        pattern = 0.3 + 0.4 * math.sin(2 * math.pi * (i / n_bars) + 4 * t)
        h = 0.027 + 0.11 * pattern  # Slightly increased height
        color_val = 0.7 + 0.2 * pattern
        ax.add_patch(plt.Rectangle((bars_x0 + i * 0.0075, bars_y0), 0.0055, h,  # Slightly larger sizes
                                   color=(0.7 * color_val, 0.85 * color_val, 1.0), alpha=0.8))
    ax.add_patch(plt.Rectangle((bx - 0.105, by - 0.095), 0.21, 0.19, fill=False,  # Slightly larger box size
                               edgecolor=(0.5, 0.6, 0.8), lw=1.5, alpha=0.5))
    
    # Branch 2: Ligand 3D conformer (ball-and-stick) - slightly larger
    bx, by, label, branch_type = branch_positions[1]
    ax.text(bx, by + 0.11, label, color=FG, fontsize=11.5, fontweight="bold", ha="center")  # Slightly increased spacing and font
    theta = 2 * np.pi * t * 0.6
    phi = 0.2 + 0.15 * np.sin(2 * np.pi * t * 0.4)
    draw_3d_conformer(ax, bx, by, theta, phi)  # This function will need scale parameter if available
    ax.add_patch(plt.Rectangle((bx - 0.105, by - 0.095), 0.21, 0.19, fill=False,  # Slightly larger box size
                               edgecolor=(0.5, 0.6, 0.8), lw=1.5, alpha=0.5))
    
    # Branch 3: RDKit 2D descriptors - Detailed molecular structure with full panel (now on left) - slightly larger
    bx, by, label, branch_type = branch_positions[2]
    ax.text(bx, by + 0.11, label, color=FG, fontsize=11.5, fontweight="bold", ha="center")  # Slightly increased spacing and font
    
    # Draw frame/box for RDKit 2D panel - slightly larger size
    panel_x0, panel_y0 = bx - 0.105, by - 0.19
    panel_w, panel_h = 0.21, 0.29
    ax.add_patch(plt.Rectangle((panel_x0, panel_y0), panel_w, panel_h, 
                               facecolor=(0.12, 0.15, 0.22), alpha=0.7,
                               edgecolor=(0.5, 0.6, 0.8), lw=2, zorder=0))
    
    # Draw detailed molecular structure (top of panel)
    mol_y = by + 0.02
    draw_detailed_rdkit_2d_molecule(ax, bx, mol_y, scale=0.75, alpha=0.95)  # Slightly increased from 0.7 to 0.75
    
    # Draw molecular fingerprint visualization (middle of panel)
    fp_y = by - 0.08
    ax.text(bx - 0.10, fp_y + 0.015, "Fingerprint (2048 bits):", 
            color=(0.75, 0.78, 0.85), fontsize=8, ha="left", fontweight="bold")
    
    # Generate real fingerprint if RDKit is available
    if HAS_RDKIT:
        try:
            mol = Chem.MolFromSmiles(DEFAULT_SMILES)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fp_list = list(fp)
                fp_bits = ''.join([str(bit) for bit in fp_list[:40]])
            else:
                fp_bits = '1010110010110101011101010010110101011101'
        except:
            fp_bits = '1010110010110101011101010010110101011101'
    else:
        # Fallback to example pattern
        fp_bits = '1010110010110101011101010010110101011101'
    
    # Scrolling effect
    scroll_offset = int(t * 5) % 20
    display_bits = fp_bits[scroll_offset:scroll_offset+25] + "..."
    ax.text(bx - 0.10, fp_y, display_bits, color=(0.6, 0.9, 0.8), 
            fontsize=6, family='monospace', ha="left", alpha=0.9)
    
    # Draw fingerprint grid visualization (small)
    if HAS_RDKIT:
        try:
            mol = Chem.MolFromSmiles(DEFAULT_SMILES)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fp_array = np.array(list(fp), dtype=int)
                # Reshape to small grid (8x16 = 128 bits shown)
                fp_grid = np.reshape(fp_array[:128], (8, 16))
                # Draw mini heatmap
                for i in range(8):
                    for j in range(16):
                        bit_val = fp_grid[i, j]
                        color = (0.4, 0.7, 0.9) if bit_val else (0.15, 0.15, 0.2)
                        x_pos = bx - 0.10 + j * 0.012
                        y_pos = fp_y - 0.03 - i * 0.003
                        ax.add_patch(plt.Rectangle((x_pos, y_pos), 0.010, 0.0025,
                                                  facecolor=color, edgecolor='none', alpha=0.8))
        except:
            pass
    
    # Draw physicochemical properties panel (bottom of panel)
    props_y = fp_y - 0.05
    ax.text(bx - 0.10, props_y + 0.015, "Physicochemical properties:", 
            color=(0.75, 0.78, 0.85), fontsize=8, ha="left", fontweight="bold")
    
    # Calculate real properties if RDKit is available
    if HAS_RDKIT:
        try:
            mol = Chem.MolFromSmiles(DEFAULT_SMILES)
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                h_donors = Descriptors.NumHDonors(mol)
                h_acceptors = Descriptors.NumHAcceptors(mol)
                props = [
                    ("Molecular weight:", f"{mw:.1f}"),
                    ("LogP:", f"{logp:.1f}"),
                    ("H-bond donors:", str(h_donors)),
                    ("H-bond acceptors:", str(h_acceptors))
                ]
            else:
                props = [
                    ("Molecular weight:", "342.4"),
                    ("LogP:", "2.7"),
                    ("H-bond donors:", "2"),
                    ("H-bond acceptors:", "5")
                ]
        except:
            props = [
                ("Molecular weight:", "342.4"),
                ("LogP:", "2.7"),
                ("H-bond donors:", "2"),
                ("H-bond acceptors:", "5")
            ]
    else:
        props = [
            ("Molecular weight:", "342.4"),
            ("LogP:", "2.7"),
            ("H-bond donors:", "2"),
            ("H-bond acceptors:", "5")
        ]
    
    for i, (label, value) in enumerate(props):
        y_pos = props_y - i * 0.018
        ax.text(bx - 0.10, y_pos, f"• {label} {value}", 
                color=(0.75, 0.78, 0.85), fontsize=7, ha="left", alpha=0.9)
    
    # Branch 4: Protein Pocket (simplified 3D representation) (now on right) - slightly larger
    bx, by, label, branch_type = branch_positions[3]
    ax.text(bx, by + 0.11, label, color=FG, fontsize=11.5, fontweight="bold", ha="center")  # Slightly increased spacing and font
    # Draw pocket residues as 3D balls
    rng = np.random.default_rng(42)
    n_res = 8
    pocket_atoms_3d = []
    pocket_colors = []
    residue_types = ['hydrophobic', 'polar', 'charged', 'aromatic']
    for i in range(n_res):
        angle = 2 * np.pi * i / n_res
        r = 0.04 + 0.01 * rng.random()
        x_3d = r * np.cos(angle)
        y_3d = r * np.sin(angle)
        z_3d = 0.01 * rng.random() - 0.005
        pocket_atoms_3d.append([x_3d, y_3d, z_3d])
        res_type = residue_types[i % len(residue_types)]
        if res_type == 'hydrophobic':
            pocket_colors.append((0.4, 0.6, 0.9))
        elif res_type == 'polar':
            pocket_colors.append((0.9, 0.7, 0.4))
        elif res_type == 'charged':
            pocket_colors.append((0.9, 0.5, 0.5))
        else:
            pocket_colors.append((0.8, 0.6, 0.9))
    
    # Rotate pocket
    rot_angle = 2 * np.pi * t * 0.3
    rot_matrix = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0],
                          [np.sin(rot_angle), np.cos(rot_angle), 0],
                          [0, 0, 1]])
    pocket_atoms_3d = np.array(pocket_atoms_3d) @ rot_matrix.T
    
    # Draw pocket residues
    pocket_bonds = [(i, (i + 1) % n_res) for i in range(n_res)]
    draw_ball_and_stick_3d(ax, bx, by, pocket_atoms_3d.tolist(), pocket_bonds,
                           atom_colors=pocket_colors,
                           bond_color=(0.6, 0.7, 0.8),
                           view_angle_x=0.3, view_angle_y=0.2,
                           atom_scale=1.35, bond_width=1.35, alpha=0.8)  # Slightly increased scale
    ax.add_patch(plt.Rectangle((bx - 0.105, by - 0.095), 0.21, 0.19, fill=False,  # Slightly larger box size
                               edgecolor=(0.5, 0.6, 0.8), lw=1.5, alpha=0.5))


# -----------------------------
# Scene 3: Pocket Detection
# -----------------------------
def draw_residue(ax, x, y, res_type, size=25):
    """Draw a realistic amino acid residue."""
    # Color by residue type
    residue_colors = {
        'hydrophobic': (0.4, 0.6, 0.9),  # Blue
        'polar': (0.9, 0.7, 0.4),        # Orange
        'charged': (0.9, 0.5, 0.5),      # Red
        'aromatic': (0.8, 0.6, 0.9),     # Purple
    }
    color = residue_colors.get(res_type, (0.7, 0.7, 0.7))
    
    # Draw residue as a small shape
    ax.scatter([x], [y], s=size, color=color, alpha=0.8,
              edgecolors=(1.0, 1.0, 1.0), linewidths=1)


def draw_3d_protein_pocket_matching(ax, cx, cy, t, scale=1.0):
    """Draw detailed 3D protein pocket matching diagram with ligand inside (GPU-accelerated).
    
    Enhanced with realistic 3D depth, shadows, and multi-layer visualization.
    
    Shows:
    - Protein surface with cavity (multi-layer depth)
    - Ligand inside pocket with colored atoms
    - Hydrophobic region (yellow, enhanced depth)
    - Polar residues (blue/red) with 3D positioning
    - Charged residues (red/blue) with 3D positioning
    - Realistic shadows and depth gradients
    """
    # Rotate for 3D effect
    rot = 2 * PI * t * 0.15
    if USE_GPU:
        rot = float(to_cpu(xp.asarray(rot)))
    
    # Add ground shadow for depth perception
    shadow_y = cy - 0.15 * scale
    from matplotlib.patches import Ellipse
    shadow_ellipse = Ellipse((cx, shadow_y), 0.30 * scale, 0.10 * scale, 
                             facecolor=(0.05, 0.05, 0.08), alpha=0.4, zorder=0)
    ax.add_patch(shadow_ellipse)
    
    # Draw protein surface with multiple layers for 3D depth - compute on GPU
    surface_angles = xp.linspace(0, 2 * PI, 60)  # More points for smoother curve
    
    # Outer shell (back layer, darker)
    surface_r_outer = 0.15 * scale
    surface_x_outer = cx + surface_r_outer * xp.cos(surface_angles + rot)
    surface_y_outer = cy + surface_r_outer * xp.sin(surface_angles + rot) * 0.65  # More elliptical
    surface_x_outer, surface_y_outer = to_cpu(surface_x_outer), to_cpu(surface_y_outer)
    ax.fill(surface_x_outer, surface_y_outer, color=(0.2, 0.35, 0.65), alpha=0.25, zorder=1)
    ax.plot(surface_x_outer, surface_y_outer, color=(0.3, 0.5, 0.75), lw=2.5, alpha=0.4, zorder=2)
    
    # Middle shell (intermediate layer)
    surface_r_mid = 0.13 * scale
    surface_x_mid = cx + surface_r_mid * xp.cos(surface_angles + rot)
    surface_y_mid = cy + surface_r_mid * xp.sin(surface_angles + rot) * 0.68
    surface_x_mid, surface_y_mid = to_cpu(surface_x_mid), to_cpu(surface_y_mid)
    ax.fill(surface_x_mid, surface_y_mid, color=(0.25, 0.45, 0.75), alpha=0.3, zorder=3)
    ax.plot(surface_x_mid, surface_y_mid, color=(0.35, 0.55, 0.85), lw=2.0, alpha=0.5, zorder=4)
    
    # Inner shell (front layer, brighter)
    surface_r = 0.12 * scale
    surface_x = cx + surface_r * xp.cos(surface_angles + rot)
    surface_y = cy + surface_r * xp.sin(surface_angles + rot) * 0.7
    surface_x, surface_y = to_cpu(surface_x), to_cpu(surface_y)
    ax.fill(surface_x, surface_y, color=(0.3, 0.5, 0.8), alpha=0.35, zorder=5)
    ax.plot(surface_x, surface_y, color=(0.4, 0.6, 0.9), lw=2.5, alpha=0.6, zorder=6)
    
    # Draw hydrophobic region (yellow, inner layer) with enhanced 3D depth
    hydro_angles = xp.linspace(0, 2 * PI, 40)
    
    # Outer hydrophobic ring (deeper, darker) - compute on GPU
    hydro_r_outer = 0.10 * scale
    hydro_x_outer = cx + hydro_r_outer * xp.cos(hydro_angles + rot)
    hydro_y_outer = cy + hydro_r_outer * xp.sin(hydro_angles + rot) * 0.7
    hydro_x_outer, hydro_y_outer = to_cpu(hydro_x_outer), to_cpu(hydro_y_outer)
    ax.fill(hydro_x_outer, hydro_y_outer, color=(0.9, 0.75, 0.3), alpha=0.35, zorder=7)
    ax.plot(hydro_x_outer, hydro_y_outer, color=(0.95, 0.7, 0.25), lw=2.0, alpha=0.5, zorder=8)
    
    # Inner hydrophobic core (closer, brighter)
    hydro_r = 0.08 * scale
    hydro_x = cx + hydro_r * xp.cos(hydro_angles + rot)
    hydro_y = cy + hydro_r * xp.sin(hydro_angles + rot) * 0.7
    hydro_x, hydro_y = to_cpu(hydro_x), to_cpu(hydro_y)
    ax.fill(hydro_x, hydro_y, color=(0.95, 0.85, 0.4), alpha=0.5, zorder=9)
    ax.plot(hydro_x, hydro_y, color=(0.95, 0.75, 0.3), lw=2.5, alpha=0.7, zorder=10)
    
    # Add depth gradient rings within hydrophobic region
    for i in range(3):
        grad_r = hydro_r + (hydro_r_outer - hydro_r) * (i + 1) / 4
        grad_alpha = 0.2 + 0.15 * (i / 2)
        grad_x = cx + grad_r * xp.cos(hydro_angles + rot)
        grad_y = cy + grad_r * xp.sin(hydro_angles + rot) * 0.7
        grad_x, grad_y = to_cpu(grad_x), to_cpu(grad_y)
        ax.plot(grad_x, grad_y, color=(0.95, 0.8, 0.35), lw=1.0, alpha=grad_alpha, zorder=8)
    
    # Draw ligand inside pocket (colored atoms) - use real RDKit 3D if available
    if HAS_RDKIT:
        try:
            mol = Chem.MolFromSmiles(DEFAULT_SMILES)
            if mol is not None:
                mol_3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_3d)
                conf = mol_3d.GetConformer()
                
                ligand_atoms_3d = []
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
                    # Scale and center coordinates (larger for 2x scale)
                    ligand_atoms_3d.append([pos.x * 0.015 * scale, pos.y * 0.015 * scale, pos.z * 0.015 * scale])
                    ligand_colors.append(atom_colors_map.get(atomic_num, (0.6, 0.6, 0.6)))
                
                ligand_bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) 
                               for bond in mol_3d.GetBonds()]
                
                # Enhanced rotation for better 3D visualization - compute on GPU
                lig_rot_y = 2 * PI * t * 0.4
                lig_rot_z = 2 * PI * t * 0.3
                
                # Rotate around Y and Z axes for full 3D rotation
                ligand_atoms_3d_gpu = to_gpu(xp.array(ligand_atoms_3d))
                rot_y = xp.array([[xp.cos(lig_rot_y), 0, xp.sin(lig_rot_y)],
                                  [0, 1, 0],
                                  [-xp.sin(lig_rot_y), 0, xp.cos(lig_rot_y)]])
                rot_z = xp.array([[xp.cos(lig_rot_z), -xp.sin(lig_rot_z), 0],
                                  [xp.sin(lig_rot_z), xp.cos(lig_rot_z), 0],
                                  [0, 0, 1]])
                ligand_atoms_3d = to_cpu(ligand_atoms_3d_gpu @ rot_y.T @ rot_z.T)
                
                # Draw ligand in pocket with enhanced 3D rendering
                draw_ball_and_stick_3d(ax, cx, cy, ligand_atoms_3d.tolist(), ligand_bonds,
                                       atom_colors=ligand_colors,
                                       bond_color=(0.6, 0.9, 0.85),
                                       view_angle_x=0.35, view_angle_y=0.25,  # Better viewing angle
                                       atom_scale=3.0 * scale, bond_width=3.0 * scale, alpha=0.95)
            else:
                # Fallback to simple representation with enhanced 3D
                ligand_atoms_3d = [
                    [0.0, 0.0, -0.02 * scale], [0.02 * scale, 0.0, 0.0], [-0.02 * scale, 0.0, 0.0],
                    [0.0, 0.02 * scale, 0.0], [0.0, -0.02 * scale, 0.0],
                ]
                ligand_bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
                ligand_colors = [(0.4, 0.4, 0.4), (0.9, 0.2, 0.2), (0.2, 0.4, 0.9),
                                (0.4, 0.4, 0.4), (0.2, 0.8, 0.2)]
                # Enhanced rotation for better 3D visualization
                lig_rot_y = 2 * np.pi * t * 0.4
                lig_rot_z = 2 * np.pi * t * 0.3
                rot_y = np.array([[np.cos(lig_rot_y), 0, np.sin(lig_rot_y)],
                                  [0, 1, 0],
                                  [-np.sin(lig_rot_y), 0, np.cos(lig_rot_y)]])
                rot_z = np.array([[np.cos(lig_rot_z), -np.sin(lig_rot_z), 0],
                                  [np.sin(lig_rot_z), np.cos(lig_rot_z), 0],
                                  [0, 0, 1]])
                ligand_atoms_3d = np.array(ligand_atoms_3d) @ rot_y.T @ rot_z.T
                draw_ball_and_stick_3d(ax, cx, cy, ligand_atoms_3d.tolist(), ligand_bonds,
                                       atom_colors=ligand_colors,
                                       bond_color=(0.6, 0.9, 0.85),
                                       view_angle_x=0.35, view_angle_y=0.25,
                                       atom_scale=3.0 * scale, bond_width=3.0 * scale, alpha=0.95)
        except:
            # Fallback to simple representation with enhanced 3D
            ligand_atoms_3d = [
                [0.0, 0.0, -0.02 * scale], [0.02 * scale, 0.0, 0.0], [-0.02 * scale, 0.0, 0.0],
                [0.0, 0.02 * scale, 0.0], [0.0, -0.02 * scale, 0.0],
            ]
            ligand_bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
            ligand_colors = [(0.4, 0.4, 0.4), (0.9, 0.2, 0.2), (0.2, 0.4, 0.9),
                            (0.4, 0.4, 0.4), (0.2, 0.8, 0.2)]
            # Enhanced rotation for better 3D visualization
            lig_rot_y = 2 * np.pi * t * 0.4
            lig_rot_z = 2 * np.pi * t * 0.3
            rot_y = np.array([[np.cos(lig_rot_y), 0, np.sin(lig_rot_y)],
                              [0, 1, 0],
                              [-np.sin(lig_rot_y), 0, np.cos(lig_rot_y)]])
            rot_z = np.array([[np.cos(lig_rot_z), -np.sin(lig_rot_z), 0],
                              [np.sin(lig_rot_z), np.cos(lig_rot_z), 0],
                              [0, 0, 1]])
            ligand_atoms_3d = np.array(ligand_atoms_3d) @ rot_y.T @ rot_z.T
            draw_ball_and_stick_3d(ax, cx, cy, ligand_atoms_3d.tolist(), ligand_bonds,
                                   atom_colors=ligand_colors,
                                   bond_color=(0.6, 0.9, 0.85),
                                   view_angle_x=0.35, view_angle_y=0.25,
                                   atom_scale=3.0 * scale, bond_width=3.0 * scale, alpha=0.95)
    else:
        # Fallback to simple representation with enhanced 3D
        ligand_atoms_3d = [
            [0.0, 0.0, -0.02 * scale], [0.02 * scale, 0.0, 0.0], [-0.02 * scale, 0.0, 0.0],
            [0.0, 0.02 * scale, 0.0], [0.0, -0.02 * scale, 0.0],
        ]
        ligand_bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
        ligand_colors = [(0.4, 0.4, 0.4), (0.9, 0.2, 0.2), (0.2, 0.4, 0.9),
                        (0.4, 0.4, 0.4), (0.2, 0.8, 0.2)]
        # Enhanced rotation for better 3D visualization - compute on GPU
        lig_rot_y = 2 * PI * t * 0.4
        lig_rot_z = 2 * PI * t * 0.3
        ligand_atoms_3d_gpu = to_gpu(xp.array(ligand_atoms_3d))
        rot_y = xp.array([[xp.cos(lig_rot_y), 0, xp.sin(lig_rot_y)],
                          [0, 1, 0],
                          [-xp.sin(lig_rot_y), 0, xp.cos(lig_rot_y)]])
        rot_z = xp.array([[xp.cos(lig_rot_z), -xp.sin(lig_rot_z), 0],
                          [xp.sin(lig_rot_z), xp.cos(lig_rot_z), 0],
                          [0, 0, 1]])
        ligand_atoms_3d = to_cpu(ligand_atoms_3d_gpu @ rot_y.T @ rot_z.T)
        draw_ball_and_stick_3d(ax, cx, cy, ligand_atoms_3d.tolist(), ligand_bonds,
                               atom_colors=ligand_colors,
                               bond_color=(0.6, 0.9, 0.85),
                               view_angle_x=0.35, view_angle_y=0.25,
                               atom_scale=3.0 * scale, bond_width=3.0 * scale, alpha=0.95)
    
    # Draw polar residues (blue/red) around ligand with 3D depth positioning
    n_polar = 8  # More residues for larger scale
    for i in range(n_polar):
        angle = 2 * np.pi * i / n_polar + rot
        r = 0.07 * scale
        # Add z-depth variation for 3D effect
        z_depth = 0.01 * scale * np.sin(angle * 2)
        x_3d = r * np.cos(angle)
        y_3d = r * np.sin(angle)
        z_3d = z_depth
        
        # Project to 2D with perspective
        x_proj, y_proj = project_3d_to_2d(x_3d, y_3d, z_3d, 0.3, 0.2)
        x_polar = cx + x_proj
        y_polar = cy + y_proj
        
        color = (0.9, 0.7, 0.4) if i % 2 == 0 else (0.4, 0.7, 0.9)  # Orange/Blue
        
        # Size based on depth (closer = larger)
        depth_factor = (z_depth + 0.02 * scale) / (0.04 * scale)
        size = (250 + 150 * depth_factor) * scale
        alpha = 0.6 + 0.3 * depth_factor
        
        ax.scatter([x_polar], [y_polar], s=size, color=color, alpha=alpha,
                  edgecolors=(1.0, 1.0, 1.0), linewidths=2.0, zorder=int(10 + depth_factor * 10))
    
    # Draw charged residues (red/blue) further out with 3D depth
    n_charged = 6  # More residues for larger scale
    for i in range(n_charged):
        angle = 2 * np.pi * i / n_charged + rot + np.pi/4
        r = 0.11 * scale
        # Add z-depth variation
        z_depth = 0.015 * scale * np.cos(angle * 1.5)
        x_3d = r * np.cos(angle)
        y_3d = r * np.sin(angle)
        z_3d = z_depth
        
        # Project to 2D with perspective
        x_proj, y_proj = project_3d_to_2d(x_3d, y_3d, z_3d, 0.3, 0.2)
        x_charged = cx + x_proj
        y_charged = cy + y_proj
        
        color = (0.9, 0.4, 0.4) if i % 2 == 0 else (0.4, 0.4, 0.9)  # Red/Blue
        
        # Size based on depth
        depth_factor = (z_depth + 0.03 * scale) / (0.06 * scale)
        size = (220 + 120 * depth_factor) * scale
        alpha = 0.5 + 0.3 * depth_factor
        
        ax.scatter([x_charged], [y_charged], s=size, color=color, alpha=alpha,
                  edgecolors=(1.0, 1.0, 1.0), linewidths=2.0, zorder=int(9 + depth_factor * 10))
    
    # Enhanced labels for regions with better positioning
    ax.text(cx, cy + 0.12 * scale, "HYDROPHOBIC\nREGION", 
            color=(0.95, 0.75, 0.3), fontsize=11 * scale, ha="center", 
            fontweight="bold", alpha=0.9, zorder=20)
    ax.text(cx - 0.10 * scale, cy, "POLAR\nRESIDUES", 
            color=(0.7, 0.8, 0.9), fontsize=10 * scale, ha="center", 
            fontweight="bold", alpha=0.8, zorder=20)
    ax.text(cx + 0.10 * scale, cy, "CHARGED\nRESIDUES", 
            color=(0.9, 0.5, 0.5), fontsize=10 * scale, ha="center", 
            fontweight="bold", alpha=0.8, zorder=20)


def scene3(ax, t):
    draw_title(ax, "3D Protein Pocket Matching", "Ligand binding in detected pocket with residue interactions")

    # Main 3D pocket matching diagram (center) - smaller and lower
    px, py = 0.50, 0.45  # Moved down from 0.50 to 0.45
    draw_3d_protein_pocket_matching(ax, px, py, t, scale=2.1)  # Reduced from 2.3 to 2.1
    
    # Pocket metrics display (right side) - moved further right to avoid overlap with 2x diagram
    metrics_x, metrics_y = 0.88, 0.50
    ax.text(metrics_x, metrics_y + 0.12, "Pocket Metrics", color=FG, fontsize=14, 
            fontweight="bold", ha="center")
    
    # Metrics box
    metrics_box = plt.Rectangle((metrics_x - 0.10, metrics_y - 0.10), 0.20, 0.20,
                                facecolor=(0.15, 0.20, 0.30), alpha=0.8,
                                edgecolor=(0.6, 0.8, 0.98), lw=2)
    ax.add_patch(metrics_box)
    
    # Metrics values (with slight animation)
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * t * 0.5)
    metrics = [
        ("Volume:", f"{342 + int(10 * pulse)} Å³"),
        ("Druggability:", f"{0.85 + 0.02 * pulse:.2f}"),
        ("Hydrophobicity:", f"{0.63 + 0.02 * pulse:.2f}"),
    ]
    
    for i, (label, value) in enumerate(metrics):
        y_pos = metrics_y + 0.05 - i * 0.06
        ax.text(metrics_x - 0.08, y_pos, label, color=(0.75, 0.78, 0.85), 
                fontsize=10, ha="left", va="center")
        ax.text(metrics_x + 0.08, y_pos, value, color=(0.6, 0.9, 0.8), 
                fontsize=10, ha="right", va="center", fontweight="bold")
    
    # Left side: Show pocket detection process - moved further left to avoid overlap with 2x diagram
    ax.text(0.08, 0.75, "Pocket Detection", color=FG, fontsize=12, fontweight="bold")
    ax.text(0.08, 0.70, "• 3D grid scan", color=(0.75, 0.78, 0.85), fontsize=10)
    ax.text(0.08, 0.65, "• Cavity identification", color=(0.75, 0.78, 0.85), fontsize=10)
    ax.text(0.08, 0.60, "• Residue mapping", color=(0.75, 0.78, 0.85), fontsize=10)
    ax.text(0.08, 0.55, "• residues", color=(0.75, 0.78, 0.85), fontsize=10)
    
    # Bottom: Interaction types (moved higher to avoid overlap with caption)
    ax.text(0.50, 0.25, "Ligand-protein interactions: H-bonds, hydrophobic, electrostatic", 
            color=(0.75, 0.78, 0.85), fontsize=11, ha="center")


# -----------------------------
# Scene 4: Cross-Attention
# -----------------------------
def scene4(ax, t):
    draw_title(ax, "Cross-Attention", "Ligand features attend to pocket residues")

    # More realistic ligand structure (left) - show actual molecular features
    ax.text(0.10, 0.80, "Ligand features", color=FG, fontsize=14, fontweight="bold")
    ligand_center = np.array([0.16, 0.47])
    
    # Draw ligand as a small molecule with distinct atoms/features
    ligand_atoms = np.array([
        [0.14, 0.62],  # Top atom
        [0.18, 0.52],  # Right atom
        [0.12, 0.42],  # Bottom-left
        [0.20, 0.38],  # Bottom-right
        [0.16, 0.30],  # Bottom center
    ])
    
    # Draw bonds
    bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)]
    for i, j in bonds:
        ax.plot([ligand_atoms[i, 0], ligand_atoms[j, 0]],
               [ligand_atoms[i, 1], ligand_atoms[j, 1]],
               color=(0.6, 0.95, 0.85), lw=2.5, alpha=0.7)
    
    # Draw atoms with different sizes/types
    atom_colors = [(0.6, 0.95, 0.85), (0.7, 0.98, 0.9), (0.5, 0.9, 0.8), 
                   (0.65, 0.97, 0.88), (0.55, 0.92, 0.82)]
    for i, (atom, color) in enumerate(zip(ligand_atoms, atom_colors)):
        ax.scatter([atom[0]], [atom[1]], s=100 + i*10, color=color, alpha=0.9,
                  edgecolors=(0.8, 1.0, 0.95), linewidths=1.5)

    # More realistic pocket residues (right) - with amino acid types
    ax.text(0.72, 0.80, "Pocket residues", color=FG, fontsize=14, fontweight="bold")
    residues = np.array([[0.78, 0.62], [0.84, 0.55], [0.76, 0.45], [0.86, 0.40], [0.80, 0.32]])
    residue_types = ['charged', 'polar', 'hydrophobic', 'aromatic', 'polar']
    
    for i, (res_pos, res_type) in enumerate(zip(residues, residue_types)):
        draw_residue(ax, res_pos[0], res_pos[1], res_type, size=140)

    # Attention beams (more realistic with varying intensities) - GPU-accelerated
    n_lig = len(ligand_atoms)
    n_res = len(residues)
    # Create meshgrid for vectorized computation on GPU
    i_indices = xp.arange(n_lig)[:, xp.newaxis]
    j_indices = xp.arange(n_res)[xp.newaxis, :]
    # Compute weights using vectorized operations on GPU
    base = 0.2 + 0.8 * (0.5 + 0.5 * xp.sin(2*PI*(t*1.2 + (i_indices*0.13 + j_indices*0.09))))
    preference = 0.3 * xp.sin((i_indices - j_indices) * 0.5)
    weights_gpu = xp.clip(base + preference, 0, 1)
    weights = to_cpu(weights_gpu)

    # Draw top-2 edges per ligand atom with gradient effect
    for i in range(len(ligand_atoms)):
        top_js = np.argsort(weights[i])[::-1][:2]
        for j in top_js:
            w = weights[i, j]
            alpha = 0.15 + 0.6 * w
            lw = 1.5 + 3.5 * w
            
            # Create gradient effect along the line
            x1, y1 = ligand_atoms[i, 0], ligand_atoms[i, 1]
            x2, y2 = residues[j, 0], residues[j, 1]
            
            # Draw main attention beam
            ax.plot([x1, x2], [y1, y2],
                    color=(0.85, 0.9, 1.0), alpha=alpha, lw=lw, zorder=1)
            
            # Add glow effect for strong attention
            if w > 0.7:
                ax.plot([x1, x2], [y1, y2],
                       color=(0.95, 0.95, 1.0), alpha=alpha*0.3, lw=lw*2, zorder=0)

    # Mid-panel label - made more visible
    ax.text(0.50, 0.22, "Stronger attention = brighter beam", color=(0.85, 0.90, 0.95), fontsize=13, ha="center", fontweight="bold", zorder=20)


# -----------------------------
# Scene 5: 2D to 3D Transition & Pocket Matching
# -----------------------------
def draw_2d_structure_detailed(ax, cx, cy, scale, alpha_val=1.0):
    """Draw a detailed 2D RDKit structure similar to image1_2d_structure.png."""
    # Draw a more complex 2D structure with multiple rings and substituents
    # Main aromatic ring (benzene-like)
    ring_radius = 0.06 * scale
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    ring_x = cx + ring_radius * np.cos(angles)
    ring_y = cy + ring_radius * np.sin(angles)
    
    # Draw aromatic ring with double bonds
    for k in range(6):
        ax.plot([ring_x[k], ring_x[(k + 1) % 6]], [ring_y[k], ring_y[(k + 1) % 6]],
                color=(0.6, 0.95, 0.85), lw=2.5 * scale, alpha=alpha_val)
        # Double bond indicators
        if k % 2 == 0:
            mid_x, mid_y = (ring_x[k] + ring_x[(k + 1) % 6]) / 2, (ring_y[k] + ring_y[(k + 1) % 6]) / 2
            perp = np.array([-(ring_y[(k + 1) % 6] - ring_y[k]), ring_x[(k + 1) % 6] - ring_x[k]])
            perp = perp / (np.linalg.norm(perp) + 1e-6) * 0.012 * scale
            ax.plot([mid_x - perp[0], mid_x + perp[0]], 
                   [mid_y - perp[1], mid_y + perp[1]],
                   color=(0.6, 0.95, 0.85), lw=1.5 * scale, alpha=alpha_val)
    
    # Carbon atoms in ring
    ax.scatter(ring_x, ring_y, s=100 * scale**2, color=(0.4, 0.4, 0.4), 
              alpha=alpha_val, edgecolors=(0.6, 0.95, 0.85), linewidths=1.5 * scale)
    
    # Add substituents (functional groups)
    # OH group
    sub1_x, sub1_y = ring_x[0], ring_y[0]
    ax.plot([sub1_x, sub1_x + 0.05 * scale], [sub1_y, sub1_y + 0.04 * scale],
            color=(0.6, 0.95, 0.85), lw=2.0 * scale, alpha=alpha_val)
    ax.scatter([sub1_x + 0.05 * scale], [sub1_y + 0.04 * scale], s=70 * scale**2, 
              color=(0.9, 0.9, 0.95), alpha=alpha_val, 
              edgecolors=(0.6, 0.95, 0.85), linewidths=1 * scale)
    
    # Methyl group
    sub2_x, sub2_y = ring_x[3], ring_y[3]
    ax.plot([sub2_x, sub2_x + 0.04 * scale], [sub2_y, sub2_y - 0.05 * scale],
            color=(0.6, 0.95, 0.85), lw=2.0 * scale, alpha=alpha_val)
    ax.scatter([sub2_x + 0.04 * scale], [sub2_y - 0.05 * scale], s=60 * scale**2,
              color=(0.5, 0.5, 0.5), alpha=alpha_val,
              edgecolors=(0.6, 0.95, 0.85), linewidths=1 * scale)
    
    # Add a second ring (fused or connected)
    ring2_center_x = ring_x[2] + 0.08 * scale
    ring2_center_y = ring_y[2]
    ring2_x = ring2_center_x + ring_radius * 0.7 * np.cos(angles)
    ring2_y = ring2_center_y + ring_radius * 0.7 * np.sin(angles)
    
    # Connection between rings
    ax.plot([ring_x[2], ring2_x[5]], [ring_y[2], ring2_y[5]],
            color=(0.6, 0.95, 0.85), lw=2.0 * scale, alpha=alpha_val)
    
    # Second ring
    for k in range(6):
        ax.plot([ring2_x[k], ring2_x[(k + 1) % 6]], [ring2_y[k], ring2_y[(k + 1) % 6]],
                color=(0.6, 0.95, 0.85), lw=2.0 * scale, alpha=alpha_val)
    ax.scatter(ring2_x, ring2_y, s=90 * scale**2, color=(0.4, 0.4, 0.4),
              alpha=alpha_val, edgecolors=(0.6, 0.95, 0.85), linewidths=1.5 * scale)


def draw_3d_pocket_matching(ax, ligand_cx, ligand_cy, pocket_cx, pocket_cy, 
                            ligand_3d_theta, ligand_3d_phi, pocket_rot, 
                            morph_t, alpha_val=1.0):
    """Draw 3D conformer fitting into protein pocket with realistic 3D visualization."""
    # Draw protein pocket (concave cavity) with 3D depth effect
    pocket_angles = np.linspace(0, 2 * np.pi, 40)
    pocket_r_base = 0.12
    pocket_r = pocket_r_base + 0.02 * np.sin(4 * pocket_angles) + 0.015 * np.cos(6 * pocket_angles)
    
    # Create 3D pocket surface (elliptical for depth)
    pocket_x = pocket_cx + pocket_r * np.cos(pocket_angles + pocket_rot)
    pocket_y = pocket_cy + pocket_r * np.sin(pocket_angles + pocket_rot) * 0.75  # Elliptical for 3D effect
    
    # Draw pocket surface with gradient and depth shading
    # Outer rim (darker, further back)
    pocket_x_outer = pocket_cx + (pocket_r + 0.01) * np.cos(pocket_angles + pocket_rot)
    pocket_y_outer = pocket_cy + (pocket_r + 0.01) * np.sin(pocket_angles + pocket_rot) * 0.75
    ax.fill(pocket_x_outer, pocket_y_outer, color=(0.2, 0.4, 0.7), alpha=alpha_val * 0.2, zorder=1)
    
    # Inner surface (brighter, closer)
    ax.fill(pocket_x, pocket_y, color=(0.25, 0.5, 0.85), alpha=alpha_val * 0.35, zorder=2)
    ax.plot(pocket_x, pocket_y, color=(0.4, 0.65, 0.95), alpha=alpha_val * 0.7, lw=2.5, zorder=3)
    
    # Draw pocket residues around the cavity with 3D positioning (bigger and more scattered)
    n_residues = 18  # Increased from 12 to 18 for more balls
    residue_types = ['hydrophobic', 'polar', 'charged', 'aromatic']
    residue_positions_3d = []
    rng = np.random.default_rng(42)  # Use seed for reproducibility
    
    for i in range(n_residues):
        base_angle = 2 * np.pi * i / n_residues + pocket_rot
        # Add more scatter by varying the distance from center (0.02 to 0.05 spread)
        res_dist = pocket_r_base + 0.02 + 0.03 * rng.random()
        # Add angular scatter to break the perfect circle
        angle = base_angle + 0.15 * (rng.random() - 0.5)
        # Add more z-variation for 3D effect and scatter
        z_offset = 0.02 * np.sin(base_angle * 2) + 0.02 * (rng.random() - 0.5)
        
        res_x_3d = res_dist * np.cos(angle)
        res_y_3d = res_dist * np.sin(angle)
        res_z_3d = z_offset
        
        # Project to 2D with perspective
        res_x_proj, res_y_proj = project_3d_to_2d(res_x_3d, res_y_3d, res_z_3d, 0.3, 0.2)
        res_x = pocket_cx + res_x_proj
        res_y = pocket_cy + res_y_proj
        
        residue_positions_3d.append((res_x, res_y, res_z_3d))
        res_type = residue_types[i % len(residue_types)]
        
        # Size based on depth - increased base size and variation for bigger balls
        depth_factor = (res_z_3d + 0.03) / 0.06  # Normalize to 0-1
        size = (60 + 40 * depth_factor) * alpha_val  # Increased from (35+15) to (60+40)
        draw_residue(ax, res_x, res_y, res_type, size=int(size))
    
    # Draw 3D ligand conformer (morphing from 2D) using ball-and-stick rendering
    # Interpolate position from 2D location to pocket center
    ligand_x = lerp(ligand_cx, pocket_cx, morph_t)
    ligand_y = lerp(ligand_cy, pocket_cy, morph_t)
    
    # Interpolate scale (starts larger, shrinks as it fits into pocket)
    scale_3d = lerp(1.3, 0.9, morph_t)
    
    # Create realistic 3D molecular structure
    # Use RDKit if available, otherwise create a complex structure
    if HAS_RDKIT and morph_t > 0.3:
        try:
            mol = Chem.MolFromSmiles(DEFAULT_SMILES)
            if mol is not None:
                mol_3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_3d)
                conf = mol_3d.GetConformer()
                
                ligand_atoms_3d = []
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
                    # Scale coordinates appropriately
                    ligand_atoms_3d.append([pos.x * 0.008 * scale_3d, 
                                          pos.y * 0.008 * scale_3d, 
                                          pos.z * 0.008 * scale_3d])
                    ligand_colors.append(atom_colors_map.get(atomic_num, (0.6, 0.6, 0.6)))
                
                ligand_bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) 
                               for bond in mol_3d.GetBonds()]
                
                # Apply rotation
                rot_y = np.array([[np.cos(ligand_3d_theta), 0, np.sin(ligand_3d_theta)],
                                  [0, 1, 0],
                                  [-np.sin(ligand_3d_theta), 0, np.cos(ligand_3d_theta)]])
                rot_z = np.array([[np.cos(ligand_3d_phi), -np.sin(ligand_3d_phi), 0],
                                  [np.sin(ligand_3d_phi), np.cos(ligand_3d_phi), 0],
                                  [0, 0, 1]])
                ligand_atoms_3d = np.array(ligand_atoms_3d) @ rot_y.T @ rot_z.T
                
                # Draw using ball-and-stick 3D rendering
                draw_ball_and_stick_3d(ax, ligand_x, ligand_y, ligand_atoms_3d.tolist(), ligand_bonds,
                                       atom_colors=ligand_colors,
                                       bond_color=(0.6, 0.9, 0.85),
                                       view_angle_x=0.35, view_angle_y=0.25,
                                       atom_scale=2.0 * scale_3d, bond_width=2.5 * scale_3d, 
                                       alpha=alpha_val * 0.95)
            else:
                raise ValueError("Failed to create molecule")
        except:
            # Fallback to complex synthetic structure
            _draw_synthetic_3d_ligand(ax, ligand_x, ligand_y, ligand_3d_theta, ligand_3d_phi, 
                                     scale_3d, alpha_val)
    else:
        # Use synthetic structure for early morph or when RDKit unavailable
        _draw_synthetic_3d_ligand(ax, ligand_x, ligand_y, ligand_3d_theta, ligand_3d_phi, 
                                 scale_3d, alpha_val)
    
    # Draw interaction lines between ligand and pocket residues (when close)
    if morph_t > 0.6:
        interaction_alpha = (morph_t - 0.6) / 0.4
        # Get ligand atom positions for interaction lines
        if HAS_RDKIT:
            try:
                mol = Chem.MolFromSmiles(DEFAULT_SMILES)
                if mol is not None:
                    mol_3d = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                    conf = mol_3d.GetConformer()
                    ligand_atoms_3d = []
                    for atom in mol_3d.GetAtoms():
                        idx = atom.GetIdx()
                        pos = conf.GetAtomPosition(idx)
                        ligand_atoms_3d.append([pos.x * 0.008 * scale_3d, 
                                              pos.y * 0.008 * scale_3d, 
                                              pos.z * 0.008 * scale_3d])
                    rot_y = np.array([[np.cos(ligand_3d_theta), 0, np.sin(ligand_3d_theta)],
                                      [0, 1, 0],
                                      [-np.sin(ligand_3d_theta), 0, np.cos(ligand_3d_theta)]])
                    rot_z = np.array([[np.cos(ligand_3d_phi), -np.sin(ligand_3d_phi), 0],
                                      [np.sin(ligand_3d_phi), np.cos(ligand_3d_phi), 0],
                                      [0, 0, 1]])
                    ligand_atoms_3d = np.array(ligand_atoms_3d) @ rot_y.T @ rot_z.T
                    
                    for i, atom_3d in enumerate(ligand_atoms_3d[:min(6, len(ligand_atoms_3d))]):
                        atom_2d_x, atom_2d_y = project_3d_to_2d(atom_3d[0], atom_3d[1], atom_3d[2], 0.35, 0.25)
                        atom_pos = (ligand_x + atom_2d_x, ligand_y + atom_2d_y)
                        
                        # Find nearest pocket residue
                        min_dist = float('inf')
                        nearest_res = None
                        for res_x, res_y, _ in residue_positions_3d:
                            dist = np.sqrt((atom_pos[0] - res_x)**2 + (atom_pos[1] - res_y)**2)
                            if dist < min_dist:
                                min_dist = dist
                                nearest_res = (res_x, res_y)
                        
                        if nearest_res and min_dist < 0.12:
                            ax.plot([atom_pos[0], nearest_res[0]], [atom_pos[1], nearest_res[1]],
                                   color=(0.95, 0.85, 0.55), alpha=interaction_alpha * 0.5, 
                                   lw=2.0, linestyle='--', zorder=4)
            except:
                pass


def _draw_synthetic_3d_ligand(ax, cx, cy, theta, phi, scale, alpha_val):
    """Draw a synthetic 3D ligand structure using ball-and-stick rendering."""
    # Create a complex multi-ring structure
    ring_radius = 0.05 * scale
    n_ring = 6
    
    # Main ring atoms
    atoms_3d = []
    atom_colors = []
    
    # Ring 1 (main aromatic ring)
    for i in range(n_ring):
        angle = 2 * np.pi * i / n_ring
        x_3d = ring_radius * np.cos(angle)
        y_3d = ring_radius * np.sin(angle)
        z_3d = 0.01 * np.sin(angle * 2)  # Slight puckering
        atoms_3d.append([x_3d, y_3d, z_3d])
        atom_colors.append((0.5, 0.5, 0.5))  # Carbon (gray)
    
    # Ring 2 (fused ring)
    ring2_offset = 0.08 * scale
    for i in range(n_ring):
        angle = 2 * np.pi * i / n_ring + np.pi / 3
        x_3d = ring2_offset * np.cos(np.pi / 3) + ring_radius * 0.8 * np.cos(angle)
        y_3d = ring2_offset * np.sin(np.pi / 3) + ring_radius * 0.8 * np.sin(angle)
        z_3d = -0.01 * np.sin(angle * 2)
        atoms_3d.append([x_3d, y_3d, z_3d])
        atom_colors.append((0.5, 0.5, 0.5))
    
    # Substituents
    sub_atoms = [
        [ring_radius * 1.4, 0, 0.02],  # Right substituent (O)
        [0, ring_radius * 1.3, -0.015],  # Top substituent (N)
        [-ring_radius * 1.2, 0, 0.01],  # Left substituent
    ]
    atoms_3d.extend(sub_atoms)
    atom_colors.extend([(0.9, 0.2, 0.2), (0.2, 0.4, 0.9), (0.5, 0.5, 0.5)])  # O, N, C
    
    # Rotate around Y and Z axes
    rot_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
    rot_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                      [np.sin(phi), np.cos(phi), 0],
                      [0, 0, 1]])
    atoms_3d = np.array(atoms_3d) @ rot_y.T @ rot_z.T
    
    # Define bonds
    bonds = []
    # Ring 1 bonds
    for i in range(n_ring):
        bonds.append((i, (i + 1) % n_ring))
    # Ring 2 bonds
    for i in range(n_ring):
        bonds.append((n_ring + i, n_ring + (i + 1) % n_ring))
    # Connection between rings
    bonds.append((0, n_ring))
    bonds.append((2, n_ring + 3))
    # Substituent bonds
    bonds.append((0, 2 * n_ring))  # First substituent
    bonds.append((2, 2 * n_ring + 1))  # Second substituent
    bonds.append((4, 2 * n_ring + 2))  # Third substituent
    
    # Draw using ball-and-stick 3D rendering
    draw_ball_and_stick_3d(ax, cx, cy, atoms_3d.tolist(), bonds, 
                          atom_colors=atom_colors,
                          bond_color=(0.7, 0.85, 0.95),
                          view_angle_x=0.35, view_angle_y=0.25,
                          atom_scale=1.8 * scale, bond_width=2.5 * scale, 
                          alpha=alpha_val * 0.95)


def scene5(ax, t):
    draw_title(ax, "2D to 3D Transition", "Ligand features attend to pocket residues")
    
    # Phase 1 (0-0.4): Show 2D structure
    # Phase 2 (0.4-0.7): Morph from 2D to 3D
    # Phase 3 (0.7-1.0): Show 3D conformer fitting into pocket
    
    if t < 0.4:
        # Phase 1: 2D structure
        phase1_t = t / 0.4
        alpha_2d = smoothstep(phase1_t)
        
        # Draw 2D structure on left
        cx_2d, cy_2d = 0.25, 0.50
        draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        
        ax.text(0.25, 0.75, "RDKit 2D Structure", color=FG, fontsize=16, fontweight="bold", ha="center")
        ax.text(0.25, 0.70, "Flat molecular representation", color=(0.75, 0.78, 0.85), fontsize=12, ha="center")
        
        # Arrow pointing right
        draw_arrow(ax, 0.40, 0.50, 0.55, 0.50, alpha=0.3 * alpha_2d)
        
    elif t < 0.7:
        # Phase 2: Morphing transition
        phase2_t = (t - 0.4) / 0.3
        morph_t = smoothstep(phase2_t)
        
        # Keep 2D visible but dimmed (not fading out)
        alpha_2d = 0.3  # Dimmed but visible
        cx_2d, cy_2d = 0.25, 0.50
        draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        
        # Keep 2D labels visible but dimmed
        ax.text(0.25, 0.75, "RDKit 2D Structure", color=FG, fontsize=16, fontweight="bold", ha="center", alpha=alpha_2d)
        ax.text(0.25, 0.70, "Flat molecular representation", color=(0.75, 0.78, 0.85), fontsize=12, ha="center", alpha=alpha_2d)
        
        # Arrow pointing right (visible throughout)
        draw_arrow(ax, 0.40, 0.50, 0.55, 0.50, alpha=0.6)
        
        # Fade in 3D with enhanced rotation
        alpha_3d = morph_t
        ligand_cx, ligand_cy = 0.25, 0.50
        pocket_cx, pocket_cy = 0.70, 0.50
        # Enhanced rotation to show 3D structure better
        ligand_theta = 2 * np.pi * t * 0.7  # Faster rotation during transition
        ligand_phi = 0.25 + 0.1 * np.sin(2 * np.pi * t * 0.5)  # Varying angle
        pocket_rot = 0.1
        
        # Add shadow during transition for depth
        if morph_t > 0.3:
            from matplotlib.patches import Ellipse
            shadow_alpha = (morph_t - 0.3) / 0.7 * 0.3
            shadow_y = lerp(ligand_cy, pocket_cy, morph_t) - 0.06
            shadow_x = lerp(ligand_cx, pocket_cx, morph_t)
            shadow_ellipse = Ellipse((shadow_x, shadow_y), 0.15 * morph_t, 0.05 * morph_t, 
                                     facecolor=(0.1, 0.1, 0.15), alpha=shadow_alpha, zorder=0)
            ax.add_patch(shadow_ellipse)
        
        draw_3d_pocket_matching(ax, ligand_cx, ligand_cy, pocket_cx, pocket_cy,
                              ligand_theta, ligand_phi, pocket_rot, morph_t, alpha_val=alpha_3d)
        
        ax.text(0.70, 0.75, "3D Conformer Generation", color=FG, fontsize=16, fontweight="bold", ha="center")
        ax.text(0.70, 0.70, f"Transition: {int(morph_t*100)}%", color=(0.75, 0.78, 0.85), fontsize=12, ha="center")
        
    else:
        # Phase 3: 3D conformer in pocket
        phase3_t = (t - 0.7) / 0.3
        alpha_3d = 1.0
        
        # Keep 2D visible but dimmed on the left
        alpha_2d = 0.3  # Dimmed but visible
        cx_2d, cy_2d = 0.25, 0.50
        draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        
        # Keep 2D labels visible but dimmed
        ax.text(0.25, 0.75, "RDKit 2D Structure", color=FG, fontsize=16, fontweight="bold", ha="center", alpha=alpha_2d)
        ax.text(0.25, 0.70, "Flat molecular representation", color=(0.75, 0.78, 0.85), fontsize=12, ha="center", alpha=alpha_2d)
        
        # Arrow pointing right (visible throughout)
        draw_arrow(ax, 0.40, 0.50, 0.55, 0.50, alpha=0.6)
        
        ligand_cx, ligand_cy = 0.25, 0.50
        pocket_cx, pocket_cy = 0.70, 0.50
        # Enhanced rotation for better 3D visualization
        ligand_theta = 2 * np.pi * t * 0.6  # Faster rotation
        ligand_phi = 0.25 + 0.15 * np.sin(2 * np.pi * t * 0.4)  # Varying viewing angle
        pocket_rot = 0.1 + 0.05 * np.sin(2 * np.pi * t * 0.2)
        morph_t = 1.0  # Fully morphed
        
        # Add subtle shadow/ground plane for depth
        shadow_y = pocket_cy - 0.08
        from matplotlib.patches import Ellipse
        shadow_ellipse = Ellipse((pocket_cx, shadow_y), 0.20, 0.06, 
                                 facecolor=(0.1, 0.1, 0.15), alpha=0.3, zorder=0)
        ax.add_patch(shadow_ellipse)
        
        draw_3d_pocket_matching(ax, ligand_cx, ligand_cy, pocket_cx, pocket_cy,
                              ligand_theta, ligand_phi, pocket_rot, morph_t, alpha_val=alpha_3d)
        
        ax.text(0.70, 0.75, "3D Pocket Matching", color=FG, fontsize=16, fontweight="bold", ha="center")
        ax.text(0.70, 0.70, "Ligand fits into binding site", color=(0.75, 0.78, 0.85), fontsize=12, ha="center")
        
        # Add fit quality indicator with 3D perspective
        fit_score = 0.85 + 0.1 * np.sin(2 * np.pi * t * 0.5)
        ax.text(0.70, 0.25, f"Binding Affinity: {fit_score:.2f}", 
               color=(0.95, 0.85, 0.55), fontsize=14, fontweight="bold", ha="center")
        
        # Add 3D rotation indicator
        ax.text(0.70, 0.18, "Rotating 3D view", color=(0.65, 0.75, 0.85), fontsize=10, ha="center", 
               style='italic', alpha=0.7)


# -----------------------------
# Scene 6: Ensemble Prediction
# -----------------------------
def scene6(ax, t):
    draw_title(ax, "Ensemble Prediction", "Multiple models vote for robust scores")

    # Line 1: Draw three model blocks (top row)
    line1_y = 0.65
    xs = [0.20, 0.42, 0.64]
    labels = ["Model 1", "Model 2", "Model 3"]
    outs = []
    for x, lab, phase in zip(xs, labels, [0.0, 0.33, 0.66]):
        ax.add_patch(plt.Rectangle((x, line1_y), 0.16, 0.16, facecolor=(0.2, 0.25, 0.35), alpha=0.9,
                                   edgecolor=(0.6, 0.8, 0.98), lw=2))
        ax.text(x + 0.08, line1_y + 0.12, lab, ha="center", va="center", color=FG, fontsize=12, fontweight="bold")
        score = 0.55 + 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * (t + phase)))
        outs.append(score)
        ax.text(x + 0.08, line1_y + 0.05, f"{score:.2f}", ha="center", va="center",
                color=(0.75, 0.9, 1.0), fontsize=14)

    # Line 2: Combine into ensemble (bottom row, centered)
    ens = float(np.mean(outs))
    ensemble_x = 0.5
    ensemble_y = 0.35
    ensemble_width = 0.20
    ensemble_height = 0.20
    
    ax.add_patch(plt.Rectangle((ensemble_x - ensemble_width/2, ensemble_y), ensemble_width, ensemble_height, 
                               facecolor=(0.25, 0.30, 0.45), alpha=0.9,
                               edgecolor=(0.95, 0.85, 0.55), lw=2))
    ax.text(ensemble_x, ensemble_y + 0.15, "Ensemble", ha="center", va="center", 
            color=FG, fontsize=14, fontweight="bold")
    ax.text(ensemble_x, ensemble_y + 0.06, f"{ens:.2f}", ha="center", va="center", 
            color=(0.95, 0.85, 0.55), fontsize=22, fontweight="bold")

    # Arrows from line 1 (models) pointing down to line 2 (ensemble)
    for x in xs:
        # Arrow from bottom of model box to top of ensemble box
        draw_arrow(ax, x + 0.08, line1_y, ensemble_x, ensemble_y + ensemble_height, alpha=0.65, lw=2.5)

    ax.text(0.05, 0.20, "Goal: stable ranking across noise & conformers", color=(0.75, 0.78, 0.85), fontsize=12)


# -----------------------------
# Scene 7: Output & Ranking
# -----------------------------
def scene7(ax, t):
    draw_title(ax, "Ranked Output", "Binding affinity scores and top hits")

    # Simulated scores that "sort" over time
    rng = np.random.default_rng(7)
    n = 10
    names = [f"Ligand_{i:02d}" for i in range(1, n + 1)]
    base_scores = rng.uniform(0.2, 0.95, size=n)

    # Interpolate from unsorted to sorted positions
    idx_unsorted = np.arange(n)
    idx_sorted = np.argsort(base_scores)[::-1]

    # target y positions
    y_positions = np.linspace(0.75, 0.18, n)
    # current permutation is a blend between unsorted and sorted
    blend = smoothstep(t)

    # Map each ligand to a y position that moves toward its sorted slot
    y_cur = np.zeros(n)
    for i in range(n):
        # current y = lerp(unsorted_slot, sorted_slot)
        u_slot = np.where(idx_unsorted == i)[0][0]
        s_slot = np.where(idx_sorted == i)[0][0]
        y_cur[i] = lerp(y_positions[u_slot], y_positions[s_slot], blend)

    # Draw bars
    x0 = 0.25
    maxw = 0.60
    y_top3_bottom = None  # Will store the y position below top 3
    
    for i in range(n):
        score = base_scores[i]
        width = maxw * score
        sorted_pos = np.where(idx_sorted == i)[0][0]
        is_top = (sorted_pos < 3)  # top 3 in sorted order
        
        # Track the bottom of the 3rd ligand (sorted index 2)
        if sorted_pos == 2:
            y_top3_bottom = y_cur[i] - 0.018  # Bottom of the 3rd ligand bar
        
        color = (0.6, 0.95, 0.85) if is_top and blend > 0.6 else (0.75, 0.9, 1.0)
        alpha = 0.95 if is_top else 0.75

        ax.text(0.06, y_cur[i], names[i], color=FG, fontsize=12, va="center")
        ax.add_patch(plt.Rectangle((x0, y_cur[i] - 0.018), width, 0.036, color=color, alpha=alpha))
        ax.text(x0 + width + 0.01, y_cur[i], f"{score:.2f}", color=(0.85, 0.9, 1.0), fontsize=11, va="center")
    
    # Draw text below top 3 ligands and line below the text
    if y_top3_bottom is not None:
        # Text position: positioned right at the bottom of 3rd ligand (moved higher) to avoid overlap with 4th ligand
        text_y = y_top3_bottom + 0.01  # Positioned at or slightly above the bottom of 3rd ligand bar
        ax.text(0.50, text_y, "Successfully Screened Candidates of Small Molecules above Threshold", 
               color=(0.95, 0.85, 0.55), fontsize=13, fontweight="bold", 
               ha="center", va="top", zorder=11)
        
        # Line position: below the text, moved down by 1 line spacing (approximately 0.04)
        line_y = text_y - 0.04
        # Draw horizontal line across the width
        ax.plot([0.05, 0.95], [line_y, line_y], 
               color=(0.95, 0.85, 0.55), lw=2.5, alpha=0.8, zorder=10)

    # Moved "Top hits advance" text higher to avoid overlapping with caption
    ax.text(0.05, 0.22, "Top hits advance to downstream validation", color=(0.75, 0.78, 0.85), fontsize=12)


# -----------------------------
# Main render loop
# -----------------------------
def draw_scene(ax, scene_idx, t):
    if scene_idx == 0:
        scene1(ax, t)
    elif scene_idx == 1:
        scene2(ax, t)
    elif scene_idx == 2:
        scene3(ax, t)
    elif scene_idx == 3:
        scene4(ax, t)  # Cross-Attention scene
    elif scene_idx == 4:
        scene6(ax, t)  # Ensemble prediction
    elif scene_idx == 5:
        scene7(ax, t)  # Output ranking
    elif scene_idx == 6:
        scene5(ax, t)  # 2D to 3D transition scene (moved to end)
    else:
        ax.text(0.5, 0.5, "Unknown scene", color=FG, ha="center", va="center")


def main():
    # Load subtitles
    subtitles = parse_srt_from_docx(SUBTITLE_PATH)
    print(f"Loaded {len(subtitles)} subtitles")
    
    # Find original scene 6 caption (to be used for scene 7) - unused variable, kept for compatibility
    scene6_time = 55.0  # Scene 5 (Ensemble) now starts at frame 550 / 10 fps = 55 seconds
    original_scene6_caption = get_subtitle_at_time(subtitles, scene6_time)
    
    writer = imageio.get_writer(OUT_PATH, fps=FPS, codec="libx264", quality=8)
    try:
        for f in range(FRAMES):
            s = scene_of_frame(f)
            t = local_t(f)
            time_seconds = f / FPS  # Current time in seconds
            
            fig, ax = setup_ax()
            draw_scene(ax, s, t)
            
            # Add subtitle if available
            # Override for Scene 1: use first caption for entire scene
            if s == 0:  # Scene 1 (0-indexed)
                if len(subtitles) > 0:
                    subtitle_text = subtitles[0]['text']  # Use first subtitle for entire scene 1
                    draw_subtitle(ax, subtitle_text)
            # Override for Scene 2 (Multi-Branch Encoding) with specific caption - smaller font
            elif s == 1:  # Scene 2 (0-indexed)
                subtitle_text = "Each ligand is encoded in 3 complementary ways: chemical semantics using ChemBERTa, 2-D geometry using RDKit 2D features, and 3-D ligand conformers using RDKit 3D conformer generation."
                draw_subtitle(ax, subtitle_text, fontsize=14)  # Reduced from 15 to 14
            # Override for Scene 3 (3D Protein Pocket Matching) with specific caption (forced 2 lines)
            elif s == 2:  # Scene 3 (0-indexed)
                subtitle_text = "pocket detection: Uses geometric/concavity analysis to find cavities\nDruggability scoring: Each pocket scored 0-1 based on residue composition"
                draw_subtitle(ax, subtitle_text)
            # Override for Scene 4 (Cross-Attention) - forced 2 lines
            elif s == 3:  # Scene 4 (0-indexed)
                subtitle_text = "Cross-attention then links ligand features with pocket residues, \nallowing the model to focus on the most relevant molecular interactions."
                draw_subtitle(ax, subtitle_text)
            # Override for Scene 5 (Ensemble Prediction) - using "Multiple predictive models..." caption
            elif s == 4:  # Scene 5 (0-indexed)
                subtitle_text = "Multiple predictive models evaluate each complex and their outputs are combined through an ensemble for robust scoring"
                draw_subtitle(ax, subtitle_text)
            # Override for Scene 6 (Ranked Output) - specific caption (forced two lines)
            elif s == 5:  # Scene 6 (0-indexed)
                subtitle_text = "The result is a ranked list of compounds by predicted binding affinity, \nenabling rapid selection of top candidates for downstream validation."
                draw_subtitle(ax, subtitle_text)
            # Override for Scene 7 (2D to 3D Transition) with specific caption
            elif s == 6:  # Scene 7 (0-indexed)
                subtitle_text = "World's 1st platform of HTS of small molecules using 3D/4Branch matching, a breakthrough from current SOTA of 2D architecture(08/2025)"
                draw_subtitle(ax, subtitle_text)
            else:
                subtitle_text = get_subtitle_at_time(subtitles, time_seconds)
                if subtitle_text:
                    draw_subtitle(ax, subtitle_text)
            
            frame = render_frame(fig)
            writer.append_data(frame)
            plt.close(fig)
            if f % 50 == 0:
                print(f"Rendered frame {f}/{FRAMES} (scene {s+1}/7)")
    finally:
        writer.close()
    print(f"Done. Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
