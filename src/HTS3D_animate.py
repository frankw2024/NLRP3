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

Image Dependencies:
  - Scene 0 (Inputs): Uses ligand image from HTS3D_ligandFigure.py (ligands/ligand_figure.png)
  - Scene 2 (Multi-Branch Encoding): Uses ligand image from HTS3D_ligandFigure.py (ligands/ligand_figure.png)
  - Scene 7 (2D to 3D): Uses ligand image from HTS3D_ligandFigure.py (ligands/ligand_figure.png) on left,
                         and protein image from HTS3D_nlrp3Figure.py (nlrp3/panel_A_nlrp3_alone.png) on right
  Run these scripts first to generate the required images before running the animation.

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
import argparse
from pathlib import Path

# Fix OpenMP error: OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
from matplotlib.transforms import Affine2D
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
# Default: 7 scenes (700 frames). With --include-scene8: 8 scenes (800 frames)
# Scene order: 0=Inputs, 1=Multi-Branch, 2=Cross-Attention, 3=Ensemble, 4=Ranked, 5=2D-to-3D, 6=2D-to-3D-duplicate, 7=3D-Pocket-Matching (optional)
FRAMES = 700  # Default: first 7 scenes only. Use --include-scene8 to generate all 8 scenes.

SCENE_LEN_S = 10  # Doubled from 5 to match 2x slower animation
SCENE_FRAMES = FPS * SCENE_LEN_S  # 100 frames/scene (doubled from 50)

OUT_PATH = "hts3d_explainer.mp4"
# Use path relative to script directory (same approach as ligand image path)
SCRIPT_DIR = Path(__file__).parent
SUBTITLE_PATH = str(SCRIPT_DIR / "codeGen" / "aniVsubtitle_generated.docx")
#SUBTITLE_PATH = str(SCRIPT_DIR / "codeGen" / "aniVsubtitle.docx")

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


def scene_of_frame(f, include_scene8=False):
    """Determine which scene a frame belongs to.
    
    New scene order:
    - Scene 0: Inputs (100 frames: 0-99)
    - Scene 1: Multi-Branch Encoding (100 frames: 100-199)
    - Scene 2: Cross-Attention (100 frames: 200-299) - was scene 3
    - Scene 3: Ensemble Prediction (50 frames: 300-349) - was scene 4
    - Scene 4: Ranked Output (100 frames: 350-449) - was scene 5
    - Scene 5: 2D to 3D Transition (150 frames: 450-599) - was scene 6
    - Scene 6: 2D to 3D Transition duplicate (100 frames: 600-699) - was scene 7
    - Scene 7: 3D Protein Pocket Matching (100 frames: 700-799) - was scene 2, now at end (optional)
    """
    if f < 200:
        return f // SCENE_FRAMES  # Scenes 0-1: 100 frames each
    elif f < 300:  # Scene 2 (Cross-Attention): 100 frames (200-299)
        return 2
    elif f < 350:  # Scene 3 (Ensemble): 50 frames (300-349)
        return 3
    elif f < 450:  # Scene 4 (Ranked): 100 frames (350-449)
        return 4
    elif f < 600:  # Scene 5 (2D to 3D): 150 frames (450-599)
        return 5
    elif f < 700:  # Scene 6 (2D to 3D duplicate): 100 frames (600-699)
        return 6
    elif include_scene8 and f < 800:  # Scene 7 (3D Protein Pocket Matching): 100 frames (700-799)
        return 7
    else:
        # Should not reach here if FRAMES is set correctly
        return 6  # Default to last scene


def local_t(f, include_scene8=False):
    """0..1 within scene."""
    s = scene_of_frame(f, include_scene8)
    if s < 2:
        # Scenes 0-1: normal duration (100 frames each)
        scene_start = s * SCENE_FRAMES
        local_frame = f - scene_start
        return local_frame / (SCENE_FRAMES - 1) if SCENE_FRAMES > 1 else 0
    elif s == 2:
        # Scene 2 (Cross-Attention): 100 frames (200-299)
        scene2_start = 200
        scene2_frames = 100
        local_frame = f - scene2_start
        return local_frame / (scene2_frames - 1) if scene2_frames > 1 else 0
    elif s == 3:
        # Scene 3 (Ensemble): 50 frames (300-349)
        scene3_start = 300
        scene3_frames = 50
        local_frame = f - scene3_start
        return local_frame / (scene3_frames - 1) if scene3_frames > 1 else 0
    elif s == 4:
        # Scene 4 (Ranked): 100 frames (350-449)
        scene4_start = 350
        scene4_frames = 100
        local_frame = f - scene4_start
        return local_frame / (scene4_frames - 1) if scene4_frames > 1 else 0
    elif s == 5:
        # Scene 5 (2D to 3D): 150 frames (450-599)
        scene5_start = 450
        scene5_frames = 150
        local_frame = f - scene5_start
        return local_frame / (scene5_frames - 1) if scene5_frames > 1 else 0
    elif s == 6:
        # Scene 6 (2D to 3D duplicate): 100 frames (600-699)
        scene6_start = 600
        scene6_frames = 100
        local_frame = f - scene6_start
        return local_frame / (scene6_frames - 1) if scene6_frames > 1 else 0
    else:
        # Scene 7 (3D Protein Pocket Matching): 100 frames (700-799)
        scene7_start = 700
        scene7_frames = 100
        local_frame = f - scene7_start
        return local_frame / (scene7_frames - 1) if scene7_frames > 1 else 0


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


def wrap_text(text, max_chars_per_line=85):
    """Wrap text into multiple lines that fit within the video width.
    
    Args:
        text: The text to wrap
        max_chars_per_line: Maximum characters per line (default 85 for fontsize 15)
    
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
    lines = wrap_text(text, max_chars_per_line=85)
    
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
        
        # Draw text without frame (frameless)
        ax.text(0.5, line_y, line, color=FG, fontsize=fontsize, fontweight="normal",
                ha="center", va="center")


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


def make_background_transparent(img, bg_color=None, color_tolerance=0.15):
    """
    Make background pixels transparent by detecting the background color from image edges.
    Preserves existing transparency if the image already has an alpha channel.
    
    Args:
        img: Image array (H, W, 3) or (H, W, 4) in range [0, 255] or [0, 1]
        bg_color: Optional RGB tuple (0-1 range) to use as background color.
                  If None, detects background color from image edges/corners.
        color_tolerance: How similar pixels must be to background to become transparent (0-1).
                        Lower values = stricter matching.
    
    Returns:
        RGBA image array in [0, 1] range with transparent background
    """
    # Convert to float [0, 1] if needed
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.copy()
        if img.max() > 1.0:
            img = img / 255.0
    
    # Handle different image formats
    if len(img.shape) == 2:  # Grayscale
        img = np.stack([img, img, img], axis=2)
    
    # Extract RGB channels and existing alpha if present
    if img.shape[2] == 4:  # RGBA - preserve existing transparency
        rgb = img[:, :, :3]
        existing_alpha = img[:, :, 3]
    else:  # RGB
        rgb = img
        existing_alpha = None
    
    # Detect background color from edges/corners if not provided
    if bg_color is None:
        h, w = rgb.shape[:2]
        # Sample from edges and corners (more reliable than center for background)
        edge_width = max(5, int(min(h, w) * 0.05))  # 5% of image size, minimum 5 pixels
        edge_samples = []
        
        # Top edge
        edge_samples.append(rgb[:edge_width, :, :].reshape(-1, 3))
        # Bottom edge
        edge_samples.append(rgb[-edge_width:, :, :].reshape(-1, 3))
        # Left edge
        edge_samples.append(rgb[:, :edge_width, :].reshape(-1, 3))
        # Right edge
        edge_samples.append(rgb[:, -edge_width:, :].reshape(-1, 3))
        
        # Combine all edge samples
        all_edge_pixels = np.concatenate(edge_samples, axis=0)
        
        # Use median color as background (more robust than mean against outliers)
        bg_color = np.median(all_edge_pixels, axis=0)
    
    bg_color = np.array(bg_color).reshape(1, 1, 3)
    
    # Calculate color distance for each pixel
    # Use Euclidean distance in RGB space
    color_diff = rgb - bg_color
    color_distance = np.sqrt(np.sum(color_diff ** 2, axis=2))
    
    # Create alpha channel: pixels similar to background become transparent
    # Use smooth transition for better edge quality
    # Pixels within tolerance become fully transparent, pixels far from background stay opaque
    normalized_distance = color_distance / color_tolerance
    new_alpha = np.clip(normalized_distance, 0.0, 1.0)
    
    # If image already had alpha, combine with new alpha (use minimum to preserve existing transparency)
    if existing_alpha is not None:
        alpha = np.minimum(new_alpha, existing_alpha)
    else:
        alpha = new_alpha
    
    # Stack RGB and alpha to create RGBA image
    rgba = np.dstack([rgb, alpha])
    
    return rgba


def blend_video_frame_with_background(video_frame, bg_color=(0.05, 0.06, 0.08), blend_factor=0.3):
    """
    Blend video frame with the dark background color to create seamless integration.
    
    Args:
        video_frame: Video frame array (H, W, 3) in range [0, 255] or [0, 1]
        bg_color: Background color RGB tuple (0-1 range)
        blend_factor: How much to blend with background (0.0 = no blend, 1.0 = full background)
    
    Returns:
        Blended frame array in [0, 1] range
    """
    # Convert frame to float [0, 1] if it's in [0, 255] range
    if video_frame.dtype == np.uint8:
        frame = video_frame.astype(np.float32) / 255.0
    else:
        frame = video_frame.copy()
        if frame.max() > 1.0:
            frame = frame / 255.0
    
    # Ensure frame has 3 channels (RGB)
    if len(frame.shape) == 2:  # Grayscale
        frame = np.stack([frame, frame, frame], axis=2)
    elif frame.shape[2] == 4:  # RGBA
        frame = frame[:, :, :3]
    
    # Blend with background color
    bg_array = np.array(bg_color).reshape(1, 1, 3)
    blended = frame * (1.0 - blend_factor) + bg_array * blend_factor
    
    # Adjust brightness slightly to match frame aesthetic (slightly darker)
    blended = blended * 0.95
    
    # Clip to valid range
    blended = np.clip(blended, 0.0, 1.0)
    
    return blended


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
    # Title centered horizontally in the frame (not left-aligned like other scenes)
    ax.text(0.50, 0.94, "High Throughput Screening 3D (HTS-3D)", 
            color=FG, fontsize=28, fontweight="bold", va="top", ha="center")
    ax.text(0.50, 0.875, "Inputs: Ligands (SMILES) + Protein Structure", 
            color=(0.75, 0.78, 0.85), fontsize=14, va="top", ha="center")

    # SMILES text flowing in (more realistic examples)
    # Made 30% smaller and aligned with center of ligand image, stopping at left edge
    smiles = [
        "CC(=O)NC1=CC=C(O)C=C1",  # Acetaminophen
        "C1=CC=C(C=C1)C(C)CC",    # Simple aromatic
        "CC(C)CC1=CC=C(C=C1)O",   # Phenol derivative
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine-like
    ]
    # Ligand image: cx=0.28, cy=0.55, width=0.30, so left edge = 0.28 - 0.15 = 0.13, but need to stop left by 0.30 fw
    # Stop at left edge of ligand image instead of continuing to 0.10
    x_base = lerp(-0.2, 0.0, smoothstep(t))
    # Center text vertically around ligand image center (cy=0.55)
    # With 4 lines and spacing 0.05: start at 0.55 + 0.075 = 0.625, then subtract 0.05 for each line
    ligand_center_y = 0.55
    text_start_y = ligand_center_y + (len(smiles) - 1) * 0.05 / 2  # Center the text block
    for i, s in enumerate(smiles):
        y = text_start_y - i * 0.05  # Centered around ligand image center
        # Use monospace font for SMILES - 30% smaller: 12 * 0.7 = 8.4, rounded to 8.5
        ax.text(x_base, y, s, color=(0.8, 0.85, 0.95), fontsize=8.5, 
               family='monospace', alpha=0.9)

    # Load and display ligand image generated by HTS3D_ligandFigure.py
    # Use Panel A image generated by HTS3D_ligandFigure.py (must run that script first)
    # This replaces the programmatically drawn molecule diagram with the generated 2D ligand figure
    ligand_image_path = SCRIPT_DIR / "ligands" / "ligand_figure.png"
    cx, cy = 0.28, 0.55
    desired_ligand_width = 0.30  # Desired width in axes coordinates
    
    if ligand_image_path.exists():
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(str(ligand_image_path))
            # Calculate aspect ratio to preserve image proportions
            img_height, img_width = img.shape[:2]
            img_aspect_ratio = img_width / img_height
            image_width = desired_ligand_width
            image_height = image_width / img_aspect_ratio  # Maintain aspect ratio
            
            # Calculate bounding box for image placement (centered at cx, cy)
            x0 = cx - image_width / 2
            y0 = cy - image_height / 2
            
            # Display image without rotation
            ax.imshow(img, extent=[x0, x0 + image_width, y0, y0 + image_height], 
                     aspect='equal', zorder=5, alpha=0.95)
        except Exception as e:
            print(f"Warning: Could not load ligand image: {e}")
            # Fallback to original molecule diagram
            morph = smoothstep((t - 0.35) / 0.65)
            draw_molecule_2d(ax, cx, cy, 0.08, morph)
    else:
        # Fallback to original molecule diagram if image doesn't exist
        print(f"Warning: Ligand image not found at {ligand_image_path}")
        print(f"  Please run HTS3D_ligandFigure.py first to generate the ligand image")
        morph = smoothstep((t - 0.35) / 0.65)
        draw_molecule_2d(ax, cx, cy, 0.08, morph)

    ax.text(0.10, 0.80, "Ligands (SMILES)", color=(0.75, 0.78, 0.85), fontsize=12)

    # Load and display protein structure image
    protein_image_path = Path(r"C:\Users\xiaon\mydoc\ant\niu\science2026\NLRP3\abby\proteinLigand1.png")
    px, py = 0.80, 0.55  # Moved right from 0.75 to 0.80 to accommodate larger size and avoid overlap
    desired_protein_width = 0.45  # 50% larger: 0.30 * 1.5 = 0.45
    
    if protein_image_path.exists():
        try:
            import matplotlib.image as mpimg
            protein_img = mpimg.imread(str(protein_image_path))
            
            # Preprocess image to make background transparent
            if isinstance(protein_img, np.ndarray):
                # Make background transparent by detecting background color and removing it
                # This works for both light and dark backgrounds
                protein_img = make_background_transparent(protein_img, bg_color=None, color_tolerance=0.12)
            
            # Calculate aspect ratio to preserve image proportions
            img_height, img_width = protein_img.shape[:2]
            img_aspect_ratio = img_width / img_height
            protein_width = desired_protein_width
            protein_height = protein_width / img_aspect_ratio  # Maintain aspect ratio
            
            # Calculate bounding box for image placement (centered at px, py)
            x0 = px - protein_width / 2
            y0 = py - protein_height / 2
            
            # Display image without rotation (RGBA image with transparency)
            ax.imshow(protein_img, extent=[x0, x0 + protein_width, y0, y0 + protein_height], 
                     aspect='equal', zorder=5)
        except Exception as e:
            print(f"Warning: Could not load protein image: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to original protein structure drawing
            rot = 2 * PI * t * 0.3  # Slower rotation
            if USE_GPU:
                rot = float(to_cpu(xp.asarray(rot)))  # Convert to Python float
            draw_protein_structure(ax, px, py, rot, alpha_val=0.4)
    else:
        # Fallback to original protein structure drawing if image doesn't exist
        print(f"Warning: Protein image not found at {protein_image_path}")
        rot = 2 * PI * t * 0.3  # Slower rotation
        if USE_GPU:
            rot = float(to_cpu(xp.asarray(rot)))  # Convert to Python float
        draw_protein_structure(ax, px, py, rot, alpha_val=0.4)

    # Move "Protein Structure" text to align with larger image (moved right to avoid overlap)
    ax.text(0.75, 0.80, "Protein Structure", color=(0.75, 0.78, 0.85), fontsize=12)

    # Connecting arrow - extended to connect to larger protein image (left edge at ~0.575)
    draw_arrow(ax, 0.40, 0.55, 0.575, 0.55, alpha=0.5)


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

    # Load and display ligand image generated by HTS3D_ligandFigure.py
    # Input molecule on left - using generated ligand image
    ligand_image_path = SCRIPT_DIR / "ligands" / "ligand_figure.png"
    # Moved left to add spacing: original 0.12, moved left by ~0.093 (1/3 of 0.28)
    cx, cy = 0.027, 0.50
    # Increased by 1/3: 0.20 * 1.333 = 0.2667
    image_width, image_height = 0.2667, 0.2667  # Image size in axes coordinates (increased by 1/3)
    
    if ligand_image_path.exists():
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(str(ligand_image_path))
            # Calculate bounding box for image placement (centered at cx, cy)
            x0 = cx - image_width / 2
            y0 = cy - image_height / 2
            ax.imshow(img, extent=[x0, x0 + image_width, y0, y0 + image_height], 
                     aspect='auto', zorder=5, alpha=0.95)
        except Exception as e:
            print(f"Warning: Could not load ligand image: {e}")
            # Fallback to original molecule diagram
            draw_molecule_2d(ax, cx, cy, 0.05, 1.0)
    else:
        # Fallback to original molecule diagram if image doesn't exist
        print(f"Warning: Ligand image not found at {ligand_image_path}")
        draw_molecule_2d(ax, cx, cy, 0.05, 1.0)
    
    ax.text(cx, 0.35, "SMILES", color=(0.75, 0.78, 0.85), fontsize=11, ha="center")

    # Four branches arranged in a grid - increased spacing between ALL diagrams by 1/3 of diagram width (0.093)
    # Branch box width = 0.28, box height = 0.2533, so 1/3 of width = 0.093
    # Horizontal spacing: 
    #   - Gap between ligand and left column: add 0.093 (left col moved right by 0.093)
    #   - Gap between columns: add 0.093 (right col moved right by additional 0.093)
    # Vertical spacing: add 0.093 to gap between rows
    # Original: left col=0.44, right col=0.65, top row=0.69, bottom row=0.41
    branch_positions = [
        (0.533, 0.69, "ChemBERTa\n(2D semantics)", "embedding"),  # Left col, top row (0.44 + 0.093)
        (0.836, 0.69, "Ligand 3D\n(Conformers)", "3d"),  # Right col, top row (0.533 + 0.28 + 0.093 = 0.906, but let's do: 0.65 + 0.093*2 = 0.836)
        (0.533, 0.317, "RDKit 2D\n(Descriptors)", "rdkit"),  # Left col, bottom row (moved right, moved down)
        (0.836, 0.317, "Protein Pocket\n(residues)", "pocket"),  # Right col, bottom row
    ]
    
    # Draw arrows from input to branches (adjusted for new spacing)
    for bx, by, _, _ in branch_positions:
        # Arrow starts from right edge of ligand image (cx + image_width/2) and ends at left edge of branch box (bx - box_w/2)
        draw_arrow(ax, cx + image_width / 2, cy, bx - 0.14, by, alpha=0.6, lw=2.0)
    
    # Branch 1: ChemBERTa embeddings - load image instead of drawing bars
    bx, by, label, branch_type = branch_positions[0]
    ax.text(bx, by + 0.135, label, color=FG, fontsize=11.5, fontweight="bold", ha="center")  # Slightly increased spacing and font
    
    # Load and display ChemBERTa image
    chemberta_image_path = Path(r"C:\Users\xiaon\mydoc\ant\niu\science2026\NLRP3\abby\chemBerta.png")
    chemberta_box_x0, chemberta_box_y0 = bx - 0.14, by - 0.1267  # Box position (half of increased box size)
    # Increased by 1/3: 0.21 * 1.333 = 0.28, 0.19 * 1.333 = 0.2533
    chemberta_box_w, chemberta_box_h = 0.28, 0.2533  # Box size increased by 1/3
    
    if chemberta_image_path.exists():
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(str(chemberta_image_path))
            
            # Make background transparent
            if isinstance(img, np.ndarray):
                img = make_background_transparent(img, bg_color=None, color_tolerance=0.12)
            
            # Calculate aspect ratio to preserve image proportions
            img_height, img_width = img.shape[:2]
            img_aspect_ratio = img_width / img_height
            
            # Fit image to box while preserving aspect ratio
            if img_aspect_ratio > (chemberta_box_w / chemberta_box_h):
                # Image is wider - fit to width
                image_width = chemberta_box_w
                image_height = image_width / img_aspect_ratio
            else:
                # Image is taller - fit to height
                image_height = chemberta_box_h
                image_width = image_height * img_aspect_ratio
            
            # Center image in the box
            image_x0 = chemberta_box_x0 + (chemberta_box_w - image_width) / 2
            image_y0 = chemberta_box_y0 + (chemberta_box_h - image_height) / 2
            
            # Display image - blend with dark background by using the image as-is
            # The image should have its own background or transparency
            ax.imshow(img, extent=[image_x0, image_x0 + image_width, image_y0, image_y0 + image_height], 
                     aspect='equal', zorder=5, alpha=0.95)
        except Exception as e:
            print(f"Warning: Could not load ChemBERTa image: {e}")
            # Fallback to original bar chart
            bars_x0, bars_y0 = bx - 0.095, by - 0.075
            n_bars = 24
            for i in range(n_bars):
                pattern = 0.3 + 0.4 * math.sin(2 * math.pi * (i / n_bars) + 4 * t)
                h = 0.027 + 0.11 * pattern
                color_val = 0.7 + 0.2 * pattern
                ax.add_patch(plt.Rectangle((bars_x0 + i * 0.0075, bars_y0), 0.0055, h,
                                           color=(0.7 * color_val, 0.85 * color_val, 1.0), alpha=0.8))
    else:
        print(f"Warning: ChemBERTa image not found at {chemberta_image_path}")
        # Fallback to original bar chart
        bars_x0, bars_y0 = bx - 0.095, by - 0.075
        n_bars = 24
        for i in range(n_bars):
            pattern = 0.3 + 0.4 * math.sin(2 * math.pi * (i / n_bars) + 4 * t)
            h = 0.027 + 0.11 * pattern
            color_val = 0.7 + 0.2 * pattern
            ax.add_patch(plt.Rectangle((bars_x0 + i * 0.0075, bars_y0), 0.0055, h,
                                       color=(0.7 * color_val, 0.85 * color_val, 1.0), alpha=0.8))
    
    # Draw box frame around the image (optional, to match other branches)
    ax.add_patch(plt.Rectangle((chemberta_box_x0, chemberta_box_y0), chemberta_box_w, chemberta_box_h, 
                               fill=False, edgecolor=(0.5, 0.6, 0.8), lw=1.5, alpha=0.5, zorder=6))
    
    # Branch 2: Ligand 3D conformer - load image instead of drawing 3D conformer
    bx, by, label, branch_type = branch_positions[1]
    ax.text(bx, by + 0.135, label, color=FG, fontsize=11.5, fontweight="bold", ha="center")  # Slightly increased spacing and font
    
    # Load and display Ligand 3D image
    ligand3d_image_path = SCRIPT_DIR / "ligands" / "ligand3D.png"
    ligand3d_box_x0, ligand3d_box_y0 = bx - 0.14, by - 0.1267  # Box position (half of increased box size)
    # Increased by 1/3: 0.21 * 1.333 = 0.28, 0.19 * 1.333 = 0.2533
    ligand3d_box_w, ligand3d_box_h = 0.28, 0.2533  # Box size increased by 1/3
    
    if ligand3d_image_path.exists():
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(str(ligand3d_image_path))
            
            # Make background transparent
            if isinstance(img, np.ndarray):
                img = make_background_transparent(img, bg_color=None, color_tolerance=0.12)
            
            # Calculate aspect ratio to preserve image proportions
            img_height, img_width = img.shape[:2]
            img_aspect_ratio = img_width / img_height
            
            # Fit image to box while preserving aspect ratio
            if img_aspect_ratio > (ligand3d_box_w / ligand3d_box_h):
                # Image is wider - fit to width
                image_width = ligand3d_box_w
                image_height = image_width / img_aspect_ratio
            else:
                # Image is taller - fit to height
                image_height = ligand3d_box_h
                image_width = image_height * img_aspect_ratio
            
            # Center image in the box
            image_x0 = ligand3d_box_x0 + (ligand3d_box_w - image_width) / 2
            image_y0 = ligand3d_box_y0 + (ligand3d_box_h - image_height) / 2
            
            # Display image - blend with dark background by using the image as-is
            # The image should have its own background or transparency
            ax.imshow(img, extent=[image_x0, image_x0 + image_width, image_y0, image_y0 + image_height], 
                     aspect='equal', zorder=5, alpha=0.95)
        except Exception as e:
            print(f"Warning: Could not load Ligand 3D image: {e}")
            # Fallback to original 3D conformer drawing
            theta = 2 * np.pi * t * 0.6
            phi = 0.2 + 0.15 * np.sin(2 * np.pi * t * 0.4)
            draw_3d_conformer(ax, bx, by, theta, phi)
    else:
        print(f"Warning: Ligand 3D image not found at {ligand3d_image_path}")
        # Fallback to original 3D conformer drawing
        theta = 2 * np.pi * t * 0.6
        phi = 0.2 + 0.15 * np.sin(2 * np.pi * t * 0.4)
        draw_3d_conformer(ax, bx, by, theta, phi)
    
    # Draw box frame around the image (to match other branches)
    ax.add_patch(plt.Rectangle((ligand3d_box_x0, ligand3d_box_y0), ligand3d_box_w, ligand3d_box_h, 
                               fill=False, edgecolor=(0.5, 0.6, 0.8), lw=1.5, alpha=0.5, zorder=6))
    
    # Branch 3: RDKit 2D descriptors - load image instead of drawing detailed panel
    bx, by, label, branch_type = branch_positions[2]
    ax.text(bx, by + 0.135, label, color=FG, fontsize=11.5, fontweight="bold", ha="center")  # Slightly increased spacing and font
    
    # Load and display RDKit 2D image
    rdkit2d_image_path = Path(r"C:\Users\xiaon\mydoc\ant\niu\science2026\NLRP3\abby\RDkit2D.png")
    rdkit2d_box_x0, rdkit2d_box_y0 = bx - 0.14, by - 0.1267  # Box position (half of increased box size)
    # Increased by 1/3: 0.21 * 1.333 = 0.28, 0.19 * 1.333 = 0.2533
    rdkit2d_box_w, rdkit2d_box_h = 0.28, 0.2533  # Box size increased by 1/3
    
    if rdkit2d_image_path.exists():
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(str(rdkit2d_image_path))
            
            # Make background transparent
            if isinstance(img, np.ndarray):
                img = make_background_transparent(img, bg_color=None, color_tolerance=0.12)
            
            # Calculate aspect ratio to preserve image proportions
            img_height, img_width = img.shape[:2]
            img_aspect_ratio = img_width / img_height
            
            # Fit image to box while preserving aspect ratio
            if img_aspect_ratio > (rdkit2d_box_w / rdkit2d_box_h):
                # Image is wider - fit to width
                image_width = rdkit2d_box_w
                image_height = image_width / img_aspect_ratio
            else:
                # Image is taller - fit to height
                image_height = rdkit2d_box_h
                image_width = image_height * img_aspect_ratio
            
            # Center image in the box
            image_x0 = rdkit2d_box_x0 + (rdkit2d_box_w - image_width) / 2
            image_y0 = rdkit2d_box_y0 + (rdkit2d_box_h - image_height) / 2
            
            # Display image - blend with dark background by using the image as-is
            # The image should have its own background or transparency
            ax.imshow(img, extent=[image_x0, image_x0 + image_width, image_y0, image_y0 + image_height], 
                     aspect='equal', zorder=5, alpha=0.95)
        except Exception as e:
            print(f"Warning: Could not load RDKit 2D image: {e}")
            # Fallback to original detailed panel (simplified version)
            panel_x0, panel_y0 = bx - 0.105, by - 0.19
            panel_w, panel_h = 0.21, 0.29
            ax.add_patch(plt.Rectangle((panel_x0, panel_y0), panel_w, panel_h, 
                                       facecolor=(0.12, 0.15, 0.22), alpha=0.7,
                                       edgecolor=(0.5, 0.6, 0.8), lw=2, zorder=0))
            mol_y = by + 0.02
            draw_detailed_rdkit_2d_molecule(ax, bx, mol_y, scale=0.75, alpha=0.95)
    else:
        print(f"Warning: RDKit 2D image not found at {rdkit2d_image_path}")
        # Fallback to original detailed panel (simplified version)
        panel_x0, panel_y0 = bx - 0.105, by - 0.19
        panel_w, panel_h = 0.21, 0.29
        ax.add_patch(plt.Rectangle((panel_x0, panel_y0), panel_w, panel_h, 
                                   facecolor=(0.12, 0.15, 0.22), alpha=0.7,
                                   edgecolor=(0.5, 0.6, 0.8), lw=2, zorder=0))
        mol_y = by + 0.02
        draw_detailed_rdkit_2d_molecule(ax, bx, mol_y, scale=0.75, alpha=0.95)
    
    # Draw box frame around the image (to match other branches)
    ax.add_patch(plt.Rectangle((rdkit2d_box_x0, rdkit2d_box_y0), rdkit2d_box_w, rdkit2d_box_h, 
                               fill=False, edgecolor=(0.5, 0.6, 0.8), lw=1.5, alpha=0.5, zorder=6))
    
    # Branch 4: Protein Pocket - load image instead of drawing 3D representation
    bx, by, label, branch_type = branch_positions[3]
    ax.text(bx, by + 0.135, label, color=FG, fontsize=11.5, fontweight="bold", ha="center")  # Slightly increased spacing and font
    
    # Load and display Protein Pocket image
    pocket_image_path = Path(r"C:\Users\xiaon\mydoc\ant\niu\science2026\NLRP3\abby\pocketResidue.png")
    pocket_box_x0, pocket_box_y0 = bx - 0.14, by - 0.1267  # Box position (half of increased box size)
    # Increased by 1/3: 0.21 * 1.333 = 0.28, 0.19 * 1.333 = 0.2533
    pocket_box_w, pocket_box_h = 0.28, 0.2533  # Box size increased by 1/3
    
    if pocket_image_path.exists():
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(str(pocket_image_path))
            
            # Make background transparent
            if isinstance(img, np.ndarray):
                img = make_background_transparent(img, bg_color=None, color_tolerance=0.12)
            
            # Calculate aspect ratio to preserve image proportions
            img_height, img_width = img.shape[:2]
            img_aspect_ratio = img_width / img_height
            
            # Fit image to box while preserving aspect ratio
            if img_aspect_ratio > (pocket_box_w / pocket_box_h):
                # Image is wider - fit to width
                image_width = pocket_box_w
                image_height = image_width / img_aspect_ratio
            else:
                # Image is taller - fit to height
                image_height = pocket_box_h
                image_width = image_height * img_aspect_ratio
            
            # Center image in the box
            image_x0 = pocket_box_x0 + (pocket_box_w - image_width) / 2
            image_y0 = pocket_box_y0 + (pocket_box_h - image_height) / 2
            
            # Display image - blend with dark background by using the image as-is
            # The image should have its own background or transparency
            ax.imshow(img, extent=[image_x0, image_x0 + image_width, image_y0, image_y0 + image_height], 
                     aspect='equal', zorder=5, alpha=0.95)
        except Exception as e:
            print(f"Warning: Could not load Protein Pocket image: {e}")
            # Fallback to original 3D pocket representation
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
                                   atom_scale=1.35, bond_width=1.35, alpha=0.8)
    else:
        print(f"Warning: Protein Pocket image not found at {pocket_image_path}")
        # Fallback to original 3D pocket representation
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
                               atom_scale=1.35, bond_width=1.35, alpha=0.8)
    
    # Draw box frame around the image (to match other branches)
    ax.add_patch(plt.Rectangle((pocket_box_x0, pocket_box_y0), pocket_box_w, pocket_box_h, 
                               fill=False, edgecolor=(0.5, 0.6, 0.8), lw=1.5, alpha=0.5, zorder=6))


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
    # 16 ligand features arranged in a gentle curve
    ax.text(0.10, 0.80, "Ligand features", color=FG, fontsize=14, fontweight="bold")
    ligand_center = np.array([0.16, 0.47])
    
    # Generate 16 ligand atoms/features in a gentle S-curve
    n_lig = 16
    ligand_atoms = []
    # Create a gentle S-curve: use parametric curve with smooth sine wave
    curve_width = 0.18  # Horizontal span of the curve
    curve_amplitude = 0.12  # Vertical amplitude of the curve (gentle S-shape)
    for i in range(n_lig):
        # Parameter t from -1 to 1 for symmetric curve
        t_param = (i / (n_lig - 1) - 0.5) * 2 if n_lig > 1 else 0.0
        # Gentle S-curve: x varies linearly, y follows a smooth sine curve
        x_offset = t_param * curve_width / 2  # Linear horizontal distribution from -width/2 to +width/2
        # S-curve using sine: creates smooth gentle S-shaped curve
        y_offset = curve_amplitude * np.sin(t_param * np.pi)  # Smooth S-curve from -amplitude to +amplitude
        ligand_atoms.append([ligand_center[0] + x_offset, ligand_center[1] + y_offset])
    ligand_atoms = np.array(ligand_atoms)
    
    # Draw atoms with different sizes/types (cycle through colors)
    atom_colors = [
        (0.6, 0.95, 0.85), (0.7, 0.98, 0.9), (0.5, 0.9, 0.8), (0.65, 0.97, 0.88),
        (0.55, 0.92, 0.82), (0.75, 0.96, 0.87), (0.58, 0.93, 0.83), (0.68, 0.97, 0.89),
        (0.62, 0.94, 0.84), (0.72, 0.98, 0.91), (0.52, 0.91, 0.81), (0.66, 0.96, 0.86),
        (0.59, 0.92, 0.82), (0.69, 0.97, 0.88), (0.63, 0.95, 0.85), (0.73, 0.99, 0.90)
    ]
    for i, (atom, color) in enumerate(zip(ligand_atoms, atom_colors)):
        ax.scatter([atom[0]], [atom[1]], s=80 + (i % 4)*8, color=color, alpha=0.9,
                  edgecolors=(0.8, 1.0, 0.95), linewidths=1.5)

    # More realistic pocket residues (right) - 16 residues with amino acid types
    ax.text(0.72, 0.80, "Pocket residues", color=FG, fontsize=14, fontweight="bold")
    # Generate 16 pocket residues in a gentle S-curve (same pattern as ligand)
    n_res = 16
    pocket_center = np.array([0.81, 0.47])
    residues = []
    residue_types = ['charged', 'polar', 'hydrophobic', 'aromatic']  # Cycle through types
    # Create a gentle S-curve: use parametric curve with smooth sine wave (same as ligand)
    curve_width = 0.18  # Horizontal span of the curve
    curve_amplitude = 0.12  # Vertical amplitude of the curve (gentle S-shape)
    for i in range(n_res):
        # Parameter t from -1 to 1 for symmetric curve
        t_param = (i / (n_res - 1) - 0.5) * 2 if n_res > 1 else 0.0
        # Gentle S-curve: x varies linearly, y follows a smooth sine curve
        x_offset = t_param * curve_width / 2  # Linear horizontal distribution from -width/2 to +width/2
        # S-curve using sine: creates smooth gentle S-shaped curve
        y_offset = curve_amplitude * np.sin(t_param * np.pi)  # Smooth S-curve from -amplitude to +amplitude
        residues.append([pocket_center[0] + x_offset, pocket_center[1] + y_offset])
    residues = np.array(residues)
    
    # Assign residue types cycling through the types
    residue_type_list = [residue_types[i % len(residue_types)] for i in range(n_res)]
    
    for i, (res_pos, res_type) in enumerate(zip(residues, residue_type_list)):
        draw_residue(ax, res_pos[0], res_pos[1], res_type, size=120)

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
    # Beam widths reduced to 1/3 of original: (1.5 + 3.5*w)/3 = 0.5 + 1.167*w
    for i in range(len(ligand_atoms)):
        top_js = np.argsort(weights[i])[::-1][:2]
        for j in top_js:
            w = weights[i, j]
            alpha = 0.15 + 0.6 * w
            lw = (1.5 + 3.5 * w) / 3.0  # 1/3 of original width: 0.5 + 1.167*w
            
            # Create gradient effect along the line
            x1, y1 = ligand_atoms[i, 0], ligand_atoms[i, 1]
            x2, y2 = residues[j, 0], residues[j, 1]
            
            # Draw main attention beam (thinner)
            ax.plot([x1, x2], [y1, y2],
                    color=(0.85, 0.9, 1.0), alpha=alpha, lw=lw, zorder=1)
            
            # Add glow effect for strong attention (also 1/3 width)
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
    # Title centered horizontally in the frame (not left-aligned like other scenes)
    ax.text(0.50, 0.94, "HTS-3D: Inhibitor ligand binding in protein pocket", 
            color=FG, fontsize=28, fontweight="bold", va="top", ha="center")
    
    # Phase 1 (0-0.2): Show 2D structure (reduced pause)
    # Phase 2 (0.2-0.5): Morph from 2D to 3D (faster transition start)
    # Phase 3 (0.5-1.0): Show 3D conformer fitting into pocket
    
    # Load video file for HTS 3D visualization (scene 7 only)
    video_path = SCRIPT_DIR / "ligands" / "YTDown.com_Shorts_Proteins-are-highly-dynamic-molecules_Media_G48pFBgbqPM_001_720p.mp4"
    pocket_cx, pocket_cy = 0.78, 0.50  # Moved further right from 0.70 to 0.78
    hts3d_image_width = 0.57  # 1.5x larger: 0.38 * 1.5 = 0.57
    hts3d_image_height = 0.57  # Default, will be recalculated if video exists
    video_reader = None
    video_frame = None
    video_fps = None
    video_frame_count = None
    hts3d_img = None  # Fallback static image
    
    if video_path.exists():
        try:
            video_reader = imageio.get_reader(str(video_path))
            video_meta = video_reader.get_meta_data()
            video_fps = video_meta.get('fps', 30)
            video_frame_count = video_reader.count_frames()
            # Calculate aspect ratio from first frame to preserve proportions
            first_frame = video_reader.get_data(0)
            img_height, img_width = first_frame.shape[:2]
            aspect_ratio = img_width / img_height
            hts3d_image_height = hts3d_image_width / aspect_ratio  # Adjust height based on aspect ratio
        except Exception as e:
            print(f"Warning: Could not load HTS 3D video: {e}")
            video_reader = None
            # Fallback to static image
            hts3d_image_path = SCRIPT_DIR / "nlrp3" / "panel_A_nlrp3_alone.png"
            if hts3d_image_path.exists():
                try:
                    import matplotlib.image as mpimg
                    hts3d_img = mpimg.imread(str(hts3d_image_path))
                    img_height, img_width = hts3d_img.shape[:2]
                    aspect_ratio = img_width / img_height
                    hts3d_image_height = hts3d_image_width / aspect_ratio
                except Exception as e2:
                    print(f"Warning: Could not load fallback HTS 3D image: {e2}")
    else:
        print(f"Warning: Video file not found at {video_path}")
        # Fallback to static image
        hts3d_image_path = SCRIPT_DIR / "nlrp3" / "panel_A_nlrp3_alone.png"
        if hts3d_image_path.exists():
            try:
                import matplotlib.image as mpimg
                hts3d_img = mpimg.imread(str(hts3d_image_path))
                img_height, img_width = hts3d_img.shape[:2]
                aspect_ratio = img_width / img_height
                hts3d_image_height = hts3d_image_width / aspect_ratio
            except Exception as e:
                print(f"Warning: Could not load fallback HTS 3D image: {e}")
    
    if t < 0.2:
        # Phase 1: 2D structure (reduced pause - faster start)
        phase1_t = t / 0.2
        alpha_2d = smoothstep(phase1_t)
        
        # Load and display ligand image generated by HTS3D_ligandFigure.py on left
        ligand_image_path = SCRIPT_DIR / "ligands" / "ligand_figure.png"
        cx_2d, cy_2d = 0.20, 0.50  # Moved further left from 0.25 to 0.20
        image_width, image_height = 0.45, 0.45  # 1.5x larger: 0.30 * 1.5 = 0.45
        
        if ligand_image_path.exists():
            try:
                import matplotlib.image as mpimg
                img = mpimg.imread(str(ligand_image_path))
                # Calculate bounding box for image placement (centered at cx_2d, cy_2d)
                x0 = cx_2d - image_width / 2
                y0 = cy_2d - image_height / 2
                ax.imshow(img, extent=[x0, x0 + image_width, y0, y0 + image_height], 
                         aspect='auto', zorder=5, alpha=alpha_2d * 0.95)
            except Exception as e:
                print(f"Warning: Could not load ligand image: {e}")
                # Fallback to original 2D structure diagram
                draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        else:
            # Fallback to original 2D structure diagram if image doesn't exist
            print(f"Warning: Ligand image not found at {ligand_image_path}")
            draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        
        ax.text(0.20, 0.72, "Inhibitor Ligand", color=FG, fontsize=16, fontweight="bold", ha="center")  # Moved with image
        
        # Arrow pointing right - extended to connect to further apart images
        draw_arrow(ax, 0.425, 0.50, 0.495, 0.50, alpha=0.3 * alpha_2d)  # Adjusted for new positions
        
        # Show HTS 3D video frame with alpha=0 (invisible) to reserve space and avoid jump when it appears
        if video_reader is not None:
            try:
                video_frame = video_reader.get_data(0)
                # Blend video frame with background for seamless integration (even when invisible)
                video_frame = blend_video_frame_with_background(video_frame, bg_color=BG, blend_factor=0.25)
                x0 = pocket_cx - hts3d_image_width / 2
                y0 = pocket_cy - hts3d_image_height / 2
                ax.imshow(video_frame, extent=[x0, x0 + hts3d_image_width, y0, y0 + hts3d_image_height], 
                         aspect='equal', zorder=5, alpha=0.0)  # Invisible but reserves space
            except Exception as e:
                print(f"Warning: Could not read video frame: {e}")
        elif hts3d_img is not None:
            x0 = pocket_cx - hts3d_image_width / 2
            y0 = pocket_cy - hts3d_image_height / 2
            ax.imshow(hts3d_img, extent=[x0, x0 + hts3d_image_width, y0, y0 + hts3d_image_height], 
                     aspect='equal', zorder=5, alpha=0.0)  # Invisible but reserves space
        
    elif t < 0.5:
        # Phase 2: Morphing transition (faster start)
        phase2_t = (t - 0.2) / 0.3
        morph_t = smoothstep(phase2_t)
        
        # Keep 2D visible but dimmed (not fading out)
        alpha_2d = 0.3  # Dimmed but visible
        cx_2d, cy_2d = 0.20, 0.50  # Moved further left from 0.25 to 0.20
        
        # Load and display ligand image generated by HTS3D_ligandFigure.py on left
        ligand_image_path = SCRIPT_DIR / "ligands" / "ligand_figure.png"
        image_width, image_height = 0.45, 0.45  # 1.5x larger: 0.30 * 1.5 = 0.45
        
        if ligand_image_path.exists():
            try:
                import matplotlib.image as mpimg
                img = mpimg.imread(str(ligand_image_path))
                # Calculate bounding box for image placement (centered at cx_2d, cy_2d)
                x0 = cx_2d - image_width / 2
                y0 = cy_2d - image_height / 2
                ax.imshow(img, extent=[x0, x0 + image_width, y0, y0 + image_height], 
                         aspect='auto', zorder=5, alpha=alpha_2d * 0.95)
            except Exception as e:
                print(f"Warning: Could not load ligand image: {e}")
                # Fallback to original 2D structure diagram
                draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        else:
            # Fallback to original 2D structure diagram if image doesn't exist
            draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        
        # Keep 2D labels visible but dimmed
        ax.text(0.20, 0.72, "2D (Piro Art)", color=FG, fontsize=16, fontweight="bold", ha="center", alpha=alpha_2d)  # Moved with image
        
        # Arrow pointing right (visible throughout) - extended to connect to further apart images
        draw_arrow(ax, 0.425, 0.50, 0.495, 0.50, alpha=0.6)  # Adjusted for new positions
        
        # Fade in 3D much faster - use accelerated fade-in curve
        # Apply a power function to make fade-in faster (e.g., phase2_t^0.5 for faster, or use a faster smoothstep)
        fast_fade_t = smoothstep(phase2_t * 2.0)  # Multiply by 2.0 to make it fade in twice as fast
        fast_fade_t = min(1.0, fast_fade_t)  # Clamp to 1.0
        alpha_3d = fast_fade_t
        ligand_cx, ligand_cy = 0.20, 0.50  # Moved further left
        pocket_cx, pocket_cy = 0.78, 0.50  # Moved further right
        # Enhanced rotation to show 3D structure better
        ligand_theta = 2 * np.pi * t * 0.7  # Faster rotation during transition
        ligand_phi = 0.25 + 0.1 * np.sin(2 * np.pi * t * 0.5)  # Varying angle
        pocket_rot = 0.1
        
        # Display HTS 3D video frame on right using pre-calculated dimensions (consistent aspect ratio)
        if video_reader is not None:
            try:
                # Calculate which frame to show based on phase2_t (0-1) within the transition phase
                frame_index = int(phase2_t * (video_frame_count - 1)) if video_frame_count > 1 else 0
                frame_index = min(frame_index, video_frame_count - 1)  # Clamp to valid range
                video_frame = video_reader.get_data(frame_index)
                # Blend video frame with background for seamless integration
                video_frame = blend_video_frame_with_background(video_frame, bg_color=BG, blend_factor=0.25)
                x0 = pocket_cx - hts3d_image_width / 2
                y0 = pocket_cy - hts3d_image_height / 2
                ax.imshow(video_frame, extent=[x0, x0 + hts3d_image_width, y0, y0 + hts3d_image_height], 
                         aspect='equal', zorder=5, alpha=alpha_3d)
            except Exception as e:
                print(f"Warning: Could not read video frame: {e}")
                # Fallback to original 3D pocket matching diagram
                draw_3d_pocket_matching(ax, ligand_cx, ligand_cy, pocket_cx, pocket_cy,
                                      ligand_theta, ligand_phi, pocket_rot, morph_t, alpha_val=alpha_3d)
        elif hts3d_img is not None:
            x0 = pocket_cx - hts3d_image_width / 2
            y0 = pocket_cy - hts3d_image_height / 2
            ax.imshow(hts3d_img, extent=[x0, x0 + hts3d_image_width, y0, y0 + hts3d_image_height], 
                     aspect='equal', zorder=5, alpha=alpha_3d)
        else:
            # Fallback to original 3D pocket matching diagram if video/image doesn't exist
            draw_3d_pocket_matching(ax, ligand_cx, ligand_cy, pocket_cx, pocket_cy,
                                  ligand_theta, ligand_phi, pocket_rot, morph_t, alpha_val=alpha_3d)
        
        # Text label above the image
        ax.text(0.78, 0.82, "HTS 3D", color=FG, fontsize=16, fontweight="bold", ha="center")  # Above video/image (moved higher to avoid overlap)
        
    else:
        # Phase 3: 3D conformer in pocket
        phase3_t = (t - 0.5) / 0.5
        alpha_3d = 1.0
        
        # Keep 2D visible but dimmed on the left
        alpha_2d = 0.3  # Dimmed but visible
        cx_2d, cy_2d = 0.20, 0.50  # Moved further left from 0.25 to 0.20
        
        # Load and display ligand image generated by HTS3D_ligandFigure.py on left
        ligand_image_path = SCRIPT_DIR / "ligands" / "ligand_figure.png"
        image_width, image_height = 0.45, 0.45  # 1.5x larger: 0.30 * 1.5 = 0.45
        
        if ligand_image_path.exists():
            try:
                import matplotlib.image as mpimg
                img = mpimg.imread(str(ligand_image_path))
                # Calculate bounding box for image placement (centered at cx_2d, cy_2d)
                x0 = cx_2d - image_width / 2
                y0 = cy_2d - image_height / 2
                ax.imshow(img, extent=[x0, x0 + image_width, y0, y0 + image_height], 
                         aspect='auto', zorder=5, alpha=alpha_2d * 0.95)
            except Exception as e:
                print(f"Warning: Could not load ligand image: {e}")
                # Fallback to original 2D structure diagram
                draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        else:
            # Fallback to original 2D structure diagram if image doesn't exist
            draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        
        # Keep 2D labels visible but dimmed
        ax.text(0.20, 0.72, "2D (SOTA)", color=FG, fontsize=16, fontweight="bold", ha="center", alpha=alpha_2d)  # Moved with image
        
        # Arrow pointing right (visible throughout) - extended to connect to further apart images
        draw_arrow(ax, 0.425, 0.50, 0.495, 0.50, alpha=0.6)  # Adjusted for new positions
        
        ligand_cx, ligand_cy = 0.20, 0.50  # Moved further left
        pocket_cx, pocket_cy = 0.78, 0.50  # Moved further right
        # Enhanced rotation for better 3D visualization
        ligand_theta = 2 * np.pi * t * 0.6  # Faster rotation
        ligand_phi = 0.25 + 0.15 * np.sin(2 * np.pi * t * 0.4)  # Varying viewing angle
        pocket_rot = 0.1 + 0.05 * np.sin(2 * np.pi * t * 0.2)
        morph_t = 1.0  # Fully morphed
        
        # Display HTS 3D video frame on right using pre-calculated dimensions (consistent aspect ratio)
        if video_reader is not None:
            try:
                # Calculate which frame to show based on t (0-1) within the entire scene
                # Map t to video frame index, looping if needed
                frame_index = int(t * (video_frame_count - 1)) if video_frame_count > 1 else 0
                frame_index = frame_index % video_frame_count  # Loop the video
                video_frame = video_reader.get_data(frame_index)
                # Blend video frame with background for seamless integration
                video_frame = blend_video_frame_with_background(video_frame, bg_color=BG, blend_factor=0.25)
                x0 = pocket_cx - hts3d_image_width / 2
                y0 = pocket_cy - hts3d_image_height / 2
                ax.imshow(video_frame, extent=[x0, x0 + hts3d_image_width, y0, y0 + hts3d_image_height], 
                         aspect='equal', zorder=5, alpha=alpha_3d)
            except Exception as e:
                print(f"Warning: Could not read video frame: {e}")
                # Fallback to original 3D pocket matching diagram
                draw_3d_pocket_matching(ax, ligand_cx, ligand_cy, pocket_cx, pocket_cy,
                                      ligand_theta, ligand_phi, pocket_rot, morph_t, alpha_val=alpha_3d)
        elif hts3d_img is not None:
            x0 = pocket_cx - hts3d_image_width / 2
            y0 = pocket_cy - hts3d_image_height / 2
            ax.imshow(hts3d_img, extent=[x0, x0 + hts3d_image_width, y0, y0 + hts3d_image_height], 
                     aspect='equal', zorder=5, alpha=alpha_3d)
        else:
            # Fallback to original 3D pocket matching diagram if video/image doesn't exist
            draw_3d_pocket_matching(ax, ligand_cx, ligand_cy, pocket_cx, pocket_cy,
                                  ligand_theta, ligand_phi, pocket_rot, morph_t, alpha_val=alpha_3d)
        
        # Text label above the image
        ax.text(0.78, 0.82, "HTS 3D", color=FG, fontsize=16, fontweight="bold", ha="center")  # Above video/image (moved higher to avoid overlap)
        
        # Add fit quality indicator with 3D perspective - moved with image
        fit_score = 0.85 + 0.1 * np.sin(2 * np.pi * t * 0.5)
        ax.text(0.78, 0.25, f"Binding Affinity: {fit_score:.2f}", 
               color=(0.95, 0.85, 0.55), fontsize=14, fontweight="bold", ha="center")  # Moved with image
        '''
        # Add 3D rotation indicator - moved with image
        ax.text(0.78, 0.18, "Rotating 3D view", color=(0.65, 0.75, 0.85), fontsize=10, ha="center",  # Moved with image 
               style='italic', alpha=0.7)
        '''

# -----------------------------
# Scene 8: 2D to 3D Transition (Duplicate of Scene 7)
# -----------------------------
def scene8(ax, t):
    # Title centered horizontally in the frame (not left-aligned like other scenes)
    ax.text(0.50, 0.94, "HTS-3D vs 2D in the State of the Art (SOTA)", 
            color=FG, fontsize=28, fontweight="bold", va="top", ha="center")
    
    # Phase 1 (0-0.2): Show 2D structure (reduced pause)
    # Phase 2 (0.2-0.5): Morph from 2D to 3D (faster transition start)
    # Phase 3 (0.5-1.0): Show 3D conformer fitting into pocket
    
    # Load and calculate aspect ratio for HTS 3D image ONCE at the beginning
    # This ensures consistent sizing throughout all phases to avoid visual jumps
    hts3d_image_path = SCRIPT_DIR / "nlrp3" / "proteinLigand1.png"
    pocket_cx, pocket_cy = 0.78, 0.50  # Moved further right from 0.70 to 0.78
    hts3d_image_width = 0.57  # 1.5x larger: 0.38 * 1.5 = 0.57
    hts3d_image_height = 0.57  # Default, will be recalculated if image exists
    hts3d_img = None
    
    if hts3d_image_path.exists():
        try:
            import matplotlib.image as mpimg
            hts3d_img = mpimg.imread(str(hts3d_image_path))
            
            # Make background transparent to blend with dark theme
            hts3d_img = make_background_transparent(hts3d_img, color_tolerance=0.05)
            
            # Calculate aspect ratio to preserve original image proportions
            img_height, img_width = hts3d_img.shape[:2]
            aspect_ratio = img_width / img_height
            hts3d_image_height = hts3d_image_width / aspect_ratio  # Adjust height based on aspect ratio
        except Exception as e:
            print(f"Warning: Could not load HTS 3D image: {e}")
            hts3d_img = None
    
    if t < 0.2:
        # Phase 1: 2D structure (reduced pause - faster start)
        phase1_t = t / 0.2
        alpha_2d = smoothstep(phase1_t)
        
        # Load and display ligand image generated by HTS3D_ligandFigure.py on left
        ligand_image_path = SCRIPT_DIR / "ligands" / "ligand_figure.png"
        cx_2d, cy_2d = 0.20, 0.50  # Moved further left from 0.25 to 0.20
        image_width, image_height = 0.45, 0.45  # 1.5x larger: 0.30 * 1.5 = 0.45
        
        if ligand_image_path.exists():
            try:
                import matplotlib.image as mpimg
                img = mpimg.imread(str(ligand_image_path))
                # Calculate bounding box for image placement (centered at cx_2d, cy_2d)
                x0 = cx_2d - image_width / 2
                y0 = cy_2d - image_height / 2
                ax.imshow(img, extent=[x0, x0 + image_width, y0, y0 + image_height], 
                         aspect='auto', zorder=5, alpha=alpha_2d * 0.95)
            except Exception as e:
                print(f"Warning: Could not load ligand image: {e}")
                # Fallback to original 2D structure diagram
                draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        else:
            # Fallback to original 2D structure diagram if image doesn't exist
            print(f"Warning: Ligand image not found at {ligand_image_path}")
            draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        
        ax.text(0.20, 0.72, "2D (SOTA)", color=FG, fontsize=16, fontweight="bold", ha="center")  # Moved with image
        
        # Arrow pointing right - extended to connect to further apart images
        draw_arrow(ax, 0.425, 0.50, 0.495, 0.50, alpha=0.3 * alpha_2d)  # Adjusted for new positions
        
        # Show HTS 3D image with alpha=0 (invisible) to reserve space and avoid jump when it appears
        if hts3d_img is not None:
            x0 = pocket_cx - hts3d_image_width / 2
            y0 = pocket_cy - hts3d_image_height / 2
            ax.imshow(hts3d_img, extent=[x0, x0 + hts3d_image_width, y0, y0 + hts3d_image_height], 
                     aspect='equal', zorder=5, alpha=0.0)  # Invisible but reserves space
        
    elif t < 0.5:
        # Phase 2: Morphing transition (faster start)
        phase2_t = (t - 0.2) / 0.3
        morph_t = smoothstep(phase2_t)
        
        # Keep 2D visible but dimmed (not fading out)
        alpha_2d = 0.3  # Dimmed but visible
        cx_2d, cy_2d = 0.20, 0.50  # Moved further left from 0.25 to 0.20
        
        # Load and display ligand image generated by HTS3D_ligandFigure.py on left
        ligand_image_path = SCRIPT_DIR / "ligands" / "ligand_figure.png"
        image_width, image_height = 0.45, 0.45  # 1.5x larger: 0.30 * 1.5 = 0.45
        
        if ligand_image_path.exists():
            try:
                import matplotlib.image as mpimg
                img = mpimg.imread(str(ligand_image_path))
                # Calculate bounding box for image placement (centered at cx_2d, cy_2d)
                x0 = cx_2d - image_width / 2
                y0 = cy_2d - image_height / 2
                ax.imshow(img, extent=[x0, x0 + image_width, y0, y0 + image_height], 
                         aspect='auto', zorder=5, alpha=alpha_2d * 0.95)
            except Exception as e:
                print(f"Warning: Could not load ligand image: {e}")
                # Fallback to original 2D structure diagram
                draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        else:
            # Fallback to original 2D structure diagram if image doesn't exist
            draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        
        # Keep 2D labels visible but dimmed
        ax.text(0.20, 0.72, "2D (SOTA)", color=FG, fontsize=16, fontweight="bold", ha="center", alpha=alpha_2d)  # Moved with image
        
        # Arrow pointing right (visible throughout) - extended to connect to further apart images
        draw_arrow(ax, 0.425, 0.50, 0.495, 0.50, alpha=0.6)  # Adjusted for new positions
        
        # Fade in 3D much faster - use accelerated fade-in curve
        # Apply a power function to make fade-in faster (e.g., phase2_t^0.5 for faster, or use a faster smoothstep)
        fast_fade_t = smoothstep(phase2_t * 2.0)  # Multiply by 2.0 to make it fade in twice as fast
        fast_fade_t = min(1.0, fast_fade_t)  # Clamp to 1.0
        alpha_3d = fast_fade_t
        ligand_cx, ligand_cy = 0.20, 0.50  # Moved further left
        pocket_cx, pocket_cy = 0.78, 0.50  # Moved further right
        # Enhanced rotation to show 3D structure better
        ligand_theta = 2 * np.pi * t * 0.7  # Faster rotation during transition
        ligand_phi = 0.25 + 0.1 * np.sin(2 * np.pi * t * 0.5)  # Varying angle
        pocket_rot = 0.1
        
        # Display HTS 3D image on right using pre-calculated dimensions (consistent aspect ratio)
        if hts3d_img is not None:
            x0 = pocket_cx - hts3d_image_width / 2
            y0 = pocket_cy - hts3d_image_height / 2
            ax.imshow(hts3d_img, extent=[x0, x0 + hts3d_image_width, y0, y0 + hts3d_image_height], 
                     aspect='equal', zorder=5, alpha=alpha_3d)
        else:
            # Fallback to original 3D pocket matching diagram if image doesn't exist
            draw_3d_pocket_matching(ax, ligand_cx, ligand_cy, pocket_cx, pocket_cy,
                                  ligand_theta, ligand_phi, pocket_rot, morph_t, alpha_val=alpha_3d)
        
        # Keep text labels above and below the image - moved with image
        ax.text(0.78, 0.75, "HTS 3D", color=FG, fontsize=16, fontweight="bold", ha="center")  # Moved with image
        ax.text(0.78, 0.70, f"Transition: {int(morph_t*100)}%", color=(0.75, 0.78, 0.85), fontsize=12, ha="center")  # Moved with image
        
    else:
        # Phase 3: 3D conformer in pocket
        phase3_t = (t - 0.5) / 0.5
        alpha_3d = 1.0
        
        # Keep 2D visible but dimmed on the left
        alpha_2d = 0.3  # Dimmed but visible
        cx_2d, cy_2d = 0.20, 0.50  # Moved further left from 0.25 to 0.20
        
        # Load and display ligand image generated by HTS3D_ligandFigure.py on left
        ligand_image_path = SCRIPT_DIR / "ligands" / "ligand_figure.png"
        image_width, image_height = 0.45, 0.45  # 1.5x larger: 0.30 * 1.5 = 0.45
        
        if ligand_image_path.exists():
            try:
                import matplotlib.image as mpimg
                img = mpimg.imread(str(ligand_image_path))
                # Calculate bounding box for image placement (centered at cx_2d, cy_2d)
                x0 = cx_2d - image_width / 2
                y0 = cy_2d - image_height / 2
                ax.imshow(img, extent=[x0, x0 + image_width, y0, y0 + image_height], 
                         aspect='auto', zorder=5, alpha=alpha_2d * 0.95)
            except Exception as e:
                print(f"Warning: Could not load ligand image: {e}")
                # Fallback to original 2D structure diagram
                draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        else:
            # Fallback to original 2D structure diagram if image doesn't exist
            draw_2d_structure_detailed(ax, cx_2d, cy_2d, scale=1.0, alpha_val=alpha_2d)
        
        # Keep 2D labels visible but dimmed
        ax.text(0.20, 0.72, "2D (SOTA)", color=FG, fontsize=16, fontweight="bold", ha="center", alpha=alpha_2d)  # Moved with image
        
        # Arrow pointing right (visible throughout) - extended to connect to further apart images
        draw_arrow(ax, 0.425, 0.50, 0.495, 0.50, alpha=0.6)  # Adjusted for new positions
        
        ligand_cx, ligand_cy = 0.20, 0.50  # Moved further left
        pocket_cx, pocket_cy = 0.78, 0.50  # Moved further right
        # Enhanced rotation for better 3D visualization
        ligand_theta = 2 * np.pi * t * 0.6  # Faster rotation
        ligand_phi = 0.25 + 0.15 * np.sin(2 * np.pi * t * 0.4)  # Varying viewing angle
        pocket_rot = 0.1 + 0.05 * np.sin(2 * np.pi * t * 0.2)
        morph_t = 1.0  # Fully morphed
        
        # Display HTS 3D image on right using pre-calculated dimensions (consistent aspect ratio)
        if hts3d_img is not None:
            x0 = pocket_cx - hts3d_image_width / 2
            y0 = pocket_cy - hts3d_image_height / 2
            ax.imshow(hts3d_img, extent=[x0, x0 + hts3d_image_width, y0, y0 + hts3d_image_height], 
                     aspect='equal', zorder=5, alpha=alpha_3d)
        else:
            # Fallback to original 3D pocket matching diagram if image doesn't exist
            draw_3d_pocket_matching(ax, ligand_cx, ligand_cy, pocket_cx, pocket_cy,
                                  ligand_theta, ligand_phi, pocket_rot, morph_t, alpha_val=alpha_3d)
        
        # Keep text labels above and below the image - moved with image
        ax.text(0.78, 0.75, "HTS 3D", color=FG, fontsize=16, fontweight="bold", ha="center")  # Moved with image
        ax.text(0.78, 0.70, "Transition", color=(0.75, 0.78, 0.85), fontsize=12, ha="center")  # Moved with image
        
        # Add fit quality indicator with 3D perspective - moved with image
        fit_score = 0.85 + 0.1 * np.sin(2 * np.pi * t * 0.5)
        ax.text(0.78, 0.25, f"Binding Affinity: {fit_score:.2f}", 
               color=(0.95, 0.85, 0.55), fontsize=14, fontweight="bold", ha="center")  # Moved with image
        '''
        # Add 3D rotation indicator - moved with image
        ax.text(0.78, 0.18, "Rotating 3D view", color=(0.65, 0.75, 0.85), fontsize=10, ha="center",  # Moved with image 
               style='italic', alpha=0.7)
        '''
# -----------------------------
# Scene 6: Ensemble Prediction
# -----------------------------
def scene6(ax, t):
    draw_title(ax, "Ensemble Prediction", "15 models (5 folds × 3 feature selection methods)")

    # Line 1: Draw three model blocks (top row) with feature selection labels
    line1_y = 0.65
    xs = [0.20, 0.42, 0.64]
    # Updated labels to reflect 5 folds × feature selection method
    labels = ["5 folds ×\nLASSO", "5 folds ×\nPCA", "5 folds ×\nMutual Info"]
    outs = []
    for x, lab, phase in zip(xs, labels, [0.0, 0.33, 0.66]):
        ax.add_patch(plt.Rectangle((x, line1_y), 0.16, 0.16, facecolor=(0.2, 0.25, 0.35), alpha=0.9,
                                   edgecolor=(0.6, 0.8, 0.98), lw=2))
        # Center text vertically in the box
        ax.text(x + 0.08, line1_y + 0.10, lab, ha="center", va="center", color=FG, fontsize=10, fontweight="bold")
        score = 0.55 + 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * (t + phase)))
        outs.append(score)
        ax.text(x + 0.08, line1_y + 0.03, f"{score:.2f}", ha="center", va="center",
                color=(0.75, 0.9, 1.0), fontsize=12)

    # Line 2: Combine into ensemble (bottom row, centered)
    # Final prediction = average of all 15 model predictions
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

    # Updated text to reflect 15 models total
    ax.text(0.50, 0.20, "15 Models Total: Average of all predictions for robust scoring", 
            color=(0.75, 0.78, 0.85), fontsize=12, ha="center")


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

    # target y positions with extra spacing between positions 7 and 8 (Ligand_08 and Ligand_09)
    # Adjusted to maintain same total height with shorter bars (0.018 instead of 0.036)
    # Original: top bar center 0.75, bottom bar center 0.18, bar height 0.036
    # Original total height: (0.75 + 0.018) - (0.18 - 0.018) = 0.768 - 0.162 = 0.606
    # New: bar height 0.018, so bar_offset = 0.009
    # To maintain total height 0.606: (top_y + 0.009) - (bottom_y - 0.009) = 0.606
    # So: top_y - bottom_y = 0.606 - 0.018 = 0.588
    # Keep top_y = 0.75 (same top bar center), then bottom_y = 0.75 - 0.588 = 0.162
    top_y = 0.75  # Top bar center (same as before)
    bottom_y = 0.162  # Bottom bar center (adjusted to maintain total height: 0.75 - 0.588 = 0.162)
    # Adjust position_7_y proportionally: original range was 0.75 to 0.18 (0.57), new range is 0.75 to 0.162 (0.588)
    # Position 7 was at 0.42, which is (0.75 - 0.42) / (0.75 - 0.18) = 0.33 / 0.57 = 0.579 of the way down
    # New position_7_y = 0.75 - 0.579 * (0.75 - 0.162) = 0.75 - 0.579 * 0.588 = 0.75 - 0.340 = 0.410
    position_7_y = 0.75 - (0.75 - 0.42) / (0.75 - 0.18) * (0.75 - bottom_y)  # Proportionally adjusted
    gap_size = 0.12  # Extra gap between position 7 and 8
    position_8_y = position_7_y - gap_size  # Position 8 (9th ligand) - with gap
    
    # Create non-uniform spacing: compress top 8, add gap, then bottom 2
    y_positions = np.zeros(n)
    # Top 8 positions (0-7): compressed into smaller space (0.75 to 0.42)
    for i in range(8):
        y_positions[i] = top_y - i * (top_y - position_7_y) / 7.0
    # Position 8 with extra gap
    y_positions[8] = position_8_y
    # Position 9 at bottom
    y_positions[9] = bottom_y
    
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
    # Bar height reduced by 50%: 0.036 -> 0.018, so offset is 0.009
    bar_height = 0.018  # 50% of original 0.036
    bar_offset = 0.009  # Half of bar height
    
    for i in range(n):
        score = base_scores[i]
        width = maxw * score
        sorted_pos = np.where(idx_sorted == i)[0][0]
        is_top = (sorted_pos < 3)  # top 3 in sorted order
        
        # Track the bottom of the 3rd ligand (sorted index 2)
        if sorted_pos == 2:
            y_top3_bottom = y_cur[i] - bar_offset  # Bottom of the 3rd ligand bar
        
        color = (0.6, 0.95, 0.85) if is_top and blend > 0.6 else (0.75, 0.9, 1.0)
        alpha = 0.95 if is_top else 0.75

        ax.text(0.06, y_cur[i], names[i], color=FG, fontsize=12, va="center")
        ax.add_patch(plt.Rectangle((x0, y_cur[i] - bar_offset), width, bar_height, color=color, alpha=alpha))
        ax.text(x0 + width + 0.01, y_cur[i], f"{score:.2f}", color=(0.85, 0.9, 1.0), fontsize=11, va="center")
    
    # Draw text below top 3 ligands and line below the text
    if y_top3_bottom is not None:
        # Text position: positioned right at the bottom of 3rd ligand (moved higher) to avoid overlap with 4th ligand
        text_y = y_top3_bottom - 0.01  # Positioned at or slightly above the bottom of 3rd ligand bar
        ax.text(0.50, text_y, "Successfully Screened Candidates of Small Molecules above Threshold", 
               color=(0.95, 0.85, 0.55), fontsize=8, fontweight="bold",  # Reduced by 50%: 13 -> 6.5
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
    """Draw the appropriate scene based on scene index.
    
    New scene order:
    - 0: Inputs (scene1)
    - 1: Multi-Branch Encoding (scene2)
    - 2: Cross-Attention (scene4) - was scene 3
    - 3: Ensemble Prediction (scene6) - was scene 4
    - 4: Ranked Output (scene7) - was scene 5
    - 5: 2D to 3D Transition (scene5) - was scene 6
    - 6: 2D to 3D Transition duplicate (scene8) - was scene 7
    - 7: 3D Protein Pocket Matching (scene3) - was scene 2, now at end
    """
    if scene_idx == 0:
        scene1(ax, t)
    elif scene_idx == 1:
        scene2(ax, t)
    elif scene_idx == 2:
        scene4(ax, t)  # Cross-Attention scene (was scene 3)
    elif scene_idx == 3:
        scene6(ax, t)  # Ensemble prediction (was scene 4)
    elif scene_idx == 4:
        scene7(ax, t)  # Output ranking (was scene 5)
    elif scene_idx == 5:
        scene5(ax, t)  # 2D to 3D transition scene (was scene 6)
    elif scene_idx == 6:
        scene8(ax, t)  # 2D to 3D transition scene duplicate (was scene 7)
    elif scene_idx == 7:
        scene3(ax, t)  # 3D Protein Pocket Matching (was scene 2, now at end)
    else:
        ax.text(0.5, 0.5, "Unknown scene", color=FG, ha="center", va="center")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate HTS-3D explainer video animation')
    parser.add_argument('--include-scene8', action='store_true',
                        help='Include scene 8 (3D Protein Pocket Matching) at the end. Default: only generate first 7 scenes.')
    args = parser.parse_args()
    
    include_scene8 = args.include_scene8
    
    # Set total frames based on whether scene 8 is included
    total_frames = 800 if include_scene8 else 700
    num_scenes = 8 if include_scene8 else 7
    
    # Load subtitles
    subtitles = parse_srt_from_docx(SUBTITLE_PATH)
    print(f"Loaded {len(subtitles)} subtitles")
    if include_scene8:
        print("Generating all 8 scenes (including 3D Protein Pocket Matching at the end)")
    else:
        print("Generating first 7 scenes only (use --include-scene8 to add 3D Protein Pocket Matching)")
    
    writer = imageio.get_writer(OUT_PATH, fps=FPS, codec="libx264", quality=8)
    try:
        for f in range(total_frames):
            s = scene_of_frame(f, include_scene8)
            t = local_t(f, include_scene8)
            time_seconds = f / FPS  # Current time in seconds
            
            fig, ax = setup_ax()
            draw_scene(ax, s, t)
            
            # Add subtitle if available
            # Override for Scene 1: use first caption for entire scene
            if s == 0:  # Scene 1 (Inputs)
                if len(subtitles) > 0:
                    subtitle_text = subtitles[0]['text']  # Use first subtitle for entire scene 1
                    draw_subtitle(ax, subtitle_text)
            # Override for Scene 2 (Multi-Branch Encoding) with specific caption - smaller font
            elif s == 1:  # Scene 2 (Multi-Branch Encoding)
                subtitle_text = "Each ligand is encoded in 3 complementary ways: chemical semantics using ChemBERTa, 2-D geometry using RDKit 2D features, and 3-D ligand conformers using RDKit 3D conformer generation."
                draw_subtitle(ax, subtitle_text, fontsize=14)  # Reduced from 15 to 14
            # Override for Scene 3 (Cross-Attention) - forced 2 lines
            elif s == 2:  # Scene 3 (Cross-Attention) - was scene 4
                subtitle_text = "Cross-attention then links ligand features with pocket residues, \nallowing the model to focus on the most relevant molecular interactions."
                draw_subtitle(ax, subtitle_text)
            # Override for Scene 4 (Ensemble Prediction) - using "Multiple predictive models..." caption
            elif s == 3:  # Scene 4 (Ensemble Prediction) - was scene 5
                subtitle_text = "Multiple predictive models evaluate each complex and their outputs are combined through an ensemble for robust scoring"
                draw_subtitle(ax, subtitle_text)
            # Override for Scene 5 (Ranked Output) - specific caption (forced two lines)
            elif s == 4:  # Scene 5 (Ranked Output) - was scene 6
                subtitle_text = "The result is a ranked list of compounds by predicted binding affinity, \nenabling rapid selection of top candidates for downstream validation."
                draw_subtitle(ax, subtitle_text)
            # Override for Scene 6 (2D to 3D Transition) with specific caption
            elif s == 5:  # Scene 6 (2D to 3D Transition) - was scene 7
                subtitle_text = "HTS-3D can screen millions of compounds in weeks, \n many times faster and cost-effective than prior art"
                draw_subtitle(ax, subtitle_text)
            # Override for Scene 7 (2D to 3D Transition duplicate) with specific caption
            elif s == 6:  # Scene 7 (2D to 3D Transition duplicate) - was scene 8
                subtitle_text = "World's 1st platform of HTS of compounds using 3D/4Branch matching,\n a breakthrough from current SOTA of 2D architecture"
                draw_subtitle(ax, subtitle_text)
            # Override for Scene 8 (3D Protein Pocket Matching) with specific caption (forced 2 lines)
            elif s == 7:  # Scene 8 (3D Protein Pocket Matching) - was scene 2, now at end
                subtitle_text = "pocket detection: Uses geometric/concavity analysis to find cavities\nDruggability scoring: Each pocket scored 0-1 based on residue composition"
                draw_subtitle(ax, subtitle_text)
            else:
                subtitle_text = get_subtitle_at_time(subtitles, time_seconds)
                if subtitle_text:
                    draw_subtitle(ax, subtitle_text)
            
            frame = render_frame(fig)
            writer.append_data(frame)
            plt.close(fig)
            if f % 50 == 0:
                print(f"Rendered frame {f}/{total_frames} (scene {s+1}/{num_scenes})")
    finally:
        writer.close()
    print(f"Done. Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
