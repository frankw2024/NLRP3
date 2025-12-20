"""
HTS-3D 30s explainer video (matplotlib) — frame-by-frame generator

Creates: hts3d_explainer.mp4 (30 seconds @ 10 fps by default)
Scenes (5s each):
  1) Inputs (SMILES -> molecule; protein silhouette)
  2) Ligand encoding (ChemBERTa embeddings + RDKit 3D)
  3) Pocket detection (protein mesh + highlighted pocket)
  4) Cross-attention (ligand nodes -> pocket residues with pulsing beams)
  5) Ensemble prediction (3 models -> ensemble score)
  6) Output ranking (bars sort; top hits highlight)

Install deps:
  pip install matplotlib numpy imageio imageio-ffmpeg

Run:
  python hts3d_explainer.py
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for rendering
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from docx import Document
import re


# -----------------------------
# Config
# -----------------------------
FPS = 10
DURATION_S = 30
W, H = 1280, 720
DPI = 100
FRAMES = FPS * DURATION_S

SCENE_LEN_S = 5
SCENE_FRAMES = FPS * SCENE_LEN_S  # 50 frames/scene

OUT_PATH = "hts3d_explainer.mp4"
SUBTITLE_PATH = "codeGen/aniVsubtitle.docx"

BG = (0.05, 0.06, 0.08)  # dark background
FG = (0.92, 0.94, 0.97)  # near-white text


# -----------------------------
# Helpers
# -----------------------------
def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def scene_of_frame(f):
    return f // SCENE_FRAMES  # 0..5


def local_t(f):
    """0..1 within scene"""
    return (f % SCENE_FRAMES) / (SCENE_FRAMES - 1)


def setup_ax():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def draw_title(ax, title, subtitle=None):
    ax.text(0.05, 0.92, title, color=FG, fontsize=28, fontweight="bold", va="top")
    if subtitle:
        ax.text(0.05, 0.875, subtitle, color=(0.75, 0.78, 0.85), fontsize=14, va="top")


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


def wrap_text(text, max_chars_per_line=70):
    """Wrap text into multiple lines that fit within the video width.
    
    Args:
        text: The text to wrap
        max_chars_per_line: Maximum characters per line (default 70 for fontsize 16)
    
    Returns:
        List of text lines
    """
    if not text:
        return []
    
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        # Check if adding this word would exceed the limit
        if current_length + word_length + 1 > max_chars_per_line and current_line:
            # Start a new line
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            # Add word to current line
            current_line.append(word)
            current_length += word_length + (1 if current_line else 0)
    
    # Add the last line
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def draw_subtitle(ax, text, y_pos=0.08, fontsize=16):
    """Draw subtitle text at the bottom of the frame with automatic line wrapping."""
    if not text:
        return
    
    # Wrap text into multiple lines
    lines = wrap_text(text, max_chars_per_line=70)
    
    if not lines:
        return
    
    # Line height in normalized coordinates (approximately 0.03 per line for fontsize 16)
    line_height = 0.035
    
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
    """Draw a DNA-like double helix structure."""
    # Parameters for the double helix
    n_turns = 2.5
    n_points = 100
    helix_radius = 0.08
    
    # Generate helix parameters
    t_helix = np.linspace(0, n_turns * 2 * np.pi, n_points)
    
    # First strand (right-handed)
    x1 = px + helix_radius * np.cos(t_helix + rot)
    y1 = py + helix_radius * np.sin(t_helix + rot) + (t_helix - n_turns * np.pi) * helix_length / (n_turns * 2 * np.pi)
    
    # Second strand (left-handed, offset by pi)
    x2 = px + helix_radius * np.cos(t_helix + rot + np.pi)
    y2 = py + helix_radius * np.sin(t_helix + rot + np.pi) + (t_helix - n_turns * np.pi) * helix_length / (n_turns * 2 * np.pi)
    
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
    """Draw a more realistic protein structure with double helix."""
    # Draw double helix as the main structure
    draw_double_helix(ax, px, py, rot, alpha_val=alpha_val, helix_length=0.25)
    
    # Add some surrounding protein context (simplified)
    # Draw a subtle background shape to suggest protein environment
    blob_t = np.linspace(0, 2 * np.pi, 200)
    blob_r = 0.15 + 0.02 * np.sin(3 * blob_t) + 0.015 * np.cos(5 * blob_t)
    bx = px + blob_r * np.cos(blob_t + rot * 0.05)
    by = py + blob_r * np.sin(blob_t + rot * 0.05)
    ax.fill(bx, by, color=(0.25, 0.5, 0.85), alpha=alpha_val * 0.2, zorder=0)


def scene1(ax, t):
    draw_title(ax, "High-Throughput 3D Screening", "Inputs: Ligands (SMILES) + Protein Structure")

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
    rot = 2 * np.pi * t * 0.3  # Slower rotation
    draw_protein_structure(ax, px, py, rot, alpha_val=0.4)

    ax.text(0.67, 0.80, "Protein Structure", color=(0.75, 0.78, 0.85), fontsize=12)

    # Connecting arrow
    draw_arrow(ax, 0.40, 0.55, 0.58, 0.55, alpha=0.5)


# -----------------------------
# Scene 2: Ligand Encoding (2D + 3D)
# -----------------------------
def draw_3d_conformer(ax, cx, cy, theta, phi=0.3):
    """Draw a more realistic 3D molecular conformer with perspective."""
    # Base structure (benzene-like ring in 3D)
    n_atoms = 6
    ring_radius = 0.08
    
    # 3D coordinates (simplified perspective projection)
    atoms_3d = []
    for i in range(n_atoms):
        angle = 2 * np.pi * i / n_atoms
        x_3d = ring_radius * np.cos(angle)
        y_3d = ring_radius * np.sin(angle) * np.cos(phi)
        z_3d = ring_radius * np.sin(angle) * np.sin(phi)
        atoms_3d.append([x_3d, y_3d, z_3d])
    
    # Rotate around Z axis
    rot_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
    atoms_3d = np.array(atoms_3d) @ rot_z.T
    
    # Project to 2D (simple orthographic)
    atoms_2d = atoms_3d[:, :2]
    atoms_2d[:, 0] += cx
    atoms_2d[:, 1] += cy
    
    # Draw bonds
    for i in range(n_atoms):
        j = (i + 1) % n_atoms
        # Vary line width based on z-depth for 3D effect
        z_depth = (atoms_3d[i, 2] + atoms_3d[j, 2]) / 2
        alpha = 0.5 + 0.4 * (z_depth / ring_radius + 1) / 2
        lw = 1.5 + 1.5 * (z_depth / ring_radius + 1) / 2
        ax.plot([atoms_2d[i, 0], atoms_2d[j, 0]], 
               [atoms_2d[i, 1], atoms_2d[j, 1]],
               color=(0.85, 0.75, 1.0), lw=lw, alpha=alpha)
    
    # Draw atoms with size based on z-depth
    for i, (atom_2d, atom_3d) in enumerate(zip(atoms_2d, atoms_3d)):
        z_depth = atom_3d[2]
        size = 60 + 40 * (z_depth / ring_radius + 1) / 2
        alpha = 0.6 + 0.3 * (z_depth / ring_radius + 1) / 2
        ax.scatter([atom_2d[0]], [atom_2d[1]], s=size, 
                  color=(0.85, 0.75, 1.0), alpha=alpha,
                  edgecolors=(0.95, 0.85, 1.0), linewidths=1)


def scene2(ax, t):
    draw_title(ax, "Ligand Encoding", "ChemBERTa (2D semantics) + RDKit (3D conformers)")

    # Split arrow
    draw_arrow(ax, 0.20, 0.55, 0.35, 0.62, alpha=0.6)
    draw_arrow(ax, 0.20, 0.55, 0.35, 0.48, alpha=0.6)

    # More realistic molecule on left
    cx, cy = 0.18, 0.55
    draw_molecule_2d(ax, cx, cy, 0.06, 1.0)

    # Left: ChemBERTa embeddings (more realistic representation)
    ax.text(0.38, 0.78, "ChemBERTa (2D semantics)", color=FG, fontsize=14, fontweight="bold")
    bars_x0, bars_y0 = 0.38, 0.25
    n_bars = 32  # More bars for realistic embedding dimension
    # Create more realistic embedding pattern (some high, some low, some medium)
    for i in range(n_bars):
        # Mix of patterns to simulate real embeddings
        pattern1 = 0.3 + 0.4 * math.sin(2 * math.pi * (i / n_bars) + 4 * t)
        pattern2 = 0.2 + 0.3 * math.sin(2 * math.pi * (i / 7) + 2 * t)
        pattern3 = 0.1 + 0.2 * (i % 3) / 2
        h = 0.05 + 0.20 * (pattern1 * 0.4 + pattern2 * 0.3 + pattern3 * 0.3)
        
        # Color varies slightly for visual interest
        color_val = 0.7 + 0.2 * (h - 0.05) / 0.20
        ax.add_patch(plt.Rectangle((bars_x0 + i * 0.010, bars_y0), 0.007, h,
                                   color=(0.7 * color_val, 0.85 * color_val, 1.0), alpha=0.7))
    ax.add_patch(plt.Rectangle((0.36, 0.22), 0.34, 0.60, fill=False,
                               edgecolor=(0.5, 0.6, 0.8), lw=2, alpha=0.45))

    # Right: RDKit 3D conformer (more realistic 3D structure)
    ax.text(0.73, 0.78, "RDKit 3D conformers", color=FG, fontsize=14, fontweight="bold")
    cx2, cy2 = 0.78, 0.50
    theta = 2 * np.pi * t * 0.8
    phi = 0.2 + 0.2 * np.sin(2 * np.pi * t * 0.5)  # Varying viewing angle
    draw_3d_conformer(ax, cx2, cy2, theta, phi)

    ax.add_patch(plt.Rectangle((0.68, 0.22), 0.28, 0.60, fill=False,
                               edgecolor=(0.5, 0.6, 0.8), lw=2, alpha=0.45))


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


def scene3(ax, t):
    draw_title(ax, "Protein Pocket Detection", "Identify binding pocket on protein surface mesh")

    # More realistic protein structure
    px, py = 0.52, 0.50
    rot = 2 * np.pi * (0.25 + 0.1 * np.sin(2*np.pi*t)) * 0.3
    draw_protein_structure(ax, px, py, rot, alpha_val=0.25)

    # Pocket region (more realistic cavity representation)
    pocket_center = np.array([0.62, 0.56])
    pulse = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * (2.0 * t)))
    
    # Draw pocket as a concave region
    pocket_angles = np.linspace(0, 2 * np.pi, 20)
    pocket_r = 0.08 + 0.02 * np.sin(3 * pocket_angles)
    pocket_x = pocket_center[0] + pocket_r * np.cos(pocket_angles)
    pocket_y = pocket_center[1] + pocket_r * np.sin(pocket_angles)
    ax.fill(pocket_x, pocket_y, color=(0.95, 0.75, 0.4), alpha=0.2 + 0.3 * pulse)
    ax.plot(pocket_x, pocket_y, color=(0.95, 0.65, 0.35), alpha=0.6 + 0.4 * pulse, lw=2.5)

    # Realistic pocket residues with different types
    rng = np.random.default_rng(123)  # deterministic
    residue_types = ['hydrophobic', 'polar', 'charged', 'aromatic']
    n_residues = 16
    
    residues_pos = pocket_center + 0.06 * rng.normal(size=(n_residues, 2))
    for i, pos in enumerate(residues_pos):
        res_type = residue_types[i % len(residue_types)]
        draw_residue(ax, pos[0], pos[1], res_type, size=35)
    
    # Draw some connecting lines between nearby residues
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            dist = np.linalg.norm(residues_pos[i] - residues_pos[j])
            if dist < 0.08:
                ax.plot([residues_pos[i, 0], residues_pos[j, 0]],
                       [residues_pos[i, 1], residues_pos[j, 1]],
                       color=(0.7, 0.7, 0.7), alpha=0.3, lw=1)
    
    ax.text(0.06, 0.20, "Pocket Identification", color=FG, fontsize=16, fontweight="bold")


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

    # Attention beams (more realistic with varying intensities)
    weights = np.zeros((len(ligand_atoms), len(residues)))
    for i in range(len(ligand_atoms)):
        for j in range(len(residues)):
            # Create more realistic attention pattern (some strong, some weak)
            base = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(2*np.pi*(t*1.2 + (i*0.13 + j*0.09))))
            # Add some structure (certain atoms prefer certain residues)
            preference = 0.3 * np.sin((i - j) * 0.5)
            weights[i, j] = np.clip(base + preference, 0, 1)

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

    # A subtle mid-panel label
    ax.text(0.42, 0.12, "Stronger attention = brighter beam", color=(0.75, 0.78, 0.85), fontsize=12)


# -----------------------------
# Scene 5: Ensemble Prediction
# -----------------------------
def scene5(ax, t):
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
# Scene 6: Output & Ranking
# -----------------------------
def scene6(ax, t):
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
    for i in range(n):
        score = base_scores[i]
        width = maxw * score
        is_top = (np.where(idx_sorted == i)[0][0] < 3)  # top 3 in sorted order
        color = (0.6, 0.95, 0.85) if is_top and blend > 0.6 else (0.75, 0.9, 1.0)
        alpha = 0.95 if is_top else 0.75

        ax.text(0.06, y_cur[i], names[i], color=FG, fontsize=12, va="center")
        ax.add_patch(plt.Rectangle((x0, y_cur[i] - 0.018), width, 0.036, color=color, alpha=alpha))
        ax.text(x0 + width + 0.01, y_cur[i], f"{score:.2f}", color=(0.85, 0.9, 1.0), fontsize=11, va="center")

    # CSV icon-ish
    ax.add_patch(plt.Rectangle((0.83, 0.14), 0.12, 0.12, facecolor=(0.2, 0.25, 0.35), alpha=0.9,
                               edgecolor=(0.6, 0.8, 0.98), lw=2))
    ax.text(0.89, 0.20, "CSV", ha="center", va="center", color=FG, fontsize=14, fontweight="bold")
    ax.text(0.05, 0.12, "Top hits advance to downstream validation", color=(0.75, 0.78, 0.85), fontsize=12)


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
        scene4(ax, t)
    elif scene_idx == 4:
        scene5(ax, t)
    elif scene_idx == 5:
        scene6(ax, t)
    else:
        ax.text(0.5, 0.5, "Unknown scene", color=FG, ha="center", va="center")


def main():
    # Load subtitles
    subtitles = parse_srt_from_docx(SUBTITLE_PATH)
    print(f"Loaded {len(subtitles)} subtitles")
    
    writer = imageio.get_writer(OUT_PATH, fps=FPS, codec="libx264", quality=8)
    try:
        for f in range(FRAMES):
            s = scene_of_frame(f)
            t = local_t(f)
            time_seconds = f / FPS  # Current time in seconds
            
            fig, ax = setup_ax()
            draw_scene(ax, s, t)
            
            # Add subtitle if available
            subtitle_text = get_subtitle_at_time(subtitles, time_seconds)
            if subtitle_text:
                draw_subtitle(ax, subtitle_text)
            
            frame = render_frame(fig)
            writer.append_data(frame)
            plt.close(fig)
            if f % 50 == 0:
                print(f"Rendered frame {f}/{FRAMES} (scene {s+1}/6)")
    finally:
        writer.close()
    print(f"Done. Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
