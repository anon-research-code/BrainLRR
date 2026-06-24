"""
visualize_rois_v5.py  —  multi-dataset ROI visualizer (CC200, 200 ROIs)

Usage:
    python visualize_rois_v5.py --dataset abide
    python visualize_rois_v5.py --dataset parkinson
    python visualize_rois_v5.py --dataset adni_nc_ad
    python visualize_rois_v5.py --dataset adni_nc_mci
    python visualize_rois_v5.py --dataset parkinson --importance-dir /path/to/dir --top-k 5
"""

import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from nilearn import plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser(description="BrainLRR ROI visualizer (all CC200 datasets)")
parser.add_argument("--dataset",
                    choices=["abide", "parkinson", "adni_nc_ad", "adni_nc_mci"],
                    required=True,
                    help="Which dataset to visualize")
parser.add_argument("--importance-dir", default=None,
                    help="Override path to directory containing mean.npy "
                         "(default: roi_importance/<dataset>)")
parser.add_argument("--top-k", type=int, default=5,
                    help="Number of top ROIs to show (default: 5)")
args = parser.parse_args()

# ============================================================
# CONFIG
# ============================================================

DATASET_TITLES = {
    "abide":       "ABIDE (ASD vs HC)",
    "parkinson":   "Parkinson's Disease (PD vs HC)",
    "adni_nc_ad":  "ADNI NC vs AD",
    "adni_nc_mci": "ADNI NC vs MCI",
}

# All four datasets use CC200 (200 ROIs) — same atlas and label file
VIZ_ROOT    = Path(__file__).parent.parent   # .../Visualization/
ATLAS_PATH  = VIZ_ROOT / "atlas/cc200_cpac_int.nii.gz"
LABELS_PATH = VIZ_ROOT / "atlas/cc200_anatomical_labels3.csv"

importance_dir = (Path(args.importance_dir)
                  if args.importance_dir
                  else VIZ_ROOT / f"results/roi_importance/{args.dataset}")
IMPORTANCE_PATH = importance_dir / "mean.npy"
OUTPUT_DIR      = VIZ_ROOT / f"results/roi_visualizations/{args.dataset}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOP_K = args.top_k

print("=" * 70)
print(f"BrainLRR ROI Visualization — {DATASET_TITLES[args.dataset]}")
print("=" * 70)
print(f"  Importance : {IMPORTANCE_PATH}")
print(f"  Output     : {OUTPUT_DIR}")
print(f"  Top-K      : {TOP_K}")

# ============================================================
# LOAD
# ============================================================

print("\n✓ Loading ROI importance")
if not IMPORTANCE_PATH.exists():
    raise FileNotFoundError(
        f"\n❌  {IMPORTANCE_PATH} not found.\n"
        f"    Run the extraction script first:\n"
        f"      python Visualization/scripts/extract_v11_importance.py --dataset {args.dataset}\n"
        f"    This requires a saved model checkpoint for the dataset."
    )
roi_importance = np.load(IMPORTANCE_PATH)
assert roi_importance.shape[0] == 200, (
    f"Expected 200 ROI scores, got {roi_importance.shape[0]}")
print(f"  Range: {roi_importance.min():.6f} – {roi_importance.max():.6f}")

print("\n✓ Loading CC200 atlas")
cc200_img  = nib.load(ATLAS_PATH)
cc200_data = np.rint(cc200_img.get_fdata()).astype(int)

print("\n✓ Loading anatomical labels")
labels_df = pd.read_csv(LABELS_PATH)

# ============================================================
# CENTROIDS (MNI)
# ============================================================

print("\n✓ Computing ROI centroids")
roi_coords = np.zeros((200, 3))
for roi in range(1, 201):
    mask = cc200_data == roi
    if mask.sum() == 0:
        continue
    vox = np.column_stack(np.where(mask))
    roi_coords[roi - 1] = nib.affines.apply_affine(
        cc200_img.affine, vox.mean(axis=0))

# ============================================================
# SELECT TOP ROIs
# ============================================================

top_indices = np.argsort(roi_importance)[-TOP_K:][::-1]
top_coords  = roi_coords[top_indices]
top_scores  = roi_importance[top_indices]

top_df = (
    pd.DataFrame({
        "ROI"       : top_indices + 1,
        "Importance": top_scores,
        "X"         : top_coords[:, 0],
        "Y"         : top_coords[:, 1],
        "Z"         : top_coords[:, 2],
    })
    .merge(labels_df, on="ROI", how="left")
)

print(f"\nTop {TOP_K} ROIs:")
print(top_df[["ROI", "Importance", "HO_Label", "AAL_Label"]])

# ============================================================
# AUTO DISPLAY LABELS (from AAL label column)
# ============================================================

def _shorten(label: str) -> str:
    if pd.isna(label) or str(label).strip() in ("", "Unlabeled / Mixed"):
        return "Unknown"
    return (str(label)
            .replace("_L", " (L)")
            .replace("_R", " (R)")
            .replace("_", " "))

display_labels = [
    _shorten(row.get("AAL_Label", ""))
    for _, row in top_df.iterrows()
]

# ============================================================
# AUTO TEXT OFFSETS — push label radially away from (0,0)
# ============================================================

def _radial_offsets(coords_xy: np.ndarray, push: float = 20.0):
    offsets = []
    for xy in coords_xy:
        norm = np.linalg.norm(xy)
        if norm < 5:          # near centre → push upward
            offsets.append((0.0, push))
        else:
            scale = push / norm
            offsets.append((xy[0] * scale, xy[1] * scale))
    return offsets

text_offsets = _radial_offsets(top_coords[:, :2], push=20.0)

# ============================================================
# COLORS
# ============================================================

_palette = [
    "#6A1B9A",  # Purple
    "#8D6E63",  # Brown
    "#FBC02D",  # Yellow
    "#1976D2",  # Blue
    "#388E3C",  # Green
    "#E53935",  # Red
    "#00ACC1",  # Cyan
    "#F57C00",  # Orange
    "#5E35B1",  # Deep purple
    "#43A047",  # Dark green
]
roi_colors = _palette[:TOP_K]

# ============================================================
# PLOT — MICCAI-STYLE SUPERIOR VIEW
# ============================================================

print("\n✓ Creating superior-view figure")
FIG_W, FIG_H = 3.5, 2.6
fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H), facecolor='white')

display = plotting.plot_glass_brain(
    None, display_mode='z', axes=ax, alpha=0.25, colorbar=False)

for coord, color in zip(top_coords, roi_colors):
    display.add_markers([coord], marker_color=color, marker_size=120, alpha=0.95)

plot_ax = display.axes['z'].ax

label_texts = []
for coord, label, color, offset in zip(top_coords, display_labels, roi_colors, text_offsets):
    x_txt = coord[0] + offset[0]
    y_txt = coord[1] + offset[1]
    txt = plot_ax.text(
        x_txt, y_txt, label,
        fontsize=11, color=color, weight='bold',
        ha='center', va='bottom',
        path_effects=[pe.withStroke(linewidth=3, foreground='white')],
        zorder=2000,
    )
    label_texts.append(txt)

# ---- de-overlap labels using their actual rendered bounding boxes ----
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
PAD_PX = 3.0
for _ in range(40):
    boxes = [t.get_window_extent(renderer) for t in label_texts]
    moved = False
    for i in range(len(label_texts)):
        for j in range(i + 1, len(label_texts)):
            bi, bj = boxes[i], boxes[j]
            if not bi.overlaps(bj):
                continue
            moved = True
            ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)  # overlap width
            oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)  # overlap height
            ci, cj = bi.x0 + bi.width / 2, bj.x0 + bj.width / 2
            cyi, cyj = bi.y0 + bi.height / 2, bj.y0 + bj.height / 2
            if ox < oy:           # cheaper to separate horizontally
                shift = ox / 2 + PAD_PX
                dxi, dyi = (-shift if ci < cj else shift), 0
                dxj, dyj = (shift if ci < cj else -shift), 0
            else:                  # separate vertically
                shift = oy / 2 + PAD_PX
                dxi, dyi = 0, (-shift if cyi < cyj else shift)
                dxj, dyj = 0, (shift if cyi < cyj else -shift)
            for txt, dx, dy in ((label_texts[i], dxi, dyi),
                                 (label_texts[j], dxj, dyj)):
                xd, yd = txt.get_position()
                px, py = plot_ax.transData.transform((xd, yd))
                xd2, yd2 = plot_ax.transData.inverted().transform((px + dx, py + dy))
                txt.set_position((xd2, yd2))
    if moved:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    else:
        break

for txt in plot_ax.texts:
    if txt.get_text() in ('L', 'R'):
        txt.set_visible(False)

x_lim = plot_ax.get_xlim()
y_lim = plot_ax.get_ylim()
plot_ax.text(x_lim[0], y_lim[0], 'L',
             fontsize=11, fontweight='bold', color='black',
             ha='left', va='bottom')
plot_ax.text(x_lim[1], y_lim[0], 'R',
             fontsize=11, fontweight='bold', color='black',
             ha='right', va='bottom')

plt.tight_layout(pad=0.3)
out_pdf = OUTPUT_DIR / f"top{TOP_K}_superior_view.pdf"
plt.savefig(out_pdf, format='pdf', bbox_inches='tight', dpi=300)
plt.close()
print(f"✓ Figure saved: {out_pdf}")

# ============================================================
# TABLE
# ============================================================

print("\n✓ Building results table")
rows = []
for rank, roi_idx in enumerate(top_indices, start=1):
    roi_number = roi_idx + 1
    coord      = roi_coords[roi_idx]
    row_label  = labels_df[labels_df["ROI"] == roi_number]

    if not row_label.empty:
        row_label  = row_label.iloc[0]
        ho_label   = row_label.get("HO_Label",         "")
        aal_label  = row_label.get("AAL_Label",          "")
        ho_ratio   = row_label.get("HO_Overlap_Ratio",  np.nan)
        aal_ratio  = row_label.get("AAL_Overlap_Ratio", np.nan)
        hemi       = row_label.get("Hemisphere",         "")
    else:
        ho_label = aal_label = hemi = ""
        ho_ratio = aal_ratio = np.nan

    rows.append({
        "Rank"             : rank,
        "ROI_Index"        : roi_number,
        "HO_Label"         : ho_label,
        "AAL_Label"        : aal_label,
        "HO_Overlap_Ratio" : ho_ratio,
        "AAL_Overlap_Ratio": aal_ratio,
        "Importance"       : roi_importance[roi_idx],
        "Hemisphere"       : hemi,
        "MNI_X"            : coord[0],
        "MNI_Y"            : coord[1],
        "MNI_Z"            : coord[2],
    })

df = pd.DataFrame(rows)
out_csv = OUTPUT_DIR / f"top{TOP_K}_table.csv"
df.to_csv(out_csv, index=False)
print(f"✓ Table saved: {out_csv}")
print(f"\nTop {TOP_K} ROI Summary ({DATASET_TITLES[args.dataset]}):")
print(df[["Rank", "ROI_Index", "HO_Label", "AAL_Label",
          "Importance", "Hemisphere"]].to_string(index=False))
