"""
Unified loader for the clean data/ directory structure:

    data/
    ├── ABIDE/
    │   ├── fc_matrices.npy   [N, 200, 200]  float32
    │   └── labels.npy        [N]            int64
    ├── ADNI/
    │   ├── fc_matrices.npy
    │   └── labels.npy
    └── PPMI/
        ├── fc_matrices.npy
        └── labels.npy

Returns (fc_tensor, labels_tensor) ready for DataLoader.
Site info is not stored here — pass site array separately if needed.
"""

import numpy as np
import torch
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def load_dataset(name: str):
    """
    Load fc_matrices and labels for a dataset by name.

    Args:
        name: 'ABIDE', 'ADNI', or 'PPMI'

    Returns:
        fc     : torch.FloatTensor  (N, n_rois, n_rois)
        labels : torch.LongTensor   (N,)
    """
    folder = DATA_ROOT / name
    fc_path  = folder / "fc_matrices.npy"
    lbl_path = folder / "labels.npy"

    if not fc_path.exists():
        raise FileNotFoundError(
            f"{fc_path} not found. "
            f"See {folder}/README.txt for preparation instructions.")

    fc     = np.load(fc_path).astype(np.float32)
    labels = np.load(lbl_path).astype(np.int64)

    assert fc.shape[0] == labels.shape[0], \
        f"fc ({fc.shape[0]}) and labels ({labels.shape[0]}) length mismatch"
    assert fc.shape[1] == fc.shape[2], \
        f"FC matrix is not square: {fc.shape}"

    return torch.from_numpy(fc), torch.from_numpy(labels)
