"""
ADNI dataset loader — ADNI_CC200.npy (659 subjects, 200 ROIs, CC200 atlas).

Label mapping in file:  0=AD, 1=EMCI, 2=LMCI, 3=NC

Two binary subsets:
    nc_vs_ad  : NC(3) vs AD(0)          → 191 + 132 = 323 subjects
    nc_vs_mci : NC(3) vs MCI(1+2)       → 191 + 336 = 527 subjects

Set cfg.dataset.label_mode to 'nc_vs_ad' or 'nc_vs_mci'.
"""
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from .preprocess import StandardScaler


def load_adni_data(cfg: DictConfig):
    data = np.load(cfg.dataset.path, allow_pickle=True).item()

    ts  = data["timeseires"] if "timeseires" in data else None
    fc  = data["corr"]                       # (659, 200, 200)
    raw_labels = data["label"].astype(int)   # 0=AD 1=EMCI 2=LMCI 3=NC
    site = data["site"]

    label_mode = getattr(cfg.dataset, "label_mode", "nc_vs_ad")

    if label_mode == "nc_vs_ad":
        keep   = (raw_labels == 3) | (raw_labels == 0)
        fc, raw_labels, site = fc[keep], raw_labels[keep], site[keep]
        if ts is not None: ts = ts[keep]
        # remap: NC=3→0, AD=0→1
        labels = np.where(raw_labels == 3, 0, 1)

    elif label_mode == "nc_vs_mci":
        keep   = (raw_labels == 3) | (raw_labels == 1) | (raw_labels == 2)
        fc, raw_labels, site = fc[keep], raw_labels[keep], site[keep]
        if ts is not None: ts = ts[keep]
        # remap: NC=3→0, EMCI/LMCI→1
        labels = np.where(raw_labels == 3, 0, 1)

    else:
        raise ValueError(f"Unknown label_mode '{label_mode}'. "
                         "Choose 'nc_vs_ad' or 'nc_vs_mci'.")

    # timeseries normalization (use fc as proxy if no ts)
    if ts is not None:
        scaler = StandardScaler(mean=np.mean(ts), std=np.std(ts))
        ts = scaler.transform(ts)
        ts_tensor = torch.from_numpy(np.array(ts)).float()
    else:
        ts_tensor = torch.zeros(len(fc), 1, 1)

    fc_tensor     = torch.from_numpy(np.array(fc)).float()
    labels_tensor = torch.from_numpy(np.array(labels)).float()

    with open_dict(cfg):
        cfg.dataset.node_sz          = fc_tensor.shape[1]
        cfg.dataset.node_feature_sz  = fc_tensor.shape[2]
        cfg.dataset.timeseries_sz    = ts_tensor.shape[-1]
        cfg.dataset.num_classes      = 2

    return ts_tensor, fc_tensor, labels_tensor, site
