"""
Parkinson's disease dataset loader (PPMI or similar cohort).

Expected .npy file format — same dict structure as abide.npy:
    data = {
        'timeseires': np.ndarray,  # (N, n_rois, n_timepoints)
        'corr':       np.ndarray,  # (N, n_rois, n_rois)  Pearson FC matrix
        'label':      np.ndarray,  # (N,) int  0=HC, 1=PD
        'site':       np.ndarray,  # (N,) str  scanner/site identifier
    }

Binary classification: HC=0 vs PD=1.
"""
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from .preprocess import StandardScaler


def load_parkinson_data(cfg: DictConfig):
    data = np.load(cfg.dataset.path, allow_pickle=True).item()

    ts     = data["timeseires"]            # (N, ROIs, T)
    fc     = data["corr"]                  # (N, ROIs, ROIs)
    labels = data["label"].astype(int)     # 0=HC, 1=PD
    site   = data["site"]

    # Drop any subjects whose label is not 0 or 1
    keep = (labels == 0) | (labels == 1)
    if not keep.all():
        ts, fc, labels, site = ts[keep], fc[keep], labels[keep], site[keep]

    ts_std = np.std(ts)
    if ts_std > 0:
        scaler = StandardScaler(mean=np.mean(ts), std=ts_std)
        ts = scaler.transform(ts)

    ts, fc, labels = [torch.from_numpy(np.array(x)).float()
                      for x in (ts, fc, labels)]

    n_hc = float((labels == 0).sum())
    n_pd = float((labels == 1).sum())
    n_total = n_hc + n_pd
    # Inverse-frequency weights: w_c = N / (n_classes * n_c)
    w_hc = n_total / (2.0 * n_hc)
    w_pd = n_total / (2.0 * n_pd)

    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = fc.shape[1], fc.shape[2]
        cfg.dataset.timeseries_sz = ts.shape[2]
        cfg.dataset.num_classes = 2
        cfg.dataset.class_weights = [w_hc, w_pd]

    return ts, fc, labels, site
