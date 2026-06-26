# Baseline (no node masking, no LRR loss)
# Usage (run from repo root):
#   python scripts/train_baseline.py --dataset abide
#   python scripts/train_baseline.py --dataset parkinson
#   python scripts/train_baseline.py --dataset adni_nc_ad
#   python scripts/train_baseline.py --dataset adni_nc_mci

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torch.optim.swa_utils as swa_utils
from omegaconf import OmegaConf, DictConfig, open_dict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import random
import os
import pandas as pd

# ===== environment and seeds =====
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ===== 1. Loss Function =====
class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=1.5, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, pred, target):
        smoothed_target = target * (1 - self.smoothing) + self.smoothing / 2
        probs = F.softmax(pred, dim=-1)
        weight = torch.tensor([1.0, self.alpha]).to(pred.device)
        pt = (probs * target).sum(dim=-1)
        log_p = (F.log_softmax(pred, dim=-1) * smoothed_target).sum(dim=-1)
        batch_weight = (target * weight).sum(dim=-1)
        loss = -batch_weight * (1 - pt) ** self.gamma * log_p
        return loss.mean()

# ===== 2. Model =====
from source.models.BNT.components import InterpretableTransformerEncoder
from source.models.BNT.ptdec import DEC

class TransPoolingEncoder(nn.Module):
    def __init__(self, input_feature_size, input_node_num, hidden_size,
                 output_node_num, pooling=True):
        super().__init__()
        self.node_embed = nn.Parameter(
            torch.randn(1, input_node_num, input_feature_size) * 0.01)
        self.ln = nn.LayerNorm(input_feature_size)
        self.transformer = InterpretableTransformerEncoder(
            d_model=input_feature_size, nhead=8, dim_feedforward=hidden_size,
            batch_first=True, dropout=0.3)
        self.pooling = pooling
        if pooling:
            in_dim = input_feature_size * input_node_num
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 256), nn.LayerNorm(256),
                nn.LeakyReLU(), nn.Linear(256, in_dim))
            self.dec = DEC(cluster_number=output_node_num,
                           hidden_dimension=input_feature_size,
                           encoder=self.encoder)

    def forward(self, x):
        x = self.ln(x + self.node_embed)
        x = self.transformer(x)
        assignment = None
        if self.pooling:
            x, assignment = self.dec(x)
        return x, assignment

class BrainNetworkTransformer(nn.Module):
    def __init__(self, config: DictConfig, num_sites):
        super().__init__()
        self.site_embed = nn.Embedding(num_sites, config.dataset.node_sz)

        sizes = config.model.sizes
        node_nums = [config.dataset.node_sz] + sizes[:-1]
        self.attention_list = nn.ModuleList([
            TransPoolingEncoder(config.dataset.node_sz, node_nums[i],
                                config.model.hidden_dim, sizes[i],
                                config.model.pooling[i])
            for i in range(len(sizes))])

        self.flatten_dim = sizes[-1] * config.dataset.node_sz
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(config.regularization.dropout),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 2))

    def forward(self, nf, site_idx, training=True):
        x = nf + self.site_embed(site_idx).unsqueeze(1)
        for atten in self.attention_list:
            x, _ = atten(x)
        return self.fc(x.reshape((x.shape[0], -1)))

# ===== 3. Trainer: OneCycleLR + SWA =====
class Trainer:
    def __init__(self, cfg, model, loaders, alpha=1.5):
        self.cfg, self.model = cfg, model
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay)
        self.criterion = BalancedFocalLoss(alpha=alpha)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=cfg.optimizer.lr,
            steps_per_epoch=len(self.train_loader),
            epochs=cfg.training.epochs, pct_start=0.3)

        self.swa_model = swa_utils.AveragedModel(model)
        self.swa_start = int(cfg.training.epochs * 0.75)
        self.swa_scheduler = swa_utils.SWALR(self.optimizer, swa_lr=1e-5)

    def train_epoch(self, epoch):
        self.model.train()
        for _, nf, label, s_idx in self.train_loader:
            nf, label, s_idx = nf.cuda(), label.float().cuda(), s_idx.cuda()
            self.optimizer.zero_grad()
            pred = self.model(nf, s_idx, training=True)
            loss = self.criterion(pred, label)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if epoch <= self.swa_start:
                self.scheduler.step()

        if epoch > self.swa_start:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()

    def evaluate(self, loader_idx=1, use_swa=False):
        eval_model = self.swa_model if use_swa else self.model
        eval_model.eval()
        loader = self.val_loader if loader_idx == 1 else self.test_loader
        labels_all, probs_all, preds_all = [], [], []
        with torch.no_grad():
            for _, nf, label, s_idx in loader:
                nf, s_idx = nf.cuda(), s_idx.cuda()
                out = eval_model(nf, s_idx, training=False)
                probs_all.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
                preds_all.extend(out.argmax(dim=1).cpu().numpy())
                labels_all.extend(label.argmax(dim=1).cpu().numpy())

        labels_all = np.array(labels_all)
        preds_all  = np.array(preds_all)
        tn, fp, fn, tp = confusion_matrix(labels_all, preds_all).ravel()
        return {
            "acc": (preds_all == labels_all).mean(),
            "auc": roc_auc_score(labels_all, probs_all),
            "sen": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "spe": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "f1":  classification_report(labels_all, preds_all,
                                         output_dict=True,
                                         zero_division=0)['macro avg']['f1-score']
        }

# ===== 4. Data loading =====
def _stratified_train_val_test_split(stratify_var, fallback_var, n):
    """70/10/20 split via two cascaded stratified splits (70/30, then 1/3:2/3 of
    the 30%). Falls back to `fallback_var` (e.g. label-only) for the second
    split if the (site,label) groups are too small."""
    stratify_var = np.asarray(stratify_var)
    fallback_var = np.asarray(fallback_var)
    dummy_X = np.zeros((n, 1))

    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    tr_idx, temp_idx = next(sss1.split(dummy_X, stratify_var))

    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=1 / 3, random_state=42)
    try:
        val_rel, test_rel = next(
            sss2.split(dummy_X[temp_idx], stratify_var[temp_idx]))
    except ValueError:
        val_rel, test_rel = next(
            sss2.split(dummy_X[temp_idx], fallback_var[temp_idx]))

    val_idx  = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    return tr_idx, val_idx, test_idx


def prepare_data(cache, dataset_name=""):
    """Compute the train/val/test split once, then derive per-site
    normalization stats and the class-imbalance weight (ALPHA) from the
    training subjects only. Returns everything needed by get_loaders so the
    split/normalization is not recomputed (or re-mutated) on every call."""
    ts, fc, labels, site = cache
    labels_np = labels.numpy()
    site = np.asarray(site)
    n = len(labels_np)

    if dataset_name.startswith("adni"):
        stratify_var = labels_np
    else:
        # stratify by site+label jointly so val/test preserve both
        # site composition and label balance
        stratify_var = np.array([f"{s}_{l}" for s, l in zip(site, labels_np)])

    try:
        tr_idx, val_idx, test_idx = _stratified_train_val_test_split(
            stratify_var, labels_np, n)
    except ValueError:
        # (site,label) groups too small even for the first split -> label-only
        tr_idx, val_idx, test_idx = _stratified_train_val_test_split(
            labels_np, labels_np, n)

    # ----- per-site normalization, fit on train subjects only -----
    fc_np = fc.numpy().copy()
    is_train = np.zeros(n, dtype=bool)
    is_train[tr_idx] = True
    for s_id in np.unique(site):
        site_mask  = (site == s_id)
        train_mask = site_mask & is_train
        if train_mask.sum() == 0:
            train_mask = is_train
        s_mean = np.mean(fc_np[train_mask], axis=0, keepdims=True)
        s_std  = np.std(fc_np[train_mask],  axis=0, keepdims=True) + 1e-8
        fc_np[site_mask] = (fc_np[site_mask] - s_mean) / s_std
    fc = torch.from_numpy(fc_np).float()

    unique_sites = list(np.unique(site))
    site_indices = torch.tensor([unique_sites.index(s) for s in site]).long()
    y_oh = F.one_hot(labels.long(), 2)

    return {
        "ts": ts, "fc": fc, "y_oh": y_oh, "site_indices": site_indices,
        "tr_idx": tr_idx, "val_idx": val_idx, "test_idx": test_idx,
        "num_sites": len(unique_sites),
        "alpha": compute_alpha(labels_np[tr_idx]),
    }


def get_loaders(cfg, prepared):
    ts, fc           = prepared["ts"], prepared["fc"]
    y_oh             = prepared["y_oh"]
    site_indices     = prepared["site_indices"]
    tr_idx, val_idx, test_idx = (
        prepared["tr_idx"], prepared["val_idx"], prepared["test_idx"])

    with open_dict(cfg):
        cfg.dataset.node_sz = fc.shape[1]
        cfg.model.sizes[0]  = fc.shape[1]

    dsets = [
        utils.TensorDataset(ts[tr_idx],   fc[tr_idx],   y_oh[tr_idx],   site_indices[tr_idx]),
        utils.TensorDataset(ts[val_idx],  fc[val_idx],  y_oh[val_idx],  site_indices[val_idx]),
        utils.TensorDataset(ts[test_idx], fc[test_idx], y_oh[test_idx], site_indices[test_idx]),
    ]
    return [utils.DataLoader(d, batch_size=cfg.dataset.batch_size,
                             shuffle=(i == 0))
            for i, d in enumerate(dsets)], prepared["num_sites"]

# ===== 5. Config =====
def build_cfg(batch_size=16):
    return OmegaConf.create({
        "dataset": {"batch_size": batch_size, "node_sz": 200},
        "model": {
            "sizes":        [200, 64],
            "pooling":      [False, True],
            "hidden_dim":   512,
        },
        "optimizer":      {"lr": 6e-5, "weight_decay": 0.03},
        "regularization": {"dropout": 0.5},
        "training":       {"epochs": 200},
    })

# ===== 6. Optuna objective =====
def objective(trial):
    cfg = build_cfg(batch_size=BATCH_SIZE)
    cfg.optimizer.lr           = trial.suggest_float("lr", 3e-5, 8e-5, log=True)
    cfg.optimizer.weight_decay = trial.suggest_float("wd", 0.01, 0.06)
    seed_everything(42)
    loaders, num_sites = get_loaders(cfg, PREPARED_DATA)
    model   = BrainNetworkTransformer(cfg, num_sites).cuda()
    trainer = Trainer(cfg, model, loaders, alpha=ALPHA)
    best_score = 0
    for epoch in range(1, 60):
        trainer.train_epoch(epoch)
        res = trainer.evaluate(1)
        score = res['auc'] * 0.5 + res['acc'] * 0.3 + res['f1'] * 0.2
        best_score = max(best_score, score)
    return best_score

# ===== 7. Dataset registry =====
DATASETS_DIR = os.environ.get("BRAINLRR_DATASETS_DIR", "./datasets")

DATASET_PATHS = {
    "abide":       f"{DATASETS_DIR}/abide_original.npy",
    "parkinson":   f"{DATASETS_DIR}/pd_combined_cc200_1.npy",
    "adni_nc_ad":  f"{DATASETS_DIR}/ADNI_NC_AD.npy",    # NC=191, AD=132,  total=323
    "adni_nc_mci": f"{DATASETS_DIR}/ADNI_NC_MCI.npy",   # NC=191, MCI=336, total=527
}
DATASET_BATCH = {
    "abide":       16,
    "parkinson":    8,
    "adni_nc_ad":   8,
    "adni_nc_mci":  8,
}
# alpha = n_majority / n_minority (inverse-frequency weighting for class 1)
def compute_alpha(labels):
    labels = np.asarray(labels)
    n0 = np.sum(labels == 0)
    n1 = np.sum(labels == 1)
    return n0 / n1

def load_cache(dataset_name):
    from source.dataset.preprocess import StandardScaler

    path = DATASET_PATHS[dataset_name]
    print(f"Loading {dataset_name} from {path}")
    data = np.load(path, allow_pickle=True).item()

    fc     = data["corr"].astype(np.float32)
    labels = data["label"].astype(int)
    site   = data["site"]

    # ── ADNI: already filtered, labels already 0/1, no timeseries ──
    if dataset_name.startswith("adni"):
        ts = np.zeros((len(fc), 1, 1), dtype=np.float32)
        print(f"  N={len(labels)}  NC={(labels==0).sum()}  pos={(labels==1).sum()}")
        return (
            torch.from_numpy(ts).float(),
            torch.from_numpy(fc).float(),
            torch.from_numpy(labels),
            site,
        )

    # ── ABIDE / Parkinson: standard dict with timeseries ─────
    ts_key  = "timeseires" if "timeseires" in data else "timeseries"
    ts_norm = StandardScaler(
        mean=np.mean(data[ts_key]),
        std=np.std(data[ts_key])
    ).transform(data[ts_key])

    return (
        torch.from_numpy(ts_norm).float(),
        torch.from_numpy(fc).float(),
        torch.from_numpy(labels),
        site,
    )

# ===== Entry point =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_PATHS.keys()),
                        default="abide")
    parser.add_argument("--n_trials", type=int, default=15,
                        help="Number of Optuna trials")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    BATCH_SIZE = DATASET_BATCH[args.dataset]

    # ── Load data ────────────────────────────────────────────
    GLOBAL_DATA_CACHE = load_cache(args.dataset)
    PREPARED_DATA = prepare_data(GLOBAL_DATA_CACHE, args.dataset)
    ALPHA = PREPARED_DATA["alpha"]
    print(f"  split sizes: train={len(PREPARED_DATA['tr_idx'])}  "
          f"val={len(PREPARED_DATA['val_idx'])}  "
          f"test={len(PREPARED_DATA['test_idx'])}")
    print(f"  ALPHA (train-only) = {ALPHA:.4f}")

    # ── Parameter count ──────────────────────────────────────
    cfg_temp = build_cfg(batch_size=BATCH_SIZE)
    loaders_temp, num_sites_temp = get_loaders(cfg_temp, PREPARED_DATA)
    model_temp = BrainNetworkTransformer(cfg_temp, num_sites_temp).cuda()
    total_params     = sum(p.numel() for p in model_temp.parameters())
    trainable_params = sum(p.numel() for p in model_temp.parameters()
                           if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    del model_temp, loaders_temp, cfg_temp
    torch.cuda.empty_cache()

    # ── Step 1: Optuna hyperparameter search ─────────────────
    print(">>> Starting hyperparameter search (Optuna)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    bp = study.best_params
    print(f"\nBest hyperparameters:")
    print(f"  lr           = {bp['lr']:.2e}")
    print(f"  weight_decay = {bp['wd']:.4f}")

    # ── Step 2: Final evaluation across seeds ────────────────
    print(f"\n{'='*60}")
    print(f" Final Evaluation  |  dataset={args.dataset.upper()}"
          f"  seeds={args.seeds}  [Baseline — no node masking]")
    print(f"{'='*60}")

    results = []
    for s in args.seeds:
        seed_everything(s)

        cfg = build_cfg(batch_size=BATCH_SIZE)
        with open_dict(cfg):
            cfg.optimizer.lr           = bp['lr']
            cfg.optimizer.weight_decay = bp['wd']
            cfg.training.epochs        = args.epochs

        loaders, num_sites = get_loaders(cfg, PREPARED_DATA)
        model   = BrainNetworkTransformer(cfg, num_sites).cuda()
        trainer = Trainer(cfg, model, loaders, alpha=ALPHA)

        for epoch in range(1, cfg.training.epochs + 1):
            trainer.train_epoch(epoch)

        test_res = trainer.evaluate(loader_idx=2, use_swa=True)
        results.append(test_res)

        print(f"Seed {s} | "
              f"ACC: {test_res['acc']:.4f} | "
              f"AUC: {test_res['auc']:.4f} | "
              f"SEN: {test_res['sen']:.4f} | "
              f"SPE: {test_res['spe']:.4f} | "
              f"F1:  {test_res['f1']:.4f}")

    # ── Summary ──────────────────────────────────────────────
    df = pd.DataFrame(results)
    print(f"\n{'─'*40}")
    print(f" Summary ({args.dataset.upper()}, n={len(args.seeds)} seeds)  [Baseline]")
    print(f"{'─'*40}")
    for m in ["acc", "auc", "sen", "spe", "f1"]:
        print(f"  {m.upper():5}: {df[m].mean():.4f} ± {df[m].std():.4f}")

    out_csv = f"results_{args.dataset}_baseline.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")
