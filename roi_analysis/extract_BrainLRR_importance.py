"""
extract_BrainLRR_importance.py  —  ROI importance from train BrainLRR checkpoints

Reads ../../saved_models/<dataset>/seed_*/model.pt (written by train_BrainLRR.py)
and writes results/roi_importance/<dataset>/{mean,std,median,all_runs}.npy

Usage (run from repo root):
    python roi_analysis/extract_BrainLRR_importance.py --dataset parkinson
    python roi_analysis/extract_BrainLRR_importance.py --dataset adni_nc_ad
    python roi_analysis/extract_BrainLRR_importance.py --dataset adni_nc_mci
    python roi_analysis/extract_BrainLRR_importance.py --dataset abide
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from omegaconf import OmegaConf, open_dict
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

VIZ_ROOT     = Path(__file__).parent.parent          # .../Visualization/
PROJECT_ROOT = VIZ_ROOT.parent                      
sys.path.insert(0, str(PROJECT_ROOT))

from source.models.BNT.components import InterpretableTransformerEncoder
from source.models.BNT.ptdec import DEC

# ─── Dataset paths  ──────────
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASET_PATHS = {
    "abide":       DATASETS_DIR / "abide_clean1.npy",
    "parkinson":   DATASETS_DIR / "pd_combined_cc200_1.npy",
    "adni_nc_ad":  DATASETS_DIR / "ADNI_NC_AD.npy",
    "adni_nc_mci": DATASETS_DIR / "ADNI_NC_MCI.npy",
}

# ─── Inline BrainNetworkTransformer —


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

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()


class BrainNetworkTransformer(nn.Module):
    def __init__(self, node_sz, sizes, pooling, hidden_dim, drop_node_p,
                 dropout, num_sites):
        super().__init__()
        self.site_embed = nn.Embedding(num_sites, node_sz)
        node_nums = [node_sz] + sizes[:-1]
        self.attention_list = nn.ModuleList([
            TransPoolingEncoder(node_sz, node_nums[i], hidden_dim, sizes[i], pooling[i])
            for i in range(len(sizes))])
        flatten_dim = sizes[-1] * node_sz
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU())
        self.classifier = nn.Linear(128, 2)

    def forward(self, nf, site_idx):
        x = nf + self.site_embed(site_idx).unsqueeze(1)
        for atten in self.attention_list:
            x, _ = atten(x)
        embed = self.fc(x.reshape((x.shape[0], -1)))
        return self.classifier(embed)

    def get_attention_weights(self):
        return [a.get_attention_weights() for a in self.attention_list]


# ─── Data loader ──────────────────────────────────────────────────────────────

def build_loaders(dataset_name, batch_size=16):
    path = DATASET_PATHS[dataset_name]
    print(f"Loading {dataset_name} from {path}")
    raw = np.load(path, allow_pickle=True).item()

    ts      = torch.from_numpy(raw["timeseires"]).float()
    fc      = torch.from_numpy(raw["corr"]).float()
    labels  = torch.from_numpy(raw["label"]).long()
    site    = raw["site"]

    # site-wise normalisation (matches training)
    fc_np = fc.numpy()
    for s_id in np.unique(site):
        mask = (site == s_id)
        s_mean = fc_np[mask].mean(axis=0, keepdims=True)
        s_std  = fc_np[mask].std(axis=0, keepdims=True) + 1e-8
        fc_np[mask] = (fc_np[mask] - s_mean) / s_std
    fc = torch.from_numpy(fc_np).float()

    unique_sites = list(np.unique(site))
    site_idx     = torch.tensor([unique_sites.index(s) for s in site]).long()
    y_oh         = F.one_hot(labels, 2)

    # stratified split matching training
    if dataset_name.startswith("adni"):
        stratify_var = labels.numpy()
    elif dataset_name == "parkinson":
        stratify_var = [f"{s}_{l}" for s, l in zip(site, labels.numpy())]
    else:
        stratify_var = site

    split = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    for tr_idx, te_idx in split.split(fc, stratify_var):
        val_size   = len(te_idx) // 3
        val_idx    = te_idx[:val_size]

    dset = utils.TensorDataset(ts[val_idx], fc[val_idx],
                                y_oh[val_idx], site_idx[val_idx])
    loader = utils.DataLoader(dset, batch_size=batch_size, shuffle=False)
    print(f"  Validation samples: {len(dset)}, batches: {len(loader)}")
    return loader, len(unique_sites)


# ─── Extraction ───────────────────────────────────────────────────────────────

def extract_importance(model, loader, device):
    model.eval()
    all_attentions = []

    with torch.no_grad():
        for _, nf, _, s_idx in tqdm(loader, desc="Extracting attention"):
            nf, s_idx = nf.to(device), s_idx.to(device)
            _ = model(nf, s_idx)

            attn_layers = model.get_attention_weights()
            batch_attns = []
            for layer_attn in attn_layers:
                if layer_attn is None:
                    continue
                if isinstance(layer_attn, list):
                    sublayer_avgs = [
                        a.mean(dim=1) if a.dim() == 4 else a
                        for a in layer_attn if a is not None
                    ]
                    if sublayer_avgs:
                        batch_attns.append(torch.stack(sublayer_avgs).mean(0))
                elif isinstance(layer_attn, torch.Tensor):
                    a = layer_attn.mean(dim=1) if layer_attn.dim() == 4 else layer_attn
                    batch_attns.append(a)

            if batch_attns:
                avg = torch.stack(batch_attns).mean(0)   # (B, n_nodes, n_nodes)
                all_attentions.append(avg.cpu().numpy())

    all_attentions = np.concatenate(all_attentions, axis=0)  # (N, R, R)
    return all_attentions.mean(axis=(0, 1))   # (R,)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        choices=list(DATASET_PATHS.keys()), required=True)
    parser.add_argument("--models-dir", default=None,
                        help="Override saved_models/<dataset> directory")
    parser.add_argument("--output-dir", default=None,
                        help="Override roi_importance/<dataset> directory")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    models_dir = (Path(args.models_dir)
                  if args.models_dir
                  else PROJECT_ROOT / f"saved_models/{args.dataset}")
    output_dir = (Path(args.output_dir)
                  if args.output_dir
                  else VIZ_ROOT / f"results/roi_importance/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_paths = sorted(models_dir.glob("seed_*/model.pt"))
    if not ckpt_paths:
        print(f"  No checkpoints found in {models_dir}")
        print(f"    Re-run training with model saving enabled:")
        print(f"      python scripts/train_BrainLRR.py --dataset {args.dataset} ...")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Found {len(ckpt_paths)} checkpoints in {models_dir}")

    all_importances = []

    for ckpt_path in ckpt_paths:
        print(f"\n{'='*60}")
        print(f"  Checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        cfg_saved = ckpt["cfg"]
        num_sites = ckpt["num_sites"]

        model = BrainNetworkTransformer(
            node_sz      = cfg_saved["node_sz"],
            sizes        = cfg_saved["sizes"],
            pooling      = cfg_saved["pooling"],
            hidden_dim   = cfg_saved["hidden_dim"],
            drop_node_p  = cfg_saved["drop_node_p"],
            dropout      = cfg_saved["dropout"],
            num_sites    = num_sites,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  Seed {ckpt['seed']} | Test AUC: {ckpt['test_metrics']['auc']:.4f}")

        loader, _ = build_loaders(args.dataset, batch_size=args.batch_size)
        roi_imp   = extract_importance(model, loader, device)

        print(f"  ROI importance: min={roi_imp.min():.6f}  max={roi_imp.max():.6f}")
        all_importances.append(roi_imp)

    all_importances = np.array(all_importances)
    mean_imp   = all_importances.mean(axis=0)
    std_imp    = all_importances.std(axis=0)
    median_imp = np.median(all_importances, axis=0)

    np.save(output_dir / "mean.npy",     mean_imp)
    np.save(output_dir / "std.npy",      std_imp)
    np.save(output_dir / "median.npy",   median_imp)
    np.save(output_dir / "all_runs.npy", all_importances)

    print(f"\n{'='*60}")
    print(f"  Done — {len(all_importances)} runs aggregated")
    print(f"   Mean importance range: [{mean_imp.min():.6f}, {mean_imp.max():.6f}]")
    print(f"   Saved to: {output_dir}/")
    print(f"\nNext: run visualization")
    print(f"  python {VIZ_ROOT / 'scripts/visualize_rois_v5.py'} --dataset {args.dataset}")


if __name__ == "__main__":
    main()
