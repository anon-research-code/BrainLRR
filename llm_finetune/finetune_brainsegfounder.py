# Fine-tune BrainSegFounder (smilelab/BrainSegFounder) on brain FC datasets —
# encoder fine-tune (full / linear-probe / last-k) + linear classification
# head.
#
# Supports: abide | parkinson | adni_nc_ad | adni_nc_mci  (same data/splits as
# finetune_llama.py)

# Usage:
#   python finetune_brainsegfounder.py --dataset abide
#   python finetune_brainsegfounder.py --dataset parkinson --seeds 42 43 44 45 46
#   python finetune_brainsegfounder.py --dataset adni_nc_ad --epochs 15
#   python finetune_brainsegfounder.py --dataset adni_nc_mci
#   python finetune_brainsegfounder.py --dataset parkinson --finetune_mode linear
#   python finetune_brainsegfounder.py --dataset parkinson --finetune_mode last_k --unfreeze_last_k 1

import argparse
import importlib.util
import math
import os
import sys
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score

from finetune_llama import (
    DATASET_CFG, load_data, get_split, seed_everything,
    youden_threshold, compute_metrics, count_params,
)


# ===== FC matrix -> image tensor =====

class FCImageDataset(Dataset):
    """Converts each subject's FC matrix into a clamped, resized square
    tensor, the input format expected by BrainSegFounderEncoder."""

    def __init__(self, mats, labels, indices, image_size=96):
        self.mats = mats
        self.labels = labels
        self.indices = np.asarray(indices)
        self.image_size = image_size

    def __len__(self):
        return len(self.indices)

    def _mat_to_image(self, mat):
        x = torch.tensor(mat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        # fc_norm is already z-scored (train-fit, mean~0 std~1, heavy tails to
        # ~+-13). Just clamp to +-3 (~99.7% of a normal dist) and use as-is —
        # zero-centered, unit-scale, no further rescaling. Per-subject min-max
        # would destroy both absolute connectivity strength and the train-fit
        # z-score normalization; remapping to [0,1] would shift the mean away
        # from the ~0-centered range these encoders' patch embeddings expect.
        x = torch.clamp(x, -3.0, 3.0)
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        # No ImageNet mean/std normalization here: BrainSegFounder's SwinViT
        # was pretrained on structural MRI volumes, not natural images, so
        # those per-channel statistics don't apply. Keep the clamped,
        # zero-centered z-score values as-is.
        return x.squeeze(0)  # [1,H,W]

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        x = self._mat_to_image(self.mats[real_idx])
        y = int(self.labels[real_idx])
        return x, y, real_idx


# ===== BrainSegFounder loading =====

def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BrainSegFounderEncoder(nn.Module):
    """Wraps BrainSegFounder's pretrained 3D Swin (swinViT) encoder.
    Turns a 2D FC-image into a shallow synthetic 3D volume (1 channel) and
    pools the last-stage feature map into a single embedding per sample."""

    def __init__(self, ssl_head, volume_size=96):
        super().__init__()
        self.ssl_head = ssl_head
        self.volume_size = int(volume_size)

    def forward(self, pixel_values=None, x=None):
        xb = pixel_values if pixel_values is not None else x
        if xb is None:
            raise ValueError("Expected pixel_values or x.")
        img = F.interpolate(xb, size=(self.volume_size, self.volume_size),
                            mode="bilinear", align_corners=False)
        vol = img.unsqueeze(2).repeat(1, 1, self.volume_size, 1, 1)
        feats = self.ssl_head.swinViT(vol.contiguous())
        feat = feats[-1] if isinstance(feats, (list, tuple)) else feats
        return feat.flatten(2).mean(dim=-1)


def load_brainsegfounder_encoder(model_id="", volume_size=96, device=None):
    if device is None:
        device = select_device()

    repo_id = model_id or "smilelab/BrainSegFounder"
    ssl_head_path = hf_hub_download(repo_id=repo_id, filename="SSL_Head.py")
    weights_path  = hf_hub_download(repo_id=repo_id, filename="model_weights_UKB-pretrain.pt")

    spec = importlib.util.spec_from_file_location("brainsegfounder_ssl_head", ssl_head_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import BrainSegFounder SSL_Head.py from {ssl_head_path}")
    ssl_head_dir = os.path.dirname(os.path.abspath(ssl_head_path))
    if ssl_head_dir not in sys.path:
        sys.path.insert(0, ssl_head_dir)
    module = importlib.util.module_from_spec(spec)
    sys.modules["brainsegfounder_ssl_head"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "SSLHead"):
        raise RuntimeError("BrainSegFounder SSL_Head.py does not define SSLHead.")

    args = Namespace(
        in_channels=1,
        feature_size=48,
        bottleneck_depth=768,
        num_swin_blocks_per_stage=[2, 2, 2, 2],
        num_heads_per_stage=[3, 6, 12, 24],
        dropout_path_rate=0.0,
        use_checkpoint=True,
        spatial_dims=3,
    )
    ssl_head = module.SSLHead(args)

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ["state_dict", "model", "net", "module"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected BrainSegFounder checkpoint format: {type(state)}")
    cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}

    model_keys = set(ssl_head.state_dict().keys())
    matched = len(model_keys.intersection(cleaned.keys()))
    if matched == 0:
        raise RuntimeError(
            "BrainSegFounder checkpoint did not match SSLHead parameter names. "
            "Refusing to run with randomly initialized weights."
        )
    missing, unexpected = ssl_head.load_state_dict(cleaned, strict=False)
    print(f"[BrainSegFounder] loaded {repo_id}; matched_keys={matched} "
          f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    encoder = BrainSegFounderEncoder(ssl_head, volume_size=volume_size)
    return encoder.to(device), repo_id


def configure_encoder_trainable(encoder, mode="full", unfreeze_last_k=1):
    """Set requires_grad on the swinViT encoder's parameters according to
    `mode`.

    mode="full"   — train every encoder parameter (current default behavior).
    mode="linear" — freeze the whole encoder; only the classification head
                    trains (tests pretrained representation quality).
    mode="last_k" — freeze everything except the last `unfreeze_last_k`
                    swinViT stages (out of layers1..layers4) — a safer
                    middle ground between linear probe and full fine-tune.
    """
    if mode == "full":
        for p in encoder.parameters():
            p.requires_grad = True
        return

    for p in encoder.parameters():
        p.requires_grad = False

    if mode == "linear":
        return

    if mode == "last_k":
        if not (1 <= unfreeze_last_k <= 4):
            raise ValueError("--unfreeze_last_k must be between 1 and 4")
        stages = [encoder.ssl_head.swinViT.layers1, encoder.ssl_head.swinViT.layers2,
                  encoder.ssl_head.swinViT.layers3, encoder.ssl_head.swinViT.layers4]
        for stage in stages[-unfreeze_last_k:]:
            for p in stage.parameters():
                p.requires_grad = True
        return

    raise ValueError(f"Unknown finetune_mode: {mode}")


# ===== Model =====

class BrainSegFounderClassifier(nn.Module):
    def __init__(self, encoder, in_features, num_classes=2, finetune_mode="full"):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(in_features, num_classes)
        self.finetune_mode = finetune_mode

    def train(self, mode=True):
        super().train(mode)
        if mode and self.finetune_mode == "linear":
            # Keep the frozen encoder in eval mode (no dropout noise) even
            # when model.train() is called for the trainable head.
            self.encoder.eval()
        return self

    def forward(self, pixel_values):
        emb = self.encoder(pixel_values=pixel_values)
        return self.head(emb)


# ===== Fine-tuning =====

def finetune_brainsegfounder(fc_norm, labels, tr_idx, val_idx,
                             epochs=10, batch_size=8, lr=1e-4,
                             model_id="", volume_size=96,
                             finetune_mode="full", unfreeze_last_k=1):
    device = select_device()
    encoder, _ = load_brainsegfounder_encoder(model_id, volume_size=volume_size, device=device)

    configure_encoder_trainable(encoder, mode=finetune_mode,
                                 unfreeze_last_k=unfreeze_last_k)
    encoder.train()

    image_size = volume_size

    with torch.no_grad():
        dummy = torch.zeros(1, 1, image_size, image_size, device=device)
        emb_dim = encoder(pixel_values=dummy).shape[1]

    model = BrainSegFounderClassifier(encoder, emb_dim, finetune_mode=finetune_mode).to(device)
    param_info = count_params(model)

    train_ds = FCImageDataset(fc_norm, labels, tr_idx, image_size=image_size)
    val_ds   = FCImageDataset(fc_norm, labels, val_idx, image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    params_to_train = [p for p in model.parameters() if p.requires_grad]
    opt     = torch.optim.AdamW(params_to_train, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_state   = None
    best_val_auc = -math.inf
    best_epoch   = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.detach())

        model.eval()
        val_true, val_score = [], []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                probs = torch.softmax(model(xb), dim=-1)[:, 1]
                val_true.extend(yb.numpy().tolist())
                val_score.extend(probs.cpu().numpy().tolist())

        try:
            val_auc = float(roc_auc_score(val_true, val_score))
        except Exception:
            val_auc = float("nan")
        val_auc_eff = val_auc if not math.isnan(val_auc) else -math.inf

        if val_auc_eff > best_val_auc:
            best_val_auc = val_auc_eff
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"  epoch={epoch} train_loss={running/len(train_loader):.4f} "
              f"val_auc={val_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        print(f"  restored best checkpoint from epoch {best_epoch} "
              f"(val_auc={best_val_auc:.4f})")

    return model, image_size, param_info


# ===== Evaluation =====

def get_probs(model, fc_norm, labels, indices, image_size, batch_size=8):
    device = next(model.parameters()).device
    ds = FCImageDataset(fc_norm, labels, indices, image_size=image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device)
            probs = torch.softmax(model(xb), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(yb.numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


# ===== Entry point =====

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        choices=list(DATASET_CFG.keys()),
                        required=True,
                        help="Dataset to fine-tune on")
    parser.add_argument("--model_id",   type=str, default="",
                        help="HuggingFace model ID for BrainSegFounder "
                             "(default: smilelab/BrainSegFounder)")
    parser.add_argument("--seeds",      nargs="+", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--volume_size", type=int,  default=96,
                        help="Synthetic 3D volume side length fed to swinViT")
    parser.add_argument("--finetune_mode", choices=["linear", "last_k", "full"],
                        default="full",
                        help="linear: freeze swinViT, train head only. "
                             "last_k: unfreeze last --unfreeze_last_k "
                             "swinViT stages (out of layers1..layers4). "
                             "full: train all encoder params (default, "
                             "current behavior).")
    parser.add_argument("--unfreeze_last_k", type=int, default=1,
                        help="Number of trailing swinViT stages to unfreeze "
                             "when --finetune_mode=last_k.")
    args = parser.parse_args()

    cfg         = DATASET_CFG[args.dataset]
    stratify_by = cfg["stratify_by"]

    # ── Load data ────────────────────────────────────────────
    fc, labels, site = load_data(args.dataset)
    print(f"  N={len(labels)}  pos={cfg['label_pos']}={(labels==1).sum()}  "
          f"neg={cfg['label_neg']}={(labels==0).sum()}")

    tr_idx, val_idx, test_idx, fc_norm = get_split(fc, labels, site, stratify_by)
    print(f"  Train={len(tr_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

    # ── Multi-seed evaluation ────────────────────────────────
    results = []

    for seed in args.seeds:
        seed_everything(seed)
        print(f"\n{'='*60}")
        print(f" Seed {seed}  [BrainSegFounder fine-tuned  dataset={args.dataset}]")
        print(f"{'='*60}")

        model, image_size, (total_params, trainable_params) = finetune_brainsegfounder(
            fc_norm, labels, tr_idx, val_idx,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            model_id    = args.model_id,
            volume_size = args.volume_size,
            finetune_mode   = args.finetune_mode,
            unfreeze_last_k = args.unfreeze_last_k,
        )

        print(f"Calibrating threshold on val set ({len(val_idx)} samples)...")
        val_labels, val_probs = get_probs(model, fc_norm, labels, val_idx,
                                          image_size, args.batch_size)
        threshold = youden_threshold(val_labels, val_probs)
        print(f"  threshold (Youden's J) = {threshold:.4f}")

        print(f"Evaluating on test set ({len(test_idx)} samples)...")
        test_labels, test_probs = get_probs(model, fc_norm, labels, test_idx,
                                            image_size, args.batch_size)
        res = compute_metrics(test_labels, test_probs, threshold=threshold)
        res["seed"]             = seed
        res["threshold"]        = threshold
        res["dataset"]          = args.dataset
        res["total_params"]     = total_params
        res["trainable_params"] = trainable_params
        res["finetune_mode"]    = args.finetune_mode
        results.append(res)

        print(f"  ACC={res['acc']:.4f}  AUC={res['auc']:.4f}  "
              f"SEN={res['sen']:.4f}  SPE={res['spe']:.4f}")

        del model
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f" SUMMARY — BrainSegFounder fine-tuned  dataset={args.dataset}  "
          f"({len(args.seeds)} seeds)")
    print(f"{'='*60}")
    for m in ["acc", "auc", "sen", "spe"]:
        print(f"  {m.upper():5}: {df[m].mean():.4f} ± {df[m].std():.4f}")

    os.makedirs("results", exist_ok=True)
    out_csv = f"results/results_{args.dataset}_brainsegfounder_finetune.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    print(f"\nModel parameters: total={total_params/1e6:.2f}M  "
          f"trainable={trainable_params/1e6:.2f}M "
          f"({100*trainable_params/total_params:.2f}%)")

    print("\nTable row (×100, mean ± std):")
    print(f"  Medical FM | BrainSegFounder | fine-tuned | {args.dataset} | "
          f"{df['auc'].mean()*100:.2f} ± {df['auc'].std()*100:.2f} | "
          f"{df['acc'].mean()*100:.2f} ± {df['acc'].std()*100:.2f} | "
          f"{df['sen'].mean()*100:.2f} ± {df['sen'].std()*100:.2f} | "
          f"{df['spe'].mean()*100:.2f} ± {df['spe'].std()*100:.2f} | "
          f"params={total_params/1e6:.2f}M (trainable={trainable_params/1e6:.2f}M)")
