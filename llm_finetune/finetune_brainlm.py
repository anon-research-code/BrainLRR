# Fine-tune BrainLM (vandijklab/brainlm) on brain FC datasets — encoder
# fine-tune (full / linear-probe / last-k) + linear classification head.
#
# Supports: abide | parkinson | adni_nc_ad | adni_nc_mci  (same data/splits as
# finetune_llama.py)
# Usage:
#   python finetune_brainlm.py --dataset abide
#   python finetune_brainlm.py --dataset parkinson --seeds 42 43 44 45 46
#   python finetune_brainlm.py --dataset adni_nc_ad --epochs 15
#   python finetune_brainlm.py --dataset adni_nc_mci
#   python finetune_brainlm.py --dataset abide --n_trials 10 --search_epochs 5
#   python finetune_brainlm.py --dataset parkinson --finetune_mode linear
#   python finetune_brainlm.py --dataset parkinson --finetune_mode last_k --unfreeze_last_k 2
#   python finetune_brainlm.py --dataset parkinson --pooling mean

import argparse
import math
import os

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

from finetune_llama import (
    DATASET_CFG, load_data, get_split, seed_everything,
    youden_threshold, compute_metrics, count_params,
)


# ===== FC matrix -> image tensor =====

class FCImageDataset(Dataset):
    """Converts each FC matrix into a 3-channel square tensor resized to the
    encoder's expected input size."""

    def __init__(self, mats, labels, indices, image_size=224):
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
        # No ImageNet mean/std normalization here: BrainLM's ViTMAE was not
        # pretrained on natural images, so those per-channel statistics don't
        # apply. Keep the [0,1]-scaled values as-is, just broadcast to 3
        # channels.
        return x.squeeze(0).repeat(3, 1, 1)  # [3,H,W]

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        x = self._mat_to_image(self.mats[real_idx])
        y = int(self.labels[real_idx])
        return x, y, real_idx


# ===== BrainLM loading =====

def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_brainlm_encoder(model_id="", device=None):
    from transformers import AutoModel
    if device is None:
        device = select_device()
    model_id = model_id or "vandijklab/brainlm"
    encoder = AutoModel.from_pretrained(model_id, subfolder="vitmae_111M",
                                        trust_remote_code=True)
    # Disable ViTMAE's random patch masking (mask_ratio=0.75 by default) so the
    # encoder sees the full image deterministically — masking is part of the
    # MAE pretraining objective, not appropriate for downstream classification.
    encoder.config.mask_ratio = 0.0
    return encoder.to(device), model_id


def infer_image_size(encoder, default_size=224):
    """Read the expected square input size from the encoder's config
    (BrainLM's ViTMAE expects 432x432, not the usual 224x224)."""
    cfg = getattr(encoder, "config", None)
    size = getattr(cfg, "image_size", None) if cfg is not None else None
    if isinstance(size, (list, tuple)):
        return int(size[-1])
    if isinstance(size, int):
        return size
    return default_size


def encode_batch(encoder, pixel_values, pooling="cls"):
    """Run the encoder and pool its output into a single embedding per sample.

    pooling="cls"  — take the CLS token (last_hidden_state[:, 0]), as in
                     BrainLM's tutorial.
    pooling="mean" — mean-pool the patch tokens (excluding CLS).
    """
    out = encoder(pixel_values=pixel_values)

    if isinstance(out, torch.Tensor):
        x = out
    elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        x = out.last_hidden_state
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    else:
        raise RuntimeError("Unable to extract embeddings from BrainLM output.")

    if x.dim() >= 3:
        if pooling == "cls":
            return x[:, 0, :]
        elif pooling == "mean":
            return x[:, 1:, :].mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
    return x.flatten(1)


def configure_encoder_trainable(encoder, mode="full", unfreeze_last_k=2):
    """Set requires_grad on the encoder's parameters according to `mode`.

    mode="full"   — train every encoder parameter (current default behavior).
    mode="linear" — freeze the whole encoder; only the classification head
                    trains (tests pretrained representation quality).
    mode="last_k" — freeze everything except the last `unfreeze_last_k`
                    transformer blocks (and the final layernorm) — a safer
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
        layers = encoder.layers  # ViTMAEModel's transformer blocks
        for layer in layers[-unfreeze_last_k:]:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(encoder, "layernorm"):
            for p in encoder.layernorm.parameters():
                p.requires_grad = True
        return

    raise ValueError(f"Unknown finetune_mode: {mode}")


# ===== Model =====

class BrainLMClassifier(nn.Module):
    def __init__(self, encoder, in_features, num_classes=2, pooling="cls",
                 finetune_mode="full"):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(in_features, num_classes)
        self.pooling = pooling
        self.finetune_mode = finetune_mode

    def train(self, mode=True):
        super().train(mode)
        if mode and self.finetune_mode == "linear":
            # Keep the frozen encoder in eval mode (no dropout noise) even
            # when model.train() is called for the trainable head.
            self.encoder.eval()
        return self

    def forward(self, pixel_values):
        emb = encode_batch(self.encoder, pixel_values, pooling=self.pooling)
        return self.head(emb)


# ===== Fine-tuning =====

def finetune_brainlm(fc_norm, labels, tr_idx, val_idx,
                     epochs=10, batch_size=16, lr=1e-4, weight_decay=0.0,
                     model_id="", pooling="cls", finetune_mode="full",
                     unfreeze_last_k=2):
    device = select_device()
    encoder, _ = load_brainlm_encoder(model_id, device=device)
    image_size = infer_image_size(encoder)

    configure_encoder_trainable(encoder, mode=finetune_mode,
                                 unfreeze_last_k=unfreeze_last_k)
    encoder.train()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        emb_dim = encode_batch(encoder, dummy, pooling=pooling).shape[1]

    model = BrainLMClassifier(encoder, emb_dim, pooling=pooling,
                              finetune_mode=finetune_mode).to(device)
    param_info = count_params(model)

    train_ds = FCImageDataset(fc_norm, labels, tr_idx, image_size=image_size)
    val_ds   = FCImageDataset(fc_norm, labels, val_idx, image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    params_to_train = [p for p in model.parameters() if p.requires_grad]
    opt     = torch.optim.AdamW(params_to_train, lr=lr, weight_decay=weight_decay)
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

def get_probs(model, fc_norm, labels, indices, image_size, batch_size=16):
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
                        help="HuggingFace model ID for BrainLM "
                             "(default: vandijklab/brainlm, vitmae_111M)")
    parser.add_argument("--seeds",      nargs="+", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--n_trials",   type=int, default=0,
                        help="If > 0, run an Optuna search for lr/weight_decay "
                             "on the first seed (val AUC objective, "
                             "--search_epochs each) before the multi-seed "
                             "evaluation, like BrainNetworkTransformer's "
                             "train_consolidatedV3_optunaV12.py.")
    parser.add_argument("--search_epochs", type=int, default=5,
                        help="Epochs per Optuna trial (kept short for speed).")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls",
                        help="How to pool encoder patch tokens into an "
                             "embedding (default: cls, as in BrainLM's "
                             "tutorial).")
    parser.add_argument("--finetune_mode", choices=["linear", "last_k", "full"],
                        default="full",
                        help="linear: freeze encoder, train head only. "
                             "last_k: unfreeze last --unfreeze_last_k "
                             "transformer blocks + final layernorm. "
                             "full: train all encoder params (default, "
                             "current behavior).")
    parser.add_argument("--unfreeze_last_k", type=int, default=2,
                        help="Number of trailing transformer blocks to "
                             "unfreeze when --finetune_mode=last_k.")
    args = parser.parse_args()

    cfg         = DATASET_CFG[args.dataset]
    stratify_by = cfg["stratify_by"]

    # ── Load data ────────────────────────────────────────────
    fc, labels, site = load_data(args.dataset)
    print(f"  N={len(labels)}  pos={cfg['label_pos']}={(labels==1).sum()}  "
          f"neg={cfg['label_neg']}={(labels==0).sum()}")

    tr_idx, val_idx, test_idx, fc_norm = get_split(fc, labels, site, stratify_by)
    print(f"  Train={len(tr_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

    # ── Optional Optuna search for lr/weight_decay ───────────
    best_lr = args.lr
    best_wd = 0.0
    if args.n_trials > 0:
        search_seed = args.seeds[0]

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            wd = trial.suggest_float("wd", 0.0, 0.1)
            seed_everything(search_seed)
            model, image_size, _ = finetune_brainlm(
                fc_norm, labels, tr_idx, val_idx,
                epochs        = args.search_epochs,
                batch_size    = args.batch_size,
                lr            = lr,
                weight_decay  = wd,
                model_id      = args.model_id,
                pooling       = args.pooling,
                finetune_mode = args.finetune_mode,
                unfreeze_last_k = args.unfreeze_last_k,
            )
            val_labels, val_probs = get_probs(model, fc_norm, labels, val_idx,
                                              image_size, args.batch_size)
            auc = roc_auc_score(val_labels, val_probs)
            del model
            torch.cuda.empty_cache()
            return auc

        print(f"\nOptuna search ({args.n_trials} trials, "
              f"{args.search_epochs} epochs each, seed={search_seed})...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)
        best_lr = study.best_params["lr"]
        best_wd = study.best_params["wd"]
        print(f"  Best lr={best_lr:.2e}  wd={best_wd:.4f}  "
              f"(val_auc={study.best_value:.4f})")

    # ── Multi-seed evaluation ────────────────────────────────
    results = []

    for seed in args.seeds:
        seed_everything(seed)
        print(f"\n{'='*60}")
        print(f" Seed {seed}  [BrainLM fine-tuned  dataset={args.dataset}]")
        print(f"{'='*60}")

        model, image_size, (total_params, trainable_params) = finetune_brainlm(
            fc_norm, labels, tr_idx, val_idx,
            epochs       = args.epochs,
            batch_size   = args.batch_size,
            lr           = best_lr,
            weight_decay = best_wd,
            model_id     = args.model_id,
            pooling      = args.pooling,
            finetune_mode = args.finetune_mode,
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
        res["pooling"]          = args.pooling
        res["finetune_mode"]    = args.finetune_mode
        results.append(res)

        print(f"  ACC={res['acc']:.4f}  AUC={res['auc']:.4f}  "
              f"SEN={res['sen']:.4f}  SPE={res['spe']:.4f}")

        del model
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f" SUMMARY — BrainLM fine-tuned  dataset={args.dataset}  "
          f"({len(args.seeds)} seeds)")
    print(f"{'='*60}")
    for m in ["acc", "auc", "sen", "spe"]:
        print(f"  {m.upper():5}: {df[m].mean():.4f} ± {df[m].std():.4f}")

    os.makedirs("results", exist_ok=True)
    out_csv = f"results/results_{args.dataset}_brainlm_finetune.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    print(f"\nModel parameters: total={total_params/1e6:.2f}M  "
          f"trainable={trainable_params/1e6:.2f}M "
          f"({100*trainable_params/total_params:.2f}%)")

    print("\nTable row (×100, mean ± std):")
    print(f"  Medical FM | BrainLM | fine-tuned | {args.dataset} | "
          f"{df['auc'].mean()*100:.2f} ± {df['auc'].std()*100:.2f} | "
          f"{df['acc'].mean()*100:.2f} ± {df['acc'].std()*100:.2f} | "
          f"{df['sen'].mean()*100:.2f} ± {df['sen'].std()*100:.2f} | "
          f"{df['spe'].mean()*100:.2f} ± {df['spe'].std()*100:.2f} | "
          f"params={total_params/1e6:.2f}M (trainable={trainable_params/1e6:.2f}M)")
