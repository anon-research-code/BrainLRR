#!/usr/bin/env python3
"""
Zero-shot inference on ABIDE using GPT-OSS-20B.

Implements the ABIDE split and metrics

- Random split 70% train / 10% val / 20% test
- Stratified sampling by site
- Metrics: AUROC, accuracy, sensitivity, specificity

Notes:
- This script performs zero-shot classification with GPT-OSS-20B by
  summarizing each connectivity matrix into a compact text prompt.
- No training is performed.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from argparse import Namespace
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:
    print("ERROR: PyTorch is required.", exc)
    sys.exit(2)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:
    print("ERROR: transformers is required.", exc)
    sys.exit(2)

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

try:
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import roc_auc_score
except Exception as exc:
    print("ERROR: scikit-learn is required.", exc)
    sys.exit(2)

try:
    from torchvision import models as tv_models
except Exception:
    tv_models = None

try:
    import timm
except Exception:
    timm = None

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


@dataclass
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def load_abide(path: str):
    data = np.load(path, allow_pickle=True).item()
    required = {"label", "corr", "pcorr", "site"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys in {path}: {sorted(missing)}")
    labels = data["label"].astype(int)
    sites = data["site"]
    corr = data["corr"]
    pcorr = data["pcorr"]
    return labels, sites, corr, pcorr


def make_split(labels, sites, seed=42) -> SplitIndices:
    # Paper: random split 70/10/20, stratified by site for ABIDE
    n = len(labels)
    idx = np.arange(n)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(idx, sites))

    # Split remaining 30% into 10% val and 20% test (i.e., 1/3 and 2/3 of temp)
    temp_sites = sites[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=2.0 / 3.0, random_state=seed)
    val_sub, test_sub = next(sss2.split(temp_idx, temp_sites))
    val_idx = temp_idx[val_sub]
    test_idx = temp_idx[test_sub]

    return SplitIndices(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def summarize_connectivity(mat: np.ndarray, topk: int = 20, precision: int = 3):
    # Use upper triangle to avoid duplication
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square matrix, got {mat.shape}")
    n = mat.shape[0]
    triu = np.triu_indices(n, k=1)
    vals = mat[triu]

    stats = {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
    }

    # Top-k strongest absolute connections
    k = min(topk, vals.size)
    if k > 0:
        order = np.argpartition(np.abs(vals), -k)[-k:]
        # sort descending by absolute value
        order = order[np.argsort(-np.abs(vals[order]))]
        pairs = []
        for idx in order:
            i = int(triu[0][idx])
            j = int(triu[1][idx])
            v = float(vals[idx])
            pairs.append((i, j, v))
    else:
        pairs = []

    def fmt(x):
        return f"{x:.{precision}f}"

    lines = [
        f"Nodes: {n}",
        "Summary statistics (upper triangle):",
        f"mean={fmt(stats['mean'])}, std={fmt(stats['std'])}, min={fmt(stats['min'])}, max={fmt(stats['max'])}, median={fmt(stats['median'])}, p25={fmt(stats['p25'])}, p75={fmt(stats['p75'])}",
    ]

    if pairs:
        conn_lines = ["Top connections (ROI_i-ROI_j: value):"]
        conn_lines.extend([f"ROI_{i}-ROI_{j}: {fmt(v)}" for i, j, v in pairs])
        lines.extend(conn_lines)

    return "\n".join(lines)


def _demo_label(label_text: str):
    # Use stripped label for demonstrations (avoid double spaces)
    return label_text.lstrip()


def build_messages(summary_text: str):
    system = (
        "You are a careful medical classification assistant. "
        "Given a summary of a subject's brain functional connectivity, "
        "classify whether the subject has Autism Spectrum Disorder (ASD) "
        "or is a neurotypical Control. "
        "Respond with exactly one word: ASD or Control. "
        "Reasoning: low."
    )
    user = f"Subject summary:\n{summary_text}\nLabel:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_few_shot_examples(
    mats,
    labels,
    train_idx,
    k,
    seed,
    topk,
    label_pos="ASD",
    label_neg="Control",
):
    if k <= 0:
        return []

    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx)
    pos_idx = train_idx[labels[train_idx] == 1]
    neg_idx = train_idx[labels[train_idx] == 0]
    k_pos = k // 2
    k_neg = k - k_pos

    pick_pos = (
        rng.choice(pos_idx, size=min(k_pos, len(pos_idx)), replace=False)
        if k_pos > 0 and len(pos_idx) > 0
        else np.array([], dtype=int)
    )
    pick_neg = (
        rng.choice(neg_idx, size=min(k_neg, len(neg_idx)), replace=False)
        if k_neg > 0 and len(neg_idx) > 0
        else np.array([], dtype=int)
    )

    picks = np.concatenate([pick_pos, pick_neg]) if (len(pick_pos) + len(pick_neg)) > 0 else np.array([], dtype=int)

    examples = []
    for idx in picks:
        summary = summarize_connectivity(mats[idx], topk=topk)
        label = _demo_label(label_pos if labels[idx] == 1 else label_neg)
        examples.append((summary, label))
    return examples


def build_messages_few_shot(summary_text: str, few_shot_examples):
    system = (
        "You are a careful medical classification assistant. "
        "Given a summary of a subject's brain functional connectivity, "
        "classify whether the subject has Autism Spectrum Disorder (ASD) "
        "or is a neurotypical Control. "
        "Respond with exactly one word: ASD or Control. "
        "Reasoning: low."
    )

    demos = []
    for i, (s, lab) in enumerate(few_shot_examples, 1):
        demos.append(f"Example {i}\nSubject summary:\n{s}\nLabel: {lab}")

    demo_text = "\n\n".join(demos)
    if demo_text:
        user = (
            f"{demo_text}\n\nNow classify the next subject.\n"
            f"Subject summary:\n{summary_text}\nLabel:"
        )
    else:
        user = f"Subject summary:\n{summary_text}\nLabel:"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class AbideImageDataset(Dataset):
    def __init__(self, mats, labels, indices, image_size=224):
        self.mats = mats
        self.labels = labels
        self.indices = np.asarray(indices)
        self.image_size = image_size

    def __len__(self):
        return len(self.indices)

    def _mat_to_image(self, mat):
        # Normalize per-sample to [0, 1] to build an image
        x = torch.tensor(mat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        minv = float(x.min())
        maxv = float(x.max())
        if maxv > minv:
            x = (x - minv) / (maxv - minv)
        else:
            x = torch.zeros_like(x)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = x.squeeze(0)  # [1,H,W]
        x = x.repeat(3, 1, 1)  # [3,H,W]
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        x = (x - mean) / std
        return x

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        mat = self.mats[real_idx]
        x = self._mat_to_image(mat)
        y = int(self.labels[real_idx])
        return x, y, real_idx


def _select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def _get_vision_model(vision_model: str):
    if tv_models is None:
        raise RuntimeError("torchvision is required for vision models.")
    if vision_model == "resnet50":
        return tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    if vision_model == "resnet101":
        return tv_models.resnet101(weights=tv_models.ResNet101_Weights.IMAGENET1K_V2)
    if vision_model == "vit_b_16":
        return tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1)
    if vision_model == "vit_l_16":
        return tv_models.vit_l_16(weights=tv_models.ViT_L_16_Weights.IMAGENET1K_V1)
    raise ValueError(f"Unknown vision model: {vision_model}")


def _replace_classifier_head(model, num_classes=2):
    # Returns (head_name, in_features)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return "fc", in_features
    if hasattr(model, "heads"):
        # torchvision ViT uses heads as nn.Sequential with a "head" Linear
        if hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
            return "heads.head", in_features
        if isinstance(model.heads, nn.Linear):
            in_features = model.heads.in_features
            model.heads = nn.Linear(in_features, num_classes)
            return "heads", in_features
    raise RuntimeError("Unsupported model head for replacement.")


def _set_classifier_identity(model):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        return in_features, "fc"
    if hasattr(model, "heads"):
        if hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Identity()
            return in_features, "heads.head"
        if isinstance(model.heads, nn.Linear):
            in_features = model.heads.in_features
            model.heads = nn.Identity()
            return in_features, "heads"
    raise RuntimeError("Unsupported model head for identity.")


def _build_fixed_head(in_features, seed=0):
    torch.manual_seed(seed)
    head = nn.Linear(in_features, 2)
    nn.init.normal_(head.weight, mean=0.0, std=0.02)
    nn.init.zeros_(head.bias)
    return head


def _select_lora_target_modules(model, vision_model, head_name, max_targets=8):
    exclude_prefixes = []
    if head_name:
        exclude_prefixes.append(head_name.split(".")[0])

    candidates = []
    for name, module in model.named_modules():
        if any(name.startswith(p) for p in exclude_prefixes):
            continue
        if vision_model in ["resnet50", "resnet101"] and isinstance(module, nn.Conv2d):
            candidates.append(name)
        elif vision_model in ["vit_b_16", "vit_l_16"] and isinstance(module, nn.Linear):
            candidates.append(name)

    if not candidates:
        return []

    # Use last modules to keep LoRA budget small
    k = min(max_targets, len(candidates))
    return candidates[-k:]


def _compute_lora_rank(base_model, target_modules, target_pct):
    total_params = sum(p.numel() for p in base_model.parameters())
    target_trainable = target_pct * total_params

    # Estimate head params (classifier)
    head_params = 0
    if hasattr(base_model, "fc"):
        head_params += sum(p.numel() for p in base_model.fc.parameters())
    if hasattr(base_model, "heads"):
        head_params += sum(p.numel() for p in base_model.heads.parameters())

    # Sum (in + out) for each target module
    name_to_module = dict(base_model.named_modules())
    s = 0
    for name in target_modules:
        mod = name_to_module.get(name)
        if mod is None or not hasattr(mod, "weight"):
            continue
        w = mod.weight
        if w is None:
            continue
        if w.dim() >= 2:
            out_features = w.shape[0]
            in_features = int(np.prod(w.shape[1:]))
            s += (out_features + in_features)
    if s <= 0:
        return 1

    remaining = max(target_trainable - head_params, s)
    r = max(1, int(remaining // s))
    return r


def _apply_lora(model_factory, target_modules, target_pct):
    # Compute rank to hit target_pct as closely as possible
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError("peft is required for LoRA fine-tuning.")
    base_model = model_factory()
    total_params = sum(p.numel() for p in base_model.parameters())
    rank = _compute_lora_rank(base_model, target_modules, target_pct)
    cfg = LoraConfig(
        r=rank,
        lora_alpha=max(1, rank),
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(base_model, cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    pct = trainable / total_params
    if abs(pct - target_pct) / target_pct > 0.2:
        print(
            f"[WARN] LoRA trainable_pct={pct:.4%} differs from target {target_pct:.4%}. "
            "Consider adjusting target modules."
        )
    return peft_model, rank, pct


def run_vision_zero_shot(mats, labels, split, vision_model, batch_size=16, seed=0):
    device = _select_device()
    model = _get_vision_model(vision_model).to(device)
    model.eval()
    total_params = _count_params(model)
    in_features, _ = _set_classifier_identity(model)
    fixed_head = _build_fixed_head(in_features, seed=seed).to(device)
    fixed_head.eval()

    test_ds = AbideImageDataset(mats, labels, split.test_idx)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    y_true, y_pred, y_score = [], [], []
    records = []

    with torch.no_grad():
        for xb, yb, idxs in test_loader:
            xb = xb.to(device)
            feats = model(xb)
            logits = fixed_head(feats)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs >= 0.5).long()
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())
            for i, idx in enumerate(idxs.numpy().tolist()):
                records.append((int(idx), int(yb[i]), int(preds[i]), float(probs[i])))

    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["total_params"] = total_params
    metrics["trainable_pct"] = 0.0
    metrics["method_note"] = "frozen pretrained backbone with fixed random linear head; no ABIDE training labels used"
    return metrics, records


def run_vision_linear_probe(
    mats,
    labels,
    split,
    vision_model,
    epochs=40,
    batch_size=16,
    lr=1e-4,
    seed=0,
):
    device = _select_device()
    torch.manual_seed(seed)
    model = _get_vision_model(vision_model).to(device)
    model.eval()
    in_features, _ = _set_classifier_identity(model)
    for p in model.parameters():
        p.requires_grad = False

    head = nn.Linear(in_features, 2).to(device)
    trainable_params = sum(p.numel() for p in head.parameters())
    total_params = _count_params(model) + trainable_params

    train_ds = AbideImageDataset(mats, labels, split.train_idx)
    val_ds = AbideImageDataset(mats, labels, split.val_idx)
    test_ds = AbideImageDataset(mats, labels, split.test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_val_auc = -math.inf
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.eval()
        head.train()
        running = 0.0
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            with torch.no_grad():
                feats = model(xb)
            logits = head(feats)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.detach())

        head.eval()
        val_true, val_score = [], []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                feats = model(xb)
                logits = head(feats)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                val_true.extend(yb.numpy().tolist())
                val_score.extend(probs.cpu().numpy().tolist())
        try:
            val_auc = float(roc_auc_score(val_true, val_score))
        except Exception:
            val_auc = float("nan")
        val_auc_for_select = val_auc if not math.isnan(val_auc) else -math.inf
        if val_auc_for_select > best_val_auc:
            best_val_auc = val_auc_for_select
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        print(f"[LinearProbe] epoch={epoch} train_loss={running/len(train_loader):.4f} val_auc={val_auc:.4f}")

    if best_state is not None:
        head.load_state_dict(best_state)
        head = head.to(device)

    model.eval()
    head.eval()
    y_true, y_pred, y_score = [], [], []
    records = []
    with torch.no_grad():
        for xb, yb, idxs in test_loader:
            xb = xb.to(device)
            feats = model(xb)
            logits = head(feats)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs >= 0.5).long()
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())
            for i, idx in enumerate(idxs.numpy().tolist()):
                records.append((int(idx), int(yb[i]), int(preds[i]), float(probs[i])))

    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["trainable_pct"] = float(trainable_params / total_params)
    metrics["total_params"] = int(total_params)
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_val_auc"] = float(best_val_auc) if best_val_auc > -math.inf else float("nan")
    metrics["method_note"] = "frozen pretrained backbone with ABIDE-trained linear probe"
    return metrics, records


def run_vision_lora(
    mats,
    labels,
    split,
    vision_model,
    lora_pct=0.005,
    epochs=5,
    batch_size=16,
    lr=1e-4,
):
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError("peft is required for LoRA fine-tuning.")

    device = _select_device()
    def model_factory():
        m = _get_vision_model(vision_model)
        m.train()
        _replace_classifier_head(m, num_classes=2)
        return m

    probe_model = model_factory()
    max_targets_list = [8, 16, 24, 32, 48, 64] if vision_model == "vit_l_16" else [8, 16]
    best = None
    best_pct = None
    best_rank = None
    best_model = None

    for max_targets in max_targets_list:
        target_modules = _select_lora_target_modules(
            probe_model, vision_model, head_name="", max_targets=max_targets
        )
        if not target_modules:
            continue
        model, rank, actual_pct = _apply_lora(model_factory, target_modules, lora_pct)
        if best_pct is None or (actual_pct <= lora_pct and actual_pct > best_pct):
            best = target_modules
            best_pct = actual_pct
            best_rank = rank
            best_model = model
        if actual_pct >= lora_pct * 0.95 and actual_pct <= lora_pct:
            break

    if best_model is None:
        # As a last resort, include all eligible modules
        target_modules = _select_lora_target_modules(
            probe_model, vision_model, head_name="", max_targets=10_000
        )
        if not target_modules:
            raise RuntimeError("Could not find target modules for LoRA.")
        best_model, best_rank, best_pct = _apply_lora(model_factory, target_modules, lora_pct)
        best = target_modules

    model = best_model.to(device)
    rank = best_rank
    actual_pct = best_pct

    # Ensure classifier head is trainable
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
    if hasattr(model, "heads"):
        for p in model.heads.parameters():
            p.requires_grad = True

    train_ds = AbideImageDataset(mats, labels, split.train_idx)
    val_ds = AbideImageDataset(mats, labels, split.val_idx)
    test_ds = AbideImageDataset(mats, labels, split.test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.detach())

        # quick validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=-1)
                correct += int((preds == yb).sum())
                total += int(yb.numel())
        acc = correct / total if total else 0.0
        print(f"[LoRA] epoch={epoch} train_loss={running/len(train_loader):.4f} val_acc={acc:.4f} (rank={rank}, trainable_pct={actual_pct:.4%})")

    # test
    model.eval()
    y_true, y_pred, y_score = [], [], []
    records = []
    with torch.no_grad():
        for xb, yb, idxs in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs >= 0.5).long()
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())
            for i, idx in enumerate(idxs.numpy().tolist()):
                records.append((int(idx), int(yb[i]), int(preds[i]), float(probs[i])))

    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["lora_rank"] = int(rank)
    metrics["trainable_pct"] = float(actual_pct)
    metrics["total_params"] = _count_params(model)
    return metrics, records


def run_vision_finetune(
    mats,
    labels,
    split,
    vision_model,
    epochs=5,
    batch_size=16,
    lr=1e-4,
):
    device = _select_device()
    model = _get_vision_model(vision_model).to(device)
    model.train()
    _replace_classifier_head(model, num_classes=2)
    # Ensure new head is on the same device
    if hasattr(model, "fc") and hasattr(model.fc, "to"):
        model.fc = model.fc.to(device)
    if hasattr(model, "heads") and hasattr(model.heads, "to"):
        model.heads = model.heads.to(device)
    model = model.to(device)

    # Full finetuning: all params trainable
    for p in model.parameters():
        p.requires_grad = True

    train_ds = AbideImageDataset(mats, labels, split.train_idx)
    val_ds = AbideImageDataset(mats, labels, split.val_idx)
    test_ds = AbideImageDataset(mats, labels, split.test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_state = None
    best_val_auc = -math.inf
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.detach())

        model.eval()
        val_true, val_score = [], []
        with torch.no_grad():
            correct = 0
            total = 0
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                preds = torch.argmax(logits, dim=-1)
                correct += int((preds == yb).sum())
                total += int(yb.numel())
                val_true.extend(yb.cpu().numpy().tolist())
                val_score.extend(probs.cpu().numpy().tolist())
        acc = correct / total if total else 0.0
        try:
            val_auc = float(roc_auc_score(val_true, val_score))
        except Exception:
            val_auc = float("nan")
        val_auc_for_select = val_auc if not math.isnan(val_auc) else -math.inf
        if val_auc_for_select > best_val_auc:
            best_val_auc = val_auc_for_select
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[FT] epoch={epoch} train_loss={running/len(train_loader):.4f} val_acc={acc:.4f} val_auc={val_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        print(f"[FT] restored best validation checkpoint from epoch {best_epoch} (val_auc={best_val_auc:.4f})")

    model.eval()
    y_true, y_pred, y_score = [], [], []
    records = []
    with torch.no_grad():
        for xb, yb, idxs in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs >= 0.5).long()
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())
            for i, idx in enumerate(idxs.numpy().tolist()):
                records.append((int(idx), int(yb[i]), int(preds[i]), float(probs[i])))

    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["trainable_pct"] = 1.0
    metrics["total_params"] = _count_params(model)
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_val_auc"] = float(best_val_auc) if best_val_auc > -math.inf else float("nan")
    metrics["method_note"] = "full fine-tuning with best validation-AUROC checkpoint restored before test"
    return metrics, records


class _BrainSegFounderEncoder(nn.Module):
    def __init__(self, ssl_head, volume_size=96):
        super().__init__()
        self.ssl_head = ssl_head
        self.volume_size = int(volume_size)
        self.config = Namespace(image_size=int(volume_size))

    def forward(self, pixel_values=None, x=None):
        xb = pixel_values if pixel_values is not None else x
        if xb is None:
            raise ValueError("Expected pixel_values or x.")
        # BrainSegFounder is a 3D Swin encoder. We embed FC images by turning the
        # 2D matrix image into a shallow synthetic volume with two input channels.
        img = xb[:, :1]
        img = F.interpolate(
            img,
            size=(self.volume_size, self.volume_size),
            mode="bilinear",
            align_corners=False,
        )
        vol = img.unsqueeze(2).repeat(1, 2, self.volume_size, 1, 1)
        if hasattr(self.ssl_head, "swinViT"):
            feats = self.ssl_head.swinViT(vol.contiguous())
            feat = feats[-1] if isinstance(feats, (list, tuple)) else feats
            return feat.flatten(2).mean(dim=-1)
        out = self.ssl_head(vol)
        return _extract_embedding_from_output(out)


def _load_brainsegfounder_encoder(model_id, volume_size=96):
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is required to load BrainSegFounder.")
    try:
        import monai  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "BrainSegFounder requires MONAI. Install it in the cluster env with: "
            "pip install monai"
        ) from exc

    repo_id = model_id or "smilelab/BrainSegFounder"
    ssl_head_path = hf_hub_download(repo_id=repo_id, filename="SSL_Head.py")
    weights_path = hf_hub_download(repo_id=repo_id, filename="model_weights_UKB-pretrain.pt")

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
        in_channels=2,
        feature_size=48,
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
    cleaned = {}
    for key, val in state.items():
        new_key = key.replace("module.", "", 1)
        cleaned[new_key] = val
    model_keys = set(ssl_head.state_dict().keys())
    matched = len(model_keys.intersection(cleaned.keys()))
    if matched == 0:
        raise RuntimeError(
            "BrainSegFounder checkpoint did not match SSLHead parameter names. "
            "Refusing to run with randomly initialized weights."
        )
    missing, unexpected = ssl_head.load_state_dict(cleaned, strict=False)
    print(
        f"[BrainSegFounder] loaded {repo_id}; "
        f"matched_keys={matched} missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
    )
    return _BrainSegFounderEncoder(ssl_head, volume_size=volume_size), repo_id


def _load_sam_med2d_encoder(model_id, checkpoint_path="", repo_dir="", device=None):
    if not checkpoint_path:
        raise RuntimeError(
            "SAM-Med2D is not a Transformers AutoModel repo. To run true SAM-Med2D, "
            "provide --sam_med2d_checkpoint pointing to the downloaded SAM-Med2D "
            "checkpoint and --sam_med2d_repo_dir pointing to the cloned SAM-Med2D repo. "
            "This avoids silently reporting a generic fallback model."
        )
    if not repo_dir:
        raise RuntimeError(
            "SAM-Med2D requires --sam_med2d_repo_dir so the official model code can be imported."
        )
    repo_dir = os.path.abspath(repo_dir)
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"SAM-Med2D checkpoint not found: {checkpoint_path}")
    if not os.path.isdir(repo_dir):
        raise RuntimeError(f"SAM-Med2D repo dir not found: {repo_dir}")

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from segment_anything import sam_model_registry
    except Exception as exc:
        raise RuntimeError(
            "Could not import SAM-Med2D segment_anything package from --sam_med2d_repo_dir."
        ) from exc

    # SAM-Med2D checkpoints are commonly ViT-B based. If a different registry key
    # is needed, set SAM_MED2D_MODEL_TYPE in the sbatch environment.
    model_type = os.environ.get("SAM_MED2D_MODEL_TYPE", "vit_b")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    encoder = _SamMed2DImageEncoder(sam.image_encoder)
    if device is not None:
        encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, f"SAM-Med2D:{checkpoint_path}"


class _SamMed2DImageEncoder(nn.Module):
    def __init__(self, image_encoder, image_size=1024):
        super().__init__()
        self.image_encoder = image_encoder
        self.config = Namespace(image_size=int(image_size))

    def forward(self, pixel_values=None, x=None):
        xb = pixel_values if pixel_values is not None else x
        if xb is None:
            raise ValueError("Expected pixel_values or x.")
        xb = F.interpolate(
            xb,
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
        )
        feat = self.image_encoder(xb)
        return _extract_embedding_from_output(feat)


def _load_medical_fm_encoder(
    medical_fm,
    model_id="",
    device=None,
    allow_fallback=False,
    brainseg_volume_size=96,
    sam_med2d_checkpoint="",
    sam_med2d_repo_dir="",
):
    if device is None:
        device = _select_device()

    chosen_model_id = model_id
    subfolder = None
    if not chosen_model_id:
        if medical_fm == "brainlm":
            chosen_model_id = "vandijklab/brainlm"
            subfolder = "vitmae_111M"
        elif medical_fm == "brainsegfounder":
            chosen_model_id = "smilelab/BrainSegFounder"
        elif medical_fm == "sam_med2d":
            chosen_model_id = "OpenGVLab/SAM-Med2D"
        else:
            raise ValueError(f"Unknown medical FM: {medical_fm}")

    if medical_fm == "brainsegfounder":
        model, loaded_id = _load_brainsegfounder_encoder(chosen_model_id, volume_size=brainseg_volume_size)
        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model, loaded_id

    if medical_fm == "sam_med2d":
        return _load_sam_med2d_encoder(
            chosen_model_id,
            checkpoint_path=sam_med2d_checkpoint,
            repo_dir=sam_med2d_repo_dir,
            device=device,
        )

    try:
        from transformers import AutoModel

        kwargs = dict(trust_remote_code=True)
        if subfolder:
            kwargs["subfolder"] = subfolder
        model = AutoModel.from_pretrained(chosen_model_id, **kwargs)
    except Exception as e:
        if allow_fallback and timm is not None:
            fallback = "vit_base_patch16_224"
            model = timm.create_model(fallback, pretrained=True, num_classes=0, global_pool="avg")
            chosen_model_id = f"{chosen_model_id} (fallback:{fallback})"
        else:
            raise RuntimeError(
                f"Failed to load requested medical FM encoder '{chosen_model_id}'. "
                "No fallback was used. If you intentionally want the generic timm "
                "ViT fallback for debugging, pass --allow_medical_fm_fallback."
            ) from e

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, chosen_model_id


def _extract_embedding_from_output(outputs):
    if isinstance(outputs, torch.Tensor):
        if outputs.dim() == 2:
            return outputs
        if outputs.dim() >= 3:
            return outputs.flatten(2).mean(dim=-1)
        return outputs.flatten(1)
    if isinstance(outputs, dict):
        for key in ["pooler_output", "last_hidden_state", "image_embeddings", "embeddings"]:
            if key in outputs and isinstance(outputs[key], torch.Tensor):
                return _extract_embedding_from_output(outputs[key])
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        x = outputs.last_hidden_state
        return x.mean(dim=1) if x.dim() >= 3 else x.flatten(1)
    if isinstance(outputs, (list, tuple)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
        x = outputs[0]
        if x.dim() == 2:
            return x
        return x.mean(dim=1) if x.dim() >= 3 else x.flatten(1)
    raise RuntimeError("Unable to extract embeddings from model output.")


def _encode_medical_batch(model, xb):
    try:
        return _extract_embedding_from_output(model(pixel_values=xb))
    except Exception:
        pass
    try:
        return _extract_embedding_from_output(model(xb))
    except Exception as e:
        raise RuntimeError(f"Medical FM forward failed for both signatures: {e}")


def _compute_embeddings(model, loader, device):
    all_emb, all_y, all_idx = [], [], []
    with torch.no_grad():
        for xb, yb, idxs in loader:
            xb = xb.to(device)
            emb = _encode_medical_batch(model, xb).detach().cpu()
            all_emb.append(emb)
            all_y.append(yb.clone())
            all_idx.append(idxs.clone())
    return torch.cat(all_emb, dim=0), torch.cat(all_y, dim=0), torch.cat(all_idx, dim=0)


def _infer_model_image_size(model, default_size=224):
    # Try common places for expected input size.
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for key in ["image_size", "input_size"]:
            v = getattr(cfg, key, None)
            if isinstance(v, int):
                return int(v)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return int(v[-1])
    # timm-style fallback
    default_cfg = getattr(model, "default_cfg", None)
    if isinstance(default_cfg, dict):
        v = default_cfg.get("input_size", None)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return int(v[-1])
    return int(default_size)


def _few_shot_prototypes(train_emb, train_y, k, seed):
    rng = np.random.default_rng(seed)
    y = train_y.numpy()
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    k_pos = max(1, k // 2)
    k_neg = max(1, k - k_pos)
    sel_pos = rng.choice(idx_pos, size=min(k_pos, len(idx_pos)), replace=False) if len(idx_pos) > 0 else np.array([], dtype=int)
    sel_neg = rng.choice(idx_neg, size=min(k_neg, len(idx_neg)), replace=False) if len(idx_neg) > 0 else np.array([], dtype=int)
    if len(sel_pos) == 0 or len(sel_neg) == 0:
        raise RuntimeError("Few-shot prototype sampling failed due to missing class samples.")
    proto_pos = train_emb[torch.tensor(sel_pos)].mean(dim=0)
    proto_neg = train_emb[torch.tensor(sel_neg)].mean(dim=0)
    return proto_neg, proto_pos


def run_medical_fm_transfer(
    mats,
    labels,
    split,
    medical_fm,
    medical_fm_model_id="",
    few_shot_k=0,
    batch_size=16,
    seed=42,
    allow_fallback=False,
    brainseg_volume_size=96,
    sam_med2d_checkpoint="",
    sam_med2d_repo_dir="",
):
    device = _select_device()
    encoder, loaded_id = _load_medical_fm_encoder(
        medical_fm,
        medical_fm_model_id,
        device=device,
        allow_fallback=allow_fallback,
        brainseg_volume_size=brainseg_volume_size,
        sam_med2d_checkpoint=sam_med2d_checkpoint,
        sam_med2d_repo_dir=sam_med2d_repo_dir,
    )
    image_size = _infer_model_image_size(encoder, default_size=224)

    test_ds = AbideImageDataset(mats, labels, split.test_idx, image_size=image_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    test_emb, test_y, test_idx = _compute_embeddings(encoder, test_loader, device)

    test_emb = torch.nn.functional.normalize(test_emb, dim=1)

    if few_shot_k > 0:
        train_ds = AbideImageDataset(mats, labels, split.train_idx, image_size=image_size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        train_emb, train_y, _ = _compute_embeddings(encoder, train_loader, device)
        train_emb = torch.nn.functional.normalize(train_emb, dim=1)
        proto_neg, proto_pos = _few_shot_prototypes(train_emb, train_y, few_shot_k, seed)
        proto_neg = torch.nn.functional.normalize(proto_neg.unsqueeze(0), dim=1).squeeze(0)
        proto_pos = torch.nn.functional.normalize(proto_pos.unsqueeze(0), dim=1).squeeze(0)
        score_pos = (test_emb @ proto_pos) - (test_emb @ proto_neg)
    else:
        d = test_emb.shape[1]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        w = torch.randn(d, generator=gen)
        w = torch.nn.functional.normalize(w.unsqueeze(0), dim=1).squeeze(0)
        score_pos = test_emb @ w

    prob_pos = torch.sigmoid(score_pos).numpy()
    pred = (prob_pos >= 0.5).astype(int)
    y_true = test_y.numpy().astype(int)

    records = []
    for i in range(len(y_true)):
        records.append((int(test_idx[i]), int(y_true[i]), int(pred[i]), float(prob_pos[i])))

    metrics = compute_metrics(y_true, pred, prob_pos)
    metrics["total_params"] = _count_params(encoder)
    metrics["trainable_pct"] = 0.0
    metrics["loaded_model_id"] = loaded_id
    metrics["image_size"] = int(image_size)
    return metrics, records


def label_logprob(model, tokenizer, prompt_ids, label_text, device):
    # Compute log-prob of label_text tokens given prompt
    label_ids = tokenizer.encode(label_text, add_special_tokens=False)
    if len(label_ids) == 0:
        raise ValueError(f"Label text tokenized to empty: {label_text!r}")

    prompt_len = prompt_ids.shape[1]
    label_ids_t = torch.tensor([label_ids], device=device)
    input_ids = torch.cat([prompt_ids, label_ids_t], dim=1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [1, seq_len, vocab]

    logprobs = torch.log_softmax(logits, dim=-1)

    # For token t in label_ids, its logprob is at position prompt_len-1 + t_idx
    lp = 0.0
    start = prompt_len - 1
    for i, tok in enumerate(label_ids):
        lp += float(logprobs[0, start + i, tok])
    return lp


def predict_sample(model, tokenizer, messages, device, label_pos="ASD", label_neg="Control"):
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # apply_chat_template may return a tensor or BatchEncoding; ensure we pass input_ids
    if hasattr(enc, "input_ids"):
        prompt_ids = enc.input_ids.to(device)
    elif isinstance(enc, dict) and "input_ids" in enc:
        prompt_ids = enc["input_ids"].to(device)
    else:
        prompt_ids = enc.to(device)

    # Use log-prob scoring for AUROC
    score_pos = label_logprob(model, tokenizer, prompt_ids, label_pos, device)
    score_neg = label_logprob(model, tokenizer, prompt_ids, label_neg, device)

    # Softmax to get probability of positive class
    m = max(score_pos, score_neg)
    prob_pos = math.exp(score_pos - m) / (math.exp(score_pos - m) + math.exp(score_neg - m))
    pred = 1 if prob_pos >= 0.5 else 0
    return pred, prob_pos, score_pos, score_neg


def compute_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    acc = float(np.mean(y_pred == y_true))

    # Sensitivity (TPR) and Specificity (TNR)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    # AUROC
    auc = float(roc_auc_score(y_true, y_score))

    return {
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "auroc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot GPT-OSS-20B on ABIDE")
    parser.add_argument("--data", default="abide.npy", help="Path to abide.npy")
    parser.add_argument("--model_id", default="Qwen/Qwen3-32B", help="HF model id")
    parser.add_argument("--corr_type", default="corr", choices=["corr", "pcorr"], help="Connectivity matrix to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--topk", type=int, default=20, help="Top-k strongest connections in prompt")
    parser.add_argument("--max_test_samples", type=int, default=0, help="If >0, limit number of test samples")
    parser.add_argument("--out_csv", default="", help="Optional path to save per-sample results")
    parser.add_argument("--out_json", default="", help="Optional path to save metrics JSON")
    parser.add_argument("--run_model", default="", help="Optional model name for reporting")
    parser.add_argument("--run_setting", default="", help="Optional setting name for reporting")
    parser.add_argument("--label_pos", default=" ASD", help="Positive label text")
    parser.add_argument("--label_neg", default=" Control", help="Negative label text")
    parser.add_argument("--device_map", default="auto", help="Transformers device_map")
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Use fast tokenizer (may fail if tokenizer.json is incompatible)",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download of model/tokenizer files (use if cache is corrupted)",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Do not reach out to the Hub; use local cache only",
    )
    parser.add_argument(
        "--few_shot_k",
        type=int,
        default=0,
        help="Number of few-shot examples (0 = zero-shot)",
    )
    parser.add_argument(
        "--few_shot_seed",
        type=int,
        default=123,
        help="Seed for few-shot sampling",
    )
    parser.add_argument(
        "--few_shot_fixed",
        action="store_true",
        help="Use a fixed few-shot set for all test samples",
    )
    parser.add_argument(
        "--vision_model",
        default="",
        choices=["", "resnet50", "resnet101", "vit_b_16", "vit_l_16"],
        help="If set, run vision model pipeline (resnet50/resnet101/vit_b_16/vit_l_16) instead of LLM",
    )
    parser.add_argument(
        "--medical_fm",
        default="",
        choices=["", "brainlm", "brainsegfounder", "sam_med2d"],
        help="Run frozen medical FM transfer (brainlm/brainsegfounder/sam_med2d).",
    )
    parser.add_argument(
        "--medical_fm_model_id",
        default="",
        help="Optional override model id/path for medical FM loading.",
    )
    parser.add_argument(
        "--allow_medical_fm_fallback",
        action="store_true",
        help="Allow fallback to timm vit_base_patch16_224 if a medical FM cannot be loaded. Disabled by default for reportable experiments.",
    )
    parser.add_argument(
        "--brainseg_volume_size",
        type=int,
        default=96,
        help="Synthetic 3D volume size for BrainSegFounder FC-image transfer.",
    )
    parser.add_argument(
        "--sam_med2d_checkpoint",
        default=os.environ.get("SAM_MED2D_CHECKPOINT", ""),
        help="Path to SAM-Med2D checkpoint for true SAM-Med2D runs.",
    )
    parser.add_argument(
        "--sam_med2d_repo_dir",
        default=os.environ.get("SAM_MED2D_REPO_DIR", ""),
        help="Path to cloned SAM-Med2D repository for true SAM-Med2D runs.",
    )
    parser.add_argument(
        "--vision_mode",
        default="zero_shot",
        choices=["zero_shot", "linear_probe", "lora", "finetune"],
        help="Vision pipeline mode: zero_shot random-head baseline, linear_probe, lora fine-tune, or full finetune",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for vision models")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for LoRA fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for LoRA fine-tuning")
    parser.add_argument(
        "--lora_target_pct",
        type=float,
        default=0.005,
        help="Target trainable parameter fraction for LoRA (e.g., 0.005 = 0.5%%)",
    )
    args = parser.parse_args()

    labels, sites, corr, pcorr = load_abide(args.data)
    mats = corr if args.corr_type == "corr" else pcorr

    split = make_split(labels, sites, seed=args.seed)
    test_idx = split.test_idx

    if args.max_test_samples and args.max_test_samples > 0:
        test_idx = test_idx[: args.max_test_samples]

    print(f"Total samples: {len(labels)}")
    print(f"Train/Val/Test sizes: {len(split.train_idx)}/{len(split.val_idx)}/{len(split.test_idx)}")
    print(f"Using test samples: {len(test_idx)}")

    if args.medical_fm:
        metrics, records = run_medical_fm_transfer(
            mats=mats,
            labels=labels,
            split=split,
            medical_fm=args.medical_fm,
            medical_fm_model_id=args.medical_fm_model_id,
            few_shot_k=args.few_shot_k,
            batch_size=args.batch_size,
            seed=args.seed,
            allow_fallback=args.allow_medical_fm_fallback,
            brainseg_volume_size=args.brainseg_volume_size,
            sam_med2d_checkpoint=args.sam_med2d_checkpoint,
            sam_med2d_repo_dir=args.sam_med2d_repo_dir,
        )
    elif args.vision_model:
        if args.vision_mode == "zero_shot":
            metrics, records = run_vision_zero_shot(
                mats=mats,
                labels=labels,
                split=split,
                vision_model=args.vision_model,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        elif args.vision_mode == "linear_probe":
            metrics, records = run_vision_linear_probe(
                mats=mats,
                labels=labels,
                split=split,
                vision_model=args.vision_model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
            )
        elif args.vision_mode == "finetune":
            metrics, records = run_vision_finetune(
                mats=mats,
                labels=labels,
                split=split,
                vision_model=args.vision_model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
        else:
            metrics, records = run_vision_lora(
                mats=mats,
                labels=labels,
                split=split,
                vision_model=args.vision_model,
                lora_pct=args.lora_target_pct,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
    else:
        print("Loading model... this may take a while.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_id,
                use_fast=args.use_fast_tokenizer,
                force_download=args.force_download,
                local_files_only=args.local_files_only,
            )
        except Exception as exc:
            if args.use_fast_tokenizer:
                raise
            print(
                f"[WARN] Fast tokenizer failed; retrying with slow tokenizer. Error: {exc}"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_id,
                use_fast=False,
                force_download=args.force_download,
                local_files_only=args.local_files_only,
            )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype="auto",
            device_map=args.device_map,
            force_download=args.force_download,
            local_files_only=args.local_files_only,
        )
        model.eval()

        y_true, y_pred, y_score = [], [], []
        records = []

        fixed_few_shot = []
        if args.few_shot_k > 0 and args.few_shot_fixed:
            fixed_few_shot = build_few_shot_examples(
                mats=mats,
                labels=labels,
                train_idx=split.train_idx,
                k=args.few_shot_k,
                seed=args.few_shot_seed,
                topk=args.topk,
                label_pos=args.label_pos,
                label_neg=args.label_neg,
            )

        for k, idx in enumerate(test_idx, 1):
            mat = mats[idx]
            summary = summarize_connectivity(mat, topk=args.topk)
            if args.few_shot_k > 0:
                if args.few_shot_fixed:
                    few_shot_examples = fixed_few_shot
                else:
                    few_shot_examples = build_few_shot_examples(
                        mats=mats,
                        labels=labels,
                        train_idx=split.train_idx,
                        k=args.few_shot_k,
                        seed=args.few_shot_seed + int(idx),
                        topk=args.topk,
                        label_pos=args.label_pos,
                        label_neg=args.label_neg,
                    )
                messages = build_messages_few_shot(summary, few_shot_examples)
            else:
                messages = build_messages(summary)
            pred, prob_pos, score_pos, score_neg = predict_sample(
                model, tokenizer, messages, device=next(model.parameters()).device,
                label_pos=args.label_pos, label_neg=args.label_neg
            )

            y_true.append(int(labels[idx]))
            y_pred.append(int(pred))
            y_score.append(float(prob_pos))

            if args.out_csv:
                records.append((int(idx), int(labels[idx]), int(pred), float(prob_pos), float(score_pos), float(score_neg)))

            if k % 10 == 0 or k == len(test_idx):
                print(f"Processed {k}/{len(test_idx)}")

        metrics = compute_metrics(y_true, y_pred, y_score)
        metrics["total_params"] = _count_params(model)
        metrics["trainable_pct"] = 0.0
    print("\nResults (Test set):")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Confusion: TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            if args.vision_model or args.medical_fm:
                w.writerow(["idx", "y_true", "y_pred", "prob_pos"])
            else:
                w.writerow(["idx", "y_true", "y_pred", "prob_pos", "score_pos", "score_neg"])
            w.writerows(records)
        print(f"Saved per-sample results to {args.out_csv}")

    if args.out_json:
        if args.run_model:
            metrics["model"] = args.run_model
        if args.run_setting:
            metrics["setting"] = args.run_setting
        metrics["seed"] = int(args.seed)
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {args.out_json}")


if __name__ == "__main__":
    main()
