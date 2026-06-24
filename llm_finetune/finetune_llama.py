# Fine-tune Llama-3.1-8B on brain FC datasets — LoRA, multi-dataset
#
# Supports: abide | parkinson | adni_nc_ad | adni_nc_mci
# Usage:
#   python finetune_llama.py --dataset abide
#   python finetune_llama.py --dataset parkinson --model_id /path/to/llama3.1-8b
#   python finetune_llama.py --dataset adni_nc_ad --seeds 42 43 44 45 46
#   python finetune_llama.py --dataset adni_nc_mci --n_epochs 3

import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit


# ===== Dataset registry =====

DATASETS_DIR = os.environ.get("BRAINLRR_DATASETS_DIR", "./datasets")

DATASET_CFG = {
    "abide": dict(
        path        = f"{DATASETS_DIR}/abide_original.npy",
        label_pos   = "ASD",
        label_neg   = "Control",
        stratify_by = "site",
        disease_ctx = (
            "Given a summary of a subject's brain functional connectivity, "
            "classify whether the subject has Autism Spectrum Disorder (ASD) "
            "or is a neurotypical Control. "
            "Respond with exactly one word: ASD or Control."
        ),
    ),
    "parkinson": dict(
        path        = f"{DATASETS_DIR}/pd_combined_cc200_1.npy",
        label_pos   = "Parkinson's Disease",
        label_neg   = "Healthy Control",
        stratify_by = "site_label",
        disease_ctx = (
            "Given a summary of a subject's brain functional connectivity, "
            "classify whether the subject has Parkinson's Disease "
            "or is a Healthy Control. "
            "Respond with exactly: Parkinson's Disease or Healthy Control."
        ),
    ),
    # Same 183 subjects as "parkinson", parcellated with the A424 atlas
    # (424 ROIs) instead of CC200 — closer to BrainLM's native pretraining
    # parcellation (424 regions). timeseries: (183, 200, 424), corr: (183, 424, 424).
    # Only usable by finetune_brainlm_timeseries.py (raw timeseries path).
    "parkinson_a424": dict(
        path        = f"{DATASETS_DIR}/pd_combined_a424_1.npy",
        label_pos   = "Parkinson's Disease",
        label_neg   = "Healthy Control",
        stratify_by = "site_label",
        disease_ctx = (
            "Given a summary of a subject's brain functional connectivity, "
            "classify whether the subject has Parkinson's Disease "
            "or is a Healthy Control. "
            "Respond with exactly: Parkinson's Disease or Healthy Control."
        ),
    ),
    "adni_nc_ad": dict(
        path        = f"{DATASETS_DIR}/ADNI_NC_AD.npy",
        label_pos   = "Alzheimer's Disease",
        label_neg   = "Normal Control",
        stratify_by = "label",
        disease_ctx = (
            "Given a summary of a subject's brain functional connectivity, "
            "classify whether the subject has Alzheimer's Disease "
            "or is a Normal Control. "
            "Respond with exactly: Alzheimer's Disease or Normal Control."
        ),
    ),
    "adni_nc_mci": dict(
        path        = f"{DATASETS_DIR}/ADNI_NC_MCI.npy",
        label_pos   = "MCI",
        label_neg   = "Normal Control",
        stratify_by = "label",
        disease_ctx = (
            "Given a summary of a subject's brain functional connectivity, "
            "classify whether the subject has Mild Cognitive Impairment (MCI) "
            "or is a Normal Control. "
            "Respond with exactly: MCI or Normal Control."
        ),
    ),
}


# ===== Reproducibility =====

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===== Parameter counting (for reporting model size in result tables) =====

def count_params(model):
    """Return (total_params, trainable_params) for a model."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ===== FC → text =====

def fc_to_text(fc_matrix: np.ndarray, top_k: int = 20, top_hubs: int = 10) -> str:
    """
    Summarise a (N, N) FC matrix as text.
    Includes: global stats, edge polarity, top hub ROIs, top-K edges.
    """
    N = fc_matrix.shape[0]
    rows, cols = np.triu_indices(N, k=1)
    vals = fc_matrix[rows, cols]

    # Global statistics
    stats = (f"mean={vals.mean():.3f}, std={vals.std():.3f}, "
             f"min={vals.min():.3f}, max={vals.max():.3f}, "
             f"median={np.median(vals):.3f}, "
             f"p25={np.percentile(vals, 25):.3f}, "
             f"p75={np.percentile(vals, 75):.3f}")

    # Edge polarity breakdown
    pos = vals[vals > 0]
    neg = vals[vals < 0]
    polarity = (f"pos={len(pos)} (mean={pos.mean():.3f}), "
                f"neg={len(neg)} (mean={neg.mean():.3f})")

    # Node strength = sum of |FC| per node (diagonal excluded)
    fc_nodiag = fc_matrix.copy()
    np.fill_diagonal(fc_nodiag, 0.0)
    strength = np.abs(fc_nodiag).sum(axis=1)
    hub_order = np.argsort(strength)[::-1][:top_hubs]
    hub_lines = [f"ROI_{i}: {strength[i]:.3f}" for i in hub_order]

    # Top-K edges by |strength|
    order = np.argsort(np.abs(vals))[::-1][:top_k]
    conn_lines = [f"ROI_{rows[i]}-ROI_{cols[i]}: {vals[i]:.3f}" for i in order]

    return (f"Nodes: {N}\n"
            f"Summary statistics:\n{stats}\n"
            f"Edge polarity: {polarity}\n"
            f"Top hub ROIs (by |strength|):\n" + "\n".join(hub_lines) + "\n"
            + "Top connections:\n" + "\n".join(conn_lines))


# ===== Prompt builders =====

def make_system_prompt(disease_ctx: str) -> str:
    return (
        "You are a careful medical classification assistant. "
        f"{disease_ctx} "
        "Reasoning: low."
    )


def make_prompt(fc_text: str, system_prompt: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Subject summary:\n{fc_text}\nLabel:"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def make_completion(label: int, label_pos: str, label_neg: str) -> str:
    label_str = label_pos if label == 1 else label_neg
    return label_str + "<|eot_id|>"


# ===== Data loading =====

def load_data(dataset_name: str):
    """Returns (fc, labels, site) as numpy arrays."""
    cfg  = DATASET_CFG[dataset_name]
    path = cfg["path"]
    print(f"Loading {dataset_name} from {path}")
    data   = np.load(path, allow_pickle=True).item()
    fc     = data["corr"].astype(np.float32)
    labels = data["label"].astype(int)
    site   = data["site"]
    return fc, labels, site


def site_zscore_fit(fc_np: np.ndarray, site, train_idx: np.ndarray) -> np.ndarray:
    """Fit per-site mean/std on train_idx only, apply to all subjects."""
    fc_out = fc_np.copy()
    for s in np.unique(site):
        site_mask  = site == s
        train_mask = site_mask.copy()
        train_mask[np.setdiff1d(np.where(site_mask)[0], train_idx)] = False
        if train_mask.sum() == 0:
            continue
        mu  = fc_np[train_mask].mean(axis=0, keepdims=True)
        std = fc_np[train_mask].std(axis=0,  keepdims=True) + 1e-8
        fc_out[site_mask] = (fc_np[site_mask] - mu) / std
    return fc_out


def get_split(fc, labels, site, stratify_by: str):
    """
    70 / 10 / 20 stratified split.
    - Site z-score fit on train only (no test leakage).
    - Val/test split within the 30% holdout is also stratified.
    stratify_by:
      'site'       — used for ABIDE
      'site_label' — used for Parkinson (combined site+label string)
      'label'      — used for ADNI (no meaningful site variation)
    """
    if stratify_by == "site":
        strat = site
    elif stratify_by == "site_label":
        strat = np.array([f"{s}_{l}" for s, l in zip(site, labels)])
    else:  # "label"
        strat = labels

    # 70 / 30 stratified split
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    for tr_idx, te_idx in sss1.split(fc, strat):
        pass

    # Split 30% holdout into val (1/3) and test (2/3), stratified by label
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=2/3, random_state=42)
    for val_idx_local, test_idx_local in sss2.split(te_idx, labels[te_idx]):
        val_idx  = te_idx[val_idx_local]
        test_idx = te_idx[test_idx_local]

    # Normalize using train statistics only
    fc_norm = site_zscore_fit(fc, site, tr_idx)
    return tr_idx, val_idx, test_idx, fc_norm


# ===== Build HuggingFace Dataset =====

def build_hf_dataset(fc_norm, labels, indices,
                     label_pos, label_neg, system_prompt, top_k=20,
                     balance=False):
    """
    balance=True oversamples the minority class by duplicating its
    text samples until both classes have equal counts. Use only for
    the training split — never for val/test.
    """
    from datasets import Dataset

    indices = np.asarray(indices)
    if balance:
        pos_idx = indices[labels[indices] == 1]
        neg_idx = indices[labels[indices] == 0]
        n_max = max(len(pos_idx), len(neg_idx))
        if len(pos_idx) < n_max:
            extra = np.random.choice(pos_idx, n_max - len(pos_idx), replace=True)
            indices = np.concatenate([indices, extra])
        elif len(neg_idx) < n_max:
            extra = np.random.choice(neg_idx, n_max - len(neg_idx), replace=True)
            indices = np.concatenate([indices, extra])

    prompts, completions = [], []
    for i in indices:
        fc_text = fc_to_text(fc_norm[i], top_k=top_k)
        prompts.append(make_prompt(fc_text, system_prompt))
        completions.append(make_completion(labels[i], label_pos, label_neg))
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


# ===== LoRA fine-tuning =====

def finetune(model_id, train_dataset, val_dataset, output_dir,
             lora_r=16, lora_alpha=32, epochs=5,
             lr=2e-4, batch_size=8, seed=42):

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                               EarlyStoppingCallback)
    from trl import SFTConfig, SFTTrainer

    print(f"\nLoading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token       = tokenizer.eos_token
    tokenizer.padding_side    = "right"
    tokenizer.model_max_length = 1024

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    param_info = count_params(model)

    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 16 // batch_size),
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        completion_only_loss=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_cfg,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    # Diagnostic: print train/eval loss per epoch and which checkpoint
    # was loaded as "best" (load_best_model_at_end). Helps catch the case
    # where early stopping selects an under-trained LoRA adapter (LoRA's
    # B matrix starts at zero, so an early checkpoint ~= base model).
    print("\n--- Loss curve ---")
    for entry in trainer.state.log_history:
        if "loss" in entry:
            print(f"  step={entry.get('step')} epoch={entry.get('epoch'):.2f} "
                  f"train_loss={entry['loss']:.4f}")
        elif "eval_loss" in entry:
            print(f"  step={entry.get('step')} epoch={entry.get('epoch'):.2f} "
                  f"eval_loss={entry['eval_loss']:.4f}")
    print(f"  best_model_checkpoint = {trainer.state.best_model_checkpoint}")
    print(f"  best_metric           = {trainer.state.best_metric}")
    print("------------------\n")

    return trainer.model, tokenizer, param_info


# ===== Evaluation =====

def label_logprob(model, prompt_ids, label_ids, device):
    """
    Mean log-prob of label_ids tokens given prompt (length-normalized).
    Sum would systematically penalize multi-token labels (e.g.
    "Parkinson's Disease" = 4 tokens vs "Healthy Control" = 2 tokens),
    biasing the comparison independent of the actual signal.
    """
    prompt_len = prompt_ids.shape[1]
    label_t    = torch.tensor([label_ids], device=device)
    input_ids  = torch.cat([prompt_ids, label_t], dim=1)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    logprobs = torch.log_softmax(logits, dim=-1)
    lp = 0.0
    for i, tok in enumerate(label_ids):
        lp += float(logprobs[0, prompt_len - 1 + i, tok])
    return lp / len(label_ids)


def get_probs(model, tokenizer, fc_norm, labels, indices,
               label_pos, label_neg, system_prompt, top_k=20):
    """Return (all_labels, all_probs) — prob_pos for each index, no thresholding."""
    model.eval()
    device = next(model.parameters()).device

    pos_ids = tokenizer.encode(label_pos, add_special_tokens=False)
    neg_ids = tokenizer.encode(label_neg, add_special_tokens=False)

    all_labels, all_probs = [], []

    for i in indices:
        fc_text    = fc_to_text(fc_norm[i], top_k=top_k)
        prompt_ids = tokenizer(make_prompt(fc_text, system_prompt),
                               return_tensors="pt",
                               truncation=True,
                               max_length=1024).input_ids.to(device)

        score_pos = label_logprob(model, prompt_ids, pos_ids, device)
        score_neg = label_logprob(model, prompt_ids, neg_ids, device)

        m        = max(score_pos, score_neg)
        prob_pos = math.exp(score_pos - m) / (
                   math.exp(score_pos - m) + math.exp(score_neg - m))

        all_probs.append(prob_pos)
        all_labels.append(int(labels[i]))

    return np.array(all_labels), np.array(all_probs)


def youden_threshold(labels, probs):
    """Threshold maximising sensitivity + specificity - 1 (Youden's J) on labels/probs."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "acc": float((preds == labels).mean()),
        "auc": float(roc_auc_score(labels, probs)),
        "sen": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "spe": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    }


# ===== Entry point =====

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        choices=list(DATASET_CFG.keys()),
                        required=True,
                        help="Dataset to fine-tune on")
    parser.add_argument("--model_id",   type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--seeds",      nargs="+", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--n_epochs",   type=int,   default=5)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--lora_r",     type=int,   default=16)
    parser.add_argument("--top_k",      type=int,   default=20,
                        help="Top-K FC edges in text summary (20 matches zero_shot.py)")
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--balance",    action="store_true",
                        help="Oversample minority class in training set "
                             "(use only if SEN/SPE collapses without it)")
    parser.add_argument("--output_dir", type=str,   default=None)
    args = parser.parse_args()

    cfg         = DATASET_CFG[args.dataset]
    label_pos   = cfg["label_pos"]
    label_neg   = cfg["label_neg"]
    stratify_by = cfg["stratify_by"]
    sys_prompt  = make_system_prompt(cfg["disease_ctx"])

    if args.output_dir is None:
        args.output_dir = f"results/llama_{args.dataset}_ckpt"

    # ── Load data ────────────────────────────────────────────
    fc, labels, site = load_data(args.dataset)
    print(f"  N={len(labels)}  pos={label_pos}={(labels==1).sum()}  "
          f"neg={label_neg}={(labels==0).sum()}")

    tr_idx, val_idx, test_idx, fc_norm = get_split(
        fc, labels, site, stratify_by)
    print(f"  Train={len(tr_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

    train_ds = build_hf_dataset(fc_norm, labels, tr_idx,
                                label_pos, label_neg, sys_prompt, args.top_k,
                                balance=args.balance)
    val_ds   = build_hf_dataset(fc_norm, labels, val_idx,
                                label_pos, label_neg, sys_prompt, args.top_k)

    # ── Multi-seed evaluation ────────────────────────────────
    results = []

    for seed in args.seeds:
        seed_everything(seed)
        print(f"\n{'='*60}")
        print(f" Seed {seed}  [Llama-3.1-8B  LoRA r={args.lora_r}  "
              f"dataset={args.dataset}]")
        print(f"{'='*60}")

        ckpt_dir = os.path.join(args.output_dir, f"seed{seed}")
        model, tokenizer, (total_params, trainable_params) = finetune(
            model_id      = args.model_id,
            train_dataset = train_ds,
            val_dataset   = val_ds,
            output_dir    = ckpt_dir,
            lora_r        = args.lora_r,
            lora_alpha    = args.lora_r * 2,
            epochs        = args.n_epochs,
            lr            = args.lr,
            batch_size    = args.batch_size,
            seed          = seed,
        )

        print(f"Calibrating threshold on val set ({len(val_idx)} samples)...")
        val_labels, val_probs = get_probs(model, tokenizer, fc_norm, labels, val_idx,
                                          label_pos, label_neg, sys_prompt, args.top_k)
        threshold = youden_threshold(val_labels, val_probs)
        print(f"  threshold (Youden's J) = {threshold:.4f}")

        print(f"Evaluating on test set ({len(test_idx)} samples)...")
        test_labels, test_probs = get_probs(model, tokenizer, fc_norm, labels, test_idx,
                                            label_pos, label_neg, sys_prompt, args.top_k)
        res = compute_metrics(test_labels, test_probs, threshold=threshold)
        res["seed"]             = seed
        res["threshold"]        = threshold
        res["dataset"]          = args.dataset
        res["total_params"]     = total_params
        res["trainable_params"] = trainable_params
        res["split_seed"]       = 42
        results.append(res)

        print(f"  ACC={res['acc']:.4f}  AUC={res['auc']:.4f}  "
              f"SEN={res['sen']:.4f}  SPE={res['spe']:.4f}")

        del model, tokenizer
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f" SUMMARY — Llama-3.1-8B fine-tuned  "
          f"dataset={args.dataset}  ({len(args.seeds)} seeds)")
    print(f"{'='*60}")
    for m in ["acc", "auc", "sen", "spe"]:
        print(f"  {m.upper():5}: {df[m].mean():.4f} ± {df[m].std():.4f}")

    os.makedirs("results", exist_ok=True)
    out_csv = f"results/results_{args.dataset}_llama_finetune.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    print(f"\nModel parameters: total={total_params/1e9:.2f}B  "
          f"trainable={trainable_params/1e6:.2f}M "
          f"({100*trainable_params/total_params:.2f}%)")

    print("\nTable row (×100, mean ± std):")
    print(f"  LLM | Llama-3.1-8B | fine-tuned | {args.dataset} | "
          f"{df['auc'].mean()*100:.2f} ± {df['auc'].std()*100:.2f} | "
          f"{df['acc'].mean()*100:.2f} ± {df['acc'].std()*100:.2f} | "
          f"{df['sen'].mean()*100:.2f} ± {df['sen'].std()*100:.2f} | "
          f"{df['spe'].mean()*100:.2f} ± {df['spe'].std()*100:.2f} | "
          f"params={total_params/1e9:.2f}B (trainable={trainable_params/1e6:.2f}M)")
