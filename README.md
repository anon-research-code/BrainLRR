# BrainLRR

BrainLRR is a research codebase for functional-connectivity-based brain disorder classification.
It combines a transformer-based brain network encoder with **low-rank representation regularization** and **stochastic ROI/node masking** to improve representation compactness and robustness.

The method is evaluated on **ABIDE**, **ADNI**, and **Parkinson’s disease** functional-connectivity classification tasks.
The repository also includes comparison pipelines for fine-tuning general and medical foundation-model backbones, including Llama-3.1, BrainLM, BrainSegFounder, ResNet50, and ViT.

## Structure

```text
source/          Core package: models, datasets, training loops, Hydra configs
scripts/         Training entry points
  train_baseline.py          Transformer baseline, Optuna HP search
  train_baseline_fixed_hp.py Transformer baseline with optional fixed HPs
  train_BrainLRR.py          BrainLRR: node masking + LRR auxiliary loss

roi_analysis/    ROI importance extraction and visualization scripts
  extract_BrainLRR_importance.py  ROI importance from BrainLRR checkpoints
  visualize_rois_v5.py         Multi-dataset ROI brain plots (CC200, 200 ROIs)
llm_finetune/    Fine-tuning scripts for Llama-3.1, BrainLM, BrainSegFounder,
                 ResNet50, and ViT on the same FC datasets
```

## Setup

Place processed `.npy` dataset files under:

```text
./datasets/
```

Alternatively, set:

```bash
export BRAINLRR_DATASETS_DIR=/path/to/datasets
```

Update dataset paths in:

```text
source/conf/dataset/*.yaml
```

Install dependencies:

```bash
pip install torch hydra-core omegaconf optuna scikit-learn pandas
```

Additional dependencies may be required for scripts under `llm_finetune/`.

## Usage

Run from the repository root.

### Transformer baseline

```bash
python scripts/train_baseline.py --dataset abide
```

### BrainLRR

```bash
python scripts/train_BrainLRR.py \
  --dataset abide \
  --drop_node_p 0.4 \
  --lrr_weight 0.0005 \
  --lr 7.74e-05 \
  --wd 0.0235
```

For foundation-model comparison experiments, see:

```text
llm_finetune/README.md
```

## Implementation Note

The transformer backbone follows the Brain Network Transformer architecture.
This repository extends the implementation with low-rank regularization, stochastic ROI masking, ablation scripts, and cross-dataset evaluation.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

Parts of the implementation are adapted from the public Brain Network Transformer codebase under the MIT License. The original copyright and license notice are retained.
