# ResNet50 and ViT-B/16 ABIDE Fine-Tuning Scripts

This package contains the scripts used to fine-tune ImageNet-pretrained
ResNet50 and ViT-B/16 backbones on the ABIDE functional-connectivity
classification task.

## Files

- `zero_shot.py`: Evaluation of FMs and task-specific models on ABIDE under zero-shot, few-shot, linear-probe, fine-tuning.

## Data

Place `abide.npy` in the same directory before launching the job. The script
expects the ABIDE dictionary to contain `label`, `corr`, `pcorr`, and `site`.

## Fine-Tuning Setup

- Split: 70% train / 10% validation / 20% test, stratified by site
- Matrix: `corr`
- Input conversion: each FC matrix is min-max normalized, resized to the model
  input size, repeated into 3 RGB channels, and ImageNet-normalized
- Models: torchvision ImageNet-pretrained `resnet50` and `vit_b_16`
- Head: original classifier replaced by a two-class ASD/Control head
- Optimizer: AdamW
- Loss: cross-entropy
- Batch size: 16
- Epochs: 40
- Learning rate: `1e-4`
- Checkpoint selection: best validation AUROC checkpoint is restored before
  final test evaluation
- Seeds: 0, 1, 2

## Run

Single-seed commands:

```bash
python zero_shot.py \
  --data abide.npy \
  --corr_type corr \
  --seed 0 \
  --vision_model resnet50 \
  --vision_mode finetune \
  --batch_size 16 \
  --epochs 40 \
  --lr 1e-4 \
  --out_csv results/resnet50_ft_seed0.csv \
  --out_json metrics/resnet50_ft_seed0.json \
  --run_model ResNet50 \
  --run_setting Finetune
```

```bash
python zero_shot.py \
  --data abide.npy \
  --corr_type corr \
  --seed 0 \
  --vision_model vit_b_16 \
  --vision_mode finetune \
  --batch_size 16 \
  --epochs 40 \
  --lr 1e-4 \
  --out_csv results/vit_b_16_ft_seed0.csv \
  --out_json metrics/vit_b_16_ft_seed0.json \
  --run_model ViT-B/16 \
  --run_setting Finetune
```
