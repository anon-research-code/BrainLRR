from omegaconf import DictConfig, open_dict
from .abide import load_abide_data
from .adni import load_adni_data
from .parkinson import load_parkinson_data
from .dataloader import init_dataloader, init_stratified_dataloader
from typing import List
import torch.utils as utils

SUPPORTED_DATASETS = ['abide', 'adni', 'parkinson']


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    assert cfg.dataset.name in SUPPORTED_DATASETS, \
        f"Dataset '{cfg.dataset.name}' not supported. Choose from {SUPPORTED_DATASETS}."

    datasets = eval(
        f"load_{cfg.dataset.name}_data")(cfg)

    dataloaders = init_stratified_dataloader(cfg, *datasets) \
        if cfg.dataset.stratified \
        else init_dataloader(cfg, *datasets)

    return dataloaders
