from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


class Paths(BaseModel):
    log: Path
    data: Path


class Files(BaseModel):
    test_data: Path
    test_labels: Path
    train_data: Path
    train_labels: Path


class Params(BaseModel):
    epochs: int
    batch_size: int
    lr: float


class MNISTConfig(BaseModel):
    paths: Paths
    files: Files
    params: Params


def dictconfig_2_mnistconfig(cfg: DictConfig) -> MNISTConfig:
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    parsed_cfg: MNISTConfig = MNISTConfig(**cfg_dict)
    return parsed_cfg
