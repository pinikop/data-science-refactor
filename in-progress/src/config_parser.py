from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator


class BaseOptimizer(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    lr: float


class SGDOptimizer(BaseOptimizer):
    momentum: float


class ADAMOptimizer(BaseOptimizer):
    eps: float


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
    optimizer: Union[SGDOptimizer, ADAMOptimizer]

    # @field_validator("optimizer", mode="after")
    # def validate_optimizer(cls, v):
    #     if not issubclass(v, BaseOptimizer):
    #         raise ValueError(f"{v} is not a subclass of BaseOptimizer")
    #     return v

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> "MNISTConfig":
        return MNISTConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore
