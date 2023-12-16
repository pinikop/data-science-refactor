from torch import nn, optim

from .config_parser import MNISTConfig


def get_optimizer(conf: MNISTConfig, model: nn.Module) -> optim.Optimizer:
    cfg = conf.optimizer
    optimizer_name = cfg.name
    if optimizer_name == "sgd":
        optimizer = optim.SGD
    elif optimizer_name == "adam":
        optimizer = optim.Adam
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizer(model.parameters(), **cfg.model_dump(exclude={"name"}))
