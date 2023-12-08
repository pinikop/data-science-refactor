from pathlib import Path

import hydra
import torch
from icecream import ic
from src.dataset import create_dataloader
from src.models import LinearNet
from src.runner import Runner, run_epoch
from src.tensorboard import TensorboardExperiment
from src.utils import generate_tensorboard_experiment_directory

ROOT_PATH = Path(__file__).parent

LOG_PATH = ROOT_PATH / "runs"

# Data configuration
DATA_DIR = ROOT_PATH / "data"
TEST_DATA = DATA_DIR / "t10k-images-idx3-ubyte.gz"
TEST_LABELS = DATA_DIR / "t10k-labels-idx1-ubyte.gz"
TRAIN_DATA = DATA_DIR / "train-images-idx3-ubyte.gz"
TRAIN_LABELS = DATA_DIR / "train-labels-idx1-ubyte.gz"


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    # Model and Optimizer
    model = LinearNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.lr)

    # Data
    train_loader = create_dataloader(
        TRAIN_DATA, TRAIN_LABELS, batch_size=cfg.params.batch_size
    )
    test_loader = create_dataloader(
        TEST_DATA, TEST_LABELS, batch_size=cfg.params.batch_size, shuffle=False
    )

    # Create the Runners
    test_runner = Runner(test_loader, model)
    train_runner = Runner(train_loader, model, optimizer)

    # Experiment Trackers
    log_dir = generate_tensorboard_experiment_directory(root=LOG_PATH)
    tracker = TensorboardExperiment(log_dir=log_dir)

    for epoch in range(cfg.params.epochs):
        run_epoch(test_runner, train_runner, tracker, epoch, cfg.params.epochs)

    tracker.flush()


if __name__ == "__main__":
    main()
