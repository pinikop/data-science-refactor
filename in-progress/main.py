import hydra
import torch
from config import MNISTConfig
from src.dataset import create_dataloader
from src.models import LinearNet
from src.runner import Runner, run_epoch
from src.tensorboard import TensorboardExperiment
from src.utils import generate_tensorboard_experiment_directory


@hydra.main(
    config_path="conf",
    config_name="config",
    version_base=None,
)
def main(cfg: MNISTConfig):
    # Model and Optimizer
    model = LinearNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.lr)

    # Data
    train_loader = create_dataloader(
        data_path=cfg.files.train_data,
        labels_path=cfg.files.train_labels,
        batch_size=cfg.params.batch_size,
    )
    test_loader = create_dataloader(
        data_path=cfg.files.test_data,
        labels_path=cfg.files.test_labels,
        batch_size=cfg.params.batch_size,
        shuffle=False,
    )

    # Create the Runners
    test_runner = Runner(test_loader, model)
    train_runner = Runner(train_loader, model, optimizer)

    # Experiment Trackers
    log_dir = generate_tensorboard_experiment_directory(root=cfg.paths.log)
    tracker = TensorboardExperiment(log_dir=log_dir)

    for epoch in range(cfg.params.epochs):
        run_epoch(test_runner, train_runner, tracker, epoch, cfg.params.epochs)

    tracker.flush()


if __name__ == "__main__":
    main()
