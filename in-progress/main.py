import hydra
import torch
from config import MNISTConfig
from omegaconf import OmegaConf
from src.dataset import create_dataloader
from src.models import LinearNet
from src.runner import Runner, run_epoch
from src.tensorboard import TensorboardExperiment
from src.utils import generate_tensorboard_experiment_directory, set_cwd_2_file_dir


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
    tracker = TensorboardExperiment(log_dir=cfg.paths.log)

    for epoch in range(cfg.params.epochs):
        run_epoch(test_runner, train_runner, tracker, epoch, cfg.params.epochs)

    tracker.flush()


@hydra.main(
    config_path="conf",
    config_name="meta_config",
    version_base=None,
)
def pre(mcfg):
    output_dir = generate_tensorboard_experiment_directory(root="./outputs/")
    cfg = OmegaConf.load(mcfg.config.file)
    cfg.hydra.run.dir = output_dir
    OmegaConf.save(cfg, mcfg.config.file)


if __name__ == "__main__":
    set_cwd_2_file_dir(__file__)
    pre()
    main()
