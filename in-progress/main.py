import hydra
from omegaconf import DictConfig
from src.config_parser import MNISTConfig
from src.dataset import create_dataloader
from src.models import LinearNet
from src.optimizers import get_optimizer
from src.runner import Runner, run_epoch
from src.tensorboard import TensorboardExperiment


@hydra.main(
    config_path="conf",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    mcfg = MNISTConfig.from_dictconfig(cfg)
    # Model and Optimizer
    model = LinearNet()
    optimizer = get_optimizer(mcfg, model)
    # Data
    train_loader = create_dataloader(
        data_path=mcfg.files.train_data,
        labels_path=mcfg.files.train_labels,
        batch_size=mcfg.params.batch_size,
    )
    test_loader = create_dataloader(
        data_path=mcfg.files.test_data,
        labels_path=mcfg.files.test_labels,
        batch_size=mcfg.params.batch_size,
        shuffle=False,
    )

    # Create the Runners
    test_runner = Runner(test_loader, model)
    train_runner = Runner(train_loader, model, optimizer)

    # Experiment Trackers
    tracker = TensorboardExperiment(log_dir=mcfg.paths.log)

    for epoch in range(mcfg.params.epochs):
        run_epoch(test_runner, train_runner, tracker, epoch, mcfg.params.epochs)

    tracker.flush()


if __name__ == "__main__":
    main()
