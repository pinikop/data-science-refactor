import os
from pathlib import Path
from typing import Any

from hydra.experimental.callback import Callback
from omegaconf import DictConfig


class HydraCallbacks(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        self._chdir(config)
        self._set_output_dir(config)

    @staticmethod
    def _chdir(config: DictConfig) -> None:
        """Override the working directory of the current run to the directory of the main.py file

        Args:
            config (DictConfig): configuration object of the current run.
        """
        file_path = Path(os.path.abspath(__file__))
        root_dir = file_path.parents[1]
        os.chdir(root_dir)
        config.hydra.runtime.cwd = root_dir

    @staticmethod
    def _set_output_dir(config: DictConfig) -> None:
        """override the output directory of the current run from timestamp to incremental ints

        Args:
            config (DictConfig): configuration object of the current run.
        """
        run_dir = Path(config.hydra.run.dir)
        children = [-1] + [
            int(c.name)
            for c in run_dir.glob("*")
            if (c.is_dir() and c.name.isnumeric())
        ]
        new_child = run_dir / str(max(children) + 1)
        config.hydra.run.dir = new_child.as_posix()
