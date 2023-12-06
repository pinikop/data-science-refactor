from pathlib import Path
from typing import Union


def generate_tensorboard_experiment_directory(
    root: Union[str, Path], parents=True
) -> str:
    """Generates a unique experiment directory.

    Args:
        root (str or Path): root directory
        parents (bool): whether to create parent directories

    Returns:
        str: experiment directory
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        root_path.mkdir(parents=parents)
    children = [
        int(c.name) for c in root_path.glob("*") if (c.is_dir() and c.name.isnumeric())
    ]
    experiment = str(max(children) + 1) if len(children) else "0"
    experiment = root_path / experiment
    experiment.mkdir(parents=parents)
    return experiment.as_posix()
