from omegaconf import DictConfig
from tasks.registry import TASKS


def create_task(cfg: DictConfig):
    """Creates the class for the task specified in cfg.

    Args:
        cfg (DictConfig): The (hydra) config to be used to build the task class.
    """
    task_type = cfg.task.type
    task_class = None
    if isinstance(task_type, str):
        task_class = TASKS.get(task_type)
    else:
        raise TypeError(
            "type must be a str or valid type, but got {}".format(type(task_type))
        )

    if task_class is None:
        raise KeyError("{} is not in the {} registry".format(task_type, TASKS.name))

    return task_class(cfg)
