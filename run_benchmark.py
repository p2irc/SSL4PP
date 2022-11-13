from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import tasks.builder as task_builder
import utils.distributed as dist_utils
import utils.misc
import wandb


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Run a benchmark on a task.

    Args:
        cfg (DictConfig): Hydra config object.

    """
    utils.misc.set_random_seed(cfg.get("seed"), cfg.get("deterministic", False))

    # intitialize distributed training
    distributed = cfg.get("distributed")
    if distributed and isinstance(distributed, DictConfig):
        dist_utils.init_distributed_mode(
            launcher=cfg.distributed.get("launcher"),
            backend=cfg.distributed.get("backend", "nccl"),
        )

    task = task_builder.create_task(cfg)

    wandb_logger = None  # for all processes except the master process
    if distributed is None or (distributed and dist_utils.is_main_process()):
        if task.cfg.get("wandb"):
            wandb_logger = wandb.init(
                job_type="test" if task.cfg.task.get("evaluate") else "train",
                config=OmegaConf.to_object(task.cfg),
                **dict(task.cfg.wandb)
            )

    print(OmegaConf.to_yaml(task.cfg, resolve=True))
    task.run(wandb_logger)

    if wandb_logger:
        ckpk_path = (
            Path(task.cfg.checkpoint.get("dir", "checkpoints"))
            .joinpath("latest.pt")
            .resolve()
        )
        if ckpk_path.exists():
            wandb.save(str(ckpk_path))

        wandb_logger.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
