"""UMAP visualization task."""
import ntpath
from typing import Any, Dict, Optional, Tuple, Union

import hydra
import numpy as np
import pandas as pd
import torch.nn
import umap
import umap.plot
from bokeh.plotting import output_file, save
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.wandb_run import Run

import datasets.builder as dataset_builder
import models.builder as model_builder
import wandb
from tasks.base import Task
from utils.logger import MetricLogger
from utils.misc import set_random_seed


class UMAPViz(Task):
    """UMAP visualization task.

    Args:
        cfg (DictConfig): configuration for the task
        **kwargs: additional arguments

    """

    def __init__(self, cfg: DictConfig, **kwargs: Any) -> None:
        """Initialize the task."""
        self._cfg = cfg
        rand_seed = self.cfg.get("seed")

        self._batch_size = self.cfg.task.dataloader.global_batch_size

        # hydra modifies the working directory, so it will look for relative
        # paths in the new working directory
        if not ntpath.isabs(self.cfg.task.dataset.root):
            self._cfg.task.dataset.root = hydra.utils.to_absolute_path(
                self.cfg.task.dataset.root
            )

        # load the dataset
        self.datasets = dataset_builder.build_dataset(
            self.cfg.task.dataset, None, rand_seed
        )

        self.train_loader, self.test_loader = dataset_builder.load_datasets(
            self.datasets,
            self.batch_size,
            self.cfg.task.dataloader.num_workers,
            seed=rand_seed,
            collate_fn=kwargs.get("collate_fn"),
        )

        # create model
        model = model_builder.build_backbone(self.cfg.task.backbone)
        model.fc = torch.nn.Identity()  # remove the fully connected layer

        # load pretrained checkpoint
        ckpt_path = self.cfg.task.pretrained_ckpt
        if ckpt_path is not None:
            if not ckpt_path.startswith("http") and not ntpath.isabs(ckpt_path):
                ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
            model = self.load_pretrained_checkpoint(
                model, ckpt_path, kwargs.get("state_dict_replacement_key", "")
            )

        # load model to device
        self.model = model
        if torch.cuda.is_available():
            self.model = model.cuda()

        # no need for training-related variables
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None
        self.train_metrics = None
        self.test_metrics = None
        self._scaler = None

    @property
    def cfg(self) -> DictConfig:
        """Get the configuration of the task."""
        return self._cfg

    @property
    def init_lr(self) -> float:
        """Get the initial learning rate.

        No need for this task.

        """
        pass

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self._batch_size

    def prepare_input(self, *args, **kwargs) -> Tuple:
        """Prepare the input for the model.

        No need for this task.

        """
        pass

    def get_loss(self, *args, **kwargs) -> Union[float, torch.Tensor]:
        """Get the loss.

        No need for this task.

        """
        pass

    def get_train_metrics(self, *args, **kwargs) -> Dict[str, Any]:
        """Get the training metrics.

        No need for this task. Here for compatibility with the base
        class.

        """
        pass

    def evaluate(self, *args, **kwargs) -> Any:
        """Evaluate the model.

        No need for this task. Here for compatibility with the base
        class.

        """
        pass

    def run(self, wandb_logger: Optional[Run] = None):
        """Run the task."""
        self.model.eval()

        if self.in_dist_mode:
            self.test_loader.sampler.set_epoch(0)

        metric_logger = MetricLogger(delimiter="  ")

        log_freq = self.cfg.get("log_freq", 1)
        representations = []

        for images, _ in metric_logger.log_every(self.test_loader, log_freq):
            # images = list(img.cuda(self.device_id, non_blocking=True) for img in images)

            with torch.cuda.amp.autocast(
                enabled=self.cfg.task.get("mixed_precision", False)
            ):
                output = self.model(images.cuda(self.device_id, non_blocking=True))

            representations.extend(output.detach().cpu().numpy())

        representations = np.asarray(representations)
        np.save("representations", representations)

        reducer = umap.UMAP(
            **self.cfg.task.umap_params, random_state=self.cfg.get("seed")
        ).fit(representations)

        # num_classes = self.datasets['test'].num_classes
        class_ids = np.array([id for _, id in self.datasets["test"].samples])
        filenames = np.array(
            [filename.split("/")[-1] for filename, _ in self.datasets["test"].samples]
        )
        # class_names = list(self.datasets['test'].class_to_idx.keys())

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])  # axis starts at 0.1, 0.1
        umap.plot.points(reducer, labels=class_ids, cmap="Spectral", ax=ax)

        # plt.scatter(embedding[:, 0], embedding[:, 1], c=class_ids, cmap='Spectral', s=5)
        # plt.gca().set_aspect('equal', 'datalim')
        # cbar = plt.colorbar(boundaries=np.arange(num_classes + 1) - 0.5)
        # cbar.set_ticks(np.arange(num_classes))
        # cbar.set_ticklabels(class_names)

        # interactive plot
        hover_data = pd.DataFrame(
            {
                "index": np.arange(len(class_ids)),
                "label": class_ids,
                "filename": filenames,
            }
        )
        hover_data["item"] = hover_data.label.map(self.datasets["test"].idx_to_class)

        i_plt = umap.plot.interactive(
            reducer, labels=class_ids, hover_data=hover_data, point_size=2
        )

        # save plots
        fig.savefig(f"{self.cfg.experiment_name}.png")
        output_file(f"{self.cfg.experiment_name}.html")
        save(i_plt)

        if wandb_logger:
            wandb_html = wandb.Html(f"{self.cfg.experiment_name}.html")
            table = wandb.Table(columns=["UMAP_embeddings"], data=[[wandb_html]])
            wandb_logger.log({f"{self.cfg.experiment_name}_UMAP_projection": table})


@hydra.main(config_path="../../configs", config_name="umap_viz_cfg")
def main(cfg: DictConfig):
    """Run the task."""
    set_random_seed(cfg.get("seed"))

    wandb_logger = None  # for all processes except the master process
    if cfg.get("wandb"):
        wandb_logger = wandb.init(
            job_type="umap-viz", config=OmegaConf.to_object(cfg), **dict(cfg.wandb)
        )

    print(OmegaConf.to_yaml(cfg, resolve=True))
    task = UMAPViz(cfg)
    task.run(wandb_logger)

    if wandb_logger:
        wandb_logger.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
