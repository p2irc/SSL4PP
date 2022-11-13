import os
import sys
import math
import ntpath
from pathlib import Path
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, OrderedDict, Tuple, Union

import torch.nn
from torch import Tensor
from omegaconf import DictConfig
from wandb.sdk.wandb_run import Run
from hydra.utils import to_absolute_path

import utils.distributed as dist_utils
import models.builder as model_builder
import trainer.builder as trainer_builder
import datasets.builder as dataset_builder
from utils.logger import MetricLogger, SmoothedValue


class Task(ABC):
    """Abstract class for instantiating a learning task.
    A task includes a dataset and dataloader, a model, an optimizer,
    a learning rate scheduler, a loss function and, optionally, some performance metrics.

    Args:
        cfg (DictConfig): a hydra config object.

    Attributes:
        cfg (DictConfig): a hydra config object containing all the information
            needed to set up the learning task.
        datasets (Dict[str, Any]): a dictionary containing the dataset objects.
        model (torch.nn.Module): the model to be trained.
        train_loader (torch.utils.data.DataLoader): the dataloader for the training set.
        test_loader (torch.utils.data.DataLoader): the dataloader for the test set.
        optimizer (torch.optim.Optimizer): the optimizer used to train the model.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): the learning rate scheduler.
        criterion (torch.nn.Module): the loss function.
        train_metrics (MetricLogger): a metric logger for the training set.
        test_metrics (MetricLogger): a metric logger for the test set.
        scaler (torch.cuda.amp.GradScaler): a scaler for mixed precision training.
        device_id (int): the id of the GPU to be used.
        in_dist_mode (bool): whether the task is running in distributed mode.
        wandb_run (wandb.sdk.wandb_run.Run): a wandb run object.
        batch_size (int): the batch size.

    """

    def __init__(self, cfg: DictConfig, **kwargs: Any) -> None:
        self._cfg = cfg
        rand_seed = self.cfg.get("seed")

        # infer learning rate before possibly changing batch size
        self._batch_size = self.cfg.task.dataloader.global_batch_size
        self._init_lr = self.cfg.task.trainer.base_lr * (
            self.batch_size
            / self.cfg.task.trainer.get("base_lr_batch_size", self.batch_size)
        )

        # hydra modifies the working directory, so it will look for relative
        # paths in the new working directory
        if not ntpath.isabs(self.cfg.task.dataset.root):
            self._cfg.task.dataset.root = to_absolute_path(self.cfg.task.dataset.root)

        self.datasets = dataset_builder.build_dataset(
            self.cfg.task.dataset, self.cfg.task.type, rand_seed
        )

        if "num_classes" in self.cfg.task.model.keys():
            # check if it has been set by the user already
            try:
                self.cfg.task.model.num_classes
            except Exception as e:
                print(e)
                self.cfg.task.model.num_classes = self.datasets["train"].num_classes
                print("num_classes has been set to ", self.cfg.task.model.num_classes)

        model = model_builder.build_model(self.cfg.task.model)

        # load pretrained checkpoint before possibly wrapping the model in DDP
        if self.cfg.checkpoint.get("pretrained"):
            ckpt_path = self.cfg.checkpoint.pretrained
            if not ckpt_path.startswith("http") and not ntpath.isabs(ckpt_path):
                ckpt_path = to_absolute_path(ckpt_path)
                self._cfg.checkpoint.pretrained = ckpt_path
            model = self.load_pretrained_checkpoint(
                model, ckpt_path, kwargs.get("state_dict_replacement_key", "")
            )

        self.model = model
        self.load_model_to_device()  # !modifies the global batch_size. Do this first before loading the dataset!

        # load the dataset
        self.train_loader, self.test_loader = dataset_builder.load_datasets(
            self.datasets,
            self.batch_size,
            self.cfg.task.dataloader.num_workers,
            self.cfg.task.dataloader.get("drop_last", True),
            rand_seed,
            collate_fn=kwargs.get("collate_fn"),
        )

        # construct trainer members
        self.optimizer = trainer_builder.build_optimizer(
            self.cfg.task.trainer.optimizer,
            self.get_model_parameters(),
            self.init_lr,
            self.cfg.task.trainer.get("use_lars", False),
        )

        total_steps = self.cfg.task.trainer.num_epochs * len(self.train_loader)
        self.lr_scheduler = trainer_builder.build_scheduler(
            self.cfg.task.trainer.lr_scheduler,
            self.optimizer,
            self.init_lr,
            self.cfg.task.trainer.get("use_lars", False),
            T_max=self.cfg.task.trainer.num_epochs
            if self.cfg.task.trainer.step_freq == "epoch"
            else total_steps,
            total_steps=total_steps,
        )

        if "loss" in self.cfg.task.trainer.keys():
            self.criterion = trainer_builder.build_loss(self.cfg.task.trainer.loss)

        self.train_metrics = None
        self.test_metrics = None

        self._scaler = None
        if self.cfg.task.trainer.get("mixed_precision"):
            self._scaler = torch.cuda.amp.GradScaler()

    @property
    def cfg(self) -> DictConfig:
        """Returns the config object."""
        return self._cfg

    @property
    def init_lr(self) -> float:
        """Returns the initial learning rate."""
        return self._init_lr

    @property
    def batch_size(self) -> int:
        """Returns the global batch size."""
        return self._batch_size

    @property
    def in_dist_mode(self):
        """Returns whether the task is running in distributed mode."""
        return (
            self.cfg.get("distributed") and dist_utils.is_dist_avail_and_initialized()
        )

    @property
    def device_id(self) -> int:
        """Returns the device id of the current process."""
        if self.in_dist_mode:
            device_id = torch.cuda.current_device()
        else:
            device_id = None
        return device_id

    @abstractmethod
    def prepare_input(self, *args, **kwargs) -> Tuple:
        """Prepares the input for the model forward pass."""
        raise NotImplementedError

    @abstractmethod
    def get_loss(self, *args, **kwargs) -> Union[float, torch.Tensor]:
        """Computes the loss for the current batch. Sum multiple losses if necessary."""

    @abstractmethod
    def get_train_metrics(self, *args, **kwargs) -> Dict[str, Any]:
        """Computes the metrics for the current batch during training."""

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Any:
        """Evaluates the model on the validation/test set."""

    @staticmethod
    def _load_state_dict(url: str) -> OrderedDict[str, Tensor]:
        """Loads a model's state_dict from the given URL.
        The URL can be a path to a file or a weblink.

        Args:
            url (str): Path to file or weblink

        Raises:
            FileNotFoundError: If the URL does not start with `http` or
            the file at the given path does not exist.

        Returns:
           OrderedDict[str, Tensor]: The state_dict, if successfully loaded.
        """
        if url.startswith("http") or os.path.isfile(url):
            if os.path.isfile(url):
                checkpoint = torch.load(url, map_location="cpu")
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
                if "state_dict" in state_dict.keys():
                    state_dict = state_dict["state_dict"]
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(url))
        return state_dict

    @staticmethod
    def _update_state_dict(
        state_dict: OrderedDict, replacement_key: str
    ) -> OrderedDict[str, Tensor]:
        """Update the given state dict by renaming the keys to match
        the current model where appropriate. This function is designed
        around MoCo v2, DenseCL and Faster-RCNN models.

        Args:
            state_dict (Dict): The(pretrained) model state_dict
            replacement_key (str): The string to append to or remove from the state_dict keys
            where necessary.

        Returns:
            Dict[str, Tensor]: The updated state_dict
        """
        new_state_dict = deepcopy(state_dict)
        for k in list(state_dict.keys()):
            # DenseCL
            if k.startswith("module.encoder_q.0") and not k.startswith(
                "module.encoder_q.1"
            ):
                new_state_dict[
                    k.replace("module.encoder_q.0.", replacement_key)
                ] = new_state_dict[k]
            elif k.startswith("encoder_q.0") and not k.startswith("encoder_q.1"):
                new_state_dict[
                    k.replace("encoder_q.0.", replacement_key)
                ] = new_state_dict[k]
            # official MoCov2
            elif k.startswith("module.encoder_q.") and not (
                k.startswith("module.encoder_q.fc")
                or k.startswith("module.encoder_q.1")
            ):
                new_state_dict[
                    k.replace("module.encoder_q.", replacement_key)
                ] = new_state_dict[k]
            elif k.startswith("encoder_q.") and not (
                k.startswith("encoder_q.fc") or k.startswith("encoder_q.1")
            ):
                new_state_dict[
                    k.replace("encoder_q.", replacement_key)
                ] = new_state_dict[k]
            # load the queue
            elif k.startswith("module.queue"):
                new_state_dict[k.replace("module.", "")] = new_state_dict[k]
            # SimSiam
            elif k.startswith("module.encoder.") and not k.startswith(
                "module.encoder.fc"
            ):
                new_state_dict[
                    k.replace("module.encoder.", replacement_key)
                ] = new_state_dict[k]
            elif k.startswith("encoder.") and not k.startswith("encoder.fc"):
                new_state_dict[k.replace("encoder.", replacement_key)] = new_state_dict[
                    k
                ]

            # plain encoder (with fc)
            # trained with DDP
            elif k.startswith("module.model."):
                new_state_dict[k.replace("module.model.", "")] = new_state_dict[k]
            elif k.startswith("module."):
                new_k = k.replace("module.", "")
                new_state_dict[new_k] = new_state_dict[k]
                # case example: plain ResNet trained with DDP
                if replacement_key != "" and not new_k.startswith(replacement_key):
                    new_state_dict[f"{replacement_key}{new_k}"] = new_state_dict[new_k]
                    del new_state_dict[new_k]
            # trained without DDP
            elif k.startswith("model."):
                # this is here to ensure that the weights of previous
                # implementations of Faster-RCNN can still be loaded.
                # It has keys like model.backbone.body
                # TODO: warn about this, but only once
                new_state_dict[k.replace("model", "")] = new_state_dict[k]
            elif k.startswith(replacement_key):
                new_state_dict[k.replace(replacement_key, "")] = new_state_dict[k]
            elif replacement_key != "" and not k.startswith(
                (replacement_key, "fc", "backbone.fpn", "rpn", "roi_heads")
            ):
                # append replacement_key
                new_state_dict[f"{replacement_key}{k}"] = new_state_dict[k]

        new_keys_set = set(list(new_state_dict.keys()))
        old_keys_set = set(list(state_dict.keys()))

        # remove renamed keys
        if len(new_keys_set - old_keys_set) > 0:
            for k in new_keys_set.intersection(old_keys_set):
                del new_state_dict[k]

        return new_state_dict

    @staticmethod
    def load_pretrained_checkpoint(
        model: torch.nn.Module, url: str, replacement_key: str
    ) -> torch.nn.Module:
        r"""
        Load a pretrained model from a checkpoint.

        Args:
            model (torch.nn.Module): model to load pretrained weights on to.
            url (str): URL of pretrained checkpoint
            replacement_key (str): The string to append to state_dict keys
            where necessary.

        Returns:
            model: model with pretrained weights loaded
        """
        print("Loading pretrained checkpoint: '{}'".format(url))
        try:
            state_dict = Task._load_state_dict(url)
            new_state_dict = Task._update_state_dict(state_dict, replacement_key)
            msg = model.load_state_dict(new_state_dict, strict=False)

            # sanity check
            print("Missing keys", msg.missing_keys)
            print("\nUnexpected keys", msg.unexpected_keys)
            if replacement_key != "":
                assert replacement_key not in "\t".join(msg.missing_keys)

            print("Loaded pre-trained model '{}'".format(url))
        except FileNotFoundError:
            print("File not found!")
            pass

        return model

    def load_model_to_device(self) -> None:
        """
        Wrap the model in DistributedDataParallel or DataParallel if using multiple
        GPUs, then load model to the GPU(s).
        """
        if self.in_dist_mode:
            if self.cfg.distributed.get("sync_bn"):
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            # set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            torch.cuda.set_device(self.device_id)
            self.model.cuda(self.device_id)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            world_size = dist_utils.get_world_size()
            self._batch_size = int(self._batch_size / world_size)  # !
            self._cfg.task.dataloader.num_workers = self.cfg.task.dataloader.get(
                "num_workers", 4
            )

            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.device_id]
            )
        elif torch.cuda.device_count() > 1:
            # DataParallel will divide and allocate batch_size to all available GPUs
            # self.model = torch.nn.DataParallel(self.model).cuda()
            raise RuntimeError("Please use DDP when with multiple GPUs")
        else:
            self.model.cuda()

    def get_model_parameters(self):
        """
        Returns the model parameter groups to be optimized.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_wandb_log_data(self, **kwargs) -> Dict[str, Any]:
        """Returns the data to be logged to wandb."""
        return kwargs

    def run(self, wandb_logger: Optional[Run] = None):
        """Run the task."""
        start_epoch = 0

        # resume from checkpoint, if specified
        if self.cfg.checkpoint.get("resume"):
            ckpt_path = self.cfg.checkpoint.resume
            if not ckpt_path.startswith("http") and not ntpath.isabs(ckpt_path):
                ckpt_path = to_absolute_path(ckpt_path)
                self._cfg.checkpoint["resume"] = ckpt_path
            start_epoch = self.resume_from_checkpoint(ckpt_path)

        if self.cfg.task.get("evaluate"):
            if self.in_dist_mode:
                self.test_loader.sampler.set_epoch(start_epoch)
            self.evaluate(wandb_logger, epoch=start_epoch)
            return

        val_freq = self.cfg.task.get("val_freq", -1)

        # train and/or evaluate
        num_epochs = self.cfg.task.trainer.num_epochs
        eval_only_at_end = self.cfg.task.get("eval_only_at_end", False)
        for epoch in range(start_epoch, num_epochs):
            if self.in_dist_mode:
                self.train_loader.sampler.set_epoch(epoch)

            # train for one epoch
            self.train_one_epoch(epoch, wandb_logger)

            # validate/test
            if (
                val_freq != -1
                and not eval_only_at_end
                and ((epoch + 1) % val_freq == 0 or (epoch + 1 == num_epochs))
            ):
                if self.test_loader is not None:
                    if self.in_dist_mode:
                        self.test_loader.sampler.set_epoch(epoch)
                    self.evaluate(wandb_logger, epoch=epoch)

            # update epoch-frequency lr schedulers
            if self.cfg.task.trainer.step_freq == "epoch":
                self.lr_scheduler.step()

            # save checkpoint on master process
            save_latest = self.cfg.checkpoint.get("keep_latest_only", True)
            if (
                save_latest
                or (epoch + 1) % self.cfg.checkpoint.get("save_freq", 1) == 0
            ):
                self.save_on_master(epoch, save_latest, start_epoch=start_epoch)
            if (epoch + 1) % num_epochs == 0 and eval_only_at_end:
                self.evaluate(wandb_logger, epoch=epoch)

    def train_one_epoch(
        self,
        epoch: int,
        wandb_logger: Optional[Run] = None,
        mode: Optional[str] = "train",
    ) -> None:
        """Train the model for one epoch.

        Args:
            epoch (int): current epoch
            wandb_logger (Optional[Run], optional): wandb logger. Defaults to None.
            mode (Optional[str], optional): train or eval. Defaults to "train".

        Raises:
            ValueError: if mode is not train or val

        Returns:
            None
        """
        if mode not in ["train", "eval"]:
            raise ValueError("mode must be train or eval")

        if mode == "eval":
            # eval mode impacts certain layers like batch_norm and dropout
            # useful for linear classification task
            self.model.eval()
        else:
            self.model.train()

        # configure metric logger
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = "Epoch: [{}]".format(epoch)

        current_step = epoch * len(self.train_loader)
        log_freq = self.cfg.get("log_freq", 1)
        for images, targets in metric_logger.log_every(
            self.train_loader, log_freq, header
        ):
            input = self.prepare_input(images=images, targets=targets)

            with torch.cuda.amp.autocast(enabled=self._scaler is not None):
                output = self.model(*input)
                loss = self.get_loss(outputs=output, targets=targets)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            # update gradients
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if not self.cfg.task.trainer.get("use_lars", False):
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.zero_grad()

            # update learning rate
            if self.cfg.task.trainer.step_freq == "batch":
                self.lr_scheduler.step()

            curr_lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss_value, lr=curr_lr)

            # compute performance metrics
            metrics = self.get_train_metrics(preds=output, targets=targets)
            if metrics is not None and isinstance(metrics, Dict):
                for key, value in metrics.items():
                    metric_logger.meters[key].update(value.item(), n=self.batch_size)

            current_step += 1
            if wandb_logger and (current_step % log_freq == 0):
                log_data = self.get_wandb_log_data(
                    **{"epoch": epoch + 1, "train/loss": loss, "learning_rate": curr_lr}
                )

                if metrics is not None and isinstance(metrics, Dict):
                    log_data.update(
                        {key: metric_logger.meters[key].global_avg for key in metrics}
                    )

                wandb_logger.log(data=log_data, step=current_step)

        # reset state for the next epoch
        if self.train_metrics is not None:
            self.train_metrics.reset()

    def resume_from_checkpoint(self, url: str) -> None:
        """
        Resume trained from a given checkpoint.

        Args:
            url (str): URL of pretrained checkpoint

        """
        start_epoch = 0
        if os.path.isfile(url):
            print("Resuming from checkpoint: '{}'".format(url))
            if self.device_id is None:
                map_loc = "cpu"
            else:
                # Map model to be loaded to specified single gpu.
                map_loc = "cuda:{}".format(self.device_id)
            checkpoint = torch.load(url, map_location=map_loc)
            start_epoch = checkpoint["epoch"]
            state_dict = checkpoint["state_dict"]

            for k in list(state_dict.keys()):
                if "mlp" in k and not "mlp_head" in k:
                    state_dict[k.replace("mlp", "mlp_head")] = state_dict[k]
                    del state_dict[k]

            # load checkpoint trained with DDP when not in dist mode
            for k in list(state_dict.keys()):
                if k.startswith("module.") and not self.in_dist_mode:
                    state_dict[k[len("module.") :]] = state_dict[k]
                    del state_dict[k]

            # load checkpoint saved without DDP for DDP-wrapped model
            for c_k, m_k in zip(
                list(state_dict.keys()), list(self.model.state_dict().keys())
            ):
                if m_k.startswith("module.") and not c_k.startswith("module."):
                    state_dict[f"module.{c_k}"] = state_dict[c_k]
                    del state_dict[c_k]

            self.model.load_state_dict(state_dict)
            if checkpoint.get("optimizer"):
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception as e:
                    print(e)
                    print(
                        "Hint: Make sure to freeze/unfreeze the same parameters as you did during training"
                    )
            if checkpoint.get("scheduler"):
                self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
            if self._scaler and checkpoint.get("scaler"):
                self._scaler.load_state_dict(checkpoint["scaler"])
            print("Loaded checkpoint '{}' (epoch {})".format(url, start_epoch))
        else:
            print("No checkpoint found at '{}'".format(url))
        return start_epoch

    def save_on_master(
        self, epoch: int, keep_latest_only: bool = True, **kwargs: Any
    ) -> None:
        """
        Save checkpoint on the master process.

        Args:
            epoch (int): Current epoch
            keep_latest_only (bool): Whether to keep only the latest checkpoint
            kwargs (Any): optional keyword arguments
        """
        if dist_utils.is_main_process():
            checkpoint_dir = self.cfg.checkpoint.get("dir", "checkpoints")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_dir = Path(checkpoint_dir).resolve()

            # remove `module` from state_dict keys to make it easier
            # to load pretrained weights without worrying about DDP or not
            state_dict = self.model.state_dict()
            for k in list(state_dict.keys()):
                if k.startswith("module."):
                    state_dict[k[len("module.") :]] = state_dict[k]
                    del state_dict[k]

            state = {
                "epoch": epoch + 1,
                "state_dict": state_dict,
                "scheduler": self.lr_scheduler.state_dict(),
                "optimizer": self.lr_scheduler.optimizer.state_dict(),
            }
            if self._scaler:
                state.update({"scaler": self._scaler.state_dict()})

            filename = checkpoint_dir.joinpath("checkpoint_{:04d}.pt".format(epoch + 1))
            latest_path = checkpoint_dir.joinpath("latest.pt")

            if not keep_latest_only:
                torch.save(state, filename)

                # create a symbolic link to the most recently save checkpoint
                if latest_path.is_symlink():
                    latest_path.unlink()
                latest_path.symlink_to(filename)
            else:
                torch.save(state, latest_path)
