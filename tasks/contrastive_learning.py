from copy import deepcopy
from typing import NamedTuple, Tuple, Union

import torch.nn
from omegaconf import DictConfig

import utils.distributed as dist_utils
from tasks.registry import TASKS

from .base import Task


@TASKS.register_class
class ContrastiveLearning(Task):
    """Task for contrastive learning.

    Args:
        cfg (DictConfig): Hydra config object.

    """

    def __init__(self, cfg: DictConfig) -> None:
        if (
            cfg.get("distributed") is None
            or not dist_utils.is_dist_avail_and_initialized()
        ):
            raise RuntimeError(
                "Only DistributedDataParallel is supported for this task."
            )

        super().__init__(cfg, state_dict_replacement_key="encoder_q.0.")

    @staticmethod
    def load_pretrained_checkpoint(
        model: torch.nn.Module, url: str, replacement_key: str
    ) -> Tuple[torch.nn.Module, NamedTuple]:
        r"""Load a pretrained model from a checkpoint.

        Args:
            model (torch.nn.Module): model to load pretrained weights on to.
            url (str): URL of pretrained checkpoint

        Returns:
            model: model with pretrained weights loaded
            msg: a NamedTuple containing missing keys and unexpected keys

        """
        print("Loading pretrained checkpoint: '{}'".format(url))
        msg = None
        try:
            state_dict = Task._load_state_dict(url)
            state_dict_q = Task._update_state_dict(
                state_dict, replacement_key
            )  # get keys for queue encoder

            # get keys for key encoder
            state_dict_k = deepcopy(state_dict_q)
            replacement_key_k = replacement_key.replace("q", "k")
            for k in list(state_dict_k.keys()):
                if not replacement_key == "" and replacement_key in k:
                    state_dict_k[
                        k.replace(replacement_key, replacement_key_k)
                    ] = state_dict_k[k]
                    del state_dict_k[k]

            # load combined state_dict
            new_state_dict = {**state_dict_q, **state_dict_k}
            msg = model.load_state_dict(new_state_dict, strict=False)

            # sanity check
            print("Missing keys", msg.missing_keys)
            print("\nUnexpected keys", msg.unexpected_keys)
            assert replacement_key not in "\t".join(msg.missing_keys) and (
                replacement_key_k not in "\t".join(msg.missing_keys)
            )

            # ensure that both encoders have the same parameters (to start with)
            for param_q, param_k in zip(
                model.encoder_q.parameters(), model.encoder_k.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            print("Loaded pre-trained model '{}'".format(url))
        except FileNotFoundError:
            pass

        return model

    def prepare_input(self, **kwargs) -> Tuple:
        images = kwargs["images"]
        images[0] = images[0].cuda(self.device_id, non_blocking=True)
        images[1] = images[1].cuda(self.device_id, non_blocking=True)
        return images[0], images[1]

    def get_loss(self, **kwargs) -> Union[float, torch.Tensor]:
        outputs = kwargs["outputs"]
        loss = self.criterion(*outputs)
        if isinstance(loss, dict):
            loss = sum(loss for loss in loss.values())
        return loss

    def get_train_metrics(self, *args, **kwargs) -> None:
        return None

    def evaluate(self, *args, **kwargs) -> None:
        return None
