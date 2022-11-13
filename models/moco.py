from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig

from models import builder
from models.registry import MODELS


@MODELS.register_class
class MoCo(nn.Module):
    """Momentum Contrast. https://arxiv.org/abs/2003.04297v1.

    The code is adapted from:
    "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".

    Args:
        backbone (Union[Dict, DictConfig]): a hydra config object containing all
            the information needed to build the backbone.
        head (Union[Dict, DictConfig]): a hydra config object containing all
            the information needed to build the projection head.
        queue_len (Optional[int]): number of embeddings in the queue. Defaults to 65536.
        momentum (Optional[float]): momentum value. Default: 0.999.

    """

    def __init__(
        self,
        backbone: Union[Dict, DictConfig],
        head: Union[Dict, DictConfig],
        queue_len: Optional[int] = 65536,
        momentum: Optional[float] = 0.999,
    ) -> None:
        super(MoCo, self).__init__()

        self.encoder_q = self._build_encoder(backbone, head)
        self.encoder_k = self._build_encoder(backbone, head)

        self.queue_len = queue_len
        self.momentum = momentum

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(
                param_q.data
            )  # ensure that both encoders have identical parameters
            param_k.requires_grad = (
                False  # freeze the key encoder, since no gradient update is expected
            )

        # create the queue
        feat_dim = self.encoder_q[1].mlp_head.fc[-1].weight.shape[0]  # NOTE !!!
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @staticmethod
    def _build_encoder(backbone_cfg: DictConfig, projector_cfg: DictConfig):
        """Builds the encoder."""
        backbone = builder.build_backbone(backbone_cfg)
        backbone.fc = nn.Identity()
        projector = builder.build_head(projector_cfg)
        return nn.Sequential(backbone, projector)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the queue with the key embeddings."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.

        *** Only supports DistributedDataParallel (DDP) model. ***

        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle.

        *** Only supports DistributedDataParallel (DDP) model. ***

        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, query_img, key_img):
        """Forward function."""
        assert key_img is not None, "2 images are required, got 1"

        # compute query features
        query_embedding = self.encoder_q(query_img)
        query_embedding = nn.functional.normalize(query_embedding, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            key_img, idx_unshuffle = self._batch_shuffle_ddp(key_img)

            # compute key features
            key_embedding = self.encoder_k(key_img)
            key_embedding = nn.functional.normalize(key_embedding, dim=1)

            # undo shuffle
            key_embedding = self._batch_unshuffle_ddp(key_embedding, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        pos_logits = torch.einsum(
            "nc,nc->n", [query_embedding, key_embedding]
        ).unsqueeze(-1)
        # negative logits: NxK
        neg_logits = torch.einsum(
            "nc,ck->nk", [query_embedding, self.queue.clone().detach()]
        )

        self._dequeue_and_enqueue(key_embedding)

        return pos_logits, neg_logits


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.

    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
