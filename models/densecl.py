from typing import Optional

import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig

from models import builder
from models.moco import MoCo, concat_all_gather
from models.registry import MODELS


@MODELS.register_class
class DenseCL(MoCo):
    """Dense Contrastive Learning. https://arxiv.org/abs/2011.09157.

    Args:
        backbone (Union[Dict, DictConfig]): a hydra config object containing all
            the information needed to build the backbone.
        head (Union[Dict, DictConfig]): a hydra config object containing all
            the information needed to build the projection head.
        queue_len (Optional[int]): number of embeddings in the queue. Defaults to 65536.
        momentum (Optional[float]): momentum value. Defaults to 0.999.

    """

    def __init__(
        self,
        backbone: DictConfig,
        head: DictConfig,
        queue_len: Optional[int] = 65536,
        momentum: Optional[float] = 0.999,
    ) -> None:
        super().__init__(backbone, head, queue_len, momentum)

        # create the second queue for dense output
        _global_feat_dim = self.encoder_q[1].mlp_head.fc[-1].weight.shape[0]
        _dense_feat_dim = self.encoder_q[1].dense_head.proj[-1].weight.shape[0]
        assert _global_feat_dim == _dense_feat_dim
        feat_dim = _global_feat_dim

        self.register_buffer("queue2", torch.randn(feat_dim, queue_len))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))

    @staticmethod
    def _build_encoder(backbone_cfg: DictConfig, projector_cfg: DictConfig):
        """Builds the encoder."""
        backbone = builder.build_backbone(backbone_cfg)
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()
        projector = builder.build_head(projector_cfg)
        return nn.Sequential(backbone, projector)

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        """Dequeue and enqueue the (pooled) keys from the dense pathway."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr : ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue2_ptr[0] = ptr

    def forward(self, query_img, key_img):
        """The forward pass of the model."""
        assert key_img is not None, "2 images are required, got 1"

        # compute query features
        query_feat_map = self.encoder_q[0](query_img)  # backbone feature map
        query_mlp_proj, query_dense_proj, _ = self.encoder_q[1](
            query_feat_map
        )  # NxC; NxCxS^2

        query_feat_map = query_feat_map.view(
            query_feat_map.size(0), query_feat_map.size(1), -1
        )
        query_feat_map = nn.functional.normalize(query_feat_map, dim=1)

        query_mlp_proj = nn.functional.normalize(query_mlp_proj, dim=1)
        query_dense_proj = nn.functional.normalize(query_dense_proj, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            key_img, idx_unshuffle = self._batch_shuffle_ddp(key_img)

            # compute key features
            key_feat_map = self.encoder_k[0](key_img)
            key_mlp_proj, key_dense_proj, key_dense_pooled = self.encoder_k[1](
                key_feat_map
            )  # NxC; NxCxS^2; NxC

            key_feat_map = key_feat_map.view(
                key_feat_map.size(0), key_feat_map.size(1), -1
            )
            key_feat_map = nn.functional.normalize(key_feat_map, dim=1)

            key_mlp_proj = nn.functional.normalize(key_mlp_proj, dim=1)
            key_dense_proj = nn.functional.normalize(key_dense_proj, dim=1)
            key_dense_pooled = nn.functional.normalize(key_dense_pooled, dim=1)

            # undo shuffle
            key_feat_map = self._batch_unshuffle_ddp(key_feat_map, idx_unshuffle)
            key_mlp_proj = self._batch_unshuffle_ddp(key_mlp_proj, idx_unshuffle)
            key_dense_proj = self._batch_unshuffle_ddp(key_dense_proj, idx_unshuffle)
            key_dense_pooled = self._batch_unshuffle_ddp(
                key_dense_pooled, idx_unshuffle
            )

        # compute global logits
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [query_mlp_proj, key_mlp_proj]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [query_mlp_proj, self.queue.clone().detach()])

        # establish visual correspondence using the backbone feature maps
        backbone_sim_matrix = torch.einsum("nca,ncb->nab", query_feat_map, key_feat_map)
        dense_sim_idx = backbone_sim_matrix.argmax(dim=2)  # NxS^2
        indexed_k_grid = torch.gather(
            key_dense_proj,
            dim=2,
            index=dense_sim_idx.unsqueeze(1).expand(-1, key_dense_proj.size(1), -1),
        )  # NxCxS^2

        # compute the dense logits
        # positive logits
        densecl_sim_q = torch.einsum(
            "nca,nca->na", query_dense_proj, indexed_k_grid
        )  # NxS^2
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1)  # NS^2X1

        query_dense_proj = query_dense_proj.permute(0, 2, 1)
        query_dense_proj = query_dense_proj.reshape(-1, query_dense_proj.size(2))
        l_neg_dense = torch.einsum(
            "nc,ck->nk", [query_dense_proj, self.queue2.clone().detach()]
        )

        self._dequeue_and_enqueue(key_mlp_proj)
        self._dequeue_and_enqueue2(key_dense_pooled)

        return l_pos, l_neg, l_pos_dense, l_neg_dense
