"""Constrastive losses for MoCo and DenseCL."""
from typing import Any

import torch

from trainer.registry import LOSSES


@LOSSES.register_class
class ContrastiveLoss:
    """Contrastive loss.

    Args:
        temperature: float
            The temperature to use for the contrastive loss.

    """

    def __init__(self, temperature: float) -> None:
        """Init method."""
        self.temperature = temperature

    def _get_loss(self, pos_logits: torch.Tensor, neg_logits: torch.Tensor):
        """Get the loss.

        Args:
            pos_logits: torch.Tensor
                The positive logits.
            neg_logits: torch.Tensor
                The negative logits.

        Returns:
            The loss.

        """
        logits = torch.cat((pos_logits, neg_logits), dim=1)
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return torch.nn.functional.cross_entropy(logits, labels)

    def __call__(self, pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> Any:
        """Forward method."""
        return self._get_loss(pos_logits, neg_logits)


@LOSSES.register_class
class DenseCLLoss(ContrastiveLoss):
    """DenseCL loss.

    Args:
        temperature: float
            The temperature to use for the contrastive loss.
        loss_lambda: float
            The lambda to use for the weighted loss.

    """

    def __init__(self, temperature: float, loss_lambda: float) -> None:
        """Init method."""
        super().__init__(temperature=temperature)
        self.loss_lambda = loss_lambda
        self.temperature = temperature

    def __call__(
        self,
        mlp_pos_logits: torch.Tensor,
        mlp_neg_logits: torch.Tensor,
        dense_pos_logits: torch.Tensor,
        dense_neg_logits: torch.Tensor,
    ) -> Any:
        """Forward method."""
        mlp_loss = self._get_loss(mlp_pos_logits, mlp_neg_logits)
        dense_loss = self._get_loss(dense_pos_logits, dense_neg_logits)

        loss_dict = dict()
        loss_dict["mlp_loss"] = mlp_loss * (1 - self.loss_lambda)
        loss_dict["dense_loss"] = dense_loss * self.loss_lambda

        return loss_dict
