"""Custom Loss Functions."""

import torch


class MSEBCELoss(torch.nn.Module):
    """A combined loss composed of a linear combionation of MSE Loss and BCEWithLogits Loss."""

    mse_loss_fn = torch.nn.MSELoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    def __init__(self, mse_weight: float, bce_weight: float) -> None:
        """Initialize losses and weight."""
        super().__init__()

        self.mse_weight = mse_weight
        self.bce_weight = bce_weight

    def forward(
        self, original: torch.Tensor, reconstructed: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Define how the losses are combined when called."""
        mse_loss = self.mse_loss_fn(reconstructed, original) * self.mse_weight
        bce_loss = self.bce_loss_fn(prediction, target) * self.bce_weight
        loss = mse_loss + bce_loss
        return loss

    def mse_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """MSE loss part."""
        return self.mse_loss_fn(reconstructed, original)

    def mse_loss_weighted(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Weighted MSE loss part."""
        return self.mse_loss_fn(reconstructed, original) * self.mse_weight

    def bce_loss(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """BCE loss part."""
        return self.bce_loss_fn(target, prediction)

    def bce_loss_weighted(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """Weighted BCE loss part."""
        return self.bce_loss_fn(target, prediction) * self.bce_weight
