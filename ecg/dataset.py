"""Dataset related code."""

import polars as pl  # is this needed?
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """Dataset to sample cases from the training dataset.

    TODO: randomly sample variable lengths
    """

    def __init__(self, X: pl.DataFrame, y: pl.DataFrame) -> None:
        """Convert data and target."""
        self.data = X
        self.target = y
        self.X = X.to_torch().to(torch.float32)
        self.y = y.to_torch().to(torch.float32)

    def __len__(self) -> int:
        """Dataset length (number of samples)."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve the idxth sample."""
        return self.X[idx], self.y[idx]

    # def get_row(self, idx: int) -> pl.DataFrame:
    #     """Retrieve the idxth sample as a dataframe."""
    #     return pl.concat([self.target, self.data], how="horizontal")
