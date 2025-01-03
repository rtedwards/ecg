"""Modelling code."""

from collections import OrderedDict

import torch
from torch import nn


class MLPAutoEncoder(nn.Module):
    """4 layer MLP autoencoder."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Define the model."""
        super().__init__()
        self.input_dim = input_dim  # (188)
        self.hidden_dim = hidden_dim

        # Building an linear encoder with Linear layer followed by Relu activation function
        # 188 ==> 16
        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(self.input_dim, 128)),
                    ("relu1", nn.ReLU()),
                    ("linear2", nn.Linear(128, 64)),
                    ("relu2", nn.ReLU()),
                    ("linear3", nn.Linear(64, 32)),
                    ("relu3", nn.ReLU()),
                    ("linear4", nn.Linear(32, 16)),
                ]
            )
        )

        # Building an linear decoder with Linear layer followed by Relu activation function
        # The Sigmoid activation function outputs the value between 0 and 1
        # 16 ==> 188
        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(16, 32)),
                    ("relu1", nn.ReLU()),
                    ("linear2", nn.Linear(32, 64)),
                    ("relu2", nn.ReLU()),
                    ("linear3", nn.Linear(64, 128)),
                    ("relu3", nn.ReLU()),
                    ("linear4", nn.Linear(128, self.input_dim)),
                    ("activation", torch.nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvAutoEncoder(nn.Module):
    """Convolutional autoencoder with 3 layers.

    Includes a classifier on the encoded embeddings.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, stride: int) -> None:
        """Define the model."""
        super().__init__()
        self.input_dim = input_dim  # (188)
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size  # how to choose kernel size?
        self.stride = 1

        # Building an linear encoder with Linear layer followed by Relu activation function
        # 188 ==> 16

        assert self.kernel_size % 2 != 0  # and (stride == 1)
        pool_padding = (kernel_size - 1) // 2

        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    # Conv1d Layer 1: Input Channels = 1, Output Channels = hidden_features
                    ("conv1", nn.Conv1d(1, self.hidden_dim, kernel_size=kernel_size, stride=self.stride)),
                    ("relu1", nn.ReLU()),
                    ("norm1", nn.BatchNorm1d(self.hidden_dim)),
                    ("pool1", nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding=pool_padding)),
                    # Conv1d Layer 2: Input Channels = hidden_features, Output Channels = hidden_features
                    ("conv2", nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, stride=self.stride)),
                    ("relu2", nn.ReLU()),
                    ("norm2", nn.BatchNorm1d(self.hidden_dim)),
                    ("pool2", nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding=pool_padding)),
                    # Conv1d Layer 2: Input Channels = hidden_features, Output Channels = hidden_features
                    ("conv3", nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, stride=self.stride)),
                    ("relu3", nn.ReLU()),
                    ("norm3", nn.BatchNorm1d(self.hidden_dim)),
                    ("pool3", nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding=pool_padding)),
                ]
            )
        )

        # Classifier - might need to add more units
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 179 for 1 convolutions, 171 for 2 convolutions, 163 for 3 convolutions
            nn.Linear(self.hidden_dim * 163, 1),
            # nn.ReLU(),
            # nn.Linear(1000, 1),
            # nn.Dropout(0.2),
            # nn.Sigmoid(),
        )
        self.activation = nn.Sigmoid()

        # Building an linear decoder with Linear layer followed by Relu activation function
        # The Sigmoid activation function outputs the value between 0 and 1
        # 16 ==> 188
        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    # Conv1d Layer 1: Input Channels = hidden_features, Output Channels = hidden_features
                    (
                        "conv1",
                        nn.ConvTranspose1d(
                            self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, stride=self.stride
                        ),
                    ),
                    ("relu1", nn.ReLU()),
                    ("norm1", nn.BatchNorm1d(self.hidden_dim)),
                    # Conv1d Layer 1: Input Channels = hidden_features, Output Channels = hidden_features
                    (
                        "conv2",
                        nn.ConvTranspose1d(
                            self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, stride=self.stride
                        ),
                    ),
                    ("relu2", nn.ReLU()),
                    ("norm2", nn.BatchNorm1d(self.hidden_dim)),
                    # Conv1d Layer 1: Input Channels = hidden_features, Output Channels = 1
                    ("conv3", nn.ConvTranspose1d(self.hidden_dim, 1, kernel_size=kernel_size, stride=self.stride)),
                    ("relu3", nn.ReLU()),
                    ("norm3", nn.BatchNorm1d(1)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model."""
        # our input is (n_batch, length_sequence)
        # we need to add a dimension for n_channels (n_batch, n_channels, length_sequence) for the Conv1D layers
        encoded = self.encoder(x[:, None, :])
        pred = self.classifier(encoded)
        decoded = self.decoder(encoded)
        return decoded.squeeze(), pred

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Classify the input."""
        encoded = self.encoder(x[:, None, :])
        pred = self.classifier(encoded)
        probabilities = self.activation(pred)
        predicted_labels = torch.where(probabilities >= threshold, 1.0, 0.0)
        return predicted_labels

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a sample.

        TODO: how to do this?  This will just reconstruct the input, not create novel generations.
        """
        encoded = self.encoder(x[:, None, :])
        decoded = self.decoder(encoded)
        return decoded.squeeze()
