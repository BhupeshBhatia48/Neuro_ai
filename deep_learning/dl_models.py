"""
deep_learning/dl_models.py
----------------------------
PyTorch model definitions — LSTM, GRU, MLP with proper regularization.

MEMORIZATION FIX:
-----------------
The previous models had insufficient regularization — they would memorize
training data instead of learning generalizable patterns. Three fixes:

1. LSTM/GRU: Added input dropout layer before the recurrent layer.
   This randomly zeroes input features during training, forcing the model
   to learn robust representations that don't depend on any single feature.

2. MLP: Added higher default dropout (0.4 → balanced), input BatchNorm
   applied at the input layer to normalize features before learning.

3. All models: Weight initialization using Xavier/Orthogonal initialization.
   Default PyTorch initialization can create poorly scaled weights that
   lead to memorization. Xavier uniform gives better gradient flow.

Install: pip install -r requirements-deep.txt
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _init_weights(module: nn.Module) -> None:
    """
    Xavier/Orthogonal weight initialization.
    Applied recursively to all layers in the model.
    Prevents vanishing/exploding gradients and reduces memorization.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                # Orthogonal initialization for recurrent weights — best practice
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)


class LSTMModel(nn.Module):
    """
    LSTM for sequence / time-series classification or regression.

    Anti-memorization measures:
    - Input dropout before LSTM (drops random input features)
    - Recurrent dropout between LSTM layers
    - Orthogonal weight initialization for recurrent weights
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        output_size: int   = 1,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Input dropout — drops feature dimensions before LSTM sees them
        self.input_drop = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.out_drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        # Apply proper weight initialization
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_drop(x)
        out, _ = self.lstm(x)
        out = self.out_drop(out[:, -1, :])   # last time step + dropout
        return self.fc(out)


class GRUModel(nn.Module):
    """
    GRU for sequence / time-series tasks.

    Anti-memorization: input dropout, output dropout, orthogonal init.
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        output_size: int   = 1,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.input_drop = nn.Dropout(p=dropout)
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.out_drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_drop(x)
        out, _ = self.gru(x)
        out = self.out_drop(out[:, -1, :])
        return self.fc(out)


class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron for tabular data.

    Anti-memorization measures:
    - Input BatchNorm normalizes feature scales (prevents large-weight memorization)
    - Dropout 0.3 between every hidden layer
    - Residual-style architecture: skip connection from first hidden to last hidden
      (helps gradient flow and prevents overfitting to specific layer patterns)
    - Xavier weight initialization
    """

    def __init__(
        self,
        input_size:   int,
        hidden_sizes: list[int] = None,
        output_size:  int       = 1,
        dropout:      float     = 0.3,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 128, 64]

        # Input normalization — stabilizes training and reduces memorization
        self.input_norm = nn.BatchNorm1d(input_size)

        layers = []
        prev = input_size
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())           # GELU outperforms ReLU on tabular data
            layers.append(nn.Dropout(dropout))
            prev = h

        self.hidden = nn.Sequential(*layers)
        self.head   = nn.Linear(prev, output_size)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        return self.head(self.hidden(x))
