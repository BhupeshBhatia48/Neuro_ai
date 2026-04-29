"""
deep_learning/dl_utils.py
---------------------------
Utilities for deep learning data preparation.

Issue #14 : Sequence windowing for time-series,
            DataLoader creation, GPU detection.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger


def detect_device() -> torch.device:
    """
    Issue #14: Detects best available device.
    Returns cuda if available, mps (Apple Silicon) as second choice, else cpu.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"[DL] GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("[DL] Apple Silicon MPS detected.")
    else:
        device = torch.device("cpu")
        logger.info("[DL] No GPU found — using CPU.")
    return device


def create_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    window_size: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Issue #14: Sliding window for time-series/sequence data.

    Parameters
    ----------
    data        : 2D array (n_samples, n_features)
    targets     : 1D array (n_samples,)
    window_size : Number of time steps per input window

    Returns
    -------
    X_seq : (n_windows, window_size, n_features)
    y_seq : (n_windows,)
    """
    X_seq, y_seq = [], []
    for i in range(len(data) - window_size):
        X_seq.append(data[i : i + window_size])
        y_seq.append(targets[i + window_size])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
    device: torch.device = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Issue #14: Creates PyTorch DataLoaders from numpy arrays.

    Parameters
    ----------
    X_train, y_train : Training arrays
    X_test, y_test   : Test arrays
    batch_size       : Mini-batch size
    device           : Target device (tensors moved here)

    Returns
    -------
    (train_loader, test_loader)
    """
    device = device or detect_device()

    def _to_tensor(arr):
        t = torch.tensor(arr, dtype=torch.float32)
        return t.to(device)

    train_ds = TensorDataset(_to_tensor(X_train), _to_tensor(y_train))
    test_ds  = TensorDataset(_to_tensor(X_test),  _to_tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    logger.info(
        f"[DL] DataLoaders created — "
        f"train_batches={len(train_loader)}  test_batches={len(test_loader)}  "
        f"batch_size={batch_size}  device={device}"
    )
    return train_loader, test_loader
