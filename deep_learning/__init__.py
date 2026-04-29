"""
deep_learning/
--------------
Experimental deep learning training pipeline.

Issue #2  : Deep learning support marked as EXPERIMENTAL.
Issue #14 : Dedicated DL trainer with GPU detection, batching,
            sequence windowing, and torch-based abstraction.

Modules
-------
dl_trainer.py   — Main DL trainer class (GPU-aware, batch training)
dl_models.py    — PyTorch model definitions (LSTM, GRU, MLP, CNN1D)
dl_utils.py     — Sequence windowing, DataLoader helpers

WARNING: These modules require PyTorch.
         Install with: pip install -r requirements-deep.txt
"""

try:
    from deep_learning.dl_trainer import DeepLearningTrainer, DLTrainingResult
    from deep_learning.dl_models import LSTMModel, GRUModel, MLPModel
    __all__ = ["DeepLearningTrainer", "DLTrainingResult", "LSTMModel", "GRUModel", "MLPModel"]
except ImportError:
    __all__ = []
