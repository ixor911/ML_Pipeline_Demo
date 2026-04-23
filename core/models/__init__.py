from .TorchModel import TorchModel
from .ModelState import ModelState

"""
Models module.

This module contains the model layer of the project:
the neural network implementation itself and the runtime wrapper used to
store model state, metadata, thresholds, and evaluation results.

Main components
---------------
TorchModel.py
    PyTorch-based tabular neural network for classification, regression,
    or multi-task learning.

    Main responsibilities:
    - build and train a small MLP for tabular features
    - handle preprocessing internally:
        * categorical one-hot encoding
        * boolean casting
        * robust scaling with median and IQR
    - support inference and persistence
    - provide a compact end-to-end interface for model training and prediction

    Main classes / functions:
    - TorchModel
    - _MLP
    - set_seed(...)

ModelState.py
    Runtime wrapper around a fitted model and its metadata.

    Main responsibilities:
    - store model identity, category, threshold, and filesystem path
    - store training / builder metadata
    - prepare candles into model-ready features via Builder
    - run inference through the underlying model
    - evaluate metrics using the configured threshold
    - manage model signatures for deduplication
    - support saving new or existing model states

    Main class:
    - ModelState

Module purpose
--------------
The models module is the bridge between engineered features and downstream
decision logic.

Typical flow:
1. A processed dataset is built from raw candles
2. TorchModel is trained on feature matrices
3. The trained model is wrapped into ModelState
4. ModelState is then used for:
   - inference
   - thresholded predictions
   - evaluation
   - persistence

Design notes
------------
- TorchModel focuses on neural network training and inference.
- ModelState focuses on runtime integration, metadata, and lifecycle management.
- This separation keeps the core neural network logic independent from the
  higher-level orchestration and storage logic.
"""