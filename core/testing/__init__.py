from .metrics_tester import *


"""
Testing module.

This module contains helper utilities for checking saved models after training.
Its purpose is to run inference on candle windows, recompute metrics for stored
models, and provide simple aggregated views of model predictions and results.

Main components
---------------
metrics_tester.py
    Metric evaluation helpers for already saved models.

    Main responsibilities:
    - load saved models from disk
    - evaluate them on a selected candle window
    - update metrics inside each ModelState
    - print the best model per category for quick inspection

    Main entry points:
    - evaluate_models_metrics(...)
    - test_saved_models_metrics_pipeline(...)

model_predictor.py
    Prediction helpers for loaded model collections.

    Main responsibilities:
    - build prediction tables for one category
    - build prediction tables for all categories
    - flatten all model predictions into one dataframe
    - provide small helpers for counting active predictions
      and selecting predictions for a single candle row

    Main entry points:
    - category_predictions(...)
    - categories_predictions(...)
    - all_models_predictions(...)
    - count_active_predictions(...)
    - build_row_predictions(...)

Module purpose
--------------
The testing module is a lightweight inspection layer on top of saved models.

Typical flow:
1. Load a processed dataset or candle window
2. Load saved models from disk
3. Run predictions or metric evaluation on the same candle set
4. Inspect results by category, by model, or by candle row

Design notes
------------
- metrics_tester.py focuses on evaluation of saved models
- model_predictor.py focuses on prediction tables and aggregation
- the module is meant for analysis and validation workflows,
  not for training or persistence itself
"""