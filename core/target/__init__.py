from .builder import Builder

"""
Target module.

This module contains the dataset-to-training-target preparation layer of the project.
Its job is to take already processed feature data and convert it into model-ready
training windows, filtered samples, and final X / y pairs.

Main components
---------------
slicing.py
    Dataset slicing utilities.

    Main responsibilities:
    - split processed datasets by candle counts
    - split processed datasets by explicit date ranges
    - provide a unified entry point for slicing through config dictionaries

    Main entry points:
    - cut_by_date(...)
    - cut_by_date_ranges(...)
    - split_by_date_ranges(...)
    - split_by_candles(...)
    - slice_data(...)

filter.py
    Feature selection and sample filtering utilities.

    Main responsibilities:
    - load and resolve feature groups
    - build strict feature whitelists
    - keep only selected feature columns
    - apply deadzone filtering on target returns
    - apply regime-based row filtering

    Main entry points:
    - load_feature_groups(...)
    - resolve_feature_columns(...)
    - keep_only_selected_features(...)
    - apply_deadzone(...)
    - apply_deadzone_windows(...)
    - apply_regime_filter(...)
    - apply_regime_filter_windows(...)

builder.py
    High-level interface for constructing final training inputs.

    Main responsibilities:
    - compute future-return targets
    - apply optional regime filtering
    - apply optional deadzone filtering
    - apply strict feature whitelist selection
    - build final X feature matrices and binary y targets
    - support both single-window and multi-window workflows

    Main entry points:
    - Builder.build(...)
    - Builder.build_windows(...)

Module purpose
--------------
The target module acts as the bridge between processed feature data
and the model-training layer.

Typical flow:
1. A processed dataset is loaded
2. slicing.py defines train / validation / test windows
3. filter.py applies feature and sample-level filtering rules
4. builder.py converts filtered windows into final X / y pairs

Design notes
------------
- slicing.py controls where the data is taken from
- filter.py controls which rows and columns are allowed
- builder.py combines these pieces into model-ready datasets
- this separation keeps the pipeline easier to debug, test, and extend
"""