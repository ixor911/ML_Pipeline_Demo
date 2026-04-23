"""
Training module.

This module contains the model training layer of the project.
Its purpose is to take prepared feature windows, train a model,
compute category-specific thresholds, and wrap the result into
runtime-ready ModelState objects.

Main components
---------------
training_engine.py
    Small helper layer for building ModelState objects from a trained model.

    Main responsibilities:
    - create one ModelState for a single evaluation category
    - create multiple ModelState objects from a threshold dictionary
    - attach category-specific thresholds and metadata to the same trained model

    Main entry points:
    - create_state(...)
    - create_states(...)

training_pipeline.py
    High-level end-to-end training pipeline.

    Main responsibilities:
    - prepare train / validation data from processed datasets
    - train a TorchModel on the prepared feature windows
    - compute validation probabilities
    - search for the best threshold per evaluation category
    - create ModelState objects for all discovered categories

    Main entry points:
    - prepare_data(...)
    - training_pipeline(...)
    - training_pipeline_list(...)

Module purpose
--------------
The training module is the bridge between prepared datasets
and runtime model objects.

Typical flow:
1. A processed dataset is sliced into train / validation windows
2. Builder converts those windows into X / y pairs
3. TorchModel is trained on the train split
4. Validation probabilities are used to find the best thresholds
5. The trained model is wrapped into one or more ModelState objects

Design notes
------------
- training_pipeline.py handles the full end-to-end training flow
- training_engine.py handles lightweight state construction
- one trained model can produce multiple ModelState objects,
  each tied to a different evaluation category and threshold
- the demo version focuses only on the initial training flow,
  without full retraining lifecycle logic
"""