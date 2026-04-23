"""
IO module.

This module contains the input / output layer of the project.
Its purpose is to load configs and datasets, persist trained models to disk,
restore saved models back into memory, and store evaluation snapshots.

Main components
---------------
loader.py
    General loading utilities for configs and datasets.

    Main responsibilities:
    - validate file paths
    - load JSON configs
    - expand grid configs into multiple combinations
    - load tabular datasets from csv / parquet / json / excel
    - load raw and processed project data from standard directories

    Main entry points:
    - load_json(...)
    - load_config(...)
    - load_config_grid(...)
    - load_dataframe(...)
    - load_raw_data(...)
    - load_processed_data(...)

model_saver.py
    Model persistence utilities.

    Main responsibilities:
    - generate unique model IDs
    - save model weights
    - save model metadata
    - create a new saved model entry
    - replace or delete saved models

    Main entry points:
    - generate_model_id(...)
    - save_model(...)
    - save_meta(...)
    - create_model(...)
    - delete_model(...)
    - replace_model(...)

model_loader.py
    Saved model loading utilities.

    Main responsibilities:
    - list categories and model IDs
    - locate model folders on disk
    - load model metadata
    - restore saved models as ModelState objects
    - load all models by category or across the full storage

    Main entry points:
    - get_all_categories(...)
    - get_category_ids(...)
    - find_model_dir(...)
    - load_model(...)
    - load_category_models(...)
    - load_all_models(...)

model_test_saver.py
    Evaluation snapshot persistence.

    Main responsibilities:
    - create unique test result filenames
    - save single evaluation results
    - save eval payloads for one model
    - save eval payloads for all loaded models in RAM

    Main entry points:
    - create_test_name(...)
    - create_test(...)
    - eval_tests(...)
    - eval_models(...)

model_selector.py
    In-memory model selection logic.

    Main responsibilities:
    - compare candidate models against saved category members
    - detect duplicates by structural signature
    - compute selector scores from evaluation metrics
    - keep only the strongest models per category
    - replace weaker saved models when better candidates appear

    Main entry points:
    - eval_model_category(...)
    - eval_model(...)
    - compare_model(...)
    - find_duplicate_by_signature(...)

Module purpose
--------------
The io module is the persistence and loading layer of the project.

Typical flow:
1. loader.py loads configs and processed datasets
2. training code creates candidate ModelState objects
3. model_selector.py decides whether a candidate should be kept
4. model_saver.py persists accepted models to disk
5. model_loader.py restores saved models later for inference or evaluation
6. model_test_saver.py stores evaluation snapshots for tracking results

Design notes
------------
- loader.py handles external files and config expansion
- model_saver.py and model_loader.py handle model lifecycle on disk
- model_test_saver.py handles evaluation result storage
- model_selector.py handles selection logic before persistence
- this separation keeps storage, loading, and selection responsibilities clean
"""