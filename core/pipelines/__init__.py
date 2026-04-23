from .candidate_batch_pipeline import *

"""
Pipelines module.

This module contains high-level orchestration flows built on top of the lower-level
project components. Its purpose is to connect data preparation, training, evaluation,
selection, and testing into reusable end-to-end workflows.

Main components
---------------
candidate_batch_pipeline.py
    Runs one full training-config pass and evaluates all produced candidates.

    Main responsibilities:
    - train one base model from a single training config
    - expand it into category-specific ModelState candidates
    - evaluate each candidate on shared test candles
    - pass candidates through the selector
    - update the in-memory best-model store

    Main entry point:
    - candidate_batch_pipeline(...)

grid_search_pipeline_async.py
    Asynchronous grid-search pipeline over many training configs.

    Main responsibilities:
    - load or receive many training configs
    - distribute training work across worker processes
    - evaluate candidate states on shared test candles
    - feed evaluated candidates into the selector
    - maintain the in-memory model pool during the search
    - optionally print and export final RAM metrics

    Main entry points:
    - grid_search_pipeline_async(...)
    - print_models_ram_metrics(...)

Module purpose
--------------
The pipelines module is the orchestration layer of the project.

Typical flow:
1. A training config or a grid of configs is prepared
2. One or more pipelines call the lower-level training code
3. Candidate ModelState objects are evaluated on test candles
4. Selector logic decides which candidates are kept
5. The resulting model pool is returned for further analysis or saving

Design notes
------------
- pipelines are intentionally built on top of lower-level modules
  such as training, eval, io, target, and testing
- candidate_batch_pipeline.py is a compact single-config workflow
- grid_search_pipeline_async.py is a multi-config parallel workflow
- the demo version focuses on training-and-selection orchestration,
  without the full production lifecycle around model management
"""