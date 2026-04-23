# ML Crypto Signal System — Demo

A modular machine learning pipeline for crypto signal generation, threshold optimization, model selection, and strategy research.

This repository is a **demo version** of a larger private project. It showcases the core architecture and main ML workflow while intentionally simplifying or removing some production-oriented parts.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ixor911/ML_Pipeline_Demo/blob/master/demo.ipynb)

---

## What this project does

The system is built around the idea that ML models should produce **probabilities and signals**, while higher-level logic decides how to use those signals.

In the current demo, the project shows how to:

- download and load market data
- build engineered feature sets for ETH, BTC, and cross-market context
- construct training targets from processed candles
- train tabular PyTorch models
- search for the best thresholds for different evaluation categories
- wrap trained models into reusable runtime states
- evaluate and compare multiple candidates
- run asynchronous grid search over many training configurations
- inspect model predictions and saved-model metrics

---

## Demo scope

This repository is **not** the full system. The demo focuses on the parts that best show the ML pipeline itself.

### Included in the demo

- feature engineering and preprocessing
- target building
- model training
- threshold evaluation
- candidate selection
- async grid search
- testing helpers for metrics and predictions

### Simplified or removed in the demo

To keep the demo safer and easier to understand, some persistence-heavy or infrastructure-heavy parts were reduced or disabled.

In particular, the demo removes or simplifies:

- model persistence workflows that actively create folders and files during normal runs
- direct interaction with the full filesystem-based model storage flow
- log generation
- Excel result generation
- convenience pipelines that are more about system maintenance than ML logic

The goal is to show the **core ML architecture**, not every operational detail of the private full version.

---

## High-level architecture

The project is split into focused modules.

```text
core/
  eval/
  experiments/
  features/
  io/
  models/
  pipelines/
  target/
  testing/
  training/
  utils/
```

### `features/`
Feature engineering layer.

Responsible for turning raw OHLCV candles into processed feature datasets.

Key parts:
- `DataProvider.py` — raw Binance data loading / reading
- `preprocessor/ETH.py` — ETH feature engineering
- `preprocessor/BTC.py` — BTC + ETHBTC feature engineering
- `preprocessor/Preprocessor.py` — unified preprocessing interface

### `target/`
Target-building layer.

Responsible for slicing datasets, filtering rows and columns, and building final `X / y` pairs.

Key parts:
- `slicing.py` — split data into windows
- `filter.py` — feature whitelist and row filtering
- `builder.py` — create final train-ready datasets

### `models/`
Model layer.

Responsible for the neural network itself and for the runtime wrapper around trained models.

Key parts:
- `TorchModel.py` — tabular PyTorch model with built-in preprocessing
- `ModelState.py` — trained-model runtime wrapper with metadata and threshold

### `eval/`
Evaluation layer.

Responsible for threshold-based metrics, constraint checking, and category-specific scoring.

Key part:
- `evaluator.py`

### `training/`
Training layer.

Responsible for preparing train/validation windows, fitting a model, selecting thresholds, and returning category-specific model states.

Key parts:
- `training_engine.py`
- `training_pipeline.py`

### `io/`
Loading and persistence utilities.

Responsible for configs, saved models, metadata, and evaluation snapshots.

Key parts:
- `loader.py`
- `model_loader.py`
- `model_saver.py`
- `model_selector.py`
- `model_test_saver.py`

### `testing/`
Post-training inspection layer.

Responsible for evaluating saved models on candle windows and building flat prediction tables.

Key parts:
- `metrics_tester.py`
- `model_predictor.py`

### `pipelines/`
High-level orchestration flows.

Responsible for connecting training, evaluation, and selection into reusable workflows.

Key parts shown in the demo:
- `candidate_batch_pipeline.py`
- `grid_search_pipeline_async.py`

---

## Core workflow

A typical flow looks like this:

1. Load or prepare processed market data
2. Slice the dataset into train / validation / test windows
3. Build final features and targets
4. Train a PyTorch model on tabular features
5. Compute probabilities on validation data
6. Search for the best threshold for each evaluation category
7. Wrap the trained model into one or more `ModelState` objects
8. Evaluate candidate states on shared test candles
9. Select the strongest models per category

This design allows **one trained model** to produce **multiple category-specific candidate states**, each with its own threshold and selection role.

---

## Evaluation philosophy

The project does not assume that one fixed threshold such as `0.5` is always correct.

Instead, thresholds are selected separately for different categories, for example:

- MCC-oriented models
- signal-frequency-oriented models
- recall / constraint-oriented models

The evaluator supports:

- threshold sweeps
- category-specific constraints
- soft constraint fitness
- normalized cross-category scoring
- final selector scores for fair comparison

This makes the system more useful for practical signal generation than a simple "train once, predict at 0.5" setup.

---

## What already exists in the full project

The private full version goes beyond this demo.

### 1. Full model persistence and maintenance workflows
The original system includes a more complete disk-based model lifecycle:

- saving model weights and metadata
- loading saved models back into RAM
- replacing weaker saved models
- managing stored model pools by category

### 2. Convenience pipelines for working with the system
The full version includes extra operational pipelines that were not kept central in the demo, for example:

- pipelines for deleting all existing saved models
- helper flows for easier maintenance of the full model storage

### 3. Model meta-analysis pipeline
There is a dedicated pipeline for analyzing saved-model metadata.

Its purpose is to detect patterns such as:

- which feature groups appear most often in top models
- which hyperparameters appear most often in top models
- what kinds of model setups dominate the strongest candidates

This is useful for understanding what the search process is actually discovering.

### 4. Backtesting and grid backtesting
A backtest system is already built in the full project.

And not only a basic backtest — there is also a **grid backtest** workflow that was used to search for better trading logic, including parameters such as:

- how many signals are required before entry
- how long positions should be kept open
- when to close positions
- how signal-based entry and exit rules affect results

So the project is not only about training models, but also about testing how those models may be used inside trading strategies.

---

## Work in progress

### Step-by-step retraining system
A step-by-step workflow is currently under development.

The idea is to:

- retrain models at each new step
- generate predictions with freshly updated models
- reduce the risk of model staleness as market conditions evolve

This is important for time-series systems where old training windows can become outdated.

---

## Planned direction: AI agent decision layers

A major next step is the integration of this ML system into a multi-layer AI-agent architecture.

A first version of the AI-agent system was already explored **before** the ML system was built, but the decision was made to first build a stronger ML foundation and only then connect it to agents.

### Planned agent structure

The current idea is a layered decision system:

#### Layer 1 — initial AI nodes
These nodes would:
- analyze outputs from the ML system
- perform their own independent information search
- produce first-level interpretations

#### Layer 2 — aggregation / deeper reasoning
These nodes would:
- analyze the outputs of the first layer
- add more independent information gathering
- refine or challenge earlier conclusions

#### Layer 3 — final decision node
This layer would produce the final action:
- buy
- hold
- sell

In other words, the ML system is planned to become the **signal and probability layer**, while AI agents become the **multi-stage decision layer** on top of it.

---

## Why this project is interesting

This is not just a single notebook or a basic model-training script.

The project already includes many system-level ideas:

- modular feature engineering
- explicit target-building layer
- reusable runtime model states
- threshold-aware evaluation
- candidate-based model selection
- asynchronous grid search
- backtest-driven strategy research
- future integration with multi-layer AI decision agents

That makes it closer to a real research / engineering system than to a typical toy ML project.

---

## Notes

- This repository is a **demo**, not the complete private production/research codebase.
- Some storage-heavy and operational workflows were intentionally simplified.
- The main goal of the demo is to show the **architecture, ML pipeline, and design thinking** behind the system.

---

## Future demo improvements

Possible future additions to this demo:

- a cleaner interactive notebook / Colab demo
- example training and prediction walkthrough
- compact visualizations of thresholds and signals
- a lightweight showcase of backtest results

---

## Status

**Demo version completed.**

The main ML modules are already presented.
The next natural step is to add:

- a polished interactive notebook or Google Colab demo
- a lightweight visual walkthrough of the training and prediction flow

