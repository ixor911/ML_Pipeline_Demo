"""
Evaluation module.

This module contains the metric and threshold evaluation layer of the project.
Its purpose is to measure model quality, select decision thresholds, and convert
raw model probabilities into category-specific evaluation results.

Main components
---------------
evaluator.py
    Core evaluation utilities for threshold-based binary classification.

    Main responsibilities:
    - compute metrics for a fixed threshold
    - validate metrics against category-specific constraints
    - build detailed constraint reports
    - normalize optimization scores across categories
    - compute final selector scores
    - brute-force thresholds and keep the best one per category
    - evaluate models on validation and optional test data

    Main entry points:
    - evaluate_threshold(...)
    - check_metrics_constraints(...)
    - get_constraint_report(...)
    - normalize_category_score(...)
    - get_final_score(...)
    - brute_force_thresholds(...)
    - compute_thresholds(...)
    - thresholds_to_metrics(...)
    - evaluate_model(...)

Module purpose
--------------
The eval module is responsible for turning raw model probabilities into
interpretable quality signals.

Typical flow:
1. A trained model produces probabilities on validation data
2. evaluator.py searches for the best threshold per evaluation category
3. metrics are computed for those thresholds
4. category constraints are checked and converted into selector-ready scores
5. the resulting metrics are used for model comparison and selection

Design notes
------------
- Evaluation is threshold-driven, not fixed to 0.5
- Different categories can optimize different metrics
- Constraints are handled both as hard pass/fail checks and as soft fitness scores
- Final selector scores combine normalized performance with constraint fit
- This keeps evaluation flexible while still allowing fair model ranking
"""