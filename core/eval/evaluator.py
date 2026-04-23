# core/eval/evaluator.py

import numpy as np
from configs import EVALUATOR_CONFIG
from copy import deepcopy
from typing import Dict, Any, Optional, Tuple


def evaluate_threshold(probs: np.ndarray, y: np.ndarray, thr: float, abs_candles: int = None):
    """
    Computes binary classification metrics for a fixed threshold.

    In addition to standard metrics such as precision, recall, F1, and MCC,
    this function also reports signal coverage statistics. If abs_candles is
    provided, the metrics are additionally normalized relative to the full
    candle window, not only to the filtered sample size.
    """
    preds = (probs >= thr).astype(np.uint8)

    # Encode prediction/target outcomes into 4 buckets:
    # 0 = TN, 1 = FN, 2 = FP, 3 = TP
    codes = preds * 2 + y.astype(np.uint8)
    counts = np.bincount(codes, minlength=4)

    tn, fn, fp, tp = map(int, [counts[0], counts[1], counts[2], counts[3]])
    total = tp + fp + tn + fn

    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = (2 * tp) / max(1, 2 * tp + fp + fn)

    # Matthews Correlation Coefficient is often more informative
    # than accuracy for imbalanced signal datasets.
    mcc_den = np.sqrt(
        (tp + fp) *
        (tp + fn) *
        (tn + fp) *
        (tn + fn)
    )
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else 0.0

    signals = tp + fp
    signals_percent = preds.mean()

    abs_percent = total / abs_candles if abs_candles is not None else None
    abs_signals = signals / abs_candles if abs_candles is not None else None

    return {
        "thr": thr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "total": total,
        "signals": signals,
        "signals_percent": signals_percent,
        "abs_candles": abs_candles,
        "abs_percent": abs_percent,
        "abs_signals": abs_signals
    }


def check_metrics_constraints(metrics: Dict[str, Any], category: str) -> bool:
    """
    Checks whether a metric dictionary satisfies all constraints
    defined for the given evaluation category.

    If a category has no constraints, it is treated as always valid.
    """
    cfg = EVALUATOR_CONFIG.get(category, {})
    constraints = cfg.get("constraints")

    if not constraints:
        return True

    for metric_name, rules in constraints.items():
        value = metrics.get(metric_name)

        # Missing metric automatically fails the constraint.
        if value is None:
            return False

        if "equals" in rules and value != rules["equals"]:
            return False

        if "min" in rules and value < rules["min"]:
            return False

        if "max" in rules and value > rules["max"]:
            return False

    return True


def _clamp01(x: float) -> float:
    """Clamps a float value into the [0, 1] range."""
    return max(0.0, min(1.0, float(x)))


def _constraint_rule_fitness(value: float | int | None, rules: Dict[str, Any]) -> float:
    """
    Converts a single constraint result into a soft fitness score in [0, 1].

    This is used when constraints are not only treated as pass/fail, but also
    as a softer measure of how close a metric is to satisfying the requirement.
    """
    if value is None:
        return 0.0

    if "equals" in rules:
        return 1.0 if value == rules["equals"] else 0.0

    if "min" in rules:
        target = float(rules["min"])
        if target <= 0:
            return 1.0
        return _clamp01(float(value) / target)

    if "max" in rules:
        target = float(rules["max"])
        value = float(value)

        if value <= target:
            return 1.0

        if value <= 0:
            return 1.0

        return _clamp01(target / value)

    return 1.0


def get_constraint_report(metrics: Dict[str, Any], category: str) -> Dict[str, Any]:
    """
    Builds a detailed constraint report for a category.

    The report includes:
    - whether all constraints passed
    - a soft aggregate constraint_fit score
    - number of violations
    - per-metric details

    Multiple constraints are aggregated with a geometric mean, which penalizes
    weak dimensions more than a simple arithmetic average would.
    """
    cfg = EVALUATOR_CONFIG.get(category, {})
    constraints = cfg.get("constraints") or {}

    if not constraints:
        return {
            "passed": True,
            "constraint_fit": 1.0,
            "violations_count": 0,
            "details": {},
        }

    details: Dict[str, Any] = {}
    fitness_values = []
    violations_count = 0

    for metric_name, rules in constraints.items():
        value = metrics.get(metric_name)
        fitness = _constraint_rule_fitness(value, rules)

        passed = True
        if value is None:
            passed = False
        elif "equals" in rules and value != rules["equals"]:
            passed = False
        elif "min" in rules and value < rules["min"]:
            passed = False
        elif "max" in rules and value > rules["max"]:
            passed = False

        if not passed:
            violations_count += 1

        details[metric_name] = {
            "value": value,
            "rules": rules,
            "passed": passed,
            "fitness": fitness,
        }

        # Small floor is used to keep geometric mean stable.
        fitness_values.append(max(fitness, 1e-12))

    if fitness_values:
        fit = float(np.prod(fitness_values) ** (1.0 / len(fitness_values)))
    else:
        fit = 1.0

    return {
        "passed": violations_count == 0,
        "constraint_fit": _clamp01(fit),
        "violations_count": violations_count,
        "details": details,
    }


def normalize_category_score(score: float | None, category: str) -> float:
    """
    Normalizes the category score into the [0, 1] range.

    This is used to make different optimization metrics more comparable
    before combining them with constraint_fit.
    """
    if score is None:
        return 0.0

    cfg = EVALUATOR_CONFIG.get(category, {})
    optimize_field = cfg.get("optimize")

    s = float(score)

    # MCC naturally lives in [-1, 1], so it needs explicit remapping.
    if optimize_field == "mcc":
        return _clamp01((s + 1.0) / 2.0)

    return _clamp01(s)


def get_final_score(metrics: Dict[str, Any], category: str) -> Dict[str, Any]:
    """
    Computes the final selector score for a category.

    Current logic:
        final_score = score_norm * constraint_fit

    This allows a model with strong raw metrics but weak constraint fit
    to be ranked lower than a more balanced candidate.
    """
    cfg = EVALUATOR_CONFIG.get(category, {})
    optimize_field = cfg.get("optimize")

    score_raw = metrics.get(optimize_field)
    score_norm = normalize_category_score(score_raw, category)

    constraint_report = get_constraint_report(metrics, category)
    constraint_fit = constraint_report["constraint_fit"]

    final_score = score_norm * constraint_fit

    return {
        "score_raw": score_raw,
        "score_norm": score_norm,
        "constraint_fit": constraint_fit,
        "final_score": final_score,
        "passed": constraint_report["passed"],
        "violations_count": constraint_report["violations_count"],
        "constraint_report": constraint_report,
    }


def update_best_metrics(
    metrics: Dict[str, Any],
    thresholds: Dict[str, Dict[str, float]],
    thr: float,
) -> Dict[str, Dict[str, float]]:
    """
    Updates the best threshold per category if the current threshold
    produces a better valid score.

    Only thresholds that satisfy all category constraints are eligible.
    """
    updated = deepcopy(thresholds)

    for category, cfg in EVALUATOR_CONFIG.items():
        optimize_field = cfg["optimize"]

        if not check_metrics_constraints(metrics, category):
            continue

        score = metrics.get(optimize_field)
        if score is None:
            continue

        best = updated.get(category)

        if best is None or score > best.get("score"):
            updated[category] = {
                "thr": thr,
                "score": score,
            }

    return updated


def brute_force_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    thr_min: float = 0.0,
    thr_max: float = 1.0,
    thr_step: float = 0.01,
    abs_candles: int = None
) -> Dict[str, Dict[str, float]]:
    """
    Brute-forces thresholds over a fixed range and keeps the best one
    for each evaluation category.

    The scan runs from high to low threshold, which can be helpful for
    categories sensitive to signal frequency.
    """
    thresholds: Dict[str, Dict[str, float]] = {}

    for thr in np.arange(thr_max, thr_min - 1e-9, -thr_step):
        thr = round(float(thr), 5)

        metrics = evaluate_threshold(probs, y, thr, abs_candles=abs_candles)

        thresholds = update_best_metrics(
            metrics=metrics,
            thresholds=thresholds,
            thr=thr
        )

    return thresholds


def compute_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    return_metrics: bool = False,
    abs_candles: int = None
) -> dict | Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    Computes the best threshold per category.

    Optionally also returns the full metrics dictionary evaluated at each
    selected threshold.
    """
    best = brute_force_thresholds(probs, y, abs_candles=abs_candles)

    thrs: Dict[str, float] = {}
    metrics_by_category: Dict[str, Dict[str, Any]] = {}

    for category, info in best.items():
        thr = info["thr"]
        thrs[category] = thr
        metrics_by_category[category] = evaluate_threshold(
            probs, y, thr, abs_candles=abs_candles
        )

    if return_metrics:
        return thrs, metrics_by_category

    return thrs


def thresholds_to_metrics(
    probs: np.ndarray,
    y: np.ndarray,
    thrs: Dict[str, float],
    abs_candles: int = None
) -> Dict[str, Dict[str, Any]]:
    """
    Converts a threshold dictionary into a metrics dictionary by evaluating
    each category at its corresponding threshold.
    """
    out: Dict[str, Dict[str, Any]] = {}

    for category, thr in thrs.items():
        out[category] = evaluate_threshold(probs, y, thr, abs_candles)

    return out


def evaluate_model(
    model,
    x_val,
    y_val,
    x_test: Optional[Any] = None,
    y_test: Optional[Any] = None,
    abs_candles: int = None,
    *args, **kwargs
):
    """
    Full model evaluation workflow.

    Steps:
    1. Compute probabilities on validation data
    2. Select best thresholds per category on validation
    3. Evaluate those thresholds:
       - on test data if provided
       - otherwise on validation data
    """
    probs_val = model.predict_proba(x_val)

    thrs, val_metrics = compute_thresholds(probs_val, y_val, return_metrics=True)

    if x_test is not None and y_test is not None:
        probs_test = model.predict_proba(x_test)

        test_metrics = thresholds_to_metrics(
            probs=probs_test,
            y=y_test,
            thrs=thrs,
            abs_candles=abs_candles
        )

        return {**test_metrics}

    return {**val_metrics}