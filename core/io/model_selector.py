# core/io/model_selector.py

"""
model_selector.py

Logic for maintaining top models per category in memory
and replacing persisted models when stronger candidates appear.
"""

from typing import Dict

from core.io import model_saver
from core.eval import evaluator
from core.models import ModelState


def find_duplicate_by_signature(
    candidate: ModelState,
    category_models: dict[str, ModelState],
) -> ModelState | None:
    """Returns an existing model with the same signature, if any."""
    cand_sig = candidate.signature

    for state in category_models.values():
        if state.signature == cand_sig:
            return state

    return None


def is_duplicate(candidate: ModelState, category_models: dict[str, ModelState]) -> bool:
    """Checks whether a candidate already exists in the category by signature."""
    return find_duplicate_by_signature(candidate, category_models) is not None


def get_state_metrics(state: ModelState) -> dict:
    """Returns cached metrics for a state."""
    return state.eval.get("metrics", {}) if state.eval else {}


def get_state_selector_info(state: ModelState) -> dict:
    """
    Computes selector metrics from test metrics and caches them in state.eval["selector"].
    """
    if state.eval is None:
        state.eval = {}

    cached = state.eval.get("selector")
    if cached is not None:
        return cached

    metrics = get_state_metrics(state)
    info = evaluator.get_final_score(metrics, state.category)
    state.eval["selector"] = info
    return info


def get_state_final_score(state: ModelState) -> float:
    """Returns the final selector score for a model state."""
    return float(get_state_selector_info(state).get("final_score", 0.0))


def find_worst_category_model(
    category_models: dict[str, ModelState],
) -> ModelState:
    """Returns the weakest model in a category by final_score."""
    worst_state = None
    worst_score = float("inf")

    for state in category_models.values():
        final_score = get_state_final_score(state)

        if final_score < worst_score:
            worst_score = final_score
            worst_state = state

    return worst_state


def compare_model(
    candidate: ModelState,
    old_state: ModelState | None,
) -> bool:
    """
    Returns True if the candidate is better than the existing model.

    Comparison is based on selector final_score:
        final_score = normalized_score * constraint_fit
    """
    cand_info = get_state_selector_info(candidate)
    cand_final = cand_info["final_score"]

    if old_state is None:
        return True

    old_info = get_state_selector_info(old_state)
    old_final = old_info["final_score"]

    return cand_final > old_final


def replace_model(
    *,
    old_state: ModelState,
    new_state: ModelState,
    category_models: Dict[str, ModelState],
) -> Dict[str, ModelState]:
    """Replaces one saved model in a category with a new candidate."""
    old_id = old_state.model_id

    if old_id is None or old_id not in category_models:
        raise ValueError("replace_model: old_state not found in category_models")

    saved_state = model_saver.replace_model(old_model_id=old_id, new_state=new_state)

    category_models.pop(old_id)
    category_models[saved_state.model_id] = saved_state

    return category_models


def eval_model_category(
    *,
    candidate: ModelState,
    category_models: dict[str, ModelState],
    limit: int,
) -> dict[str, ModelState]:
    """
    Evaluates a candidate inside one category.

    Rules:
    - skip duplicates by signature
    - save directly if there is free space
    - otherwise compare against the weakest saved model
    """
    if candidate.category is None:
        return category_models

    # Skip exact structural duplicates.
    if is_duplicate(candidate, category_models):
        return category_models

    # Save directly if the category is not full yet.
    if len(category_models) < limit:
        saved_state = model_saver.create_model(candidate)
        category_models[saved_state.model_id] = saved_state
        return category_models

    # Otherwise compare with the weakest existing model.
    worst_state = find_worst_category_model(category_models)

    if compare_model(candidate, worst_state):
        return replace_model(
            old_state=worst_state,
            new_state=candidate,
            category_models=category_models,
        )

    return category_models


def eval_model(
    *,
    candidate: ModelState,
    models_ram: dict[str, dict[str, ModelState]],
    limit: int,
) -> dict[str, dict[str, ModelState]]:
    """
    Evaluates a candidate model inside the in-memory model store.

    The candidate is routed to its category and processed there.
    """
    category = candidate.category

    if category not in models_ram:
        models_ram[category] = {}

    models_ram[category] = eval_model_category(
        candidate=candidate,
        category_models=models_ram[category],
        limit=limit,
    )

    return models_ram