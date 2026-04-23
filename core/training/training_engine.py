# core/training/training_engine.py

from __future__ import annotations

from typing import Dict

from core.models import ModelState
from core.models.TorchModel import TorchModel


# =========================================================
# State builders
# =========================================================

def create_state(
    *,
    model: TorchModel,
    meta: dict,
    category: str,
    thr: float,
) -> ModelState:
    """
    Creates a single ModelState for one evaluation category.

    A separate state is created per category because the same trained model
    can be reused with different thresholds and selection buckets.
    """
    state = ModelState(
        model_id=None,
        category=category,
        model=model,
        meta={**meta, "category": category},
        thr=thr,
        eval={},
    )

    # Ensure structural signature is available immediately.
    state.ensure_signature()
    return state


def create_states(
    *,
    model: TorchModel,
    meta: dict,
    thrs: Dict[str, float],
) -> Dict[str, ModelState]:
    """
    Creates one ModelState per threshold category.

    Example:
        thrs = {
            "mcc": 0.42,
            "tpr90": 0.37,
            ...
        }

    Returns:
        Dict[str, ModelState]:
            Mapping category -> ModelState
    """
    states: dict = {}

    for category, thr in thrs.items():
        if thr is None:
            continue

        state = create_state(
            model=model,
            meta=meta,
            category=category,
            thr=thr,
        )
        states[category] = state

    return states


