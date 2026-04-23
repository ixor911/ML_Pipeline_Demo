# core/pipelines/candidate_batch_pipeline.py

from typing import Dict, Any, List
import pandas as pd

from core.training.training_pipeline import training_pipeline_list
from core.io.model_selector import eval_model
from core.models import ModelState


def candidate_batch_pipeline(
    *,
    training_config: Dict[str, Any],
    test_candles: pd.DataFrame,
    models_ram: Dict[str, Dict[str, ModelState]],
    limit: int,
) -> Dict[str, Dict[str, ModelState]]:
    """
    Runs one full train-config pass and evaluates all produced candidates.

    Flow:
    1. train one base model from the provided training config
    2. convert it into a batch of candidate ModelState objects
       (one per threshold / category)
    3. evaluate each candidate on the same test candle set
    4. pass each candidate through the selector

    The selector decides whether a candidate should:
    - be added directly
    - replace a weaker saved model
    - be rejected
    """
    # Unpack the main config sections used by the training pipeline.
    datapath = training_config.get("datapath")
    model = training_config.get("model")
    slicing = training_config.get("slicing")
    builder = training_config.get("builder")

    # Train one model and expand it into multiple candidate states.
    candidate_states: List[ModelState] = training_pipeline_list(
        datapath=datapath,
        model_config=model,
        slicing_config=slicing,
        builder_config=builder
    )

    # Evaluate every candidate on the shared test candles,
    # then let the selector decide whether to keep it.
    for state in candidate_states:
        state.evaluate_metrics(candles=test_candles)

        models_ram = eval_model(
            candidate=state,
            models_ram=models_ram,
            limit=limit,
        )

    return models_ram