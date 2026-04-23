# core/testing/metrics_tester.py

from typing import Dict, Any
import pandas as pd

from core.io.model_loader import load_all_models
from core.io import loader

from core.models import ModelState
from core.target import slicing


def evaluate_models_metrics(
    *,
    models_ram: Dict[str, Dict[str, ModelState]],
    candles: pd.DataFrame,
) -> Dict[str, Dict[str, ModelState]]:
    """
    Evaluates all loaded models on the same candle window.

    The computed metrics are written back into each ModelState.
    """
    for category, models in models_ram.items():
        for model_id, state in models.items():
            state.evaluate_metrics(candles=candles)

    return models_ram


def test_saved_models_metrics_pipeline(
    *,
    df: pd.DataFrame,
    slicing_config: Dict[str, Any],
    show: bool = True,
) -> Dict[str, Dict[str, ModelState]]:
    """
    Full testing pipeline for saved models.

    Flow:
    1. Slice the input dataframe into windows
    2. Use the first window as the evaluation candle set
    3. Load all saved models from disk
    4. Evaluate each model on the same candle window
    5. Optionally print the best model per category
    """
    windows = slicing.slice_data(df=df, slicing_config=slicing_config)

    # By convention, the first sliced window is used for evaluation.
    candles = windows[0]

    # Load all persisted models grouped by category.
    models_ram: Dict[str, Dict[str, ModelState]] = load_all_models()

    # Recompute metrics for every model on the selected candle window.
    models_ram = evaluate_models_metrics(
        models_ram=models_ram,
        candles=candles,
    )

    if show:
        _show_best_models(models_ram)

    return models_ram


# ============================================================
# Console output
# ============================================================

def _show_best_models(
    models_ram: Dict[str, Dict[str, ModelState]]
) -> None:
    """
    Prints the best model per category based on state.score.
    """
    print("\n=== BEST MODELS PER CATEGORY ===")

    for category, models in models_ram.items():
        best_state = None
        best_score = None

        # Find the strongest model inside the current category.
        for state in models.values():
            score = getattr(state, "score", None)
            if score is None:
                continue

            if best_score is None or score > best_score:
                best_score = score
                best_state = state

        if best_state is None:
            print(f"[{category}] no valid models")
            continue

        metrics = best_state.eval.get("metrics", {})
        print(
            f"[{category}] "
            f"id={best_state.model_id} "
            f"score={best_score:.4f} "
            f"signals={metrics.get('signals')} "
            f"thr={best_state.thr}"
        )


# ============================================================
# Main smoke test
# ============================================================

if __name__ == "__main__":
    datapath = "ETHUSDT_6_1h.csv"

    # Load processed feature dataset.
    data = loader.load_processed_data(datapath)

    # Evaluate all saved models on the most recent 5000 candles.
    slicing_config = {
        "type": "candles",
        "candles": [5000],
        "from_tail": True,
        "end_date": None,
        "time_col": "open_time_eth",
    }

    test_saved_models_metrics_pipeline(
        df=data,
        slicing_config=slicing_config,
        show=True,
    )