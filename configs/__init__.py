import json
import paths


with open(paths.CONFIGS_DIR / "evaluator_metrics.json") as file:
    EVALUATOR_CONFIG = json.load(file)

