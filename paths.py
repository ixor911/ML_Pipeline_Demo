from pathlib import Path


ROOT = Path(__file__).resolve().parent


CORE_DIR = ROOT / "core"


DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


SAVES_DIR = ROOT / "saves"

MODEL_SAVES_DIR = SAVES_DIR / "models"
BEST_MODELS_PRECISION_DIR = MODEL_SAVES_DIR / "precision"
BEST_MODELS_MCC_DIR       = MODEL_SAVES_DIR / "mcc"
BEST_MODELS_TPR90_DIR       = MODEL_SAVES_DIR / "tpr90"
ARCHIVED_MODELS_DIR       = MODEL_SAVES_DIR / "archive"  # если будешь старые складывать

SCALERS_DIR = SAVES_DIR / "scalers"


LOGS_DIR = ROOT / "logs"
EXPERIMENT_LOGS_DIR = LOGS_DIR / "experiments"  # текстовые summary по grid / full_run
BACKTEST_LOGS_DIR   = LOGS_DIR / "backtests"    # текстовые отчёты по стратегиям


CONFIGS_DIR = ROOT / "configs"

