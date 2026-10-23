"""Reuse the preserved log-XGBoost helpers without mutating their globals."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'experiments/series_weight_v1'
for name in ('baseline', 'baseline_outputs', 'xgboost_core', 'xgboost_evaluation'):
    if name in sys.modules and Path(sys.modules[name].__file__).resolve() != SOURCE / (name + '.py'):
        raise RuntimeError('Use the documented standalone command in a fresh Python process')
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('global_preserved_runner', SOURCE / 'run_experiment.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
import xgboost_core as core
from baseline import SERIES_KEY
from xgboost_evaluation import evaluate_forecast
