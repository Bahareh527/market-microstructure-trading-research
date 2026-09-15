from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_microstructure.pipeline import run_experiment


result = run_experiment(ROOT / "results", ROOT / "data")
print(json.dumps(result, indent=2))
