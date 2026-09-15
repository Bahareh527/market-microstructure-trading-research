from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_microstructure.synthetic import SyntheticLOBConfig, generate_synthetic_lob


output = ROOT / "data" / "sample_lob.csv"
generate_synthetic_lob(SyntheticLOBConfig(rows=1_000, seed=42)).to_csv(output, index=False)
print(f"Wrote {output}")
