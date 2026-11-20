"""PW4 A1: fetch the original UCI Concrete Strength dataset once.

Run from PW4/: python fetch_concrete.py
Afterwards sessionA.py reads the local CSV and needs no internet.
"""
from pathlib import Path

from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

concrete = fetch_ucirepo(id=165)
print("=== UCI METADATA ===")
print(concrete.metadata)
print("\n=== ORIGINAL VARIABLES (CHECK THE COLUMN ORDER) ===")
print(concrete.variables.to_string(index=False))

X = concrete.data.features.copy()
y = concrete.data.targets.iloc[:, 0].rename("strength")
expected_columns = [
    "cement", "slag", "ash", "water", "superplastic",
    "coarse_agg", "fine_agg", "age",
]
if X.shape[1] != len(expected_columns) or len(y) != len(X):
    raise ValueError("Unexpected UCI dataset shape; verify variables before renaming")
X.columns = expected_columns

df = X.assign(strength=y)
path = DATA / "concrete.csv"
df.to_csv(path, index=False)
print(f"\nSaved {len(df)} rows and {len(df.columns)} columns to {path}")
print("New columns:", list(df.columns))
