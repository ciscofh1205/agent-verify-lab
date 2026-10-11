"""Leakage checks. Run:  python tests/test_no_leakage.py   (prints PASS or fails with an error)"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.verify import DATA_PATH, RuntimeVerifier, temporal_split  # noqa: E402

log = pd.read_csv(DATA_PATH)
train, test = temporal_split(log)

# 1. No claim on both sides, and the split is in time order.
assert set(train.case_id).isdisjoint(test.case_id)
assert train.groupby("case_id").timestamp.min().max() <= test.groupby("case_id").timestamp.min().min()

# 2. The verifier learns nothing from the test set: scramble every test claim, re-split, refit.
#    If any test information reached fit(), the mined rules would change.
scrambled = test.copy()
scrambled["activity"] = scrambled.groupby("case_id")["activity"].transform(lambda s: s.iloc[::-1].values)
rules_clean = RuntimeVerifier().fit(train).rules_
train2, _ = temporal_split(pd.concat([train, scrambled]))
rules_scrambled = RuntimeVerifier().fit(train2).rules_
assert rules_clean == rules_scrambled, "leakage: test data changed the mined rules"

print(f"PASS: {train.case_id.nunique()} train / {test.case_id.nunique()} test claims, "
      f"{len(rules_clean)} rules mined from train only")
