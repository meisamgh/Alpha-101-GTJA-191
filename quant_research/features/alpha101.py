"""Validated representative Alpha-101 factors in normalized panel form.

The legacy implementation is deliberately not imported: it mutates inputs, uses deprecated APIs,
and contains incomplete formula translations. Add factors here only after formula-level tests.
"""
import numpy as np
import pandas as pd


def compute_alpha_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_index()
    g = p.groupby(level="symbol", group_keys=False)
    out = pd.DataFrame(index=p.index)
    out["alpha_012"] = np.sign(g.volume.diff()) * -g.close.diff()
    out["alpha_101"] = (p.close - p.open) / (p.high - p.low + 0.001)
    return out
