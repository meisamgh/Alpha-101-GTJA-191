from quant_research.features.regime import hard_regime_gate
from quant_research.features.technical import compute_features


def test_regime_features_are_trailing_only(panel):
    cutoff = panel.index.get_level_values("date").unique()[100]
    base = hard_regime_gate(compute_features(panel)).loc[:cutoff]
    changed = panel.copy()
    changed.loc[changed.index.get_level_values("date") > cutoff, "close"] *= 3
    candidate = hard_regime_gate(compute_features(changed)).loc[:cutoff]
    assert base.equals(candidate)
