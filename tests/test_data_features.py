import pandas as pd

from quant_research.data.validation import validate_panel
from quant_research.features.alpha101 import compute_alpha_features
from quant_research.features.gtja191 import compute_gtja_features
from quant_research.features.technical import compute_features


def test_synthetic_panel_valid(panel):
    assert validate_panel(panel).valid


def test_features_do_not_change_when_future_is_perturbed(panel):
    cutoff = panel.index.get_level_values("date").unique()[90]
    baseline = compute_features(panel).loc[:cutoff]
    changed = panel.copy()
    future = changed.index.get_level_values("date") > cutoff
    changed.loc[future, ["open", "high", "low", "close", "adjusted_close", "volume"]] *= 10
    pd.testing.assert_frame_equal(baseline, compute_features(changed).loc[:cutoff])


def test_representative_alpha_formulas(panel):
    a = compute_alpha_features(panel)
    row = panel.index[50]
    symbol = row[1]
    asset = panel.xs(symbol, level="symbol")
    expected = (asset.close.diff().loc[row[0]] * -1) * asset.volume.diff().loc[row[0]].__class__(
        1 if asset.volume.diff().loc[row[0]] > 0 else -1
    )
    assert a.loc[row, "alpha_012"] == expected
    assert compute_gtja_features(panel).index.equals(panel.index)
