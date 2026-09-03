import pandas as pd

from quant_research.targets.returns import make_return_targets
from quant_research.validation.purged_cv import PurgedWalkForwardSplit


def test_purging_removes_overlapping_labels(panel):
    targets = make_return_targets(panel, (5,))
    splitter = PurgedWalkForwardSplit(60, 10, 10)
    train, validation, _ = next(splitter.split(panel.index, targets.label_end_5d))
    validation_start = panel.index[validation].get_level_values("date").min()
    assert (targets.label_end_5d.iloc[train] < validation_start).all()


def test_embargo_removes_early_test_dates(panel):
    targets = make_return_targets(panel, (5,))
    base = PurgedWalkForwardSplit(60, 10, 10, embargo_days=0)
    embargo = PurgedWalkForwardSplit(60, 10, 10, embargo_days=3)
    _, _, test0 = next(base.split(panel.index, targets.label_end_5d))
    _, validation, test1 = next(embargo.split(panel.index, targets.label_end_5d))
    val_end = panel.index[validation].get_level_values("date").max()
    assert len(test1) < len(test0)
    assert (panel.index[test1].get_level_values("date") > val_end + pd.Timedelta(days=3)).all()
