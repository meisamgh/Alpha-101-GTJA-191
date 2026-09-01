"""Date-block walk-forward splits with interval purging and embargo."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    min_train_dates: int
    validation_dates: int
    test_dates: int
    step_dates: int | None = None
    embargo_days: int = 0

    def split(
        self, index: pd.MultiIndex, label_end: pd.Series
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        dates = pd.Index(index.get_level_values("date").unique()).sort_values()
        step = self.step_dates or self.test_dates
        start = self.min_train_dates
        while start + self.validation_dates + self.test_dates <= len(dates):
            validation = dates[start : start + self.validation_dates]
            test_start = start + self.validation_dates
            test = dates[test_start : test_start + self.test_dates]
            train_cutoff = validation[0]
            row_dates = index.get_level_values("date")
            train_mask = row_dates < train_cutoff
            # Purge any training label whose outcome interval reaches validation.
            train_mask &= pd.to_datetime(label_end).to_numpy() < np.datetime64(train_cutoff)
            val_mask = row_dates.isin(validation)
            test_mask = row_dates.isin(test)
            if self.embargo_days:
                embargo_end = validation[-1] + pd.Timedelta(days=self.embargo_days)
                test_mask &= row_dates > embargo_end
            yield np.flatnonzero(train_mask), np.flatnonzero(val_mask), np.flatnonzero(test_mask)
            start += step
