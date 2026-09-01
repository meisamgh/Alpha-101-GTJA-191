import pytest

from quant_research.experiments.synthetic import make_synthetic_panel


@pytest.fixture(scope="session")
def panel():
    return make_synthetic_panel(days=140, assets=20)
