import pytest
from theseus.common import load_config, project_root


@pytest.fixture
def cfg():
    return load_config(project_root() / "configs/h100.yaml")


@pytest.fixture
def scenario():
    return {
        "id": "fixture",
        "dataset": "synthetic_test",
        "description": "Test product",
        "price_scale": 1.0,
        "volume_scale": 100.0,
        "normalized_costs": [1.0, 1.0],
    }
