from __future__ import annotations

import pytest

from ddxkb.data import Patient, load_kb, load_validation
from ddxkb.kb import KB
from helpers import TOY_CONDITIONS, TOY_EVIDENCES, real_data_present


@pytest.fixture
def toy_kb() -> KB:
    return KB.from_json(TOY_CONDITIONS, TOY_EVIDENCES)


@pytest.fixture(scope="session")
def real_kb() -> KB:
    if not real_data_present():
        pytest.skip("DDXPlus release files not present")
    return load_kb()


@pytest.fixture(scope="session")
def validation() -> list[Patient]:
    if not real_data_present():
        pytest.skip("DDXPlus release files not present")
    return load_validation()
