import json
from pathlib import Path

import pytest

DATASETS_DIR = Path(__file__).parent / "golden_datasets"


@pytest.fixture
def docs_golden():
    return json.loads((DATASETS_DIR / "docs_team.json").read_text())


@pytest.fixture
def research_golden():
    return json.loads((DATASETS_DIR / "research_team.json").read_text())


@pytest.fixture
def technical_golden():
    return json.loads((DATASETS_DIR / "technical_team.json").read_text())


@pytest.fixture
def knowledge_golden():
    return json.loads((DATASETS_DIR / "knowledge_team.json").read_text())
