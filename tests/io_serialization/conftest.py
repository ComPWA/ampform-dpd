from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ampform_dpd.io.serialization.format import ModelDefinition


@pytest.fixture(scope="session")
def model_definition() -> ModelDefinition:
    this_directory = Path(__file__).parent
    doc_directory = this_directory.parent.parent / "docs"
    with open(doc_directory / "Lc2ppiK.json") as stream:
        return json.load(stream)
