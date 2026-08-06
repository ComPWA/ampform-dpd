from __future__ import annotations

import json
from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

from ampform_dpd.io.serialization import load_workspace

if TYPE_CHECKING:
    from pathlib import Path

    from ampform_dpd.io.serialization.format import ModelDefinition


@pytest.mark.parametrize("from_path", [False, True], ids=["mapping", "path"])
def test_load_workspace(
    model_definition: ModelDefinition, tmp_path: Path, from_path: bool
):
    source = model_definition
    if from_path:
        source = tmp_path / "model.json"
        source.write_text(json.dumps(model_definition))
    workspace = load_workspace(source)
    assert tuple(workspace.decays) == ("default_model",)
    assert tuple(workspace.distributions) == ("default_model",)
    assert set(workspace.functions) == {
        item["name"] for item in model_definition["functions"]
    }
    assert isinstance(workspace.definition, MappingProxyType)
    with pytest.raises(TypeError):
        workspace.distributions["new"] = workspace.distributions["default_model"]  # ty: ignore[invalid-assignment]


def test_load_workspace_with_latex_renderer(model_definition: ModelDefinition):
    workspace = load_workspace(model_definition, to_latex=lambda name: f"latex({name})")
    decay = workspace.decays["default_model"]
    assert decay.initial_state.latex == "latex(Lc)"
    assert all(state.latex.startswith("latex(") for state in decay.final_state.values())
    assert any(
        "latex(" in str(symbol)
        for symbol in workspace.distributions["default_model"].parameter_defaults
    )


def test_loads_multiple_distributions(model_definition: ModelDefinition):
    definition = deepcopy(model_definition)
    second = deepcopy(definition["distributions"][0])
    second["name"] = "second"
    definition["distributions"].append(second)
    workspace = load_workspace(definition)
    assert tuple(workspace.distributions) == ("default_model", "second")


@pytest.mark.parametrize("collection", ["distributions", "functions"])
def test_rejects_duplicate_names(model_definition: ModelDefinition, collection: str):
    definition = deepcopy(model_definition)
    if collection == "distributions":
        definition["distributions"].append(deepcopy(definition["distributions"][0]))
    else:
        definition["functions"].append(deepcopy(definition["functions"][0]))
    with pytest.raises(ValueError, match=rf"Duplicate {collection} names"):
        load_workspace(definition)


def test_reports_unsupported_unreferenced_function(model_definition: ModelDefinition):
    definition = deepcopy(model_definition)
    definition["functions"].append({"name": "orphan", "type": "Unknown"})
    with pytest.raises(NotImplementedError, match=r"orphan.*Unknown"):
        load_workspace(definition)
