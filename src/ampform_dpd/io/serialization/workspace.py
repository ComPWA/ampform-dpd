from __future__ import annotations

import json
import pathlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from ampform_dpd.io.serialization.amplitude import formulate
from ampform_dpd.io.serialization.decay import to_decay
from ampform_dpd.io.serialization.dynamics import (
    PropagatorDynamicsBuilder,
    formulate_dynamics,
    formulate_form_factor,
    identity_function,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ampform_dpd import AmplitudeModel, DefinedExpression
    from ampform_dpd.decay import ThreeBodyDecay
    from ampform_dpd.io.serialization.format import ModelDefinition


@dataclass(frozen=True)
class Workspace:
    """Backend-independent formulation of a serialized amplitude model."""

    definition: Mapping[str, Any]
    decays: Mapping[str, ThreeBodyDecay]
    distributions: Mapping[str, AmplitudeModel]
    functions: Mapping[str, DefinedExpression]
    kinematics: Mapping[str, Any]
    reference_points: tuple[Mapping[str, Any], ...]
    checksums: tuple[Mapping[str, Any], ...]


def load_workspace(
    source: str | pathlib.Path | Mapping[str, Any],
    *,
    builders: Mapping[str, PropagatorDynamicsBuilder] | None = None,
    to_latex: Callable[[str], str] | None = None,
) -> Workspace:
    """Load and formulate every distribution in a serialized model."""
    definition = _load_definition(source)
    _raise_on_duplicate_names(definition)
    render_name = to_latex if to_latex is not None else identity_function
    selected_definitions = {
        distribution["name"]: _select_distribution(definition, distribution)
        for distribution in definition["distributions"]
    }
    decays = {
        name: to_decay(selected, to_latex=render_name)
        for name, selected in selected_definitions.items()
    }
    distributions = {
        name: formulate(
            selected,
            additional_builders=dict(builders) if builders is not None else None,
            to_latex=render_name,
        )
        for name, selected in selected_definitions.items()
    }
    functions = _formulate_functions(definition, builders, render_name)
    kinematics = {
        distribution["name"]: distribution["decay_description"]["kinematics"]
        for distribution in definition["distributions"]
    }
    checksums = definition.get("misc", {}).get("amplitude_model_checksums", [])
    return Workspace(
        definition=_freeze(definition),
        decays=MappingProxyType(decays),
        distributions=MappingProxyType(distributions),
        functions=MappingProxyType(functions),
        kinematics=_freeze(kinematics),
        reference_points=tuple(
            _freeze(point) for point in definition.get("parameter_points", [])
        ),
        checksums=tuple(_freeze(checksum) for checksum in checksums),
    )


def _load_definition(
    source: str | pathlib.Path | Mapping[str, Any],
) -> ModelDefinition:
    if isinstance(source, Mapping):
        return dict(source)  # ty: ignore[invalid-return-type]
    with pathlib.Path(source).open() as stream:
        return json.load(stream)


def _raise_on_duplicate_names(definition: ModelDefinition) -> None:
    for collection_name in ("distributions", "functions"):
        names = [item["name"] for item in definition[collection_name]]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            msg = f"Duplicate {collection_name} names: {', '.join(duplicates)}"
            raise ValueError(msg)


def _select_distribution(
    definition: ModelDefinition, distribution: Mapping[str, Any]
) -> ModelDefinition:
    selected = dict(definition)
    selected["distributions"] = [dict(distribution)]
    return selected  # ty: ignore[invalid-return-type]


def _formulate_functions(
    definition: ModelDefinition,
    builders: Mapping[str, PropagatorDynamicsBuilder] | None,
    to_latex: Callable[[str], str],
) -> dict[str, DefinedExpression]:
    formulated = {}
    for function in definition["functions"]:
        name = function["name"]
        formulated[name] = _formulate_function(name, definition, builders, to_latex)
    return formulated


def _formulate_function(
    name: str,
    definition: ModelDefinition,
    builders: Mapping[str, PropagatorDynamicsBuilder] | None,
    to_latex: Callable[[str], str],
) -> DefinedExpression:
    for distribution in definition["distributions"]:
        model = _select_distribution(definition, distribution)
        for chain in distribution["decay_description"]["chains"]:
            for propagator in chain["propagators"]:
                if propagator.get("parametrization") == name:
                    single_propagator_chain = dict(chain)
                    single_propagator_chain["propagators"] = [propagator]
                    return formulate_dynamics(
                        cast("Any", single_propagator_chain),
                        model,
                        to_latex=to_latex,
                        additional_definitions=(
                            dict(builders) if builders is not None else None
                        ),
                    )
            for vertex in chain["vertices"]:
                if vertex.get("formfactor") == name:
                    return formulate_form_factor(vertex, model)
    function_type = next(
        function["type"]
        for function in definition["functions"]
        if function["name"] == name
    )
    msg = (
        f"Cannot formulate function {name!r} of type {function_type!r}: "
        "it has no propagator or form-factor context"
    )
    raise NotImplementedError(msg)


def _freeze(value: Any, /) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value
