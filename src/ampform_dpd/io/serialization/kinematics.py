from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import sympy as sp
from ampform.kinematics.phasespace import Kallen

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ampform_dpd.io.serialization.workspace import Workspace


def formulate_kinematic_map(  # ruff: ignore[too-many-locals]
    workspace: Workspace, distribution: str | None = None
) -> Mapping[sp.Symbol, sp.Expr]:
    r"""Map a serialized isobar mass and helicity angle to all invariants.

    The serialized angle :math:`\theta_{ij}` is measured between particle :math:`i` and
    spectator :math:`k` in the rest frame of the :math:`(ij)` isobar. Its ordered pair
    must follow the cyclic convention ``(1, 2)``, ``(2, 3)``, or ``(3, 1)``.
    """
    distribution_name = _select_distribution(workspace, distribution)
    definition = _find_distribution(workspace, distribution_name)
    variables = definition.get("variables", [])
    coordinate = _select_isobar_coordinate(variables)
    i, j = coordinate["node"]
    k, *_ = {1, 2, 3} - {i, j}
    mass_name, _, cosine_name = coordinate["mass_phi_costheta"]
    mass = sp.Symbol(mass_name, nonnegative=True)
    cosine = sp.Symbol(cosine_name, real=True)
    m0, m1, m2, m3 = sp.symbols("m:4", nonnegative=True)
    masses = {0: m0, 1: m1, 2: m2, 3: m3}
    mi, mj, mk = masses[i], masses[j], masses[k]
    invariants = {
        index: sp.Symbol(f"sigma{index}", nonnegative=True) for index in (1, 2, 3)
    }
    sigma_k = mass**2
    denominator = sp.sqrt(Kallen(m0**2, mk**2, sigma_k)) * sp.sqrt(
        Kallen(sigma_k, mi**2, mj**2)
    )
    sigma_j = (
        mk**2
        + mi**2
        + (cosine * denominator + (sigma_k + mi**2 - mj**2) * (m0**2 - sigma_k - mk**2))
        / (2 * sigma_k)
    )
    sigma_i = m0**2 + m1**2 + m2**2 + m3**2 - sigma_j - sigma_k
    return MappingProxyType({
        invariants[i]: sigma_i,
        invariants[j]: sigma_j,
        invariants[k]: sigma_k,
    })


def _select_distribution(workspace: Workspace, distribution: str | None) -> str:
    if distribution is not None:
        if distribution not in workspace.distributions:
            msg = f"Unknown distribution {distribution!r}"
            raise KeyError(msg)
        return distribution
    names = tuple(workspace.distributions)
    if len(names) != 1:
        msg = "Select a distribution when the workspace contains multiple distributions"
        raise ValueError(msg)
    return names[0]


def _find_distribution(workspace: Workspace, name: str) -> Mapping[str, Any]:
    for definition in workspace.definition["distributions"]:
        if definition["name"] == name:
            return definition
    msg = f"Missing definition for distribution {name!r}"
    raise KeyError(msg)


def _select_isobar_coordinate(variables: Any) -> Mapping[str, Any]:
    node_size = 2
    coordinate_size = 3
    candidates = []
    for variable in variables:
        node = variable.get("node")
        names = variable.get("mass_phi_costheta")
        if (
            isinstance(node, (tuple, list))
            and len(node) == node_size
            and all(isinstance(item, int) for item in node)
            and isinstance(names, (tuple, list))
            and len(names) == coordinate_size
        ):
            candidates.append(variable)
    if len(candidates) != 1:
        msg = f"Expected one invariant-mass/helicity-angle coordinate, found {len(candidates)}"
        raise ValueError(msg)
    i, j = candidates[0]["node"]
    if (i, j) not in {(1, 2), (2, 3), (3, 1)}:
        msg = f"Unsupported helicity-angle orientation ({i}, {j})"
        raise ValueError(msg)
    return candidates[0]
