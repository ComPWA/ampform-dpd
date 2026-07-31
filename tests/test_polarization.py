from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import sympy as sp

from ampform_dpd.polarization import formulate_final_state_decay_angles

if TYPE_CHECKING:
    from ampform_dpd.decay import FinalStateID


@pytest.mark.parametrize(
    ("state_id", "expected_momenta"),
    [
        (2, {"p2", "q2"}),
        (1, {"p1", "p3", "q1"}),
        (3, {"p1", "p3", "q3"}),
    ],
    ids=["spectator", "isobar-product-1", "isobar-product-3"],
)
def test_formulate_final_state_decay_angles(
    state_id: FinalStateID, expected_momenta: set[str]
):
    """The expressions must reproduce angles used to construct the daughter momentum."""
    angles = formulate_final_state_decay_angles(state_id, reference_subsystem=2)
    assert [str(s) for s in angles] == [f"phi_{state_id}", f"theta_{state_id}"]

    momenta = _generate_three_body_event()
    expected_φ, expected_θ = 1.234, 0.789
    chain = _compute_boost_chain(momenta, state_id)
    daughter_direction = np.array([
        0.5,
        0.3 * np.sin(expected_θ) * np.cos(expected_φ),
        0.3 * np.sin(expected_θ) * np.sin(expected_φ),
        0.3 * np.cos(expected_θ),
    ])
    momenta[f"q{state_id}"] = np.linalg.inv(chain) @ daughter_direction

    symbols = sorted(
        {s for expr in angles.values() for s in expr.free_symbols}, key=str
    )
    assert {str(s) for s in symbols} >= expected_momenta
    arrays = [momenta[str(s)][None, :] for s in symbols]
    φ_expr, θ_expr = angles.values()
    computed_φ = sp.lambdify(symbols, φ_expr.doit(), "numpy")(*arrays)
    computed_θ = sp.lambdify(symbols, θ_expr.doit(), "numpy")(*arrays)
    assert computed_φ[0] == pytest.approx(expected_φ)
    assert computed_θ[0] == pytest.approx(expected_θ)


def _generate_three_body_event() -> dict[str, np.ndarray]:
    """Generate one J/ψ → K⁰ Σ⁺ p̄ event via the subsystem-2 chain (isobar = p̄K⁰)."""
    rng = np.random.default_rng(seed=42)
    m0, m1, m2, m3 = 3.0969, 0.4976, 1.1894, 0.9383
    m_isobar = rng.uniform(m1 + m3 + 0.1, m0 - m2 - 0.1)
    p_isobar, p2 = _two_body_decay(m0, m_isobar, m2, rng)
    p3_isobar, p1_isobar = _two_body_decay(m_isobar, m3, m1, rng)
    return {
        "p1": _boost_from_rest_frame_of(p_isobar, p1_isobar),
        "p2": p2,
        "p3": _boost_from_rest_frame_of(p_isobar, p3_isobar),
    }


def _compute_boost_chain(momenta: dict[str, np.ndarray], state_id: int) -> np.ndarray:
    """Matrix that transforms CM momenta into the helicity frame of a final state.

    Follows the same rotation-boost sequence as
    :func:`ampform.kinematics.angles.compute_helicity_angles`: directly from the
    center-of-momentum frame for the spectator of subsystem 2, or through the
    isobar rest frame for its decay products.
    """
    if state_id == 2:
        return _helicity_transform(momenta["p2"])
    isobar_transform = _helicity_transform(momenta["p1"] + momenta["p3"])
    p_in_isobar = isobar_transform @ momenta[f"p{state_id}"]
    return _helicity_transform(p_in_isobar) @ isobar_transform


def _helicity_transform(p: np.ndarray) -> np.ndarray:
    """Rotate ``p`` onto the z-axis and boost into its rest frame."""
    φ = np.arctan2(p[2], p[1])
    θ = np.arccos(p[3] / np.linalg.norm(p[1:]))
    β = np.linalg.norm(p[1:]) / p[0]
    return _boost_z(β) @ _rotation_y(-θ) @ _rotation_z(-φ)


def _two_body_decay(
    parent_mass: float, m_a: float, m_b: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    p = np.sqrt(
        (parent_mass**2 - (m_a + m_b) ** 2) * (parent_mass**2 - (m_a - m_b) ** 2)
    ) / (2 * parent_mass)
    cos_θ = rng.uniform(-1, 1)
    sin_θ = np.sqrt(1 - cos_θ**2)
    φ = rng.uniform(-np.pi, +np.pi)
    direction = np.array([sin_θ * np.cos(φ), sin_θ * np.sin(φ), cos_θ])
    momentum = p * direction
    return (
        np.array([np.sqrt(m_a**2 + p**2), *momentum]),
        np.array([np.sqrt(m_b**2 + p**2), *-momentum]),
    )


def _boost_from_rest_frame_of(parent: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Boost ``p`` from the rest frame of ``parent`` to the frame of ``parent``."""
    mass = np.sqrt(parent[0] ** 2 - parent[1:] @ parent[1:])
    momentum_norm = np.linalg.norm(parent[1:])
    direction = parent[1:] / momentum_norm
    γ = parent[0] / mass
    βγ = momentum_norm / mass
    p_parallel = p[1:] @ direction
    p_perpendicular = p[1:] - p_parallel * direction
    energy = γ * p[0] + βγ * p_parallel
    parallel = βγ * p[0] + γ * p_parallel
    return np.array([energy, *(p_perpendicular + parallel * direction)])


def _boost_z(β: float) -> np.ndarray:
    γ = 1 / np.sqrt(1 - β**2)
    matrix = np.eye(4)
    matrix[0, 0] = matrix[3, 3] = γ
    matrix[0, 3] = matrix[3, 0] = -γ * β
    return matrix


def _rotation_y(angle: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[1, 1] = matrix[3, 3] = np.cos(angle)
    matrix[1, 3] = np.sin(angle)
    matrix[3, 1] = -np.sin(angle)
    return matrix


def _rotation_z(angle: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[1, 1] = matrix[2, 2] = np.cos(angle)
    matrix[1, 2] = -np.sin(angle)
    matrix[2, 1] = np.sin(angle)
    return matrix
