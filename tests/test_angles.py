import numpy as np
import pytest
import sympy as sp
from ampform.kinematics.phasespace import Kallen, compute_third_mandelstam

from ampform_dpd.angles import (
    formulate_scattering_angle,
    formulate_theta_hat_angle,
    formulate_zeta_angle,
)

m0, m1, m2, m3 = sp.symbols("m:4", nonnegative=True)
σ1, σ2, σ3 = sp.symbols("sigma1:4", nonnegative=True)


def describe_formulate_scattering_angle():
    def it_matches_reference_for_subsystem_23():
        assert formulate_scattering_angle(2, 3)[1] == sp.acos(
            (
                2 * σ1 * (-(m1**2) - m2**2 + σ3)
                - (m0**2 - m1**2 - σ1) * (m2**2 - m3**2 + σ1)
            )
            / (sp.sqrt(Kallen(m0**2, m1**2, σ1)) * sp.sqrt(Kallen(σ1, m2**2, m3**2)))
        )

    def it_matches_reference_for_subsystem_31():
        assert formulate_scattering_angle(3, 1)[1] == sp.acos(
            (
                2 * σ2 * (-(m2**2) - m3**2 + σ1)
                - (m0**2 - m2**2 - σ2) * (-(m1**2) + m3**2 + σ2)
            )
            / (sp.sqrt(Kallen(m0**2, m2**2, σ2)) * sp.sqrt(Kallen(σ2, m3**2, m1**2)))
        )


def describe_formulate_theta_hat_angle():
    def it_matches_reference_expression():
        assert formulate_theta_hat_angle(1, 2)[1] == sp.acos(
            (
                (m0**2 + m1**2 - σ1) * (m0**2 + m2**2 - σ2)
                - 2 * m0**2 * (σ3 - m1**2 - m2**2)
            )
            / (sp.sqrt(Kallen(m0**2, m2**2, σ2)) * sp.sqrt(Kallen(m0**2, σ1, m1**2)))
        )

    def it_is_antisymmetric():
        assert formulate_theta_hat_angle(1, 2)[1] == -formulate_theta_hat_angle(2, 1)[1]

    @pytest.mark.parametrize("i", [1, 2, 3])
    def it_vanishes_for_equal_indices(i: int):
        assert formulate_theta_hat_angle(i, i)[1] == 0


def describe_formulate_zeta_angle():
    @pytest.mark.parametrize("rotated_state", [1, 2, 3])
    @pytest.mark.parametrize("aligned_subsystem", [1, 2, 3])
    @pytest.mark.parametrize("reference_subsystem", [1, 2, 3])
    def it_has_zero_massless_limit(
        rotated_state, aligned_subsystem, reference_subsystem
    ):
        masses = {i: sp.S.Zero if i == rotated_state else sp.S.One for i in (1, 2, 3)}
        energies = {i: sp.sqrt(1 + mass**2) for i, mass in masses.items()}
        substitutions = {
            sp.Symbol(f"m{i}", nonnegative=True): mass for i, mass in masses.items()
        }
        substitutions[m0] = sum(energies.values())
        for k in (1, 2, 3):
            i, j = sorted({1, 2, 3} - {k})
            substitutions[sp.Symbol(f"sigma{k}", nonnegative=True)] = (
                masses[i] ** 2 + masses[j] ** 2 + 2 * energies[i] * energies[j] + 1
            )
        _, angle = formulate_zeta_angle(
            rotated_state, aligned_subsystem, reference_subsystem
        )
        assert angle.doit().subs(substitutions).simplify() == 0

    def it_satisfies_equation_a6():
        """Test Eq. (A6), https://journals.aps.org/prd/pdf/10.1103/PhysRevD.101.034033#page=10."""
        for i in [1, 2, 3]:
            for k in [1, 2, 3]:
                _, ζi_k0 = formulate_zeta_angle(i, k, 0)
                _, ζi_ki = formulate_zeta_angle(i, k, i)
                _, ζi_kk = formulate_zeta_angle(i, k, k)
                assert ζi_ki == ζi_k0
                assert ζi_kk == 0

    @pytest.mark.parametrize(
        ("ζ1_expr", "ζ2_expr", "ζ3_expr"),
        [
            (
                formulate_zeta_angle(1, 2, 3)[1],
                formulate_zeta_angle(1, 2, 1)[1],
                formulate_zeta_angle(1, 1, 3)[1],
            ),
            (
                formulate_zeta_angle(2, 3, 1)[1],
                formulate_zeta_angle(2, 3, 2)[1],
                formulate_zeta_angle(2, 2, 1)[1],
            ),
            (
                formulate_zeta_angle(3, 1, 2)[1],
                formulate_zeta_angle(3, 1, 3)[1],
                formulate_zeta_angle(3, 3, 2)[1],
            ),
        ],
    )
    def it_satisfies_angle_sum(ζ1_expr: sp.Expr, ζ2_expr: sp.Expr, ζ3_expr: sp.Expr):
        """Test Eq. (A9), https://journals.aps.org/prd/pdf/10.1103/PhysRevD.101.034033#page=11."""
        σ3_expr = compute_third_mandelstam(σ1, σ2, m0, m1, m2, m3)
        masses = {
            m0: 2.3,
            m1: 0.94,
            m2: 0.14,
            m3: 0.49,
            σ1: 1.2,
            σ2: 3.0,
            σ3: σ3_expr,
        }
        ζ1 = float(ζ1_expr.doit().xreplace(masses).xreplace(masses))
        ζ2 = float(ζ2_expr.doit().xreplace(masses).xreplace(masses))
        ζ3 = float(ζ3_expr.doit().xreplace(masses).xreplace(masses))
        np.testing.assert_almost_equal(ζ1, ζ2 + ζ3, decimal=14)
