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


def test_formulate_scattering_angle():
    assert formulate_scattering_angle(2, 3)[1] == sp.acos(
        (
            2 * σ1 * (-(m1**2) - m2**2 + σ3)
            - (m0**2 - m1**2 - σ1) * (m2**2 - m3**2 + σ1)
        )
        / (
            sp.sqrt(Kallen(m0**2, m1**2, σ1))
            * sp.sqrt(Kallen(σ1, m2**2, m3**2))
        )
    )
    assert formulate_scattering_angle(3, 1)[1] == sp.acos(
        (
            2 * σ2 * (-(m2**2) - m3**2 + σ1)
            - (m0**2 - m2**2 - σ2) * (-(m1**2) + m3**2 + σ2)
        )
        / (
            sp.sqrt(Kallen(m0**2, m2**2, σ2))
            * sp.sqrt(Kallen(σ2, m3**2, m1**2))
        )
    )


def test_formulate_theta_hat_angle():
    assert formulate_theta_hat_angle(1, 2)[1] == sp.acos(
        (
            (m0**2 + m1**2 - σ1) * (m0**2 + m2**2 - σ2)
            - 2 * m0**2 * (σ3 - m1**2 - m2**2)
        )
        / (
            sp.sqrt(Kallen(m0**2, m2**2, σ2))
            * sp.sqrt(Kallen(m0**2, σ1, m1**2))
        )
    )
    assert formulate_theta_hat_angle(1, 2)[1] == -formulate_theta_hat_angle(2, 1)[1]
    for i in [1, 2, 3]:
        assert formulate_theta_hat_angle(i, i)[1] == 0


def test_formulate_zeta_angle_equation_A6():
    """Test Eq. (A6), https://journals.aps.org/prd/pdf/10.1103/PhysRevD.101.034033#page=10.
    """
    for i in [1, 2, 3]:
        for k in [1, 2, 3]:
            _, ζi_k0 = formulate_zeta_angle(i, k, 0)
            _, ζi_ki = formulate_zeta_angle(i, k, i)
            _, ζi_kk = formulate_zeta_angle(i, k, k)
            assert ζi_ki == ζi_k0
            assert ζi_kk == 0


@pytest.mark.parametrize(
    ("ζ1", "ζ2", "ζ3"),
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
def test_formulate_zeta_angle_sum_rule(ζ1: sp.Expr, ζ2: sp.Expr, ζ3: sp.Expr):
    """Test Eq. (A9), https://journals.aps.org/prd/pdf/10.1103/PhysRevD.101.034033#page=11.
    """
    σ3_expr = compute_third_mandelstam(σ1, σ2, m0, m1, m2, m3)
    masses = {m0: 2.3, m1: 0.94, m2: 0.14, m3: 0.49, σ1: 1.2, σ2: 3.0, σ3: σ3_expr}
    ζ1 = float(ζ1.doit().xreplace(masses).xreplace(masses))
    ζ2 = float(ζ2.doit().xreplace(masses).xreplace(masses))
    ζ3 = float(ζ3.doit().xreplace(masses).xreplace(masses))
    np.testing.assert_almost_equal(ζ1, ζ2 + ζ3, decimal=14)
