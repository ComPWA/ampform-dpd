"""Spin-density matrices for formulating polarized intensities.

The distribution of a decay that originates from a polarized initial state depends on
the orientation of the decay plane with respect to the production frame. This
orientation is parametrized by three Euler angles :math:`(\\phi, \\theta, \\chi)` and
the polarization itself is described by a spin-density matrix :math:`\\rho`. See
:cite:`Marangotto:2019ucc` and `this section
<https://redeboer.github.io/phd-thesis/chapter3.html#sec-differential-decay-rate>`_ for
the theory behind this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp
from sympy.physics.matrices import msigma

if TYPE_CHECKING:
    from typing import SupportsFloat

EULER_ANGLES: tuple[sp.Symbol, sp.Symbol, sp.Symbol] = sp.symbols(
    "phi theta chi", real=True
)
r""":math:`(\phi, \theta, \chi)` Euler angles that orient the decay plane.

The angles rotate the production frame onto the frame in which the helicities of the
three-body decay are defined. See `formulate_polarized_intensity
<.DalitzPlotDecompositionBuilder.formulate_polarized_intensity>`.
"""


def create_spin_density_matrix(spin: SupportsFloat) -> sp.MatrixSymbol:
    r"""Create a `~sympy.matrices.expressions.MatrixSymbol` :math:`\rho` for a spin-density matrix.

    The matrix has dimension :math:`2j+1` and its rows and columns are ordered from
    highest to lowest spin projection, that is, index :math:`i` corresponds to spin
    projection :math:`m = j - i`. This matches the standard basis ordering of the Pauli
    matrices (see `formulate_polarized_spin_half_density`).

    >>> rho = create_spin_density_matrix(spin=1)
    >>> rho.shape
    (3, 3)
    >>> create_spin_density_matrix(spin=0.75)
    Traceback (most recent call last):
        ...
    ValueError: Spin must be a multiple of 1/2, got 0.75
    """
    return sp.MatrixSymbol(R"\rho", *2 * [_get_dimension(spin)])


def formulate_unpolarized_density(spin: SupportsFloat) -> sp.ImmutableDenseMatrix:
    r"""Formulate the spin-density matrix of an unpolarized particle.

    The unpolarized density is proportional to the identity matrix,
    :math:`\rho = \tfrac{1}{2j+1}\mathbb{1}`, meaning that all spin projections are
    equally populated and that there are no coherences between them.

    >>> formulate_unpolarized_density(spin=0.5)
    Matrix([
    [1/2,   0],
    [  0, 1/2]])
    """
    n = _get_dimension(spin)
    return sp.ImmutableDenseMatrix(sp.eye(n) / n)


def formulate_polarized_spin_half_density(Px, Py, Pz) -> sp.ImmutableDenseMatrix:  # noqa: N803
    r"""Formulate the spin-density matrix of a polarized spin-½ particle.

    The density matrix is expressed in terms of a polarization vector
    :math:`\vec{P} = (P_x, P_y, P_z)` as
    :math:`\rho = \tfrac{1}{2}\left(\mathbb{1} + \vec{P}\cdot\vec\sigma\right)`, with
    :math:`\vec\sigma` the Pauli matrices.

    >>> Px, Py, Pz = sp.symbols("P_x P_y P_z", real=True)
    >>> formulate_polarized_spin_half_density(Px, Py, Pz)
    Matrix([
    [    P_z/2 + 1/2, P_x/2 - I*P_y/2],
    [P_x/2 + I*P_y/2,     1/2 - P_z/2]])
    """
    polarization = Px * msigma(1) + Py * msigma(2) + Pz * msigma(3)
    return sp.ImmutableDenseMatrix((sp.eye(2) + polarization) / 2)


def formulate_ee_annihilation_density() -> sp.ImmutableDenseMatrix:
    r"""Formulate the spin-density matrix of a vector meson from :math:`e^+e^-` annihilation.

    A vector meson (such as the :math:`J/\psi`) that is produced through unpolarized
    :math:`e^+e^-` annihilation can only have spin projections :math:`m = \pm 1` along
    the beam axis, without coherence between them:

    >>> formulate_ee_annihilation_density()
    Matrix([
    [1/2, 0,   0],
    [  0, 0,   0],
    [  0, 0, 1/2]])
    """
    return sp.ImmutableDenseMatrix(sp.diag(1, 0, 1) / 2)


def _get_dimension(spin: SupportsFloat) -> int:
    two_j = 2 * float(spin)
    if two_j != int(two_j) or two_j < 0:
        msg = f"Spin must be a multiple of 1/2, got {float(spin)}"
        raise ValueError(msg)
    return int(two_j) + 1
