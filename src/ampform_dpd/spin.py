from __future__ import annotations

from decimal import Decimal
from typing import Generator, SupportsFloat, SupportsInt

import sympy as sp


def generate_ls_couplings(
    parent_spin: SupportsFloat,
    child1_spin: SupportsFloat,
    child2_spin: SupportsFloat,
    max_L: int = 3,
) -> list[tuple[int, sp.Rational]]:
    r"""
    >>> generate_ls_couplings(1.5, 0.5, 0)
    [(1, 1/2), (2, 1/2)]
    """
    s1 = float(child1_spin)
    s2 = float(child2_spin)
    angular_momenta = create_rational_range(0, max_L)
    coupled_spins = create_rational_range(abs(s1 - s2), s1 + s2)
    ls_couplings = {
        (int(L), S)
        for L in angular_momenta
        for S in coupled_spins
        if abs(L - S) <= parent_spin <= L + S
    }
    return sorted(ls_couplings)


def filter_parity_violating_ls(
    ls_couplings: list[tuple[int, sp.Rational]],
    parent_parity: SupportsInt,
    child1_parity: SupportsInt,
    child2_parity: SupportsInt,
) -> list[tuple[int, sp.Rational]]:
    r"""
    >>> LS = generate_ls_couplings(0.5, 1.5, 0)  # Λc → Λ(1520)π
    >>> LS
    [(1, 3/2), (2, 3/2)]
    >>> filter_parity_violating_ls(LS, +1, -1, -1)
    [(2, 3/2)]
    """
    η0, η1, η2 = (
        int(parent_parity),
        int(child1_parity),
        int(child2_parity),
    )
    return [(L, S) for L, S in ls_couplings if η0 == η1 * η2 * (-1) ** L]


def create_spin_range(spin: SupportsFloat) -> list[sp.Rational]:
    """
    >>> create_spin_range(1.5)
    [-3/2, -1/2, 1/2, 3/2]
    """
    return create_rational_range(-spin, spin)


def create_rational_range(
    __from: SupportsFloat, __to: SupportsFloat
) -> list[sp.Rational]:
    """
    >>> create_rational_range(-0.5, +1.5)
    [-1/2, 1/2, 3/2]
    """
    spin_range = arange(float(__from), +float(__to) + 0.5)
    return list(map(sp.Rational, spin_range))


def arange(x_1: float, x_2: float, delta: float = 1.0) -> Generator[float, None, None]:
    current = Decimal(x_1)
    while current < x_2:
        yield float(current)
        current += Decimal(delta)
