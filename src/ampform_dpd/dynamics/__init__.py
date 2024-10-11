"""Functions for dynamics lineshapes and kinematics."""

# pyright:reportPrivateUsage=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp
from ampform.dynamics.form_factor import FormFactor
from ampform.kinematics.phasespace import Kallen
from ampform.sympy import unevaluated

if TYPE_CHECKING:
    from sympy.printing.latex import LatexPrinter


@unevaluated
class RelativisticBreitWigner(sp.Expr):
    s: Any
    mass0: Any
    gamma0: Any
    m1: Any
    m2: Any
    angular_momentum: Any
    meson_radius: Any
    _latex_repr_ = (
        R"\mathcal{{R}}_{{{angular_momentum}}}\left({s}, {mass0}, {gamma0}\right)"
    )

    def evaluate(self):
        from ampform.dynamics import EnergyDependentWidth  # noqa: PLC0415

        s, m0, w0, m1, m2, angular_momentum, meson_radius = self.args
        width = EnergyDependentWidth(
            s=s,
            mass0=m0,
            gamma0=w0,
            m_a=m1,
            m_b=m2,
            angular_momentum=angular_momentum,
            meson_radius=meson_radius,
            name=Rf"\Gamma_{{{sp.latex(angular_momentum)}}}",
        )
        return (m0 * w0) / (m0**2 - s - width * m0 * sp.I)


@unevaluated
class P(sp.Expr):
    s: Any
    mi: Any
    mj: Any
    _latex_repr_ = R"p_{{_{{{mi},{mj}}}}}\left({s}\right)"

    def evaluate(self):
        s, mi, mj = self.args
        return sp.sqrt(Kallen(s, mi**2, mj**2)) / (2 * sp.sqrt(s))


@unevaluated
class Q(sp.Expr):
    s: Any
    m0: Any
    mk: Any
    _latex_repr_ = R"q_{{_{{{m0},{mk}}}}}\left({s}\right)"

    def evaluate(self):
        s, m0, mk = self.args
        return sp.sqrt(Kallen(s, m0**2, mk**2)) / (2 * m0)  # <-- not s!


@unevaluated
class BreitWignerMinL(sp.Expr):
    s: Any
    decaying_mass: Any
    spectator_mass: Any
    resonance_mass: Any
    resonance_width: Any
    child2_mass: Any
    child1_mass: Any
    l_dec: Any
    l_prod: Any
    R_dec: Any
    R_prod: Any
    _latex_repr_ = R"\mathcal{{R}}^\mathrm{{BW}}_{{{l_dec},{l_prod}}}\left({s}\right)"

    def evaluate(self):  # noqa: PLR0914
        s, m_top, m_spec, m0, Γ0, m1, m2, l_dec, l_prod, R_dec, R_prod = self.args
        q = Q(s, m_top, m_spec)
        q0 = Q(m0**2, m_top, m_spec)
        p = P(s, m1, m2)
        p0 = P(m0**2, m1, m2)
        width = EnergyDependentWidth(s, m0, Γ0, m1, m2, l_dec, R_dec)
        return sp.Mul(
            (q / q0) ** l_prod,
            BlattWeisskopf(q * R_prod, l_prod) / BlattWeisskopf(q0 * R_prod, l_prod),
            1 / (m0**2 - s - sp.I * m0 * width),
            (p / p0) ** l_dec,
            BlattWeisskopf(p * R_dec, l_dec) / BlattWeisskopf(p0 * R_dec, l_dec),
            evaluate=False,
        )


@unevaluated
class BuggBreitWigner(sp.Expr):
    s: Any
    m0: Any
    Γ0: Any
    m1: Any
    m2: Any
    γ: Any
    _latex_repr_ = R"\mathcal{{R}}^\mathrm{{Bugg}}\left({s}\right)"

    def evaluate(self):
        s, m0, Γ0, m1, m2, γ = self.args
        s_A = m1**2 - m2**2 / 2  # Adler zero  # noqa: N806
        g_squared = sp.Mul(
            (s - s_A) / (m0**2 - s_A),
            m0 * Γ0 * sp.exp(-γ * s),
            evaluate=False,
        )
        return 1 / (m0**2 - s - sp.I * g_squared)


@unevaluated
class FlattéSWave(sp.Expr):
    # https://github.com/ComPWA/polarimetry/blob/34f5330/julia/notebooks/model0.jl#L151-L161
    s: Any
    m0: Any
    widths: tuple[Any, Any]
    masses1: tuple[Any, Any]
    masses2: tuple[Any, Any]
    _latex_repr_ = R"\mathcal{{R}}^\mathrm{{Flatté}}\left({s}\right)"

    def evaluate(self):
        s, m0, (Γ1, Γ2), (ma1, mb1), (ma2, mb2) = self.args
        p = P(s, ma1, mb1)
        p0 = P(m0**2, ma2, mb2)
        q = P(s, ma2, mb2)
        q0 = P(m0**2, ma2, mb2)
        Γ1 *= (p / p0) * m0 / sp.sqrt(s)
        Γ2 *= (q / q0) * m0 / sp.sqrt(s)
        Γ = Γ1 + Γ2
        return 1 / (m0**2 - s - sp.I * m0 * Γ)


@unevaluated
class EnergyDependentWidth(sp.Expr):
    s: Any
    m0: Any
    Γ0: Any
    m1: Any
    m2: Any
    L: Any
    R: Any

    def evaluate(self):
        s, m0, Γ0, m1, m2, L, R = self.args
        p = P(s, m1, m2)
        p0 = P(m0**2, m1, m2)
        ff = BlattWeisskopf(p * R, L) ** 2
        ff0 = BlattWeisskopf(p0 * R, L) ** 2
        return sp.Mul(
            Γ0,
            (p / p0) ** (2 * L + 1),
            m0 / sp.sqrt(s),
            ff / ff0,
            evaluate=False,
        )

    def _latex_repr_(self, printer: LatexPrinter) -> str:
        s = printer._print(self.s)  # pyright:ignore[reportPrivateUsage]
        if self.L == 0:
            if self.m1 == 0 and self.m2 == 0:
                return printer._print(self.Γ0)  # pyright:ignore[reportPrivateUsage]
            return Rf"\Gamma\left({s}\right)"
        L = printer._print(self.L)  # pyright:ignore[reportPrivateUsage]
        return Rf"\Gamma_{{{L}}}\left({s}\right)"


@unevaluated
class BlattWeisskopf(sp.Expr):
    z: Any
    L: Any
    _latex_repr_ = R"F_{{{L}}}\left({z}\right)"

    def evaluate(self) -> sp.Piecewise:
        z, L = self.args
        cases = {
            0: 1,
            1: 1 / (1 + z**2),  # type:ignore[operator]
            2: 1 / (9 + 3 * z**2 + z**4),  # type:ignore[operator]
        }
        return sp.Piecewise(*[
            (sp.sqrt(expr), sp.Eq(L, l_val)) for l_val, expr in cases.items()
        ])


@unevaluated
class MultichannelBreitWigner(sp.Expr):
    s: Any
    mass: Any
    channels: tuple[ChannelArguments, ...]

    def evaluate(self):
        s = self.s
        m0 = self.mass
        width = sum(channel.evaluate() for channel in self.channels)
        return BreitWigner(s, m0, width)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        latex = R"\mathcal{R}^\mathrm{BW}_\mathrm{multi}\left("
        latex += printer._print(self.s) + "; "
        latex += ", ".join(printer._print(channel.width) for channel in self.channels)
        latex += R"\right)"
        return latex


@unevaluated
class ChannelArguments(sp.Expr):
    s: Any
    m0: Any
    width: Any
    m1: Any = 0
    m2: Any = 0
    angular_momentum: Any = 0
    meson_radius: Any = 1
    _latex_repr_ = R"\Gamma^\text{channel}\left({{s}}, {{m0}}, {{width}}\right)"

    def evaluate(self) -> sp.Expr:
        s, m0, Γ0, m1, m2, L, R = self.args
        ff = FormFactor(s, m1, m2, L, R) ** 2
        return Γ0 * m0 / sp.sqrt(s) * ff  # type:ignore[operator]


@unevaluated
class BreitWigner(sp.Expr):
    s: Any
    mass: Any
    width: Any
    m1: Any = 0
    m2: Any = 0
    angular_momentum: Any = 0
    meson_radius: Any = 1

    def evaluate(self):
        width = self.energy_dependent_width()
        expr = SimpleBreitWigner(self.s, self.mass, width)
        if self.angular_momentum == 0 and self.m1 == 0 and self.m2 == 0:
            return expr.evaluate()
        return expr

    def energy_dependent_width(self) -> sp.Expr:
        s, m0, Γ0, m1, m2, L, d = self.args
        if L == 0 and m1 == 0 and m2 == 0:
            return Γ0  # type:ignore[return-value]
        return EnergyDependentWidth(s, m0, Γ0, m1, m2, L, d)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.s)
        function_symbol = R"\mathcal{R}^\mathrm{BW}"
        mass = printer._print(self.mass)
        width = printer._print(self.width)
        arg = Rf"\left({s}; {mass}, {width}\right)"
        L = printer._print(self.angular_momentum)
        if isinstance(self.angular_momentum, sp.Integer):
            return Rf"{function_symbol}_{{L={L}}}{arg}"
        return Rf"{function_symbol}_{{{L}}}{arg}"


@unevaluated
class SimpleBreitWigner(sp.Expr):
    s: Any
    mass: Any
    width: Any
    _latex_repr_ = R"\mathcal{{R}}^\mathrm{{BW}}\left({s}; {mass}, {width}\right)"

    def evaluate(self):
        s, m0, Γ0 = self.args
        return 1 / (m0**2 - s - m0 * Γ0 * 1j)
