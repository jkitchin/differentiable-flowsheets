"""Product blending pool with nonlinear blending rules.

A refinery makes its margin in the blender: component streams are mixed
into finished products (gasoline, jet, ULSD, fuel oil) whose *properties*
must meet specifications.  A planning LP blends those properties linearly by
volume, sometimes through a blending index, and protects the plan with a
back-off.  The real blend is nonlinear -- octane numbers interact, vapor
pressure and viscosity blend through indices -- and :class:`BlendPool` is
the rigorous side of that picture: the nonlinear rules refiners actually
use, differentiable in the recipe and in every component property.

Rules implemented (each cited at its function):

=====================  =================================================
property               rule
=====================  =================================================
RON, MON               Ethyl RT-70 interaction model (:func:`ethyl_rt70`)
RVP                    RVP^1.25 blending index (:func:`rvp_index_blend`);
                       Raoult on the pseudocomponents (:func:`raoult_rvp`)
S, N, CCR              by mass
aromatics, olefins,    by volume
benzene, SG, smoke pt
flash point            Hu-Burns index (:func:`flash_point_blend`)
freeze, cloud, pour,   power-law temperature index
CFPP                   (:func:`temperature_index_blend`)
viscosity              Refutas viscosity blending number, by mass
                       (:func:`refutas_blend`)
distillation           from the blend's pseudocomponent composition
                       (TBP, then Riazi-Daubert TBP -> ASTM D86)
cetane index           ASTM D4737 or D976 on the blend's density and D86
                       distillation -- computed, never blended
=====================  =================================================

Two ways to describe a component:

* **property mode** -- :meth:`BlendComponent.from_properties`: a dict of
  measured or unit-reported properties.  Enough for every blended property.
* **stream mode** -- :meth:`BlendComponent.from_stream`: a difflow stream
  of pseudocomponent molar flows on a shared
  :class:`~difflow_refinery.characterization.BlendCharacterization`.  Adds the
  product stream itself, an exact mass and volume balance, and the
  properties that only composition can give (distillation, cetane index,
  Raoult RVP).  Unit-reported properties (RON/MON from a reformer, say)
  are passed as overrides and take precedence.

Volumes are ideal-mixing volumes at 15 degC from ``SG``: the product
volume is the sum of the component volumes, and the product ``SG`` is the
volume average of the component ``SG``.

Example::

    pool = BlendPool("gasoline")
    res = pool([reformate, fcc_gasoline, alkylate, butane],
               recipe=[0.35, 0.35, 0.25, 0.05])
    res.properties["RON"], res.margins["RON >= 91"]
    pool.linear_blend_error(components, recipe)   # what a backoff must cover
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow_refinery.characterization import (
    PSI,
    R_GAS,
    RHO_WATER_15C,
    BlendCharacterization,
)

jax.config.update("jax_enable_x64", True)

#: 100 degF, the Reid vapor pressure test temperature (K).
T_RVP = 310.92777777777775

_C0 = 273.15


# ---------------------------------------------------------------------------
# Blending rules.  Each takes component values and VOLUME fractions ``v``
# (any non-negative weights; they are normalised here), so a recipe in flows
# works as well as one in fractions.
# ---------------------------------------------------------------------------

def _normalise(v: Array) -> Array:
    # An empty pool has no properties; dividing by one instead of zero keeps
    # an optimizer that touches the origin out of NaN (see
    # ``BlendPool.spec_margins(weighted=True)``).
    v = jnp.asarray(v, dtype=jnp.float64)
    total = jnp.sum(v)
    return v / jnp.where(total > 0, total, 1.0)


def volume_blend(v: Array, p: Array) -> Array:
    """Linear blend by volume: ``sum(v_i p_i) / sum(v_i)``."""
    v = _normalise(v)
    return jnp.sum(v * jnp.asarray(p))


def mass_blend(v: Array, sg: Array, p: Array) -> Array:
    """Linear blend by mass, with mass weights ``v_i SG_i``.

    The basis for properties specified as weight fractions (sulfur,
    nitrogen, CCR).
    """
    w = _normalise(jnp.asarray(v) * jnp.asarray(sg))
    return jnp.sum(w * jnp.asarray(p))


@dataclass(frozen=True)
class EthylRT70(ParamsMixin):
    """Coefficients of the Ethyl RT-70 octane blending model.

    Healy, Maassen & Peterson, "A new approach to blending octanes", API
    Division of Refining, 24th midyear meeting, New York (1959); Ethyl Corp.
    report RT-70.  The values are the 75-blend fit as restated in the
    gasoline-blending literature (e.g. Singh, Forbes, Vermeer & Woo,
    *Comput. Chem. Eng.* 24 (2000) 1027).

    Attributes:
        a1: RON sensitivity interaction coefficient.
        a2: RON olefin-spread coefficient.
        a3: RON aromatic-spread coefficient (zero in the published fit).
        b1: MON olefin interaction coefficient.
        b2: MON olefin-spread coefficient.
        b3: MON aromatic-spread coefficient (on the squared spread / 1e4).
    """

    a1: float = 0.03224
    a2: float = 0.00101
    a3: float = 0.0
    b1: float = 0.04450
    b2: float = 0.00081
    b3: float = -0.0645


def ethyl_rt70(v: Array, ron: Array, mon: Array, olefins: Array,
               aromatics: Array, coeffs: EthylRT70 = EthylRT70()
               ) -> tuple[Array, Array]:
    """Blend RON and MON with the Ethyl RT-70 interaction model.

    With volume averages written as bars, ``s = RON - MON`` the
    sensitivity and ``O``, ``A`` the olefin and aromatic contents (vol%)::

        RON = r + a1 (rs - r s) + a2 (O^2 - O O) + a3 (A^2 - A A)^2 / 1e4
        MON = m + b1 (mO - m O) + b2 (O^2 - O O) + b3 (A^2 - A A)^2 / 1e4

    where ``rs`` means the average of the product ``r_i s_i`` and ``r s``
    the product of the averages.  Every correction is a *spread* (a
    covariance or variance across the components), so the model reduces
    exactly to the linear volume blend whenever the components agree --
    one component, or components with identical sensitivity, olefins and
    aromatics.  Healy, Maassen & Peterson (1959); see :class:`EthylRT70`.

    Args:
        v: Volume fractions (or flows) of the components.
        ron, mon: Component research and motor octane numbers.
        olefins, aromatics: Component olefin and aromatic contents (vol%).
        coeffs: Model coefficients.

    Returns:
        ``(RON, MON)`` of the blend.
    """
    v = _normalise(v)

    def avg(x):
        return jnp.sum(v * x)

    ron, mon = jnp.asarray(ron), jnp.asarray(mon)
    olefins, aromatics = jnp.asarray(olefins), jnp.asarray(aromatics)
    s = ron - mon
    r_bar, m_bar, s_bar = avg(ron), avg(mon), avg(s)
    o_bar, a_bar = avg(olefins), avg(aromatics)
    o_spread = avg(olefins ** 2) - o_bar ** 2
    a_spread = avg(aromatics ** 2) - a_bar ** 2
    c = coeffs
    blend_ron = (r_bar + c.a1 * (avg(ron * s) - r_bar * s_bar)
                 + c.a2 * o_spread + c.a3 * a_spread ** 2 / 1e4)
    blend_mon = (m_bar + c.b1 * (avg(mon * olefins) - m_bar * o_bar)
                 + c.b2 * o_spread + c.b3 * a_spread ** 2 / 1e4)
    return blend_ron, blend_mon


def rvp_index_blend(v: Array, rvp: Array, exponent: float = 1.25) -> Array:
    """Blend Reid vapor pressure through the ``RVP^1.25`` index, by volume.

    ``RVP_blend = (sum v_i RVP_i^1.25)^(1/1.25)``.  The Chevron blending
    index; Gary, Handwerk & Kaiser, *Petroleum Refining: Technology and
    Economics*, chapter on product blending.  Units cancel,
    so any pressure unit works.
    """
    v = _normalise(v)
    return jnp.sum(v * jnp.asarray(rvp) ** exponent) ** (1.0 / exponent)


def raoult_rvp(moles: Array, psat: Array, molar_volume: Array,
               vapor_liquid_ratio: float = 4.0, T: float = T_RVP) -> Array:
    """Reid vapor pressure from Raoult's law on the pseudocomponents.

    The RVP bomb has four volumes of air space per volume of liquid, so the
    measured pressure is that of a liquid that has *lost* some of its light
    ends to the vapor.  With ``c_j = r v_L Psat_j / (R T)`` (``r`` the
    vapor/liquid volume ratio, ``v_L`` the liquid molar volume) the
    equilibrium liquid is ``x_j ~ z_j / (1 + c_j)`` and the pressure is
    ``sum x_j Psat_j`` -- closed form, because the vapor moles scale with the
    same pressure Raoult's law returns.

    Ideal liquid, no dissolved-air correction: this is the rigorous
    *check* on the index, not a replacement for a measured RVP.

    Args:
        moles: Pseudocomponent molar amounts (any scale).
        psat: Pseudocomponent vapor pressures at ``T`` (Pa).
        molar_volume: Pseudocomponent liquid molar volumes (m^3/mol).
        vapor_liquid_ratio: Air-space to liquid volume ratio (4 for D323).
        T: Test temperature (K); 100 degF by default.

    Returns:
        Vapor pressure (Pa).
    """
    z = moles / jnp.sum(moles)
    v_liq = jnp.sum(z * molar_volume)
    c = vapor_liquid_ratio * v_liq * psat / (R_GAS * T)
    x = z / (1.0 + c)
    x = x / jnp.sum(x)
    return jnp.sum(x * psat)


def flash_point_blend(v: Array, flash_K: Array) -> Array:
    """Blend flash points (K) through the Hu-Burns index, by volume.

    ``log10 BI = -6.1188 + 2414 / (T - 42.6)`` with ``T`` in K -- Hu & Burns,
    *Hydrocarbon Processing* 49(11) (1970); restated in Riazi,
    *Characterization and Properties of Petroleum Fractions* (ASTM MNL50,
    2005), Ch. 3.  The same index as Wickey & Chittenden's
    ``log10 BI = -6.1188 + 4345.2 / (T_F + 383)`` in degF.  The index falls
    steeply with temperature, so a little low-flash component pulls the
    blend's flash point down hard -- which is the physics.
    """
    v = _normalise(v)
    log10_bi = -6.1188 + 2414.0 / (jnp.asarray(flash_K) - 42.6)
    log10_blend = jax.nn.logsumexp(log10_bi * jnp.log(10.0), b=v) / jnp.log(10.0)
    return 42.6 + 2414.0 / (log10_blend + 6.1188)


#: Exponents of the power-law temperature indices ``BI = T^n`` (T in K).
#: Cloud (n = 1/0.05) and pour (n = 1/0.08) are Hu & Burns (1970), as
#: restated in Riazi MNL50 Ch. 3.  Freeze point and CFPP have no index of
#: their own in that source: freeze point is a crystal-*disappearance*
#: temperature like the cloud point and takes its exponent, and CFPP a
#: flow-plugging temperature like the pour point and takes that one.  Both
#: are approximations; override through ``BlendPool(rules=...)``.
TEMPERATURE_INDEX_EXPONENTS = {
    "cloud_C": 1.0 / 0.05,
    "freeze_C": 1.0 / 0.05,
    "pour_C": 1.0 / 0.08,
    "CFPP_C": 1.0 / 0.08,
}


def temperature_index_blend(v: Array, T_K: Array, exponent: float) -> Array:
    """Blend a cold-flow temperature through ``BI = T^n``, by volume.

    ``T_blend = (sum v_i T_i^n)^(1/n)`` with absolute temperatures; a scale
    change of the unit (K vs degR) cancels, an offset (degC) does not.
    Evaluated in log space since ``T^20`` is ~1e50.
    """
    v = _normalise(v)
    log_t = jnp.log(jnp.asarray(T_K))
    return jnp.exp(jax.nn.logsumexp(exponent * log_t, b=v) / exponent)


def refutas_vbn(nu: Array) -> Array:
    """Refutas viscosity blending number of a kinematic viscosity (cSt).

    ``VBN = 14.534 ln(ln(nu + 0.8)) + 10.975`` -- the Refutas correlation,
    as given in Maples, *Petroleum Refinery Process Economics*, 2nd ed.
    (2000).
    """
    return 14.534 * jnp.log(jnp.log(jnp.asarray(nu) + 0.8)) + 10.975


def refutas_viscosity(vbn: Array) -> Array:
    """Invert :func:`refutas_vbn`: ``nu = exp(exp((VBN - 10.975)/14.534)) - 0.8``."""
    return jnp.exp(jnp.exp((vbn - 10.975) / 14.534)) - 0.8


def refutas_blend(v: Array, sg: Array, nu: Array) -> Array:
    """Blend kinematic viscosity (cSt) by the Refutas method.

    The VBN blends linearly by *mass*; the component viscosities must be at
    the same temperature, and so is the result.
    """
    w = _normalise(jnp.asarray(v) * jnp.asarray(sg))
    return refutas_viscosity(jnp.sum(w * refutas_vbn(nu)))


# -- distillation -------------------------------------------------------------

#: Riazi & Daubert (1986) TBP <-> ASTM D86 interconversion, ``TBP = a D86^b``
#: in K, per volume-percent point; adopted by the API Technical Data Book,
#: restated in Riazi MNL50 Ch. 3.
TBP_D86 = {
    0: (0.9177, 1.0019), 10: (0.5564, 1.0900), 30: (0.7617, 1.0425),
    50: (0.9013, 1.0176), 70: (0.8821, 1.0226), 90: (0.9552, 1.0110),
    95: (0.8177, 1.0355),
}


def tbp_to_d86(tbp_K: Array, percent: int) -> Array:
    """ASTM D86 temperature (K) from the TBP temperature at ``percent``."""
    if percent not in TBP_D86:
        raise ValueError(f"no TBP->D86 coefficients at {percent}%; "
                         f"available: {sorted(TBP_D86)}")
    a, b = TBP_D86[percent]
    return (tbp_K / a) ** (1.0 / b)


def tbp_evaporated(T: Array, Tb: Array, phi: Array, width: float) -> Array:
    """Volume fraction boiling below ``T`` on a smoothed TBP curve.

    Each pseudocomponent's volume fraction ``phi_j`` is spread around its
    boiling point by a logistic of scale ``width`` (K), which turns the
    pseudocomponent staircase into a differentiable curve.
    """
    return jnp.sum(phi * jax.nn.sigmoid((T - Tb) / width))


def tbp_temperature(fraction: Array, Tb: Array, phi: Array,
                    width: float, iterations: int = 80) -> Array:
    """Invert :func:`tbp_evaporated`: the TBP temperature (K) at ``fraction``.

    Bisection to round-off, then one Newton step with the derivative held
    fixed, which attaches the implicit-function gradient
    ``dT = -(dE/dtheta) / (dE/dT)`` to the result.
    """
    lo = jnp.min(Tb) - 40.0 * width
    hi = jnp.max(Tb) + 40.0 * width
    Tb_s, phi_s = jax.lax.stop_gradient(Tb), jax.lax.stop_gradient(phi)
    frac_s = jax.lax.stop_gradient(fraction)

    def body(_, bracket):
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        below = tbp_evaporated(mid, Tb_s, phi_s, width) < frac_s
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(0, iterations, body,
                               (jax.lax.stop_gradient(lo),
                                jax.lax.stop_gradient(hi)))
    T0 = 0.5 * (lo + hi)
    dEdT = jax.grad(tbp_evaporated)(T0, Tb_s, phi_s, width)
    return T0 - (tbp_evaporated(T0, Tb, phi, width) - fraction) / dEdT


def cetane_index_d976(density: Array, t50_C: Array) -> Array:
    """Two-variable cetane index, ASTM D976.

    ``CI = 454.74 - 1641.416 D + 774.74 D^2 - 0.554 B + 97.803 (log10 B)^2``
    with ``D`` the density at 15 degC (g/mL) and ``B`` the D86 50% point
    (degC).
    """
    D, B = density, t50_C
    return (454.74 - 1641.416 * D + 774.74 * D ** 2 - 0.554 * B
            + 97.803 * jnp.log10(B) ** 2)


def cetane_index_d4737(density: Array, t10_C: Array, t50_C: Array,
                       t90_C: Array) -> Array:
    """Four-variable cetane index, ASTM D4737 Procedure A.

    ``CI = 45.2 + 0.0892 T10N + (0.131 + 0.901 B) T50N
    + (0.0523 - 0.420 B) T90N + 0.00049 (T10N^2 - T90N^2) + 107 B + 60 B^2``

    with ``B = exp(-3.5 (D - 0.85)) - 1``, ``T10N = T10 - 215``,
    ``T50N = T50 - 260``, ``T90N = T90 - 310`` (D86, degC) and ``D`` the
    density at 15 degC (g/mL).  At the reference fuel (0.85, 215, 260, 310)
    every correction vanishes and CI = 45.2.
    """
    B = jnp.exp(-3.5 * (density - 0.85)) - 1.0
    t10n, t50n, t90n = t10_C - 215.0, t50_C - 260.0, t90_C - 310.0
    return (45.2 + 0.0892 * t10n + (0.131 + 0.901 * B) * t50n
            + (0.0523 - 0.420 * B) * t90n
            + 0.00049 * (t10n ** 2 - t90n ** 2) + 107.0 * B + 60.0 * B ** 2)


# ---------------------------------------------------------------------------
# Which rule each blended property follows.
# ---------------------------------------------------------------------------

#: Blended properties and their default rule.  ``"volume"``/``"mass"`` are
#: linear on that basis; the rest are the nonlinear rules above.
PROPERTY_RULES: dict[str, str] = {
    "SG": "volume",
    "S_ppm": "mass", "N_ppm": "mass", "CCR_wt": "mass",
    "aromatics_vol": "volume", "olefins_vol": "volume",
    "benzene_vol": "volume", "naphthenes_vol": "volume",
    "paraffins_vol": "volume", "smoke_mm": "volume",
    "RON": "ethyl", "MON": "ethyl",
    "RVP_psi": "index",
    "flash_C": "index",
    "cloud_C": "index", "freeze_C": "index", "pour_C": "index",
    "CFPP_C": "index",
    "viscosity_cSt": "refutas",
}

#: Allowed rule names per property (the first is the default).
_ALLOWED = {
    "RON": ("ethyl", "volume"), "MON": ("ethyl", "volume"),
    "RVP_psi": ("index", "raoult", "volume"),
    "flash_C": ("index", "volume"),
    "cloud_C": ("index", "volume"), "freeze_C": ("index", "volume"),
    "pour_C": ("index", "volume"), "CFPP_C": ("index", "volume"),
    "viscosity_cSt": ("refutas", "mass", "volume"),
}

_DISTILLATION_POINTS = (10, 50, 90, 95)


@dataclass
class BlendSpec(ParamsMixin):
    """A product specification on one blended property.

    Attributes:
        property: Property name, e.g. ``"RON"`` or ``"S_ppm"``.
        op: ``">="`` (a minimum) or ``"<="`` (a maximum).
        limit: The specification value, in the property's units.
        scale: Divisor applied to the margin, so margins on different
            properties are comparable as constraints.  Default 1.
        name: Label; defaults to ``"<property> <op> <limit>"``.
    """

    property: str
    op: str
    limit: float
    scale: float = 1.0
    name: str | None = None

    def __post_init__(self):
        if self.op not in (">=", "<="):
            raise ValueError(f"spec op must be '>=' or '<=', got {self.op!r}")
        if self.name is None:
            self.name = f"{self.property} {self.op} {self.limit:g}"

    def margin(self, value: Array) -> Array:
        """Signed margin: positive when met, zero at the limit."""
        m = value - self.limit if self.op == ">=" else self.limit - value
        return m / self.scale


#: Illustrative product specifications.  Real ones are grade-, season- and
#: market-specific -- pass your own ``specs=``.  Gasoline: US regular
#: (AKI 87) summer; jet: ASTM D1655 Jet A; ULSD: ASTM D975 No. 2-D S15;
#: fuel oil: IMO 2020 very-low-sulfur RMG 380.
PRODUCT_SPECS: dict[str, list[BlendSpec]] = {
    "gasoline": [BlendSpec("RON", ">=", 91.0), BlendSpec("MON", ">=", 82.0),
                 BlendSpec("RVP_psi", "<=", 9.0),
                 BlendSpec("S_ppm", "<=", 10.0)],
    "jet": [BlendSpec("S_ppm", "<=", 3000.0),
            BlendSpec("freeze_C", "<=", -40.0),
            BlendSpec("smoke_mm", ">=", 18.0),
            BlendSpec("flash_C", ">=", 38.0)],
    "ulsd": [BlendSpec("S_ppm", "<=", 15.0),
             BlendSpec("cetane_index", ">=", 40.0),
             BlendSpec("flash_C", ">=", 52.0),
             BlendSpec("T90_d86_C", "<=", 338.0)],
    "fuel_oil": [BlendSpec("S_ppm", "<=", 5000.0),
                 BlendSpec("viscosity_cSt", "<=", 380.0),
                 BlendSpec("SG", "<=", 0.991),
                 BlendSpec("CCR_wt", "<=", 18.0)],
}

#: Composition-derived properties each product reports in stream mode.
PRODUCT_DERIVED: dict[str, tuple[str, ...]] = {
    "gasoline": ("E70_tbp", "E100_tbp", "T10_d86_C", "T50_d86_C",
                 "T90_d86_C", "RVP_raoult_psi"),
    "jet": ("T10_d86_C", "T90_d86_C"),
    "ulsd": ("T10_d86_C", "T50_d86_C", "T90_d86_C", "T95_d86_C",
             "cetane_index"),
    "fuel_oil": ("T50_d86_C",),
}


# ---------------------------------------------------------------------------
# Components.
# ---------------------------------------------------------------------------

@dataclass
class BlendComponent:
    """One blend component: its properties, and optionally its composition.

    Build one with :meth:`from_properties` or :meth:`from_stream`.

    Attributes:
        name: Component name.
        properties: Blended properties, ``{name: value}``; keys from
            :data:`PROPERTY_RULES`.  ``SG`` is required.
        characterization: The pseudocomponent grid (stream mode only).
        moles_per_volume: Pseudocomponent mol per m^3 of component at
            15 degC (stream mode only).
        available_volume: Volume flow the source stream carries (m^3/s at
            15 degC; stream mode only).
        T, P: Source stream temperature (K) and pressure (Pa).
    """

    name: str
    properties: dict[str, Array]
    characterization: BlendCharacterization | None = None
    moles_per_volume: Array | None = None
    available_volume: Array | None = None
    T: Array | None = None
    P: Array | None = None

    def __post_init__(self):
        unknown = sorted(set(self.properties) - set(PROPERTY_RULES))
        if unknown:
            raise ValueError(
                f"component {self.name!r}: unknown properties {unknown}; "
                f"blended properties are {sorted(PROPERTY_RULES)}")
        if "SG" not in self.properties:
            raise ValueError(f"component {self.name!r} needs an SG: every "
                             "volume/mass conversion goes through it")

    @property
    def has_composition(self) -> bool:
        return self.moles_per_volume is not None

    @classmethod
    def from_properties(cls, name: str, **properties: Any) -> "BlendComponent":
        """A component described by its properties alone.

        Example::

            alkylate = BlendComponent.from_properties(
                "alkylate", SG=0.70, RON=96.0, MON=93.5, RVP_psi=4.5,
                S_ppm=5.0, olefins_vol=0.5, aromatics_vol=0.5)
        """
        props = {k: jnp.asarray(v, dtype=jnp.float64)
                 for k, v in properties.items()}
        return cls(name=name, properties=props)

    @classmethod
    def from_stream(cls, name: str, stream: Mapping[str, Any],
                    characterization: BlendCharacterization,
                    **overrides: Any) -> "BlendComponent":
        """A component computed from a stream of pseudocomponent flows.

        ``SG`` comes from the composition (volume-weighted, ideal mixing at
        15 degC), each of the characterization's ``qualities`` on its own
        basis, and ``RVP_psi`` from :func:`raoult_rvp`.  ``overrides`` -- a
        RON/MON reported by the reformer, a measured flash point -- win
        over anything computed.

        Args:
            name: Component name.
            stream: difflow stream with ``F_<pseudocomponent>`` flows (mol/s).
            characterization: The grid the stream's species belong to.
            **overrides: Property values, keys from :data:`PROPERTY_RULES`.
        """
        char = characterization
        moles = char.flows(stream)
        vol = moles * char.molar_volume
        V = jnp.sum(vol)
        phi = vol / V
        mass = moles * char.mw
        w = mass / jnp.sum(mass)
        props: dict[str, Array] = {"SG": jnp.sum(phi * char.SG)}
        for key, vec in char.qualities.items():
            basis = char.QUALITY_BASIS[key]
            props[key] = jnp.sum((w if basis == "mass" else phi) * vec)
        props["RVP_psi"] = raoult_rvp(moles, char.psat(T_RVP),
                                      char.molar_volume) / PSI
        for k, v in overrides.items():
            props[k] = jnp.asarray(v, dtype=jnp.float64)
        return cls(name=name, properties=props, characterization=char,
                   moles_per_volume=moles / V, available_volume=V,
                   T=jnp.asarray(stream.get("T", 288.15), dtype=jnp.float64),
                   P=jnp.asarray(stream.get("P", 101325.0), dtype=jnp.float64))


# ---------------------------------------------------------------------------
# The pool.
# ---------------------------------------------------------------------------

@dataclass
class BlendResult:
    """What :class:`BlendPool` returns for one recipe.

    Attributes:
        volume: Product volume flow at 15 degC (recipe units; m^3/s in
            stream mode).
        mass: Product mass flow (kg per recipe volume unit; kg/s in stream
            mode).
        properties: ``{property: value}`` of the blend.
        margins: ``{spec name: signed margin}``; positive is on spec.
        stream: Product stream (stream mode only, else ``None``).
        volume_fractions: The recipe as volume fractions.
    """

    volume: Array
    mass: Array
    properties: dict[str, Array]
    margins: dict[str, Array]
    stream: dict[str, Array] | None
    volume_fractions: Array

    def margin_vector(self) -> Array:
        """Margins as an array, in spec order."""
        return jnp.stack(list(self.margins.values())) if self.margins \
            else jnp.zeros(0)

    def violations(self, temperature: float | None = None) -> Array:
        """Spec violations ``max(-margin, 0)``; see :func:`smooth_violation`."""
        return smooth_violation(self.margin_vector(), temperature)


def smooth_violation(margin: Array, temperature: float | None = None
                     ) -> Array:
    """``max(-margin, 0)``, or its softplus smoothing at ``temperature``.

    The smooth form is ``t * logaddexp(-margin/t, 0)``: it is within
    ``t ln 2`` of the kink, and its derivative at an active spec
    (``margin = 0``) is exactly ``-1/2``, so gradients exist and are
    informative right at the boundary.  (The branchless
    ``max(u,0) + log1p(exp(-|u|))`` has the same value and a derivative of
    *zero* there -- see ``difflow.stochastic``.)
    """
    margin = jnp.asarray(margin)
    if temperature is None:
        return jnp.maximum(-margin, 0.0)
    t = temperature
    return t * jnp.logaddexp(-margin / t, 0.0)


class BlendPool:
    """A steady-state product blending pool with nonlinear blending rules.

    Args:
        product: ``"gasoline"``, ``"jet"``, ``"ulsd"`` or ``"fuel_oil"``.
            Picks the default specs and the composition-derived properties
            to report; any other name is allowed with explicit ``specs``.
        specs: :class:`BlendSpec` list, or tuples ``(property, op, limit)``.
            Defaults to :data:`PRODUCT_SPECS` for the product.
        rules: Per-property rule overrides, ``{property: rule}``.  Rules:
            ``"volume"``, ``"mass"`` (linear), ``"ethyl"`` (RON/MON),
            ``"index"`` (RVP, flash and cold-flow temperatures),
            ``"raoult"`` (RVP, stream mode), ``"refutas"`` (viscosity).
            Also ``"octane"`` as shorthand for both RON and MON,
            ``"cetane": "d4737" | "d976"``, and ``"<property>_exponent"``
            for a cold-flow index exponent.
        ethyl: RT-70 coefficients.
        distillation_width: Logistic smoothing (K) of the TBP staircase.
        derived: Composition-derived properties to report in stream mode;
            defaults to :data:`PRODUCT_DERIVED` for the product.

    Call it with a list of :class:`BlendComponent` and a recipe; see
    :meth:`__call__`.
    """

    def __init__(self, product: str = "gasoline",
                 specs: Sequence[BlendSpec | tuple] | None = None,
                 rules: Mapping[str, Any] | None = None,
                 ethyl: EthylRT70 = EthylRT70(),
                 distillation_width: float = 5.0,
                 derived: Sequence[str] | None = None):
        if specs is None:
            if product not in PRODUCT_SPECS:
                raise ValueError(
                    f"no default specs for product {product!r}; pass specs= "
                    f"(products with defaults: {sorted(PRODUCT_SPECS)})")
            specs = PRODUCT_SPECS[product]
        self.product = product
        self.specs = [s if isinstance(s, BlendSpec) else BlendSpec(*s)
                      for s in specs]
        names = [s.name for s in self.specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate spec names: {names}")
        self.rules = dict(PROPERTY_RULES)
        self.exponents = dict(TEMPERATURE_INDEX_EXPONENTS)
        self.cetane = "d4737"
        for key, rule in dict(rules or {}).items():
            if key == "octane":
                self._set_rule("RON", rule)
                self._set_rule("MON", rule)
            elif key == "cetane":
                if rule not in ("d4737", "d976"):
                    raise ValueError("cetane rule must be 'd4737' or 'd976'")
                self.cetane = rule
            elif key.endswith("_exponent"):
                prop = key[: -len("_exponent")]
                if prop not in self.exponents:
                    raise ValueError(f"no temperature index for {prop!r}")
                self.exponents[prop] = float(rule)
            else:
                self._set_rule(key, rule)
        self.ethyl = ethyl
        self.distillation_width = distillation_width
        self.derived = tuple(derived if derived is not None
                             else PRODUCT_DERIVED.get(product, ()))

    def _set_rule(self, prop: str, rule: str) -> None:
        if prop not in PROPERTY_RULES:
            raise ValueError(f"unknown property {prop!r}")
        allowed = _ALLOWED.get(prop, ("volume", "mass"))
        if rule not in allowed:
            raise ValueError(
                f"rule {rule!r} not available for {prop!r}; choose from "
                f"{allowed}")
        self.rules[prop] = rule

    def __repr__(self) -> str:
        return (f"BlendPool(product={self.product!r}, "
                f"specs={[s.name for s in self.specs]})")

    # -- recipe ---------------------------------------------------------------

    @staticmethod
    def _stream_mode(components: Sequence[BlendComponent]) -> bool:
        modes = {c.has_composition for c in components}
        if len(modes) > 1:
            raise ValueError(
                "mix of stream-mode and property-mode components: either "
                "every component is built from a stream (and the pool "
                "returns a product stream) or none is")
        stream = modes.pop()
        if stream:
            chars = {id(c.characterization) for c in components}
            if len(chars) > 1:
                raise ValueError("stream-mode components must share one "
                                 "BlendCharacterization object")
        return stream

    def volumes(self, components: Sequence[BlendComponent], recipe: Any,
                basis: str = "volume_fraction",
                total_volume: Any = 1.0) -> Array:
        """Component volume flows into the pool.

        Args:
            components: The blend components.
            recipe: One entry per component.
            basis: ``"volume_fraction"`` (scaled to ``total_volume``),
                ``"volume_flow"`` (used as given), or ``"split"`` (fraction
                of each component's available stream volume; stream mode).
            total_volume: Product volume for the ``"volume_fraction"`` basis.
        """
        r = jnp.asarray(recipe, dtype=jnp.float64)
        if r.shape != (len(components),):
            raise ValueError(f"recipe needs {len(components)} entries, got "
                             f"shape {r.shape}")
        if basis == "volume_fraction":
            return total_volume * r / jnp.sum(r)
        if basis == "volume_flow":
            return r
        if basis == "split":
            if not self._stream_mode(components):
                raise ValueError("basis='split' needs stream-mode components")
            return r * jnp.stack([c.available_volume for c in components])
        raise ValueError(f"unknown recipe basis {basis!r}")

    # -- evaluation -----------------------------------------------------------

    def _column(self, components, prop) -> Array:
        missing = [c.name for c in components if prop not in c.properties]
        if missing:
            raise KeyError(f"{prop!r} missing on components {missing}")
        return jnp.stack([c.properties[prop] for c in components])

    def _available(self, components) -> list[str]:
        common = set(components[0].properties)
        for c in components[1:]:
            common &= set(c.properties)
        return [p for p in PROPERTY_RULES if p in common]

    def blend_properties(self, components: Sequence[BlendComponent],
                         v: Array) -> dict[str, Array]:
        """Nonlinear blend of every property all components carry.

        ``v`` are component volumes (any scale).  Composition-derived
        properties are added in stream mode.
        """
        props = {}
        sg = self._column(components, "SG")
        available = self._available(components)
        stream_mode = self._stream_mode(components)
        for p in available:
            rule = self.rules[p]
            col = self._column(components, p)
            if p in ("RON", "MON") and rule == "ethyl":
                # RT-70 gives both numbers; take only this one, so a rule set
                # to "volume" on the other is not overwritten.
                need = ("RON", "MON", "olefins_vol", "aromatics_vol")
                lacking = [q for q in need if q not in available]
                if lacking:
                    raise KeyError(
                        f"the Ethyl RT-70 octane rule needs {lacking} on "
                        "every component; supply them or pass "
                        "rules={'octane': 'volume'}")
                ron, mon = ethyl_rt70(
                    v, self._column(components, "RON"),
                    self._column(components, "MON"),
                    self._column(components, "olefins_vol"),
                    self._column(components, "aromatics_vol"), self.ethyl)
                props[p] = ron if p == "RON" else mon
            elif rule == "volume":
                props[p] = volume_blend(v, col)
            elif rule == "mass":
                props[p] = mass_blend(v, sg, col)
            elif p == "RVP_psi" and rule == "index":
                props[p] = rvp_index_blend(v, col)
            elif p == "RVP_psi" and rule == "raoult":
                if not stream_mode:
                    raise ValueError("the Raoult RVP rule needs stream-mode "
                                     "components")
                props[p] = self._raoult_psi(self._product_moles(components, v),
                                            components[0].characterization)
            elif p == "flash_C":
                props[p] = flash_point_blend(v, col + _C0) - _C0
            elif p in self.exponents:
                props[p] = temperature_index_blend(
                    v, col + _C0, self.exponents[p]) - _C0
            elif p == "viscosity_cSt":
                props[p] = refutas_blend(v, sg, col)
            else:  # pragma: no cover - guarded by _set_rule
                raise ValueError(f"no rule {rule!r} for {p!r}")
        if "RON" in props and "MON" in props:
            props["AKI"] = 0.5 * (props["RON"] + props["MON"])
        props["density_kgm3"] = props["SG"] * RHO_WATER_15C
        if stream_mode:
            char = components[0].characterization
            props.update(self._derived(self._product_moles(components, v),
                                       char))
        return props

    @staticmethod
    def _product_moles(components, v) -> Array:
        return jnp.sum(jnp.stack([vi * c.moles_per_volume
                                  for vi, c in zip(v, components)]), axis=0)

    @staticmethod
    def _raoult_psi(moles, char) -> Array:
        return raoult_rvp(moles, char.psat(T_RVP), char.molar_volume) / PSI

    def _derived(self, moles: Array, char: BlendCharacterization
                 ) -> dict[str, Array]:
        """Composition-derived properties of a pseudocomponent mixture."""
        out: dict[str, Array] = {}
        if not self.derived:
            return out
        phi = char.volume_fractions(moles)
        w = self.distillation_width
        tbp = {}

        def tbp_at(pct):
            if pct not in tbp:
                tbp[pct] = tbp_temperature(pct / 100.0, char.Tb, phi, w)
            return tbp[pct]

        def d86_C(pct):
            return tbp_to_d86(tbp_at(pct), pct) - _C0

        for name in self.derived:
            if name in ("E70_tbp", "E100_tbp"):
                T = _C0 + float(name[1:-4])
                out[name] = 100.0 * tbp_evaporated(T, char.Tb, phi, w)
            elif name.endswith("_tbp_C"):
                out[name] = tbp_at(int(name[1:-6])) - _C0
            elif name.endswith("_d86_C"):
                out[name] = d86_C(int(name[1:-6]))
            elif name == "RVP_raoult_psi":
                out[name] = self._raoult_psi(moles, char)
            elif name == "cetane_index":
                dens = jnp.sum(phi * char.SG) * RHO_WATER_15C / 1000.0
                if self.cetane == "d976":
                    out[name] = cetane_index_d976(dens, d86_C(50))
                else:
                    out[name] = cetane_index_d4737(dens, d86_C(10),
                                                   d86_C(50), d86_C(90))
            else:
                raise ValueError(f"unknown derived property {name!r}")
        return out

    def __call__(self, components: Sequence[BlendComponent], recipe: Any,
                 basis: str = "volume_fraction",
                 total_volume: Any = 1.0) -> BlendResult:
        """Blend ``components`` by ``recipe``.

        Args:
            components: :class:`BlendComponent` list.
            recipe: One entry per component, on ``basis``.
            basis: ``"volume_fraction"``, ``"volume_flow"`` or ``"split"``
                (see :meth:`volumes`).
            total_volume: Product volume for ``"volume_fraction"``.

        Returns:
            :class:`BlendResult` with the product stream (stream mode), its
            properties and the signed spec margins.
        """
        components = list(components)
        V = self.volumes(components, recipe, basis, total_volume)
        props = self.blend_properties(components, V)
        sg = self._column(components, "SG")
        mass = jnp.sum(V * sg) * RHO_WATER_15C
        stream = None
        if self._stream_mode(components):
            char = components[0].characterization
            moles = self._product_moles(components, V)
            stream = {f"F_{n}": moles[j] for j, n in enumerate(char.names)}
            T = jnp.stack([c.T for c in components])
            P = jnp.stack([c.P for c in components])
            stream["T"] = jnp.sum(V * T) / jnp.sum(V)
            stream["P"] = jnp.min(P)
        return BlendResult(volume=jnp.sum(V), mass=mass, properties=props,
                           margins=self._margins(props), stream=stream,
                           volume_fractions=V / jnp.sum(V))

    def _margins(self, props: Mapping[str, Array]) -> dict[str, Array]:
        out = {}
        for s in self.specs:
            if s.property not in props:
                raise KeyError(
                    f"spec {s.name!r}: the blend has no {s.property!r} "
                    f"(available: {sorted(props)})")
            out[s.name] = s.margin(props[s.property])
        return out

    # -- optimizer and planning hooks ----------------------------------------

    def spec_margins(self, components: Sequence[BlendComponent], recipe: Any,
                     basis: str = "volume_fraction",
                     weighted: bool = False) -> Array:
        """Signed spec margins as a vector (spec order); ``>= 0`` is on spec.

        Smooth in the recipe and in every component property, so usable
        directly as inequality constraints ``g(recipe) >= 0``.

        Args:
            weighted: Multiply each margin by the product volume.  A property
                is intensive -- homogeneous of degree zero in the flows and
                0/0 at an empty pool -- and an optimizer over volume flows
                that wanders towards zero meets NaN.  ``V * margin >= 0`` is
                the same constraint wherever ``V > 0``, stays defined at
                ``V = 0``, and is the form a planning LP's blending rows take
                (``sum_i V_i (p_i - limit) >= 0``).
        """
        res = self(components, recipe, basis)
        m = res.margin_vector()
        return m * res.volume if weighted else m

    def spec_violations(self, components: Sequence[BlendComponent],
                        recipe: Any, basis: str = "volume_fraction",
                        temperature: float | None = None) -> Array:
        """Spec violations, ``max(-margin, 0)`` or smoothed.

        With ``temperature`` set the softplus form of
        :func:`smooth_violation` is used, which has a finite, non-zero
        gradient at an active spec -- what a penalty method needs.
        """
        return smooth_violation(self.spec_margins(components, recipe, basis),
                                temperature)

    def linear_properties(self, components: Sequence[BlendComponent],
                          recipe: Any, basis: str = "volume_fraction",
                          exact: Sequence[str] = ()) -> dict[str, Array]:
        """Every property blended *linearly by volume* -- the planning-LP view.

        Composition-derived properties are the volume average of each
        component's own value.

        Args:
            exact: Properties the LP already models with the pool's own rule
                -- RVP through its ``RVP^1.25`` index, sulfur by mass -- and
                which therefore take that rule here instead of the linear
                one.  Only rules that are linear in the volumes (``"index"``
                on RVP, ``"mass"``, ``"volume"``) are allowed: the point is
                to describe an LP.
        """
        components = list(components)
        V = self.volumes(components, recipe, basis)
        available = self._available(components)
        sg = self._column(components, "SG")
        for p in exact:
            rule = self.rules.get(p)
            if not (rule in ("mass", "volume")
                    or (p == "RVP_psi" and rule == "index")):
                raise ValueError(
                    f"{p!r} uses rule {rule!r}, which an LP cannot hold "
                    "exactly; exact= takes properties whose pool rule is "
                    "linear in the volumes")
        props = {}
        for p in available:
            col = self._column(components, p)
            if p in exact and p == "RVP_psi":
                props[p] = rvp_index_blend(V, col)
            elif p in exact and self.rules[p] == "mass":
                props[p] = mass_blend(V, sg, col)
            else:
                props[p] = volume_blend(V, col)
        if "RON" in props and "MON" in props:
            props["AKI"] = 0.5 * (props["RON"] + props["MON"])
        props["density_kgm3"] = props["SG"] * RHO_WATER_15C
        if self._stream_mode(components) and self.derived:
            char = components[0].characterization
            own = [self._derived(c.moles_per_volume, char)
                   for c in components]
            for name in self.derived:
                props[name] = volume_blend(
                    V, jnp.stack([d[name] for d in own]))
        return props

    def linear_blend_error(self, components: Sequence[BlendComponent],
                           recipe: Any, basis: str = "volume_fraction",
                           exact: Sequence[str] = ()) -> dict[str, Array]:
        """Nonlinear minus linear-by-volume blend, per property.

        This is the error a planning LP makes at this recipe, and so the
        quantity its back-off on each spec is meant to cover.  ``exact``
        names properties the LP models with the pool's own (linear-in-volume)
        rule, whose error is then zero; see :meth:`linear_properties`.
        """
        nonlinear = self(components, recipe, basis).properties
        linear = self.linear_properties(components, recipe, basis, exact)
        return {k: nonlinear[k] - linear[k] for k in linear if k in nonlinear}

    def backoff(self, components: Sequence[BlendComponent], recipe: Any,
                basis: str = "volume_fraction",
                exact: Sequence[str] = ()) -> dict[str, Array]:
        """Per-spec back-off that makes the LP's view exact at ``recipe``.

        ``max(linear margin - nonlinear margin, 0)``, keyed by spec name: how
        far the linear blend overstates each margin.  Tightening each LP row
        by this amount and re-solving, until the plan stops moving, is the
        successive back-off a planner runs against a rigorous blend model.
        """
        err = self.linear_blend_error(components, recipe, basis, exact)
        out = {}
        for s in self.specs:
            if s.property not in err:
                raise KeyError(f"spec {s.name!r}: no linear view of "
                               f"{s.property!r}")
            # nonlinear margin minus linear margin, in margin units
            d = err[s.property] if s.op == ">=" else -err[s.property]
            out[s.name] = jnp.maximum(-d / s.scale, 0.0)
        return out

    def as_block(self, components: Sequence[BlendComponent],
                 name: str | None = None,
                 outputs: Sequence[str] | None = None,
                 lb: Any = None, ub: Any = None, u0: Any = None,
                 **kwargs: Any):
        """A :class:`difflow.planning.Block` over the blend recipe.

        Levers ``u`` are the component volume flows into the pool
        (``"<component>_V"``).  Outputs are the product volume
        (``"volume"``), the chosen properties, and every spec margin
        (``"margin:<spec name>"``) -- the form ``difflow.planning`` turns
        into delta vectors and ``>= 0`` constraint rows.

        Args:
            components: The blend components (held fixed).
            name: Block name; defaults to the product.
            outputs: Properties to expose; defaults to every spec property.
            lb, ub, u0: Bounds and linearisation point on the volumes.
                ``ub`` defaults to each component's available volume in
                stream mode.
            **kwargs: Passed to :class:`~difflow.planning.Block`.
        """
        from difflow.planning import Block

        components = list(components)
        if outputs is None:
            outputs = list(dict.fromkeys(s.property for s in self.specs))
        y_names = (["volume"] + list(outputs)
                   + [f"margin:{s.name}" for s in self.specs])
        n = len(components)
        if lb is None:
            lb = [0.0] * n
        if ub is None and self._stream_mode(components):
            ub = [float(c.available_volume) for c in components]

        def fn(u):
            res = self(components, u, basis="volume_flow")
            return jnp.stack([res.volume]
                             + [res.properties[p] for p in outputs]
                             + list(res.margins.values()))

        return Block(name=name or self.product, fn=fn,
                     u_names=[f"{c.name}_V" for c in components],
                     y_names=y_names, lb=lb, ub=ub, u0=u0, **kwargs)
