"""Hydrocracking kinetics on the pseudo-component grid: continuous lumping, or discrete lumps.

:class:`HCKinetics` is a kinetic model for
:class:`~difflow_refinery.hydroprocessing.reactor.TrickleBedReactor` (the
interface of the shared hydroprocessing stack, #306). It cracks the molecules
of each cut into lighter cuts and C1-C5 gas, and -- optionally, on by
default -- runs the hydrotreating network of
:class:`~difflow_refinery.hydrotreating.kinetics.HDTKinetics` (HDS, HDN,
aromatics saturation) alongside it with the cracking catalyst's own
constants, because a cracking bed keeps desulfurizing and saturating.

The per-cut attributes are the hydrotreater's (:data:`HDT_ATTRIBUTES`) plus
one more, ``"cracked"``: the number of molecules in the cut that are cracked
PRODUCT (a count, no element). It rides with the molecules like every other
attribute and is what lets the unit report the gravity of a cut that holds
both feed molecules and cracked ones (:mod:`.unit`).

Cracking rate (per cut ``i``, molecules per s per kg of catalyst)::

    r_i = f k_i(T) c_i h(c_H2) / (1 + K_N(T) c_Norg + K_NH3 c_NH3)

with ``f = activity * effectiveness * wetting``, ``c_i`` the cut's
fugacity-equivalent liquid concentration (mol/m^3), ``c_Norg`` the total
ORGANIC nitrogen concentration of the liquid (all cuts, both nitrogen
classes -- the nitrogen the pretreat bed did not remove), ``h = (c_H2 /
c_ref)^m`` (``m = 0`` by default: first order in the hydrocarbon, as the
continuous-lumping model is), and ``k_i(T) = k_max exp(-E/R (1/T - 1/T_ref))
kappa_i``. The organic-N term is the Langmuir adsorption-inhibition form:
basic nitrogen compounds adsorb on the acid sites of the cracking function.
Its constants are ILLUSTRATIVE (chosen so that 10 wppm of organic N slows
cracking by a few tens of per cent); there is no transferable published set.

Where the cracked molecules go -- the two schemes (``scheme=``):

* ``"continuous"`` (default): **continuous lumping** after Laxminarasimhan,
  Verma & Ramachandran (1996). A cut's normalised boiling point is
  ``theta = (Tb - T_low)/(T_high - T_low)``; its reactivity is
  ``kappa = theta^(1/alpha)`` (``k = k_max theta^(1/alpha)``); the species-type
  distribution ``D(k) = dN/dk = N0/(alpha k_max^(1/alpha)) k^(1/alpha - 1)``;
  and the yield distribution of species of reactivity ``k`` from cracking
  species of reactivity ``K`` is::

      p(k, K) = [exp(-((k/K)^a0 - 0.5)^2 / a1) - exp(-0.25/a1) + delta (1 - k/K)] / (S0 sqrt(2 pi))

  with ``S0`` set by mass conservation, ``int_0^K p(k, K) D(k) dk = 1``. On
  the grid, the cracked mass of cut ``i`` is distributed over the lighter
  bins ``j`` (every lighter cut, and a gas bin below the lightest cut) in
  proportion to ``int_bin p(k(theta), K_i) D(k(theta)) dk/dtheta dtheta``
  (Gauss-Legendre, :data:`N_QUAD` nodes per bin), normalised over the bins
  -- which IS the ``S0`` normalisation, discretised. Products of cut ``i`` land
  only in cuts lighter than ``i`` (the integral stops at ``i``'s lower cut
  edge): a cracked molecule always leaves its parent's cut.

  These forms are the model of Laxminarasimhan et al. (1996) AS RESTATED IN
  THE LATER LITERATURE THAT USES IT, from recollection; the paper itself
  could not be reached, so its equation numbers are not given and the forms
  are marked unverified. Note what the restated forms imply, which a reader
  of the paper should check: with ``k = k_max theta^(1/alpha)`` and the
  ``D(k)`` above, the number of species per unit ``theta`` goes as
  ``theta^(1/alpha^2 - 1)`` -- uniform only at ``alpha = 1``. The code
  follows the forms as stated (``species_density`` is computed from them, not
  assumed uniform).

* ``"discrete"``: discrete lumps (e.g. VGO -> diesel -> kerosene -> naphtha
  -> gas), the scheme of Stangeland (1974) and of Mohanty, Saraf & Kunzru
  (1991) in its lump form. Lumps are TBP ranges (``lump_edges``); every cut in
  lump ``L`` cracks with ``kappa = lump_k[L]``; its products are shared
  between the lighter lumps by the selectivity row ``lump_selectivity[L]``
  and spread uniformly in ``theta`` over each product lump's cuts (that
  spreading is an assumption of this coding, not of the lump scheme). The
  default constants are ILLUSTRATIVE, not any published set.

Both schemes reduce to the same thing at run time -- a reactivity vector
``kappa`` and a row-stochastic distribution matrix ``W`` (cut -> gas bin and
lighter cuts) -- computed once per solve by :meth:`HCKinetics.prepare` from
the cut boiling points (traceable: a TBP point of the assay moves them).

What a cracked molecule turns into (property assignment -- a MODELLING
ASSUMPTION, stated in the docs):

* the product landing in cut ``j`` has the boiling point of cut ``j`` and the
  specific gravity of a **saturation-adjusted Watson K**: ``SG_j = (1.8
  Tb_j)^(1/3) / (Kw_feed + dKw)``, with ``Kw_feed`` the mass-average Watson K
  of the fresh feed's cuts and ``dKw`` (default +0.3, ILLUSTRATIVE) the
  shift to the more paraffinic/naphthenic cracked product;
* its molecular weight is Twu (1984) from ``(Tb_j, SG_j)`` (the
  characterization's correlation), its refractive index the composition
  module's Riazi-Daubert Huang index, its hydrogen content Goossens (1997),
  and its hydrocarbon types the Riazi-Daubert (1986) 2B4.1 estimate -- the
  composition module's own chain (#305), applied to the assigned ``(Tb, SG)``;
  its aromatics are split mono/di/poly by the hydrotreater's
  ``DEFAULT_AROMATIC_SPLIT``; no olefins;
* it carries no sulfur or nitrogen: a cracked molecule's heteroatoms leave as
  H2S and NH3 (hydrocracked products are heteroatom-poor; the uncracked
  molecules keep theirs and the hydrotreating network removes them);
* the gas bin's carbon becomes C1-C5 in the mole shares :data:`HCU_GAS_SPLIT`
  (ILLUSTRATIVE: hydrocracker LPG is isobutane-rich).

**Hydrogen consumption is the hydrogen balance** of that event: H atoms in
the products (cuts, gas, H2S, NH3) less those of the parent molecule, halved
-- so it follows the conversion and the product slate, not a separate
correlation. Carbon, sulfur and nitrogen are conserved by construction, and
hydrogen through the H2 drawn from the gas; :func:`~difflow_refinery.hydroprocessing.reactor.check_element_conservation`
returns zero to round-off.

**Heat** is per H2 consumed, split into C-C scission and saturation: each
cracking event that makes ``n`` product molecules from one parent breaks
``n - 1`` bonds, each with one H2 and the heat of n-hexane + H2 -> n-butane +
ethane (:data:`SCISSION_HEAT`, -42.7 kJ/mol, the hydrotreater's cracking-leak
model reaction); the remaining H2 (saturation of the product, and the
heteroatom removal) releases the benzene + 3 H2 -> cyclohexane heat per H2
(:data:`SATURATION_HEAT_PER_H2`, -68.4 kJ/mol H2). Both from the
model-compound thermochemistry of :mod:`difflow_refinery.hydrotreating.kinetics`.

References:
    Laxminarasimhan, C.S., Verma, R.P. and Ramachandran, P.A., "Continuous
        lumping model for simulation of hydrocracking", AIChE J. 42(9),
        2645-2653 (1996), doi:10.1002/aic.690420925 (bibliographic details
        confirmed by web search; the paper itself was not reached --
        equations as restated in later literature, numbering and the
        published parameter values UNVERIFIED and NOT used).
    Stangeland, B.E., "A kinetic model for the prediction of hydrocracker
        yields", Ind. Eng. Chem. Process Des. Dev. 13(1), 71-76 (1974),
        doi:10.1021/i260049a013 (title and DOI by web search; issue and
        pages as in the issue -- unverified). Discrete form, qualitative use.
    Mohanty, S., Saraf, D.N. and Kunzru, D., "Modeling of a hydrocracking
        reactor", Fuel Process. Technol. 29, 1-17 (1991) (title and pages
        unverified). Discrete lumps, qualitative use.
    Twu, C.H., Fluid Phase Equilib. 16, 137-150 (1984) -- MW of the products
        (via :mod:`difflow_refinery.correlations`, as cited there).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow_refinery import correlations as corr
from difflow_refinery.composition import (
    HC_TYPES, density_20c, estimate_refractive_index, hydrogen_goossens, riazi_daubert_pna)
from difflow_refinery.hydroprocessing.layout import ATOMIC_MASS, GAS_ELEMENTS, Layout
from difflow_refinery.hydroprocessing.reactor import Rates, ReactionContext
from difflow_refinery.hydrotreating.feed import DEFAULT_AROMATIC_SPLIT, aromatic_split
from difflow_refinery.hydrotreating.kinetics import (
    AROMATIC_CLASSES, AROMATIC_THERMO, CRACK_HEAT, HDT_ATTRIBUTE_ELEMENTS, HDT_ATTRIBUTES,
    HDTKineticParams, HDTKinetics, R_GAS)

jax.config.update("jax_enable_x64", True)

#: The hydrocracker's per-cut attributes: the hydrotreater's plus a count of
#: cracked-product molecules.
HCU_ATTRIBUTES: tuple[str, ...] = HDT_ATTRIBUTES + ("cracked",)
HCU_ATTRIBUTE_ELEMENTS: tuple[str | None, ...] = HDT_ATTRIBUTE_ELEMENTS + (None,)

#: Gas made by cracking, mole shares by species. ILLUSTRATIVE: hydrocracker
#: LPG and light gas are rich in iso-paraffins and poor in C1-C2 (the
#: qualitative finding of the hydrocracking literature, e.g. Stangeland 1974);
#: the numbers are chosen here, from no particular unit.
HCU_GAS_SPLIT: dict[str, float] = {
    "methane": 0.03, "ethane": 0.05, "propane": 0.22, "isobutane": 0.25, "n_butane": 0.12,
    "isopentane": 0.22, "n_pentane": 0.11,
}

#: Heat (J/mol) of one C-C scission with one H2: n-hexane + H2 -> n-butane +
#: ethane (the hydrotreater's cracking-leak model reaction).
SCISSION_HEAT: float = CRACK_HEAT
#: Heat (J per mol H2) of saturation: benzene + 3 H2 -> cyclohexane, per H2.
SATURATION_HEAT_PER_H2: float = AROMATIC_THERMO[2][0] / 3.0

#: Gauss-Legendre nodes per bin of the continuous-lumping quadrature.
N_QUAD: int = 8

SCHEMES = ("continuous", "discrete")


@dataclass
class HCKineticParams(ParamsMixin):
    """Constants of :class:`HCKinetics` (all ILLUSTRATIVE unless stated).

    Attributes:
        activity: Cracking-catalyst activity multiplier a(t) (1 fresh); the
            deactivation handle (``difflow.reconciliation.tracking``).
        effectiveness: Effectiveness factor (multiplier).
        wetting: Wetting efficiency (multiplier).
        scheme: ``"continuous"`` (Laxminarasimhan et al. 1996) or ``"discrete"`` (lumps).
        k_max: Rate constant of the heaviest species at ``T_ref`` (theta = 1),
            m^3 liquid per kg catalyst per s.
        E: Activation energy of cracking, J/mol.
        T_ref: Reference temperature of the Arrhenius and adsorption terms, K.
        T_low: Boiling point at theta = 0, K (the mixture's lightest; methane by default).
        T_high: Boiling point at theta = 1, K.
        alpha: Reactivity exponent: ``k = k_max theta^(1/alpha)``.
        a0: Yield-distribution shape exponent.
        a1: Yield-distribution width.
        delta: Yield-distribution light-end (gas) term.
        K_N: Organic-nitrogen adsorption constant at ``T_ref``, m^3/mol.
        dH_N: Its adsorption enthalpy, J/mol (negative: weakens with T).
        K_nh3: NH3 adsorption constant, m^3/mol (0 by default).
        h2_order: Order in dissolved H2 (0: first order in hydrocarbon only).
        c_ref: Reference H2 concentration of the H2 term, mol/m^3.
        dKw: Watson-K shift of the cracked product over the fresh feed's.
        lump_edges: Discrete scheme: TBP boundaries between liquid lumps, K
            (lightest first; the gas lump is everything below the lightest cut).
        lump_k: Discrete scheme: relative reactivity of each liquid lump.
        lump_selectivity: Discrete scheme: row L = shares of a lump-L
            parent's cracked carbon going to (gas, lump 0, lump 1, ...).
        hdt: :class:`HDTKineticParams` of the hydrotreating reactions on the
            cracking catalyst, or ``None`` to run cracking alone.
    """

    activity: float = 1.0
    effectiveness: float = 1.0
    wetting: float = 1.0
    scheme: str = "continuous"
    k_max: float = 4.0e-7
    E: float = 170e3
    T_ref: float = 653.15
    T_low: float = 111.66
    T_high: float = 873.15
    alpha: float = 0.55
    a0: float = 1.0
    a1: float = 0.2
    delta: float = 0.02
    K_N: float = 1.0
    dH_N: float = -60e3
    K_nh3: float = 0.0
    h2_order: float = 0.0
    c_ref: float = 1000.0
    dKw: float = 0.3
    lump_edges: tuple = (438.15, 533.15, 643.15)
    lump_k: tuple = (0.10, 0.25, 0.45, 1.0)
    lump_selectivity: tuple = (
        (1.00, 0.00, 0.00, 0.00, 0.00),     # naphtha -> gas
        (0.10, 0.90, 0.00, 0.00, 0.00),     # kerosene -> gas, naphtha
        (0.06, 0.44, 0.50, 0.00, 0.00),     # diesel -> gas, naphtha, kerosene
        (0.04, 0.26, 0.30, 0.40, 0.00),     # VGO -> gas, naphtha, kerosene, diesel
    )
    hdt: HDTKineticParams | None = field(default_factory=lambda: CRACK_BED_HDT_PARAMS)


#: Hydrotreating constants on the cracking catalyst (NiMo on a zeolite /
#: amorphous silica-alumina): HDS and HDN continue, aromatics saturate faster
#: than on a CoMo diesel catalyst, and the HDT model's own cracking leak is off
#: (the cracking model does the cracking). ILLUSTRATIVE, like every HDT constant.
CRACK_BED_HDT_PARAMS = HDTKineticParams(
    T_ref=653.15, hds_k=(4.0e-5, 2.0e-5, 1.2e-5, 1.0e-5, 6.0e-6), hdn_k=(4.0e-7, 7.0e-7),
    hda_k=(8.0e-6, 3.0e-6, 6.0e-7), crack_k=0.0)


def _register_params():
    data = [f for f in HCKineticParams.__dataclass_fields__ if f != "scheme"]
    jax.tree_util.register_dataclass(HCKineticParams, data_fields=data, meta_fields=["scheme"])


_register_params()


@dataclass(frozen=True)
class Prepared:
    """Per-solve constants of the cracking model (a pytree; see :meth:`HCKinetics.prepare`).

    Attributes:
        kappa: ``(n_cut,)`` relative reactivity of each cut (``k_i / k_max``).
        W: ``(n_cut, n_cut + 1)`` share of each cut's cracked carbon going to
            the gas bin (column 0) and to each cut (columns 1..); rows sum to 1
            and only lighter cuts receive.
        n_C: ``(n_cut,)`` carbon atoms per cracked-product molecule landing in each cut.
        h_per_C: ``(n_cut,)`` hydrogen atoms per carbon atom of that product.
        aromatics: ``(n_cut, 3)`` mono/di/poly-aromatic molecule fraction of it.
        naphthenes: ``(n_cut,)`` naphthenic molecule fraction of it.
        SG: ``(n_cut,)`` its specific gravity; MW: its molecular weight.
        gas_C, gas_H: C and H atoms per gas molecule (the mole-share average).
        gas_shares: ``(n_gas,)`` mole share of each layout gas in the cracked gas.
    """

    kappa: Array
    W: Array
    n_C: Array
    h_per_C: Array
    aromatics: Array
    naphthenes: Array
    SG: Array
    MW: Array
    gas_C: Array
    gas_H: Array
    gas_shares: Array


jax.tree_util.register_dataclass(Prepared, data_fields=["kappa", "W", "n_C", "h_per_C", "aromatics", "naphthenes",
                                                        "SG", "MW", "gas_C", "gas_H", "gas_shares"],
                                 meta_fields=[])


@dataclass(frozen=True)
class HCState:
    """What :meth:`HCKinetics.rates` takes as ``params``: the constants and the prepared tables."""

    p: HCKineticParams
    prep: Prepared


jax.tree_util.register_dataclass(HCState, data_fields=["p", "prep"], meta_fields=[])


def yield_distribution(u: Array, a0, a1, delta) -> Array:
    """The (unnormalised) Laxminarasimhan et al. (1996) yield distribution at ``u = k/K``.

    ``exp(-(u^a0 - 0.5)^2 / a1) - exp(-0.25/a1) + delta (1 - u)`` (the
    ``1/(S0 sqrt(2 pi))`` factor is the normalisation, applied by the
    caller). Zero at ``u = 1`` (a species does not crack into itself) and
    ``delta`` at ``u = 0`` (the light-end yield). Form as restated in the
    later literature (unverified against the paper; see module docstring).
    """
    u = jnp.asarray(u, dtype=float)
    return jnp.exp(-(u ** a0 - 0.5) ** 2 / a1) - jnp.exp(-0.25 / a1) + delta * (1.0 - u)


def species_density(theta: Array, alpha) -> Array:
    """``dN/dtheta`` (up to ``N0``) implied by ``k = k_max theta^(1/alpha)`` and the stated ``D(k)``.

    ``D(k) dk/dtheta = theta^((1/alpha)(1/alpha - 1)) theta^(1/alpha - 1) / alpha^2
    = theta^(1/alpha^2 - 1) / alpha^2``.
    """
    return jnp.asarray(theta, dtype=float) ** (1.0 / alpha**2 - 1.0) / alpha**2


def _gl(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(x), jnp.asarray(w)


def continuous_distribution(theta_i: Array, theta_lo: Array, theta_hi: Array, theta_gas: Array,
                            alpha, a0, a1, delta, n_quad: int = N_QUAD) -> Array:
    """The continuous-lumping distribution matrix ``W`` (see :class:`Prepared`).

    Args:
        theta_i: ``(n,)`` normalised boiling point of each cut (the parent's ``K``).
        theta_lo, theta_hi: ``(n,)`` normalised lower and upper edges of each cut.
        theta_gas: Upper edge of the gas bin (the lightest cut's lower edge).
    """
    x, w = _gl(n_quad)
    n = theta_i.shape[0]
    lo_bins = jnp.concatenate([jnp.zeros(1), theta_lo])          # (n+1,)
    hi_bins = jnp.concatenate([jnp.reshape(theta_gas, (1,)), theta_hi])

    def row(th_i, th_cap):
        a = jnp.minimum(lo_bins, th_cap)
        b = jnp.minimum(hi_bins, th_cap)
        half = 0.5 * (b - a)
        mid = 0.5 * (b + a)
        t = mid[:, None] + half[:, None] * x[None, :]            # (n+1, q)
        t = jnp.maximum(t, 1e-12)
        u = (t / th_i) ** (1.0 / alpha)
        g = yield_distribution(u, a0, a1, delta) * species_density(t, alpha)
        mass = half * jnp.sum(g * w[None, :], axis=1)
        mass = jnp.maximum(mass, 0.0)
        return mass / jnp.sum(mass)

    return jax.vmap(row)(theta_i, theta_lo)


def discrete_distribution(theta_i: Array, theta_lo: Array, theta_hi: Array, theta_gas: Array,
                          lump_edges_theta: Array, lump_selectivity: Array) -> Array:
    """The discrete-lump distribution matrix ``W``.

    A parent in lump ``L`` sends ``sel[L, 0]`` of its carbon to the gas bin
    and ``sel[L, 1 + M]`` to lump ``M``, spread over the part of lump ``M``
    lighter than the parent's cut in proportion to theta-width; the row is
    renormalised over what is reachable.
    """
    edges = jnp.concatenate([jnp.reshape(theta_gas, (1,)), lump_edges_theta, jnp.asarray([jnp.inf])])
    sel = jnp.asarray(lump_selectivity, dtype=float)
    lump_of = jnp.clip(jnp.searchsorted(lump_edges_theta, theta_i), 0,
                       sel.shape[0] - 1)

    def row(L, cap):
        # overlap of each cut bin with each lump, below the parent's lower edge
        a = jnp.minimum(theta_lo, cap)
        b = jnp.minimum(theta_hi, cap)
        lo_l, hi_l = edges[:-1], edges[1:]
        ov = jnp.maximum(jnp.minimum(b[:, None], hi_l[None, :]) - jnp.maximum(a[:, None], lo_l[None, :]), 0.0)
        tot = jnp.sum(ov, axis=0)                                # reachable width of each lump
        share = sel[L, 1:]
        share = jnp.where(tot > 0, share, 0.0)
        cut = ov @ (share / jnp.where(tot > 0, tot, 1.0))
        out = jnp.concatenate([sel[L, :1], cut])
        return out / jnp.sum(out)

    return jax.vmap(row)(lump_of, theta_lo)


def product_properties(Tb: Array, Kw: Array) -> dict:
    """Properties assigned to a cracked product boiling at ``Tb`` with Watson K ``Kw`` (see module docstring).

    Returns ``SG``, ``MW`` (Twu 1984), ``H`` (Goossens 1997 mass fraction),
    ``types`` ``(n, 4)`` (P, N, A, O volume fractions, Riazi-Daubert 1986 on
    the simplex, no olefins), ``n_C`` and ``h_per_C``.
    """
    Tb = jnp.asarray(Tb, dtype=float)
    SG = corr.sg_from_watson_k(Tb, Kw)
    MW, _, _ = corr.critical_properties(Tb, SG, "twu")
    n20 = estimate_refractive_index(Tb, SG)
    d20 = density_20c(SG)
    H = hydrogen_goossens(n20, d20, MW)
    CH = (1.0 - H) / H
    xp, xn, xa = riazi_daubert_pna(MW, SG, n20, CH, d20)
    t = 1e-3
    pos = t * jnp.logaddexp(jnp.stack([xp, xn, xa], axis=-1) / t, 0.0)
    pna = pos / jnp.sum(pos, axis=-1, keepdims=True)
    types = jnp.concatenate([pna, jnp.zeros_like(xp)[:, None]], axis=-1)
    n_C = MW * (1.0 - H) / ATOMIC_MASS["C"]
    h_per_C = (H / ATOMIC_MASS["H"]) / ((1.0 - H) / ATOMIC_MASS["C"])
    return {"SG": SG, "MW": MW, "H": H, "types": types, "n_C": n_C, "h_per_C": h_per_C}


@dataclass(frozen=True)
class HCKinetics:
    """The hydrocracking kinetic model (see the module docstring).

    Attributes:
        layout: The stream layout: attributes :data:`HCU_ATTRIBUTES`; gases
            hydrogen, hydrogen_sulfide, ammonia and at least one of
            :data:`HCU_GAS_SPLIT`'s.
        hdt: The :class:`HDTKinetics` run alongside (or ``None``).
        scheme: ``"continuous"`` or ``"discrete"`` (static; must match the params').
        aromatic_table: The mono/di/poly split applied to cracked-product aromatics.
    """

    layout: Layout
    hdt: HDTKinetics | None = None
    scheme: str = "continuous"
    aromatic_table: tuple = DEFAULT_AROMATIC_SPLIT
    attributes: tuple[str, ...] = HCU_ATTRIBUTES
    attribute_elements: tuple = HCU_ATTRIBUTE_ELEMENTS

    def __post_init__(self):
        if tuple(self.layout.attributes) != HCU_ATTRIBUTES:
            raise ValueError("the layout's attributes must be HCU_ATTRIBUTES")
        if self.scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        for g in ("hydrogen", "hydrogen_sulfide", "ammonia"):
            if g not in self.layout.gases:
                raise ValueError(f"the layout needs gas {g!r}")
        if not any(g in self.layout.gases for g in HCU_GAS_SPLIT):
            raise ValueError("the layout carries none of the cracked-gas species")

    # ----- per-solve tables ---------------------------------------------------

    def gas_shares(self) -> np.ndarray:
        """``(n_gas,)`` mole shares of the cracked gas over the layout's gases (normalised)."""
        lay = self.layout
        s = np.asarray([HCU_GAS_SPLIT.get(g, 0.0) for g in lay.gases])
        return s / s.sum()

    def prepare(self, p: HCKineticParams, Tb: Array, T_lo: Array, T_hi: Array, Kw_feed) -> HCState:
        """The per-solve tables from the cut boiling points and edges (K) and the feed's Watson K."""
        if p.scheme != self.scheme:
            raise ValueError(f"params.scheme {p.scheme!r} does not match the model's {self.scheme!r}")
        span = p.T_high - p.T_low
        th = lambda T: jnp.clip((jnp.asarray(T, dtype=float) - p.T_low) / span, 1e-9, None)
        th_i, th_lo, th_hi = th(Tb), th(T_lo), th(T_hi)
        th_gas = th_lo[0]
        if self.scheme == "continuous":
            kappa = th_i ** (1.0 / p.alpha)
            W = continuous_distribution(th_i, th_lo, th_hi, th_gas, p.alpha, p.a0, p.a1, p.delta)
        else:
            le = th(jnp.asarray(p.lump_edges, dtype=float))
            lump_of = jnp.searchsorted(le, th_i)
            kappa = jnp.asarray(p.lump_k, dtype=float)[jnp.clip(lump_of, 0, len(p.lump_k) - 1)]
            W = discrete_distribution(th_i, th_lo, th_hi, th_gas, le, jnp.asarray(p.lump_selectivity, dtype=float))
        pr = product_properties(Tb, Kw_feed + p.dKw)
        types = pr["types"]
        arom = types[:, HC_TYPES.index("aromatics")][:, None] * aromatic_split(Tb, self.aromatic_table)
        lay = self.layout
        shares = jnp.asarray(self.gas_shares())
        gC = float(sum(s * GAS_ELEMENTS[g].get("C", 0) for g, s in zip(lay.gases, self.gas_shares())))
        gH = float(sum(s * GAS_ELEMENTS[g].get("H", 0) for g, s in zip(lay.gases, self.gas_shares())))
        prep = Prepared(kappa=kappa, W=W, n_C=pr["n_C"], h_per_C=pr["h_per_C"], aromatics=arom,
                        naphthenes=types[:, HC_TYPES.index("naphthenes")], SG=pr["SG"], MW=pr["MW"],
                        gas_C=jnp.asarray(gC), gas_H=jnp.asarray(gH), gas_shares=shares)
        return HCState(p=p, prep=prep)

    # ----- rates --------------------------------------------------------------

    def cracking_rate(self, ctx: ReactionContext, st: HCState) -> Array:
        """``(n_cut,)`` molecules cracked per s per kg catalyst."""
        p, prep = st.p, st.prep
        T = ctx.T
        Tr = p.T_ref
        k = p.k_max * jnp.exp(-p.E / R_GAS * (1.0 / T - 1.0 / Tr))
        ai = {a: i for i, a in enumerate(HCU_ATTRIBUTES)}
        c_norg = jnp.sum(ctx.c_attr[:, ai["N_basic"]] + ctx.c_attr[:, ai["N_non_basic"]])
        K_N = p.K_N * jnp.exp(-p.dH_N / R_GAS * (1.0 / T - 1.0 / Tr))
        inhib = 1.0 + K_N * jnp.maximum(c_norg, 0.0) + p.K_nh3 * jnp.maximum(ctx.c_gas("ammonia"), 0.0)
        h2 = (jnp.maximum(ctx.c_gas("hydrogen"), 0.0) / p.c_ref) ** p.h2_order
        f = p.activity * p.effectiveness * p.wetting
        return f * k * prep.kappa * jnp.maximum(ctx.c_cut, 0.0) * h2 / inhib

    def rates(self, ctx: ReactionContext, st: HCState) -> Rates:
        lay = self.layout
        prep = st.prep
        ai = {a: i for i, a in enumerate(HCU_ATTRIBUTES)}
        gi = {g: lay.gas_index(g) for g in lay.gases}
        r = self.cracking_rate(ctx, st)                              # (nc,)
        pm = ctx.per_molecule
        s_cols = [ai[a] for a in HCU_ATTRIBUTES if a.startswith("S_")]
        n_cols = [ai[a] for a in HCU_ATTRIBUTES if a.startswith("N_")]
        C_par = pm[:, ai["C"]]
        H_par = pm[:, ai["H"]]
        S_par = jnp.sum(pm[:, s_cols], axis=1)
        N_par = jnp.sum(pm[:, n_cols], axis=1)
        rc = r * C_par                                               # carbon cracked, per cut
        C_to = rc @ prep.W                                           # (nc+1,) gas bin then cuts
        Cg, Cj = C_to[0], C_to[1:]
        mol_j = Cj / prep.n_C
        H_j = Cj * prep.h_per_C
        mol_g = Cg / prep.gas_C
        d_cut = -r + mol_j
        d_attr = -r[:, None] * pm
        d_attr = d_attr.at[:, ai["C"]].add(Cj)
        d_attr = d_attr.at[:, ai["H"]].add(H_j)
        for k, c in enumerate(AROMATIC_CLASSES):
            d_attr = d_attr.at[:, ai[f"A_{c}"]].add(mol_j * prep.aromatics[:, k])
        d_attr = d_attr.at[:, ai["naphthenes"]].add(mol_j * prep.naphthenes)
        d_attr = d_attr.at[:, ai["cracked"]].add(mol_j)
        d_gas = mol_g * prep.gas_shares
        d_gas = d_gas.at[gi["hydrogen_sulfide"]].add(jnp.sum(r * S_par))
        d_gas = d_gas.at[gi["ammonia"]].add(jnp.sum(r * N_par))
        H_prod = jnp.sum(H_j) + mol_g * prep.gas_H + 2.0 * jnp.sum(r * S_par) + 3.0 * jnp.sum(r * N_par)
        h2 = 0.5 * (H_prod - jnp.sum(r * H_par))                     # H2 consumed
        d_gas = d_gas.at[gi["hydrogen"]].add(-h2)
        scissions = jnp.sum(mol_j) + mol_g - jnp.sum(r)
        heat = -(scissions * SCISSION_HEAT + (h2 - scissions) * SATURATION_HEAT_PER_H2)
        out = Rates(gas=d_gas, cut=d_cut, attr=d_attr, heat=heat)
        if self.hdt is not None and st.p.hdt is not None:
            h = self.hdt.rates(ctx, st.p.hdt)
            out = Rates(gas=out.gas + h.gas, cut=out.cut + h.cut, attr=out.attr + h.attr, heat=out.heat + h.heat)
        return out


__all__ = ["HCU_ATTRIBUTES", "HCU_ATTRIBUTE_ELEMENTS", "HCU_GAS_SPLIT", "SCISSION_HEAT",
           "SATURATION_HEAT_PER_H2", "N_QUAD", "SCHEMES", "HCKineticParams", "CRACK_BED_HDT_PARAMS",
           "Prepared", "HCState", "HCKinetics", "yield_distribution", "species_density",
           "continuous_distribution", "discrete_distribution", "product_properties"]
