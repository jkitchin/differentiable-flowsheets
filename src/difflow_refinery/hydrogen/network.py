r"""The refinery hydrogen header: producers, consumers, an optional PSA, purge, import and export.

A header is a pool of gas at one pressure. Its supplies are

* **producers** with a fixed flow and composition (the reformer's net gas,
  a hydrogen plant at a fixed rate, a purge from another unit), any share of
  which can be routed through the header's PSA;
* **swing sources** (:class:`Import`, :class:`H2Plant`) that make up a
  deficit, in order, each up to its capacity.

Its sinks are the **consumers** (hydrotreater and hydrocracker makeup) and
the **purge**, to fuel gas or to export. Every consumer draws the header's
gas, so every consumer on a header receives makeup at the header's purity;
consumers that need different purities go on different headers.

Equations (all mol/s, per species on :data:`HEADER_GASES`)
-----------------------------------------------------------

With ``F_p`` producer ``p``'s gas and ``a_p`` its PSA share, the gas eligible
for the PSA is ``E = sum_p a_p F_p`` and the bypass ``B = sum_p (1 - a_p) F_p``.
The PSA processes ``s E`` (``s = 1``, or set by a purity target, below) with
H2 recovery ``R`` and product purity ``y_P``::

    product H2        = R s E_H2
    product impurity  = R s E_H2 (1 - y_P) / y_P, split like the feed's impurities
    tail gas          = s E - product               (to fuel gas)

The fixed supply is ``S = B + (1 - s) E + product``. A consumer's makeup
demand is an H2 flow ``d_j`` (what :class:`~difflow_refinery.hydrotreating.Hydrotreater`
reports as ``h2.makeup``), optionally a linear response to the purity it
receives, ``d_j(y) = d_j0 + (dd_j/dy) (y - y_ref,j)``. The swing sources
fill ``need = sum_j d_j + purge_min - S_H2`` in order, ``x_k = clip(need_k, 0,
cap_k)``; the header gas is ``G = S + sum_k X_k`` and its purity
``y = G_H2 / sum(G)``. Consumer ``j`` takes ``M_j = d_j G / G_H2`` and the
purge is what is left, ``G - sum_j M_j`` -- its H2 is the **surplus**. A
negative surplus is a deficit the swing sources could not cover: the
answer is still returned (it is what a planner writes a row on) and
:attr:`H2NetworkResult.feasible` says so.

When a consumer's demand responds to purity, ``y`` is the fixed point of
``y -> purity(d(y))``, iterated a fixed number of times (the gain is the
slope times ``dy/dd``, tiny in practice); the residual is reported.

PSA purity target: with :attr:`PSA.target_purity` set, ``s`` is the share
for which the header purity equals the target, which is linear in ``s``::

    s = (y* B_tot - A) / (E_H2 (R - 1) - y* (R E_H2 / y_P - E_tot))

(``A``, ``B_tot`` the header's H2 and total flow at ``s = 0``), clipped to
``[0, 1]``. With a swing source active its flow moves with ``s``, so ``s``
and the swing flows are alternated :data:`PSA_TARGET_PASSES` times;
``<h>.psa.target_error`` (header purity minus target) reports what was
reached, and is nonzero when the clip binds (the PSA cannot make the target).

Every flow is a JAX array, so the network is differentiable (forward and
reverse mode) in anything that sets a producer's gas, a demand, a PSA
parameter or a capacity. The clips (swing capacity, the PSA share) are
kinks: a gradient taken exactly at one is one-sided.

Partial pressure
----------------
A consumer's ``min_pH2`` is a spec on the makeup's H2 partial pressure at the
consumer's makeup pressure, ``y P``. It is not the reactor-inlet H2 partial
pressure, which also depends on the unit's recycle purity: that is the
hydrotreater's ``reactor.pH2_in`` output, which
:func:`~difflow_refinery.hydrogen.close_hydrotreater_loop` reports.

Not modelled: compression power, header pressure drop, multicomponent PSA
selectivity (impurities slip in the feed's proportions), H2S in the header
(the hydrotreaters' purge has been amine-scrubbed; a purge routed here is
mapped onto :data:`HEADER_GASES` and the rest is reported as dropped).

Sources
-------
The superstructure (sources to sinks through a header, a purifier with a
recovery and a product purity, purge to fuel) is the usual one of refinery
hydrogen-network analysis, e.g. Alves and Towler (2002), *Ind. Eng. Chem.
Res.* 41, 5759 (background only; nothing here reproduces a number from it,
unverified). The PSA defaults (88 % recovery, 99.9 mol% product) are
illustrative values inside the range usually quoted for refinery PSA units
(unverified against a source).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery.hydroprocessing.layout import GAS_ELEMENTS, gas_mw
from difflow_refinery.hydroprocessing.recycle import MOL_PER_NM3

jax.config.update("jax_enable_x64", True)

#: The species a header carries, lightest first (the hydroprocessing gas names).
HEADER_GASES: tuple[str, ...] = ("hydrogen", "methane", "ethane", "propane", "isobutane", "n_butane",
                                 "isopentane", "n_pentane", "n_hexane")
N_GAS = len(HEADER_GASES)
H2 = 0  # index of hydrogen
#: Molar masses (kg/mol) on :data:`HEADER_GASES`.
MW = np.asarray([gas_mw(g) for g in HEADER_GASES]) / 1000.0
#: Carbon number of each header gas (the folding order of :func:`fold_composition`).
CARBON = np.asarray([GAS_ELEMENTS[g].get("C", 0) for g in HEADER_GASES])

#: Reformer species -> header gas. Anything else (C6+) is lumped into ``n_hexane``
#: on a MOLE basis (H2 and the total moles are conserved; the mass of that trace is
#: the n-hexane molar mass's).
REFORMER_SPECIES: dict[str, str] = {
    "H2": "hydrogen", "C1": "methane", "C2": "ethane", "C3": "propane", "iC4": "isobutane",
    "nC4": "n_butane", "iC5": "isopentane", "nC5": "n_pentane"}

#: Alternations of PSA share and swing flows when a purity target meets an active swing.
PSA_TARGET_PASSES = 8

#: Nm^3/h per mol/s (0 C, 1 atm).
NM3_H_PER_MOL_S = 3600.0 / MOL_PER_NM3


def gas_vector(flows) -> Array:
    """A flow (or fraction) vector on :data:`HEADER_GASES` from ``{gas: value}`` or an array."""
    if isinstance(flows, Mapping):
        unknown = sorted(set(flows) - set(HEADER_GASES))
        if unknown:
            raise ValueError(f"{unknown} are not header gases {HEADER_GASES}")
        return jnp.stack([jnp.asarray(flows.get(g, 0.0), dtype=float) for g in HEADER_GASES])
    v = jnp.asarray(flows, dtype=float)
    if v.shape != (N_GAS,):
        raise ValueError(f"a header gas vector has shape ({N_GAS},), got {v.shape}")
    return v


def _impurity_vector(impurity) -> Array:
    """Normalised impurity mole fractions (no hydrogen) from a gas name, a dict or a vector."""
    if isinstance(impurity, str):
        impurity = {impurity: 1.0}
    v = gas_vector(impurity).at[H2].set(0.0)
    return v / jnp.sum(v)


def fold_composition(y, gases: Sequence[str]) -> dict:
    """Map a header composition onto ``gases`` (e.g. a hydrotreater layout's).

    A header gas the target does not carry is folded, on a mole basis, into
    the heaviest target gas with no more carbon atoms (methane at worst), so
    the H2 fraction -- the purity -- is unchanged. Hydrogen and methane must
    be in ``gases``.
    """
    gases = list(gases)
    for g in ("hydrogen", "methane"):
        if g not in gases:
            raise ValueError(f"{g} must be among the target gases")
    y = gas_vector(y)
    out = {g: jnp.asarray(0.0) for g in gases}
    present = [i for i, g in enumerate(HEADER_GASES) if g in gases]
    for i, g in enumerate(HEADER_GASES):
        if g in gases:
            tgt = g
        else:
            cand = [j for j in present if CARBON[j] <= CARBON[i] and j != H2]
            tgt = HEADER_GASES[max(cand)]
        out[tgt] = out[tgt] + y[i]
    return out


# ----- supplies ----------------------------------------------------------------


@dataclass
class Producer:
    """A fixed supply of gas to a header.

    Attributes:
        name: Name (prefixes its outputs).
        flows: Gas flows (mol/s) on :data:`HEADER_GASES` (a vector or ``{gas: mol/s}``).
        P: Delivery pressure (Pa); must be at least the header's.
        header: The header it feeds.
        to_psa: Share of this gas routed to the header's PSA (0 to 1).
    """

    name: str
    flows: Array
    P: float = 25e5
    header: str = "main"
    to_psa: float = 0.0

    def __post_init__(self):
        self.flows = gas_vector(self.flows)

    @property
    def h2(self) -> Array:
        return self.flows[H2]

    @property
    def total(self) -> Array:
        return jnp.sum(self.flows)

    @property
    def purity(self) -> Array:
        return self.flows[H2] / jnp.sum(self.flows)

    @classmethod
    def of_purity(cls, name: str, h2, purity, impurity="methane", **kw) -> "Producer":
        """``h2`` mol/s of hydrogen at mole fraction ``purity``; the balance is ``impurity``
        (a gas name, or ``{gas: share}`` normalised over the non-H2 part)."""
        h2 = jnp.asarray(h2, dtype=float)
        purity = jnp.asarray(purity, dtype=float)
        imp = _impurity_vector(impurity)
        F = imp * h2 * (1.0 - purity) / purity
        return cls(name, F.at[H2].set(h2), **kw)

    @classmethod
    def from_reformer(cls, result, name: str = "reformer", P=None, **kw) -> "Producer":
        """The net gas of a solved :class:`~difflow_refinery.reforming.CatalyticReformer`.

        Species map by :data:`REFORMER_SPECIES`; C6+ traces are lumped into
        ``n_hexane`` (mole basis). ``P`` defaults to the net gas's pressure
        (the separator's). Differentiable: a traced reformer result gives a
        traced producer.
        """
        from difflow_refinery.reforming import species as sp
        F_net = result.flows("net_gas")
        F = jnp.zeros(N_GAS)
        for k, s in enumerate(sp.NAMES):
            g = REFORMER_SPECIES.get(s, "n_hexane")
            F = F.at[HEADER_GASES.index(g)].add(F_net[k])
        if P is None:
            P = result.streams["net_gas"]["P"]
        return cls(name, F, P=P, **kw)

    @classmethod
    def from_flows(cls, name: str, flows: Mapping, **kw) -> "Producer":
        """From any ``{gas: mol/s}`` (e.g. a hydrotreater purge's gas): gases not on
        :data:`HEADER_GASES` are dropped -- :func:`dropped_flows` says how much."""
        return cls(name, {g: v for g, v in flows.items() if g in HEADER_GASES}, **kw)


@dataclass
class Import:
    """A swing supply: as much as the header needs, up to ``capacity`` mol/s of H2.

    Attributes:
        name: Name (prefixes its outputs).
        purity: H2 mole fraction.
        impurity: The non-H2 part (a gas name or ``{gas: share}``).
        capacity: Most H2 it supplies (mol/s); ``inf`` for an import pipeline.
        P: Delivery pressure (Pa).
    """

    name: str = "import"
    purity: float = 0.999
    impurity: object = "methane"
    capacity: float = float("inf")
    P: float = 25e5

    def composition(self) -> Array:
        y = jnp.asarray(self.purity, dtype=float)
        return (_impurity_vector(self.impurity) * (1.0 - y)).at[H2].set(y)


@dataclass
class H2Plant(Import):
    """A hydrogen plant (SMR + PSA) run as the swing: an :class:`Import` with a capacity.

    A plant at a FIXED rate is a :meth:`Producer.of_purity` instead. The
    defaults (99.9 mol%, impurity methane) are illustrative.
    """

    name: str = "h2_plant"
    capacity: float = 100.0


@dataclass
class PSA:
    """A pressure-swing adsorption unit on a header.

    Attributes:
        recovery: H2 recovery to product (fraction). Default 0.88 is
            illustrative (unverified against a source).
        purity: Product H2 mole fraction. Default 0.999 is illustrative.
        target_purity: If set, the share of the PSA-eligible gas processed
            is solved so that the HEADER purity is this (see the module
            docstring); else all of it is processed.
    """

    recovery: float = 0.88
    purity: float = 0.999
    target_purity: float | None = None


@dataclass
class Header:
    """A hydrogen header.

    Attributes:
        name: Name (prefixes its outputs).
        P: Header pressure (Pa). Every supply must deliver at least this.
        psa: Optional :class:`PSA` on the producers' ``to_psa`` shares.
        swing: Swing sources (:class:`Import`, :class:`H2Plant`), filled in order.
        min_purge: H2 (mol/s) that must leave as purge even at balance
            (the swing sources make it up).
        purge_to: ``"fuel"`` or ``"export"``.
    """

    name: str = "main"
    P: float = 20e5
    psa: PSA | None = None
    swing: tuple = ()
    min_purge: float = 0.0
    purge_to: str = "fuel"

    def __post_init__(self):
        if self.purge_to not in ("fuel", "export"):
            raise ValueError("purge_to is 'fuel' or 'export'")
        self.swing = tuple(self.swing)


# ----- consumers ---------------------------------------------------------------


@dataclass
class Consumer:
    """A makeup-hydrogen consumer (hydrotreater, hydrocracker).

    Attributes:
        name: Name (prefixes its outputs).
        h2_demand: Makeup H2 it takes (mol/s of H2) at ``purity_ref``.
        header: The header it draws from.
        P: Makeup delivery pressure (Pa), for the partial-pressure spec and report.
        min_purity: Required makeup H2 mole fraction, or None.
        min_pH2: Required makeup H2 partial pressure ``y P`` (Pa), or None.
        purity_ref: The purity at which ``h2_demand`` holds (None: demand
            does not respond to purity).
        d_demand_d_purity: ``dd/dy`` (mol/s per unit mole fraction) about
            ``purity_ref`` -- a linear response, e.g. a secant from two
            solves of the unit (:func:`close_hydrotreater_loop` builds it).
    """

    name: str
    h2_demand: float
    header: str = "main"
    P: float | None = None
    min_purity: float | None = None
    min_pH2: float | None = None
    purity_ref: float | None = None
    d_demand_d_purity: float = 0.0

    def demand_at(self, y) -> Array:
        d = jnp.asarray(self.h2_demand, dtype=float)
        if self.purity_ref is None:
            return d
        return d + jnp.asarray(self.d_demand_d_purity, dtype=float) * (y - self.purity_ref)

    @property
    def required_purity(self):
        """The binding purity spec (``min_purity`` or ``min_pH2 / P``), or None."""
        req = []
        if self.min_purity is not None:
            req.append(jnp.asarray(self.min_purity, dtype=float))
        if self.min_pH2 is not None:
            if self.P is None:
                raise ValueError(f"{self.name}: min_pH2 needs the makeup pressure P")
            req.append(jnp.asarray(self.min_pH2, dtype=float) / self.P)
        if not req:
            return None
        return req[0] if len(req) == 1 else jnp.maximum(req[0], req[1])

    @classmethod
    def from_hydrotreater(cls, name: str, result, params=None, **kw) -> "Consumer":
        """From a solved :class:`~difflow_refinery.hydrotreating.Hydrotreater`:
        demand = its ``h2.makeup``, ``purity_ref`` = its makeup's H2 fraction,
        ``P`` = its reactor pressure (from ``params``, default params if None)."""
        from difflow_refinery.hydrotreating import HydrotreaterParams
        p = params or HydrotreaterParams()
        tot = sum(float(v) for v in p.makeup.values())
        kw.setdefault("P", p.P)
        kw.setdefault("purity_ref", float(p.makeup.get("hydrogen", 0.0)) / tot)
        return cls(name, result.outputs["h2.makeup"], **kw)


# ----- the result --------------------------------------------------------------


@dataclass
class H2NetworkResult:
    """A balanced hydrogen network.

    Attributes:
        outputs: Flat ``{name: value}`` (see :data:`OUTPUT_UNITS` for the patterns):
            ``h2.surplus`` (mol/s, negative = deficit), ``h2.supply``,
            ``h2.demand``, ``fuel_gas.*``, ``export.*``, and per header
            ``<h>.purity``, ``<h>.h2_surplus``, ``<h>.psa.*``, ``<h>.<swing>.h2``,
            per consumer ``<c>.purity``, ``<c>.makeup_mol_s``, ``<c>.makeup_h2``,
            ``<c>.pH2``, ``<c>.purity_margin``, per producer ``<p>.h2``.
        streams: ``{name: flow vector on HEADER_GASES}``: each producer, each
            swing source, ``<h>.psa_product``, ``<h>.psa_tail``, ``<h>.gas``
            (the header pool), ``<c>.makeup``, ``<h>.purge``, ``fuel_gas``, ``export``.
        balances: Relative closures over the whole network: ``total`` (moles),
            ``hydrogen``, ``mass``; and ``makeup_h2`` (largest relative
            difference between a consumer's makeup H2 and its demand).
        headers: Names of the headers.
        consumers: Names of the consumers.
    """

    outputs: dict
    streams: dict
    balances: dict
    headers: tuple
    consumers: tuple
    _consumer_header: dict = field(default_factory=dict, repr=False)
    _required: dict = field(default_factory=dict, repr=False)
    _pressure: dict = field(default_factory=dict, repr=False)
    _purge_to: dict = field(default_factory=dict, repr=False)

    def makeup_composition(self, consumer: str, gases: Sequence[str] | None = None) -> dict:
        """Consumer ``consumer``'s makeup as mole fractions; with ``gases``, folded onto them
        (:func:`fold_composition`). Floats when concrete -- the shape
        :class:`~difflow_refinery.hydrotreating.HydrotreaterParams` takes as ``makeup``."""
        M = self.streams[f"{consumer}.makeup"]
        y = M / jnp.sum(M)
        d = fold_composition(y, gases) if gases is not None else dict(zip(HEADER_GASES, y))
        try:
            return {k: float(v) for k, v in d.items() if float(v) > 0.0}
        except jax.errors.ConcretizationTypeError:
            return d

    @property
    def feasible(self) -> dict:
        """Concrete checks: ``<h>.balanced`` (surplus >= 0), ``<h>.psa_purity`` (the PSA's
        product is purer than its feed), ``<h>.pressure`` (every supply at or above the
        header), ``<c>.purity`` (spec met). Empty entries for checks that do not apply."""
        o = self.outputs
        f = {}
        for h in self.headers:
            f[f"{h}.balanced"] = bool(o[f"{h}.h2_surplus"] >= -1e-9)
            if f"{h}.psa.feed_purity" in o:
                f[f"{h}.psa_purity"] = bool(o[f"{h}.psa.purity"] >= o[f"{h}.psa.feed_purity"])
            f[f"{h}.pressure"] = all(bool(P >= self._pressure[h]) for P in self._pressure[f"{h}.supplies"])
        for c in self.consumers:
            if f"{c}.purity_margin" in o:
                f[f"{c}.purity"] = bool(o[f"{c}.purity_margin"] >= -1e-12)
        return f

    def table(self) -> str:
        """A plain-text summary."""
        o = self.outputs
        lines = []
        for h in self.headers:
            lines.append(f"header {h}: purity {100 * float(o[f'{h}.purity']):.2f} mol%, "
                         f"H2 surplus {float(o[f'{h}.h2_surplus']):.2f} mol/s "
                         f"({float(o[f'{h}.h2_surplus']) * NM3_H_PER_MOL_S / 1e3:.2f} kNm3/h) to "
                         f"{self._purge_to[h]}")
        for c in self.consumers:
            s = (f"  {c}: makeup {float(o[f'{c}.makeup_mol_s']):.2f} mol/s, "
                 f"H2 {float(o[f'{c}.makeup_h2']):.2f} mol/s at {100 * float(o[f'{c}.purity']):.2f} mol%")
            if f"{c}.purity_margin" in o:
                s += f", margin {100 * float(o[f'{c}.purity_margin']):+.2f} mol%"
            lines.append(s)
        lines.append(f"fuel gas {float(o['fuel_gas.mol_s']):.2f} mol/s ({float(o['fuel_gas.kg_s']):.3f} kg/s, "
                     f"H2 {float(o['fuel_gas.h2_mol_s']):.2f} mol/s); export {float(o['export.mol_s']):.2f} mol/s")
        lines.append("closures: " + ", ".join(f"{k} {float(v):.0e}" for k, v in self.balances.items()))
        return "\n".join(lines)


def _rel(a, b):
    return jnp.abs(a - b) / jnp.maximum(jnp.maximum(jnp.abs(a), jnp.abs(b)), 1e-300)


# ----- the network -------------------------------------------------------------


@dataclass
class HydrogenNetwork:
    """Producers and consumers on one or more headers (see the module docstring).

    Attributes:
        producers: :class:`Producer` s.
        consumers: :class:`Consumer` s.
        headers: :class:`Header` s; default one ``Header("main")``.
        n_iter: Fixed-point iterations on the header purity (only matters when
            a consumer's demand responds to purity).
    """

    producers: Sequence[Producer]
    consumers: Sequence[Consumer]
    headers: Sequence[Header] = field(default_factory=lambda: (Header(),))
    n_iter: int = 30

    def __post_init__(self):
        self.producers = tuple(self.producers)
        self.consumers = tuple(self.consumers)
        self.headers = tuple(self.headers)
        names = [h.name for h in self.headers]
        if len(set(names)) != len(names):
            raise ValueError("header names must be unique")
        every = ([p.name for p in self.producers] + [c.name for c in self.consumers]
                 + [s.name for h in self.headers for s in h.swing])
        if len(set(every)) != len(every):
            raise ValueError("producer, consumer and swing-source names must be unique")
        bad = [x.name for x in self.producers + self.consumers if x.header not in names]
        if bad:
            raise ValueError(f"{bad} feed or draw from a header that is not defined ({names})")

    def header(self, name: str) -> Header:
        return {h.name: h for h in self.headers}[name]

    def replace(self, producers=None, consumers=None, headers=None) -> "HydrogenNetwork":
        """A copy with some parts swapped (others kept)."""
        return dataclasses.replace(self, producers=self.producers if producers is None else producers,
                                   consumers=self.consumers if consumers is None else consumers,
                                   headers=self.headers if headers is None else headers)

    def with_consumer(self, name: str, **changes) -> "HydrogenNetwork":
        """A copy with consumer ``name``'s fields changed."""
        cons = tuple(dataclasses.replace(c, **changes) if c.name == name else c for c in self.consumers)
        return self.replace(consumers=cons)

    def _solve_header(self, h: Header, out: dict, streams: dict):
        prods = [p for p in self.producers if p.header == h.name]
        cons = [c for c in self.consumers if c.header == h.name]
        z = jnp.zeros(N_GAS)
        E = sum((jnp.asarray(p.to_psa, dtype=float) * p.flows for p in prods), z)
        Bp = sum(((1.0 - jnp.asarray(p.to_psa, dtype=float)) * p.flows for p in prods), z)
        comps = [s.composition() for s in h.swing]
        psa = h.psa
        if psa is not None:
            R = jnp.asarray(psa.recovery, dtype=float)
            yP = jnp.asarray(psa.purity, dtype=float)
            E_imp = E.at[H2].set(0.0)
            imp_share = E_imp / jnp.maximum(jnp.sum(E_imp), 1e-300)

        def supplies(y):
            d = sum((c.demand_at(y) for c in cons), jnp.asarray(0.0))
            # swing flows are set against the fixed supply at s, and s against the swing
            # flows (purity target): one pass of each per purity iteration.
            def fixed(s):
                if psa is None:
                    return Bp + E, z, z
                prod_h2 = R * s * E[H2]
                prod = (imp_share * prod_h2 * (1.0 - yP) / yP).at[H2].set(prod_h2)
                return Bp + (1.0 - s) * E + prod, prod, s * E - prod

            def swing(S):
                need = d + jnp.asarray(h.min_purge, dtype=float) - S[H2]
                xs = []
                for s_, cmp in zip(h.swing, comps):
                    x = jnp.clip(need, 0.0, jnp.asarray(s_.capacity, dtype=float))
                    xs.append(x)
                    need = need - x
                X = sum((x * cmp / cmp[H2] for x, cmp in zip(xs, comps)), z)
                return xs, X

            s = jnp.asarray(1.0)
            s_raw = s
            if psa is not None and psa.target_purity is not None:
                # s is linear in the target at fixed swing flows; the swing flows move with s,
                # so alternate (exact in one pass when the swing sources are idle).
                ys = jnp.asarray(psa.target_purity, dtype=float)
                den = E[H2] * (R - 1.0) - ys * (R * E[H2] / yP - jnp.sum(E))
                den = jnp.where(jnp.abs(den) > 1e-300, den, 1e-300)
                s = jnp.asarray(0.0)
                for _ in range(PSA_TARGET_PASSES if h.swing else 1):
                    S0, _, _ = fixed(jnp.asarray(0.0))
                    _, Xs = swing(fixed(s)[0])
                    A = S0[H2] + Xs[H2]
                    Btot = jnp.sum(S0) + jnp.sum(Xs)
                    s_raw = (ys * Btot - A) / den
                    s = jnp.clip(s_raw, 0.0, 1.0)
            S, prod, tail = fixed(s)
            xs, X = swing(S)
            return d, s, s_raw, S, prod, tail, xs, X

        def purity(y):
            _, _, _, S, _, _, _, X = supplies(y)
            G = S + X
            return G[H2] / jnp.sum(G)

        y0 = purity(jnp.asarray(1.0))
        if any(c.purity_ref is not None and not (isinstance(c.d_demand_d_purity, (int, float))
                                                and c.d_demand_d_purity == 0.0) for c in cons):
            y = jax.lax.fori_loop(0, self.n_iter, lambda i, y: purity(y), y0)
        else:
            y = y0
        resid = jnp.abs(purity(y) - y)
        d, s, s_raw, S, prod, tail, xs, X = supplies(y)
        G = S + X
        yG = G / jnp.sum(G)
        hn = h.name
        takes = sum((c.demand_at(y) for c in cons), jnp.asarray(0.0))
        purge = G - takes * yG / yG[H2]
        streams[f"{hn}.gas"] = G
        streams[f"{hn}.purge"] = purge
        streams[f"{hn}.psa_product"] = prod
        streams[f"{hn}.psa_tail"] = tail
        for sw, x, cmp in zip(h.swing, xs, comps):
            streams[sw.name] = x * cmp / cmp[H2]
            out[f"{sw.name}.h2"] = x
            out[f"{sw.name}.mol_s"] = jnp.sum(streams[sw.name])
        for c in cons:
            dj = c.demand_at(y)
            M = dj * yG / yG[H2]
            streams[f"{c.name}.makeup"] = M
            out[f"{c.name}.makeup_h2"] = M[H2]
            out[f"{c.name}.makeup_mol_s"] = jnp.sum(M)
            out[f"{c.name}.makeup_nm3_h"] = M[H2] * NM3_H_PER_MOL_S
            out[f"{c.name}.purity"] = y
            if c.P is not None:
                out[f"{c.name}.pH2"] = y * c.P
            req = c.required_purity
            if req is not None:
                out[f"{c.name}.purity_margin"] = y - req
        out[f"{hn}.purity"] = y
        out[f"{hn}.loop_residual"] = resid
        out[f"{hn}.h2_supply"] = S[H2] + X[H2]
        out[f"{hn}.h2_demand"] = takes
        out[f"{hn}.h2_surplus"] = purge[H2]
        out[f"{hn}.purge_mol_s"] = jnp.sum(purge)
        if psa is not None:
            out[f"{hn}.psa.share"] = s
            out[f"{hn}.psa.share_unclipped"] = s_raw
            if psa.target_purity is not None:
                out[f"{hn}.psa.target_error"] = y - jnp.asarray(psa.target_purity, dtype=float)
            out[f"{hn}.psa.h2_product"] = prod[H2]
            out[f"{hn}.psa.tail_gas_mol_s"] = jnp.sum(tail)
            out[f"{hn}.psa.tail_gas_h2"] = tail[H2]
            out[f"{hn}.psa.feed_purity"] = E[H2] / jnp.maximum(jnp.sum(E), 1e-300)
            out[f"{hn}.psa.purity"] = jnp.asarray(psa.purity, dtype=float)
        return purge, tail

    def solve(self) -> H2NetworkResult:
        """Balance every header (pure JAX; trace it with ``jax.jit``/``grad``/``jacfwd``)."""
        out, streams = {}, {}
        z = jnp.zeros(N_GAS)
        fuel, export = z, z
        for p in self.producers:
            streams[p.name] = p.flows
            out[f"{p.name}.h2"] = p.flows[H2]
            out[f"{p.name}.purity"] = p.purity
        pressure = {}
        for h in self.headers:
            purge, tail = self._solve_header(h, out, streams)
            fuel = fuel + tail
            if h.purge_to == "fuel":
                fuel = fuel + purge
            else:
                export = export + purge
            pressure[h.name] = h.P
            pressure[f"{h.name}.supplies"] = ([p.P for p in self.producers if p.header == h.name]
                                              + [s.P for s in h.swing])
        streams["fuel_gas"] = fuel
        streams["export"] = export
        hs = [h.name for h in self.headers]
        out["h2.supply"] = sum(out[f"{h}.h2_supply"] for h in hs)
        out["h2.demand"] = sum(out[f"{h}.h2_demand"] for h in hs)
        out["h2.surplus"] = sum(out[f"{h}.h2_surplus"] for h in hs)
        out["h2.surplus_nm3_h"] = out["h2.surplus"] * NM3_H_PER_MOL_S
        out["fuel_gas.mol_s"] = jnp.sum(fuel)
        out["fuel_gas.kg_s"] = fuel @ MW
        out["fuel_gas.h2_mol_s"] = fuel[H2]
        out["export.mol_s"] = jnp.sum(export)
        out["export.h2_mol_s"] = export[H2]
        out["export.kg_s"] = export @ MW
        swing_total = sum((streams[s.name] for h in self.headers for s in h.swing), z)
        out["swing.h2"] = swing_total[H2]
        # balances over the network: in = producers + swing, out = makeups + fuel + export
        F_in = sum((p.flows for p in self.producers), z) + swing_total
        F_out = sum((streams[f"{c.name}.makeup"] for c in self.consumers), z) + fuel + export
        bal = {"total": _rel(jnp.sum(F_in), jnp.sum(F_out)), "hydrogen": _rel(F_in[H2], F_out[H2]),
               "mass": _rel(F_in @ MW, F_out @ MW)}
        y = {}
        for c in self.consumers:
            yy = out[f"{c.name}.purity"]
            y[c.name] = _rel(out[f"{c.name}.makeup_h2"], c.demand_at(yy))
        bal["makeup_h2"] = jnp.max(jnp.stack(list(y.values()))) if y else jnp.asarray(0.0)
        return H2NetworkResult(outputs=out, streams=streams, balances=bal, headers=tuple(hs),
                               consumers=tuple(c.name for c in self.consumers),
                               _consumer_header={c.name: c.header for c in self.consumers},
                               _required={c.name: c.required_purity for c in self.consumers},
                               _pressure=pressure, _purge_to={h.name: h.purge_to for h in self.headers})

    __call__ = solve


#: Units of the outputs, by suffix (``<name>.<suffix>``).
OUTPUT_UNITS: dict[str, str] = {
    "h2": "mol/s", "mol_s": "mol/s", "kg_s": "kg/s", "h2_mol_s": "mol/s", "purity": "mol frac",
    "makeup_h2": "mol/s", "makeup_mol_s": "mol/s", "makeup_nm3_h": "Nm3/h", "pH2": "Pa",
    "purity_margin": "mol frac", "h2_supply": "mol/s", "h2_demand": "mol/s", "h2_surplus": "mol/s",
    "purge_mol_s": "mol/s", "surplus": "mol/s", "surplus_nm3_h": "Nm3/h", "supply": "mol/s",
    "demand": "mol/s", "share": "-", "h2_product": "mol/s", "tail_gas_mol_s": "mol/s",
    "tail_gas_h2": "mol/s", "feed_purity": "mol frac", "loop_residual": "mol frac",
}


def output_units(key: str) -> str:
    """The unit of an output key (by its last dotted part)."""
    return OUTPUT_UNITS.get(key.rsplit(".", 1)[-1], "-")


def dropped_flows(flows: Mapping) -> dict:
    """The part of ``{gas: mol/s}`` that :meth:`Producer.from_flows` drops (non-header gases)."""
    return {g: v for g, v in flows.items() if g not in HEADER_GASES}


__all__ = ["HEADER_GASES", "REFORMER_SPECIES", "NM3_H_PER_MOL_S", "Producer", "Import", "H2Plant", "PSA",
           "Header", "Consumer", "HydrogenNetwork", "H2NetworkResult", "gas_vector", "fold_composition",
           "dropped_flows", "OUTPUT_UNITS", "output_units"]
