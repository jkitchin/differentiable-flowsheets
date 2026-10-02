"""The alkylation reactor: stoichiometry closed by the Sauer-Colville-Burwick yield.

A well-mixed liquid-phase reactor in which every olefin reacts with
isobutane by one of two routes:

* **Alkylation** -- ``C_nH_2n + iC4H10 -> C_(n+4)H_(2n+10)``: propylene to
  the dimethylpentanes, butenes to the trimethylpentanes and
  2,5-dimethylhexane, amylenes to 2,2,5-trimethylhexane, split by a
  per-olefin selectivity table.
* **Heavy ends** -- ``(8/n) C_nH_2n + iC4H10 -> C12H26``: olefin dimerised
  before it alkylates (polymerisation), the C12 lump carried as n-dodecane's
  properties (:data:`~difflow_refinery.alkylation.species.HEAVY_END`).

Both routes conserve carbon and hydrogen exactly, so mass is conserved by
construction whatever the split. The split is what the correlation sets. The
Sauer-Colville-Burwick yield ``Y(r)`` (vol alkylate per vol olefin,
:func:`~difflow_refinery.alkylation.correlations.alkylate_yield`) rises with
the isobutane/olefin ratio ``r`` to a maximum ``Y_max`` at ``r = 9.87``. The
model reads ``eps = Y(r) / Y_max`` as an alkylation efficiency and, for each
olefin ``j``, sends the fraction ``h_j`` to heavy ends that makes that
olefin's volume yield ``eps`` times its all-alkylation yield::

    (1 - h_j) Y_A,j + h_j Y_H,j = eps Y_A,j
    =>  h_j = Y_A,j (1 - eps) / (Y_A,j - Y_H,j)

with ``Y_A,j``/``Y_H,j`` the stoichiometric volume yields of the two routes
at 60 F (COSTALD volumes). For the butenes ``Y_A`` is 1.73-1.81
(trans-2-butene 1.76), within 2.5 % of ``Y_max`` = 1.770, so on a butene
feed the reactor reproduces the correlation's yield to about 1 %; propylene
(``Y_A`` 1.80) and the amylenes (1.67-1.72) follow its *shape* about their
own stoichiometric yields. This mapping is a modelling
choice of this module, not part of the published correlation.

The alkylate's motor octane is the correlation's
(:func:`~difflow_refinery.alkylation.correlations.motor_octane`) in ``r``
and the acid strength, plus two linear corrections that are NOT from Sauer et
al. -- the published correlation has neither variable -- and whose default
coefficients are illustrative assumptions of this module:

* temperature: ``mon_per_K * (T - T_ref)`` (default -0.1 MON/K, about -0.55
  per 10 F; colder runs make better alkylate);
* olefin space velocity: ``mon_per_sv * (SV - sv_ref)`` (default -2 MON per
  v/h/v).

Set both to zero to recover the published correlation exactly; refit all of
them to unit data before planning with them. Acid consumption (H2SO4)
follows from the octane through the F-4 performance number and the dilution
factor (Sauer et al., as in ``process.gms``). HF has no open correlation: its
consumption (acid make-up / ASO make, lb/bbl) must be supplied, and the
H2SO4 yield and octane correlations are used for it as they stand.

Heat. The heat of alkylation is computed, not correlated: Hess's law on the
liquid heats of formation at 298.15 K (``Hf(g) - Hvap(298 K)``; see
:mod:`~difflow_refinery.alkylation.species`), the temperature dependence of
the heat of reaction between 298 K and the reactor neglected. The
refrigeration duty is that heat plus the sensible heat of cooling the reactor
feed to the reactor temperature (Peng-Robinson liquid enthalpy); with
effluent/auto-refrigeration it is removed by vaporising isobutane, reported as
the refrigerant isobutane rate (Watson's latent heat from the CRC value at
Tb).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Mapping

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows, make_stream

from difflow_refinery.alkylation import correlations as corr
from difflow_refinery.alkylation.species import (
    ALKYLATE, ALKYLATION_SPECIES, BARREL, HEAVY_END, OLEFINS, property_array, species,
)

_DAY = 86400.0
#: lb per kg.
LB_PER_KG = 1.0 / 0.45359237


class AlkylationRangeWarning(UserWarning):
    """An operating variable is outside the range the correlation was fitted on.

    ``process.gms`` bounds the regression variables -- I/O ratio 3 to 12,
    acid strength 85 to 93 wt%, motor octane 90 to 95 -- to where Sauer et
    al.'s plant operated. Outside them the quadratics are extrapolations (the
    yield, for one, falls again above ``r = 9.87``).
    """


#: Default alkylation selectivities: mole fraction of each olefin's
#: alkylate going to each product. ILLUSTRATIVE assumptions of this module,
#: chosen to reflect the qualitative picture (isobutylene and 2-butenes
#: alkylate mostly to trimethylpentanes, 1-butene gives more
#: dimethylhexane, propylene gives dimethylpentanes) -- not taken from a
#: source. Every product must have carbon number n_olefin + 4.
DEFAULT_SELECTIVITY: dict[str, dict[str, float]] = {
    "propylene": {"2_3_dimethylpentane": 0.6, "2_4_dimethylpentane": 0.4},
    "1_butene": {"2_2_4_trimethylpentane": 0.30, "2_3_4_trimethylpentane": 0.35,
                 "2_5_dimethylhexane": 0.35},
    "cis_2_butene": {"2_2_4_trimethylpentane": 0.45, "2_3_4_trimethylpentane": 0.40,
                     "2_5_dimethylhexane": 0.15},
    "trans_2_butene": {"2_2_4_trimethylpentane": 0.45, "2_3_4_trimethylpentane": 0.40,
                       "2_5_dimethylhexane": 0.15},
    "isobutylene": {"2_2_4_trimethylpentane": 0.65, "2_3_4_trimethylpentane": 0.20,
                    "2_5_dimethylhexane": 0.15},
    "1_pentene": {"2_2_5_trimethylhexane": 1.0},
    "2_methyl_2_butene": {"2_2_5_trimethylhexane": 1.0},
}


@dataclass(repr=False)
class AlkylationReactorParams(ParamsMixin):
    """Parameters of the alkylation reactor.

    Attributes:
        T: Reactor temperature (K).
        P: Reactor pressure (Pa); liquid phase.
        acid: ``"H2SO4"`` or ``"HF"``.
        acid_strength: Titratable acid strength (wt%) -- the ``strength`` of
            the correlation.
        space_velocity: Olefin space velocity (v olefin / h / v acid).
        correlation: The Sauer-Colville-Burwick coefficients.
        T_ref: Temperature (K) at which the temperature correction vanishes.
        mon_per_K: Motor octane change per K above ``T_ref`` (illustrative).
        sv_ref: Space velocity at which its correction vanishes (1/h).
        mon_per_sv: Motor octane change per unit space velocity above
            ``sv_ref`` (illustrative).
        ron_minus_mon: Alkylate sensitivity RON - MON. Default 2.5, the
            sensitivity of the illustrative alkylate in
            :class:`~difflow_refinery.blending.BlendComponent`'s example
            (RON 96, MON 93.5); unverified.
        hf_acid_lb_per_bbl: HF consumption (lb per bbl alkylate). Required
            when ``acid == "HF"``; there is no open correlation for it.
        olefin_conversion: Fraction of each olefin converted (1: complete).
        selectivity: ``{olefin: {product: mole fraction}}`` of the
            alkylation route (see :data:`DEFAULT_SELECTIVITY`).
    """

    T: float = 283.15
    P: float = 6.0e5
    acid: str = "H2SO4"
    acid_strength: float = 89.0
    space_velocity: float = 0.3
    correlation: corr.SauerCorrelation = field(default_factory=corr.SauerCorrelation)
    T_ref: float = 283.15
    mon_per_K: float = -0.1
    sv_ref: float = 0.3
    mon_per_sv: float = -2.0
    ron_minus_mon: float = 2.5
    hf_acid_lb_per_bbl: float | None = None
    olefin_conversion: float = 1.0
    selectivity: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_SELECTIVITY.items()})


@lru_cache(maxsize=None)
def alkylation_thermo():
    """Peng-Robinson thermo for the alkylation species (kij = 0).

    A :class:`~difflow.thermo.CubicThermo` -- ideal-gas enthalpy from the
    database Cp plus the PR departure -- over :data:`ALKYLATION_SPECIES`.
    Used for the fractionation K-values and for the reactor's sensible
    heat. No binary interaction parameters: for C3-C5 paraffin/olefin pairs
    the published PR kij are small (|kij| < ~0.01), and none are tabulated in
    difflow.
    """
    from difflow.database import get_critical_props, get_species_data
    from difflow.eos import PengRobinson
    from difflow.thermo import CubicThermo, IdealThermo

    names = list(ALKYLATION_SPECIES)
    ideal = IdealThermo({s: get_species_data(s) for s in names})
    eos = PengRobinson({s: get_critical_props(s) for s in names})
    return CubicThermo(ideal, eos)


def flows_array(stream: Stream) -> Array:
    """The stream's molar flows (mol/s) in :data:`ALKYLATION_SPECIES` order.

    A species the stream does not carry counts as zero; a species it carries
    that the alkylation unit does not know is an error.
    """
    f = get_flows(stream)
    unknown = sorted(set(f) - set(ALKYLATION_SPECIES))
    if unknown:
        raise KeyError(f"species {unknown} are not alkylation species; known: "
                       f"{list(ALKYLATION_SPECIES)}")
    return jnp.stack([jnp.asarray(f.get(s, 0.0), dtype=jnp.float64) for s in ALKYLATION_SPECIES])


def stream_from_array(F: Array, T, P) -> Stream:
    """A stream carrying every alkylation species, from a flow array."""
    return make_stream({s: F[i] for i, s in enumerate(ALKYLATION_SPECIES)}, T, P)


def feed_stream(flows: Mapping[str, float], T: float, P: float) -> Stream:
    """A stream over every alkylation species, zero where ``flows`` is silent.

    This is the shape an FCC (#308) LPG stream plugs in as: a plain difflow
    stream with real-species flows (mol/s).
    """
    unknown = sorted(set(flows) - set(ALKYLATION_SPECIES))
    if unknown:
        raise KeyError(f"species {unknown} are not alkylation species")
    return make_stream({s: float(flows.get(s, 0.0)) for s in ALKYLATION_SPECIES}, T, P)


_IDX = {s: i for i, s in enumerate(ALKYLATION_SPECIES)}
_V60 = property_array(ALKYLATION_SPECIES, "v60")       # m^3/mol
_MW = property_array(ALKYLATION_SPECIES, "MW")         # g/mol
_HF_LIQ = np.array([species(s).Hf_liquid for s in ALKYLATION_SPECIES])  # J/mol
_OLEFIN_MASK = np.array([s in OLEFINS for s in ALKYLATION_SPECIES], dtype=float)
_ALKYLATE_MASK = np.array([s in ALKYLATE for s in ALKYLATION_SPECIES], dtype=float)


def standard_volume(F: Array) -> Array:
    """Liquid volume flow at 60 F (m^3/s) of a flow array (ideal mixing)."""
    return jnp.sum(F * _V60)


def to_bpd(volume_m3_s: Array) -> Array:
    """m^3/s at 60 F -> standard barrels per day."""
    return volume_m3_s * _DAY / BARREL


def isobutane_latent_heat(T: Array) -> Array:
    """Latent heat of isobutane (J/mol): Watson's relation from the CRC Tb value.

    ``dHvap(T) = dHvap(Tb) ((1 - T/Tc) / (1 - Tb/Tc))^0.38`` -- Watson,
    Ind. Eng. Chem. 35, 398 (1943); dHvap(Tb) = 21300 J/mol at Tb =
    261.42 K (CRC), Tc = 407.8 K (database).
    """
    s = species("isobutane")
    return 21300.0 * ((1.0 - T / s.Tc) / (1.0 - s.Tb / s.Tc)) ** 0.38


def _route_tables(selectivity: Mapping[str, Mapping[str, float]]):
    """Per-olefin stoichiometric columns of the two routes (numpy, static).

    Returns ``(nu_A, nu_H, Y_A, Y_H)``: for each olefin ``j`` (in
    :data:`OLEFINS` order) the change in every species per mole of olefin by
    route A (alkylation) and route H (heavy ends), and the two routes'
    standard-volume yields per volume of olefin.
    """
    n = len(ALKYLATION_SPECIES)
    iC4 = _IDX["isobutane"]
    nu_A, nu_H, Y_A, Y_H = [], [], [], []
    for olefin, n_c in OLEFINS.items():
        sel = selectivity.get(olefin)
        if sel is None:
            raise ValueError(f"no selectivity for olefin {olefin!r}")
        total = sum(sel.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"selectivity of {olefin!r} sums to {total}, not 1")
        a = np.zeros(n)
        a[_IDX[olefin]] -= 1.0
        a[iC4] -= 1.0
        for prod, s in sel.items():
            sp = species(prod)
            if prod not in ALKYLATE or sp.n_C != n_c + 4 or sp.n_H != 2 * n_c + 10:
                raise ValueError(f"{prod!r} is not an alkylation product of {olefin!r} "
                                 f"(C{n_c}= + iC4 gives C{n_c + 4}H{2 * n_c + 10})")
            a[_IDX[prod]] += s
        h = np.zeros(n)
        h[_IDX[olefin]] -= 1.0
        h[iC4] -= n_c / 8.0
        h[_IDX[HEAVY_END]] += n_c / 8.0
        v_ol = _V60[_IDX[olefin]]
        nu_A.append(a)
        nu_H.append(h)
        Y_A.append(float(np.sum(np.where(a > 0, a, 0.0) * _V60)) / v_ol)
        Y_H.append(n_c / 8.0 * _V60[_IDX[HEAVY_END]] / v_ol)
    return np.array(nu_A), np.array(nu_H), np.array(Y_A), np.array(Y_H)


def _concrete(x) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


class AlkylationReactor:
    """Isobutane alkylation of C3-C5 olefins over H2SO4 or HF.

    ``reactor(inlet) -> (effluent, info)``. The inlet is a difflow stream of
    real species (:data:`~difflow_refinery.alkylation.species.ALKYLATION_SPECIES`;
    absent ones are zero); the external isobutane/olefin ratio is read off
    it, so the reactor sees whatever the fresh feed, the makeup and the
    recycle put together. The effluent leaves as a liquid at the reactor
    temperature and pressure; the acid phase is not carried (the settler is
    ideal) and the acid consumed is reported in ``info`` as an operating cost.

    ``info`` (all JAX scalars):

    ========================  ===================================================
    ``io_ratio``              external isobutane/olefin ratio, vol/vol at 60 F
    ``yield_correlation``     Sauer et al.'s yield ``Y(r)``, vol/vol olefin
    ``efficiency``            ``Y(r) / Y_max``
    ``alkylate_yield``        the reactor's own yield, vol alkylate / vol olefin
    ``heavy_fraction``        olefin-volume-weighted share sent to heavy ends
    ``MON``, ``RON``          alkylate motor and research octane
    ``F4``, ``dilution``      F-4 performance number, acid dilution factor
    ``acid_lb_per_bbl``       acid consumed per bbl of alkylate
    ``acid_klb_d``            acid consumed, 1000 lb/d (``process.gms`` units)
    ``olefin_bpd``            olefin reacted, bbl/d
    ``alkylate_bpd``          alkylate made (C7+ products), bbl/d
    ``isobutane_consumed``    mol/s; ``isobutane_consumed_bpd`` in bbl/d
    ``heat_of_reaction``      heat released, W
    ``sensible_duty``         heat removed cooling the feed to ``T``, W
    ``refrigeration_duty``    the two together, W
    ``refrigerant_isobutane`` isobutane vaporised to remove it, mol/s
    ========================  ===================================================
    """

    outlet_names = ("effluent", "reactor_info")

    def __init__(self, params: AlkylationReactorParams | None = None):
        self.params = params if params is not None else AlkylationReactorParams()
        p = self.params
        if p.acid not in ("H2SO4", "HF"):
            raise ValueError(f"acid must be 'H2SO4' or 'HF', got {p.acid!r}")
        if p.acid == "HF" and p.hf_acid_lb_per_bbl is None:
            raise ValueError(
                "acid='HF' needs hf_acid_lb_per_bbl: HF consumption (acid "
                "regeneration losses, ASO make) has no open correlation and must "
                "be supplied from the unit's data")
        self._nu_A, self._nu_H, self._Y_A, self._Y_H = _route_tables(p.selectivity)
        self._dH_A = self._nu_A @ _HF_LIQ    # J per mol olefin, route A
        self._dH_H = self._nu_H @ _HF_LIQ    # J per mol olefin, route H
        self._olefin_idx = np.array([_IDX[o] for o in OLEFINS])

    def __call__(self, inlet: Stream):
        """React the inlet; returns ``(effluent, info)``."""
        p = self.params
        c = p.correlation
        F = flows_array(inlet)
        F_ol = F[self._olefin_idx]
        V_ol = jnp.sum(F_ol * _V60[self._olefin_idx])
        V_ic4 = F[_IDX["isobutane"]] * _V60[_IDX["isobutane"]]
        r = V_ic4 / V_ol
        Y = corr.alkylate_yield(r, c)
        eps = Y / corr.max_alkylate_yield(c)
        h = jnp.clip(self._Y_A * (1.0 - eps) / (self._Y_A - self._Y_H), 0.0, 1.0)
        x = F_ol * p.olefin_conversion
        xi_A, xi_H = x * (1.0 - h), x * h
        F_out = F + xi_A @ self._nu_A + xi_H @ self._nu_H
        # The olefins left are set exactly: summed through the routes they come
        # out at +/- round-off, and a -1e-17 flow trips the columns' checks.
        F_out = F_out.at[self._olefin_idx].set(F_ol * (1.0 - p.olefin_conversion))

        V_reacted = jnp.sum(x * _V60[self._olefin_idx])
        V_alk = jnp.sum((F_out - F) * _ALKYLATE_MASK * _V60)
        heavy = jnp.sum(x * h * _V60[self._olefin_idx]) / V_reacted

        T, S = jnp.asarray(p.T, dtype=jnp.float64), jnp.asarray(p.acid_strength, dtype=jnp.float64)
        mon = (corr.motor_octane(r, S, c) + p.mon_per_K * (T - p.T_ref)
               + p.mon_per_sv * (p.space_velocity - p.sv_ref))
        f4 = corr.f4_performance(mon, c)
        dil = corr.acid_dilution(f4, c)
        alk_bpd = to_bpd(V_alk)
        if p.acid == "H2SO4":
            acid_per_bbl = corr.acid_per_alkylate(mon, S, c)
        else:
            acid_per_bbl = jnp.asarray(p.hf_acid_lb_per_bbl, dtype=jnp.float64)

        q_rxn = -(jnp.sum(xi_A * self._dH_A) + jnp.sum(xi_H * self._dH_H))
        thermo = alkylation_thermo()
        flows_in = {s: F[i] for i, s in enumerate(ALKYLATION_SPECIES)}
        q_sens = (thermo.stream_enthalpy(flows_in, inlet["T"], "liquid", p.P)
                  - thermo.stream_enthalpy(flows_in, T, "liquid", p.P))
        q_ref = q_rxn + q_sens
        info = {
            "io_ratio": r,
            "yield_correlation": Y,
            "efficiency": eps,
            "alkylate_yield": V_alk / V_reacted,
            "heavy_fraction": heavy,
            "MON": mon,
            "RON": mon + p.ron_minus_mon,
            "F4": f4,
            "dilution": dil,
            "acid_lb_per_bbl": acid_per_bbl,
            "acid_klb_d": acid_per_bbl * alk_bpd / 1000.0,
            "olefin_bpd": to_bpd(V_reacted),
            "alkylate_bpd": alk_bpd,
            "isobutane_consumed": F[_IDX["isobutane"]] - F_out[_IDX["isobutane"]],
            "isobutane_consumed_bpd": to_bpd((F[_IDX["isobutane"]] - F_out[_IDX["isobutane"]])
                                             * _V60[_IDX["isobutane"]]),
            "heat_of_reaction": q_rxn,
            "sensible_duty": q_sens,
            "refrigeration_duty": q_ref,
            "refrigerant_isobutane": q_ref / isobutane_latent_heat(T),
        }
        self._check_ranges(r, S, mon, F_out[_IDX["isobutane"]])
        return stream_from_array(F_out, T, p.P), info

    @staticmethod
    def _check_ranges(r, S, mon, ic4_out):
        r, S, mon, ic4 = (_concrete(v) for v in (r, S, mon, ic4_out))
        if r is None:
            return
        for name, v, (lo, hi) in (("I/O ratio", r, corr.PROCESS_BOUNDS["ratio"]),
                                  ("acid strength", S, corr.PROCESS_BOUNDS["strength"]),
                                  ("motor octane", mon, corr.PROCESS_BOUNDS["octane"])):
            if not lo <= v <= hi:
                warnings.warn(f"{name} {v:.4g} is outside the correlation's range "
                              f"[{lo:g}, {hi:g}]", AlkylationRangeWarning, stacklevel=3)
        if ic4 is not None and ic4 < 0.0:
            warnings.warn("the reactor consumed more isobutane than it was fed "
                          "(I/O ratio far too low); the effluent has negative isobutane",
                          AlkylationRangeWarning, stacklevel=3)

    def route_yields(self) -> dict[str, tuple[float, float]]:
        """``{olefin: (Y_A, Y_H)}``: stoichiometric volume yields of both routes."""
        return {o: (float(a), float(b)) for o, a, b in zip(OLEFINS, self._Y_A, self._Y_H)}

    def heats_of_reaction(self) -> dict[str, tuple[float, float]]:
        """``{olefin: (dH_A, dH_H)}``, liquid heat of reaction per mol olefin (J/mol)."""
        return {o: (float(a), float(b)) for o, a, b in zip(OLEFINS, self._dH_A, self._dH_H)}


__all__ = [
    "AlkylationRangeWarning", "AlkylationReactor", "AlkylationReactorParams",
    "DEFAULT_SELECTIVITY", "alkylation_thermo", "feed_stream", "flows_array",
    "isobutane_latent_heat", "standard_volume", "stream_from_array", "to_bpd",
]
