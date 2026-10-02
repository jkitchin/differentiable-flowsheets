"""The VGO hydrocracker: pretreat reactor, cracking reactor, HP separator, recycle gas, fractionator, UCO recycle.

:class:`Hydrocracker` is a single-stage, series-flow hydrocracker assembled
from the shared hydroprocessing pieces (:mod:`difflow_refinery.hydroprocessing`)::

    fresh VGO --> [pretreat beds] --> (+ UCO recycle) --> [cracking beds] --> cooler --> HPS
                     ^   quench           quench ^ ^ ^                                  |   |
    treat gas -------+---------------------------+-+-+                          vapour  |   | liquid
         ^                                                                             v   |
         +---- compressor <---- purge <---- amine <---- KO drum <----------------------+   |
         ^ makeup H2                                                                       v
                         off-gas, LPG, light/heavy naphtha, kerosene, diesel, UCO <-- fractionator
                                                                     |
                                     UCO bleed <---------------------+---> UCO recycle (to the cracking reactor)

* **Pretreat** -- the hydrotreating kinetics (:class:`~difflow_refinery.hydrotreating.kinetics.HDTKinetics`)
  with a VGO parameter set (:data:`~.feed.VGO_PRETREAT_PARAMS`): HDS, HDN
  (what protects the cracking catalyst) and aromatics saturation.
* **Cracking** -- :class:`~.kinetics.HCKinetics`: continuous lumping (or
  discrete lumps) on the pseudo-component grid, organic-N inhibition, the
  hydrotreating network continuing on the cracking catalyst. The pretreat
  effluent goes to it whole (H2S and NH3 included: series flow).
* **HP separator and recycle-gas loop** -- the hydrotreater's (PR flash,
  knock-out, amine, purge, ideal-gas compressor, makeup to an H2/oil spec),
  closed by a Newton tear on the recycle gas.
* **Fractionator** -- a documented simplified split by TBP cut points
  (:mod:`.fractionator`), not a column.
* **UCO recycle** -- a fraction of the fractionator bottoms returns to the
  cracking reactor inlet: a tear on the UCO stream (molecules and attributes
  of every UCO cut), solved by successive substitution around the gas-loop
  Newton, with an adjoint (implicit) gradient (:mod:`.fixed_point`).

Everything is differentiable -- with respect to every spec, every kinetic
constant, the feed rate and the characterization (a TBP point of the assay)
-- by implicit-function gradients of both tears and the flashes and the
discrete adjoint of the bed integrations. Reverse mode only: the reactor's
diffrax adjoint is a ``custom_vjp``.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow_refinery import correlations as corr
from difflow_refinery.assay import LIGHT_END_SG
from difflow_refinery.blending import cetane_index_d4737, tbp_temperature, tbp_to_d86
from difflow_refinery.characterization import RHO_WATER_15C, BlendCharacterization
from difflow_refinery.composition import NITROGEN_CLASSES, SULFUR_CLASSES
from difflow_refinery.hydrocracking.feed import VGO_PRETREAT_PARAMS, hcu_feed, hcu_layout
from difflow_refinery.hydrocracking.fixed_point import fixed_point
from difflow_refinery.hydrocracking.fractionator import (
    DEFAULT_CUT_POINTS, LIQUID_PRODUCTS, PRODUCTS, cut_shares, fractionate, uco_cuts)
from difflow_refinery.hydrocracking.kinetics import HCKineticParams, HCKinetics, HCU_ATTRIBUTES
from difflow_refinery.hydroprocessing.layout import ELEMENTS, GAS_ELEMENTS, Flows, gas_mw, relative_balance_error
from difflow_refinery.hydroprocessing.reactor import (
    ReactorOptions, ReactorResult, TrickleBedReactor, integrate_bed, stream_enthalpy, vapor_enthalpy)
from difflow_refinery.hydroprocessing.recycle import (
    MOL_PER_NM3, amine_scrub, compress, knockout, makeup_for_ratio, makeup_vector, purge_split, solve_tear)
from difflow_refinery.hydroprocessing.separator import HPSeparator
from difflow_refinery.hydroprocessing.solve import newton_scalar
from difflow_refinery.hydroprocessing.thermo import Components
from difflow_refinery.hydrotreating.feed import DEFAULT_AROMATIC_SPLIT, cut_indices
from difflow_refinery.hydrotreating.kinetics import AROMATIC_CLASSES, HDTKineticParams, HDTKinetics, crack_targets
from difflow_refinery.hydrotreating.unit import BARREL, C_TO_K, SCF_PER_NM3, VOLUME_INCREMENTS, TargetSpec
from difflow_refinery.thermo import LIGHT_ENDS, RHO_WATER_60F

jax.config.update("jax_enable_x64", True)


class HydrocrackerConvergenceWarning(UserWarning):
    """The recycle-gas tear or the UCO recycle did not converge."""


@dataclass
class HydrocrackerParams(ParamsMixin):
    """Operating specs of :class:`Hydrocracker`.

    Quench rates are fractions of the TOTAL treat gas; the pretreat reactor's
    first bed gets what the quenches leave.

    Attributes:
        T_pretreat: Pretreat first-bed inlet temperature (K).
        quench_pretreat: Quench into each later pretreat bed (fractions of the treat gas).
        pretreat_beds: Catalyst share of each pretreat bed (sums to 1).
        lhsv_pretreat: Pretreat LHSV on fresh feed, 1/h.
        T_crack: Cracking first-bed inlet temperature (K); an interstage
            exchanger is implied (its duty is not computed).
        quench_crack: ``None`` (default): every cracking bed's inlet is held at
            ``T_crack`` and the quench each needs is solved (with the gas loop;
            the pretreat inlet gets what the quenches leave, ``crack.gas_left``);
            or the quench into each later cracking bed as fixed fractions of
            the treat gas (inlets then follow -- a fixed quench is a
            knife-edge: the beds run away or die out with a few kelvin).
        crack_beds: Catalyst share of each cracking bed (sums to 1).
        lhsv_crack: Cracking LHSV on FRESH feed, 1/h (catalyst volume = fresh
            feed rate / LHSV, so a recycle loads the same catalyst harder).
        catalyst_density: Loaded catalyst density, kg/m^3.
        P: Reactor pressure (Pa), uniform.
        h2_oil: Treat-gas H2 per fresh feed, Nm^3/m^3 (quench included).
        purge: Fraction of the scrubbed separator gas purged.
        makeup: Makeup-gas composition, ``{gas: mole fraction}``.
        h2s_removal: Fraction of the recycle gas's H2S removed by the amine.
        nh3_removal: Fraction of its NH3 removed (wash water).
        hps_T: HP separator temperature (K).
        loop_dP: Reactor-to-separator pressure drop, made up by the compressor (Pa).
        compressor_eta: Recycle compressor isentropic efficiency.
        uco_recycle: Fraction of the fractionator UCO returned to the cracking
            reactor (0 once-through); the rest is the UCO bleed.
        cut_points: Fractionator TBP cut points LN/HN, HN/kero, kero/diesel,
            diesel/UCO (K).
        cut_width: Half-width of the fractionator's smooth split (K).
        pretreat_kinetics: :class:`HDTKineticParams` of the pretreat bed.
        crack_kinetics: :class:`HCKineticParams` of the cracking bed.
        kij: PR binary interaction parameters (dict of name pairs), or None.
        reactor: :class:`ReactorOptions`.
        tear_tol: Recycle-gas tear tolerance (scaled residual).
        uco_tol: UCO recycle tolerance (max scaled change per pass).
        uco_max_steps: UCO substitution limit.
    """

    T_pretreat: float = 643.15
    quench_pretreat: tuple = (0.06,)
    pretreat_beds: tuple = (0.5, 0.5)
    lhsv_pretreat: float = 1.5
    T_crack: float = 653.15
    quench_crack: tuple | None = None
    crack_beds: tuple = (0.15, 0.18, 0.20, 0.22, 0.25)
    lhsv_crack: float = 1.5
    catalyst_density: float = 800.0
    P: float = 150e5
    h2_oil: float = 1500.0
    purge: float = 0.05
    makeup: dict = field(default_factory=lambda: {"hydrogen": 0.99, "methane": 0.01})
    h2s_removal: float = 0.99
    nh3_removal: float = 1.0
    hps_T: float = 323.15
    loop_dP: float = 8e5
    compressor_eta: float = 0.75
    uco_recycle: float = 0.0
    cut_points: tuple = DEFAULT_CUT_POINTS
    cut_width: float = 15.0
    pretreat_kinetics: HDTKineticParams = field(default_factory=lambda: VGO_PRETREAT_PARAMS)
    crack_kinetics: HCKineticParams = field(default_factory=HCKineticParams)
    kij: dict | None = None
    reactor: ReactorOptions = field(default_factory=ReactorOptions)
    tear_tol: float = 1e-11
    uco_tol: float = 1e-12
    uco_max_steps: int = 80


#: Units of the scalar outputs (per-product keys are ``<product>.<quantity>``,
#: see :data:`PRODUCT_OUTPUT_UNITS`).
OUTPUT_UNITS: dict[str, str] = {
    "conversion.per_pass": "-", "conversion.overall": "-", "naphtha_to_middle_distillate": "-",
    "feed.rate": "kg/s", "feed.volume": "m3/s", "feed.S_wppm": "wppm", "feed.N_wppm": "wppm",
    "feed.370plus": "kg/s", "pretreat.N_wppm": "wppm", "pretreat.S_wppm": "wppm",
    "wabt.pretreat": "K", "wabt.crack": "K", "pretreat.dT_total": "K", "crack.dT_total": "K",
    "h2.chemical": "mol/s", "h2.chemical_nm3_m3": "Nm3/m3", "h2.chemical_scf_bbl": "scf/bbl",
    "h2.chemical_wt": "wt% of feed", "h2.consumed_by_balance": "mol/s", "h2.makeup": "mol/s",
    "h2.makeup_nm3_m3": "Nm3/m3", "h2.purge": "mol/s", "h2.dissolved": "mol/s",
    "recycle.rate": "mol/s", "recycle.h2_purity": "-", "purge.rate": "mol/s", "makeup.rate": "mol/s",
    "compressor.power": "W", "compressor.T_out": "K", "reactor.pH2_in": "Pa",
    "uco.recycle_rate": "kg/s", "uco.bmci": "-", "diesel.cetane_index": "-",
    "catalyst.pretreat": "kg", "catalyst.crack": "kg", "liquid.volume_yield": "-",
    "tear.residual": "-", "uco.residual": "-", "uco.steps": "-", "T_shift": "K",
}
PRODUCT_OUTPUT_UNITS: dict[str, str] = {
    "rate": "kg/s", "yield": "- (mass, fresh feed)", "volume": "m3/s (60 F)", "volume_yield": "-", "sg": "-",
    "api": "API", "S_wppm": "wppm", "N_wppm": "wppm", "H_wt": "wt%", "aromatics_vol": "vol%",
    "T05": "K", "T10": "K", "T50": "K", "T90": "K", "T95": "K",
}


@dataclass(frozen=True)
class HydrocrackerResult:
    """A solved hydrocracker.

    Attributes:
        outputs: ``{name: value}`` (see :data:`OUTPUT_UNITS`, :data:`PRODUCT_OUTPUT_UNITS`).
        streams: ``{name: Flows}``: ``feed``, ``makeup``, ``treat_gas``,
            ``pretreat_out``, ``crack_in``, ``crack_out``, ``hps_vapor``,
            ``hps_liquid``, ``hps_water``, ``acid_gas``, ``purge``, ``recycle``,
            every fractionator product of :data:`~.fractionator.PRODUCTS`
            (``uco`` is the whole bottoms), ``uco_bleed`` and ``uco_recycle``.
        pretreat, crack: The two :class:`~difflow_refinery.hydroprocessing.reactor.ReactorResult` s.
        product_grid: Per-component properties of the product grid (light ends then cuts).
        balances: ``{"mass", "C", "H", "S", "N": relative error}`` over the unit.
        converged: Whether both tears converged.
        product_names: Names of the product grid's components.
    """

    outputs: dict
    streams: dict
    pretreat: Any
    crack: Any
    product_grid: Any
    balances: dict
    converged: Array
    product_names: tuple = ()

    @property
    def product_char(self) -> BlendCharacterization:
        """:class:`BlendCharacterization` of the product grid, for ``BlendComponent.from_stream``."""
        g = self.product_grid
        return BlendCharacterization(names=list(self.product_names), Tb=g.Tb, SG=g.SG, MW=g.MW, Tc=g.Tc,
                                     Pc=g.Pc, omega=g.omega, qualities=dict(g.qualities))

    def product_stream(self, name: str, T=298.15, P=101325.0) -> dict:
        """Product ``name`` as an ``F_<component>`` stream on :attr:`product_char`."""
        f = self.streams[name]
        names = list(self.product_names)
        n_light = len(names) - f.cut.shape[0]
        out = {f"F_{names[i]}": self.streams["_light_" + name][i] for i in range(n_light)}
        for i, c in enumerate(names[n_light:]):
            out[f"F_{c}"] = f.cut[i]
        out["T"] = jnp.asarray(T)
        out["P"] = jnp.asarray(P)
        return out

    def table(self) -> str:
        """The headline outputs as text."""
        o = {k: float(v) for k, v in self.outputs.items() if np.ndim(v) == 0}
        rows = [
            f"conversion (370 C+): per pass {100 * o['conversion.per_pass']:.1f} %, overall "
            f"{100 * o['conversion.overall']:.1f} %; UCO recycle {o['uco.recycle_rate']:.2f} kg/s",
            f"WABT pretreat {o['wabt.pretreat'] - C_TO_K:.1f} C (dT {o['pretreat.dT_total']:.1f} K), "
            f"cracking {o['wabt.crack'] - C_TO_K:.1f} C (dT {o['crack.dT_total']:.1f} K); "
            f"organic N to cracker {o['pretreat.N_wppm']:.1f} wppm",
            "yields (wt% fresh feed): " + ", ".join(
                f"{p} {100 * o[p + '.yield']:.2f}" for p in ("off_gas", "lpg") + LIQUID_PRODUCTS[:-1] + ("uco_bleed",)),
            f"H2: chemical {o['h2.chemical_nm3_m3']:.0f} Nm3/m3 ({o['h2.chemical_scf_bbl']:.0f} scf/bbl, "
            f"{o['h2.chemical_wt']:.2f} wt%), makeup {o['h2.makeup_nm3_m3']:.0f} Nm3/m3",
            f"kerosene SG {o['kerosene.sg']:.4f}; diesel SG {o['diesel.sg']:.4f}, cetane index "
            f"{o['diesel.cetane_index']:.1f}, S {o['diesel.S_wppm']:.1f} wppm; UCO BMCI {o['uco.bmci']:.1f}",
        ]
        return "\n".join(rows)


jax.tree_util.register_dataclass(HydrocrackerResult, data_fields=["outputs", "streams", "pretreat", "crack",
                                                                  "product_grid", "balances", "converged"],
                                 meta_fields=["product_names"])


@dataclass(frozen=True)
class _ProductGrid:
    Tb: Array
    SG: Array
    MW: Array
    Tc: Array
    Pc: Array
    omega: Array
    qualities: dict


jax.tree_util.register_dataclass(_ProductGrid, data_fields=["Tb", "SG", "MW", "Tc", "Pc", "omega", "qualities"],
                                 meta_fields=[])


def bmci(vabp_K, SG):
    """US Bureau of Mines Correlation Index: ``48640 / VABP(K) + 473.7 SG - 456.8``.

    The standard form (Smith, H.M., US Bureau of Mines Tech. Paper 610,
    1940, as given e.g. in Gary, Handwerk & Kaiser; source not checked --
    unverified). VABP is the volume-average boiling point. 0 for n-paraffins,
    100 for benzene by construction.
    """
    return 48640.0 / vabp_K + 473.7 * SG - 456.8


def _tree_float(p):
    return jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype=float), p)


class Hydrocracker:
    """VGO hydrocracker on a characterization (see the module docstring).

    Args:
        char: The :class:`~difflow_refinery.assay.Characterization`, with a
            composition (``characterize(assay, composition=True)``). Concrete.
        feed: A concrete feed stream (``F_<char.names>``, e.g. a VDU LVGO +
            HVGO), used to pick the cuts: every pseudo-component from the
            lightest up to the heaviest carrying more than ``trace`` of the
            feed's pseudo-component MASS. Or give ``cuts``.
        params: :class:`HydrocrackerParams`.
        cuts: The cuts explicitly (lightest first).
        recycle: Build the UCO recycle tear (default: ``params.uco_recycle > 0``).
            Static; a once-through unit cannot be given a recycle later.
        target: Optional :class:`~difflow_refinery.hydrotreating.unit.TargetSpec`
            on any output (e.g. ``"conversion.per_pass"`` or ``"wabt.crack"``),
            met by shifting the cracking inlet temperature.
        trace: See ``feed``. What is dropped is :attr:`dropped_mass_fraction`.
        uco_margin: How far (K) the UCO cut point may move below its value at
            construction before the UCO tear misses a cut (see :func:`.fractionator.uco_cuts`).
    """

    def __init__(self, char, feed: Mapping | None = None, params: HydrocrackerParams | None = None,
                 cuts: Sequence[str] | None = None, recycle: bool | None = None,
                 target: TargetSpec | None = None, trace: float = 1e-4,
                 aromatic_split=DEFAULT_AROMATIC_SPLIT, uco_margin: float = 30.0):
        if char.composition is None:
            raise ValueError("the characterization needs a composition: characterize(assay, composition=True)")
        self.char = char
        self.params = p = params or HydrocrackerParams()
        self.target = target
        self.aromatic_split = aromatic_split
        mw = dict(zip(char.names, np.asarray(char.component_MW)))
        if cuts is None:
            if feed is None:
                raise ValueError("give a feed (to pick the cuts) or cuts")
            m = np.asarray([float(feed.get(f"F_{c}", 0.0)) * mw[c] for c in char.pseudo_names])
            used = np.nonzero(m > trace * m.sum())[0]
            if used.size == 0:
                raise ValueError("the feed carries none of the characterization's pseudo-components")
            cuts = char.pseudo_names[: int(used.max()) + 1]
        self.cuts = tuple(cuts)
        self.layout = hcu_layout(char, self.cuts)
        self.dropped_mass_fraction = 0.0
        if feed is not None:
            mm = {n: float(feed.get(f"F_{n}", 0.0)) * mw[n] for n in char.names}
            kept = set(self.cuts) | set(self.layout.gases)
            tot = sum(mm.values())
            self.dropped_mass_fraction = sum(v for n, v in mm.items() if n not in kept) / tot if tot else 0.0
        idx = cut_indices(char, self.cuts)
        comp = char.composition
        cpm = np.asarray(char.component_MW)[idx] * np.asarray(comp.carbon)[idx] / 12.0107
        self.hdt_kinetics = HDTKinetics(self.layout, crack_targets(cpm))
        ck = p.crack_kinetics
        self.kinetics = HCKinetics(self.layout, self.hdt_kinetics if ck.hdt is not None else None,
                                   scheme=ck.scheme, aromatic_table=aromatic_split)
        for nm, q, b in (("pretreat", p.quench_pretreat, p.pretreat_beds), ("crack", p.quench_crack, p.crack_beds)):
            if q is not None and len(q) != len(b) - 1:
                raise ValueError(f"quench_{nm} needs {len(b) - 1} fractions")
        self.pretreat_reactor = TrickleBedReactor(self.layout, self.hdt_kinetics, p.reactor)
        self.crack_reactor = TrickleBedReactor(self.layout, self.kinetics, p.reactor)
        self.recycle = (p.uco_recycle > 0) if recycle is None else bool(recycle)
        Tb_all = np.concatenate([np.zeros(len(char.light_names)), np.asarray(char.Tb)])[idx]
        self.uco_index = uco_cuts(Tb_all, float(p.cut_points[-1]), float(p.cut_width), uco_margin)
        self._jit = jax.jit(self._run)

    # ----- inputs --------------------------------------------------------------

    def theta(self, feed: Mapping, char=None, params: HydrocrackerParams | None = None) -> dict:
        """The differentiable inputs of a solve, as a pytree."""
        char = self.char if char is None else char
        p = self.params if params is None else params
        if tuple(char.names) != tuple(self.char.names):
            raise ValueError("char must have the same components as the one the unit was built on")
        lay = self.layout
        idx = jnp.asarray(cut_indices(char, self.cuts))
        pidx = idx - len(char.light_names)
        skip = ("makeup", "pretreat_kinetics", "crack_kinetics", "kij", "reactor", "tear_tol", "uco_tol",
                "uco_max_steps")
        num = {f.name: jnp.asarray(getattr(p, f.name), dtype=float) for f in dataclasses.fields(p)
               if f.name not in skip and getattr(p, f.name) is not None}
        light = list(char.light_names)
        return {
            "oil": hcu_feed(char, feed, lay, self.aromatic_split),
            "cut": {
                "Tb": char.Tb[pidx], "SG": char.SG[pidx], "MW": char.MW[pidx], "Tc": char.Tc[pidx],
                "Pc": char.Pc[pidx], "omega": char.omega[pidx], "hvap_nb": char.hvap_nb[pidx],
                "cp_ig": char.cp_ig_coeffs[pidx], "T_lo": char.cut_edges[:-1][pidx],
                "T_hi": char.cut_edges[1:][pidx],
            },
            "unit_attr": hcu_feed(char, {f"F_{c}": 1.0 for c in self.cuts}, lay, self.aromatic_split).attr,
            "makeup_y": makeup_vector(lay, p.makeup),
            "num": num,
            "pre": _tree_float(p.pretreat_kinetics),
            "crk": _tree_float(p.crack_kinetics),
            "light_feed": jnp.stack([jnp.asarray(feed.get(f"F_{n}", 0.0), dtype=float) for n in light])
            if light else jnp.zeros(0),
        }

    def _comps(self, th) -> Components:
        c = th["cut"]
        return Components.build(self.layout, c["Tb"], c["SG"], c["MW"], c["Tc"], c["Pc"], c["omega"],
                                c["hvap_nb"], c["cp_ig"], kij=self.params.kij)

    def _feed_volume(self, th) -> Array:
        """Fresh-feed standard liquid volume, m^3/s at 60 F."""
        c = th["cut"]
        m = th["oil"].cut_mass(self.layout)
        V = jnp.sum(m / (c["SG"] * RHO_WATER_60F))
        for i, nme in enumerate(self.char.light_names):
            if nme in LIGHT_END_SG:
                V = V + th["light_feed"][i] * LIGHT_ENDS[nme][0] / 1000.0 / (LIGHT_END_SG[nme] * RHO_WATER_60F)
        return V

    def _crack_state(self, th):
        lay = self.layout
        c = th["cut"]
        m = th["oil"].cut_mass(lay)
        Kw = jnp.sum(m * corr.watson_k(c["Tb"], c["SG"])) / jnp.sum(m)
        return self.kinetics.prepare(th["crk"], c["Tb"], c["T_lo"], c["T_hi"], Kw)

    # ----- the UCO recycle vector ------------------------------------------------

    def _uco_flows(self, u: Array) -> Flows:
        lay = self.layout
        nu = self.uco_index.size
        cut = jnp.zeros(lay.n_cut).at[self.uco_index].set(u[:nu])
        attr = jnp.zeros((lay.n_cut, lay.n_attr)).at[self.uco_index].set(u[nu:].reshape(nu, lay.n_attr))
        return Flows(jnp.zeros(lay.n_gas), cut, attr)

    def _uco_vector(self, f: Flows) -> Array:
        return jnp.concatenate([f.cut[self.uco_index], f.attr[self.uco_index].reshape(-1)])

    # ----- one pass round the gas loop ----------------------------------------------

    def _evaluate(self, x, A, adjoint=None):
        """One pass from recycle gas ``x`` (gas flows, compressor outlet T); ``A = (th, comps, cst, uco, T_shift)``."""
        th, comps, cst, uco, T_shift = A
        lay = self.layout
        p = self.params
        n = th["num"]
        oil = th["oil"]
        Q = self._feed_volume(th)
        W_pre_tot = Q * 3600.0 / n["lhsv_pretreat"] * n["catalyst_density"]
        W_crk_tot = Q * 3600.0 / n["lhsv_crack"] * n["catalyst_density"]
        W_pre = [W_pre_tot * n["pretreat_beds"][k] for k in range(len(p.pretreat_beds))]
        W_crk = [W_crk_tot * n["crack_beds"][k] for k in range(len(p.crack_beds))]
        h2_target = n["h2_oil"] * Q * MOL_PER_NM3
        P_hps = n["P"] - n["loop_dP"]
        R = Flows(jnp.maximum(x[:-2], 0.0), jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
        T_gas = x[-2]
        makeup, M = makeup_for_ratio(R, lay, th["makeup_y"], h2_target)
        gas = R + makeup
        q_pre = n["quench_pretreat"]
        n_crk = len(p.crack_beds)
        if n_crk == 1:
            s_crk = jnp.asarray(0.0)
        elif p.quench_crack is None:
            # the cracking quench, a tear unknown: the share of the treat gas the
            # cracking beds' quench valves take (solved with the gas loop)
            # (clipped to a physical range so a Newton iterate cannot starve or flood
            # the pretreat bed; the clip is inactive at any solution the unit accepts)
            s_crk = jnp.clip(x[-1], 0.0, 0.9)
        else:
            s_crk = jnp.sum(n["quench_crack"])
        gas1 = gas.scale(1.0 - s_crk)
        quench1 = [q_pre[k] / (1.0 - s_crk) for k in range(q_pre.shape[0])] if len(p.pretreat_beds) > 1 else None
        rx1 = self.pretreat_reactor(oil, gas1, [n["T_pretreat"]], T_gas, n["P"], W_pre, comps, th["pre"],
                                    quench=quench1, adjoint=adjoint)
        crack_in = rx1.outlet + uco
        T2_in = n["T_crack"] + T_shift
        if n_crk > 1 and p.quench_crack is None:
            rx2 = self._crack_beds(crack_in, gas, T2_in, T_gas, n["P"], W_crk, comps, cst, adjoint)
            s_next = jnp.sum(rx2.quench)
        else:
            if n_crk == 1:
                gas2, quench2 = gas.scale(0.0), None
            else:
                q_crk = n["quench_crack"]
                gas2, quench2 = gas.scale(s_crk), [q_crk[k] / s_crk for k in range(q_crk.shape[0])]
            rx2 = self.crack_reactor(crack_in, gas2, [T2_in], T_gas, n["P"], W_crk, comps, cst,
                                     quench=quench2, adjoint=adjoint)
            s_next = s_crk
        vap, liq, water, fr = HPSeparator(lay)(rx2.outlet, n["hps_T"], P_hps, comps)
        vap, liq = knockout(vap, liq)
        sweet, absorbed = amine_scrub(vap, lay, n["h2s_removal"], n["nh3_removal"])
        rec, purge = purge_split(sweet, n["purge"])
        T2, Wc = compress(rec, lay, comps, n["hps_T"], P_hps, n["P"], n["compressor_eta"])
        prods = fractionate(liq, lay, th["cut"]["Tb"], [n["cut_points"][k] for k in range(4)], n["cut_width"])
        out = dict(R=R, makeup=makeup, M=M, gas=gas, rx1=rx1, rx2=rx2, crack_in=crack_in, vap=vap, liq=liq,
                   water=water, absorbed=absorbed, rec=rec, purge=purge, T2=T2, Wc=Wc, W_pre=W_pre, W_crk=W_crk,
                   Q=Q, prods=prods, uco_in=uco, s_crk=s_crk)
        return jnp.concatenate([rec.gas, T2[None], jnp.reshape(s_next, (1,))]), out

    def _crack_beds(self, stream, gas, T_in, T_gas, P, W, comps, cst, adjoint=None) -> ReactorResult:
        """Cracking beds with every inlet held at ``T_in``, the quench into each later bed explicit.

        The first bed takes the pretreat effluent (and the UCO) with no gas of
        its own; the quench into bed ``k+1`` is the treat-gas fraction whose
        heating from ``T_gas`` to ``T_in`` absorbs the cooling of bed ``k``'s
        outlet to ``T_in`` -- the enthalpy balance of
        :class:`~difflow_refinery.hydroprocessing.reactor.TrickleBedReactor`'s
        ``quench=None`` mode (same functions, same linearised K-values). Here
        the total is a tear unknown of the gas loop instead of a Newton solve
        of its own, so a pass runs each bed once.
        """
        lay = self.layout
        kin = self.kinetics
        opts = self.crack_reactor.options if adjoint is None else dataclasses.replace(
            self.crack_reactor.options, adjoint=adjoint)
        Ws = jnp.stack([jnp.asarray(w, dtype=float) for w in W])
        last = jnp.arange(len(W)) == len(W) - 1
        T_in = jnp.asarray(T_in, dtype=float)

        def body(s, xs):
            Wk, is_last = xs
            bed = integrate_bed(kin, cst, lay, comps, s, T_in, P, Wk, opts)
            km, out = bed.k_model, bed.outlet
            dH = stream_enthalpy(out, lay, comps, km, bed.T_out) - stream_enthalpy(out, lay, comps, km, T_in)
            per = vapor_enthalpy(gas, lay, comps, T_in) - vapor_enthalpy(gas, lay, comps, T_gas)
            qk = jnp.where(is_last, 0.0, dH / per)
            return out + gas.scale(qk), (bed, qk)

        _, (beds, qs) = jax.lax.scan(body, stream, (Ws, last))
        q = jnp.concatenate([jnp.zeros(1), qs[:-1]])
        wabt = jnp.sum(Ws * (beds.T_in + 2.0 * beds.T_out) / 3.0) / jnp.sum(Ws)
        bed_list = tuple(jax.tree_util.tree_map(lambda a, k=k: a[k], beds) for k in range(len(W)))
        return ReactorResult(beds=bed_list, quench=q, outlet=bed_list[-1].outlet, T_out=beds.T_out[-1],
                             wabt=wabt, delta_T=beds.T_out - beds.T_in)

    #: Substitution passes before the gas-tear Newton (cold and warm start).
    PRE_PASSES = 12
    PRE_PASSES_WARM = 1

    def _gas_x0(self, th):
        lay = self.layout
        h2_target = jax.lax.stop_gradient(th["num"]["h2_oil"] * self._feed_volume(th) * MOL_PER_NM3)
        hi, mi = lay.gas_index("hydrogen"), lay.gas_index("methane")
        x0 = jnp.concatenate([(jnp.zeros(lay.n_gas) + 1e-4 * h2_target).at[hi].set(0.85 * h2_target)
                              .at[mi].set(0.08 * h2_target),
                              jax.lax.stop_gradient(th["num"]["hps_T"] + 25.0)[None], jnp.asarray([0.4])])
        scale = jnp.concatenate([jnp.full(lay.n_gas, 1e-2 * h2_target).at[hi].set(h2_target),
                                 jnp.asarray([100.0, 0.1])])
        return x0, scale

    def _gas_loop(self, th, comps, cst, uco, T_shift, x0=None, adjoint=None):
        """Converge the recycle-gas tear for a given UCO recycle; returns the converged pass (with ``"tear"``)."""
        A = (th, comps, cst, uco, T_shift)
        x_def, scale = self._gas_x0(th)
        n_pre = self.PRE_PASSES if x0 is None else self.PRE_PASSES_WARM
        x0 = x_def if x0 is None else x0
        # a few successive-substitution passes before Newton: from the default
        # start Newton's first steps overshoot the cracking-quench share (whose
        # row is coupled to the whole gas composition); substitution brings
        # every entry but the slow methane build-up close first
        A_s = jax.lax.stop_gradient(A)
        x0 = jax.lax.fori_loop(0, n_pre, lambda _, x: self._evaluate(x, A_s, adjoint="forward")[0],
                               jax.lax.stop_gradient(x0))
        sol = solve_tear(lambda x, A: self._evaluate(x, A, adjoint=adjoint)[0], x0, A, scale=scale,
                         tol=self.params.tear_tol, max_step=2.0,
                         g_iter=lambda x, A: self._evaluate(x, A, adjoint="forward")[0], jac="fwd")
        _, out = self._evaluate(sol.value, A, adjoint=adjoint)
        out["tear"] = sol
        return out

    def _solve_loops(self, th, comps, cst, T_shift):
        """Both tears: the gas loop alone (once-through), or the UCO recycle around it."""
        lay = self.layout
        n = th["num"]
        if not self.recycle:
            out = self._gas_loop(th, comps, cst, Flows.zeros(lay), T_shift)
            out["uco_sol"] = None
            return out
        nu = self.uco_index.size
        n_uco = nu * (1 + lay.n_attr)
        x_def, gscale = self._gas_x0(th)
        oil = th["oil"]
        # scale of the UCO entries: the fresh feed's flows in the UCO cuts (floored per column)
        ref = jax.lax.stop_gradient(self._uco_vector(oil))
        cmax = jax.lax.stop_gradient(jnp.concatenate([
            jnp.full(nu, jnp.max(oil.cut)),
            jnp.tile(jnp.max(jnp.abs(oil.attr), axis=0), nu)]))
        uscale = jnp.maximum(jnp.abs(ref), 1e-6 * cmax + 1e-300)

        def G(z, A, adjoint=None):
            th, comps, cst, T_shift = A
            uco = self._uco_flows(z[:n_uco])
            o = self._gas_loop(th, comps, cst, uco, T_shift, x0=jax.lax.stop_gradient(z[n_uco:]), adjoint=adjoint)
            new = self._uco_vector(o["prods"]["uco"].scale(th["num"]["uco_recycle"]))
            return jnp.concatenate([new, o["tear"].value])

        z0 = jnp.concatenate([jnp.zeros(n_uco), x_def])
        scale = jnp.concatenate([uscale, jnp.full(x_def.size, jnp.inf)])
        sol = fixed_point(G, z0, (th, comps, cst, T_shift), scale=scale, tol=self.params.uco_tol,
                          max_steps=int(self.params.uco_max_steps),
                          G_iter=lambda z, A: G(z, A, adjoint="forward"))
        z = sol.value
        out = self._gas_loop(th, comps, cst, self._uco_flows(z[:n_uco]), T_shift,
                             x0=jax.lax.stop_gradient(z[n_uco:]))
        out["uco_sol"] = sol
        return out

    def _run(self, th):
        comps = self._comps(th)
        cst = self._crack_state(th)
        if self.target is None:
            shift = jnp.asarray(0.0)
            out = self._solve_loops(th, comps, cst, shift)
        else:
            tgt = self.target

            def res(sh, A):
                th, comps, cst = A
                o = self._solve_loops(th, comps, cst, sh)
                return (self._outputs(th, comps, cst, o)[0][tgt.output] - tgt.target) / tgt.scale

            sol = newton_scalar(res, 0.0, (th, comps, cst), tol=1e-9, max_step=20.0)
            shift = sol.value
            out = self._solve_loops(th, comps, cst, shift)
        outputs, streams, grid, balances, conv = self._outputs(th, comps, cst, out, full=True)
        outputs["T_shift"] = shift
        return HydrocrackerResult(outputs=outputs, streams=streams, pretreat=out["rx1"], crack=out["rx2"],
                                  product_grid=grid, balances=balances, converged=conv,
                                  product_names=tuple(self._product_lights()) + tuple(self.cuts))

    # ----- products and outputs --------------------------------------------------------

    def _product_lights(self) -> list[str]:
        """Gases that are liquid blend components (C3+ with a standard liquid gravity), in layout order."""
        return [g for g in self.layout.gases if g in LIGHT_END_SG and g in LIGHT_ENDS
                and g not in ("methane", "ethane", "water")]

    def _cut_properties(self, th, cst, flows: Flows):
        """Per-cut MW, SG and qualities of the cracked/treated molecules in ``flows``."""
        lay = self.layout
        c = th["cut"]
        ai = {a: i for i, a in enumerate(lay.attributes)}
        pm = flows.per_molecule()
        inc = jnp.zeros(lay.n_attr)
        for a, v in VOLUME_INCREMENTS.items():
            inc = inc.at[ai[a]].set(v)
        v_feed = c["MW"] / (1000.0 * c["SG"] * RHO_WATER_60F)
        v0 = v_feed - th["unit_attr"] @ inc
        prep = cst.prep
        v_prod = prep.MW / (1000.0 * prep.SG * RHO_WATER_60F)
        prod_attr_inc = sum(prep.aromatics[:, k] * VOLUME_INCREMENTS[f"A_{a}"] for k, a in enumerate(AROMATIC_CLASSES))
        v0p = v_prod - prod_attr_inc
        f_cr = pm[:, ai["cracked"]]
        v_cut = (1.0 - f_cr) * v0 + f_cr * v0p + pm @ inc
        mw_cut = flows.cut_mw(lay)
        has = flows.cut > 1e-20 * jnp.sum(flows.cut)
        mw_cut = jnp.where(has, mw_cut, c["MW"])
        v_cut = jnp.where(has, v_cut, v_feed)
        sg = mw_cut / (1000.0 * v_cut * RHO_WATER_60F)
        safe = lambda a: jnp.where(has, a, 0.0)
        s_cols = [ai[f"S_{k}"] for k in SULFUR_CLASSES]
        n_cols = [ai[f"N_{k}"] for k in NITROGEN_CLASSES]
        q = {
            "S_ppm": safe(1e6 * pm[:, s_cols].sum(1) * 32.065 / mw_cut),
            "N_ppm": safe(1e6 * pm[:, n_cols].sum(1) * 14.0067 / mw_cut),
            "aromatics_vol": safe(100.0 * sum(pm[:, ai[f"A_{k}"]] for k in AROMATIC_CLASSES)),
            "olefins_vol": safe(100.0 * pm[:, ai["olefins"]]),
            "naphthenes_vol": safe(100.0 * pm[:, ai["naphthenes"]]),
            "H_wt": safe(100.0 * pm[:, ai["H"]] * 1.00794 / mw_cut),
        }
        q["paraffins_vol"] = 100.0 - q["aromatics_vol"] - q["olefins_vol"] - q["naphthenes_vol"]
        return mw_cut, sg, q

    def _outputs(self, th, comps, cst, o, full=False):
        lay = self.layout
        p = self.params
        n = th["num"]
        c = th["cut"]
        oil = th["oil"]
        rx1, rx2 = o["rx1"], o["rx2"]
        prods = dict(o["prods"])
        rho = n["uco_recycle"]
        uco_bleed = prods["uco"].scale(1.0 - rho)
        uco_rec_out = prods["uco"].scale(rho)
        prods["uco_bleed"] = uco_bleed
        mw_cut, sg_cut, qual = self._cut_properties(th, cst, rx2.outlet)
        lights = self._product_lights()
        lv = lambda i: jnp.asarray([LIGHT_ENDS[nm][i] for nm in lights], dtype=float).reshape(-1)
        l_sg = jnp.asarray([LIGHT_END_SG[nm] for nm in lights], dtype=float).reshape(-1)
        cat = lambda a, b: jnp.concatenate([a, b])
        zl = jnp.zeros(len(lights))
        l_h = jnp.asarray([100.0 * GAS_ELEMENTS[nm]["H"] * 1.00794 / gas_mw(nm) for nm in lights],
                          dtype=float).reshape(-1)
        grid = _ProductGrid(
            Tb=cat(lv(1), c["Tb"]), SG=cat(l_sg, sg_cut), MW=cat(jnp.asarray([gas_mw(nm) for nm in lights],
                                                                          dtype=float).reshape(-1), mw_cut),
            Tc=cat(lv(2), c["Tc"]), Pc=cat(lv(3), c["Pc"]), omega=cat(lv(4), c["omega"]),
            qualities={k: cat(l_h if k == "H_wt" else (100.0 + zl if k == "paraffins_vol" else zl), v)
                       for k, v in qual.items()})

        def light_flows(f: Flows):
            return jnp.stack([f.gas[lay.gas_index(nm)] for nm in lights]) if lights else jnp.zeros(0)

        feed_mass = oil.mass(lay)
        Q = o["Q"]
        outputs = {}

        def props(f: Flows, pre: str, tbp: bool = True):
            moles = cat(light_flows(f), f.cut)
            m = moles * grid.MW
            vol = m * 1e-3 / (grid.SG * RHO_WATER_60F)
            V = jnp.sum(vol)
            phi = vol / jnp.where(V > 0, V, 1.0)
            w = m / jnp.where(jnp.sum(m) > 0, jnp.sum(m), 1.0)
            sg = jnp.sum(phi * grid.SG)
            outputs[f"{pre}.volume"] = V
            outputs[f"{pre}.volume_yield"] = V / Q
            outputs[f"{pre}.sg"] = sg
            outputs[f"{pre}.api"] = 141.5 / sg - 131.5
            outputs[f"{pre}.S_wppm"] = jnp.sum(w * grid.qualities["S_ppm"])
            outputs[f"{pre}.N_wppm"] = jnp.sum(w * grid.qualities["N_ppm"])
            outputs[f"{pre}.H_wt"] = jnp.sum(w * grid.qualities["H_wt"])
            outputs[f"{pre}.aromatics_vol"] = jnp.sum(phi * grid.qualities["aromatics_vol"])
            if tbp:
                for pct in (5, 10, 50, 90, 95):
                    outputs[f"{pre}.T{pct:02d}" if pct < 10 else f"{pre}.T{pct}"] = tbp_temperature(
                        pct / 100.0, grid.Tb, phi, 5.0)
            return phi, sg

        for name in ("off_gas", "lpg") + LIQUID_PRODUCTS + ("uco_bleed",):
            f = prods[name]
            outputs[f"{name}.rate"] = f.mass(lay)
            outputs[f"{name}.yield"] = f.mass(lay) / feed_mass
        for name in ("lpg",) + LIQUID_PRODUCTS:
            phi, sg = props(prods[name], name, tbp=name != "lpg")
            if name == "diesel":
                d86 = {pct: tbp_to_d86(outputs[f"diesel.T{pct}"], pct) - C_TO_K for pct in (10, 50, 90)}
                outputs["diesel.cetane_index"] = cetane_index_d4737(sg * RHO_WATER_15C / 1000.0, d86[10], d86[50],
                                                                    d86[90])
            if name == "uco":
                outputs["uco.bmci"] = bmci(jnp.sum(phi * grid.Tb), sg)
        liq_vol = sum(outputs[f"{nm}.volume"] for nm in ("lpg",) + LIQUID_PRODUCTS[:-1]) \
            + outputs["uco.volume"] * (1.0 - rho)
        outputs["liquid.volume_yield"] = liq_vol / Q
        # conversion on the fractionator's own UCO cut
        above = cut_shares(c["Tb"], jnp.stack([n["cut_points"][k] for k in range(4)]), n["cut_width"])[:, -1]
        fresh_heavy = jnp.sum(oil.cut_mass(lay) * above)
        uco_total = prods["uco"].mass(lay)
        uco_in = o["uco_in"].mass(lay)
        outputs["feed.370plus"] = fresh_heavy
        outputs["conversion.per_pass"] = 1.0 - uco_total / (fresh_heavy + uco_in)
        outputs["conversion.overall"] = 1.0 - uco_bleed.mass(lay) / fresh_heavy
        outputs["naphtha_to_middle_distillate"] = (prods["light_naphtha"].mass(lay) + prods["heavy_naphtha"].mass(lay)) \
            / (prods["kerosene"].mass(lay) + prods["diesel"].mass(lay))
        outputs["uco.recycle_rate"] = uco_in
        # feed and pretreat
        ai = {a: i for i, a in enumerate(lay.attributes)}
        s_cols = [ai[f"S_{k}"] for k in SULFUR_CLASSES]
        n_cols = [ai[f"N_{k}"] for k in NITROGEN_CLASSES]
        outputs["feed.rate"] = feed_mass
        outputs["feed.volume"] = Q
        outputs["feed.S_wppm"] = 1e6 * jnp.sum(oil.attr[:, s_cols]) * 32.065e-3 / feed_mass
        outputs["feed.N_wppm"] = 1e6 * jnp.sum(oil.attr[:, n_cols]) * 14.0067e-3 / feed_mass
        hc1 = jnp.sum(rx1.outlet.cut_mass(lay))
        outputs["pretreat.N_wppm"] = 1e6 * jnp.sum(rx1.outlet.attr[:, n_cols]) * 14.0067e-3 / hc1
        outputs["pretreat.S_wppm"] = 1e6 * jnp.sum(rx1.outlet.attr[:, s_cols]) * 32.065e-3 / hc1
        # reactors
        outputs["wabt.pretreat"] = rx1.wabt
        outputs["wabt.crack"] = rx2.wabt
        outputs["pretreat.dT_total"] = jnp.sum(rx1.delta_T)
        outputs["crack.dT_total"] = jnp.sum(rx2.delta_T)
        for tag, rx in (("pretreat", rx1), ("crack", rx2)):
            for k, b in enumerate(rx.beds):
                outputs[f"{tag}.bed{k + 1}.T_in"] = b.T_in
                outputs[f"{tag}.bed{k + 1}.dT"] = b.T_out - b.T_in
        for k in range(len(p.crack_beds)):
            outputs[f"crack.bed{k + 1}.quench"] = rx2.quench[k]
        # share of the cracking reactor's gas left for its first bed (negative: the quench pool is too small)
        outputs["crack.gas_left"] = 1.0 - o["s_crk"] - jnp.sum(n["quench_pretreat"])
        outputs["catalyst.pretreat"] = sum(o["W_pre"])
        outputs["catalyst.crack"] = sum(o["W_crk"])
        # hydrogen: chemical consumption by the H balance on everything but H2
        hi = lay.gas_index("hydrogen")
        Hmat = jnp.asarray(lay.gas_element_matrix())[:, ELEMENTS.index("H")]
        not_h2 = jnp.ones(lay.n_gas).at[hi].set(0.0)

        def H_nonH2(f: Flows):
            return jnp.sum(f.gas * Hmat * not_h2) + jnp.sum(f.attribute(lay, "H"))

        outs_all = [o["purge"], o["absorbed"], o["water"]] + [prods[k] for k in PRODUCTS if k != "uco"] \
            + [uco_bleed]
        chem = (sum(H_nonH2(f) for f in outs_all) - H_nonH2(oil) - H_nonH2(o["makeup"])
                - H_nonH2(o["uco_in"]) + H_nonH2(uco_rec_out)) / 2.0
        H2_in = o["makeup"].gas[hi] + oil.gas[hi]
        H2_out = sum(f.gas[hi] for f in outs_all)
        outputs["h2.chemical"] = chem
        outputs["h2.consumed_by_balance"] = H2_in - H2_out
        outputs["h2.chemical_nm3_m3"] = chem / MOL_PER_NM3 / Q
        outputs["h2.chemical_scf_bbl"] = chem / MOL_PER_NM3 * SCF_PER_NM3 / (Q / BARREL)
        outputs["h2.chemical_wt"] = 100.0 * chem * 2.01588e-3 / feed_mass
        outputs["h2.makeup"] = o["makeup"].gas[hi]
        outputs["h2.makeup_nm3_m3"] = o["makeup"].gas[hi] / MOL_PER_NM3 / Q
        outputs["h2.purge"] = o["purge"].gas[hi]
        outputs["h2.dissolved"] = o["liq"].gas[hi]
        rec = o["rec"]
        outputs["recycle.rate"] = jnp.sum(rec.gas)
        outputs["recycle.h2_purity"] = rec.gas[hi] / jnp.sum(rec.gas)
        outputs["purge.rate"] = jnp.sum(o["purge"].gas)
        outputs["makeup.rate"] = o["M"]
        b1 = rx1.beds[0].inlet
        outputs["reactor.pH2_in"] = n["P"] * b1.gas[hi] / (jnp.sum(b1.gas) + jnp.sum(b1.cut))
        outputs["compressor.power"] = o["Wc"]
        outputs["compressor.T_out"] = o["T2"]
        outputs["tear.residual"] = o["tear"].residual
        usol = o["uco_sol"]
        outputs["uco.residual"] = usol.residual if usol is not None else jnp.asarray(0.0)
        outputs["uco.steps"] = jnp.asarray(usol.steps, dtype=float) if usol is not None else jnp.asarray(0.0)
        # balances: in = fresh oil + makeup; out = purge + acid gas + HPS water + products + UCO bleed
        # (+ the recycle tear's own residual: recycled UCO computed less recycled UCO assumed)
        tot_in = oil + o["makeup"]
        tot_out = outs_all[0]
        for f in outs_all[1:]:
            tot_out = tot_out + f
        tot_out = tot_out + uco_rec_out - o["uco_in"]
        e_in, e_out = tot_in.elements(lay), tot_out.elements(lay)
        balances = {"mass": relative_balance_error(tot_in.mass(lay), tot_out.mass(lay))}
        for k, e in enumerate(ELEMENTS):
            balances[e] = relative_balance_error(e_in[k], e_out[k])
        conv = o["tear"].converged
        if usol is not None:
            conv = conv & usol.converged
        if not full:
            return outputs, None, None, None, None
        streams = {"feed": oil, "makeup": o["makeup"], "treat_gas": o["gas"], "pretreat_out": rx1.outlet,
                   "crack_in": o["crack_in"], "crack_out": rx2.outlet, "hps_vapor": o["vap"], "hps_liquid": o["liq"],
                   "hps_water": o["water"], "acid_gas": o["absorbed"], "purge": o["purge"], "recycle": rec,
                   "uco_recycle": o["uco_in"]}
        streams.update(prods)
        for name in ("lpg",) + LIQUID_PRODUCTS + ("uco_bleed",):
            streams["_light_" + name] = light_flows(prods[name])
        return outputs, streams, grid, balances, conv

    # ----- public ---------------------------------------------------------------------

    def solve(self, feed: Mapping, char=None, params: HydrocrackerParams | None = None,
              warn: bool = True) -> HydrocrackerResult:
        """Solve the unit for ``feed`` (``F_<char.names>``, mol/s).

        ``char`` and ``params`` may replace the unit's (same names, same
        shape-setting settings: bed counts, kinetic scheme, tolerances) -- how
        a gradient with respect to the assay or a spec is taken.
        """
        if params is not None:
            for k in ("reactor", "tear_tol", "uco_tol", "uco_max_steps", "kij", "makeup"):
                if getattr(params, k) != getattr(self.params, k):
                    raise ValueError(f"params.{k} shapes the solve; build a new Hydrocracker to change it")
            for k in ("pretreat_beds", "crack_beds", "quench_pretreat", "quench_crack"):
                a, b = getattr(params, k), getattr(self.params, k)
                if (a is None) != (b is None) or (a is not None and len(a) != len(b)):
                    raise ValueError("the bed counts shape the solve; build a new Hydrocracker")
            if params.crack_kinetics.scheme != self.params.crack_kinetics.scheme or \
                    (params.crack_kinetics.hdt is None) != (self.params.crack_kinetics.hdt is None):
                raise ValueError("the cracking scheme shapes the solve; build a new Hydrocracker")
        res = self._jit(self.theta(feed, char, params))
        if warn:
            try:
                ok = bool(res.converged)
            except jax.errors.ConcretizationTypeError:
                ok = True
            if not ok:
                warnings.warn(f"hydrocracker did not converge (gas tear residual "
                              f"{float(res.outputs['tear.residual']):.2e}, UCO residual "
                              f"{float(res.outputs['uco.residual']):.2e})", HydrocrackerConvergenceWarning,
                              stacklevel=2)
        return res

    __call__ = solve


__all__ = ["Hydrocracker", "HydrocrackerParams", "HydrocrackerResult", "HydrocrackerConvergenceWarning",
           "OUTPUT_UNITS", "PRODUCT_OUTPUT_UNITS", "bmci"]
