"""The residue desulfurizer (ARDS/RDS): trickle beds with HDS, HDM and CCR reduction, and its products.

:class:`ResidueDesulfurizer` puts the residue kinetics (:mod:`.kinetics`) into
the shared trickle-bed reactor (:class:`~difflow_refinery.hydroprocessing.reactor.TrickleBedReactor`)::

    atmospheric residue --+--> [bed 1] --quench--> [bed 2] ... --> product separation --+--> gas (H2, H2S, NH3, C1-C5, water)
                          |                                                             +--> distillate (cuts below the cut point)
    treat gas ------------+                                                             +--> desulfurized residue (cuts above it)

* **Treat gas once through.** The treat gas is given (rate per m^3 of oil
  and composition, by default 90 % H2 / 10 % CH4 -- a scrubbed recycle gas
  with makeup), rather than solved as a recycle loop. That is the recycle
  loop with an ideal amine scrubber and the makeup set to hold the
  recycle purity; :mod:`difflow_refinery.hydroprocessing.recycle` has the
  pieces to close the loop, which matters for the hydrogen balance but
  little for the product sulfur. Chemical hydrogen consumption is reported
  from the hydrogen balance.
* **Product separation** is an ideal component split of the reactor
  effluent: every gas and light end to the gas, every cut boiling below
  ``product_cut_T`` (350 C) to the distillate, the rest to the desulfurized
  residue. It stands in for the hot and cold separators and the
  fractionator; the balances close through it exactly.
* **Catalyst grading** (HDM catalyst in front of HDS catalyst) is lumped
  into one average catalyst for all beds.
* No feed heater (bed-1 inlet temperature is a spec, the treat gas is at it)
  and no bed pressure drop.

Everything is differentiable -- in every parameter, the feed and the
characterization -- through the discrete adjoint of the bed integrations
(and the implicit-function gradient of the quench mixing).
Kinetic constants are ILLUSTRATIVE (see :mod:`.kinetics`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow_refinery.blending import BlendComponent
from difflow_refinery.characterization import RHO_WATER_15C, BlendCharacterization
from difflow_refinery.composition import NITROGEN_CLASSES, SULFUR_CLASSES
from difflow_refinery.hydroprocessing.layout import ATOMIC_MASS, ELEMENTS, Flows, relative_balance_error
from difflow_refinery.hydroprocessing.reactor import ReactorOptions, TrickleBedReactor
from difflow_refinery.hydroprocessing.recycle import MOL_PER_NM3, makeup_vector
from difflow_refinery.hydroprocessing.thermo import Components
from difflow_refinery.hydrotreating.feed import cut_indices
from difflow_refinery.hydrotreating.kinetics import AROMATIC_CLASSES, crack_targets
from difflow_refinery.residue.feed import DEFAULT_RESIDUE_S_SHARE, rds_feed, rds_layout
from difflow_refinery.residue.kinetics import (
    CCR_MW, METAL_MW, RDSKineticParams, RDSKinetics, conversion_targets)
from difflow_refinery.thermo import RHO_WATER_60F

jax.config.update("jax_enable_x64", True)

C_TO_K = 273.15

#: Liquid molar-volume change (m^3/mol) per molecule converted to the next
#: more saturated type -- the hydrotreater's model-compound values (see
#: ``difflow_refinery.hydrotreating.unit.VOLUME_INCREMENTS``), restated here
#: so the residue unit does not depend on the hydrotreater's unit module.
VOLUME_INCREMENTS: dict[str, float] = {
    "A_mono": -20.1e-6, "A_di": -31.1e-6, "A_poly": -42.1e-6, "olefins": -5.5e-6, "naphthenes": 0.0,
}


@dataclass
class RDSParams(ParamsMixin):
    """Operating specs of :class:`ResidueDesulfurizer`.

    Attributes:
        T_in: Bed inlet temperatures (K): only the first bed's when ``quench``
            is given; one per bed when ``quench`` is ``None`` (then the
            quench each later bed needs is solved for).
        quench: Quench into each later bed as a fraction of the treat gas,
            or ``None``.
        bed_fractions: Share of the catalyst in each bed (sums to 1).
        P: Reactor pressure (Pa).
        lhsv: Liquid hourly space velocity, 1/h (feed standard volume at 60 F
            per hour per volume of catalyst).
        catalyst_density: Loaded catalyst density, kg/m^3.
        gas_oil: Treat gas (all species) to oil ratio, Nm^3 per m^3 of feed
            (60 F), reactor inlet including quench.
        treat_gas: Treat-gas composition ``{gas: mole fraction}``.
        T_gas: Quench-gas temperature (K).
        product_cut_T: Normal boiling point (K) dividing the distillate from
            the desulfurized residue in the product separation.
        T_conv: Cuts boiling at or above this (K) convert (see :mod:`.kinetics`).
        kinetics: :class:`~.kinetics.RDSKineticParams`.
        kij: PR binary interaction parameters, or ``None`` for the defaults.
        reactor: :class:`~difflow_refinery.hydroprocessing.reactor.ReactorOptions`.
    """

    T_in: tuple = (646.15,)
    quench: tuple | None = (0.2, 0.25)
    bed_fractions: tuple = (0.25, 0.35, 0.4)
    P: float = 150e5
    lhsv: float = 0.25
    catalyst_density: float = 800.0
    gas_oil: float = 1000.0
    treat_gas: dict = field(default_factory=lambda: {"hydrogen": 0.90, "methane": 0.10})
    T_gas: float = 343.15
    product_cut_T: float = 623.15
    T_conv: float = 811.15
    kinetics: RDSKineticParams = field(default_factory=RDSKineticParams)
    kij: dict | None = None
    reactor: ReactorOptions = field(default_factory=lambda: ReactorOptions(rtol=1e-8, atol=1e-10))


#: Units of the scalar outputs of :class:`RDSResult`.
OUTPUT_UNITS: dict[str, str] = {
    "feed.rate": "kg/s", "feed.volume": "m3/s", "feed.S_wt": "wt%", "feed.NiV_wppm": "wppm",
    "feed.CCR_wt": "wt%", "feed.N_wppm": "wppm",
    "residue.rate": "kg/s", "residue.yield": "-", "residue.S_wt": "wt%", "residue.S_wppm": "wppm",
    "residue.NiV_wppm": "wppm", "residue.CCR_wt": "wt%", "residue.N_wppm": "wppm", "residue.sg": "-",
    "distillate.rate": "kg/s", "distillate.yield": "-", "distillate.S_wppm": "wppm", "distillate.sg": "-",
    "gas.rate": "kg/s", "hds.conversion": "-", "hdm.conversion": "-", "ccr.reduction": "-",
    "hdn.conversion": "-", "conversion": "-",
    "h2.chemical": "mol/s", "h2.chemical_nm3_m3": "Nm3/m3", "h2.chemical_wt": "wt% of feed",
    "h2.treat": "mol/s", "metals.deposit": "kg/s", "h2s.make": "mol/s",
    "wabt": "K", "reactor.T_out": "K", "reactor.dT_total": "K", "catalyst.mass": "kg",
}


@dataclass(frozen=True)
class RDSResult:
    """A solved residue desulfurizer.

    Attributes:
        outputs: ``{name: value}`` (see :data:`OUTPUT_UNITS`; also
            ``bed<k>.T_in``, ``bed<k>.dT``, ``bed<k>.quench``).
        streams: ``{name: Flows}``: ``feed``, ``treat_gas``, ``reactor_out``,
            ``gas``, ``distillate``, ``residue``.
        reactor: The :class:`~difflow_refinery.hydroprocessing.reactor.ReactorResult`.
        product_grid: Per-cut properties of the treated cuts (see :attr:`product_char`).
        balances: ``{"mass", "C", "H", "S", "N", "NiV"}`` relative errors across the unit.
        converged: Whether every bed integration finished.
        cut_names: The layout's cut names (static).
    """

    outputs: dict
    streams: dict
    reactor: Any
    product_grid: dict
    balances: dict
    converged: Array
    cut_names: tuple = ()

    @property
    def product_char(self) -> BlendCharacterization:
        """:class:`BlendCharacterization` of the treated cuts, for ``BlendComponent.from_stream``."""
        g = self.product_grid
        return BlendCharacterization(names=list(self.cut_names), Tb=g["Tb"], SG=g["SG"], MW=g["MW"],
                                     Tc=g["Tc"], Pc=g["Pc"], omega=g["omega"], qualities=dict(g["qualities"]))

    def product_stream(self, name: str = "residue", T=323.15, P=101325.0) -> dict:
        """``F_<cut>`` stream (mol/s) of ``"residue"`` or ``"distillate"`` on :attr:`product_char`."""
        f = self.streams[name]
        out = {f"F_{c}": f.cut[i] for i, c in enumerate(self.cut_names)}
        out["T"] = jnp.asarray(T, dtype=float)
        out["P"] = jnp.asarray(P, dtype=float)
        return out

    def blend_component(self, name: str = "residue", *, property_mode: bool = True,
                        viscosity_T_C: float = 50.0, **overrides) -> BlendComponent:
        """The product ``name`` as a fuel-oil :class:`~difflow_refinery.blending.BlendComponent`.

        SG, sulfur, nitrogen and CCR from the treated cuts; viscosity at
        ``viscosity_T_C`` estimated by :mod:`difflow_refinery.properties`
        (Abbott at 100/210 F, Walther between, from the treated cuts' TBP 50 %
        point and gravity -- marked unverified there). ``property_mode=True``
        (default) returns a property-mode component, which blends with
        cutters from any other characterization (a pool takes one mode).
        """
        c = BlendComponent.from_stream(name, self.product_stream(name), self.product_char,
                                       estimate=("viscosity_cSt",), viscosity_T_C=viscosity_T_C, **overrides)
        if not property_mode:
            return c
        keep = ("SG", "S_ppm", "N_ppm", "CCR_wt", "viscosity_cSt")
        return BlendComponent.from_properties(name, **{k: c.properties[k] for k in keep if k in c.properties})

    def volume(self, name: str = "residue") -> Array:
        """Standard volume flow (m^3/s at 15 C) of ``"residue"`` or ``"distillate"``."""
        g = self.product_grid
        f = self.streams[name]
        return jnp.sum(f.cut * g["MW"] * 1e-3 / (g["SG"] * RHO_WATER_15C))

    def table(self) -> str:
        """The headline outputs as text."""
        o = {k: float(v) for k, v in self.outputs.items() if np.ndim(v) == 0}
        return "\n".join([
            f"WABT {o['wabt'] - C_TO_K:.1f} C, total bed dT {o['reactor.dT_total']:.1f} K",
            f"feed: S {o['feed.S_wt']:.2f} wt%, Ni+V {o['feed.NiV_wppm']:.0f} wppm, CCR {o['feed.CCR_wt']:.1f} wt%",
            f"desulfurized residue: S {o['residue.S_wt']:.3f} wt%, Ni+V {o['residue.NiV_wppm']:.1f} wppm, "
            f"CCR {o['residue.CCR_wt']:.2f} wt%, SG {o['residue.sg']:.4f}, yield {100 * o['residue.yield']:.1f} %",
            f"HDS {100 * o['hds.conversion']:.1f} %, HDM {100 * o['hdm.conversion']:.1f} %, "
            f"CCR reduction {100 * o['ccr.reduction']:.1f} %, conversion {100 * o['conversion']:.1f} %",
            f"distillate {100 * o['distillate.yield']:.1f} % (S {o['distillate.S_wppm']:.0f} wppm), "
            f"H2 chemical {o['h2.chemical_nm3_m3']:.0f} Nm3/m3 ({o['h2.chemical_wt']:.2f} wt%)",
        ])


jax.tree_util.register_dataclass(RDSResult, data_fields=["outputs", "streams", "reactor", "product_grid",
                                                         "balances", "converged"], meta_fields=["cut_names"])


class ResidueDesulfurizer:
    """Residue desulfurizer on a characterization (see the module docstring).

    Args:
        char: The characterization, with a composition
            (``characterize(assay, composition=True)``). Concrete (it fixes
            the layout).
        feed: A concrete feed stream (``F_<char.names>``) used only to pick
            the cuts: every pseudo-component from the lightest one the
            conversion can reach up to the heaviest one with flow. Or give ``cuts``.
        params: :class:`RDSParams`.
        cuts: The cuts explicitly (lightest first), instead of ``feed``.
        residue_s_share: Refractory sulfur share table (see :mod:`.feed`).
        trace: Cuts carrying less than this fraction of the feed's
            pseudo-component moles do not count as "with flow" when picking.
    """

    def __init__(self, char, feed: Mapping | None = None, params: RDSParams | None = None,
                 cuts: Sequence[str] | None = None, residue_s_share=DEFAULT_RESIDUE_S_SHARE,
                 trace: float = 1e-9):
        if char.composition is None:
            raise ValueError("the characterization needs a composition: characterize(assay, composition=True)")
        self.char = char
        self.params = params or RDSParams()
        self.residue_s_share = residue_s_share
        p = self.params
        names = list(char.pseudo_names)
        k = len(char.light_names)
        comp = char.composition
        cpm_all = (np.asarray(char.component_MW)[k:] * np.asarray(comp.carbon)[k:] / ATOMIC_MASS["C"])
        Tb_all = np.asarray(char.Tb)
        if cuts is None:
            if feed is None:
                raise ValueError("give a feed (to pick the cuts) or cuts")
            fl = np.asarray([float(feed.get(f"F_{c}", 0.0)) for c in names])
            used = np.nonzero(fl > trace * fl.sum())[0]
            if used.size == 0:
                raise ValueError("the feed carries none of the characterization's pseudo-components")
            lo, hi = int(used.min()), int(used.max())
            tgt, _ = conversion_targets(cpm_all[: hi + 1], Tb_all[: hi + 1], p.T_conv)
            reach = [t for t, i in zip(tgt, range(hi + 1)) if t >= 0 and i >= lo]
            lo = min([lo] + reach)
            cuts = names[lo: hi + 1]
        self.cuts = tuple(cuts)
        self.layout = rds_layout(char, self.cuts)
        #: Mass fraction of ``feed`` on components the layout does not carry
        #: (0 when ``cuts`` were given or nothing was dropped).
        self.dropped_mass_fraction = 0.0
        if feed is not None:
            mw = dict(zip(char.names, np.asarray(char.component_MW)))
            mass = {nm: float(feed.get(f"F_{nm}", 0.0)) * mw[nm] for nm in char.names}
            kept = set(self.cuts) | set(self.layout.gases)
            tot = sum(mass.values())
            self.dropped_mass_fraction = sum(v for nm, v in mass.items() if nm not in kept) / tot if tot else 0.0
        idx = cut_indices(char, self.cuts) - k
        cpm = cpm_all[idx]
        ct, cm = conversion_targets(cpm, Tb_all[idx], p.T_conv)
        self.kinetics = RDSKinetics(self.layout, crack_targets(cpm), ct, cm)
        n_beds = len(p.bed_fractions)
        if p.quench is None and len(p.T_in) != n_beds:
            raise ValueError(f"T_in needs {n_beds} temperatures (one per bed) when quench is None")
        if p.quench is not None and len(p.quench) != n_beds - 1:
            raise ValueError(f"quench needs {n_beds - 1} fractions")
        self.reactor = TrickleBedReactor(self.layout, self.kinetics, p.reactor)
        self._distillate = jnp.asarray(Tb_all[idx] < p.product_cut_T, dtype=float)
        self._jit = jax.jit(self._run)

    # ----- inputs ------------------------------------------------------------

    def theta(self, feed: Mapping, char=None, params: RDSParams | None = None) -> dict:
        """The differentiable inputs of a solve, as a pytree."""
        char = self.char if char is None else char
        p = self.params if params is None else params
        if tuple(char.names) != tuple(self.char.names):
            raise ValueError("char must have the same components as the one the unit was built on")
        lay = self.layout
        idx = cut_indices(char, self.cuts) - len(char.light_names)
        num = {f: jnp.asarray(getattr(p, f), dtype=float)
               for f in ("P", "lhsv", "catalyst_density", "gas_oil", "T_gas")}
        num["T_in"] = jnp.asarray(p.T_in, dtype=float)
        num["quench"] = jnp.asarray(p.quench if p.quench is not None else (), dtype=float)
        num["bed_fractions"] = jnp.asarray(p.bed_fractions, dtype=float)
        unit = {f"F_{c}": 1.0 for c in self.cuts}
        return {
            "oil": rds_feed(char, feed, lay, self.residue_s_share),
            "unit_attr": rds_feed(char, unit, lay, self.residue_s_share).attr,
            "cut": {"Tb": char.Tb[idx], "SG": char.SG[idx], "MW": char.MW[idx], "Tc": char.Tc[idx],
                    "Pc": char.Pc[idx], "omega": char.omega[idx], "omega_vp": char.omega_vp[idx],
                    "hvap_nb": char.hvap_nb[idx], "cp_ig": char.cp_ig_coeffs[idx]},
            "treat_y": makeup_vector(lay, p.treat_gas),
            "num": num,
            "kin": p.kinetics,
        }

    # ----- the solve ------------------------------------------------------------

    def _run(self, th) -> RDSResult:
        p = self.params
        lay = self.layout
        n = th["num"]
        c = th["cut"]
        oil = th["oil"]
        comps = Components.build(lay, c["Tb"], c["SG"], c["MW"], c["Tc"], c["Pc"], c["omega"], c["hvap_nb"],
                                 c["cp_ig"], kij=p.kij)
        Q = jnp.sum(oil.cut_mass(lay) / (c["SG"] * RHO_WATER_60F))        # m^3/s at 60 F
        W_tot = Q * 3600.0 / n["lhsv"] * n["catalyst_density"]
        W = [W_tot * n["bed_fractions"][k] for k in range(len(p.bed_fractions))]
        gas_tot = n["gas_oil"] * Q * MOL_PER_NM3
        treat = Flows(th["treat_y"] * gas_tot, jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
        T_in = [n["T_in"][k] for k in range(n["T_in"].shape[0])]
        quench = None if p.quench is None else [n["quench"][k] for k in range(n["quench"].shape[0])]
        rx = self.reactor(oil, treat, T_in, n["T_gas"], n["P"], W, comps, th["kin"], quench=quench)
        out = rx.outlet
        # --- ideal product separation
        d = self._distillate
        gas = Flows(out.gas, jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
        dist = Flows(jnp.zeros(lay.n_gas), out.cut * d, out.attr * d[:, None])
        resid = Flows(jnp.zeros(lay.n_gas), out.cut * (1.0 - d), out.attr * (1.0 - d)[:, None])
        # --- treated-cut properties
        ai = {a: i for i, a in enumerate(lay.attributes)}
        pm = out.per_molecule()
        inc = jnp.zeros(lay.n_attr)
        for a, v in VOLUME_INCREMENTS.items():
            inc = inc.at[ai[a]].set(v)
        v_feed = c["MW"] / (1000.0 * c["SG"] * RHO_WATER_60F)
        v_cut = v_feed - th["unit_attr"] @ inc + pm @ inc
        has = out.cut > 1e-20 * jnp.sum(out.cut)
        mw_cut = jnp.where(has, out.cut_mw(lay), c["MW"])
        v_cut = jnp.where(has, v_cut, v_feed)
        sg_cut = mw_cut / (1000.0 * v_cut * RHO_WATER_60F)
        safe = lambda a: jnp.where(has, a, 0.0)
        s_cols = [ai[f"S_{k}"] for k in SULFUR_CLASSES] + [ai["S_residue"]]
        n_cols = [ai[f"N_{k}"] for k in NITROGEN_CLASSES]
        s_ppm = safe(1e6 * pm[:, s_cols].sum(1) * ATOMIC_MASS["S"] / mw_cut)
        n_ppm = safe(1e6 * pm[:, n_cols].sum(1) * ATOMIC_MASS["N"] / mw_cut)
        ccr_wt = safe(100.0 * pm[:, ai["CCR"]] * CCR_MW / mw_cut)
        arom = safe(100.0 * sum(pm[:, ai[f"A_{k}"]] for k in AROMATIC_CLASSES))
        olef = safe(100.0 * pm[:, ai["olefins"]])
        naph = safe(100.0 * pm[:, ai["naphthenes"]])
        grid = {"Tb": c["Tb"], "SG": sg_cut, "MW": mw_cut, "Tc": c["Tc"], "Pc": c["Pc"], "omega": c["omega_vp"],
                "qualities": {"S_ppm": s_ppm, "N_ppm": n_ppm, "CCR_wt": ccr_wt, "aromatics_vol": arom,
                              "olefins_vol": olef, "naphthenes_vol": naph,
                              "paraffins_vol": 100.0 - arom - olef - naph}}

        # --- outputs
        def S_mass(f):          # kg/s
            return jnp.sum(f.attr[:, s_cols]) * ATOMIC_MASS["S"] / 1000.0

        def N_mass(f):
            return jnp.sum(f.attr[:, n_cols]) * ATOMIC_MASS["N"] / 1000.0

        def M_mass(f):
            return jnp.sum(f.attribute(lay, "NiV")) * METAL_MW / 1000.0

        def CCR_mass(f):
            return jnp.sum(f.attribute(lay, "CCR")) * CCR_MW / 1000.0

        o: dict[str, Array] = {}
        m_feed = oil.mass(lay)
        m_feed_cuts = jnp.sum(oil.cut_mass(lay))
        o["feed.rate"] = m_feed
        o["feed.volume"] = Q
        o["feed.S_wt"] = 100.0 * S_mass(oil) / m_feed_cuts
        o["feed.N_wppm"] = 1e6 * N_mass(oil) / m_feed_cuts
        o["feed.NiV_wppm"] = 1e6 * M_mass(oil) / m_feed_cuts
        o["feed.CCR_wt"] = 100.0 * CCR_mass(oil) / m_feed_cuts
        for name, f in (("residue", resid), ("distillate", dist)):
            m = jnp.sum(f.cut_mass(lay))
            ms = jnp.maximum(m, 1e-300)
            o[f"{name}.rate"] = m
            o[f"{name}.yield"] = m / m_feed
            o[f"{name}.S_wppm"] = 1e6 * S_mass(f) / ms
            o[f"{name}.N_wppm"] = 1e6 * N_mass(f) / ms
            o[f"{name}.NiV_wppm"] = 1e6 * M_mass(f) / ms
            o[f"{name}.CCR_wt"] = 100.0 * CCR_mass(f) / ms
            vol = f.cut * mw_cut * 1e-3 / (sg_cut * RHO_WATER_15C)
            o[f"{name}.sg"] = jnp.sum(f.cut * mw_cut * 1e-3) / RHO_WATER_15C / jnp.maximum(jnp.sum(vol), 1e-300)
        o["residue.S_wt"] = o["residue.S_wppm"] / 1e4
        o["gas.rate"] = gas.mass(lay)
        liq_out = resid + dist
        o["hds.conversion"] = 1.0 - S_mass(liq_out) / S_mass(oil)
        def removed(f_in, f_out):       # 0 when the feed carries none
            return jnp.where(f_in > 0.0, 1.0 - f_out / jnp.where(f_in > 0.0, f_in, 1.0), 0.0)

        o["hdn.conversion"] = removed(N_mass(oil), N_mass(liq_out))
        o["hdm.conversion"] = removed(M_mass(oil), M_mass(liq_out))
        o["ccr.reduction"] = removed(CCR_mass(oil), CCR_mass(liq_out))
        heavy = jnp.asarray(np.asarray(self.kinetics.conv_targets) >= 0, dtype=float)
        o["conversion"] = 1.0 - jnp.sum(out.cut_mass(lay) * heavy) / jnp.maximum(
            jnp.sum(oil.cut_mass(lay) * heavy), 1e-300)
        o["metals.deposit"] = M_mass(oil) - M_mass(out)
        hi, si = lay.gas_index("hydrogen"), lay.gas_index("hydrogen_sulfide")
        o["h2.treat"] = treat.gas[hi]
        o["h2.chemical"] = treat.gas[hi] + oil.gas[hi] - out.gas[hi]
        o["h2.chemical_nm3_m3"] = o["h2.chemical"] / MOL_PER_NM3 / Q
        o["h2.chemical_wt"] = 100.0 * o["h2.chemical"] * 2.0 * ATOMIC_MASS["H"] / 1000.0 / m_feed
        o["h2s.make"] = out.gas[si] - oil.gas[si] - treat.gas[si]
        o["wabt"] = rx.wabt
        o["reactor.T_out"] = rx.T_out
        o["reactor.dT_total"] = jnp.sum(rx.delta_T)
        o["catalyst.mass"] = W_tot
        for k, b in enumerate(rx.beds):
            o[f"bed{k + 1}.T_in"] = b.T_in
            o[f"bed{k + 1}.dT"] = b.T_out - b.T_in
            o[f"bed{k + 1}.quench"] = rx.quench[k]
        # --- balances across the unit (in: oil + treat gas; out: gas + distillate + residue + deposit)
        ins = oil + treat
        outs = gas + dist + resid
        bal = {"mass": relative_balance_error(ins.mass(lay), outs.mass(lay))}
        e_in, e_out = ins.elements(lay), outs.elements(lay)
        for i, e in enumerate(ELEMENTS):
            bal[e] = relative_balance_error(e_in[i], e_out[i])
        bal["NiV"] = relative_balance_error(M_mass(ins), M_mass(outs) + o["metals.deposit"])
        steps_ok = jnp.all(jnp.stack([b.steps < p.reactor.max_steps for b in rx.beds]))
        finite = jnp.all(jnp.isfinite(outs.ravel()))
        return RDSResult(outputs=o, streams={"feed": oil, "treat_gas": treat, "reactor_out": out, "gas": gas,
                                             "distillate": dist, "residue": resid},
                         reactor=rx, product_grid=grid, balances=bal, converged=steps_ok & finite,
                         cut_names=self.cuts)

    def solve(self, feed: Mapping, char=None, params: RDSParams | None = None, jit: bool = True) -> RDSResult:
        """Solve the unit for ``feed`` (``F_<char.names>``, mol/s).

        ``params`` may differ from the construction's in numbers (traced
        values included) but not in structure (bed count, quench mode,
        ``T_conv``/``product_cut_T`` which fix the layout's static maps).
        """
        p = self.params if params is None else params
        if params is not None:
            for f in ("bed_fractions", "T_in"):
                if len(getattr(params, f)) != len(getattr(self.params, f)):
                    raise ValueError(f"{f} must keep its length ({len(getattr(self.params, f))})")
            if (params.quench is None) != (self.params.quench is None):
                raise ValueError("quench mode (given or solved) is fixed at construction")
            for f in ("T_conv", "product_cut_T", "kij"):
                if getattr(params, f) is not getattr(self.params, f) and getattr(params, f) != getattr(self.params, f):
                    raise ValueError(f"{f} is fixed at construction")
        th = self.theta(feed, char, p)
        return self._jit(th) if jit else self._run(th)


__all__ = ["VOLUME_INCREMENTS", "RDSParams", "OUTPUT_UNITS", "RDSResult", "ResidueDesulfurizer"]
