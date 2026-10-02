"""The hydrotreater: reactor, HP separator, recycle-gas loop, stripper, products.

:class:`Hydrotreater` assembles the shared hydroprocessing pieces
(:mod:`difflow_refinery.hydroprocessing`) around the hydrotreating kinetics::

    feed oil ---+--> [bed 1] --quench--> [bed 2] ... --> effluent cooler --> HPS
                |                                                          |  |
    treat gas --+    makeup H2 --+                                  vapour |  | liquid
         ^                       |                                         v  |
         +------ compressor <----+---- purge <---- amine <---- KO drum ----+  |
                                                                              v
                       off-gas, wild naphtha, sour water <--- drum <--- steam stripper
                                                                              |
                                                                     treated product

Degrees of freedom (:class:`HydrotreaterParams`): bed inlet temperatures (or
the first one and the quench rates), reactor pressure, LHSV, treat-gas
H2/oil ratio, purge fraction, makeup-gas composition, amine H2S removal,
separator temperature, stripper feed temperature, steam and drum
temperature; and the kinetic parameters, catalyst activity among them.
A WABT or a product-sulfur target is a :class:`TargetSpec` that replaces the
inlet temperatures (all shifted together) -- a scalar Newton solve around
the whole unit, differentiated implicitly.

Everything is differentiable -- with respect to every parameter, the feed
rate and the characterization (so a TBP point of the assay) -- through
implicit-function gradients of the recycle tear, the flashes and the
stripper, and the discrete adjoint of the bed integrations.
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
from difflow_refinery.assay import LIGHT_END_SG
from difflow_refinery.blending import cetane_index_d4737, tbp_temperature, tbp_to_d86
from difflow_refinery.characterization import RHO_WATER_15C, BlendCharacterization
from difflow_refinery.composition import NITROGEN_CLASSES, SULFUR_CLASSES
from difflow_refinery.hydroprocessing.layout import ELEMENTS, Flows, Layout, relative_balance_error
from difflow_refinery.hydroprocessing.layout import gas_mw
from difflow_refinery.hydroprocessing.reactor import (
    ReactorOptions, TrickleBedReactor, k_model_at, stream_enthalpy, vapor_enthalpy)
from difflow_refinery.hydroprocessing.recycle import (
    MOL_PER_NM3, amine_scrub, compress, knockout, makeup_for_ratio, makeup_vector, purge_split, solve_tear)
from difflow_refinery.hydroprocessing.separator import HPSeparator, flash_components
from difflow_refinery.hydroprocessing.solve import newton_scalar
from difflow_refinery.hydroprocessing.stripper import (
    StripperSpec, cut_pseudo_components, overhead_drum, strip)
from difflow_refinery.hydroprocessing.thermo import Components
from difflow_refinery.hydrotreating.feed import (
    DEFAULT_AROMATIC_SPLIT, cut_indices, hdt_feed, hdt_layout)
from difflow_refinery.hydrotreating.kinetics import (
    AROMATIC_CLASSES, HDTKineticParams, HDTKinetics, crack_targets)
from difflow_refinery.thermo import LIGHT_ENDS, RHO_WATER_60F, ColumnThermo

jax.config.update("jax_enable_x64", True)

C_TO_K = 273.15
BARREL = 0.158987294928
SCF_PER_NM3 = 37.326  # standard ft^3 (60 F, 14.696 psia) per normal m^3 (0 C, 1 atm), ideal gas

#: Liquid molar-volume change (m^3/mol) when one molecule of each type is
#: converted to the next more saturated one, used to compute the treated
#: product's gravity. From liquid molar volumes at 60 F of model compounds
#: (DIPPR 105 correlations of Perry's 8th ed. as tabulated in the
#: ``chemicals`` package): mono-aromatic -> naphthene the mean of benzene ->
#: cyclohexane (+19.2), toluene -> methylcyclohexane (+21.3) and
#: ethylbenzene -> ethylcyclohexane (+19.9 cm^3/mol); di -> mono naphthalene
#: (subcooled) -> tetralin (+11.0); poly -> di taken equal; olefin ->
#: paraffin the mean of 1-hexene -> n-hexane (+5.6) and 1-decene -> n-decane
#: (+5.4). Heteroatom removal is taken to change the volume by nothing.
VOLUME_INCREMENTS: dict[str, float] = {
    "A_mono": -20.1e-6, "A_di": -31.1e-6, "A_poly": -42.1e-6, "olefins": -5.5e-6, "naphthenes": 0.0,
}


#: Largest PR-flash residual (in ln K) a converged solve accepts.
FLASH_TOL = 1e-8

#: Outlets that carry no real gas by construction: the stripper sends every
#: real species (H2, H2S, NH3, C1-C6) overhead, so its bottoms has none.
_GAS_FREE_STREAMS: tuple[str, ...] = ("product",)


class HydrotreaterConvergenceWarning(UserWarning):
    """The recycle loop, a bed or the stripper did not converge."""


@dataclass
class HydrotreaterParams(ParamsMixin):
    """Operating specs of :class:`Hydrotreater`.

    Attributes:
        T_in: Bed inlet temperatures (K): one per bed when the quench is
            solved for, else only the first bed's.
        quench: ``None`` (solve each later bed's quench for its ``T_in``),
            or ``n_beds - 1`` quench rates as fractions of the treat gas.
        bed_fractions: Share of the catalyst in each bed (sums to 1).
        P: Reactor pressure (Pa).
        lhsv: Liquid hourly space velocity, 1/h: feed standard liquid volume
            (60 F) per hour per volume of catalyst.
        catalyst_density: Catalyst bulk (loaded) density, kg/m^3.
        h2_oil: Treat-gas H2 to oil ratio, Nm^3 H2 (0 C, 1 atm) per m^3 of
            feed (60 F), at the reactor inlet including quench.
        purge: Fraction of the scrubbed separator gas purged.
        makeup: Makeup-gas composition, ``{gas: mole fraction}``.
        h2s_removal: Fraction of the recycle gas's H2S the amine removes.
        nh3_removal: Fraction of its NH3 removed (wash water / amine).
        hps_T: High-pressure separator temperature (K).
        loop_dP: Pressure drop from reactor inlet to separator (Pa); the
            recycle compressor makes it up.
        compressor_eta: Recycle compressor isentropic efficiency.
        stripper_feed_T: Stripper feed temperature after its heater (K).
        stripper_P: Stripper top pressure (Pa).
        stripper_dP: Stripper pressure rise per stage (Pa).
        steam_ratio: Stripping steam, kg per kg of stripper feed.
        steam_T: Stripping steam temperature (K).
        drum_T: Stripper overhead drum temperature (K).
        drum_dP: Overhead drum pressure below the stripper top (Pa).
        stripper_stages: Equilibrium stages in the stripper.
        kinetics: :class:`HDTKineticParams`.
        kij: PR binary interaction parameters (dict of name pairs), or None
            for the defaults.
        reactor: :class:`ReactorOptions`.
        tear_tol: Recycle-tear convergence tolerance (scaled residual).
        heater_efficiency: Charge-heater thermal efficiency, absorbed over
            fired duty. ILLUSTRATIVE default 0.85 (a typical fired process
            heater with some heat recovery; not from a cited source).
        heater_inlet_T: Temperature (K) of the combined oil and treat gas
            entering the charge heater -- the feed/effluent exchanger's
            cold-side outlet. ``None`` (default): no exchanger, the oil enters
            at its feed temperature and the treat gas at the compressor
            discharge temperature, so the heater does all the heating.
    """

    T_in: tuple = (613.15,)
    quench: tuple | None = (0.15,)
    bed_fractions: tuple = (0.4, 0.6)
    P: float = 50e5
    lhsv: float = 1.0
    catalyst_density: float = 800.0
    h2_oil: float = 300.0
    purge: float = 0.05
    makeup: dict = field(default_factory=lambda: {"hydrogen": 0.97, "methane": 0.03})
    h2s_removal: float = 0.99
    nh3_removal: float = 1.0
    hps_T: float = 323.15
    loop_dP: float = 3e5
    compressor_eta: float = 0.75
    stripper_feed_T: float = 503.15
    stripper_P: float = 7e5
    stripper_dP: float = 1e3
    steam_ratio: float = 0.01
    steam_T: float = 523.15
    drum_T: float = 313.15
    drum_dP: float = 0.3e5
    stripper_stages: int = 6
    kinetics: HDTKineticParams = field(default_factory=HDTKineticParams)
    kij: dict | None = None
    reactor: ReactorOptions = field(default_factory=ReactorOptions)
    tear_tol: float = 1e-11
    heater_efficiency: float = 0.85
    heater_inlet_T: float | None = None


@dataclass(frozen=True)
class TargetSpec:
    """Hit ``output == target`` by shifting every bed inlet temperature by the same amount.

    ``output`` is a key of :attr:`HydrotreaterResult.outputs`, e.g.
    ``"wabt"`` (K) or ``"product.S_wppm"``. ``scale`` normalises the residual.
    """

    output: str
    target: float
    scale: float = 1.0


# =============================================================================
# Result
# =============================================================================


@dataclass(frozen=True)
class HydrotreaterResult:
    """A solved hydrotreater.

    Attributes:
        outputs: ``{name: value}`` of every reported quantity (see
            :data:`OUTPUT_UNITS` for names and units).
        streams: ``{name: Flows}``: ``feed``, ``makeup``, ``treat_gas``,
            ``reactor_out``, ``hps_vapor``, ``hps_liquid``, ``hps_water``,
            ``acid_gas``, ``purge``, ``recycle``, ``stripper_overhead``,
            ``product``, ``wild_naphtha``, ``off_gas``, ``sour_water``,
            ``steam``.
        reactor: The :class:`~difflow_refinery.hydroprocessing.reactor.ReactorResult`.
        product_grid: Per-component properties of the treated product grid
            (light ends then cuts; shared by product and wild naphtha); see
            :attr:`product_char`.
        balances: ``{"mass", "C", "H", "S", "N": relative error}`` across the unit.
        converged: Whether the tear, the beds and the stripper converged.
    """

    outputs: dict
    streams: dict
    reactor: Any
    product_grid: Any
    balances: dict
    converged: Array
    product_names: tuple = ()
    gas_names: tuple = ()

    @property
    def product_char(self) -> BlendCharacterization:
        """:class:`BlendCharacterization` of the treated product grid, for ``BlendComponent.from_stream``."""
        g = self.product_grid
        return BlendCharacterization(names=list(self.product_names), Tb=g.Tb, SG=g.SG, MW=g.MW, Tc=g.Tc,
                                     Pc=g.Pc, omega=g.omega, qualities=dict(g.qualities))

    def product_stream(self, name: str = "product", T=298.15, P=101325.0, gases: bool = True) -> dict:
        """A liquid outlet as an ``F_<name>`` stream, mol/s (for ``BlendComponent.from_stream``).

        The stream carries the components of :attr:`product_char` (the liquid
        light ends and the treated cuts) and, with ``gases=True`` (default), the
        real gases dissolved in it that are not on that grid -- H2, H2S, NH3,
        methane and ethane, under their own ``F_<gas>`` keys -- so that a
        downstream mass balance on it closes to round-off (:meth:`stream_mass`;
        #333). The stripper bottoms (``"product"``) carries none by
        construction (the stripper sends every real gas overhead), so its
        stream is the blend grid only either way. The wild naphtha does carry
        them: ``BlendComponent.from_stream`` refuses species outside its grid,
        so blend it with ``gases=False``, which leaves them out --
        ``outputs["naphtha.dissolved_gas_rate"]`` is the mass that drops (or
        fractionate it, :meth:`fractionate`, which sends them to an off-gas).
        Water is never in these streams (the drum decants all of it).
        """
        f = self.streams[name]
        names = list(self.product_names)
        n_light = len(names) - f.cut.shape[0]
        gi = {g: i for i, g in enumerate(self.gas_names)}
        out = {}
        for i in range(n_light):
            out[f"F_{names[i]}"] = f.gas[gi[names[i]]]
        for i, c in enumerate(names[n_light:]):
            out[f"F_{c}"] = f.cut[i]
        if gases and name not in _GAS_FREE_STREAMS:
            for g, i in gi.items():
                if g not in names and g != "water":
                    out[f"F_{g}"] = f.gas[i]
        out["T"] = jnp.asarray(T)
        out["P"] = jnp.asarray(P)
        return out

    def stream_mass(self, stream: Mapping) -> Array:
        """Mass flow (kg/s) of a stream from :meth:`product_stream` or :meth:`fractionate`.

        Components of :attr:`product_char` at its molar masses, real gases at
        theirs (the atomic weights every balance of the unit closes on).
        """
        g = self.product_grid
        mw = dict(zip(self.product_names, [g.MW[i] for i in range(len(self.product_names))]))
        m = jnp.asarray(0.0)
        for k, v in stream.items():
            if not k.startswith("F_"):
                continue
            n = k[2:]
            m = m + jnp.asarray(v) * (mw[n] if n in mw else gas_mw(n))
        return m / 1000.0

    def fractionate(self, cut_points=None, products=None, width=None, feeds=("product",), T=298.15,
                    P=101325.0):
        """Split the liquid product(s) at TBP cut points (#328); see :func:`.fractionator.fractionate`."""
        from difflow_refinery.hydrotreating import fractionator as fr
        return fr.fractionate(self, fr.DEFAULT_CUT_POINTS if cut_points is None else cut_points,
                              fr.DEFAULT_PRODUCTS if products is None else products,
                              fr.DEFAULT_WIDTH if width is None else width, feeds=feeds, T=T, P=P)

    def table(self) -> str:
        """The headline outputs as text."""
        o = {k: float(v) for k, v in self.outputs.items() if np.ndim(v) == 0}
        rows = [
            f"WABT {o['wabt'] - C_TO_K:.1f} C, bed dT " + ", ".join(
                f"{float(t):.1f}" for t in np.asarray(self.reactor.delta_T)) + " K",
            f"product: S {o['product.S_wppm']:.1f} wppm, N {o['product.N_wppm']:.1f} wppm, "
            f"SG {o['product.sg']:.4f}, aromatics {o['product.aromatics_vol']:.1f} vol%, "
            f"cetane index {o['product.cetane_index']:.1f}",
            f"yields (mass): product {100 * o['product.yield']:.2f} %, wild naphtha "
            f"{100 * o['naphtha.yield']:.2f} %, gas {100 * o['gas.yield']:.3f} %",
            f"H2: chemical {o['h2.chemical_nm3_m3']:.1f} Nm3/m3 ({o['h2.chemical_scf_bbl']:.0f} scf/bbl), "
            f"makeup {o['h2.makeup_nm3_m3']:.1f} Nm3/m3, purity {100 * o['recycle.h2_purity']:.1f} %",
            f"recycle compressor {o['compressor.power'] / 1e3:.1f} kW, purge {o['purge.rate']:.3f} mol/s",
        ]
        return "\n".join(rows)


jax.tree_util.register_dataclass(HydrotreaterResult, data_fields=["outputs", "streams", "reactor",
                                                                  "product_grid", "balances", "converged"],
                                 meta_fields=["product_names", "gas_names"])


@dataclass(frozen=True)
class _ProductGrid:
    """Arrays of the treated-product grid (a pytree; see :attr:`HydrotreaterResult.product_char`)."""

    Tb: Array
    SG: Array
    MW: Array
    Tc: Array
    Pc: Array
    omega: Array
    qualities: dict


jax.tree_util.register_dataclass(_ProductGrid, data_fields=["Tb", "SG", "MW", "Tc", "Pc", "omega", "qualities"],
                                 meta_fields=[])

#: Units of the scalar outputs.
OUTPUT_UNITS: dict[str, str] = {
    "wabt": "K", "reactor.T_out": "K", "reactor.dT_total": "K",
    "product.S_wppm": "wppm", "product.N_wppm": "wppm", "product.sg": "-", "product.api": "API",
    "product.H_wt": "wt%", "product.aromatics_vol": "vol%", "product.mono_aromatics_vol": "vol%",
    "product.di_aromatics_vol": "vol%", "product.poly_aromatics_vol": "vol%", "product.olefins_vol": "vol%",
    "product.T10": "K", "product.T50": "K", "product.T90": "K", "product.T95": "K",
    "product.cetane_index": "-", "product.rate": "kg/s", "product.yield": "-", "product.volume_yield": "-",
    "naphtha.rate": "kg/s", "naphtha.yield": "-", "gas.yield": "-",
    "h2.chemical": "mol/s", "h2.chemical_nm3_m3": "Nm3/m3", "h2.chemical_scf_bbl": "scf/bbl",
    "h2.chemical_wt": "wt% of feed", "h2.makeup": "mol/s", "h2.makeup_nm3_m3": "Nm3/m3",
    "h2.dissolved": "mol/s", "h2.purge": "mol/s", "h2.reaction_sum": "mol/s",
    "recycle.h2_purity": "-", "recycle.rate": "mol/s", "purge.rate": "mol/s",
    "reactor.pH2_in": "Pa", "compressor.power": "W", "compressor.T_out": "K",
    "feed.rate": "kg/s", "feed.volume": "m3/s", "feed.S_wppm": "wppm", "feed.N_wppm": "wppm",
    "catalyst.mass": "kg", "hds.conversion": "-", "hdn.conversion": "-",
    "tear.residual": "-", "stripper.residual": "-", "flash.residual": "-",
    "yields.total": "-", "feed.h2_water_rate": "kg/s",
    "product.dissolved_gas_rate": "kg/s", "naphtha.dissolved_gas_rate": "kg/s",
    "heater.duty": "W", "heater.fired_duty": "W", "heater.inlet_T": "K", "feed_effluent.duty": "W",
}


# =============================================================================
# The unit
# =============================================================================


class Hydrotreater:
    """Hydrotreater on a characterization (see the module docstring).

    Args:
        char: The :class:`~difflow_refinery.assay.Characterization`, carrying
            a composition (``characterize(assay, composition=True)``). Concrete
            (it fixes the layout); a traced one with the same names may be
            passed to :meth:`solve`.
        feed: A concrete feed stream (``F_<char.names>``), used only to pick
            the cuts the unit carries: every cut with flow, plus every
            lighter pseudo-component (the cracking leak's products). Or give
            ``cuts``.
        params: :class:`HydrotreaterParams`.
        cuts: The cuts explicitly (lightest first), instead of ``feed``.
        target: Optional :class:`TargetSpec` replacing the inlet temperatures.
        aromatic_split: Mono/di/poly split table (see :mod:`.feed`).
        trace: Cuts heavier than the heaviest one carrying more than this
            fraction of the feed's pseudo-component moles are left out of the
            layout (a column product carries every cut at some 1e-20 level).
            What that drops is :attr:`dropped_mass_fraction`.
    """

    def __init__(self, char, feed: Mapping | None = None, params: HydrotreaterParams | None = None,
                 cuts: Sequence[str] | None = None, target: TargetSpec | None = None,
                 aromatic_split=DEFAULT_AROMATIC_SPLIT, trace: float = 1e-9):
        if char.composition is None:
            raise ValueError("the characterization needs a composition: characterize(assay, composition=True)")
        self.char = char
        self.params = params or HydrotreaterParams()
        self.target = target
        self.aromatic_split = aromatic_split
        if cuts is None:
            if feed is None:
                raise ValueError("give a feed (to pick the cuts) or cuts")
            fl = np.asarray([float(feed.get(f"F_{c}", 0.0)) for c in char.pseudo_names])
            used = np.nonzero(fl > trace * fl.sum())[0]
            if used.size == 0:
                raise ValueError("the feed carries none of the characterization's pseudo-components")
            cuts = char.pseudo_names[: int(used.max()) + 1]
        self.cuts = tuple(cuts)
        self.layout = hdt_layout(char, self.cuts)
        #: Mass fraction of ``feed`` on components the layout does not carry
        #: (0 when ``cuts`` were given or nothing was dropped).
        self.dropped_mass_fraction = 0.0
        if feed is not None:
            mw = dict(zip(char.names, np.asarray(char.component_MW)))
            m = {n: float(feed.get(f"F_{n}", 0.0)) * mw[n] for n in char.names}
            kept = set(self.cuts) | set(self.layout.gases)
            tot = sum(m.values())
            self.dropped_mass_fraction = sum(v for n, v in m.items() if n not in kept) / tot if tot else 0.0
        idx = cut_indices(char, self.cuts)
        # cracking targets from the (concrete) carbon number per molecule
        comp = char.composition
        cpm = np.asarray(char.component_MW)[idx] * np.asarray(comp.carbon)[idx] / 12.0107
        self.kinetics = HDTKinetics(self.layout, crack_targets(cpm))
        p = self.params
        n_beds = len(p.bed_fractions)
        if p.quench is None and len(p.T_in) != n_beds:
            raise ValueError(f"T_in needs {n_beds} temperatures (one per bed) when quench is None")
        if p.quench is not None and len(p.quench) != n_beds - 1:
            raise ValueError(f"quench needs {n_beds - 1} fractions")
        self.reactor = TrickleBedReactor(self.layout, self.kinetics, p.reactor)
        self.stripper = StripperSpec(n_stages=int(p.stripper_stages))
        self._jit = jax.jit(self._run)

    # ----- inputs ------------------------------------------------------------

    def theta(self, feed: Mapping, char=None, params: HydrotreaterParams | None = None) -> dict:
        """The differentiable inputs of a solve, as a pytree."""
        char = self.char if char is None else char
        p = self.params if params is None else params
        if tuple(char.names) != tuple(self.char.names):
            raise ValueError("char must have the same components as the one the unit was built on")
        lay = self.layout
        oil = hdt_feed(char, feed, lay, self.aromatic_split)
        idx = jnp.asarray(cut_indices(char, self.cuts))
        k = len(char.light_names)
        pidx = idx - k
        num = {f.name: jnp.asarray(getattr(p, f.name), dtype=float) for f in dataclasses.fields(p)
               if f.name not in ("T_in", "quench", "bed_fractions", "makeup", "kinetics", "kij", "reactor",
                                 "stripper_stages", "tear_tol", "heater_inlet_T")}
        num["heater_inlet_T"] = jnp.asarray(0.0 if p.heater_inlet_T is None else p.heater_inlet_T, dtype=float)
        num["T_in"] = jnp.asarray(p.T_in, dtype=float)
        num["quench"] = jnp.asarray(p.quench if p.quench is not None else (), dtype=float)
        num["bed_fractions"] = jnp.asarray(p.bed_fractions, dtype=float)
        kin = {f.name: (getattr(p.kinetics, f.name) if f.name == "hds_form"
                        else jnp.asarray(getattr(p.kinetics, f.name), dtype=float))
               for f in dataclasses.fields(p.kinetics)}
        kin.pop("hds_form")
        light = [n for n in char.light_names]
        return {
            "oil": oil,
            "feed_T": jnp.asarray(feed.get("T", 298.15), dtype=float),
            "cut": {
                "Tb": char.Tb[pidx], "SG": char.SG[pidx], "MW": char.MW[pidx], "Tc": char.Tc[pidx],
                "Pc": char.Pc[pidx], "omega": char.omega[pidx], "hvap_nb": char.hvap_nb[pidx],
                "cp_ig": char.cp_ig_coeffs[pidx], "T_lo": char.cut_edges[:-1][pidx],
                "T_hi": char.cut_edges[1:][pidx],
            },
            "unit_attr": hdt_feed(char, {f"F_{c}": 1.0 for c in self.cuts}, lay, self.aromatic_split).attr,
            "makeup_y": makeup_vector(lay, p.makeup),
            "num": num,
            "kin": kin,
            "light_feed": jnp.stack([jnp.asarray(feed.get(f"F_{n}", 0.0), dtype=float) for n in light])
            if light else jnp.zeros(0),
        }

    # ----- the solve ---------------------------------------------------------

    def _kin_params(self, kin) -> HDTKineticParams:
        d = {k: (tuple(v[i] for i in range(v.shape[0])) if hasattr(v, "shape") and v.ndim else v)
             for k, v in kin.items()}
        return HDTKineticParams(hds_form=self.params.kinetics.hds_form, **d)

    def _comps(self, th) -> Components:
        c = th["cut"]
        return Components.build(self.layout, c["Tb"], c["SG"], c["MW"], c["Tc"], c["Pc"], c["omega"],
                                c["hvap_nb"], c["cp_ig"], kij=self.params.kij)

    def _evaluate(self, x, A, adjoint=None):
        """One pass round the loop from recycle gas ``x`` (gas flows, then the compressor outlet T).

        A pure function of ``(x, A)``, ``A = (th, comps, kin, T_shift)``.
        Returns the next ``x`` and everything the pass computed.
        """
        th, comps, kin, T_shift = A
        lay = self.layout
        n = th["num"]
        oil = th["oil"]
        Q = self._feed_volume(th)
        W_tot = Q * 3600.0 / n["lhsv"] * n["catalyst_density"]
        W = [W_tot * n["bed_fractions"][k] for k in range(len(self.params.bed_fractions))]
        h2_target = n["h2_oil"] * Q * MOL_PER_NM3
        P_hps = n["P"] - n["loop_dP"]
        T_in = [n["T_in"][k] + T_shift for k in range(n["T_in"].shape[0])]
        quench = None if self.params.quench is None else [n["quench"][k] for k in range(n["quench"].shape[0])]
        R = Flows(jnp.maximum(x[:-1], 0.0), jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
        T_gas = x[-1]
        makeup, M = makeup_for_ratio(R, lay, th["makeup_y"], h2_target)
        gas = R + makeup
        rx = self.reactor(oil, gas, T_in, T_gas, n["P"], W, comps, kin, quench=quench, adjoint=adjoint)
        vap, liq, water, fr = HPSeparator(lay)(rx.outlet, n["hps_T"], P_hps, comps)
        vap, liq = knockout(vap, liq)
        sweet, absorbed = amine_scrub(vap, lay, n["h2s_removal"], n["nh3_removal"])
        rec, purge = purge_split(sweet, n["purge"])
        T2, Wc = compress(rec, lay, comps, n["hps_T"], P_hps, n["P"], n["compressor_eta"])
        out = dict(R=R, makeup=makeup, M=M, gas=gas, rx=rx, vap=vap, liq=liq, water=water, fr=fr,
                   absorbed=absorbed, rec=rec, purge=purge, T2=T2, Wc=Wc, W=W, Q=Q,
                   h2_target=h2_target, P_hps=P_hps)
        return jnp.concatenate([rec.gas, T2[None]]), out

    def _loop(self, th, comps, kin, T_shift):
        """Converge the recycle tear; returns the converged pass's results (with ``"tear"``)."""
        lay = self.layout
        n = th["num"]
        A = (th, comps, kin, T_shift)
        h2_target = jax.lax.stop_gradient(n["h2_oil"] * self._feed_volume(th) * MOL_PER_NM3)
        hi, mi = lay.gas_index("hydrogen"), lay.gas_index("methane")
        x0 = jnp.concatenate([(jnp.zeros(lay.n_gas) + 1e-4 * h2_target).at[hi].set(0.85 * h2_target)
                              .at[mi].set(0.08 * h2_target),
                              jax.lax.stop_gradient(n["hps_T"] + 25.0)[None]])
        scale = jnp.concatenate([jnp.full(lay.n_gas, 1e-2 * h2_target).at[hi].set(h2_target),
                                 jnp.asarray([100.0])])
        sol = solve_tear(lambda x, A: self._evaluate(x, A)[0], x0, A, scale=scale,
                         tol=self.params.tear_tol, max_step=2.0,
                         g_iter=lambda x, A: self._evaluate(x, A, adjoint="forward")[0], jac="fwd")
        _, out = self._evaluate(sol.value, A)
        out["tear"] = sol
        return out

    def _feed_volume(self, th) -> Array:
        """Feed standard liquid volume (m^3/s at 60 F), cuts and liquid light ends."""
        c = th["cut"]
        oil = th["oil"]
        m = oil.cut_mass(self.layout)
        V = jnp.sum(m / (c["SG"] * RHO_WATER_60F))
        for i, nme in enumerate(self.char.light_names):
            if nme in LIGHT_END_SG:
                V = V + th["light_feed"][i] * LIGHT_ENDS[nme][0] / 1000.0 / (LIGHT_END_SG[nme] * RHO_WATER_60F)
        return V

    def _run(self, th):
        p = self.params
        lay = self.layout
        n = th["num"]
        comps = self._comps(th)
        kin = self._kin_params(th["kin"])
        if self.target is None:
            out = self._loop(th, comps, kin, jnp.asarray(0.0))
            shift = jnp.asarray(0.0)
        else:
            tgt = self.target

            def res(sh, A):
                th, comps, kin = A
                o = self._loop(th, comps, kin, sh)
                return (self._outputs(th, comps, o)[0][tgt.output] - tgt.target) / tgt.scale

            sol = newton_scalar(res, 0.0, (th, comps, kin), tol=1e-9, max_step=30.0)
            shift = sol.value
            out = self._loop(th, comps, kin, shift)
        outputs, streams, pchar, balances, conv = self._outputs(th, comps, out, full=True)
        outputs["T_shift"] = shift
        return HydrotreaterResult(outputs=outputs, streams=streams, reactor=out["rx"], product_grid=pchar,
                                  balances=balances, converged=conv,
                                  product_names=tuple(self._product_lights()) + tuple(self.cuts),
                                  gas_names=tuple(lay.gases))

    def _outputs(self, th, comps, o, full=False):
        """Stripper, products, outputs and balances from a converged loop evaluation."""
        p = self.params
        lay = self.layout
        n = th["num"]
        c = th["cut"]
        oil = th["oil"]
        liq = o["liq"]
        # --- stripper and drum
        pc = cut_pseudo_components(lay, liq, c["Tb"], c["SG"], c["MW"], c["Tc"], c["Pc"], c["omega"],
                                   c["T_lo"], c["T_hi"])
        steam = n["steam_ratio"] * liq.mass(lay)
        st = strip(liq, lay, pc, n["stripper_feed_T"], n["stripper_P"], n["stripper_dP"], steam, n["steam_T"],
                   self.stripper, feed_inlet_T=n["hps_T"])
        off, naph, sour, frd = overhead_drum(st.overhead, lay, comps, n["drum_T"], n["stripper_P"] - n["drum_dP"])
        prod = st.bottoms
        # --- treated-cut properties (the reactor outlet's per-molecule attributes)
        rx = o["rx"]
        pm = rx.outlet.per_molecule()
        ai = {a: i for i, a in enumerate(lay.attributes)}
        unit_attr = th["unit_attr"]
        inc = jnp.zeros(lay.n_attr)
        for a, v in VOLUME_INCREMENTS.items():
            inc = inc.at[ai[a]].set(v)
        v_feed = c["MW"] / (1000.0 * c["SG"] * RHO_WATER_60F)
        v0 = v_feed - unit_attr @ inc
        v_cut = v0 + pm @ inc                                    # m^3/mol of treated molecules
        mw_cut = rx.outlet.cut_mw(lay)
        has = rx.outlet.cut > 1e-20 * jnp.sum(rx.outlet.cut)
        mw_cut = jnp.where(has, mw_cut, c["MW"])
        v_cut = jnp.where(has, v_cut, v_feed)
        sg_cut = mw_cut / (1000.0 * v_cut * RHO_WATER_60F)
        mass_pm = mw_cut  # g/mol
        safe = lambda a: jnp.where(has, a, 0.0)
        s_ppm = safe(1e6 * (pm[:, [ai[f"S_{k}"] for k in SULFUR_CLASSES]].sum(1) * 32.065) / mass_pm)
        n_ppm = safe(1e6 * (pm[:, [ai[f"N_{k}"] for k in NITROGEN_CLASSES]].sum(1) * 14.0067) / mass_pm)
        aro = {k: safe(100.0 * pm[:, ai[f"A_{k}"]]) for k in AROMATIC_CLASSES}
        olef = safe(100.0 * pm[:, ai["olefins"]])
        naphth = safe(100.0 * pm[:, ai["naphthenes"]])
        arom = aro["mono"] + aro["di"] + aro["poly"]
        h_wt = safe(100.0 * pm[:, ai["H"]] * 1.00794 / mass_pm)
        # product blend characterization: liquid light ends + cuts
        char = self.char
        lights = self._product_lights()
        l_Tb = jnp.asarray([LIGHT_ENDS[nm][1] for nm in lights], dtype=float).reshape(-1)
        l_sg = jnp.asarray([LIGHT_END_SG[nm] for nm in lights], dtype=float).reshape(-1)
        # molar masses from the atomic weights the unit's balances close on, not the crude unit's
        # tabulated ones (they differ in the fifth figure), so a balance on these streams
        # downstream closes to round-off (#333)
        l_mw = jnp.asarray([gas_mw(nm) for nm in lights], dtype=float).reshape(-1)
        l_tc = jnp.asarray([LIGHT_ENDS[nm][2] for nm in lights], dtype=float).reshape(-1)
        l_pc = jnp.asarray([LIGHT_ENDS[nm][3] for nm in lights], dtype=float).reshape(-1)
        l_w = jnp.asarray([LIGHT_ENDS[nm][4] for nm in lights], dtype=float).reshape(-1)
        zl = jnp.zeros(len(lights))
        cat = lambda a, b: jnp.concatenate([a, b])
        pchar = _ProductGrid(
            Tb=cat(l_Tb, c["Tb"]), SG=cat(l_sg, sg_cut),
            MW=cat(l_mw, mw_cut), Tc=cat(l_tc, c["Tc"]), Pc=cat(l_pc, c["Pc"]), omega=cat(l_w, c["omega"]),
            qualities={"S_ppm": cat(zl, s_ppm), "N_ppm": cat(zl, n_ppm), "aromatics_vol": cat(zl, arom),
                       "olefins_vol": cat(zl, olef), "naphthenes_vol": cat(zl, naphth),
                       "paraffins_vol": cat(100.0 * (1.0 - 0.0 * zl), 100.0 - arom - olef - naphth)})

        def light_flows(f: Flows):
            return jnp.stack([f.gas[lay.gas_index(nm)] for nm in lights]) if lights else jnp.zeros(0)

        def props(f: Flows, prefix: str):
            moles = cat(light_flows(f), f.cut)
            vol = moles * pchar.MW * 1e-3 / (pchar.SG * RHO_WATER_15C)
            phi = vol / jnp.sum(vol)
            m = moles * pchar.MW
            w = m / jnp.sum(m)
            out = {}
            sg = jnp.sum(phi * pchar.SG)
            out[f"{prefix}.sg"] = sg
            out[f"{prefix}.api"] = 141.5 / sg - 131.5
            out[f"{prefix}.S_wppm"] = jnp.sum(w * pchar.qualities["S_ppm"])
            out[f"{prefix}.N_wppm"] = jnp.sum(w * pchar.qualities["N_ppm"])
            out[f"{prefix}.aromatics_vol"] = jnp.sum(phi * pchar.qualities["aromatics_vol"])
            for k in AROMATIC_CLASSES:
                out[f"{prefix}.{k}_aromatics_vol"] = jnp.sum(phi * cat(zl, aro[k]))
            out[f"{prefix}.olefins_vol"] = jnp.sum(phi * pchar.qualities["olefins_vol"])
            out[f"{prefix}.H_wt"] = jnp.sum(w * cat(100.0 * jnp.asarray(
                [GAS_H_FRACTION(nm) for nm in lights], dtype=float).reshape(-1), h_wt))
            tb = {}
            for pct in (5, 10, 50, 90, 95):
                tb[pct] = tbp_temperature(pct / 100.0, pchar.Tb, phi, 5.0)
                out[f"{prefix}.T{pct:02d}" if pct < 10 else f"{prefix}.T{pct}"] = tb[pct]
            d86 = {pct: tbp_to_d86(tb[pct], pct) - C_TO_K for pct in (10, 50, 90)}
            dens = sg * RHO_WATER_15C / 1000.0
            out[f"{prefix}.cetane_index"] = cetane_index_d4737(dens, d86[10], d86[50], d86[90])
            out[f"{prefix}.rate"] = jnp.sum(m) / 1000.0
            out[f"{prefix}.volume"] = jnp.sum(vol)
            return out

        outputs = {}
        outputs.update(props(prod, "product"))
        nprops = props(naph, "naphtha")
        outputs.update({k: v for k, v in nprops.items()
                        if k in ("naphtha.rate", "naphtha.sg", "naphtha.S_wppm", "naphtha.T50", "naphtha.volume")})
        # feed qualities
        feed_mass = oil.mass(lay)
        S_feed = jnp.sum(oil.attr[:, [ai[f"S_{k}"] for k in SULFUR_CLASSES]]) * 32.065 / 1000.0
        N_feed = jnp.sum(oil.attr[:, [ai[f"N_{k}"] for k in NITROGEN_CLASSES]]) * 14.0067 / 1000.0
        Q = o["Q"]
        outputs["feed.rate"] = feed_mass
        outputs["feed.volume"] = Q
        outputs["feed.S_wppm"] = 1e6 * S_feed / feed_mass
        outputs["feed.N_wppm"] = 1e6 * N_feed / feed_mass
        outputs["product.yield"] = outputs["product.rate"] / feed_mass
        outputs["product.volume_yield"] = outputs["product.volume"] / Q
        outputs["naphtha.yield"] = outputs["naphtha.rate"] / feed_mass
        # unconverted heteroatoms: everything still on a cut, in every outlet that carries cuts
        cut_out = prod + naph + off
        s_cols = [ai[f"S_{k}"] for k in SULFUR_CLASSES]
        n_cols = [ai[f"N_{k}"] for k in NITROGEN_CLASSES]
        outputs["hds.conversion"] = 1.0 - jnp.sum(cut_out.attr[:, s_cols]) * 32.065 / 1000.0 / S_feed
        outputs["hdn.conversion"] = 1.0 - jnp.sum(cut_out.attr[:, n_cols]) * 14.0067 / 1000.0 / jnp.maximum(
            N_feed, 1e-300)
        # reactor
        outputs["wabt"] = rx.wabt
        outputs["reactor.T_out"] = rx.T_out
        outputs["reactor.dT_total"] = jnp.sum(rx.delta_T)
        for k, b in enumerate(rx.beds):
            outputs[f"bed{k + 1}.T_in"] = b.T_in
            outputs[f"bed{k + 1}.dT"] = b.T_out - b.T_in
            outputs[f"bed{k + 1}.quench"] = rx.quench[k]
        outputs["catalyst.mass"] = sum(o["W"])
        # hydrogen
        hi = lay.gas_index("hydrogen")
        gas_out = (o["purge"] + o["absorbed"] + o["water"] + off + sour)
        H2_out = gas_out.gas[hi] + naph.gas[hi] + prod.gas[hi]
        H2_in = o["makeup"].gas[hi] + oil.gas[hi]
        outputs["h2.makeup"] = o["makeup"].gas[hi]
        outputs["h2.makeup_nm3_m3"] = o["makeup"].gas[hi] / MOL_PER_NM3 / Q
        outputs["h2.purge"] = o["purge"].gas[hi]
        outputs["h2.dissolved"] = o["liq"].gas[hi]
        # chemical consumption by the hydrogen balance on everything but H2
        Hmat = jnp.asarray(lay.gas_element_matrix())[:, ELEMENTS.index("H")]
        not_h2 = jnp.ones(lay.n_gas).at[hi].set(0.0)

        def H_atoms_nonH2(f: Flows):
            return jnp.sum(f.gas * Hmat * not_h2) + jnp.sum(f.attribute(lay, "H"))

        steam_f = Flows(jnp.zeros(lay.n_gas).at[lay.gas_index("water")].set(steam / 18.01528 * 1000.0),
                        jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
        outs_all = [o["purge"], o["absorbed"], o["water"], off, naph, sour, prod]
        H_out = sum(H_atoms_nonH2(f) for f in outs_all)
        H_in = H_atoms_nonH2(oil) + H_atoms_nonH2(o["makeup"]) + H_atoms_nonH2(steam_f)
        chem = (H_out - H_in) / 2.0
        outputs["h2.chemical"] = chem
        outputs["h2.consumed_by_balance"] = H2_in - H2_out
        outputs["h2.chemical_nm3_m3"] = chem / MOL_PER_NM3 / Q
        outputs["h2.chemical_scf_bbl"] = chem / MOL_PER_NM3 * SCF_PER_NM3 / (Q / BARREL)
        outputs["h2.chemical_wt"] = 100.0 * chem * 2.01588 / 1000.0 / feed_mass
        rec = o["rec"]
        outputs["recycle.rate"] = jnp.sum(rec.gas)
        outputs["recycle.h2_purity"] = rec.gas[hi] / jnp.sum(rec.gas)
        outputs["purge.rate"] = jnp.sum(o["purge"].gas)
        outputs["makeup.rate"] = o["M"]
        gas_t = o["gas"]
        b1 = rx.beds[0].inlet
        outputs["reactor.pH2_in"] = n["P"] * b1.gas[hi] / (jnp.sum(b1.gas) + jnp.sum(b1.cut))
        outputs["compressor.power"] = o["Wc"]
        outputs["compressor.T_out"] = o["T2"]
        outputs.update(self._charge_heater(th, comps, o))
        # gas yield (#332): everything that leaves as gas -- the gas outlets plus the real gases
        # dissolved in the liquid products off their blend grid -- less the makeup gas's own
        # hydrocarbons; H2 and water excluded. Feed light ends count where they leave (C5s in the
        # wild naphtha's yield, C3-C4 mostly in the gas), so the three yields add up to the feed's
        # non-H2, non-water mass plus the chemical hydrogen (``yields.total``).
        not_h2w = jnp.asarray([0.0 if g in ("hydrogen", "water") else 1.0 for g in lay.gases])
        off_grid = jnp.asarray([0.0 if (g in ("hydrogen", "water") or g in lights) else 1.0
                                for g in lay.gases])
        gas_out = jnp.sum((off + o["purge"] + o["absorbed"] + o["water"] + sour).gas_mass(lay) * not_h2w)
        diss = {k: jnp.sum(f.gas_mass(lay) * off_grid) for k, f in (("product", prod), ("naphtha", naph))}
        outputs["gas.yield"] = (gas_out + diss["product"] + diss["naphtha"]
                                - jnp.sum(o["makeup"].gas_mass(lay) * not_h2w)) / feed_mass
        outputs["yields.total"] = outputs["product.yield"] + outputs["naphtha.yield"] + outputs["gas.yield"]
        hw = jnp.asarray([1.0 if g in ("hydrogen", "water") else 0.0 for g in lay.gases])
        outputs["feed.h2_water_rate"] = jnp.sum(oil.gas_mass(lay) * hw)
        # real gases dissolved in the liquid products, off their blend grid (#333): H2, H2S, NH3, C1, C2
        h2m = jnp.asarray([1.0 if g == "hydrogen" else 0.0 for g in lay.gases])
        for k, f in (("product", prod), ("naphtha", naph)):
            outputs[f"{k}.dissolved_gas_rate"] = diss[k] + jnp.sum(f.gas_mass(lay) * h2m)
        outputs["tear.residual"] = o["tear"].residual
        outputs["stripper.residual"] = st.residual
        outputs["stripper.bottoms_T"] = st.outputs["bottoms.T"]
        # balances: in = oil + makeup + steam; out = purge + acid gas + HPS water + off-gas + naphtha + sour water + product
        tot_in = oil + o["makeup"] + steam_f
        tot_out = outs_all[0]
        for f in outs_all[1:]:
            tot_out = tot_out + f
        e_in, e_out = tot_in.elements(lay), tot_out.elements(lay)
        balances = {"mass": relative_balance_error(tot_in.mass(lay), tot_out.mass(lay))}
        for k, e in enumerate(ELEMENTS):
            balances[e] = relative_balance_error(e_in[k], e_out[k])
        # converged: the tear and the stripper, and every PR flash (bed inlets, HPS, drum) -- a
        # flash that fails returns its start with a large residual rather than raising (#332)
        flash_res = jnp.stack([b.flash_residual for b in rx.beds] + [o["fr"].residual, frd.residual])
        outputs["flash.residual"] = jnp.max(flash_res)
        conv = (o["tear"].converged & st.converged & jnp.all(flash_res < FLASH_TOL)
                & jnp.isfinite(outputs["product.S_wppm"]) & jnp.isfinite(outputs["h2.chemical"]))
        if not full:
            return outputs, None, None, None, None
        streams = {"feed": oil, "makeup": o["makeup"], "treat_gas": gas_t, "reactor_out": rx.outlet,
                   "hps_vapor": o["vap"], "hps_liquid": liq, "hps_water": o["water"], "acid_gas": o["absorbed"],
                   "purge": o["purge"], "recycle": rec, "stripper_overhead": st.overhead, "product": prod,
                   "wild_naphtha": naph, "off_gas": off, "sour_water": sour, "steam": steam_f,
                   "_light_product": light_flows(prod), "_light_wild_naphtha": light_flows(naph)}
        return outputs, streams, pchar, balances, conv

    def _charge_heater(self, th, comps, o) -> dict:
        """Charge-heater duty (#332): bed-1 inlet enthalpy less that of what enters the heater.

        The oil and the first bed's share of the treat gas (the treat gas less
        every quench) are heated to the bed-1 inlet temperature, which stays the
        spec. What enters the heater is either the oil at its feed temperature
        and the gas at the recycle compressor's discharge (``heater_inlet_T is
        None``), or both mixed at ``heater_inlet_T`` after a feed/effluent
        exchanger, whose cold-side duty is then reported too (its hot side is
        not modelled, so a temperature cross is not checked). Enthalpies are
        the reactor's own (:func:`~difflow_refinery.hydroprocessing.reactor.stream_enthalpy`:
        PR K-values and phase split, ideal-gas and liquid enthalpies on one
        basis) at the reactor pressure; feed water, if any, is taken as vapour
        (the reactor's convention), which understates the duty by its latent
        heat. Fired duty is the absorbed duty over ``heater_efficiency``.
        """
        lay = self.layout
        n = th["num"]
        P = n["P"]
        b1 = o["rx"].beds[0]
        oil = th["oil"]
        gas1 = b1.inlet - oil
        H_bed = stream_enthalpy(b1.inlet, lay, comps, b1.k_model, b1.T_in)
        # the oil is pumped to reactor pressure and enters as liquid (no H2 in it yet, so it is
        # below its bubble point there); water, if any, on the reactor's vapour convention
        T_f = th["feed_T"]
        H_oil = jnp.sum(flash_components(oil, lay, comps) * comps.h_liquid(T_f))
        if "water" in lay.gases:
            H_oil = H_oil + oil.gas[lay.gas_index("water")] * ColumnThermo.water_h_vapor(T_f)
        H_cold = H_oil + vapor_enthalpy(gas1, lay, comps, o["T2"])
        out = {}
        if self.params.heater_inlet_T is None:
            H_in = H_cold
            out["heater.inlet_T"] = jnp.minimum(th["feed_T"], o["T2"])
            out["feed_effluent.duty"] = jnp.asarray(0.0)
        else:
            T_h = n["heater_inlet_T"]
            km_mix, _ = k_model_at(b1.inlet, lay, comps, T_h, P)
            H_in = stream_enthalpy(b1.inlet, lay, comps, km_mix, T_h)
            out["heater.inlet_T"] = T_h
            out["feed_effluent.duty"] = H_in - H_cold
        out["heater.duty"] = H_bed - H_in
        out["heater.fired_duty"] = out["heater.duty"] / n["heater_efficiency"]
        return out

    def _product_lights(self) -> list[str]:
        """Light ends that are blend components of the liquid products (C3+ with a liquid SG)."""
        return [nm for nm in self.char.light_names if nm in LIGHT_END_SG]

    def solve(self, feed: Mapping, char=None, params: HydrotreaterParams | None = None,
              warn: bool = True) -> HydrotreaterResult:
        """Solve the unit for ``feed`` (``F_<char.names>``, mol/s).

        ``char`` and ``params`` may replace the unit's (same names, same
        layout-shaping settings: bed count, quench mode, stripper stages) --
        that is how a gradient with respect to the assay or a spec is taken.
        """
        if params is not None:
            static = ("stripper_stages", "tear_tol", "reactor")
            if (params.heater_inlet_T is None) != (self.params.heater_inlet_T is None):
                raise ValueError("heater_inlet_T None vs a value shapes the solve; build a new Hydrotreater")
            for k in static:
                if getattr(params, k) != getattr(self.params, k):
                    raise ValueError(f"params.{k} shapes the solve; build a new Hydrotreater to change it")
            if params.kij != self.params.kij or params.kinetics.hds_form != self.params.kinetics.hds_form:
                raise ValueError("kij and kinetics.hds_form shape the solve; build a new Hydrotreater")
            if (params.quench is None) != (self.params.quench is None) or \
                    len(params.bed_fractions) != len(self.params.bed_fractions) or \
                    len(params.T_in) != len(self.params.T_in):
                raise ValueError("the bed count and quench mode shape the solve; build a new Hydrotreater")
        res = self._jit(self.theta(feed, char, params))
        if warn:
            try:
                ok = bool(res.converged)
            except jax.errors.ConcretizationTypeError:
                ok = True
            if not ok:
                warnings.warn(f"hydrotreater did not converge (tear residual "
                              f"{float(res.outputs['tear.residual']):.2e}, stripper residual "
                              f"{float(res.outputs['stripper.residual']):.2e}, largest flash residual "
                              f"{float(res.outputs['flash.residual']):.2e})",
                              HydrotreaterConvergenceWarning, stacklevel=2)
        return res

    __call__ = solve


def GAS_H_FRACTION(name: str) -> float:
    """Hydrogen mass fraction of a light end (paraffin)."""
    from difflow_refinery.hydroprocessing.layout import GAS_ELEMENTS, gas_mw
    return GAS_ELEMENTS[name]["H"] * 1.00794 / gas_mw(name)


__all__ = ["Hydrotreater", "HydrotreaterParams", "HydrotreaterResult", "TargetSpec", "OUTPUT_UNITS",
           "VOLUME_INCREMENTS", "HydrotreaterConvergenceWarning"]
