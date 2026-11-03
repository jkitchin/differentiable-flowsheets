"""Isobaric energy recovery: the pressure exchanger.

A pressure exchanger [M5] hands the pressure of the RO concentrate (brine)
to an equal volume of incoming seawater. The equations are the ones
WaterTAP 1.8.0's ``PressureExchanger`` uses (efficiency form, optional
volumetric leakage and mixing):

- pressure transfer: ``P_feed,out - P_feed,in = eta (P_brine,in - P_brine,out)``
- equal low pressures: ``P_brine,out = P_feed,in``
- volumetric flows: ``V_feed,out = (1 - L) V_brine,in`` (``L`` the leakage
  fraction; ``L = 0`` is WaterTAP's equal-flow constraint)
- mixing: ``C_feed,out = (1 - x) C_feed,in + x C_brine,in`` (mass
  concentration), the salt moving from the brine to the feed side, the
  low-pressure feed volume unchanged.

In a sequential flowsheet the equal-flow condition is met by drawing that
volume from the seawater fed to the exchanger: the rest leaves through a
third outlet, ``bypass``, which goes to the high-pressure pump. Feeding the
exchanger the whole plant intake therefore reproduces the usual split
without a separate splitter whose fraction would have to be iterated.

The efficiency, leakage and mixing defaults (0.95, 0, 0) are the values WaterTAP's
``RO_with_energy_recovery`` example fixes; they are design inputs, not
sourced performance data (ILLUSTRATIVE).
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream
from difflow_water.chemistry import seawater as sw
from difflow_water.chemistry.seawater import MW_SEA_SALT, MW_WATER
from difflow_water.streams import SALT, WATER, mass_flows, salinity, volumetric_flow


@dataclass(repr=False)
class PressureExchangerParams(ParamsMixin):
    """Parameters of a pressure exchanger.

    Attributes:
        efficiency: Pressure-transfer efficiency ``eta`` (-). ILLUSTRATIVE default.
        mixing: Volumetric mixing fraction ``x`` of brine into the
            pressurized feed (-). ILLUSTRATIVE default.
        leakage: Volumetric leakage fraction ``L``: the pressurized feed is
            ``(1 - L)`` times the brine volume (-). ILLUSTRATIVE default.
    """

    efficiency: float | Array = 0.95
    mixing: float | Array = 0.0
    leakage: float | Array = 0.0


def _stream_from_vol_and_conc(V, C, T, P) -> Stream:
    """A seawater stream of volume flow V (m^3/s) and salt mass concentration C (kg/m^3)."""
    S = sw.salinity_from_mass_concentration(C, T)
    m = V * sw.density(T, S)
    return make_stream({WATER: m * (1.0 - S) / MW_WATER, SALT: m * S / MW_SEA_SALT}, T, P)


class PressureExchanger:
    """Pressure exchanger with a feed bypass outlet.

    Inlets: ``brine`` (high-pressure RO concentrate) and ``feed``
    (low-pressure seawater, at least the brine's volume). Outlets: the
    pressurized feed (``hp_feed``), the depressurized ``brine`` and the
    unused ``bypass`` seawater (at the feed pressure).

    Example:
        >>> from difflow_water.streams import make_seawater_stream
        >>> brine = make_seawater_stream(flow_vol=5e-4, salinity=0.07, P=60e5)
        >>> feed = make_seawater_stream(flow_vol=1e-3, salinity=0.035, P=2e5)
        >>> hp, lp, bypass, info = PressureExchanger(PressureExchangerParams())(brine, feed)
        >>> round(float(hp["P"]) / 1e5, 1)
        56.9
    """

    symbol = "PX"
    equations = [
        r"P^{out}_{f} - P^{in}_{f} = \eta\,(P^{in}_{b} - P^{out}_{b}),\qquad P^{out}_{b} = P^{in}_{f}",
        r"V^{out}_{f} = (1 - L)\,V^{in}_{b}",
        r"C^{out}_{f} = (1-x)\,C^{in}_{f} + x\,C^{in}_{b}",
    ]
    assumptions = [
        "Isothermal.",
        "Pressurized feed volume (1 - L) times the brine volume; the feed in excess of it bypasses.",
        "Mixing moves salt only; the low-pressure side volume is unchanged.",
    ]
    references = [
        "Stover, Desalination 203 (2007) 168-175, doi:10.1016/j.desal.2006.03.528.",
        "WaterTAP 1.8.0 PressureExchanger (equations as coded).",
    ]
    parameter_symbols = {"efficiency": "eta", "mixing": "x", "leakage": "L"}
    parameter_units = {"efficiency": "-", "mixing": "-", "leakage": "-"}
    outlet_roles = ["hp_feed", "brine", "bypass"]

    def __init__(self, params: PressureExchangerParams):
        """Initialize.

        Args:
            params: :class:`PressureExchangerParams`.
        """
        self.params = params

    def __call__(self, brine: Stream, feed: Stream) -> tuple[Stream, Stream, Stream, dict]:
        """Exchange pressure from the brine to the feed.

        Args:
            brine: High-pressure concentrate (seawater stream).
            feed: Low-pressure seawater.

        Returns:
            ``(hp_feed, brine_out, bypass, info)``; ``info`` has
            ``flow_vol_exchanged`` (m^3/s), ``bypass_fraction`` (of the
            feed volume), ``salinity_hp_feed``, ``salinity_increase``
            and ``deltaP_feed``/``deltaP_brine`` (Pa).
        """
        p = self.params
        Tb, Tf = brine["T"], feed["T"]
        Vb = volumetric_flow(brine)
        Vf = volumetric_flow(feed)
        Sb, Sf = salinity(brine), salinity(feed)
        Cb = Sb * sw.density(Tb, Sb)
        Cf = Sf * sw.density(Tf, Sf)
        dP_brine = feed["P"] - brine["P"]
        P_hp = feed["P"] - p.efficiency * dP_brine
        C_hp = (1.0 - p.mixing) * Cf + p.mixing * Cb
        V_hp = (1.0 - p.leakage) * Vb
        hp = _stream_from_vol_and_conc(V_hp, C_hp, Tf, P_hp)
        # salt moved to the feed side leaves the brine; brine volume unchanged
        salt_moved = V_hp * (C_hp - Cf)
        mw_b, ms_b = mass_flows(brine)
        lp = make_stream({WATER: mw_b / MW_WATER, SALT: (ms_b - salt_moved) / MW_SEA_SALT},
                         Tb, feed["P"])
        # the drawn volume leaves the feed at the feed's own composition
        frac = V_hp / Vf
        bypass = {k: (v * (1.0 - frac) if k.startswith("F_") else v) for k, v in feed.items()}
        info = {
            "flow_vol_exchanged": V_hp,
            "flow_vol_brine": Vb,
            "bypass_fraction": 1.0 - frac,
            "salinity_hp_feed": sw.salinity_from_mass_concentration(C_hp, Tf),
            "salinity_increase": sw.salinity_from_mass_concentration(C_hp, Tf) - Sf,
            "deltaP_feed": P_hp - feed["P"],
            "deltaP_brine": dP_brine,
        }
        return hp, lp, bypass, info
