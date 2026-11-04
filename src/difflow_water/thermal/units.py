"""Palette operations for thermal desalination: whole-plant MVC and MED blocks."""

from __future__ import annotations

import jax.numpy as jnp

from difflow.streams import Stream
from difflow_water.thermal.med import MEDParams, med_performance
from difflow_water.thermal.mvc import MVCParams, mvc_performance


def _scalars(out: dict) -> dict:
    """The scalar (and per-effect array) entries of a performance dict."""
    return {k: v for k, v in out.items()
            if not isinstance(v, dict) and jnp.ndim(v) <= 1}


class MVCDesalinator:
    """Single-stage mechanical vapor compression plant (WaterTAP's MVC layout).

    Inlet: seawater ``feed``. Outlets: ``distillate`` and ``brine``. The
    whole plant (feed pump, preheaters, evaporator, compressor, condenser,
    brine and distillate pumps) is one block; see
    :func:`~difflow_water.thermal.mvc.mvc_performance`.

    Example:
        >>> from difflow_water.streams import make_seawater_stream
        >>> feed = make_seawater_stream(flow_mass=44.444, salinity=0.1)
        >>> dist, brine, info = MVCDesalinator(MVCParams())(feed)
        >>> round(float(info["recovery"]), 2)
        0.5
    """

    symbol = "MVC"
    equations = [
        r"Q + H_{feed} = H_{brine} + H_{vap},\qquad Q = U A\,\mathrm{LMTD}",
        r"P_{out} = r P_{in},\quad T_s = T_{in} r^{1 - 1/\gamma},\quad \eta (h_{out} - h_{in}) = h_s - h_{in}",
        r"m_v (h_{v,out} - h_{l}(T_c)) = Q\qquad \text{(complete condensation)}",
        r"\mathrm{LMTD} \approx \left(\Delta T_1 \Delta T_2 \tfrac{\Delta T_1 + \Delta T_2}{2}\right)^{1/3}\ \text{(Chen)}",
    ]
    assumptions = [
        "WaterTAP 1.8.0 mvc_single_stage layout and equations; steady state.",
        "Brine and vapor leave the evaporator at the brine temperature; brine at its vapor pressure.",
        "Compressor isentropic temperature from gamma = 1.3 (ideal gas).",
        "Brine temperature and preheater areas are inputs; evaporator area is computed.",
        "Pressure ratio solved for saturated condensate (default) or given (mode='fixed_pressure_ratio').",
    ]
    references = [
        "Sharqawy, Lienhard & Zubair, Desalin. Water Treat. 16 (2010) 354-380, doi:10.5004/dwt.2010.1079.",
        "El-Dessouky & Ettouney, Fundamentals of Salt Water Desalination (2002), doi:10.1016/B978-0-444-50810-2.X5000-3.",
        "WaterTAP 1.8.0 flowsheets/mvc/mvc_single_stage.py (equations as coded).",
    ]
    parameter_symbols = {"recovery": "r_w", "T_brine": "T_b", "pressure_ratio": "r",
                         "compressor_efficiency": "eta", "U_evaporator": "U"}
    parameter_units = {"recovery": "-", "T_brine": "K", "pressure_ratio": "-",
                       "compressor_efficiency": "-", "gamma": "-",
                       "U_evaporator": "W/(m^2 K)", "A_hx_distillate": "m^2",
                       "A_hx_brine": "m^2", "U_hx": "W/(m^2 K)", "feed_pump_dP": "Pa",
                       "hx_dP": "Pa", "distillate_pump_dP": "Pa", "pump_efficiency": "-",
                       "brine_outlet_pressure": "Pa", "mode": "-"}
    numerical_method = "Newton on the plant's 8 coupled balances (optimistix; implicit gradients)."
    outlet_roles = ["distillate", "brine"]

    def __init__(self, params: MVCParams):
        """Initialize.

        Args:
            params: :class:`~difflow_water.thermal.mvc.MVCParams`.
        """
        self.params = params

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict]:
        """Desalinate a seawater feed.

        Returns:
            ``(distillate, brine, info)``; ``info`` holds the scalar outputs of
            :func:`~difflow_water.thermal.mvc.mvc_performance` plus ``recovery``.
        """
        out = mvc_performance(feed, self.params)
        info = _scalars(out)
        info["recovery"] = jnp.asarray(self.params.recovery, dtype=jnp.float64)
        return out["distillate"], out["brine"], info


class MEDDesalinator:
    """Parallel-feed multi-effect distillation plant.

    Inlet: seawater ``feed`` (at the intake temperature). Outlets:
    ``distillate`` and ``brine``; see
    :func:`~difflow_water.thermal.med.med_performance`.

    Example:
        >>> from difflow_water.streams import make_seawater_stream
        >>> feed = make_seawater_stream(flow_mass=20.0, salinity=0.035)
        >>> dist, brine, info = MEDDesalinator(MEDParams(n_effects=4))(feed)
        >>> float(info["GOR"]) > 1.0
        True
    """

    symbol = "MED"
    equations = [
        r"Q_i + F_i h(T_f, S_f) = B_i h(T_i, S_i) + D_i h_v(T_i),\qquad F_i S_f = B_i S_i",
        r"Q_1 = m_s h_{fg}(T_s),\qquad Q_{i} = D_{i-1}\,[h_v(T_{i-1}) - h_w(T_{i-1} - \mathrm{BPE}_{i-1})]",
        r"A_i = Q_i / (U\,(T_{c,i} - T_i)),\qquad \mathrm{GOR} = \textstyle\sum_i D_i / m_s",
    ]
    assumptions = [
        "Parallel feed, equal feed to each effect; effect temperatures linear from T_top to T_last.",
        "Each effect's vapor condenses in the next at its saturation temperature (T - BPE).",
        "No feed preheaters between effects, no distillate flashing; condensate leaves saturated.",
        "Final condenser preheats the whole feed; the rest of its heat goes to cooling seawater.",
    ]
    references = [
        "Sharqawy, Lienhard & Zubair, Desalin. Water Treat. 16 (2010) 354-380, doi:10.5004/dwt.2010.1079.",
        "El-Dessouky & Ettouney, Fundamentals of Salt Water Desalination (2002), doi:10.1016/B978-0-444-50810-2.X5000-3.",
    ]
    parameter_symbols = {"n_effects": "N", "steam_flow": "m_s", "T_steam": "T_s",
                         "T_top": "T_1", "T_last": "T_N", "T_feed": "T_f",
                         "U_effect": "U"}
    parameter_units = {"n_effects": "-", "steam_flow": "kg/s", "T_steam": "K",
                       "T_top": "K", "T_last": "K", "T_feed": "K",
                       "U_effect": "W/(m^2 K)", "U_condenser": "W/(m^2 K)"}
    numerical_method = "Effects solved in sequence; Newton (fixed count) on each energy balance."
    outlet_roles = ["distillate", "brine"]

    def __init__(self, params: MEDParams):
        """Initialize.

        Args:
            params: :class:`~difflow_water.thermal.med.MEDParams`.
        """
        self.params = params
        if int(self.params.n_effects) < 1:
            raise ValueError(f"n_effects must be at least 1, got {self.params.n_effects}")

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict]:
        """Desalinate a seawater feed.

        Returns:
            ``(distillate, brine, info)``; ``info`` holds the scalar and
            per-effect outputs of :func:`~difflow_water.thermal.med.med_performance`.
        """
        out = med_performance(feed, self.params)
        return out["distillate"], out["brine"], _scalars(out)
