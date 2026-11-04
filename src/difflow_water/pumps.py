"""Water pumps: isothermal, incompressible, with a mechanical efficiency.

``W_fluid = V dP`` and ``W_mech = W_fluid / eta``, with ``V`` the inlet
volumetric flow from the seawater density [S1]. This is the isothermal
pump WaterTAP's flowsheets use (IDAES ``Pump``; ``work_fluid`` and
``work_mechanical``), so pump energies compare one for one.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow_water.streams import volumetric_flow


@dataclass(repr=False)
class WaterPumpParams(ParamsMixin):
    """Parameters of a water pump.

    Attributes:
        outlet_pressure: Outlet pressure (Pa); takes precedence over
            ``deltaP`` when given.
        deltaP: Pressure rise (Pa), used when ``outlet_pressure`` is None.
            The default 0 is a placeholder (a palette drop builds a pump
            that does nothing until one of the two is set).
        efficiency: Mechanical (wire-to-water) efficiency (-).
    """

    outlet_pressure: float | Array | None = None
    deltaP: float | Array = 0.0
    efficiency: float | Array = 0.8


class WaterPump:
    """Isothermal water pump.

    Example:
        >>> from difflow_water.streams import make_seawater_stream
        >>> feed = make_seawater_stream(flow_vol=1e-3)
        >>> out, info = WaterPump(WaterPumpParams(outlet_pressure=101325.0 + 1e5))(feed)
        >>> round(float(info["work_mechanical"]), 1)
        125.0
    """

    symbol = "Pump"
    equations = [r"W_{fluid} = \dot V\,\Delta P,\qquad W_{mech} = W_{fluid}/\eta"]
    assumptions = ["Isothermal, incompressible; inlet volumetric flow."]
    references = ["WaterTAP 1.8.0 / IDAES Pump (isothermal), as used in the WaterTAP RO flowsheets."]
    parameter_symbols = {"outlet_pressure": "P_out", "deltaP": "dP", "efficiency": "eta"}
    parameter_units = {"outlet_pressure": "Pa", "deltaP": "Pa", "efficiency": "-"}

    def __init__(self, params: WaterPumpParams):
        """Initialize.

        Args:
            params: :class:`WaterPumpParams`.
        """
        if params.outlet_pressure is None and params.deltaP is None:
            raise ValueError("give outlet_pressure or deltaP")
        self.params = params

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        """Pump a seawater stream.

        Args:
            inlet: Seawater stream.

        Returns:
            ``(outlet, info)``; ``info`` has ``deltaP`` (Pa), ``flow_vol``
            (m^3/s), ``work_fluid`` and ``work_mechanical`` (W).
        """
        p = self.params
        if p.outlet_pressure is not None:
            dP = jnp.asarray(p.outlet_pressure, dtype=jnp.float64) - inlet["P"]
        else:
            dP = jnp.asarray(p.deltaP, dtype=jnp.float64)
        V = volumetric_flow(inlet)
        out = dict(inlet)
        out["P"] = inlet["P"] + dP
        w = V * dP
        return out, {"deltaP": dP, "flow_vol": V, "work_fluid": w,
                     "work_mechanical": w / p.efficiency}
