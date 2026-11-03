"""Multi-stage RO train: the concentrate of each stage feeds the next.

Stages are :class:`~difflow_water.membranes.ro.ROStage` units in series on
the concentrate side, with an optional booster pump before each stage after
the first (isothermal, ``W = V dP / eta``, as :class:`~difflow_water.pumps.
WaterPump`). The permeates are mixed. There is no physics here beyond the
stage and the pump: this is the usual single-pass, multi-stage arrangement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow_water.membranes.ro import ROStage, ROStageParams
from difflow_water.pumps import WaterPump, WaterPumpParams
from difflow_water.streams import combine_seawater, mass_flows, volumetric_flow


@dataclass(repr=False)
class ROTrainParams(ParamsMixin):
    """Parameters of an RO train.

    Attributes:
        stages: One :class:`ROStageParams` per stage, in flow order.
        boost: Pressure added before each stage after the first (Pa); a list
            of ``len(stages) - 1`` values, or None for no interstage boost.
        booster_efficiency: Efficiency of the interstage booster pumps (-).
    """

    stages: list = field(default_factory=list)
    boost: list | None = None
    booster_efficiency: float | Array = 0.8


class ROTrain:
    """Concentrate-staged RO train.

    Example:
        >>> import difflow_water as dw
        >>> feed = dw.make_seawater_stream(flow_vol=1e-3, salinity=0.035, P=65e5)
        >>> s = dw.ROStageParams(A_w=4.2e-12, B=3.5e-8, area=30.0, n_steps=10)
        >>> conc, perm, info = dw.ROTrain(dw.ROTrainParams(stages=[s, s], boost=[5e5]))(feed)
        >>> len(info["stages"])
        2
    """

    symbol = "RO train"
    equations = [r"F^{feed}_{k+1} = F^{conc}_{k},\quad P^{feed}_{k+1} = P^{conc}_{k} + \Delta P^{boost}_{k}"]
    assumptions = ["Stages as ROStage; permeates fully mixed at the lowest permeate pressure."]
    references = ["WaterTAP 1.8.0 multi-stage RO flowsheets (arrangement)."]
    parameter_symbols = {"boost": "dP_boost", "booster_efficiency": "eta"}
    parameter_units = {"stages": "-", "boost": "Pa", "booster_efficiency": "-"}
    outlet_roles = ["concentrate", "permeate"]

    def __init__(self, params: ROTrainParams):
        """Initialize.

        Args:
            params: :class:`ROTrainParams`.
        """
        if not params.stages:
            raise ValueError("an RO train needs at least one stage")
        if params.boost is not None and len(params.boost) != len(params.stages) - 1:
            raise ValueError("boost needs one value per stage after the first")
        self.params = params

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict]:
        """Run the feed through every stage.

        Args:
            feed: Seawater stream at the first stage's feed pressure.

        Returns:
            ``(concentrate, permeate, info)``; ``info`` has ``stages`` (each
            stage's info), ``recovery_mass_water`` (train), ``salinity_permeate``,
            ``work_boosters`` (W, total) and ``permeate_flow_vol`` (m^3/s).
        """
        p = self.params
        infos, perms = [], []
        work = jnp.asarray(0.0)
        x = feed
        for i, sp in enumerate(p.stages):
            if i > 0 and p.boost is not None:
                x, pi = WaterPump(WaterPumpParams(deltaP=p.boost[i - 1],
                                                  efficiency=p.booster_efficiency))(x)
                work = work + pi["work_mechanical"]
            x, perm, info = ROStage(sp if isinstance(sp, ROStageParams) else ROStageParams(**sp))(x)
            infos.append(info)
            perms.append(perm)
        permeate = combine_seawater(*perms)
        mw_f, _ = mass_flows(feed)
        mw_p, ms_p = mass_flows(permeate)
        return x, permeate, {
            "stages": infos,
            "recovery_mass_water": mw_p / mw_f,
            "salinity_permeate": ms_p / (mw_p + ms_p),
            "work_boosters": work,
            "permeate_flow_vol": volumetric_flow(permeate),
        }
