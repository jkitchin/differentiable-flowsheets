"""Water distribution networks.

A library, not palette operations: :class:`NetworkModel` solves a whole
:class:`WaterNetwork` equation-oriented (heads and flows together, as EPANET
does), differentiably. See ``docs/unit-operations-water.md``.
"""

import json
from importlib import resources

from difflow_water.network.hydraulics import (
    NetworkConvergenceWarning,
    NetworkModel,
    darcy_friction,
)
from difflow_water.network.model import WaterNetwork, pump_curve

EXAMPLES = ("Net1", "Net3")


def load_example(name: str) -> tuple[WaterNetwork, dict]:
    """An EPANET example network and EPANET 2.2's own solution of it at t = 0.

    Args:
        name: "Net1" or "Net3" (EPANET's example networks, U.S. EPA, public
            domain; converted to SI by WNTR, demands and levels at t = 0).

    Returns:
        ``(network, epanet)``; ``epanet`` has ``head``, ``pressure``,
        ``demand`` (by node name) and ``flow`` (by link name), SI units.
    """
    if name not in EXAMPLES:
        raise ValueError(f"example must be one of {EXAMPLES}")
    rec = json.loads(resources.files("difflow_water.network.data").joinpath(f"{name}.json").read_text())
    return WaterNetwork.from_dict(rec["network"]), rec["epanet"]


__all__ = ["WaterNetwork", "NetworkModel", "NetworkConvergenceWarning", "pump_curve",
           "darcy_friction", "load_example", "EXAMPLES"]
