"""Standard benchmark networks, and a MATPOWER case-struct importer.

Every case here is public benchmark data reproduced verbatim from the
MATPOWER distribution, so a result computed with this plugin can be
compared line by line against MATPOWER, PYPOWER, PowerModels or
pandapower on the same case. Believing a new power-flow implementation
without that comparison is not an option --- sign conventions on phase
shift, the ``tap = 0`` sentinel and the half-charging split are each
easy to get wrong in a way that costs a few percent and looks
plausible.

Cases
-----

======================  ==========================================
:func:`case3`           a hand-built 3-bus loop, the smallest thing
                        with a meshed flow to reason about
:func:`case5`           PJM 5-bus: linear costs and two binding line
                        ratings, the standard congestion / LMP demo
:func:`case9`           WSCC 9-bus, 3 machines: the classic
                        power-flow and OPF benchmark
:func:`case14`          IEEE 14-bus: three tap-changing transformers
                        and a shunt capacitor, so it exercises the
                        parts of the branch model the smaller cases
                        do not
:func:`radial_feeder`   a 7-bus distribution feeder: no loops, so it
                        also admits the sequential sweep of
                        :mod:`difflow_power.flowsheet`
======================  ==========================================

Importing your own
------------------

:func:`from_matpower` takes a MATPOWER / PYPOWER case struct --- the
plain dict of arrays that ``pypower.case9()`` returns --- and builds a
:class:`~difflow_power.network.PowerNetwork`. It handles the format's
sentinels (``tap = 0`` means 1, ``rateA = 0`` means unlimited),
converts degrees to radians and MW to per unit, and drops
out-of-service components rather than solving with them at zero
impedance.
"""

from __future__ import annotations

from typing import Any, Mapping

import math

from difflow_power.network import Branch, Bus, Generator, Load, PowerNetwork

#: MATPOWER bus type codes
_MATPOWER_BUS_KIND = {1: "pq", 2: "pv", 3: "slack"}


def from_matpower(mpc: Mapping[str, Any], name: str | None = None) -> PowerNetwork:
    """Build a network from a MATPOWER / PYPOWER case struct.

    Args:
        mpc: a mapping with ``"baseMVA"``, ``"bus"``, ``"gen"``,
            ``"branch"`` and optionally ``"gencost"``, whose values are
            the case format's numeric matrices (any nested sequence;
            numpy arrays work).
        name: label for the resulting network.

    Returns:
        A :class:`~difflow_power.network.PowerNetwork` with bus ids
        being the case's bus numbers as strings, branch ids
        ``"br<k>"``, generator ids ``"g<k>"`` and load ids ``"d<bus>"``.

    Raises:
        ValueError: if a ``gencost`` row is piecewise-linear (model 1),
            which this plugin does not carry --- a piecewise-linear
            cost makes the OPF an MILP or needs a constrained-cost-
            variable reformulation, and silently fitting a polynomial
            to it would misprice the dispatch.

    Notes:
        Out-of-service components (``status = 0``) are dropped. Bus
        shunts, voltage limits and angle-difference limits are carried
        through; areas, zones and ramp rates are not, because nothing
        in this plugin reads them.
    """
    base_mva = float(mpc["baseMVA"])
    buses: dict[str, Bus] = {}
    loads: dict[str, Load] = {}

    for row in mpc["bus"]:
        bid = str(int(row[0]))
        buses[bid] = Bus(
            kind=_MATPOWER_BUS_KIND[int(row[1])],
            base_kv=float(row[9]) or 1.0,
            vm_min=float(row[12]),
            vm_max=float(row[11]),
            vm_setpoint=float(row[7]),
            g_shunt_mw=float(row[4]),
            b_shunt_mvar=float(row[5]),
            va_reference=math.radians(float(row[8]))
            if int(row[1]) == 3 else 0.0,
            name=f"bus {bid}",
        )
        if float(row[2]) != 0.0 or float(row[3]) != 0.0:
            loads[f"d{bid}"] = Load(
                bus=bid, p_mw=float(row[2]), q_mvar=float(row[3]),
                name=f"load {bid}",
            )

    costs = _polynomial_costs(mpc.get("gencost"))
    generators: dict[str, Generator] = {}
    k = 0
    for i, row in enumerate(mpc["gen"]):
        if float(row[7]) <= 0.0:          # out of service
            continue
        k += 1
        generators[f"g{k}"] = Generator(
            bus=str(int(row[0])),
            p_min_mw=float(row[9]),
            p_max_mw=float(row[8]),
            q_min_mvar=float(row[4]),
            q_max_mvar=float(row[3]),
            cost=costs[i] if costs else (0.0,),
            p_mw=float(row[1]),
            q_mvar=float(row[2]),
            vm_setpoint=float(row[5]),
            name=f"gen {k} @ bus {int(row[0])}",
        )

    branches: dict[str, Branch] = {}
    k = 0
    for row in mpc["branch"]:
        if float(row[10]) <= 0.0:         # out of service
            continue
        k += 1
        branches[f"br{k}"] = Branch(
            from_bus=str(int(row[0])),
            to_bus=str(int(row[1])),
            r=float(row[2]),
            x=float(row[3]),
            b=float(row[4]),
            tap=float(row[8]),
            shift=math.radians(float(row[9])),
            rate_mva=float(row[5]) or None,
            angle_min=math.radians(float(row[11])) if len(row) > 11 else None,
            angle_max=math.radians(float(row[12])) if len(row) > 12 else None,
            name=f"{int(row[0])}-{int(row[1])}",
        )

    return PowerNetwork(
        buses=buses, branches=branches, generators=generators,
        loads=loads, base_mva=base_mva, name=name,
    )


def _polynomial_costs(gencost) -> list[tuple[float, ...]] | None:
    """Pull polynomial coefficients out of a ``gencost`` matrix."""
    if gencost is None:
        return None
    out = []
    for row in gencost:
        model = int(row[0])
        if model == 1:
            raise ValueError(
                "piecewise-linear generator costs (gencost model 1) are "
                "not supported: they make the OPF non-smooth, and fitting "
                "a polynomial to them would misprice the dispatch. Convert "
                "the case to model 2 (polynomial) costs first."
            )
        n = int(row[3])
        out.append(tuple(float(c) for c in row[4:4 + n]))
    return out


# =============================================================================
# Benchmark cases
# =============================================================================


def case3() -> PowerNetwork:
    """A 3-bus loop with two units and one load.

    Not a standard benchmark --- it is the smallest network in which
    power divides between two paths, which makes it the right size for
    a doctest, a hand-checked Jacobian, or reasoning about why an OPF
    moved a dispatch. Costs are chosen so the cheap unit is on the
    electrically longer path, and the OPF therefore has a real decision
    to make.
    """
    return PowerNetwork(
        name="case3",
        base_mva=100.0,
        buses={
            "1": Bus(kind="slack", base_kv=230.0, vm_setpoint=1.0),
            "2": Bus(kind="pv", base_kv=230.0, vm_setpoint=1.0),
            "3": Bus(kind="pq", base_kv=230.0),
        },
        branches={
            "l12": Branch("1", "2", 0.01, 0.10, 0.02, rate_mva=120.0,
                          name="1-2"),
            "l23": Branch("2", "3", 0.01, 0.10, 0.02, rate_mva=120.0,
                          name="2-3"),
            "l13": Branch("1", "3", 0.02, 0.20, 0.02, rate_mva=120.0,
                          name="1-3"),
        },
        generators={
            "g1": Generator("1", 10.0, 200.0, -80.0, 80.0,
                            cost=(0.11, 20.0, 100.0), p_mw=100.0,
                            name="expensive unit"),
            "g2": Generator("2", 10.0, 150.0, -60.0, 60.0,
                            cost=(0.085, 12.0, 80.0), p_mw=60.0,
                            name="cheap unit"),
        },
        loads={"d3": Load("3", 150.0, 50.0, name="load")},
    )


#: PJM 5-bus case (MATPOWER ``case5.m``), verbatim.
_CASE5 = {
    "baseMVA": 100.0,
    "bus": [
        [1, 2, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [2, 1, 300, 98.61, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [3, 2, 300, 98.61, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [4, 3, 400, 131.47, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [5, 2, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
    ],
    "gen": [
        [1, 40, 0, 30, -30, 1, 100, 1, 40, 0],
        [1, 170, 0, 127.5, -127.5, 1, 100, 1, 170, 0],
        [3, 323.49, 0, 390, -390, 1, 100, 1, 520, 0],
        [4, 0, 0, 150, -150, 1, 100, 1, 200, 0],
        [5, 466.51, 0, 450, -450, 1, 100, 1, 600, 0],
    ],
    "branch": [
        [1, 2, 0.00281, 0.0281, 0.00712, 400, 400, 400, 0, 0, 1, -360, 360],
        [1, 4, 0.00304, 0.0304, 0.00658, 0, 0, 0, 0, 0, 1, -360, 360],
        [1, 5, 0.00064, 0.0064, 0.03126, 0, 0, 0, 0, 0, 1, -360, 360],
        [2, 3, 0.00108, 0.0108, 0.01852, 0, 0, 0, 0, 0, 1, -360, 360],
        [3, 4, 0.00297, 0.0297, 0.00674, 0, 0, 0, 0, 0, 1, -360, 360],
        [4, 5, 0.00297, 0.0297, 0.00674, 240, 240, 240, 0, 0, 1, -360, 360],
    ],
    "gencost": [
        [2, 0, 0, 2, 14, 0],
        [2, 0, 0, 2, 15, 0],
        [2, 0, 0, 2, 30, 0],
        [2, 0, 0, 2, 40, 0],
        [2, 0, 0, 2, 10, 0],
    ],
}


def case5() -> PowerNetwork:
    """PJM 5-bus system (MATPOWER ``case5``).

    The standard teaching case for congestion pricing: costs are
    LINEAR, so the economic merit order is unambiguous, and two of the
    six branches are rated tightly enough to bind. Without the ratings
    the cheapest units would serve everything; with them, expensive
    generation is forced on behind the constraint and the locational
    marginal prices separate. Two units share bus 1, which also
    exercises the var- and MW-sharing rules of
    :mod:`difflow_power.powerflow`.
    """
    return from_matpower(_CASE5, name="case5 (PJM 5-bus)")


#: WSCC 9-bus case (MATPOWER ``case9.m``), verbatim.
_CASE9 = {
    "baseMVA": 100.0,
    "bus": [
        [1, 3, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [2, 2, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [3, 2, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [4, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [5, 1, 90, 30, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [6, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [7, 1, 100, 35, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [8, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [9, 1, 125, 50, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
    ],
    "gen": [
        [1, 0, 0, 300, -300, 1, 100, 1, 250, 10],
        [2, 163, 0, 300, -300, 1, 100, 1, 300, 10],
        [3, 85, 0, 300, -300, 1, 100, 1, 270, 10],
    ],
    "branch": [
        [1, 4, 0, 0.0576, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [4, 5, 0.017, 0.092, 0.158, 250, 250, 250, 0, 0, 1, -360, 360],
        [5, 6, 0.039, 0.170, 0.358, 150, 150, 150, 0, 0, 1, -360, 360],
        [3, 6, 0, 0.0586, 0, 300, 300, 300, 0, 0, 1, -360, 360],
        [6, 7, 0.0119, 0.1008, 0.209, 150, 150, 150, 0, 0, 1, -360, 360],
        [7, 8, 0.0085, 0.072, 0.149, 250, 250, 250, 0, 0, 1, -360, 360],
        [8, 2, 0, 0.0625, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [8, 9, 0.032, 0.161, 0.306, 250, 250, 250, 0, 0, 1, -360, 360],
        [9, 4, 0.010, 0.085, 0.176, 250, 250, 250, 0, 0, 1, -360, 360],
    ],
    "gencost": [
        [2, 1500, 0, 3, 0.11, 5, 150],
        [2, 2000, 0, 3, 0.085, 1.2, 600],
        [2, 3000, 0, 3, 0.1225, 1, 335],
    ],
}


def case9() -> PowerNetwork:
    """WSCC 3-machine 9-bus system (MATPOWER ``case9``).

    The most-used benchmark in the field: three machines behind
    step-up transformers, three loads, one loop. Small enough to check
    by hand, meshed enough to be a real power flow. MATPOWER's
    reference answers, which
    ``tests/power/test_powerflow.py`` asserts against, are
    ``Pg = (71.95, 163.00, 85.00)`` MW with 4.955 MW of losses for the
    power flow, and an OPF optimum of $5296.69/h.
    """
    return from_matpower(_CASE9, name="case9 (WSCC 9-bus)")


#: IEEE 14-bus case (MATPOWER ``case14.m``), verbatim.
_CASE14 = {
    "baseMVA": 100.0,
    "bus": [
        [1, 3, 0, 0, 0, 0, 1, 1.06, 0, 0, 1, 1.06, 0.94],
        [2, 2, 21.7, 12.7, 0, 0, 1, 1.045, -4.98, 0, 1, 1.06, 0.94],
        [3, 2, 94.2, 19, 0, 0, 1, 1.01, -12.72, 0, 1, 1.06, 0.94],
        [4, 1, 47.8, -3.9, 0, 0, 1, 1.019, -10.33, 0, 1, 1.06, 0.94],
        [5, 1, 7.6, 1.6, 0, 0, 1, 1.02, -8.78, 0, 1, 1.06, 0.94],
        [6, 2, 11.2, 7.5, 0, 0, 1, 1.07, -14.22, 0, 1, 1.06, 0.94],
        [7, 1, 0, 0, 0, 0, 1, 1.062, -13.37, 0, 1, 1.06, 0.94],
        [8, 2, 0, 0, 0, 0, 1, 1.09, -13.36, 0, 1, 1.06, 0.94],
        [9, 1, 29.5, 16.6, 0, 19, 1, 1.056, -14.94, 0, 1, 1.06, 0.94],
        [10, 1, 9, 5.8, 0, 0, 1, 1.051, -15.1, 0, 1, 1.06, 0.94],
        [11, 1, 3.5, 1.8, 0, 0, 1, 1.057, -14.79, 0, 1, 1.06, 0.94],
        [12, 1, 6.1, 1.6, 0, 0, 1, 1.055, -15.07, 0, 1, 1.06, 0.94],
        [13, 1, 13.5, 5.8, 0, 0, 1, 1.05, -15.16, 0, 1, 1.06, 0.94],
        [14, 1, 14.9, 5, 0, 0, 1, 1.036, -16.04, 0, 1, 1.06, 0.94],
    ],
    "gen": [
        [1, 232.4, -16.9, 10, 0, 1.06, 100, 1, 332.4, 0],
        [2, 40, 42.4, 50, -40, 1.045, 100, 1, 140, 0],
        [3, 0, 23.4, 40, 0, 1.01, 100, 1, 100, 0],
        [6, 0, 12.2, 24, -6, 1.07, 100, 1, 100, 0],
        [8, 0, 17.4, 24, -6, 1.09, 100, 1, 100, 0],
    ],
    "branch": [
        [1, 2, 0.01938, 0.05917, 0.0528, 0, 0, 0, 0, 0, 1, -360, 360],
        [1, 5, 0.05403, 0.22304, 0.0492, 0, 0, 0, 0, 0, 1, -360, 360],
        [2, 3, 0.04699, 0.19797, 0.0438, 0, 0, 0, 0, 0, 1, -360, 360],
        [2, 4, 0.05811, 0.17632, 0.034, 0, 0, 0, 0, 0, 1, -360, 360],
        [2, 5, 0.05695, 0.17388, 0.0346, 0, 0, 0, 0, 0, 1, -360, 360],
        [3, 4, 0.06701, 0.17103, 0.0128, 0, 0, 0, 0, 0, 1, -360, 360],
        [4, 5, 0.01335, 0.04211, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [4, 7, 0, 0.20912, 0, 0, 0, 0, 0.978, 0, 1, -360, 360],
        [4, 9, 0, 0.55618, 0, 0, 0, 0, 0.969, 0, 1, -360, 360],
        [5, 6, 0, 0.25202, 0, 0, 0, 0, 0.932, 0, 1, -360, 360],
        [6, 11, 0.09498, 0.1989, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [6, 12, 0.12291, 0.25581, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [6, 13, 0.06615, 0.13027, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [7, 8, 0, 0.17615, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [7, 9, 0, 0.11001, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [9, 10, 0.03181, 0.0845, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [9, 14, 0.12711, 0.27038, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [10, 11, 0.08205, 0.19207, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [12, 13, 0.22092, 0.19988, 0, 0, 0, 0, 0, 0, 1, -360, 360],
        [13, 14, 0.17093, 0.34802, 0, 0, 0, 0, 0, 0, 1, -360, 360],
    ],
    "gencost": [
        [2, 0, 0, 3, 0.0430292599, 20, 0],
        [2, 0, 0, 3, 0.25, 20, 0],
        [2, 0, 0, 3, 0.01, 40, 0],
        [2, 0, 0, 3, 0.01, 40, 0],
        [2, 0, 0, 3, 0.01, 40, 0],
    ],
}


def case14() -> PowerNetwork:
    """IEEE 14-bus system (MATPOWER ``case14``).

    Worth having because of what the smaller cases lack: three
    tap-changing transformers (branches 4-7, 4-9 and 5-6, with taps
    0.978, 0.969 and 0.932) and a 19 MVAr shunt capacitor at bus 9. If
    the tap convention or the shunt sign is wrong, this case says so
    and ``case9`` does not. Four of the five units have no MW capacity
    to speak of and exist to hold voltage, which is a fair picture of a
    real system.
    """
    return from_matpower(_CASE14, name="case14 (IEEE 14-bus)")


def radial_feeder() -> PowerNetwork:
    """A 7-bus radial distribution feeder at 12.47 kV.

    Loops: none. That matters --- a radial network is the one topology
    where the sequential forward/backward sweep of
    :mod:`difflow_power.flowsheet` applies, so this is the case those
    tests use. The impedances are typical overhead distribution values
    (high R/X ratio, about 1:1, unlike transmission's 1:10), which is
    also why distribution voltage profiles sag on REAL power and
    transmission ones on reactive.

    A single substation source at bus ``s`` feeds a trunk of four buses
    with two laterals hanging off it.
    """
    return PowerNetwork(
        name="radial feeder (12.47 kV)",
        base_mva=10.0,
        buses={
            "s": Bus(kind="slack", base_kv=12.47, vm_setpoint=1.02,
                     vm_min=0.95, vm_max=1.05, name="substation"),
            "n1": Bus(base_kv=12.47, vm_min=0.95, vm_max=1.05),
            "n2": Bus(base_kv=12.47, vm_min=0.95, vm_max=1.05),
            "n3": Bus(base_kv=12.47, vm_min=0.95, vm_max=1.05),
            "n4": Bus(base_kv=12.47, vm_min=0.95, vm_max=1.05),
            "l1": Bus(base_kv=12.47, vm_min=0.95, vm_max=1.05,
                      name="lateral 1"),
            "l2": Bus(base_kv=12.47, vm_min=0.95, vm_max=1.05,
                      name="lateral 2"),
        },
        branches={
            "t0": Branch("s", "n1", 0.030, 0.035, 0.0, name="trunk 0"),
            "t1": Branch("n1", "n2", 0.045, 0.050, 0.0, name="trunk 1"),
            "t2": Branch("n2", "n3", 0.060, 0.062, 0.0, name="trunk 2"),
            "t3": Branch("n3", "n4", 0.055, 0.058, 0.0, name="trunk 3"),
            "b1": Branch("n2", "l1", 0.090, 0.080, 0.0, name="lateral 1"),
            "b2": Branch("n3", "l2", 0.110, 0.095, 0.0, name="lateral 2"),
        },
        generators={
            "sub": Generator("s", -50.0, 50.0, -30.0, 30.0,
                             cost=(0.0, 55.0, 0.0), p_mw=5.0,
                             name="grid supply"),
        },
        loads={
            "d1": Load("n1", 0.60, 0.25),
            "d2": Load("n2", 0.90, 0.40),
            "d3": Load("n3", 0.75, 0.30),
            "d4": Load("n4", 1.10, 0.45),
            "dl1": Load("l1", 0.50, 0.20),
            "dl2": Load("l2", 0.65, 0.28),
        },
    )


#: every benchmark case, by name
CASES = {
    "case3": case3,
    "case5": case5,
    "case9": case9,
    "case14": case14,
    "radial_feeder": radial_feeder,
}


def load_case(name: str) -> PowerNetwork:
    """Build a benchmark case by name; see :data:`CASES`."""
    try:
        return CASES[name]()
    except KeyError:
        raise KeyError(
            f"unknown case {name!r}; available: {sorted(CASES)}"
        ) from None
