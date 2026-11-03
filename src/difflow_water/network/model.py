"""Water distribution network model: nodes, links, and their compiled arrays.

A :class:`WaterNetwork` holds junctions, reservoirs, tanks, pipes, pumps and
valves in SI units (m, m^3/s). It is a plain description: no solving here.
:meth:`WaterNetwork.compile` turns it into :class:`NetworkArrays`, the index
arrays and parameter vectors the equation set in
:mod:`difflow_water.network.hydraulics` works on.

Conventions follow EPANET 2.2 [N8]: link flow is positive from the start node
to the end node, head loss ``h = H_start - H_end``; a junction's demand and
emitter flow leave the network; tanks are fixed-head nodes at
``elevation + level`` in a steady-state (single period) solve.

Example:
    >>> net = WaterNetwork()
    >>> net.add_reservoir("R", head=100.0)
    >>> net.add_junction("J", elevation=20.0, demand=0.01)
    >>> net.add_pipe("P", "R", "J", length=1000.0, diameter=0.2, roughness=120.0)
    >>> sorted(net.junctions)
    ['J']
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HEADLOSS_MODELS = ("H-W", "D-W")
VALVE_TYPES = ("PRV", "PSV", "FCV", "TCV")


@dataclass
class Junction:
    """A demand node. ``demand`` (m^3/s, leaving the network) at ``elevation`` (m).

    ``emitter`` is the discharge coefficient C of an emitter (or leak),
    ``q = C p^emitter_exponent`` with p the pressure head (m); 0 for none.
    """

    elevation: float
    demand: float = 0.0
    emitter: float = 0.0
    emitter_exponent: float = 0.5


@dataclass
class Reservoir:
    """A fixed-head node (m)."""

    head: float


@dataclass
class Tank:
    """A storage tank; a fixed head of ``elevation + level`` in a steady solve."""

    elevation: float
    level: float
    min_level: float = 0.0
    max_level: float = float("inf")
    diameter: float = 0.0


@dataclass
class Pipe:
    """A pipe from ``start`` to ``end``.

    ``roughness`` is the Hazen-Williams C (H-W) or the absolute roughness in
    m (D-W). ``minor_loss`` is K on the velocity head. ``check_valve`` blocks
    reverse flow; ``status`` "open" or "closed".
    """

    start: str
    end: str
    length: float
    diameter: float
    roughness: float
    minor_loss: float = 0.0
    check_valve: bool = False
    status: str = "open"


@dataclass
class Pump:
    """A pump, head gain ``h_G = A - B q^C`` (m, m^3/s), times ``speed``^2 laws.

    Build the curve with :func:`pump_curve`. Pumps do not run backwards
    (as in EPANET, reverse flow closes them). A constant-power pump [N8]
    has ``power`` (W) > 0 and head gain ``power / (rho g Q)``; its curve
    coefficients are ignored.
    """

    start: str
    end: str
    A: float = 0.0
    B: float = 0.0
    C: float = 1.0
    speed: float = 1.0
    status: str = "open"
    power: float = 0.0


@dataclass
class Valve:
    """A control valve: PRV, PSV (setting: pressure head, m), FCV (m^3/s), TCV (K).

    Open, a valve is a smooth pipe of ``diameter`` with ``minor_loss`` (and,
    as in EPANET, an ``f = 0.02`` friction loss over twice the diameter).
    """

    start: str
    end: str
    kind: str
    diameter: float
    setting: float
    minor_loss: float = 0.0
    status: str = "active"


def pump_curve(points) -> tuple[float, float, float]:
    """``(A, B, C)`` of ``h = A - B q^C`` from EPANET's 1- or 3-point pump curves.

    One point ``(q_d, h_d)``: EPANET adds a shutoff head of 133 % of the
    design head at zero flow and zero head at twice the design flow [N8]
    (taken here as 4/3, which makes the exponent exactly 2). Three points,
    the first at zero flow: ``A = h_0``, ``C`` and ``B`` through the other two.

    Args:
        points: ``[(q, h)]`` in m^3/s and m.

    Returns:
        ``(A, B, C)``.
    """
    pts = [(float(q), float(h)) for q, h in points]
    if len(pts) == 1:
        qd, hd = pts[0]
        pts = [(0.0, 4.0 / 3.0 * hd), (qd, hd), (2.0 * qd, 0.0)]
    if len(pts) != 3 or pts[0][0] != 0.0:
        raise ValueError("give one (q, h) point, or three with the first at q = 0")
    (_, h0), (q1, h1), (q2, h2) = pts
    C = np.log((h0 - h1) / (h0 - h2)) / np.log(q1 / q2)
    B = (h0 - h1) / q1**C
    return h0, B, float(C)


@dataclass
class NetworkArrays:
    """The compiled network (numpy index arrays + parameter vectors).

    Node indices run over junctions first, then fixed-head nodes
    (reservoirs, then tanks). Link kind codes: 0 pipe, 1 pump, 2 PRV,
    3 PSV, 4 FCV, 5 TCV.
    """

    junctions: list
    fixed: list
    links: list
    elevation: np.ndarray
    demand: np.ndarray
    emitter: np.ndarray
    emitter_exponent: np.ndarray
    fixed_head: np.ndarray
    start: np.ndarray
    end: np.ndarray
    kind: np.ndarray
    length: np.ndarray
    diameter: np.ndarray
    roughness: np.ndarray
    minor_loss: np.ndarray
    check: np.ndarray
    closed: np.ndarray
    pump_A: np.ndarray
    pump_B: np.ndarray
    pump_C: np.ndarray
    speed: np.ndarray
    setting: np.ndarray
    pump_power: np.ndarray
    headloss: str
    auto_closed: list = field(default_factory=list)

    @property
    def n_junctions(self) -> int:
        return len(self.junctions)

    @property
    def n_links(self) -> int:
        return len(self.links)


@dataclass
class WaterNetwork:
    """A water distribution network (SI units).

    Attributes:
        headloss: "H-W" (Hazen-Williams) or "D-W" (Darcy-Weisbach).
        viscosity: Kinematic viscosity for D-W (m^2/s).
    """

    junctions: dict = field(default_factory=dict)
    reservoirs: dict = field(default_factory=dict)
    tanks: dict = field(default_factory=dict)
    pipes: dict = field(default_factory=dict)
    pumps: dict = field(default_factory=dict)
    valves: dict = field(default_factory=dict)
    headloss: str = "H-W"
    viscosity: float = 1.0e-6

    # -- building -------------------------------------------------------
    def add_junction(self, name, elevation, demand=0.0, emitter=0.0, emitter_exponent=0.5):
        self.junctions[name] = Junction(elevation, demand, emitter, emitter_exponent)

    def add_reservoir(self, name, head):
        self.reservoirs[name] = Reservoir(head)

    def add_tank(self, name, elevation, level, **kw):
        self.tanks[name] = Tank(elevation, level, **kw)

    def add_pipe(self, name, start, end, length, diameter, roughness, **kw):
        self.pipes[name] = Pipe(start, end, length, diameter, roughness, **kw)

    def add_pump(self, name, start, end, curve=None, speed=1.0, status="open", power=0.0):
        if (curve is None) == (power <= 0.0):
            raise ValueError("give a pump curve or a constant power, not both")
        A, B, C = pump_curve(curve) if curve is not None else (0.0, 0.0, 1.0)
        self.pumps[name] = Pump(start, end, A, B, C, speed, status, power)

    def add_valve(self, name, start, end, kind, diameter, setting, **kw):
        if kind not in VALVE_TYPES:
            raise ValueError(f"valve kind must be one of {VALVE_TYPES}")
        self.valves[name] = Valve(start, end, kind, diameter, setting, **kw)

    # -- serialization --------------------------------------------------
    def to_dict(self) -> dict:
        from dataclasses import asdict
        return {
            "headloss": self.headloss, "viscosity": self.viscosity,
            **{k: {n: asdict(v) for n, v in getattr(self, k).items()}
               for k in ("junctions", "reservoirs", "tanks", "pipes", "pumps", "valves")},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WaterNetwork":
        net = cls(headloss=d.get("headloss", "H-W"), viscosity=d.get("viscosity", 1.0e-6))
        for key, typ in (("junctions", Junction), ("reservoirs", Reservoir), ("tanks", Tank),
                         ("pipes", Pipe), ("pumps", Pump), ("valves", Valve)):
            for name, rec in d.get(key, {}).items():
                getattr(net, key)[name] = typ(**{k: v for k, v in rec.items()
                                                 if not k.startswith("_")})
        return net

    # -- compiling -------------------------------------------------------
    def compile(self) -> NetworkArrays:
        """Index arrays and parameter vectors for the equation set."""
        if self.headloss not in HEADLOSS_MODELS:
            raise ValueError(f"headloss must be one of {HEADLOSS_MODELS}")
        if not (self.reservoirs or self.tanks):
            raise ValueError("a network needs at least one reservoir or tank")
        J = list(self.junctions)
        F = list(self.reservoirs) + list(self.tanks)
        idx = {n: i for i, n in enumerate(J + F)}
        links, rows = [], []
        for n, p in self.pipes.items():
            links.append(n)
            rows.append((p.start, p.end, 0, p.length, p.diameter, p.roughness, p.minor_loss,
                         p.check_valve, p.status == "closed", 0.0, 0.0, 1.0, 1.0, 0.0, 0.0))
        for n, p in self.pumps.items():
            links.append(n)
            rows.append((p.start, p.end, 1, 0.0, 1.0, 0.0, 0.0, True, p.status == "closed",
                         p.A, p.B, p.C, p.speed, 0.0, p.power))
        code = {"PRV": 2, "PSV": 3, "FCV": 4, "TCV": 5}
        for n, v in self.valves.items():
            links.append(n)
            # TCV: the setting is the minor-loss coefficient (EPANET)
            K = v.setting if v.kind == "TCV" else v.minor_loss
            rows.append((v.start, v.end, code[v.kind], 2.0 * v.diameter, v.diameter, 0.0, K,
                         v.kind in ("PRV", "PSV"), v.status == "closed", 0.0, 0.0, 1.0, 1.0,
                         v.setting, 0.0))
        for r in rows:
            for node in r[:2]:
                if node not in idx:
                    raise ValueError(f"link endpoint {node!r} is not a node of the network")
        auto_closed = self._dead_end_pumps(links, rows, idx, J, F)
        if auto_closed:
            rows = [r[:8] + (True,) + r[9:] if n in auto_closed else r for n, r in zip(links, rows)]
        a = lambda i, dt=float: np.array([r[i] for r in rows], dtype=dt)  # noqa: E731
        return NetworkArrays(
            junctions=J, fixed=F, links=links,
            elevation=np.array([self.junctions[n].elevation for n in J], float),
            demand=np.array([self.junctions[n].demand for n in J], float),
            emitter=np.array([self.junctions[n].emitter for n in J], float),
            emitter_exponent=np.array([self.junctions[n].emitter_exponent for n in J], float),
            fixed_head=np.array([self.reservoirs[n].head for n in self.reservoirs]
                                + [self.tanks[n].elevation + self.tanks[n].level for n in self.tanks],
                                float),
            start=np.array([idx[r[0]] for r in rows], int),
            end=np.array([idx[r[1]] for r in rows], int),
            kind=a(2, int), length=a(3), diameter=a(4), roughness=a(5), minor_loss=a(6),
            check=a(7, bool), closed=a(8, bool), pump_A=a(9), pump_B=a(10), pump_C=a(11),
            speed=a(12), setting=a(13), pump_power=a(14), headloss=self.headloss,
            auto_closed=auto_closed,
        )

    def _dead_end_pumps(self, links, rows, idx, J, F) -> list:
        """Pumps that can deliver no flow: their outlet side, without them and
        without closed links, holds no demand, emitter, reservoir or tank.

        EPANET's status check closes such a pump (a constant-power pump with
        nowhere to send water would otherwise raise an unbounded head), so
        they are compiled as closed; :attr:`NetworkArrays.auto_closed` lists them.
        """
        sink = np.zeros(len(J) + len(F), bool)
        sink[len(J):] = True
        for i, n in enumerate(J):
            jn = self.junctions[n]
            sink[i] = jn.demand != 0.0 or jn.emitter != 0.0
        out = []
        for k, (name, r) in enumerate(zip(links, rows)):
            if r[2] != 1 or r[8]:
                continue
            parent = list(range(len(sink)))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for j, rr in enumerate(rows):
                if j != k and not rr[8]:
                    parent[find(idx[rr[0]])] = find(idx[rr[1]])
            root = find(idx[r[1]])
            if not any(sink[i] for i in range(len(sink)) if find(i) == root):
                out.append(name)
        return out
