"""Read EPANET input (INP) files into a WaterNetwork and its Schedule.

The INP format and its units are those of the EPANET 2.2 manual [N8]
(Appendix A "Units of Measurement" and the input-file chapter; content
verified). Everything is converted to SI (m, m^3/s, W, s):

- US customary units when the flow units are CFS, GPM, MGD, IMGD or AFD
  (lengths and heads in ft, pipe diameters in inches, pressures in psi,
  D-W roughness in 1e-3 ft, power in hp); SI otherwise (m, mm, m, mm, kW).
- Flow units, from their definitions (US gallon 3.785411784 L, Imperial
  gallon 4.54609 L, acre-foot 1233.48183754752 m^3).
- psi to metres of head with EPANET's 0.4333 psi/ft, as WNTR converts (the
  EPANET source was not read: unverified).

Read: TITLE, JUNCTIONS, RESERVOIRS, TANKS, PIPES, PUMPS (HEAD curves,
POWER, SPEED, PATTERN), VALVES (PRV, PSV, FCV, TCV), DEMANDS, EMITTERS,
STATUS, PATTERNS, CURVES, CONTROLS (simple controls), TIMES, OPTIONS
(UNITS, HEADLOSS, VISCOSITY, PATTERN, DEMAND MULTIPLIER, EMITTER EXPONENT),
QUALITY, REACTIONS (kept on the schedule for water quality). Not read (an
error if present and non-empty): rule-based controls, tank volume curves,
multi-point pump curves, GPV and PBV valves, Chezy-Manning head loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from difflow_water.network.model import Pump, Valve, WaterNetwork, pump_curve

FT = 0.3048
INCH = 0.0254
GAL = 3.785411784e-3
IGAL = 4.54609e-3
FLOW_UNITS = {
    "CFS": FT**3, "GPM": GAL / 60.0, "MGD": 1e6 * GAL / 86400.0, "IMGD": 1e6 * IGAL / 86400.0,
    "AFD": 1233.48183754752 / 86400.0,
    "LPS": 1e-3, "LPM": 1e-3 / 60.0, "MLD": 1e6 * 1e-3 / 86400.0, "CMH": 1.0 / 3600.0,
    "CMD": 1.0 / 86400.0,
}
US_UNITS = ("CFS", "GPM", "MGD", "IMGD", "AFD")
PSI_TO_M = FT / 0.4333
HP = 745.699872


@dataclass
class Control:
    """A simple control: set ``link`` to ``status`` ("open"/"closed") or ``setting``
    when ``kind`` ("above", "below", "time", "clocktime") holds."""

    link: str
    status: str | None
    setting: float | None
    kind: str
    node: str | None = None
    value: float = 0.0


@dataclass
class Schedule:
    """The time-varying part of an EPANET model (SI, seconds).

    Attributes:
        patterns: name -> multipliers.
        demands: junction -> [(base demand m^3/s, pattern name or None)].
        reservoir_patterns: reservoir -> pattern name (head multipliers).
        pump_patterns: pump -> pattern name (speed multipliers).
        controls: simple controls.
        duration, hydraulic_step, pattern_step, pattern_start, report_step,
        start_clocktime: times (s).
        default_pattern: pattern for demands with none (EPANET: "1" if it exists).
        demand_multiplier: global demand multiplier.
        quality, reactions: raw quality data, for water quality.
    """

    patterns: dict = field(default_factory=dict)
    demands: dict = field(default_factory=dict)
    reservoir_patterns: dict = field(default_factory=dict)
    pump_patterns: dict = field(default_factory=dict)
    controls: list = field(default_factory=list)
    duration: float = 0.0
    hydraulic_step: float = 3600.0
    pattern_step: float = 3600.0
    pattern_start: float = 0.0
    report_step: float = 3600.0
    start_clocktime: float = 0.0
    default_pattern: str | None = None
    demand_multiplier: float = 1.0
    quality: dict = field(default_factory=dict)
    reactions: dict = field(default_factory=dict)

    def multiplier(self, pattern: str | None, t: float) -> float:
        """Pattern multiplier at time t (s); 1 for no pattern."""
        if pattern is None or pattern not in self.patterns:
            return 1.0
        m = self.patterns[pattern]
        k = int((t + self.pattern_start) // self.pattern_step) % len(m)
        return m[k]

    def demand(self, junction: str, t: float) -> float:
        """Total demand of a junction at time t (m^3/s)."""
        return self.demand_multiplier * sum(
            base * self.multiplier(pat if pat is not None else self.default_pattern, t)
            for base, pat in self.demands.get(junction, []))


def _seconds(text: str, unit: str | None = None) -> float:
    """EPANET time: "h:mm[:ss]", or a number with an optional unit."""
    t = text.strip().upper()
    if ":" in t:
        parts = [float(p) for p in t.split(":")]
        while len(parts) < 3:
            parts.append(0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    v = float(t)
    u = (unit or "HOURS").upper()
    return v * {"SEC": 1, "SECONDS": 1, "MIN": 60, "MINUTES": 60, "HOURS": 3600,
                "HOUR": 3600, "DAYS": 86400, "DAY": 86400}.get(u, 3600)


def _clock(words: list[str]) -> float:
    t = _seconds(words[0])
    if len(words) > 1 and words[1].upper() == "PM" and t < 12 * 3600:
        t += 12 * 3600
    if len(words) > 1 and words[1].upper() == "AM" and t >= 12 * 3600:
        t -= 12 * 3600
    return t


def _sections(text: str) -> dict:
    out, cur = {}, None
    for raw in text.splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = line[1:-1].upper()
            out.setdefault(cur, [])
            continue
        if cur is not None:
            out[cur].append(line.split())
    return out


def read_inp(path) -> tuple[WaterNetwork, Schedule]:
    """Read an EPANET INP file.

    Args:
        path: The INP file.

    Returns:
        ``(network, schedule)``: the network at its initial state (base
        demands times their t = 0 multipliers, initial tank levels, initial
        statuses) and the time-varying data for extended-period simulation.
    """
    sec = _sections(Path(path).read_text(errors="replace"))
    opts = {}
    for w in sec.get("OPTIONS", []):
        key = w[0].upper()
        if key in ("SPECIFIC", "DEMAND", "EMITTER", "MINIMUM", "REQUIRED", "PRESSURE"):
            key = " ".join(x.upper() for x in w[:2])
            opts[key] = w[2:]
        else:
            opts[key] = w[1:]
    units = opts.get("UNITS", ["GPM"])[0].upper()
    if units not in FLOW_UNITS:
        raise ValueError(f"unknown flow units {units!r}")
    us = units in US_UNITS
    qf = FLOW_UNITS[units]
    L = FT if us else 1.0                    # length, elevation, head
    D = INCH if us else 1e-3                 # pipe diameter
    P = PSI_TO_M if us else 1.0              # pressure -> m of head
    headloss = opts.get("HEADLOSS", ["H-W"])[0].upper()
    if headloss not in ("H-W", "D-W"):
        raise ValueError(f"head loss {headloss!r} is not supported (H-W, D-W)")
    for unsupported in ("RULES",):
        if sec.get(unsupported):
            raise ValueError(f"[{unsupported}] is not supported")
    viscosity = 1.0e-6 * float(opts.get("VISCOSITY", ["1.0"])[0])
    emitter_exp = float(opts.get("EMITTER EXPONENT", ["0.5"])[0])
    net = WaterNetwork(headloss=headloss, viscosity=viscosity)
    sch = Schedule(demand_multiplier=float(opts.get("DEMAND MULTIPLIER", ["1.0"])[0]))

    for w in sec.get("PATTERNS", []):
        sch.patterns.setdefault(w[0], []).extend(float(x) for x in w[1:])
    curves = {}
    for w in sec.get("CURVES", []):
        curves.setdefault(w[0], []).append((float(w[1]), float(w[2])))
    sch.default_pattern = opts.get("PATTERN", [None])[0] or ("1" if "1" in sch.patterns else None)

    for w in sec.get("JUNCTIONS", []):
        base = float(w[2]) * qf if len(w) > 2 else 0.0
        pat = w[3] if len(w) > 3 else None
        net.add_junction(w[0], elevation=float(w[1]) * L, emitter_exponent=emitter_exp)
        sch.demands[w[0]] = [(base, pat)] if base != 0.0 else []
    if sec.get("DEMANDS"):
        seen = set()
        for w in sec["DEMANDS"]:
            if w[0] not in seen:            # [DEMANDS] replaces the junction's demand
                sch.demands[w[0]] = []
                seen.add(w[0])
            sch.demands[w[0]].append((float(w[1]) * qf, w[2] if len(w) > 2 else None))
    for w in sec.get("RESERVOIRS", []):
        net.add_reservoir(w[0], head=float(w[1]) * L)
        if len(w) > 2:
            sch.reservoir_patterns[w[0]] = w[2]
    for w in sec.get("TANKS", []):
        if len(w) > 7 and w[7] not in ("*", ""):
            raise ValueError(f"tank {w[0]}: volume curves are not supported")
        net.add_tank(w[0], elevation=float(w[1]) * L, level=float(w[2]) * L,
                     min_level=float(w[3]) * L, max_level=float(w[4]) * L,
                     diameter=float(w[5]) * L)
    for w in sec.get("PIPES", []):
        status = w[7].upper() if len(w) > 7 else "OPEN"
        rough = float(w[5])
        if headloss == "D-W":
            rough *= 1e-3 * FT if us else 1e-3
        net.add_pipe(w[0], w[1], w[2], length=float(w[3]) * L, diameter=float(w[4]) * D,
                     roughness=rough, minor_loss=float(w[6]) if len(w) > 6 else 0.0,
                     check_valve=status == "CV", status="closed" if status == "CLOSED" else "open")
    for w in sec.get("PUMPS", []):
        kv = {w[i].upper(): w[i + 1] for i in range(3, len(w) - 1, 2)}
        speed = float(kv.get("SPEED", 1.0))
        if "POWER" in kv:
            power = float(kv["POWER"]) * (HP if us else 1e3)
            net.pumps[w[0]] = Pump(w[1], w[2], speed=speed, power=power)
        elif "HEAD" in kv:
            pts = [(q * qf, h * L) for q, h in curves[kv["HEAD"]]]
            if len(pts) not in (1, 3):
                raise ValueError(f"pump {w[0]}: only 1- and 3-point curves are supported")
            A, B, C = pump_curve(pts)
            net.pumps[w[0]] = Pump(w[1], w[2], A, B, C, speed=speed)
        else:
            raise ValueError(f"pump {w[0]} has neither HEAD nor POWER")
        if "PATTERN" in kv:
            sch.pump_patterns[w[0]] = kv["PATTERN"]
    for w in sec.get("VALVES", []):
        kind = w[4].upper()
        if kind not in ("PRV", "PSV", "FCV", "TCV"):
            raise ValueError(f"valve {w[0]}: {kind} valves are not supported")
        s = float(w[5])
        setting = s * P if kind in ("PRV", "PSV") else s * qf if kind == "FCV" else s
        net.valves[w[0]] = Valve(w[1], w[2], kind, float(w[3]) * D, setting,
                                 minor_loss=float(w[6]) if len(w) > 6 else 0.0)
    for w in sec.get("EMITTERS", []):
        # flow units / (pressure unit)^gamma -> m^3/s / m^gamma
        net.junctions[w[0]].emitter = float(w[1]) * qf / P**emitter_exp
    for w in sec.get("STATUS", []):
        name, val = w[0], w[1].upper()
        if name in net.pipes:
            net.pipes[name].status = "closed" if val == "CLOSED" else "open"
        elif name in net.pumps:
            if val in ("OPEN", "CLOSED"):
                net.pumps[name].status = val.lower()
            else:
                net.pumps[name].speed = float(val)
        elif name in net.valves:
            if val in ("OPEN", "CLOSED"):
                net.valves[name].status = val.lower()
            else:
                v = net.valves[name]
                s_ = float(val)
                v.setting = s_ * P if v.kind in ("PRV", "PSV") else s_ * qf if v.kind == "FCV" else s_
    for w in sec.get("CONTROLS", []):
        sch.controls.append(_control(w, net, L, P, qf))
    for w in sec.get("TIMES", []):
        words = [x.upper() for x in w]
        if words[0] == "DURATION":
            sch.duration = _seconds(w[1], w[2] if len(w) > 2 else None)
        elif words[:2] == ["HYDRAULIC", "TIMESTEP"]:
            sch.hydraulic_step = _seconds(w[2], w[3] if len(w) > 3 else None)
        elif words[:2] == ["PATTERN", "TIMESTEP"]:
            sch.pattern_step = _seconds(w[2], w[3] if len(w) > 3 else None)
        elif words[:2] == ["PATTERN", "START"]:
            sch.pattern_start = _seconds(w[2], w[3] if len(w) > 3 else None)
        elif words[:2] == ["REPORT", "TIMESTEP"]:
            sch.report_step = _seconds(w[2], w[3] if len(w) > 3 else None)
        elif words[:2] == ["START", "CLOCKTIME"]:
            sch.start_clocktime = _clock(w[2:])
    sch.quality = {w[0]: float(w[1]) for w in sec.get("QUALITY", [])}
    sch.reactions = [w for w in sec.get("REACTIONS", [])]
    # initial demands and reservoir heads
    for j in net.junctions:
        net.junctions[j].demand = sch.demand(j, 0.0)
    for r, pat in sch.reservoir_patterns.items():
        net.reservoirs[r].head *= sch.multiplier(pat, 0.0)
    return net, sch


def _control(w, net, L, P, qf) -> Control:
    """``LINK id status|setting IF NODE id ABOVE|BELOW value`` or ``... AT TIME|CLOCKTIME t``."""
    words = [x.upper() for x in w]
    link = w[1]
    val = words[2]
    status = val.lower() if val in ("OPEN", "CLOSED") else None
    setting = None if status else float(w[2])
    if "IF" in words:
        i = words.index("IF")
        node, rel, v = w[i + 2], words[i + 3], float(w[i + 4])
        # tank/junction level or pressure: tanks in length units, junctions in pressure units
        scale = L if node in net.tanks or node in net.reservoirs else P
        return Control(link, status, setting, rel.lower(), node, v * scale)
    i = words.index("AT")
    if words[i + 1] == "TIME":
        return Control(link, status, setting, "time", None,
                       _seconds(w[i + 2], w[i + 3] if len(w) > i + 3 else None))
    return Control(link, status, setting, "clocktime", None, _clock(w[i + 2:]))
