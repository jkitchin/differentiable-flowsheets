"""Node unit operations: sources, loads, shunts, generators, junctions.

Everything a branch connects. Between them they carry the whole of a
network's non-branch physics, which is less than it sounds: a bus is a
place where complex powers sum and one voltage is shared, and the only
component with any nonlinearity of its own is the constant-power load.

That load is where a power flow's difficulty actually lives. A constant
IMPEDANCE would make the whole system linear in ``V``; a constant
CURRENT would make it linear in the voltage phasor. Constant power ---
what a motor, an inverter or a tariff-driven consumer actually behaves
like on the timescale of a power flow --- makes ``S = V conj(Y V)``, and
that is the nonlinearity Newton's method is spending its iterations on.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream

from difflow_power.physics import polynomial_cost
from difflow_power.streams import (
    complex_power,
    complex_voltage,
    from_complex,
    power_stream,
)


#: shared literature references for the node units
_NODE_REFS = [
    "Wood, A.J., Wollenberg, B.F., Sheble, G.B., Power Generation, "
    "Operation, and Control, 3rd ed., Wiley (2013).",
    "Grainger, J.J., Stevenson, W.D., Power System Analysis, "
    "McGraw-Hill (1994), ch. 9.",
]

#: references for the sequential feeder units, whose tear and split
#: conventions come from the distribution-sweep literature rather than
#: from transmission practice
_FEEDER_REFS = [
    "Kersting, W.H., Distribution System Modeling and Analysis, "
    "4th ed., CRC Press (2017).",
    "Shirmohammadi, D. et al., IEEE Trans. Power Syst. 3(2), "
    "753-762 (1988).",
]


@dataclass
class SlackSourceParams(ParamsMixin):
    """Parameters of a slack source.

    Attributes:
        vm_setpoint: regulated voltage magnitude (pu).
        va_reference: the angle everything else is measured from (rad).
    """

    vm_setpoint: float = 1.0
    va_reference: float = 0.0


class SlackSource:
    """Pin a feed to a regulated voltage, passing its power through.

    The electrical analogue of the gas plugin's ``SourceHead``: it fixes
    the potential and lets the flow be whatever the network demands. In
    a sequential feeder solve the power it carries is the tear variable
    --- the substation infeed, which must come out equal to the total
    load plus the losses nobody knows until the solve is done.
    """

    symbol = "Slack"
    equations = [
        r"|V| = V^\mathrm{set},\quad \theta = \theta^\mathrm{ref}",
        r"S_\mathrm{out} = S_\mathrm{in}",
    ]
    assumptions = [
        "The source holds its voltage for any power drawn from it "
        "(infinite short-circuit capacity)",
    ]
    references = _NODE_REFS
    parameter_units = {"vm_setpoint": "pu", "va_reference": "rad"}

    def __init__(self, params: SlackSourceParams):
        self.params = params

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        outlet = power_stream(
            inlet["F_P"],
            inlet["F_Q"],
            self.params.vm_setpoint,
            self.params.va_reference,
        )
        return outlet, {
            "vm": jnp.asarray(self.params.vm_setpoint),
            "supplied_p": jnp.asarray(inlet["F_P"]),
            "supplied_q": jnp.asarray(inlet["F_Q"]),
        }


@dataclass
class LoadParams(ParamsMixin):
    """Parameters of a constant-power load.

    Attributes:
        p_pu: real demand (pu), positive drawing power.
        q_pu: reactive demand (pu), positive drawing lagging vars.
    """

    p_pu: float = 0.0
    q_pu: float = 0.0


class LoadDraw:
    """Subtract a constant-power demand from a stream.

    Constant power means constant regardless of the voltage it is served
    at, which is what makes it the hard component: as voltage sags the
    current rises to compensate, which sags the voltage further. That
    positive feedback is what a P-V curve's nose is, and why a heavily
    loaded feeder has no solution rather than a poor one.

    ``info["current"]`` reports the current drawn at the stream's own
    voltage, which is what a backward sweep accumulates.
    """

    symbol = "Load"
    equations = [
        r"S_\mathrm{out} = S_\mathrm{in} - (P_d + jQ_d)",
        r"I_d = \overline{(P_d + jQ_d)/V}",
    ]
    assumptions = [
        "Constant power: the demand does not vary with the voltage it "
        "is served at (no ZIP dependence)",
    ]
    references = _NODE_REFS
    parameter_units = {"p_pu": "pu", "q_pu": "pu"}

    def __init__(self, params: LoadParams):
        self.params = params

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        s_load = self.params.p_pu + 1j * self.params.q_pu
        v = complex_voltage(inlet)
        remaining = complex_power(inlet) - s_load
        return from_complex(remaining, v), {
            "drawn_p": jnp.asarray(self.params.p_pu),
            "drawn_q": jnp.asarray(self.params.q_pu),
            "current": jnp.abs(jnp.conj(s_load / v)),
        }


@dataclass
class ShuntParams(ParamsMixin):
    """Parameters of a fixed shunt.

    Attributes:
        g_pu: shunt conductance (pu). Always a loss.
        b_pu: shunt susceptance (pu). Positive is a CAPACITOR, which
            injects vars; negative is a reactor, which absorbs them.
    """

    g_pu: float = 0.0
    b_pu: float = 0.0


class ShuntDraw:
    """Subtract a shunt admittance's draw from a stream.

    Unlike a load, a shunt is constant IMPEDANCE: it draws
    ``|V|^2 conj(Y)``, so its var output falls with the square of the
    voltage. That is the well-known weakness of capacitor banks for
    voltage support --- they give least where they are needed most ---
    and it falls straight out of the model rather than having to be
    remembered.
    """

    symbol = "Shunt"
    equations = [
        r"S_\mathrm{sh} = |V|^2 \overline{(g + jb)}",
        r"S_\mathrm{out} = S_\mathrm{in} - S_\mathrm{sh}",
    ]
    assumptions = [
        "Constant impedance: the var output falls with the square of "
        "the voltage",
    ]
    references = _NODE_REFS
    parameter_units = {"g_pu": "pu", "b_pu": "pu"}

    def __init__(self, params: ShuntParams):
        self.params = params

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        v = complex_voltage(inlet)
        y = self.params.g_pu + 1j * self.params.b_pu
        s_shunt = jnp.abs(v) ** 2 * jnp.conj(y)
        return from_complex(complex_power(inlet) - s_shunt, v), {
            "drawn_p": jnp.real(s_shunt),
            "drawn_q": jnp.imag(s_shunt),
        }


@dataclass
class GeneratorParams(ParamsMixin):
    """Parameters of a generator injection.

    Attributes:
        p_pu: real output (pu).
        q_pu: reactive output (pu).
        cost: polynomial cost coefficients, highest order first, with
            real power in MW and cost in $/h.
        base_mva: the base ``p_pu`` is on, needed to evaluate the cost.
    """

    p_pu: float = 0.0
    q_pu: float = 0.0
    cost: tuple[float, ...] = (0.0,)
    base_mva: float = 100.0


class GeneratorInject:
    """Add a generator's output to a stream, and price it.

    ``info["cost"]`` is the unit's cost in $/h at its current output,
    which is what makes a flowsheet objective assembled from these
    differentiable with respect to dispatch --- the gradient of the
    objective is the offer curve.
    """

    symbol = "Gen"
    equations = [
        r"S_\mathrm{out} = S_\mathrm{in} + (P_g + jQ_g)",
        r"C(P_g) = c_2 P_g^2 + c_1 P_g + c_0",
    ]
    assumptions = [
        "Cost is a polynomial in real power only; reactive output is "
        "free at the margin",
    ]
    references = _NODE_REFS
    parameter_units = {
        "p_pu": "pu", "q_pu": "pu", "base_mva": "MVA", "cost": "$/h",
    }

    def __init__(self, params: GeneratorParams):
        self.params = params

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        v = complex_voltage(inlet)
        injected = self.params.p_pu + 1j * self.params.q_pu
        p_mw = jnp.asarray(self.params.p_pu) * self.params.base_mva
        return from_complex(complex_power(inlet) + injected, v), {
            "injected_p": jnp.asarray(self.params.p_pu),
            "injected_q": jnp.asarray(self.params.q_pu),
            "cost": polynomial_cost(p_mw, self.params.cost),
        }


class BusNode:
    """Sum power at a bus; every inlet shares the first one's voltage.

    That shared voltage IS the definition of a bus, and taking it from
    the first inlet is the sequential-modular way of saying so: the
    equation-oriented formulation would instead carry one voltage
    variable per bus and constrain the rest to equal it. The two agree
    at a converged solution; the sequential form just cannot detect a
    disagreement, so a flowsheet must be built so that the first inlet
    is the one whose voltage was actually computed.
    """

    symbol = "Bus"
    equations = [
        r"S = \sum_k S_k",
        r"V = V_1 \quad \text{(every inlet shares the bus voltage)}",
    ]
    assumptions = [
        "All inlets are electrically at the same point, so the first "
        "inlet's voltage is taken as the bus voltage",
    ]
    references = _NODE_REFS

    def __call__(self, *inlets: Stream) -> tuple[Stream, dict]:
        if not inlets:
            raise ValueError("BusNode needs at least one inlet")
        v = complex_voltage(inlets[0])
        total = sum(complex_power(s) for s in inlets[1:])
        total = complex_power(inlets[0]) + total
        return from_complex(total, v), {
            "n_inlets": len(inlets),
            "vm": jnp.abs(v),
            "net_p": jnp.real(total),
            "net_q": jnp.imag(total),
        }


class PowerSplit:
    """Divide a bus's outgoing power between two branches.

    ``fraction`` of the real AND reactive power goes to the first
    outlet. Both outlets keep the bus's voltage, since they leave the
    same bus.

    A split fraction is NOT a physical parameter --- how power divides
    between two paths is determined by their impedances, not chosen ---
    so this unit belongs only in a sequential decomposition, where the
    fraction is a tear variable a fixed-point iteration solves for.
    """

    symbol = "Split"
    equations = [
        r"S_1 = \alpha S,\quad S_2 = (1 - \alpha) S",
        r"V_1 = V_2 = V \quad \text{(both outlets leave the same bus)}",
    ]
    assumptions = [
        "The split fraction is a TEAR VARIABLE, not a physical "
        "parameter: how power actually divides is set by the downstream "
        "impedances",
    ]
    references = _FEEDER_REFS
    parameter_units = {"fraction": "-"}

    def __init__(self, params: "SplitParams"):
        self.params = params

    def __call__(self, inlet: Stream) -> tuple[tuple[Stream, Stream], dict]:
        v = complex_voltage(inlet)
        total = complex_power(inlet)
        first = self.params.fraction * total
        return (
            from_complex(first, v),
            from_complex(total - first, v),
        ), {"fraction": jnp.asarray(self.params.fraction)}


@dataclass
class SplitParams(ParamsMixin):
    """Parameters of a :class:`PowerSplit`.

    Attributes:
        fraction: share of the inlet power sent to the first outlet.
    """

    fraction: float = 0.5


@dataclass
class LadderCloseParams(ParamsMixin):
    """Parameters of a :class:`LadderClose`.

    Attributes:
        vm_setpoint: the source's regulated voltage (pu), re-emitted so
            the tear's voltage entries are a constant and converge in
            one pass.
        va_reference: the source's angle (rad).
    """

    vm_setpoint: float = 1.0
    va_reference: float = 0.0


class LadderClose:
    """Correct a feeder's infeed guess by the residual at its open end.

    A ladder feeder has exactly one unknown: the complex power the
    source has to push in. Everything downstream follows explicitly ---
    but the losses, and therefore the infeed, are not known until the
    flow is. So the infeed is a tear, and this unit closes the loop.

    The naive closure --- recycling the open end's leftover power back
    as the next infeed --- has the WRONG fixed point: it converges where
    ``leftover == infeed``, and the answer wanted is where
    ``leftover == 0``. Correcting instead::

        infeed_next = infeed - leftover

    has its fixed point exactly at ``leftover = 0``, and since the
    leftover is very nearly ``infeed - (load + loss)``, the correction
    is close to an exact Newton step: it converges in a handful of
    passes rather than not at all.
    """

    symbol = "Ladder close"
    equations = [
        r"S^\mathrm{in}_{k+1} = S^\mathrm{in}_k - S^\mathrm{leftover}_k",
        r"S^\mathrm{leftover} \to 0 \quad \text{at the fixed point}",
    ]
    assumptions = [
        "The feeder is a ladder: one unknown, the source infeed",
        "The leftover is very nearly (infeed - load - loss), which makes "
        "the correction close to an exact Newton step",
    ]
    references = _FEEDER_REFS
    numerical_method = "Fixed-point tear on the source infeed"
    parameter_units = {"vm_setpoint": "pu", "va_reference": "rad"}

    def __init__(self, params: LadderCloseParams):
        self.params = params

    def __call__(
        self, tail: Stream, infeed: Stream
    ) -> tuple[Stream, dict]:
        leftover = complex_power(tail)
        corrected = complex_power(infeed) - leftover
        return power_stream(
            jnp.real(corrected),
            jnp.imag(corrected),
            self.params.vm_setpoint,
            self.params.va_reference,
        ), {
            "leftover_p": jnp.real(leftover),
            "leftover_q": jnp.imag(leftover),
            "infeed_p": jnp.real(corrected),
            "infeed_q": jnp.imag(corrected),
        }
