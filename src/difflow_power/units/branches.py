"""Branch unit operations: the pi-model as explicit, invertible maps.

A branch relates the complex voltage and power at its two ends. The
relation is a 2x2 admittance block (see
:func:`difflow_power.physics.branch_admittances`), and what makes a
sequential-modular power flow possible at all is that the block can be
solved EXPLICITLY in either direction:

- :class:`SeriesBranch` --- given the from-end voltage and the power
  entering there, get the to-end voltage and the power leaving. This is
  the forward pass of a feeder sweep.
- :class:`BranchDrop` --- given the from-end voltage and the CURRENT,
  get the to-end voltage. This is the voltage propagation of a
  backward/forward sweep, and is linear.
- :class:`BranchFlow` --- given both end voltages, get both end powers.
  This is what an equation-oriented formulation uses and what closes a
  loop.

There is no iteration in any of them: a branch is not where a power flow
is hard. The difficulty is entirely in the constant-power loads, which
is why :class:`~difflow_power.units.nodes.LoadDraw` is the unit a sweep
iterates around.

All three take the SAME :class:`BranchParams`, so a flowsheet can swap
which one it uses without touching the data --- which is the point of
having them as separate units rather than as modes of one.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream

from difflow_power.physics import branch_admittances
from difflow_power.streams import (
    complex_power,
    complex_voltage,
    from_complex,
    power_stream,
)


#: shared literature references for the branch units. The MATPOWER
#: paper is the authoritative statement of the tap/shift convention this
#: model follows; Grainger and Stevenson derive the underlying pi-model.
_BRANCH_REFS = [
    "Zimmerman, R.D., Murillo-Sanchez, C.E., Thomas, R.J., "
    "IEEE Trans. Power Syst. 26(1), 12-19 (2011).",
    "Grainger, J.J., Stevenson, W.D., Power System Analysis, "
    "McGraw-Hill (1994), ch. 6.",
]

#: the 2x2 admittance block, shared by every branch unit
_BRANCH_BLOCK = [
    r"t = \tau e^{j\theta},\quad y_s = 1/(r + jx)",
    r"Y_{ff} = (y_s + jb/2)/\tau^2,\quad Y_{ft} = -y_s/\overline{t}",
    r"Y_{tf} = -y_s/t,\quad Y_{tt} = y_s + jb/2",
]


@dataclass
class BranchParams(ParamsMixin):
    """Parameters of a line, transformer or phase shifter.

    Attributes:
        r: series resistance (pu).
        x: series reactance (pu).
        b: total line charging susceptance (pu), half at each end.
        g: total line charging conductance (pu).
        tap: off-nominal turns ratio magnitude at the from end.
        shift: phase shift (radians).
    """

    r: float = 0.0
    x: float = 0.1
    b: float = 0.0
    g: float = 0.0
    tap: float = 1.0
    shift: float = 0.0

    def admittances(self):
        """``(yff, yft, ytf, ytt)`` for this branch."""
        return branch_admittances(
            self.r, self.x, self.b, self.tap, self.shift, self.g
        )


class SeriesBranch:
    """Propagate voltage and power along a branch, from end to to end.

    Given the sending-end voltage and the complex power entering the
    branch there, both the receiving-end voltage and the power leaving
    follow in closed form:

    .. math::

        I_f = \\overline{(S_f / V_f)}, \\quad
        V_t = (I_f - Y_{ff} V_f) / Y_{ft}, \\quad
        S_t^{out} = -V_t \\overline{(Y_{tf} V_f + Y_{tt} V_t)}

    The sign on ``S_t^{out}`` makes the returned stream carry power
    *onward* rather than *into* the branch, so a chain of these composes
    the way a flowsheet expects. The difference between what went in and
    what comes out is the branch's own loss, which
    ``info["loss_p"]`` reports.

    Example:
        >>> unit = SeriesBranch(BranchParams(r=0.01, x=0.1))
        >>> inlet = power_stream(1.0, 0.2, 1.0, 0.0)
        >>> outlet, info = unit(inlet)
        >>> bool(info["loss_p"] > 0)      # a passive branch always loses
        True
    """

    symbol = "Branch \u2192"
    equations = _BRANCH_BLOCK + [
        r"I_f = \overline{(S_f / V_f)}",
        r"V_t = (I_f - Y_{ff} V_f) / Y_{ft}",
        r"S_t^\mathrm{out} = -V_t \overline{(Y_{tf} V_f + Y_{tt} V_t)}",
    ]
    assumptions = [
        "Balanced positive-sequence steady state",
        "The sending-end complex power and voltage are both known",
    ]
    references = _BRANCH_REFS
    parameter_units = {
        "r": "pu", "x": "pu", "b": "pu", "g": "pu",
        "tap": "-", "shift": "rad",
    }

    def __init__(self, params: BranchParams):
        self.params = params

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        yff, yft, ytf, ytt = self.params.admittances()
        v_from = complex_voltage(inlet)
        s_from = complex_power(inlet)

        i_from = jnp.conj(s_from / v_from)
        v_to = (i_from - yff * v_from) / yft
        s_to_in = v_to * jnp.conj(ytf * v_from + ytt * v_to)

        outlet = from_complex(-s_to_in, v_to)
        return outlet, {
            "loss_p": jnp.real(s_from + s_to_in),
            "loss_q": jnp.imag(s_from + s_to_in),
            "current": jnp.abs(i_from),
            "apparent_power": jnp.abs(s_from),
        }


class BranchDrop:
    """Propagate voltage across a branch at a known current.

    Linear in the voltage, and therefore the cheap half of a
    backward/forward sweep: given the sending voltage and the current
    the backward pass computed, the receiving voltage is one division.
    The stream's power is carried through unchanged --- this unit
    updates voltage only, and the caller is responsible for the flows.

    Use :class:`SeriesBranch` instead when the power, not the current,
    is what is known.
    """

    symbol = "Branch (drop)"
    equations = _BRANCH_BLOCK + [
        r"V_t = (I_f - Y_{ff} V_f) / Y_{ft}",
    ]
    assumptions = [
        "Balanced positive-sequence steady state",
        "The branch current is known; the stream's power is carried "
        "through unchanged",
    ]
    references = _BRANCH_REFS
    parameter_units = {
        "r": "pu", "x": "pu", "b": "pu", "g": "pu",
        "tap": "-", "shift": "rad",
    }

    def __init__(self, params: BranchParams):
        self.params = params

    def __call__(self, inlet: Stream, current: Array) -> tuple[Stream, dict]:
        yff, yft, _, _ = self.params.admittances()
        v_from = complex_voltage(inlet)
        v_to = (current - yff * v_from) / yft
        outlet = power_stream(
            inlet["F_P"], inlet["F_Q"], jnp.abs(v_to), jnp.angle(v_to)
        )
        return outlet, {"voltage_drop": jnp.abs(v_from) - jnp.abs(v_to)}


class BranchFlow:
    """Both end powers of a branch from both end voltages.

    The equation-oriented form: no inversion, no assumption about which
    end is upstream, and the only one of the three that works on a
    branch closing a loop, where neither end's power is known in advance.

    Both returned streams carry power INTO the branch from their own
    end, so their sum is the loss --- the convention
    :func:`difflow_power.residuals.branch_flows` uses.
    """

    symbol = "Branch (EO)"
    equations = _BRANCH_BLOCK + [
        r"S_f = V_f \overline{(Y_{ff} V_f + Y_{ft} V_t)}",
        r"S_t = V_t \overline{(Y_{tf} V_f + Y_{tt} V_t)}",
        r"S_\mathrm{loss} = S_f + S_t",
    ]
    assumptions = [
        "Balanced positive-sequence steady state",
        "Both end voltages are known; neither end is privileged, so this "
        "form also works on a branch closing a loop",
    ]
    references = _BRANCH_REFS
    parameter_units = {
        "r": "pu", "x": "pu", "b": "pu", "g": "pu",
        "tap": "-", "shift": "rad",
    }

    def __init__(self, params: BranchParams):
        self.params = params

    def __call__(
        self, from_stream: Stream, to_stream: Stream
    ) -> tuple[tuple[Stream, Stream], dict]:
        yff, yft, ytf, ytt = self.params.admittances()
        v_from = complex_voltage(from_stream)
        v_to = complex_voltage(to_stream)
        s_from = v_from * jnp.conj(yff * v_from + yft * v_to)
        s_to = v_to * jnp.conj(ytf * v_from + ytt * v_to)
        return (
            from_complex(s_from, v_from),
            from_complex(s_to, v_to),
        ), {
            "loss_p": jnp.real(s_from + s_to),
            "loss_q": jnp.imag(s_from + s_to),
            "apparent_power": jnp.maximum(jnp.abs(s_from), jnp.abs(s_to)),
        }


class Transformer(SeriesBranch):
    """A tap-changing or phase-shifting transformer.

    Identical mathematics to :class:`SeriesBranch` --- the branch model
    already carries the complex tap --- and a separate name only because
    a flowsheet reads better when a transformer is called one. It
    refuses parameters that make it a line, so a mislabelled component
    is caught at construction rather than by a puzzling result.
    """

    symbol = "Transformer"
    equations = _BRANCH_BLOCK + [
        r"V'_f = V_f / t \quad \text{(ideal ratio at the from end)}",
        r"S_t^\mathrm{out} = -V_t \overline{(Y_{tf} V_f + Y_{tt} V_t)}",
    ]
    assumptions = [
        "Ideal (lossless) turns ratio at the FROM end, followed by the "
        "series impedance",
        "No magnetising branch or core loss",
    ]
    references = _BRANCH_REFS
    parameter_units = {
        "r": "pu", "x": "pu", "tap": "-", "shift": "rad",
    }

    def __init__(self, params: BranchParams):
        if params.tap == 1.0 and params.shift == 0.0:
            raise ValueError(
                "a Transformer needs an off-nominal tap or a phase "
                "shift; with tap=1 and shift=0 this is a line, so use "
                "SeriesBranch"
            )
        if params.b != 0.0:
            raise ValueError(
                f"a Transformer should not carry line charging "
                f"(b={params.b}); that is a line's shunt capacitance"
            )
        super().__init__(params)
