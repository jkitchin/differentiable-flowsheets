"""Sequential-modular power flow: the backward/forward sweep.

Newton on the full system (:mod:`difflow_power.powerflow`) is the right
method for a meshed transmission network. It is a poor one for a
distribution feeder: a feeder has a high R/X ratio, which makes the
decoupling Newton relies on invalid, and it is radial, which makes a
far cheaper method available.

That method is the backward/forward sweep, and it is genuinely a
sequential-modular flowsheet solve: units evaluated in a topological
schedule with one tear, iterated to a fixed point.

The sweep
---------

The tear is the VOLTAGE PROFILE. Given a guess for it:

1. **Backward, leaves to root.** At each bus, KCL says the current into
   its parent branch is what is left after the bus's own injection,
   its shunt, and its children's branches have taken their share::

       I_b->parent = conj(S_b / V_b) - Y_sh,b V_b
                     - sum over children c of I_b->(b,c)

   Each child term is the current at B's OWN end of that branch,
   evaluated from the two voltages. That distinction matters: the two
   ends of a branch carry different currents, differing by what the
   branch's charging absorbs, and the textbook shortcut of reusing the
   child's own accumulated current is valid only after the charging has
   been folded into the bus shunts. One pass gives every branch
   current --- no iteration, no matrix.

2. **Forward, root to leaves.** The branch relation
   ``I_b-> = Y_tf V_a + Y_tt V_b`` inverts for the child voltage::

       V_b = (I_b->parent - Y_tf V_parent) / Y_tt

   so one forward pass from the fixed slack voltage gives every bus
   voltage.

Iterating the two is a contraction, so it converges LINEARLY: on the
12.47 kV example case the factor is about 0.4 per pass, or five passes
per decade of accuracy --- eight passes to 1e-4, eighteen to 1e-8,
twenty-eight to floating-point. That is many more iterations than
Newton's five, and still much less work, because a pass is O(n) with no
Jacobian formed, factorised or differentiated.

It is exact, not an approximation: line charging, bus shunts and
transformer taps are all carried through the 2x2 admittance block that
:mod:`difflow_power.units.branches` uses, and the converged profile
matches Newton to 1e-12 on every bus.

The tear is solved with ``optimistix``'s fixed-point iteration, so the
whole thing is jittable and its gradients come from the implicit
function theorem at the converged profile rather than from unrolling
the sweeps.

What it will not do
-------------------

Loops. The backward pass needs each bus to have exactly one parent, and
a meshed network does not. :class:`RadialFeederFlowsheet` refuses to
build on one rather than sweeping a spanning tree and silently
returning the wrong answer --- use
:func:`difflow_power.powerflow.solve_power_flow` there.

The schedule is a Python loop over a static topology, so it unrolls
into the traced graph. For a feeder of a few hundred buses that is
fine; past a few thousand the trace itself becomes the cost, and the
equation-oriented Newton is the better tool again.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow.flowsheet import Flowsheet, Unit
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream

from difflow_power.network import PowerNetwork
from difflow_power.physics import branch_admittances
from difflow_power.streams import SPECIES, from_complex, power_stream
from difflow_power.units import (
    BranchParams,
    LadderClose,
    LadderCloseParams,
    LoadDraw,
    LoadParams,
    SeriesBranch,
    SlackSource,
    SlackSourceParams,
)


def bus_stream_name(bus_id: str) -> str:
    """Flowsheet stream carrying a bus's voltage and net power."""
    return f"bus_{bus_id}"


def branch_stream_name(branch_id: str) -> str:
    """Flowsheet stream carrying a branch's flow."""
    return f"branch_{branch_id}"


@dataclass
class FeederTree(ParamsMixin):
    """The rooted tree a radial network's sweep schedule follows.

    Attributes:
        root: the slack bus, where the sweep starts and ends.
        order: bus ids in breadth-first order from the root. The
            backward pass walks it in reverse, the forward pass
            forwards.
        parent: bus id -> ``(parent bus id, branch id, reversed)``.
            ``reversed`` is True when the branch's stored direction runs
            child -> parent, in which case the admittance block's two
            ends swap.
        children: bus id -> list of child bus ids.
    """

    root: str
    order: list[str]
    parent: dict[str, tuple[str, str, bool]]
    children: dict[str, list[str]] = field(default_factory=dict)


def feeder_tree(network: PowerNetwork, root: str | None = None) -> FeederTree:
    """Root a radial network at its slack bus.

    Args:
        network: a network with no loops.
        root: the bus to root at; defaults to the slack bus.

    Returns:
        A :class:`FeederTree`.

    Raises:
        ValueError: if the network has loops. A sweep on a spanning tree
            of a meshed network converges to something, and that
            something is not the power flow.
    """
    if not network.is_radial:
        raise ValueError(
            f"network has {network.cycle_rank} loop(s); the backward/"
            "forward sweep needs a radial network. Use "
            "difflow_power.powerflow.solve_power_flow for a meshed one."
        )
    root = root or network.slack_bus
    incident: dict[str, list[tuple[str, str]]] = {b: [] for b in network.buses}
    for bid, br in network.branches.items():
        incident[br.from_bus].append((bid, br.to_bus))
        incident[br.to_bus].append((bid, br.from_bus))

    order = [root]
    parent: dict[str, tuple[str, str, bool]] = {}
    children: dict[str, list[str]] = {b: [] for b in network.buses}
    seen = {root}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for branch_id, other in incident[node]:
            if other in seen:
                continue
            seen.add(other)
            reversed_ = network.branches[branch_id].from_bus != node
            parent[other] = (node, branch_id, reversed_)
            children[node].append(other)
            order.append(other)
            queue.append(other)
    return FeederTree(root=root, order=order, parent=parent, children=children)


class RadialFeederFlowsheet:
    """Sequential-modular power flow for a radial network.

    Args:
        network: a radial network.
        root: the bus to sweep from; defaults to the slack bus.

    Example:
        >>> import difflow_power as dp
        >>> fs = dp.RadialFeederFlowsheet(dp.cases.radial_feeder())
        >>> streams = fs.solve()
        >>> round(float(streams["bus_s"]["P"]), 3)     # slack voltage
        1.02
    """

    def __init__(self, network: PowerNetwork, root: str | None = None):
        self.network = network
        self.tree = feeder_tree(network, root)
        self._blocks = {
            bid: branch_admittances(
                br.r, br.x, br.b, br.tap, br.shift, br.g
            )
            for bid, br in network.branches.items()
        }

    # -- the two passes ---------------------------------------------------

    def _ends(self, branch_id: str, reversed_: bool):
        """The (near, far) admittance pair for a branch traversed
        parent -> child, swapping the block if the branch is stored
        child -> parent."""
        yff, yft, ytf, ytt = self._blocks[branch_id]
        return (ytf, ytt) if not reversed_ else (yft, yff)

    def _current_into(
        self, branch_id: str, near_is_from: bool, v_near: Array, v_far: Array
    ) -> Array:
        """Current flowing from the near bus INTO a branch.

        The 2x2 block is not symmetric once a branch carries a tap or
        charging, so which end is which matters: ``I_f`` and ``I_t`` are
        different currents, and their SUM is what the branch's own shunt
        halves absorb. Confusing the two is the classic way to build a
        sweep that converges neatly to the wrong voltage profile.
        """
        yff, yft, ytf, ytt = self._blocks[branch_id]
        if near_is_from:
            return yff * v_near + yft * v_far
        return ytf * v_far + ytt * v_near

    def backward(
        self, v: dict[str, Array], injection: dict[str, Array]
    ) -> dict[str, Array]:
        """Currents into each bus's parent branch, from KCL at every bus.

        At bus ``b``, everything injected has to leave through the
        branches and the shunt, so the current into the parent branch is
        what the bus's own children and shunt did not take::

            I_b->parent = conj(S_b / V_b) - Y_sh,b V_b
                          - sum over children c of I_b->(b,c)

        Each child term is evaluated at B's OWN end of that branch, not
        at the child's --- the two differ by the branch's charging
        current, and using the child's accumulated value instead
        (which is the textbook shortcut, valid only once the charging
        has been folded into the bus shunts) is a real error on a
        cabled feeder.

        Args:
            v: bus id -> complex voltage (pu).
            injection: bus id -> complex power injected at the bus (pu),
                generation minus load.

        Returns:
            Bus id -> current into that bus's parent branch, at the bus's
            own end. The root has no entry.
        """
        shunt = self.network.shunt_array_pu()
        index = self.network.bus_index
        currents: dict[str, Array] = {}
        for bus in reversed(self.tree.order):
            if bus == self.tree.root:
                continue
            total = jnp.conj(injection[bus] / v[bus]) - shunt[index[bus]] * v[bus]
            for child in self.tree.children[bus]:
                _, branch_id, child_reversed = self.tree.parent[child]
                total = total - self._current_into(
                    branch_id, not child_reversed, v[bus], v[child]
                )
            currents[bus] = total
        return currents

    def forward(
        self, v_root: Array, currents: dict[str, Array]
    ) -> dict[str, Array]:
        """Bus voltages from the root outwards, given the branch currents."""
        v = {self.tree.root: v_root}
        for bus in self.tree.order:
            if bus == self.tree.root:
                continue
            parent, branch_id, reversed_ = self.tree.parent[bus]
            y_near, y_far = self._ends(branch_id, reversed_)
            v[bus] = (currents[bus] - y_near * v[parent]) / y_far
        return v

    # -- the fixed point --------------------------------------------------

    def _injection(
        self, demand: tuple[Array, Array] | None = None
    ) -> dict[str, Array]:
        """Complex power injected at each bus (pu), generation minus load."""
        pd, qd = demand if demand is not None else self.network.load_arrays_pu()
        index = self.network.bus_index
        out = {b: -(pd[index[b]] + 1j * qd[index[b]]) for b in self.network.buses}
        for gen in self.network.generators.values():
            if gen.bus == self.tree.root:
                continue      # the slack supplies whatever balances
            out[gen.bus] = out[gen.bus] + (
                gen.p_mw + 1j * gen.q_mvar
            ) / self.network.base_mva
        return out

    def sweep(
        self, v: dict[str, Array], injection: dict[str, Array]
    ) -> dict[str, Array]:
        """One backward pass followed by one forward pass."""
        return self.forward(v[self.tree.root], self.backward(v, injection))

    def solve_voltages(
        self,
        *,
        demand: tuple[Array, Array] | None = None,
        v0: dict[str, Array] | None = None,
        rtol: float = 1e-12,
        atol: float = 1e-12,
        max_steps: int = 200,
    ) -> tuple[dict[str, Array], dict[str, Any]]:
        """Iterate the sweep to a fixed point in the voltage profile.

        Jit- and grad-safe: the fixed point is found by ``optimistix``,
        whose gradients come from the implicit function theorem at the
        converged profile, so differentiating with respect to ``demand``
        costs one linear solve rather than an unrolled sweep history.

        Returns:
            ``(voltages, stats)``; ``stats["converged"]`` and
            ``stats["num_steps"]``.
        """
        injection = self._injection(demand)
        root = self.tree.root
        v_root = (
            self.network.buses[root].vm_setpoint
            * jnp.exp(1j * self.network.buses[root].va_reference)
        )
        if v0 is None:
            v0 = {b: v_root for b in self.tree.order}
        # The ROOT is deliberately not part of the packed vector. Its
        # voltage is pinned, so the sweep returns it unchanged, and
        # including it would give the fixed-point map an identity block
        # -- making ``I - dG/dv`` singular and the implicit-function
        # gradient undefined, even though the forward solve is fine.
        order = list(self.tree.order[1:])
        n = len(order)

        def pack(v):
            return jnp.stack(
                [jnp.real(v[b]) for b in order]
                + [jnp.imag(v[b]) for b in order]
            )

        def unpack(x):
            v = {b: x[i] + 1j * x[n + i] for i, b in enumerate(order)}
            v[self.tree.root] = v_root
            return v

        def step(x, args):
            return pack(self.sweep(unpack(x), args))

        solver = optx.FixedPointIteration(rtol=rtol, atol=atol)
        sol = optx.fixed_point(
            step, solver, pack(v0), args=injection,
            max_steps=max_steps, throw=False,
        )
        stats = dict(sol.stats)
        stats["converged"] = sol.result == optx.RESULTS.successful
        return unpack(sol.value), stats

    def solve(
        self,
        *,
        demand: tuple[Array, Array] | None = None,
        **kwargs: Any,
    ) -> dict[str, Stream]:
        """Solve and return difflow streams for every bus and branch.

        Stream names come from :func:`bus_stream_name` and
        :func:`branch_stream_name`. A bus stream carries the bus's
        voltage in its ``P``/``T`` slots and its NET injection in
        ``F_P``/``F_Q``; a branch stream carries the from-end voltage
        and the power entering the branch there.
        """
        v, stats = self.solve_voltages(demand=demand, **kwargs)
        injection = self._injection(demand)
        currents = self.backward(v, injection)

        streams: dict[str, Stream] = {}
        for bus in self.tree.order:
            if bus == self.tree.root:
                supplied = jnp.asarray(0.0 + 0j)
                for child in self.tree.children[bus]:
                    _, branch_id, child_reversed = self.tree.parent[child]
                    supplied = supplied + v[bus] * jnp.conj(
                        self._current_into(
                            branch_id, not child_reversed, v[bus], v[child]
                        )
                    )
                streams[bus_stream_name(bus)] = from_complex(supplied, v[bus])
            else:
                streams[bus_stream_name(bus)] = from_complex(
                    injection[bus], v[bus]
                )
        for bid, br in self.network.branches.items():
            yff, yft, ytf, ytt = self._blocks[bid]
            vf, vt = v[br.from_bus], v[br.to_bus]
            s_from = vf * jnp.conj(yff * vf + yft * vt)
            streams[branch_stream_name(bid)] = from_complex(s_from, vf)
        self.last_solve_stats = stats
        return streams

    def make_objective_fn(
        self, objective_fn: Callable[[dict[str, Stream]], Array]
    ) -> Callable[[tuple[Array, Array]], Array]:
        """A differentiable objective of the demand.

        The returned callable takes ``(pd, qd)`` per-bus arrays (pu),
        solves the sweep and evaluates ``objective_fn`` on the streams,
        so ``jax.grad`` of it answers "how does this objective move with
        load at each bus?" --- the feeder analogue of an LMP.
        """

        def objective(demand: tuple[Array, Array]) -> Array:
            return objective_fn(self.solve(demand=demand))

        return objective


# =============================================================================
# A difflow Flowsheet for a ladder feeder
# =============================================================================


def build_ladder_flowsheet(
    network: PowerNetwork, root: str | None = None
) -> tuple[Flowsheet, list[str]]:
    """Build a genuine :class:`difflow.Flowsheet` for a LADDER feeder.

    A ladder is a radial network with no branching: every bus has at
    most one child, so the whole network is one chain. That is the case
    where the sequential schedule is a straight line and the only tear
    is the substation infeed --- the power the source must push in,
    which is not known until the losses are, which are not known until
    the flow is.

    The assembled flowsheet is::

        infeed (tear) -> SlackSource -> [SeriesBranch -> LoadDraw] x N
                      -> LadderClose(end, infeed) -> infeed_next
        recycle: infeed_next -> infeed

    At convergence the leftover power at the open end of the ladder is
    zero: everything pushed in has been consumed by loads and losses.
    :class:`~difflow_power.units.nodes.LadderClose` is what makes that
    the fixed point rather than the leftover simply equalling the
    infeed.

    Args:
        network: a radial, non-branching network.
        root: the bus to start from; defaults to the slack bus.

    Returns:
        ``(flowsheet, order)`` --- the flowsheet, and the bus order the
        ladder follows.

    Raises:
        ValueError: if the network branches. Use
            :class:`RadialFeederFlowsheet` for a branching feeder;
            representing a split sequentially needs a tear per lateral,
            which the sweep avoids entirely.

    Example:
        >>> import difflow_power as dp
        >>> net = dp.cases.radial_feeder()      # this one branches
        >>> dp.build_ladder_flowsheet(net)      # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ValueError: ...
    """
    tree = feeder_tree(network, root)
    branching = [b for b, kids in tree.children.items() if len(kids) > 1]
    if branching:
        raise ValueError(
            f"buses {branching} have more than one downstream branch, so "
            "this feeder is not a ladder. Use RadialFeederFlowsheet, "
            "whose sweep handles branching with no extra tears."
        )

    fs = Flowsheet(
        list(SPECIES), default_flow=0.0, default_T=0.0, default_P=1.0
    )
    base = network.base_mva
    pd, qd = network.load_arrays_pu()
    index = network.bus_index

    fs.add_feed("infeed", power_stream(0.0, 0.0, 1.0, 0.0))
    source = network.buses[tree.root]
    fs.add_unit(
        Unit(
            name=f"src_{tree.root}",
            operation=SlackSource(
                SlackSourceParams(source.vm_setpoint, source.va_reference)
            ),
            inlet_names=["infeed"],
            outlet_names=[bus_stream_name(tree.root)],
        )
    )

    previous = bus_stream_name(tree.root)
    for bus in tree.order[1:]:
        _, branch_id, reversed_ = tree.parent[bus]
        br = network.branches[branch_id]
        params = BranchParams(br.r, br.x, br.b, br.g, br.tap, br.shift)
        if reversed_:
            # The chain runs against the branch's stored direction, so
            # the tap sits at the far end; inverting the ratio moves it.
            params = BranchParams(
                br.r, br.x, br.b, br.g, 1.0 / br.tap, -br.shift
            )
        arrival = f"{branch_stream_name(branch_id)}_out"
        fs.add_unit(
            Unit(
                name=f"line_{branch_id}",
                operation=SeriesBranch(params),
                inlet_names=[previous],
                outlet_names=[arrival],
            )
        )
        fs.add_unit(
            Unit(
                name=f"load_{bus}",
                operation=LoadDraw(
                    LoadParams(float(pd[index[bus]]), float(qd[index[bus]]))
                ),
                inlet_names=[arrival],
                outlet_names=[bus_stream_name(bus)],
            )
        )
        previous = bus_stream_name(bus)

    fs.add_unit(
        Unit(
            name="close",
            operation=LadderClose(
                LadderCloseParams(source.vm_setpoint, source.va_reference)
            ),
            inlet_names=[previous, "infeed"],
            outlet_names=["infeed_next"],
        )
    )
    fs.add_recycle("infeed_next", "infeed")
    return fs, list(tree.order)
