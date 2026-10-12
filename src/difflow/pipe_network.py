"""Looped and branched liquid pipe networks.

:class:`PipeNetwork` holds nodes and branches. A node either has a fixed
pressure (a reservoir, a pressurized header, a discharge to atmosphere) or a
specified external demand (volumetric flow leaving the network there;
negative for an injection). A branch joins two nodes and is a
:class:`~difflow.units.pipe.Pipe` (Darcy-Weisbach with fittings and elevation,
from :mod:`difflow.fluids`) or a :class:`ResistanceBranch`
(``dP = r Q|Q|^(n-1)``, e.g. a Hazen-Williams or textbook ``k Q^2`` element).

:meth:`PipeNetwork.solve` returns the branch flows, the node pressures and
convergence information. Two methods:

* ``"newton"`` (default) -- simultaneous Newton on branch flows and free-node
  pressures (the "global gradient" / hybrid node-loop formulation of Todini
  and Pilati, 1988, used by EPANET): one momentum equation per branch,
  ``dP_k(Q_k) = P_from - P_to``, and one continuity equation per free node.
  It is started from the solution of the network with every branch
  linearized at 1 m/s and solved with :func:`optimistix.root_find`, so
  ``jax.grad`` of any output with respect to demands, fixed pressures or pipe
  parameters (``D``, roughness, ``L``, ``mu``, ...) is the
  implicit-function-theorem derivative at the solution.
* ``"hardy_cross"`` -- the classical Hardy Cross (1936) loop-flow correction,
  sweeping the loops of a cycle basis (Gauss-Seidel order) for a fixed
  number of iterations; fixed-pressure nodes are joined through a virtual
  datum node, which adds the "pseudo-loops" between reservoirs. Differentiable
  by unrolling. Offered for teaching and cross-checking; it converges
  linearly and can need many sweeps.

Sign conventions: a branch flow is positive from its ``from`` node to its
``to`` node; ``dP`` of a branch is ``P_from - P_to`` (friction and fittings
in the flow direction, plus the static head ``rho g dz`` of a Pipe). The
network is single-phase, incompressible and isothermal: all pipes see one
liquid (numbers in each ``PipeParams``, or a ``fluid`` stream for
``rho``/``mu`` that are None or callables).

This is a library, deliberately not a palette operation: a network is a
whole hydraulic system, not a unit with inlet and outlet streams, so it lives
outside :mod:`difflow.units` (whose exported classes are auto-registered).

References:
    Cross, H. (1936). Analysis of flow in networks of conduits or conductors.
    Univ. Illinois Eng. Exp. Station Bull. 286.
    Todini, E. and Pilati, S. (1988). A gradient algorithm for the analysis
    of pipe networks. In *Computer Applications in Water Supply*, Vol. 1,
    Research Studies Press, 1-20.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow.streams import Stream
from difflow.units.pipe import Pipe

#: Flow (m^3/s) that smooths ``|Q|`` at zero flow: ``|Q| ~ sqrt(Q^2 + Q_EPS^2)``.
Q_EPS = 1e-12

NETWORK_METHODS = ("newton", "hardy_cross")


@dataclass
class ResistanceBranch:
    """A lumped branch with ``dP = r Q|Q|^(n-1) + dP_static`` (Pa).

    Attributes:
        r: Resistance coefficient (Pa per (m^3/s)^n).
        n: Flow exponent: 2 for a turbulent ``k Q^2`` element, 1.852 for
            Hazen-Williams, 1 for laminar flow.
        dP_static: Flow-independent pressure difference ``P_from - P_to``
            added to the loss (Pa), e.g. ``rho g dz`` of an elevation change.

    Example:
        >>> b = ResistanceBranch(r=2.0e9, n=2.0)
        >>> float(b.signed_dp(-0.01))
        -200000.0
    """

    r: float
    n: float = 2.0
    dP_static: float = 0.0

    def signed_dp(self, Q: Array | float, fluid: Stream | None = None) -> Array:
        """``P_from - P_to`` (Pa) at flow ``Q`` (m^3/s, signed)."""
        Q = jnp.asarray(Q, dtype=jnp.float64)
        Qa = jnp.sqrt(Q**2 + Q_EPS**2)
        return (jnp.asarray(self.r) * Q * Qa ** (jnp.asarray(self.n) - 1.0)
                + jnp.asarray(self.dP_static, dtype=jnp.float64))

    def reference_flow(self, fluid: Stream | None = None) -> Array | None:
        """No natural flow scale (None)."""
        return None


def _pipe_signed_dp(pipe: Pipe, Q: Array, fluid: Stream | None) -> Array:
    """``P_from - P_to`` of a :class:`Pipe` for a signed flow.

    Friction and fittings act against the flow (odd in ``Q``); the static head
    ``rho g dz`` does not change sign. ``|Q|`` is smoothed as
    ``sqrt(Q^2 + Q_EPS^2)``, so the loss is ``R_laminar Q`` near zero flow and
    the function is differentiable through flow reversal.
    """
    Qa = jnp.sqrt(Q**2 + Q_EPS**2)
    h = pipe.hydraulics_at_flow(Qa, fluid)
    return Q / Qa * (h["dP_friction"] + h["dP_minor"]) + h["dP_static"]


def _branch_dp(element, Q: Array, fluid: Stream | None) -> Array:
    if isinstance(element, Pipe):
        return _pipe_signed_dp(element, Q, fluid)
    return element.signed_dp(Q, fluid)


class PipeNetworkResult(NamedTuple):
    """Solution of a :class:`PipeNetwork`.

    Attributes:
        Q: Branch flows (m^3/s), in branch order, positive from ``from`` to ``to``.
        P: Node pressures (Pa), in node order.
        branch_flows: ``{branch name: Q}``.
        node_pressures: ``{node name: P}``.
        branch_dP: ``{branch name: P_from - P_to}`` (Pa) from the branch model
            at the solution flow.
        supply: ``{fixed-pressure node: net flow it delivers into the network}``
            (m^3/s; negative when the network discharges into it).
        converged: Bool array: the solver reported success and the residuals
            are below tolerance.
        mass_imbalance: Largest absolute continuity error at a free node (m^3/s).
        energy_residual: Largest absolute momentum (loop) error over the
            branches (Pa): ``|dP_k(Q_k) - (P_from - P_to)|``.
        iterations: Newton steps taken, or Hardy Cross sweeps.
    """

    Q: Array
    P: Array
    branch_flows: dict
    node_pressures: dict
    branch_dP: dict
    supply: dict
    converged: Array
    mass_imbalance: Array
    energy_residual: Array
    iterations: Array


class PipeNetwork:
    """A liquid pipe network with fixed-pressure and demand nodes.

    Build it with :meth:`add_node` and :meth:`add_branch`, then call
    :meth:`solve`. See the module docstring for the formulation.

    Example:
        Two parallel pipes from a 3 bar header to a 2 L/s draw-off:

        >>> from difflow import Pipe, PipeParams, PipeNetwork
        >>> water = dict(rho=998.2, mu=1.002e-3)
        >>> net = PipeNetwork()
        >>> net.add_node("header", pressure=3e5)
        >>> net.add_node("user", demand=2e-3)
        >>> net.add_branch("a", "header", "user", Pipe(PipeParams(L=50.0, D=0.0409, **water)))
        >>> net.add_branch("b", "header", "user", Pipe(PipeParams(L=80.0, D=0.0409, **water)))
        >>> res = net.solve()
        >>> bool(res.converged), round(float(res.Q.sum()) * 1e3, 9)
        (True, 2.0)
    """

    def __init__(self) -> None:
        self.nodes: list[str] = []
        self.fixed_pressure: dict[str, object] = {}
        self.demand: dict[str, object] = {}
        self.branches: list[tuple[str, str, str, object]] = []

    # ------------------------------------------------------------------
    def add_node(self, name: str, pressure: Array | float | None = None,
                 demand: Array | float = 0.0) -> None:
        """Add a node.

        Args:
            name: Node name (unique).
            pressure: Fixed pressure (Pa) for a reservoir or header node; None
                for a free node whose pressure is solved for.
            demand: External volumetric flow leaving the network at a free
                node (m^3/s); negative for an injection. Must be 0 at a
                fixed-pressure node (its flow is an output, ``supply``).

        Raises:
            ValueError: Duplicate name, or a demand on a fixed-pressure node.
        """
        if name in self.nodes:
            raise ValueError(f"node {name!r} already exists")
        if pressure is not None:
            if not (isinstance(demand, (int, float)) and demand == 0.0):
                raise ValueError(f"node {name!r} has a fixed pressure; its flow is solved for, "
                                 "so it cannot also have a demand")
            self.fixed_pressure[name] = pressure
        else:
            self.demand[name] = demand
        self.nodes.append(name)

    def add_branch(self, name: str, from_node: str, to_node: str, element) -> None:
        """Add a branch from ``from_node`` to ``to_node``.

        Args:
            name: Branch name (unique).
            from_node: Upstream node for positive flow.
            to_node: Downstream node for positive flow.
            element: A :class:`~difflow.units.pipe.Pipe` (its ``dz`` or
                elevation ``profile`` is the elevation gain from ``from_node``
                to ``to_node``) or a :class:`ResistanceBranch`.

        Raises:
            ValueError: Unknown node, a self-loop, or a duplicate name.
        """
        if any(b[0] == name for b in self.branches):
            raise ValueError(f"branch {name!r} already exists")
        for n in (from_node, to_node):
            if n not in self.nodes:
                raise ValueError(f"branch {name!r}: unknown node {n!r}; add it first")
        if from_node == to_node:
            raise ValueError(f"branch {name!r} starts and ends at {from_node!r}")
        self.branches.append((name, from_node, to_node, element))

    # ------------------------------------------------------------------
    def _check(self) -> None:
        if not self.fixed_pressure:
            raise ValueError("a pipe network needs at least one fixed-pressure node "
                             "(otherwise the pressure level is undetermined)")
        if not self.branches:
            raise ValueError("the network has no branches")
        # connectivity
        adj = {n: set() for n in self.nodes}
        for _, a, b, _ in self.branches:
            adj[a].add(b)
            adj[b].add(a)
        seen, todo = set(), deque([next(iter(self.fixed_pressure))])
        while todo:
            n = todo.popleft()
            if n in seen:
                continue
            seen.add(n)
            todo.extend(adj[n] - seen)
        missing = [n for n in self.nodes if n not in seen]
        if missing:
            raise ValueError(f"nodes {missing} are not connected to the network")
        for n in self.nodes:
            if n not in self.fixed_pressure and not adj[n] - {n}:
                raise ValueError(f"node {n!r} has no branches")

    def _incidence(self):
        """``M[node, branch]``: +1 at the ``from`` node, -1 at the ``to`` node."""
        idx = {n: i for i, n in enumerate(self.nodes)}
        M = [[0.0] * len(self.branches) for _ in self.nodes]
        for k, (_, a, b, _) in enumerate(self.branches):
            M[idx[a]][k] = 1.0
            M[idx[b]][k] = -1.0
        return jnp.asarray(M, dtype=jnp.float64)

    def loops(self) -> list[dict[str, float]]:
        """A cycle basis of the branch graph (fixed-pressure nodes not joined).

        Each loop is ``{branch name: +1 or -1}``, the sign being the branch
        orientation along the loop's direction of travel, so at a solution
        ``sum_k sign_k (P_from - P_to)_k = 0`` around every loop (the "zero
        head loss around a loop" condition). Pseudo-loops between
        fixed-pressure nodes are not included.
        """
        names = [b[0] for b in self.branches]
        basis = self._cycle_basis(self.nodes, [(a, b) for _, a, b, _ in self.branches],
                                  root=self.nodes[0])
        return [{names[k]: s for k, s in loop.items()} for loop in basis]

    @staticmethod
    def _cycle_basis(nodes, edges, root) -> list:
        """Fundamental cycles of a BFS spanning tree (multigraph-safe)."""
        adj = {n: [] for n in nodes}
        for k, (a, b) in enumerate(edges):
            adj[a].append((k, b, 1.0))
            adj[b].append((k, a, -1.0))
        parent = {root: None}  # node -> (edge k, parent node, sign of edge going parent->node)
        order = [root]
        tree = set()
        q = deque([root])
        while q:
            u = q.popleft()
            for k, v, sgn in adj[u]:
                if v not in parent:
                    parent[v] = (k, u, sgn)
                    tree.add(k)
                    order.append(v)
                    q.append(v)

        def path_to_root(n):
            out = []  # list of (edge, sign along node->root direction)
            while parent[n] is not None:
                k, u, sgn = parent[n]
                out.append((k, -sgn))  # walking node -> parent reverses parent->node
                n = u
            return out

        loops = []
        for k, (a, b) in enumerate(edges):
            if k in tree:
                continue
            # loop: a -> b along chord, then b -> root, then root -> a
            loop = {k: 1.0}
            pb = path_to_root(b)
            pa = path_to_root(a)
            for e, s in pb:
                loop[e] = loop.get(e, 0.0) + s
            for e, s in pa:
                loop[e] = loop.get(e, 0.0) - s
            loops.append({e: s for e, s in loop.items() if s != 0.0})
        return loops

    # ------------------------------------------------------------------
    def _dp_vector(self, Q: Array, fluid: Stream | None) -> Array:
        return jnp.stack([_branch_dp(el, Q[k], fluid) for k, (_, _, _, el) in enumerate(self.branches)])

    def _static_and_conductance(self, fluid, Q_scale):
        """Static ``dP`` (flow-independent part) and a linearized conductance per branch."""
        st, cond = [], []
        for _, _, _, el in self.branches:
            s0 = _branch_dp(el, jnp.asarray(0.0), fluid)
            if isinstance(el, Pipe):
                A = el._area(jnp.asarray(el.params.D, dtype=jnp.float64))
                Q_ref = A * 1.0  # 1 m/s
            else:
                Q_ref = Q_scale
            loss = _branch_dp(el, Q_ref, fluid) - s0
            st.append(s0)
            cond.append(Q_ref / loss)
        return jnp.stack(st), jnp.stack(cond)

    def _linear_start(self, fluid, M, free, fixed, d_free, P_fixed, Q_scale):
        """Flows and free pressures of the network linearized at 1 m/s."""
        st, c = self._static_and_conductance(fluid, Q_scale)
        sg = jax.lax.stop_gradient
        st, c = sg(st), sg(c)
        Mf, Mx = M[jnp.asarray(free, dtype=int)], M[jnp.asarray(fixed, dtype=int)]
        # branch: Q = c (M^T P - st); continuity at free nodes: M_f Q = -d_f
        L_ff = (Mf * c) @ Mf.T
        L_fx = (Mf * c) @ Mx.T
        rhs = -d_free + (Mf * c) @ st - L_fx @ P_fixed
        if len(free):
            P_free = jnp.linalg.solve(L_ff, rhs)
        else:
            P_free = jnp.zeros((0,))
        P_all = jnp.zeros(len(self.nodes)).at[jnp.asarray(free, dtype=int)].set(P_free)
        P_all = P_all.at[jnp.asarray(fixed, dtype=int)].set(P_fixed)
        Q = c * (M.T @ P_all - st)
        # Q carries the (exact, linear) dependence on demands and fixed
        # pressures through continuity, which Hardy Cross preserves; the
        # conductances are stop-gradient, so only the starting point uses them.
        return Q, P_free

    def solve(
        self,
        fluid: Stream | None = None,
        method: str = "newton",
        tol: float = 1e-10,
        max_steps: int = 100,
        iterations: int = 200,
        Q0: Array | None = None,
    ) -> PipeNetworkResult:
        """Solve for the branch flows and node pressures.

        Args:
            fluid: Liquid stream, only needed when a Pipe's ``rho``/``mu`` are
                None (liquid_properties) or callables of the stream.
            method: ``"newton"`` (default) or ``"hardy_cross"``.
            tol: Convergence tolerance on the scaled residuals (flows relative
                to the total demand, pressures relative to the spread of fixed
                pressures or 1e4 Pa).
            max_steps: Newton step limit.
            iterations: Hardy Cross sweeps (fixed: the loop is unrolled for
                differentiation).
            Q0: Optional initial branch flows for Newton (m^3/s); default from
                the linearized network. Hardy Cross always starts from the
                linearized network (its start must satisfy continuity).

        Returns:
            :class:`PipeNetworkResult`.

        Raises:
            ValueError: No fixed-pressure node, a disconnected node, or an
                unknown method.
        """
        if method not in NETWORK_METHODS:
            raise ValueError(f"method must be one of {NETWORK_METHODS}, got {method!r}")
        self._check()
        fixed = [i for i, n in enumerate(self.nodes) if n in self.fixed_pressure]
        free = [i for i, n in enumerate(self.nodes) if n not in self.fixed_pressure]
        P_fixed = jnp.stack([jnp.asarray(self.fixed_pressure[self.nodes[i]], dtype=jnp.float64)
                             for i in fixed])
        d_free = (jnp.stack([jnp.asarray(self.demand[self.nodes[i]], dtype=jnp.float64) for i in free])
                  if free else jnp.zeros((0,)))
        M = self._incidence()
        sg = jax.lax.stop_gradient
        Q_scale = jnp.maximum(jnp.sum(jnp.abs(sg(d_free))), 1e-6)
        P_spread = sg(jnp.max(P_fixed) - jnp.min(P_fixed))
        P_scale = jnp.maximum(P_spread, 1e4)

        Q_lin, P_lin = self._linear_start(fluid, M, free, fixed, d_free, P_fixed, Q_scale)
        if method == "newton":
            if Q0 is not None:
                Q_lin = jnp.asarray(Q0, dtype=jnp.float64)
            Q, P_free, steps, ok = self._solve_newton(fluid, M, free, fixed, d_free, P_fixed,
                                                      Q_lin, P_lin, Q_scale, P_scale, tol, max_steps)
        else:
            Q, P_free, steps, ok = self._solve_hardy_cross(fluid, M, free, fixed, d_free, P_fixed,
                                                           Q_lin, Q_scale, P_scale, tol, iterations)

        P = jnp.zeros(len(self.nodes))
        if free:
            P = P.at[jnp.asarray(free)].set(P_free)
        P = P.at[jnp.asarray(fixed)].set(P_fixed)
        dp = self._dp_vector(Q, fluid)
        mom = dp - M.T @ P
        cont = M[jnp.asarray(free, dtype=int)] @ Q + d_free if free else jnp.zeros((0,))
        mass_err = jnp.max(jnp.abs(cont)) if free else jnp.asarray(0.0)
        energy_err = jnp.max(jnp.abs(mom))
        converged = ok & (mass_err <= 1e3 * tol * Q_scale) & (energy_err <= 1e3 * tol * P_scale)
        supply = {self.nodes[i]: (M[i] @ Q) for i in fixed}
        names = [b[0] for b in self.branches]
        return PipeNetworkResult(
            Q=Q, P=P,
            branch_flows={n: Q[k] for k, n in enumerate(names)},
            node_pressures={n: P[i] for i, n in enumerate(self.nodes)},
            branch_dP={n: dp[k] for k, n in enumerate(names)},
            supply=supply,
            converged=converged,
            mass_imbalance=mass_err,
            energy_residual=energy_err,
            iterations=jnp.asarray(steps),
        )

    # ------------------------------------------------------------------
    def _solve_newton(self, fluid, M, free, fixed, d_free, P_fixed, Q_lin, P_lin,
                      Q_scale, P_scale, tol, max_steps):
        nb = len(self.branches)
        free_idx = jnp.asarray(free, dtype=int)
        fixed_idx = jnp.asarray(fixed, dtype=int)
        Mf = M[free_idx]
        n_nodes = len(self.nodes)

        def unpack(x, args):
            P_fixed_, d_free_, Qs, Ps = args
            Q = x[:nb] * Qs
            P = jnp.zeros(n_nodes).at[fixed_idx].set(P_fixed_)
            if free:
                P = P.at[free_idx].set(x[nb:] * Ps)
            return Q, P

        def resid(x, args):
            P_fixed_, d_free_, Qs, Ps = args
            Q, P = unpack(x, args)
            r_mom = (self._dp_vector(Q, fluid) - M.T @ P) / Ps
            if not free:
                return r_mom
            r_cont = (Mf @ Q + d_free_) / Qs
            return jnp.concatenate([r_mom, r_cont])

        args = (P_fixed, d_free, Q_scale, P_scale)
        x0 = jax.lax.stop_gradient(jnp.concatenate([Q_lin / Q_scale, P_lin / P_scale]))
        sol = optx.root_find(resid, optx.Newton(rtol=tol, atol=tol), x0, args,
                             max_steps=max_steps, throw=False)
        Q, P = unpack(sol.value, args)
        ok = sol.result == optx.RESULTS.successful
        return Q, P[free_idx] if free else jnp.zeros((0,)), sol.stats["num_steps"], ok

    def _solve_hardy_cross(self, fluid, M, free, fixed, d_free, P_fixed, Q_lin,
                           Q_scale, P_scale, tol, iterations):
        nb = len(self.branches)
        # augmented graph: a datum node "G" (P = 0) joined to every fixed node
        G = "__datum__"
        nodes = [G] + list(self.nodes)
        edges = [(a, b) for _, a, b, _ in self.branches]
        edges += [(G, self.nodes[i]) for i in fixed]
        loops = self._cycle_basis(nodes, edges, root=G)
        n_e = len(edges)
        O = jnp.asarray([[lp.get(k, 0.0) for k in range(n_e)] for lp in loops],
                        dtype=jnp.float64).reshape(len(loops), n_e)
        # virtual branch G -> fixed node i carries that node's supply, dP = 0 - P_i
        q_virtual = jnp.stack([M[i] @ Q_lin for i in fixed])
        q = jnp.concatenate([Q_lin, q_virtual])

        def dp_all(q_):
            return jnp.concatenate([self._dp_vector(q_[:nb], fluid), -P_fixed])

        def ddp_all(q_):
            return jax.jvp(dp_all, (q_,), (jnp.ones_like(q_),))[1]

        def sweep(_, q_):
            for li in range(len(loops)):
                o = O[li]
                dp = dp_all(q_)
                ddp = ddp_all(q_)
                dq = -jnp.dot(o, dp) / jnp.sum(o**2 * ddp)
                q_ = q_ + dq * o
            return q_

        if len(loops):
            q = jax.lax.fori_loop(0, iterations, sweep, q)
        Q = q[:nb]
        # pressures along the spanning tree from the datum (P_G = 0)
        dp = dp_all(q)
        P_of = {G: jnp.asarray(0.0)}
        adj = {n: [] for n in nodes}
        for k, (a, b) in enumerate(edges):
            adj[a].append((k, b, 1.0))
            adj[b].append((k, a, -1.0))
        todo = deque([G])
        while todo:
            u = todo.popleft()
            for k, v, sgn in adj[u]:
                if v in P_of:
                    continue
                # sgn = +1: edge u -> v, P_v = P_u - dp_k ; sgn = -1: edge v -> u, P_v = P_u + dp_k
                P_of[v] = P_of[u] - sgn * dp[k]
                todo.append(v)
        P_free = (jnp.stack([P_of[self.nodes[i]] for i in free]) if free else jnp.zeros((0,)))
        loop_err = jnp.max(jnp.abs(O @ dp)) if len(loops) else jnp.asarray(0.0)
        ok = loop_err <= 1e3 * tol * P_scale
        return Q, P_free, iterations, ok


__all__ = [
    "PipeNetwork",
    "PipeNetworkResult",
    "ResistanceBranch",
    "NETWORK_METHODS",
    "Q_EPS",
]
