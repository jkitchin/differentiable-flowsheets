"""Assembly and solution of the delta-base linear program.

The LP has one column per block input, one per block output, and one per
elastic spec slack.  Its rows are:

* **model rows** ``y_b - J_b u_b = y0_b - J_b u0_b`` — the delta vectors;
* **link rows** ``u_target - y_source = 0`` — the network connectivity;
* **spec rows** ``a . x - s <= rhs`` — product/quality constraints, elastic
  by default so the LP always has a feasible point to report;
* **bounds**, which carry both the physical limits and the trust region.

The objective is linear in the variables, which is what keeps the problem an
LP: pricing is applied to block outputs (and optionally to levers).  Anything
bilinear — pooling and blending in particular — stays nonconvex no matter how
good the unit linearisation is, and is explicitly out of scope; see the module
documentation.

The model is solved with HiGHS through :func:`scipy.optimize.linprog` (or
:func:`scipy.optimize.milp` when integer columns are present) and can be
emitted as a Pyomo ``ConcreteModel`` via :meth:`LPModel.to_pyomo` for reuse in
the IDAES/Pyomo ecosystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

_OPS = {"<=", ">=", "==", "<", ">", "="}
_CANON = {"<": "<=", ">": ">=", "=": "=="}


@dataclass
class Spec:
    """A linear constraint on planning variables.

    Attributes:
        expr: Either a qualified variable name, or a dict of
            ``{qualified name: coefficient}`` for a linear combination.
        op: ``"<="``, ``">="`` or ``"=="``.
        rhs: Right-hand side.
        elastic: Allow violation at a price (``penalty`` per unit).  Elastic
            specs keep the LP feasible; the *realised* violation is charged
            against the nonlinear model by the planner, never against the LP's
            own prediction.
        penalty: Cost per unit of slack.  Defaults to the planner's
            ``penalty``.
        name: Optional label used in reports.
        backoff: Constraint back-off subtracted from an upper bound (added to
            a lower bound), typically ``kappa * sigma`` from coefficient
            covariance.  See :mod:`difflow.planning.backoff`.
    """

    expr: Any
    op: str
    rhs: float
    elastic: bool = True
    penalty: float | None = None
    name: str | None = None
    backoff: float = 0.0

    def __post_init__(self):
        if self.op not in _OPS:
            raise ValueError(f"spec operator must be one of {sorted(_OPS)}, "
                             f"got {self.op!r}")
        self.op = _CANON.get(self.op, self.op)
        if isinstance(self.expr, str):
            self.coeffs = {self.expr: 1.0}
        elif isinstance(self.expr, Mapping):
            if not self.expr:
                raise ValueError("spec expression is empty")
            self.coeffs = {k: float(v) for k, v in self.expr.items()}
        else:
            raise TypeError(
                "spec expression must be a variable name or a "
                "{name: coefficient} dict")
        self.rhs = float(self.rhs)
        if self.name is None:
            lead = next(iter(self.coeffs))
            self.name = (lead if len(self.coeffs) == 1
                         else f"{lead}+{len(self.coeffs) - 1}more")

    @property
    def effective_rhs(self) -> float:
        """Right-hand side after applying the back-off."""
        if self.op == "<=":
            return self.rhs - self.backoff
        if self.op == ">=":
            return self.rhs + self.backoff
        return self.rhs

    def value(self, values: Mapping[str, float]) -> float:
        """Evaluate the left-hand side from a variable dict."""
        return sum(c * float(values[k]) for k, c in self.coeffs.items())

    def violation(self, values: Mapping[str, float]) -> float:
        """Non-negative violation of the *stated* bound (back-off excluded).

        Back-off is a safety margin the planner chooses, not a promise made to
        anyone downstream, so violating it is not a violation of the spec.
        """
        v = self.value(values)
        if self.op == "<=":
            return max(0.0, v - self.rhs)
        if self.op == ">=":
            return max(0.0, self.rhs - v)
        return abs(v - self.rhs)

    def __repr__(self) -> str:
        lhs = (next(iter(self.coeffs)) if len(self.coeffs) == 1
               else " + ".join(f"{c:g}*{k}" for k, c in self.coeffs.items()))
        back = f", backoff={self.backoff:g}" if self.backoff else ""
        return f"Spec({lhs} {self.op} {self.rhs:g}{back})"


def as_spec(item: Any, default_penalty: float | None = None) -> Spec:
    """Coerce a tuple or :class:`Spec` into a :class:`Spec`.

    Accepts ``(name, op, rhs)``, ``(name, op, rhs, penalty)`` or a
    :class:`Spec`.
    """
    if isinstance(item, Spec):
        if item.penalty is None:
            item.penalty = default_penalty
        return item
    if isinstance(item, (tuple, list)):
        if len(item) == 3:
            return Spec(item[0], item[1], item[2], penalty=default_penalty)
        if len(item) == 4:
            return Spec(item[0], item[1], item[2], penalty=float(item[3]))
    raise TypeError(
        "specs must be Spec objects or (variable, op, rhs[, penalty]) tuples, "
        f"got {item!r}")


@dataclass
class LPModel:
    """A structured linear (or mixed-integer linear) program.

    The problem solved is::

        minimize    c . x
        subject to  A_eq x  = b_eq
                    A_ub x <= b_ub
                    lb <= x <= ub
                    x_j integral for j in integer_cols

    Attributes:
        columns: Column names in order.  Block variables keep their qualified
            names; slacks are ``"slack[<spec>]"``.
        c: Objective coefficients (minimisation).
        A_eq, b_eq: Equality rows (model and link rows).
        A_ub, b_ub: Inequality rows (specs).
        lb, ub: Column bounds, already intersected with the trust region.
        eq_names, ub_names: Row labels.
        integer_cols: Indices of integer-restricted columns.
        sense: ``1.0`` if the caller's objective was a minimisation, ``-1.0``
            when it was a maximisation folded into ``c``.
        objective_offset: Constant added to ``c . x`` to recover the caller's
            objective.
        specs: The specs that produced the inequality rows.
        slack_cols: ``{spec name: column index}`` for elastic slacks.
    """

    columns: list[str]
    c: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    eq_names: list[str] = field(default_factory=list)
    ub_names: list[str] = field(default_factory=list)
    integer_cols: list[int] = field(default_factory=list)
    sense: float = -1.0
    objective_offset: float = 0.0
    specs: list[Spec] = field(default_factory=list)
    slack_cols: dict[str, int] = field(default_factory=dict)
    sos2_sets: list[list[int]] = field(default_factory=list)

    @property
    def n_cols(self) -> int:
        return len(self.columns)

    def col(self, name: str) -> int:
        """Column index of a variable by name."""
        try:
            return self.columns.index(name)
        except ValueError:
            raise KeyError(f"{name!r} is not a column of this LP")

    def solve(self, solver: str = "auto") -> "LPSolution":
        """Solve with HiGHS via SciPy.

        Args:
            solver: ``"auto"`` picks ``milp`` when integer columns are present
                and ``linprog`` otherwise.  ``"linprog"`` or ``"milp"`` force
                one.

        Returns:
            An :class:`LPSolution`.
        """
        from scipy.optimize import linprog, milp, Bounds, LinearConstraint

        use_milp = (solver == "milp"
                    or (solver == "auto" and bool(self.integer_cols)))
        if solver not in ("auto", "linprog", "milp"):
            raise ValueError(f"unknown solver {solver!r}")
        if solver == "linprog" and self.integer_cols:
            raise ValueError(
                "this model has integer columns; use solver='milp'")

        lb = np.where(np.isfinite(self.lb), self.lb, -np.inf)
        ub = np.where(np.isfinite(self.ub), self.ub, np.inf)

        if use_milp:
            integrality = np.zeros(self.n_cols)
            integrality[self.integer_cols] = 1
            constraints = []
            if self.A_eq.size:
                constraints.append(
                    LinearConstraint(self.A_eq, self.b_eq, self.b_eq))
            if self.A_ub.size:
                constraints.append(
                    LinearConstraint(self.A_ub, -np.inf, self.b_ub))
            res = milp(c=self.c, constraints=constraints,
                       bounds=Bounds(lb, ub), integrality=integrality)
            x = None if res.x is None else np.asarray(res.x)
            duals = {}
        else:
            res = linprog(
                self.c,
                A_ub=self.A_ub if self.A_ub.size else None,
                b_ub=self.b_ub if self.A_ub.size else None,
                A_eq=self.A_eq if self.A_eq.size else None,
                b_eq=self.b_eq if self.A_eq.size else None,
                bounds=list(zip(lb, ub)), method="highs")
            x = None if res.x is None else np.asarray(res.x)
            duals = _extract_duals(res, self)

        success = bool(getattr(res, "success", False)) and x is not None
        obj = (float(self.c @ x) if success else float("nan"))
        return LPSolution(
            model=self, x=x, success=success,
            status=int(getattr(res, "status", -1)),
            message=str(getattr(res, "message", "")),
            lp_objective=obj,
            objective=(self.sense * obj + self.objective_offset
                       if success else float("nan")),
            duals=duals)

    def to_pyomo(self, name: str = "delta_base_plan"):
        """Emit the model as a Pyomo ``ConcreteModel``.

        difflow.planning deliberately does not own a solver.  Emitting Pyomo
        lets the plan compose with the existing Pyomo/IDAES ecosystem — the
        same model can be handed to CBC, Gurobi, CPLEX or an IDAES
        superstructure — instead of competing with it.

        Args:
            name: Model name.

        Returns:
            A ``pyomo.environ.ConcreteModel`` with ``m.x`` indexed by column
            name, ``m.obj``, ``m.eq``, ``m.ub`` and, where applicable,
            ``m.sos2``.

        Raises:
            ImportError: If Pyomo is not installed.
        """
        try:
            import pyomo.environ as pyo
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "to_pyomo() requires Pyomo. Install it with "
                "`pip install pyomo` (difflow.planning keeps it optional so "
                "the module works without a Pyomo installation)."
            ) from exc

        m = pyo.ConcreteModel(name=name)
        cols = list(self.columns)
        index_of = {c: i for i, c in enumerate(cols)}
        m.COLS = pyo.Set(initialize=cols, ordered=True)
        integer = set(self.integer_cols)

        def _bounds(_m, j):
            i = index_of[j]
            lo = float(self.lb[i]) if np.isfinite(self.lb[i]) else None
            hi = float(self.ub[i]) if np.isfinite(self.ub[i]) else None
            return (lo, hi)

        def _domain(_m, j):
            return pyo.Integers if index_of[j] in integer else pyo.Reals

        m.x = pyo.Var(m.COLS, bounds=_bounds, domain=_domain)

        m.obj = pyo.Objective(
            expr=sum(float(self.c[i]) * m.x[cols[i]]
                     for i in range(self.n_cols) if self.c[i] != 0.0),
            sense=pyo.minimize)

        if self.A_eq.size:
            m.EQ = pyo.Set(initialize=list(range(self.A_eq.shape[0])),
                           ordered=True)

            def _eq(_m, r):
                row = self.A_eq[r]
                nz = np.nonzero(row)[0]
                return (sum(float(row[i]) * _m.x[cols[i]] for i in nz)
                        == float(self.b_eq[r]))

            m.eq = pyo.Constraint(m.EQ, rule=_eq)

        if self.A_ub.size:
            m.UB = pyo.Set(initialize=list(range(self.A_ub.shape[0])),
                           ordered=True)

            def _ub(_m, r):
                row = self.A_ub[r]
                nz = np.nonzero(row)[0]
                return (sum(float(row[i]) * _m.x[cols[i]] for i in nz)
                        <= float(self.b_ub[r]))

            m.ub = pyo.Constraint(m.UB, rule=_ub)

        if self.sos2_sets:
            m.SOS2SETS = pyo.Set(initialize=list(range(len(self.sos2_sets))),
                                 ordered=True)
            m.sos2 = pyo.SOSConstraint(
                m.SOS2SETS,
                var=m.x,
                index={k: [cols[i] for i in s]
                       for k, s in enumerate(self.sos2_sets)},
                sos=2)

        m.difflow_sense = self.sense
        m.difflow_offset = self.objective_offset
        return m

    def __repr__(self) -> str:
        return (f"LPModel(cols={self.n_cols}, eq={self.A_eq.shape[0]}, "
                f"ub={self.A_ub.shape[0]}, int={len(self.integer_cols)})")


def _extract_duals(res, model: "LPModel") -> dict[str, Any]:
    """Pull HiGHS marginals into named dicts, tolerating their absence."""
    duals: dict[str, Any] = {}
    eq = getattr(res, "eqlin", None)
    if eq is not None and getattr(eq, "marginals", None) is not None:
        vals = np.asarray(eq.marginals)
        duals["eq"] = {n: float(v) for n, v in zip(model.eq_names, vals)}
    ineq = getattr(res, "ineqlin", None)
    if ineq is not None and getattr(ineq, "marginals", None) is not None:
        vals = np.asarray(ineq.marginals)
        duals["ub"] = {n: float(v) for n, v in zip(model.ub_names, vals)}
    upper = getattr(res, "upper", None)
    lower = getattr(res, "lower", None)
    if upper is not None and getattr(upper, "marginals", None) is not None:
        duals["upper"] = {n: float(v) for n, v
                          in zip(model.columns, np.asarray(upper.marginals))}
    if lower is not None and getattr(lower, "marginals", None) is not None:
        duals["lower"] = {n: float(v) for n, v
                          in zip(model.columns, np.asarray(lower.marginals))}
    return duals


@dataclass
class LPSolution:
    """Solution of an :class:`LPModel`.

    Attributes:
        model: The model that was solved.
        x: Primal solution, or ``None`` when infeasible.
        success: Whether the solver reported optimality.
        status: SciPy status code.
        message: Solver message.
        lp_objective: Value of ``c . x`` (a minimisation).
        objective: The caller's objective, sign- and offset-corrected.
        duals: Marginals by row/bound name, where the solver provided them.
    """

    model: LPModel
    x: np.ndarray | None
    success: bool
    status: int
    message: str
    lp_objective: float
    objective: float
    duals: dict[str, Any] = field(default_factory=dict)

    def values(self) -> dict[str, float]:
        """Primal solution as ``{column name: value}``."""
        if self.x is None:
            return {}
        return {n: float(v) for n, v in zip(self.model.columns, self.x)}

    def __getitem__(self, name: str) -> float:
        if self.x is None:
            raise KeyError("LP has no solution")
        return float(self.x[self.model.col(name)])

    def slacks(self) -> dict[str, float]:
        """Elastic slack activity by spec name."""
        if self.x is None:
            return {}
        return {name: float(self.x[j])
                for name, j in self.model.slack_cols.items()}

    def active_bounds(self, atol: float = 1e-7) -> dict[str, str]:
        """Columns sitting on a bound, and which one.

        Returns:
            ``{column: "lb"|"ub"}`` for columns within ``atol`` of a finite
            bound.
        """
        if self.x is None:
            return {}
        out = {}
        for i, name in enumerate(self.model.columns):
            lo, hi = self.model.lb[i], self.model.ub[i]
            if np.isfinite(lo) and abs(self.x[i] - lo) <= atol:
                out[name] = "lb"
            elif np.isfinite(hi) and abs(self.x[i] - hi) <= atol:
                out[name] = "ub"
        return out

    def __repr__(self) -> str:
        return (f"LPSolution(success={self.success}, "
                f"objective={self.objective:.6g})")
