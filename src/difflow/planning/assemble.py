"""Turn a linearised network into an :class:`~difflow.planning.lp.LPModel`.

This is the translation layer: delta vectors in, LP columns and rows out.
It is separated from the trust-region loop so that a single linearisation can
be inspected, exported to Pyomo, or re-solved under different prices without
re-running the planner.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from difflow.planning.linearize import Linearization
from difflow.planning.lp import LPModel, as_spec
from difflow.planning.network import Network
from difflow.planning.piecewise import PiecewiseData


def trust_region_bounds(lo: np.ndarray, hi: np.ndarray, center: np.ndarray,
                        radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Intersect physical bounds with a trust region around ``center``.

    The radius is a fraction of each variable's bound range.  Where a variable
    is unbounded the range is undefined, so the radius is applied to
    ``max(|center|, 1)`` instead, which keeps the step scale-aware without
    requiring the caller to invent artificial bounds.

    Args:
        lo: Physical lower bounds.
        hi: Physical upper bounds.
        center: Trust-region centre.
        radius: Fraction of range (``>= 0``).

    Returns:
        ``(lb, ub)`` arrays.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    center = np.asarray(center, dtype=float)
    span = hi - lo
    finite = np.isfinite(span)
    # np.where evaluates both branches, so substitute a finite placeholder in
    # the unbounded slots before combining -- otherwise `radius * inf` leaks a
    # nan into the bounds when radius is zero.
    scale = np.where(finite, np.where(finite, span, 0.0),
                     np.maximum(np.abs(center), 1.0))
    step = radius * scale
    return (np.maximum(lo, center - step), np.minimum(hi, center + step))


def build_lp(network: Network,
             linearizations: Mapping[str, Linearization],
             prices: Mapping[str, float],
             specs: Sequence[Any] = (),
             centers: Mapping[str, Any] | None = None,
             radius: float = 0.3,
             penalty: float = 1e4,
             sense: str = "max",
             piecewise: Mapping[str, PiecewiseData] | None = None) -> LPModel:
    """Assemble the delta-base LP.

    Args:
        network: The block network.
        linearizations: ``{block name: Linearization}`` covering every block.
        prices: ``{qualified variable: price}``.  Positive prices on outputs
            are revenues; put a negative price on a lever to charge for it.
            Variables absent from the dict are free of charge.
        specs: Product/quality constraints; see
            :class:`~difflow.planning.lp.Spec`.
        centers: ``{block name: input array}`` for the trust region.  Defaults
            to each linearisation point.
        radius: Trust-region radius as a fraction of each variable's range.
            ``None`` or a non-finite value disables the trust region.
        penalty: Default cost per unit of elastic spec slack.
        sense: ``"max"`` (default) or ``"min"``.
        piecewise: ``{block name: PiecewiseData}``.  Named blocks get an SOS2
            piecewise-linear model of their distinguished variable instead of
            a single delta vector, which makes the problem a MILP.  The
            trust region is not applied to that variable: the piecewise model
            is valid across the whole sampled range, which is the point of
            building it.

    Returns:
        An :class:`LPModel`.

    Raises:
        KeyError: If a price or spec names an unknown variable, or a block has
            no linearisation.
    """
    if sense not in ("max", "min"):
        raise ValueError(f"sense must be 'max' or 'min', got {sense!r}")
    sign = -1.0 if sense == "max" else 1.0

    missing = [b.name for b in network.blocks if b.name not in linearizations]
    if missing:
        raise KeyError(f"no linearisation supplied for block(s) {missing}")

    columns: list[str] = []
    col_of: dict[str, int] = {}
    lb_list: list[float] = []
    ub_list: list[float] = []

    def add_col(name: str, lo: float, hi: float) -> int:
        if name in col_of:
            raise ValueError(f"duplicate LP column {name!r}")
        col_of[name] = len(columns)
        columns.append(name)
        lb_list.append(lo)
        ub_list.append(hi)
        return col_of[name]

    tr_active = radius is not None and np.isfinite(radius)
    piecewise = dict(piecewise or {})
    unknown_pw = sorted(set(piecewise) - {b.name for b in network.blocks})
    if unknown_pw:
        raise KeyError(f"piecewise data for unknown block(s) {unknown_pw}")

    # -- columns: block inputs (bounded, trust-region clipped) then outputs
    for b in network.blocks:
        lo = np.asarray(b.lb, dtype=float)
        hi = np.asarray(b.ub, dtype=float)
        if centers is not None and b.name in centers:
            center = np.asarray(centers[b.name], dtype=float)
        else:
            center = np.asarray(linearizations[b.name].u0, dtype=float)
        if tr_active:
            tr_lo, tr_hi = trust_region_bounds(lo, hi, center, float(radius))
            pw = piecewise.get(b.name)
            if pw is not None:
                # The piecewise model spans its whole grid, so the trust
                # region must not clip the variable it was built for.
                tr_lo[pw.index] = max(float(lo[pw.index]),
                                      float(pw.breakpoints[0]))
                tr_hi[pw.index] = min(float(hi[pw.index]),
                                      float(pw.breakpoints[-1]))
            lo, hi = tr_lo, tr_hi
        for j, name in enumerate(b.qualified_u()):
            add_col(name, float(lo[j]), float(hi[j]))
    for b in network.blocks:
        for name in b.qualified_y():
            add_col(name, -np.inf, np.inf)

    spec_objs = [as_spec(s, penalty) for s in specs]
    slack_cols: dict[str, int] = {}
    for s in spec_objs:
        if s.elastic:
            label = f"slack[{s.name}]"
            base = label
            k = 1
            while label in col_of:
                k += 1
                label = f"{base}#{k}"
            slack_cols[s.name] = add_col(label, 0.0, np.inf)

    # -- piecewise columns: convex-combination weights and interval binaries
    lam_cols: dict[str, list[int]] = {}
    z_cols: dict[str, list[int]] = {}
    integer_cols: list[int] = []
    for bname, pw in piecewise.items():
        K = pw.n_points
        lam_cols[bname] = [add_col(f"{bname}.lambda[{k}]", 0.0, 1.0)
                           for k in range(K)]
        z = [add_col(f"{bname}.z[{k}]", 0.0, 1.0) for k in range(K - 1)]
        z_cols[bname] = z
        integer_cols.extend(z)

    n = len(columns)

    # -- equality rows: model rows then link rows ------------------------
    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []
    eq_names: list[str] = []

    for b in network.blocks:
        if b.name in piecewise:
            continue
        lin = linearizations[b.name]
        J = np.asarray(lin.J, dtype=float)
        y0 = np.asarray(lin.y0, dtype=float)
        u0 = np.asarray(lin.u0, dtype=float)
        u_cols = [col_of[nm] for nm in b.qualified_u()]
        for i, yname in enumerate(b.qualified_y()):
            row = np.zeros(n)
            row[col_of[yname]] = 1.0
            for j, cj in enumerate(u_cols):
                row[cj] -= J[i, j]
            eq_rows.append(row)
            eq_rhs.append(float(y0[i] - J[i] @ u0))
            eq_names.append(f"model[{yname}]")

    for link in network.links:
        row = np.zeros(n)
        row[col_of[link.target]] = 1.0
        row[col_of[link.source]] -= 1.0
        eq_rows.append(row)
        eq_rhs.append(0.0)
        eq_names.append(f"link[{link.source}->{link.target}]")

    # -- inequality rows: specs -----------------------------------------
    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []
    ub_names: list[str] = []

    # -- piecewise rows --------------------------------------------------
    sos2_sets: list[list[int]] = []
    for bname, pw in piecewise.items():
        b = network.block(bname)
        lam = lam_cols[bname]
        z = z_cols[bname]
        K = pw.n_points
        u_cols = [col_of[nm] for nm in b.qualified_u()]
        sos2_sets.append(list(lam))

        row = np.zeros(n)
        row[lam] = 1.0
        eq_rows.append(row)
        eq_rhs.append(1.0)
        eq_names.append(f"pw_convex[{bname}]")

        row = np.zeros(n)
        row[u_cols[pw.index]] = 1.0
        for k, c in enumerate(lam):
            row[c] -= float(pw.breakpoints[k])
        eq_rows.append(row)
        eq_rhs.append(0.0)
        eq_names.append(f"pw_grid[{bname}.{pw.variable}]")

        for i, yname in enumerate(b.qualified_y()):
            row = np.zeros(n)
            row[col_of[yname]] = 1.0
            for k, c in enumerate(lam):
                row[c] -= float(pw.y[k, i])
            rhs = 0.0
            for m, cm in enumerate(u_cols):
                if m == pw.index:
                    continue
                coef = float(pw.cross_jacobian[i, m])
                if coef == 0.0:
                    continue
                row[cm] -= coef
                rhs -= coef * float(pw.center[m])
            eq_rows.append(row)
            eq_rhs.append(rhs)
            eq_names.append(f"pw_model[{yname}]")

        row = np.zeros(n)
        row[z] = 1.0
        eq_rows.append(row)
        eq_rhs.append(1.0)
        eq_names.append(f"pw_interval[{bname}]")

        # SOS2 adjacency: lambda_k may only be nonzero on a chosen interval.
        for k in range(K):
            row = np.zeros(n)
            row[lam[k]] = 1.0
            for i in (k - 1, k):
                if 0 <= i <= K - 2:
                    row[z[i]] -= 1.0
            ub_rows.append(row)
            ub_rhs.append(0.0)
            ub_names.append(f"pw_sos2[{bname}.lambda[{k}]]")

    for s in spec_objs:
        for var in s.coeffs:
            if var not in col_of:
                raise KeyError(
                    f"spec {s!r} references unknown variable {var!r}. "
                    f"Known variables: {sorted(col_of)[:8]}...")
        base = np.zeros(n)
        for var, c in s.coeffs.items():
            base[col_of[var]] += c
        slack = slack_cols.get(s.name)

        def _emit(coef: np.ndarray, rhs: float, tag: str) -> None:
            row = coef.copy()
            if slack is not None:
                row[slack] -= 1.0
            ub_rows.append(row)
            ub_rhs.append(rhs)
            ub_names.append(tag)

        rhs = s.effective_rhs
        if s.op in ("<=", "=="):
            _emit(base, rhs, f"spec[{s.name}<=]")
        if s.op in (">=", "=="):
            _emit(-base, -rhs, f"spec[{s.name}>=]")

    # -- objective -------------------------------------------------------
    c = np.zeros(n)
    for var, price in prices.items():
        if var not in col_of:
            raise KeyError(
                f"price given for unknown variable {var!r}. Known variables "
                f"include {sorted(col_of)[:8]}...")
        c[col_of[var]] += sign * float(price)
    for s in spec_objs:
        j = slack_cols.get(s.name)
        if j is not None:
            pen = penalty if s.penalty is None else s.penalty
            # A slack is always a cost, whichever way the objective points.
            c[j] += abs(float(pen))

    return LPModel(
        columns=columns,
        c=c,
        A_eq=(np.stack(eq_rows) if eq_rows else np.zeros((0, n))),
        b_eq=np.asarray(eq_rhs, dtype=float),
        A_ub=(np.stack(ub_rows) if ub_rows else np.zeros((0, n))),
        b_ub=np.asarray(ub_rhs, dtype=float),
        lb=np.asarray(lb_list, dtype=float),
        ub=np.asarray(ub_list, dtype=float),
        eq_names=eq_names,
        ub_names=ub_names,
        integer_cols=integer_cols,
        sense=sign,
        specs=spec_objs,
        slack_cols=slack_cols,
        sos2_sets=sos2_sets)
