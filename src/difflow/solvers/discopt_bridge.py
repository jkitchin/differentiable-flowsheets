"""Bridge a difflow residual into a ``discopt`` model -- and its one hard limit.

``discopt.modeling.implicit(g, u_inputs, n_unknowns)`` takes a residual
``g(u, v) = 0`` and compiles it into a differentiable inner Newton solve
whose derivatives come from ``jax.lax.custom_root``. That is exactly what
:func:`difflow.solvers.residual.as_residual` produces, so a difflow unit or
section can be dropped into an algebraic discopt model as one opaque block.

The restriction, stated plainly
-------------------------------
``dm.implicit`` is built on ``dm.custom``, which produces a ``CustomCall``
node. A ``CustomCall`` is an opaque, AD-only callable, and discopt's
guarantees are all built on being able to *see* the algebra:

* **No global optimality certificate.** The solve reports
  ``status="feasible"`` with the certificate withheld -- ``bound`` and
  ``gap`` do not describe global optimality. Which root the inner Newton
  lands in *is* the definition of ``v``, so two starting points in the same
  box can legitimately give two different "optima".
* **Relaxation compilation and ``.nl`` export raise.** The Rust tape
  refuses a ``CustomCall`` outright -- ``_nl_expr_compiler.py`` raises
  ``UnsupportedForTape("CustomCall (dm.custom) has no tape equivalent")`` --
  so the model falls back to the JAX evaluator. In practice ``Model.to_nl``
  on a model containing an *indexed* implicit node fails earlier still, with
  ``ValueError: Cannot resolve indexed expression: custom:...[i]``; the exact
  message depends on where the walk gives up, but the export never succeeds.
* **Integer or binary variables make the solve raise.** Spatial branch and
  bound has no valid node relaxation for an opaque callable, so discopt
  refuses rather than returning an unsound bound.

Verified against discopt's source: the gate lives in
``discopt/solver.py`` (``_model_contains_custom_call`` ->
``_custom_call_reduced_admissible`` -> the ``ValueError`` when
``not _is_pure_continuous(model)``). Current discopt has one refinement
worth knowing: a ``CustomCall`` whose body traces soundly through discopt's
McCormick-box (MCBox) intrinsics *is* globally relaxable, integers included.
A difflow residual never qualifies -- ``dm.implicit``'s forward is a
``jax.lax.while_loop`` Newton iteration over raw ``jnp`` intrinsics, which
is outside MCBox scope by construction -- so for difflow the restriction is
the strict one above. :func:`check_no_integrality` enforces it at build
time rather than letting you discover it at solve time.

**So you cannot wrap a difflow flowsheet as a UDF and then put binaries
around it in one discopt model.** If you need integrality or a global bound
over a flowsheet, decompose: solve the flowsheet-shaped subproblem with the
NLP bridge (:mod:`difflow.solvers.pounce_bridge`) inside a master problem
that owns the binaries, or re-express the block in discopt's own algebraic
language (``dm.exp``, ``dm.log``, ...) where a relaxation exists. discopt's
``implicit(..., formulation="full_space")`` is the second option applied to
this exact node: it lowers ``v`` to real variables and the residuals to real
equality constraints, which keeps a certificate reachable -- but it requires
the residual to be written in discopt operators, so a JAX-traced difflow
residual cannot be passed to it.
"""

from __future__ import annotations

from typing import Any, Sequence

from difflow.solvers._lazy import require
from difflow.solvers.residual import ResidualView, as_residual

__all__ = [
    "DiscoptIntegralityError",
    "CUSTOMCALL_RESTRICTION",
    "check_no_integrality",
    "integer_variables",
    "as_implicit",
]

#: One-paragraph statement of the ``CustomCall`` restriction, for error
#: messages and for anything that wants to surface it in a report.
CUSTOMCALL_RESTRICTION = (
    "A difflow residual enters a discopt model through dm.implicit, which "
    "builds a CustomCall: an opaque AD-only node. Such a model is solved on "
    "the LOCAL NLP path only -- no global optimality certificate, and "
    "relaxation compilation and .nl export raise. Most importantly the solve "
    "RAISES if any integer or binary variable is present, because spatial "
    "branch and bound has no valid node relaxation for an opaque callable. "
    "Decompose instead: keep the binaries in a master problem and solve the "
    "flowsheet block as an NLP, or re-express the block in discopt's "
    "algebraic operators."
)


class DiscoptIntegralityError(TypeError):
    """A discopt model mixes integer/binary variables with a difflow block.

    Raised by :func:`check_no_integrality` at *build* time. discopt raises
    the equivalent ``ValueError`` at solve time; catching it early is the
    point, because by then the model is already written.
    """


def integer_variables(model) -> list[str]:
    """Names of the model's non-continuous variables.

    Args:
        model: A ``discopt.modeling.Model``.

    Returns:
        Names of every binary or general-integer variable, in model order.

    Raises:
        ImportError: If discopt is not installed.
    """
    dm = require("discopt.modeling")
    var_type = dm.VarType
    return [
        v.name
        for v in model._variables
        if getattr(v, "var_type", var_type.CONTINUOUS) is not var_type.CONTINUOUS
    ]


def check_no_integrality(model) -> None:
    """Refuse a discopt model that has integer or binary variables.

    Args:
        model: A ``discopt.modeling.Model`` about to receive a difflow
            implicit block.

    Raises:
        DiscoptIntegralityError: If any variable is binary or integer.
        ImportError: If discopt is not installed.

    Example:
        >>> m = dm.Model()                     # doctest: +SKIP
        >>> b = m.binary("y")                  # doctest: +SKIP
        >>> check_no_integrality(m)            # doctest: +SKIP
        Traceback (most recent call last):
        DiscoptIntegralityError: ...
    """
    bad = integer_variables(model)
    if bad:
        raise DiscoptIntegralityError(
            f"model has {len(bad)} integer/binary variable(s) "
            f"({', '.join(bad[:5])}{', ...' if len(bad) > 5 else ''}). "
            + CUSTOMCALL_RESTRICTION
        )


def as_implicit(
    model,
    unit_or_section,
    u_inputs: Sequence,
    *,
    view: ResidualView | None = None,
    x0=None,
    name: str = "difflow",
    allow_integrality: bool = False,
    **residual_kwargs: Any,
):
    """Add a difflow unit or section to a discopt model as an implicit block.

    The block's internal states ``v`` become an expression node you index
    (``node[i]``); the model's own expressions supply ``u``.

    Args:
        model: A ``discopt.modeling.Model``.
        unit_or_section: Anything :func:`~difflow.solvers.residual.as_residual`
            accepts, or ``None`` when ``view`` is given.
        u_inputs: The model expressions feeding the block, in the order the
            residual expects them. Their flattened values become ``u``, so
            there must be ``view.n_inputs`` scalars in total.
        view: A prebuilt :class:`~difflow.solvers.residual.ResidualView`, to
            avoid rebuilding it.
        x0: Starting guess for ``v``. Defaults to ``view.v0``, which is the
            flowsheet's own sequential-modular estimate -- much better than
            ``dm.implicit``'s default of zeros, where a difflow residual is
            usually singular.
        name: Display name for the node.
        allow_integrality: Skip the integrality check. Only useful to
            reproduce discopt's own solve-time error; the solve will still
            raise.
        **residual_kwargs: Forwarded to
            :func:`~difflow.solvers.residual.as_residual`.

    Returns:
        ``(node, view)`` -- the discopt expression and the residual view.

    Raises:
        DiscoptIntegralityError: If the model already has integer or binary
            variables. See :data:`CUSTOMCALL_RESTRICTION`.
        ValueError: If ``u_inputs`` does not carry ``view.n_inputs`` scalars.
        ImportError: If discopt is not installed.

    Example:
        >>> node, view = as_implicit(m, flowsheet, [T_feed_expr])  # doctest: +SKIP
        >>> m.minimize(node[0])                                    # doctest: +SKIP
    """
    dm = require("discopt.modeling")
    if not allow_integrality:
        check_no_integrality(model)

    if view is None:
        if unit_or_section is None:
            raise ValueError("pass unit_or_section or view")
        view = as_residual(unit_or_section, **residual_kwargs)

    n_u = 0
    for expr in u_inputs:
        shape = tuple(getattr(expr, "shape", ()) or ())
        n_u += 1 if not shape else int(_prod(shape))
    if n_u != view.n_inputs:
        raise ValueError(
            f"u_inputs flatten to {n_u} scalars but the residual expects "
            f"{view.n_inputs} ({view.u_names[:4]} ...). dm.implicit "
            "concatenates the inputs in argument order, so they must line up "
            "exactly."
        )

    node = dm.implicit(
        view,
        list(u_inputs),
        view.n_unknowns,
        x0=view.v0 if x0 is None else x0,
        name=name,
    )
    return node, view


def _prod(shape) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out
