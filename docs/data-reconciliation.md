# Data Reconciliation

## Overview

Plant measurements are noisy, and taken at face value they contradict the model: nominations do not close, balances do not balance, pressure drops do not match flows. **Data reconciliation** finds the smallest statistically weighted adjustment to the measurements that makes them satisfy the model equations. Because the model carries information the sensors do not, the reconciled estimates are *more* precise than the raw measurements — often by a factor of two or three.

`difflow.reconciliation` provides:

- **Constrained weighted least squares** against any differentiable residual function $F(x, \theta) = 0$ — a flowsheet, a gas network, or an equation set you write yourself.
- **Joint parameter estimation**: mark a variable unmeasured and it is *estimated* rather than reconciled, with a standard error, in the same solve.
- **Covariance of the estimates**, from the inverse KKT matrix, and the sensitivity $\partial \hat x/\partial y$ by automatic differentiation.
- **Gross error detection**: the global $\chi^2$ test, the measurement test, and serial elimination.
- **Observability and redundancy classification**, run *before* the solve so an ill-posed problem raises a named error instead of returning `NaN`.
- **Sensor placement**: what a proposed meter would buy, before anyone buys it.

Everything that makes this work — the constraint Jacobian, the covariance, the sensitivities — is derivative information. A conventional flowsheet package hand-derives it per unit operation; a differentiable one gets it from `jax.jacobian`.

The module is domain-agnostic. `difflow_gas.residuals` shows how a plugin supplies its equation set; see [`examples/28_data_reconciliation.ipynb`](../examples/28_data_reconciliation.ipynb) for the worked gas-network case.

---

## Mathematical Formulation

### The problem

$$\min_x \; (x - y)^T W (x - y) \quad \text{subject to} \quad F(x, \theta) = 0$$

where $y$ are the measurements, $W = \operatorname{diag}(1/\sigma_i^2)$, and $F$ are the model equations. Entries with $\sigma_i = \infty$ get $W_{ii} = 0$: they are **estimated**, not reconciled. A *finite* $\sigma$ on an unmetered variable acts as a Bayesian prior, which is the graceful way to handle a weakly identified parameter.

### The KKT system

The first-order conditions are

$$\begin{bmatrix} W & A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} \Delta x \\ \lambda \end{bmatrix} = \begin{bmatrix} -W(x - y) \\ -F(x) \end{bmatrix}, \qquad A = \frac{\partial F}{\partial x},$$

iterated to convergence. This is Gauss–Newton: it drops the $\sum_k \lambda_k \nabla^2 F_k$ term of the exact Newton Jacobian, which is where a pipe law's non-smooth $q|q|$ second derivative would enter. Exact gradients are recovered afterwards from one implicit-function-theorem correction that *does* use the full Jacobian, so `jax.grad` through `reconcile` is exact with respect to $y$, $\sigma$ and $\theta$.

### Solvability is observability

For $W \succeq 0$, the KKT matrix is nonsingular **iff** $A$ has full row rank and $Z^T W Z \succ 0$ on a basis $Z$ of $\ker A$. With $W$ diagonal and zero exactly on the unmeasured entries, the second condition collapses to

> the unmeasured columns $A_U$ must have full column rank

which is precisely the classical observability condition. `classify` runs this test first and raises `ReconciliationStructureError` naming the culprits.

Two consequences worth knowing:

- **Boundary flows must be state variables, not fixed parameters.** With them fixed, the node-balance block of $A$ is the incidence matrix, whose rank is only $n_{\text{nodes}} - 1$, and the KKT matrix is singular for purely structural reasons.
- **Ranks come from `svd(A)`, never from the eigenvalues of $A^T A$.** Squaring the matrix squares its condition number, and a structurally zero singular value reappears near $\sqrt{\varepsilon}\,\sigma_{\max}$ — an unobservable system reported as full rank.

### Covariance

$$\Sigma_{\hat x} = [K^{-1}]_{11}$$

For a fully measured problem this equals the textbook projection $\Sigma - \Sigma A^T (A\Sigma A^T)^{-1} A \Sigma$, but unlike it, it stays well defined when a variable is unmeasured — so the standard error of an estimated parameter comes from the same expression as the reconciled variance of a metered flow. The adjustment covariance is $\Sigma_{\text{adj}} = \Sigma - \Sigma_{\hat x}$.

`measurement_sensitivity` computes $S = \partial \hat x/\partial y$ by differentiating the solver. For **linear** constraints $S \Sigma S^T = \Sigma_{\hat x}$ exactly. For nonlinear ones the two differ by the curvature term the covariance formula drops: they agree to machine precision when the data are consistent (the multipliers vanish) and diverge in proportion to how inconsistent the data are. The classical formula is itself a linearization; the discrepancy is a useful diagnostic of how nonlinear the model is over the range the adjustments span.

### Scaling

The KKT matrix mixes $W$ (units of 1/variable²) with $A$ (units of residual/variable), so its conditioning depends on the unit system — the same gas network posed in Pa and Pa² rather than bar and bar² is many orders worse conditioned, past what float64 resolves. Variables are scaled by $d_i = \sigma_i$, which makes the scaled weight matrix a 0/1 mask (so $1/\sigma^2$ is never evaluated and an infinite $\sigma$ cannot produce a `NaN`), and residual rows are equilibrated to unit 2-norm. This is on by default; pass `scaling=False` only to observe what it prevents.

### Gross error detection

The **global test** statistic is the optimal objective itself, distributed $\chi^2$ on the degrees of redundancy $m - \operatorname{rank}(A_U)$. The often-quoted form $F(y)^T (A\Sigma A^T)^{-1} F(y)$ is a special case that cannot be evaluated at all when a variable is unmeasured, and carries the wrong degrees of freedom.

The **measurement test** standardizes each adjustment, $z_i = (\hat x_i - y_i)/\sqrt{\Sigma_{\text{adj},ii}}$, which is standard normal under the null hypothesis. A measurement nothing checks has $\Sigma_{\text{adj},ii} = 0$ and is reported as untestable rather than given a spurious $z$. Because that and the redundancy classification are read from the same $\Sigma_{\hat x}$, they cannot disagree.

Least squares smears a gross error across neighbouring measurements, so identification is reliable only where redundancy is high; `serial_elimination` discards the prime suspect and re-tests until the data are clean.

---

## API Reference

### `reconcile`

```python
def reconcile(
    residual_fn,                    # F(x, params) -> (m,) Array
    y: Array,
    sigma: Array,                   # inf entry => unmeasured
    *,
    params=None, names=None, x0=None,
    unmeasured_init=None, unmeasured_scale=None,
    scaling: Scaling | bool = True,
    max_steps: int = 20, tol: float = 1e-9,
    method: str = "gauss_newton",
    check_structure: bool = True, rank_tol=None,
) -> ReconcileResult
```

**Parameters:**
- `residual_fn` — the model equations, JAX-traceable.
- `y` — measurements. Entries with infinite `sigma` are ignored and may be `nan`.
- `sigma` — standard deviations; `inf` marks a variable to estimate.
- `params` — extra argument threaded to `residual_fn`; `jax.grad` with respect to it answers *how would the reconciled state move if this fixed parameter changed* — a different question from estimating it.
- `check_structure` — run the observability test first. Disable only inside a `jit`/`vmap` sweep whose structure you have already validated.

**Returns** a `ReconcileResult` with `x`, `x_named`, `adjustment`, `objective`, `covariance`, `std`, `converged`, `structure` and `scaling`, plus a `summary()` table.

### Other entry points

```python
classify(residual_fn, x, sigma, *, scaling, names=None) -> StructureReport
reconciled_covariance(residual_fn, x, sigma, *, scaling) -> Array
measurement_sensitivity(residual_fn, y, sigma, *, x0, scaling) -> Array

global_test(result, alpha=0.05) -> GlobalTestResult
measurement_test(result, alpha=0.05, bonferroni=True) -> MeasurementTestResult
serial_elimination(residual_fn, y, sigma, *, alpha=0.05, max_removed=3) -> list

sensor_value(residual_fn, x, sigma, *, target, candidate, candidate_sigma) -> dict
sensor_ranking(residual_fn, x, sigma, *, target, candidates, candidate_sigma) -> list
```

`StructureReport.classes` assigns every variable one of `measured-redundant`, `measured-just-determined`, `unmeasured-observable` or `unmeasured-unobservable`, and `summary()` prints the table.

---

## Worked Example

A three-stream splitter whose meters do not close:

```python
import jax.numpy as jnp
from difflow.reconciliation import reconcile, global_test, measurement_test

def balance(x, params=None):
    return jnp.array([x[0] - x[1] - x[2]])       # feed = top + bottom

y     = jnp.array([100.0, 62.0, 40.0])           # out by 2 units
sigma = jnp.array([2.0, 1.0, 1.0])               # the feed meter is worst

res = reconcile(balance, y, sigma, names=["feed", "top", "bottom"])
print(res.summary())
print(global_test(res))
```

The feed meter, being the least trusted, absorbs most of the adjustment; the reconciled standard deviations are all below the raw `sigma`. With one equation and three measurements the degree of redundancy is 1.

To *estimate* a quantity instead of reconciling it, give it an infinite sigma:

```python
sigma = jnp.array([2.0, 1.0, jnp.inf])           # the bottoms meter failed
res = reconcile(balance, y, sigma, names=["feed", "top", "bottom"])
res.x_named["bottom"]                            # inferred from the balance
res.std["bottom"]                                # and its standard error
```

Ask for one unknown too many and the problem is diagnosed rather than silently failing:

```python
sigma = jnp.array([2.0, jnp.inf, jnp.inf])
reconcile(balance, y, sigma, names=["feed", "top", "bottom"])
# ReconciliationStructureError: reconciliation problem is not solvable:
# 1 unmeasured variable(s) cannot be determined from the constraints.
# Unobservable: top, bottom. ...
```

---

## Gas Networks

`difflow_gas` supplies the equation set for transmission networks. `difflow_gas.residuals.network_residuals` is the single definition of that set — nodal balances, resistance laws, valve relations, **and** the compressor relation $p_{\text{to}} = r\, p_{\text{from}}$. [`difflow_gas.verify`](unit-operations-gas.md) is the reporting layer over it, unflattening the same residual vector into labelled dicts of floats.

`verify` reports every block except the compressor relation, which a sequential solve satisfies by construction — there is nothing to check. A reconciliation cannot drop it, and on the example network it turns out to be exactly what makes the loop observable.

```python
import jax
import difflow_gas as dg

layout = dg.gas_state_layout(net, efficiency_arcs=["p3"])   # estimate fouling
sigma  = dg.measurement_sigma(layout)                       # meter accuracies
y      = dg.perturb(x_true, sigma, jax.random.PRNGKey(0))   # simulated data

res = dg.reconcile_network(net, y, sigma, layout, ratios={"cs1": 1.2})
p_bar, q_kg_s, supply = dg.reconciled_values(res, layout)
dg.verify.residuals_from_values(p_bar, q_kg_s, net).ok      # True
```

Measured nominations are deliberately kept out of `GasNetwork`, which rejects supplies that do not sum to zero — real nominations do not, and making them close is what the reconciliation is for.

See [`examples/28_data_reconciliation.ipynb`](../examples/28_data_reconciliation.ipynb) for the full treatment: variance reduction, a biased flow meter found and eliminated, an unmeasured pipe-fouling factor estimated with its standard error, the observability boundary, and sensor placement.

## Reconciling data vs. updating the model

Both are the same optimisation — a variable with a finite `sigma` is a measurement you may move at a cost, one with `sigma = inf` is a parameter you may move for free — so the interesting question is not *how* to update a model but **when you are entitled to**. Letting a parameter float absorbs whatever is wrong, including a broken sensor, and hands back a confident wrong number.

The practical discipline is two clocks: reconcile routinely with parameters **fixed**, so the global test stays a genuine instrument-health monitor; re-estimate parameters only as a deliberate campaign. The trigger is the pair of tests read together — persistent rejection with the *same* sensor blamed every day is an instrument fault, while persistent rejection with a *wandering* suspect is model drift.

[`examples/29_model_updating.ipynb`](../examples/29_model_updating.ipynb) works this through on a pipe that fouls over a 45-day campaign, including the case where a free parameter manufactures a fouling estimate out of a biased flow meter.
