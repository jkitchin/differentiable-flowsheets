# Feedback Control: PI(D) and Linear MPC

`difflow.control` closes loops around difflow models, at the two levels a plant
runs them:

- **`PIDController`** is a dynamic unit. It sits in a `DynamicFlowsheet`, measures
  a stream and drives a port, with output limits and back-calculation
  anti-windup. The closed loop is one JAX function of the gains, so `jax.grad`
  of an integrated squared error tunes it.
- **`LinearMPC`** is a receding-horizon controller on a linear(ized)
  discrete-time model, a `difflow.mhe.StateSpaceModel` linearized by autodiff or
  matrices you give, with bounds on the inputs. `run_mpc` runs it in closed loop
  against any JAX plant.

Both are JAX throughout: no Python control flow on traced values, no SciPy.

## Table of Contents

1. [A PI loop on a jacketed reactor](#a-pi-loop-on-a-jacketed-reactor)
2. [PIDController](#pidcontroller)
3. [Tuning with jax.grad](#tuning-with-jaxgrad)
4. [Limits and anti-windup](#limits-and-anti-windup)
5. [LinearMPC](#linearmpc)
6. [From a flowsheet to a state-space model](#from-a-flowsheet-to-a-state-space-model)
7. [Limits of what is here](#limits-of-what-is-here)
8. [API summary](#api-summary)

---

## A PI loop on a jacketed reactor

The controller's output travels as a **signal**: a stream with no flows whose
temperature `T` carries the output (in whatever units it has). A
`DynamicCSTR` in `mode="jacket"` takes its coolant temperature from an inlet
port when `Tc` names one, so the wiring is two `add_unit` calls:

```python
import jax
import jax.numpy as jnp
from difflow import make_stream
from difflow.dynamic import DynamicCSTR, DynamicFlowsheet, PIDController

jax.config.update("jax_enable_x64", True)

def rate(C, T, p):
    return jnp.array([1e-3 * jnp.exp(-4000.0 * (1 / T - 1 / 340.0)) * C["A"]])

fs = DynamicFlowsheet(["A", "B"])
fs.add_feed("feed", make_stream({"A": 0.5, "B": 9.5}, T=330.0, P=2e5))
fs.add_unit(DynamicCSTR(volume=1.0, rate_fn=rate, stoich=jnp.array([[-1.0], [1.0]]),
                        species_order=["A", "B"], dH_rxn=jnp.array([-60e3]),
                        mode="jacket", UA=2e3, Tc="coolant", name="reactor"),
            inlets={"inlet": "feed", "coolant": "Tc"}, outlets={"outlet": "product"})
fs.add_unit(PIDController(Kc=5.0, tau_I=200.0, setpoint=340.0, bias=330.0,
                          u_min=300.0, u_max=345.0, name="TC"),
            inlets={"measurement": "product"}, outlets={"output": "Tc"})

y0 = fs.initial_state().at[2].set(345.0)          # start 5 K hot
res = fs.simulate((0.0, 600.0), y0=y0, method="RK4", n_steps=60)
T = res.trajectory.y[:, 2]                        # reactor.T
Tc = res.stream_history("Tc")["T"]                # the controller output over time
```

The reactor's outlet reads its state and its process feed, not the coolant port
(`DynamicCSTR` declares this through `output_dependencies`), so the flowsheet
evaluates the reactor outlet, then the controller, then the reactor's
derivatives, all in one right-hand-side call: there is no algebraic loop to
break. `duty="heat"` (in `"specified_duty"` mode) and `UA="ua"` take a signal
the same way. Anything else reads the signal as `inputs[port]["T"]` in an
`InstantaneousUnit` (a valve that turns it into a feed flow, say).

`DynamicFlowsheetResult.stream_history(name, params, args)` re-evaluates a
stream at every saved time (one `vmap`); pass the `params` and `args` the
simulation used.

## PIDController

The control law is the parallel ("ISA") form with derivative action on the
filtered measurement and back-calculation anti-windup (Åström and Hägglund,
*Advanced PID Control*, ISA 2006, §3.5):

$$e = s\,(r - y),\qquad v = b + K_c e + I + D,\qquad u = \mathrm{clip}(v, u_\mathrm{min}, u_\mathrm{max})$$

$$\frac{dI}{dt} = \frac{K_c}{\tau_I}\,e + \frac{u - v}{\tau_T},\qquad
D = -s K_c \tau_D \frac{dy_f}{dt},\qquad \frac{dy_f}{dt} = \frac{y - y_f}{\alpha\,\tau_D}$$

| Argument | Meaning |
|----------|---------|
| `Kc` | gain, output units per measured unit |
| `tau_I` | integral time (s); `None` for no integral state |
| `tau_D` | derivative time (s); `None` for no derivative state |
| `alpha` | derivative filter time as a fraction of `tau_D` (default 0.1) |
| `setpoint` | `r`: a constant, `f(t)`, `f(t, args)` or an inlet port (cascade) |
| `bias` | `b`, the output at zero error and zero integral |
| `u_min`, `u_max` | output limits (`None`: unbounded) |
| `tau_T` | tracking time constant; default `sqrt(tau_I*tau_D)` with D, else `tau_I`; `inf` disables anti-windup |
| `action` | `"reverse"` (`e = r - y`, the output falls as the measurement rises) or `"direct"` (`e = y - r`) |
| `measure` | key of the measured stream (`"T"`, `"P"`, `"F_A"`) or `f(stream) -> value` |
| `I0` | initial integral term (output units) |

States: `I` (the integral term, in output units, so a gain change is bumpless)
and, with derivative action, `y_f`, which starts at the measurement so there is
no derivative kick at `t = 0`. The measured stream comes in on port
`"measurement"` (or the only inlet that is not a setpoint port); the output
leaves on `"output"`.

Every tuning value may be a constant, `f(t)` or `f(t, args)`, and lives under
its own key in the unit's `params`. `fs.simulate(params=...)` reaches every unit
of the flowsheet, so two controllers in one flowsheet take distinct traced gains
through `args` (`Kc=lambda t, a: a["Kc_level"]`), not through `params`.

## Tuning with jax.grad

Because the gains are read per call from `params`, the closed-loop trajectory is
a differentiable function of them:

```python
def ise(log_gains):
    Kc, tau_I = jnp.exp(log_gains)
    r = fs.simulate((0.0, 600.0), y0=y0, method="RK4", n_steps=60,
                    params={"Kc": Kc, "tau_I": tau_I})
    return jnp.sum((r.trajectory.y[:, 2] - 340.0) ** 2)

value, grad = jax.jit(jax.value_and_grad(ise))(jnp.log(jnp.array([5.0, 200.0])))
```

Hand `value_and_grad` to `scipy.optimize.minimize(jac=True)` or any optimizer;
optimize logarithms to keep the gains positive, and add a penalty on output
moves (`jnp.diff` of `stream_history("Tc", params)["T"]`) so the optimum is not
a controller that slams the valve.

## Limits and anti-windup

While the output is inside its limits `u = v` and the tracking term vanishes.
While it is pinned, a plain clipped PI keeps integrating the error: the integral
**winds up**, and the loop overshoots and takes long to unwind once the error
changes sign. The back-calculation term `(u - v)/tau_T` pulls `I` back toward
the value the limit implies, with time constant `tau_T`. A short `tau_T` behaves
like conditional integration (stop integrating while saturated); a long one lets
some useful integral build; `tau_T=float("inf")` is the clipped PI that winds up.
The clip is `jnp.minimum`/`jnp.maximum`, so gradients are exact (zero with
respect to the unclipped output while saturated).

## LinearMPC

At every sample the controller predicts `horizon` steps with a linear model in
deviations from an operating point,

$$\delta x_{k+1} = A\,\delta x_k + B\,\delta u_k + c_0,\qquad y = y_\mathrm{op} + C\,\delta x,$$

and chooses the input sequence that minimizes

$$\sum_{i=1}^{N} \lVert y_i + b - r_i\rVert^2_{Q} + \sum_{k=0}^{N-1} \lVert u_k - u_{k-1}\rVert^2_{R} + \lVert u_k - u_\mathrm{op}\rVert^2_{R_u}
\quad\text{s.t.}\quad u_\mathrm{min} \le u_k \le u_\mathrm{max},$$

applies the first input and repeats. `b`, the measured output minus the model's,
is a constant output disturbance: it makes the loop offset-free on a plant the
linear model does not match. The problem is a box-constrained QP in the
`horizon * n_u` inputs; `box_qp` solves it exactly (a primal-dual active-set
method, one dense solve per iteration, with a projected-gradient polish if the
active set has not settled) under `lax.fori_loop`, so the whole closed loop is a
`lax.scan`.

```python
from difflow.control import LinearMPC, run_mpc
from difflow.mhe import StateSpaceModel

model = StateSpaceModel.from_ode(                      # a first-order lag, sampled at 1 s
    lambda t, x, u, th: (2.0 * u - x) / 10.0, lambda x, u, th: x,
    n_x=1, n_y=1, n_u=1, dt=1.0)
mpc = LinearMPC.from_model(model, x_op=jnp.zeros(1), u_op=jnp.zeros(1), horizon=20,
                           q=1.0, r_move=0.1, u_min=-1.0, u_max=1.0)
res = run_mpc(mpc, model, x0=jnp.zeros(1), n_steps=60, y_ref=1.5)
res.u.max()        # 1.0: the bound binds on the way up
```

`LinearMPC.from_model` takes `A`, `B`, `C` as Jacobians of the model's step and
measurement at `(x_op, u_op)` (process noise at zero) and `c0 = f(x_op, u_op) -
x_op`, zero at a steady state. Matrices can also be given directly:
`LinearMPC(A, B, C, horizon, x_op=..., u_op=..., y_op=...)`.

`run_mpc(mpc, plant, x0, n_steps, y_ref)` takes the plant as `plant(x, u) ->
x_next` (any JAX function, e.g. a `difflow.dynamic.integrate` over one sample of
the nonlinear model) or a `StateSpaceModel`. `measure(x) -> y` defaults to the
model's `observe`, or `y_op + C (x - x_op)`. `y_ref` is one value for every
sample or one per sample. It returns `MPCResult(x, y, u, kkt)`: the plant states
and measurements (`n_steps + 1` rows), the applied inputs (`n_steps` rows) and
the QP's KKT residual at each sample. `mpc.solve(xm, u_prev, y_ref, bias)` is one
sample, for a loop of your own (with a state estimate from `difflow.mhe`, say).

## From a flowsheet to a state-space model

`StateSpaceModel.from_flowsheet(fs, h, n_y=..., dt=..., n_u=...)` makes a
`DynamicFlowsheet` the discrete-time model both `difflow.mhe` and `LinearMPC`
read. Its state is the combined state; the input `u` reaches the flowsheet as
its `args` (map it with `u_to_args=`), and `theta` as its `params`:

```python
fs2 = DynamicFlowsheet(["A", "B"])
fs2.add_feed("feed", make_stream({"A": 0.5, "B": 9.5}, T=330.0, P=2e5))
fs2.add_unit(DynamicCSTR(volume=1.0, rate_fn=rate, stoich=jnp.array([[-1.0], [1.0]]),
                         species_order=["A", "B"], dH_rxn=jnp.array([-60e3]),
                         mode="jacket", UA=2e3, Tc=lambda t, args: args[0], name="reactor"),
             inlets={"inlet": "feed"}, outlets={"outlet": "product"})
plant = StateSpaceModel.from_flowsheet(fs2, lambda x, u, th: x[2:3], n_y=1, n_u=1, dt=10.0)
x_op = fs2.steady_state(args=jnp.array([330.0]))
mpc = LinearMPC.from_model(plant, x_op, jnp.array([330.0]), horizon=20,
                           r_move=0.05, u_min=300.0, u_max=345.0)
res = run_mpc(mpc, plant, x_op, 60, y_ref=340.0)
```

## Limits of what is here

- **Continuous-time PID.** The controller integrates with the flowsheet; there is
  no sample-and-hold or dead time. A digital controller with a long sampling
  period is a discrete loop of your own.
- **Input bounds only in the MPC.** Output constraints and move-rate limits make
  the QP general rather than a box and are not offered; penalize them through
  `q` and `r_move` instead.
- **One linearization.** `LinearMPC` is linear in deviations around one
  operating point; re-linearize (`from_model` at a new point) for a nonlinear
  plant that moves far.
- **No estimator inside `run_mpc`.** The model state is propagated open loop from
  the applied inputs and corrected only through the output bias. With a state
  estimate, call `mpc.solve` yourself.
- The measurement map's dependence on `u` is ignored in the prediction.
- `simulate(params=...)` reaches every unit; distinct gains for two controllers
  go through `args`.

## API summary

```python
from difflow.control import (
    PIDController,          # also difflow.dynamic.PIDController
    LinearMPC,              # .from_model, .solve, .prediction_matrices
    run_mpc, MPCResult,
    box_qp,                 # min 0.5 u'Hu + g'u, lb <= u <= ub
)
from difflow.mhe import StateSpaceModel   # .from_flowsheet
# DynamicFlowsheetResult.stream_history(name, params=None, args=None)
```

Tests: `tests/test_control.py`.
