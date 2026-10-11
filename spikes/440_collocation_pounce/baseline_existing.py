"""Baseline on origin/main in this venv: jit primal and jacfwd of the planning model (WAIT lever), at +5 C."""
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

from difflow_refinery.reforming import CatalyticReformer, ReformerParams, lean_naphtha
from difflow_refinery.reforming.planning import ReformerPlanningModel

KEYS = ["reformate.RON", "h2.net_mol_s"]
t = time.time()
model = ReformerPlanningModel(CatalyticReformer(ReformerParams()), lean_naphtha(), ["wait"], KEYS)
print(f"model build (eager base solve) {time.time()-t:.1f}s", flush=True)
u = jnp.array([505.0])


def timed(label, fn, reps=3):
    ts = []
    for _ in range(reps):
        t = time.time()
        out = jax.block_until_ready(fn(u))
        ts.append(time.time() - t)
    print(f"{label}: first {ts[0]:.1f}s later {[round(x, 1) for x in ts[1:]]}  rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB",
          flush=True)
    return out


y = timed("jit primal", jax.jit(model))
print("  y", np.asarray(y), flush=True)
J = timed("jit jacfwd", jax.jit(jax.jacfwd(model)), reps=3)
print("  J", np.asarray(J).ravel(), flush=True)
