"""Stage 1: time one residual and one residual+Jacobian evaluation of a single bed (compile and warm).

usage: stage1_timing.py beds.pkl NE K grading
"""
import pickle
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from difflow.dynamic.collocation import Collocation
from difflow_refinery.reforming.kinetics import ReformingKinetics
from difflow_refinery.reforming.reactor import integrate_bed
import eo_bed as eb

beds = pickle.load(open(sys.argv[1], "rb"))
NE, K, g = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
mesh = Collocation.make(NE, K, grading=g)
kin = ReformingKinetics()
b = beds[500.0][0]
F, T, P, W = jnp.asarray(b["F"]), b["T"], b["P"], b["W"]
shape = (NE, K, eb.NV)
res = jax.jit(lambda y, F, T, P, W: eb.bed_residual(y.reshape(shape), F, T, P, W, kin, mesh).ravel())
jac = jax.jit(jax.jacfwd(lambda y, F, T, P, W: eb.bed_residual(y.reshape(shape), F, T, P, W, kin, mesh).ravel()))
Y0 = eb.initial_guess(F, T, mesh).ravel()


def timeit(label, fn):
    t = time.time()
    jax.block_until_ready(fn(Y0, F, T, P, W))
    first = time.time() - t
    ts = []
    for _ in range(5):
        t = time.time()
        jax.block_until_ready(fn(Y0, F, T, P, W))
        ts.append(time.time() - t)
    print(f"{label}: first call (compile+run) {first:.2f}s; warm best {min(ts)*1e3:.1f} ms, spread {min(ts)*1e3:.1f}-{max(ts)*1e3:.1f} ms",
          flush=True)


print(f"mesh NE={NE} K={K} grading={g}: {NE*K*eb.NV} unknowns", flush=True)
timeit("residual", res)
timeit("dense jacobian (jacfwd, 31 x NE x K tangents)", jac)
solver = eb.make_newton(mesh, kin)
t = time.time()
Y, it, nr = solver(Y0.reshape(shape), F, T, P, W)
print(f"damped Newton from flat start: {it} iterations, |r|={nr:.1e}, {time.time()-t:.1f}s (compiled)", flush=True)
# the diffrax bed for scale
t = time.time()
jax.block_until_ready(integrate_bed(F, T, P, W, kin))
print(f"diffrax integrate_bed first call {time.time()-t:.2f}s", flush=True)
ts = []
for _ in range(5):
    t = time.time()
    jax.block_until_ready(integrate_bed(F, T, P, W, kin))
    ts.append(time.time() - t)
print(f"diffrax integrate_bed warm best {min(ts)*1e3:.1f} ms spread {min(ts)*1e3:.1f}-{max(ts)*1e3:.1f} ms", flush=True)
jt = jax.jit(jax.jacfwd(lambda Fi: integrate_bed(Fi, T, P, W, kin)[0]))
t = time.time()
jax.block_until_ready(jt(F))
print(f"diffrax bed jacfwd wrt inlet flows (29 tangents) first {time.time()-t:.1f}s", flush=True)
ts = []
for _ in range(3):
    t = time.time()
    jax.block_until_ready(jt(F))
    ts.append(time.time() - t)
print(f"  warm best {min(ts)*1e3:.0f} ms spread {min(ts)*1e3:.0f}-{max(ts)*1e3:.0f} ms", flush=True)
print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
