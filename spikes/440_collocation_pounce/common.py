"""Shared setup for the stage 2/3 scripts."""
import json
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from difflow.dynamic.collocation import Collocation
from difflow.solvers import differentiable_problem, pounce_problem
from difflow_refinery.reforming import CatalyticReformer, ReformerParams, lean_naphtha
from difflow_refinery.reforming.reactor import flows_of, integrate_bed, stream_of
from difflow_refinery.reforming import species as sp
import eo_bed as eb
import eo_reformer as er

OPTS = {"tol": 1e-9, "max_iter": 200, "print_level": 0, "hessian_approximation": "limited-memory"}


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def z_cold(eo, p):
    """One sequential-modular pass from ``initial_recycle``: beds by diffrax profile, separator by flash."""
    ref, feed = eo.reformer, eo.feed
    rp = ref.params.update(H2_HC=p[2]).with_wait(p[0] + 273.15)
    feedp = feed.scaled(p[1] / eo.base_rate)
    c = eo.conditions(p)
    R = flows_of(ref.initial_recycle(feedp, rp))
    F = c["feed_F"] + R
    Ys = []
    for k in range(eo.n_rx):
        F0 = jnp.sum(F)
        Yk = []
        for w in np.asarray(eo.mesh.t).ravel():
            Fo, To, co, _ = integrate_bed(F, c["T_in"][k], c["P_rx"], c["Ws"][k] * w, rp.kinetics, rtol=1e-8)
            Yk.append(jnp.concatenate([Fo / F0, jnp.atleast_1d(co), jnp.atleast_1d(To)]))
        Ys.append(jnp.stack(Yk).reshape(eo.mesh.n_elements, eo.mesh.K, eb.NV))
        Fo, To, _, _ = integrate_bed(F, c["T_in"][k], c["P_rx"], c["Ws"][k], rp.kinetics, rtol=1e-8)
        F = Fo
        T_last = To
    _, ops, _ = eo._ops(p)
    liq, vap, _ = ops["separator"](stream_of(F, T_last, c["P_rx"]))
    return eo.pack(Ys, R, flows_of(liq), flows_of(vap))


def solve_with(prob, z):
    t = time.time()
    x, info = prob.solve(x0=np.asarray(z))
    return jnp.asarray(x), info, time.time() - t
