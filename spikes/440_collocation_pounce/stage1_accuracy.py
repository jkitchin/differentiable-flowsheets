"""Stage 1: collocation bed vs diffrax bed, over WAIT 480-530 C, three beds."""
import pickle
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
kin = ReformingKinetics()
configs = [(int(a), int(b), float(c)) for a, b, c in
           (s.split(",") for s in sys.argv[2:])] if len(sys.argv) > 2 else [
    (6, 3, 2.0), (8, 3, 2.0), (12, 3, 2.0), (16, 3, 2.0), (8, 3, 3.0), (12, 3, 3.0)]

def hybrid(n1, n2, g, split=0.5, K=3):
    from difflow.dynamic.collocation import graded_edges
    e1 = graded_edges(n1, g, 0.0, split)
    e2 = np.linspace(split, 1.0, n2 + 1)[1:]
    return Collocation(K, tuple(float(e) for e in np.concatenate([e1, e2])))


for NE, K, g in configs:
    if g < 0:   # hybrid code: NE = 10*n1 + n2, g = -grading
        mesh = hybrid(NE // 10, NE % 10, -g, K=K)
        NE = mesh.n_elements
    else:
        mesh = Collocation.make(NE, K, grading=g)
    worst = dict(F=0.0, Fabs=0.0, T=0.0)
    niters = []
    newton = eb.make_newton(mesh, kin)
    for wait, bl in beds.items():
        for k, b in enumerate(bl):
            F, T, P, W = jnp.asarray(b["F"]), b["T"], b["P"], b["W"]
            Fr, Tr, cr, _ = integrate_bed(F, T, P, W, kin, rtol=1e-11)
            Y0 = eb.initial_guess(F, T, mesh)
            Y, it, nr = newton(Y0, F, T, P, W)
            Fo, To, co = eb.outlet(Y, F)
            sig = Fr > 1e-3 * jnp.sum(F)
            rel = float(jnp.max(jnp.abs(Fo - Fr)[sig] / Fr[sig]))
            ab = float(jnp.max(jnp.abs(Fo - Fr)) / jnp.sum(F))
            dT = float(abs(To - Tr))
            worst["F"] = max(worst["F"], rel)
            worst["Fabs"] = max(worst["Fabs"], ab)
            worst["T"] = max(worst["T"], dT)
            niters.append((it, nr))
            print(f"  NE={NE} K={K} g={g} WAIT={wait} bed{k+1}: relF={rel:.2e} absF/F0={ab:.2e} dT={dT:.2e} K "
                  f"coke_rel={abs(float(co - cr)) / float(cr):.1e} newton={it} |r|={nr:.1e}", flush=True)
    print(f"NE={NE} K={K} grading={g}: unknowns/bed={NE*K*eb.NV} worst relF={worst['F']:.2e} "
          f"worst absF/F0={worst['Fabs']:.2e} worst dT={worst['T']:.2e} K", flush=True)
