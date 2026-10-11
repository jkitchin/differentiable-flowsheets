"""Mesh study at the answer level: EO solve at WAIT+5 C vs the stored existing-solver numbers (no existing solve).

usage: stage2c.py spec [spec ...]   spec = K:n1:g:s:n2   (n1 geometric elements of ratio g over [0, s], then n2 uniform)
Reference (CatalyticReformer at WAIT=505 C, lean naphtha, tol 1e-11; reproduced in stage2b/diag_gap logs).
"""
from common import *  # noqa
from difflow.dynamic.collocation import graded_edges

REF = {"reformate.RON": 95.79937555, "h2.net_mol_s": 144.03648238, "rx1.dT": -69.64936631}

ref = CatalyticReformer(ReformerParams())
feed = lean_naphtha()
for spec in sys.argv[1:]:
    K, n1, g, s, n2 = spec.split(":")
    K, n1, g, s, n2 = int(K), int(n1), float(g), float(s), int(n2)
    e = np.concatenate([graded_edges(n1, g, 0.0, s), np.linspace(s, 1.0, n2 + 1)[1:]]) if n2 else graded_edges(n1, g)
    mesh = Collocation(K, tuple(float(x) for x in e))
    eo = er.EOReformer(ref, feed, mesh)
    p505 = eo.p0.at[0].add(5.0)
    zc = z_cold(eo, p505)
    t = time.time()
    f, gfun, bd = eo.bounds(zc, detect=True)
    prob = pounce_problem(lambda x: f(x, p505), lambda x: gfun(x, p505), bd, options=OPTS, sparse=True)
    x, info, dt0 = solve_with(prob, zc)
    x, info, dt = solve_with(prob, zc)
    out = eo.result(x, p505).outputs()
    print(f"{spec}: NE={mesh.n_elements} n={eo.n} setup+first {time.time()-t:.0f}s warm solve {dt:.1f}s iters={info['iter_count']} "
          f"RON err {float(out['reformate.RON']) - REF['reformate.RON']:+.2e}  H2 rel {float(out['h2.net_mol_s'])/REF['h2.net_mol_s']-1:+.2e}  "
          f"rx1.dT err {float(out['rx1.dT']) - REF['rx1.dT']:+.1e} K", flush=True)
