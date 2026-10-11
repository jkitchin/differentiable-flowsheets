"""Stage 2 (final form): EO reformer solved by pounce at WAIT+5 C; accuracy and timing.

usage: stage2b.py NE K grading
"""
from common import *  # noqa

NE, K, g = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
mesh = Collocation.make(NE, K, grading=g)
ref = CatalyticReformer(ReformerParams())
feed = lean_naphtha()
t = time.time()
base = ref.solve(feed, tol=1e-11, max_iter=400)
bo = {k: float(v) for k, v in base.outputs().items()}
print(f"existing base solve {time.time()-t:.1f}s iters={base.iterations}; base RON {bo['reformate.RON']:.8f} "
      f"H2 {bo['h2.net_mol_s']:.8f}", flush=True)
t = time.time()
r5 = CatalyticReformer(ReformerParams().with_wait(505.0 + 273.15)).solve(feed, tear_initial=base.tear, tol=1e-11,
                                                                         max_iter=400)
t_exist = time.time() - t
o5 = {k: float(v) for k, v in r5.outputs().items()}
print(f"existing +5C from stored tear {t_exist:.1f}s iters={r5.iterations}; RON {o5['reformate.RON']:.8f} "
      f"H2 {o5['h2.net_mol_s']:.8f}", flush=True)
t = time.time()
r5c = CatalyticReformer(ReformerParams().with_wait(505.0 + 273.15)).solve(feed, tol=1e-11, max_iter=400)
print(f"existing +5C cold (initial_recycle) {time.time()-t:.1f}s iters={r5c.iterations} conv={r5c.converged}", flush=True)

eo = er.EOReformer(ref, feed, mesh)
p_hot = eo.p0.at[0].add(5.0)
z0 = eo.z_from_result(base)
t = time.time()
f, gfun, bd = eo.bounds(z0, detect=True)
print(f"n=m={eo.n}; patterns [{eo.pattern_note}] {time.time()-t:.1f}s: jac nnz {len(bd.jac_pattern[0])}, "
      f"hess nnz {len(bd.hess_pattern[0])}", flush=True)
prob = pounce_problem(lambda x: f(x, p_hot), lambda x: gfun(x, p_hot), bd, options=OPTS, sparse=True)

for label, start in (("warm (base solution)", z0), ("cold (initial_recycle, one sequential pass)", z_cold(eo, p_hot))):
    for rep in range(3):
        x, info, dt = solve_with(prob, start)
        print(f"{label} rep{rep}: {dt:.2f}s status={info['status']} iters={info['iter_count']} "
              f"max|g|={float(jnp.max(jnp.abs(gfun(x, p_hot)))):.2e}", flush=True)
    out = eo.result(x, p_hot).outputs()
    print(f"  {label}: RON {float(out['reformate.RON']):.8f} (existing {o5['reformate.RON']:.8f}, "
          f"abs {abs(float(out['reformate.RON']) - o5['reformate.RON']):.2e}); "
          f"H2 {float(out['h2.net_mol_s']):.8f} (existing {o5['h2.net_mol_s']:.8f}, "
          f"rel {abs(float(out['h2.net_mol_s']) / o5['h2.net_mol_s'] - 1):.2e})", flush=True)
worst = max((abs(float(out[k]) - o5[k]) / abs(o5[k]), k) for k in o5 if abs(o5[k]) > 1e-9)
print("worst relative difference over all outputs:", worst)
rel = {k: abs(float(out[k]) - o5[k]) / abs(o5[k]) for k in o5 if abs(o5[k]) > 1e-9}
print("outputs with rel diff > 1e-4:", {k: f"{v:.1e}" for k, v in rel.items() if v > 1e-4})
bal = eo.result(x, p_hot).balances()
print("balances:", {k: f"{float(v):.1e}" for k, v in bal.items()})
print(f"peak RSS {rss_mb():.0f} MB")
