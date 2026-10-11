"""Stage 3 (clean process, no existing-solver solve): flat-profile start robustness, jit(jacrev) compile/warm/RSS,
and the factor_reuse=False variants.

usage: stage3c.py NE K grading
"""
from common import *  # noqa

NE, K, g = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
mesh = Collocation.make(NE, K, grading=g)
ref = CatalyticReformer(ReformerParams())
feed = lean_naphtha()
eo = er.EOReformer(ref, feed, mesh)
p505 = eo.p0.at[0].add(5.0)
t = time.time()
zc = z_cold(eo, eo.p0)     # one sequential diffrax pass from initial_recycle (no flowsheet solve)
print(f"cold start vector {time.time()-t:.1f}s, rss {rss_mb():.0f} MB", flush=True)
f, gfun, bd = eo.bounds(zc, detect=True)
KEYS = ("reformate.RON", "h2.net_mol_s")

# a FLAT profile start: every collocation point of a bed at that bed's inlet state
Ys, R, Fl, Fv = eo.split(zc)
c = eo.conditions(eo.p0)
F_in = c["feed_F"] + R
Yf = []
for k in range(eo.n_rx):
    z = jnp.concatenate([F_in / jnp.sum(F_in), jnp.zeros(1), jnp.atleast_1d(c["T_in"][k])])
    Yf.append(jnp.broadcast_to(z, Ys[k].shape))
    F_in = Ys[k][-1, -1, :eb.NS] * jnp.sum(F_in)   # next bed's inlet from the sequential pass outlet
zflat = eo.pack(Yf, R, Fl, Fv)
print("max|r| at flat start", float(jnp.max(jnp.abs(eo.residual(zflat, eo.p0)))), flush=True)
prob = pounce_problem(lambda x: f(x, eo.p0), lambda x: gfun(x, eo.p0), bd, options=OPTS, sparse=True)
for label, z in (("sequential-pass profile", zc), ("flat profile (every point at the bed inlet)", zflat)):
    if label.startswith("flat"):
        prob.add_option("max_iter", 60)            # a flat start must not run unbounded
        prob.add_option("max_cpu_time", 120.0)
    for rep in range(1 if label.startswith("flat") else 2):
        x, info, dt = solve_with(prob, z)
        print(f"{label}: {dt:.2f}s status={info['status']} ({info.get('status_msg')}) iters={info['iter_count']} "
              f"max|g|={float(jnp.max(jnp.abs(gfun(x, eo.p0)))):.1e}", flush=True)
print(f"rss {rss_mb():.0f} MB", flush=True)

for reuse in (True, False):
    t = time.time()
    jp = differentiable_problem(f, gfun, bd, options=OPTS, sparse=True, factor_reuse=reuse)
    print(f"== factor_reuse={reuse}: build {time.time()-t:.1f}s", flush=True)
    zj = jnp.asarray(zc)
    fun = lambda p: eo.minimal_outputs(jp.solve(p, zj), p, KEYS)
    jj = jax.jit(jax.jacrev(fun))
    for label, p in (("jit(jacrev) at 500C", eo.p0), ("jit(jacrev) at 505C", p505), ("jit(jacrev) at 505C again", p505),
                     ("jit(jacrev) at 505C again", p505)):
        t = time.time()
        J = jax.block_until_ready(jj(p))
        print(f"  {label}: {time.time()-t:.2f}s  WAIT column {np.asarray(J)[:, 0]}  rss {rss_mb():.0f} MB", flush=True)
    for label, fn in (("jit(grad RON)", jax.jit(jax.grad(lambda p: fun(p)[0]))),
                      ("jit(hessian RON)", jax.jit(jax.hessian(lambda p: fun(p)[0]))),
                      ("jacfwd", jax.jacfwd(fun))):
        try:
            t = time.time()
            out = jax.block_until_ready(fn(p505))
            print(f"  {label}: ok {time.time()-t:.1f}s", flush=True)
        except Exception as exc:  # noqa
            print(f"  {label}: FAILS {type(exc).__name__}: {str(exc).splitlines()[0][:200]}", flush=True)
    del jp
print(f"peak RSS {rss_mb():.0f} MB", flush=True)
