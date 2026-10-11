"""Stage 3: derivatives of the optimal products w.r.t. the levers through pounce.jax.JaxProblem.

usage: stage3.py NE K grading
levers p = [WAIT (C), feed rate (kg/s), H2/HC]; outputs [reformate.RON, h2.net_mol_s].
"""
import traceback

from common import *  # noqa

NE, K, g = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
mesh = Collocation.make(NE, K, grading=g)
ref = CatalyticReformer(ReformerParams())
feed = lean_naphtha()
base = ref.solve(feed, tol=1e-11, max_iter=400)
eo = er.EOReformer(ref, feed, mesh)
z0 = eo.z_from_result(base)
f, gfun, bd = eo.bounds(z0, detect=True)
print(f"n=m={eo.n} jac nnz {len(bd.jac_pattern[0])} hess nnz {len(bd.hess_pattern[0])} [{eo.pattern_note}]", flush=True)
t = time.time()
jp = differentiable_problem(f, gfun, bd, options=OPTS, sparse=True)
print(f"differentiable_problem build {time.time()-t:.1f}s", flush=True)
KEYS = ("reformate.RON", "h2.net_mol_s")
p_base = eo.p0
p505 = eo.p0.at[0].add(5.0)
z0j = jnp.asarray(z0)


def fun(p):
    return eo.minimal_outputs(jp.solve(p, z0j), p, KEYS)


def timed(label, fn, *a, reps=3):
    ts, out = [], None
    for i in range(reps):
        t = time.time()
        out = fn(*a)
        jax.block_until_ready(out)
        ts.append(time.time() - t)
    print(f"  {label}: first {ts[0]:.2f}s, later best {min(ts[1:]):.2f}s spread {min(ts[1:]):.2f}-{max(ts[1:]):.2f}s"
          if reps > 1 else f"  {label}: {ts[0]:.2f}s", flush=True)
    return out


def attempt(label, fn, *a, reps=3):
    try:
        return timed(label, fn, *a, reps=reps)
    except Exception as exc:  # noqa
        msg = str(exc).strip().splitlines()
        print(f"  {label}: FAILS with {type(exc).__name__}: {' | '.join(msg[:3])[:400]}", flush=True)
        return None


print("== value", flush=True)
v = attempt("eager fun(p_base)", fun, p_base)
print("   ", v, flush=True)
jf = jax.jit(fun)
v = attempt("jit fun(p_base)", jf, p_base)
v5 = attempt("jit fun(p+5C)", jf, p505)
print("    base", np.asarray(v), "505", np.asarray(v5), flush=True)

print("== reverse mode", flush=True)
Jr = attempt("jacrev eager", jax.jacrev(fun), p_base)
jjr = jax.jit(jax.jacrev(fun))
Jr = attempt("jit(jacrev)", jjr, p_base)
print("   J (rows RON,H2; cols WAIT,feed,H2HC):\n", np.asarray(Jr), flush=True)
Jr5 = attempt("jit(jacrev) at +5C", jjr, p505)
print("   J at +5C:\n", np.asarray(Jr5), flush=True)
gr = attempt("jit(grad RON)", jax.jit(jax.grad(lambda p: fun(p)[0])), p_base)
print("   grad RON", np.asarray(gr), flush=True)

print("== forward mode / other transforms", flush=True)
attempt("jacfwd", jax.jacfwd(fun), p_base, reps=1)
attempt("jit(jacfwd)", jax.jit(jax.jacfwd(fun)), p_base, reps=1)
attempt("jvp", lambda p: jax.jvp(fun, (p,), (jnp.array([1.0, 0.0, 0.0]),)), p_base, reps=1)
pb = jnp.stack([p_base, p505])
attempt("vmap(fun)", jax.vmap(fun), pb, reps=1)
attempt("jit(vmap(fun))", jax.jit(jax.vmap(fun)), pb, reps=1)
attempt("vmap(jacrev(fun))", jax.vmap(jax.jacrev(fun)), pb, reps=1)
attempt("jit(vmap(jacrev))", jax.jit(jax.vmap(jax.jacrev(fun))), pb, reps=1)
attempt("hessian (2nd order)", jax.jit(jax.hessian(lambda p: fun(p)[0])), p_base, reps=1)
attempt("jp.vmap_solve", lambda pp: jp.vmap_solve(pp, z0j)[:, :3], pb, reps=1)

print("== alternative: solve_with_jacobian (KKT sensitivity, not a JAX transform)", flush=True)


def jac_via_kkt(p):
    x, _, J = jp.solve_with_jacobian(p, z0j)
    dz = lambda pp: eo.minimal_outputs(x, pp, KEYS)
    dx = lambda xx: eo.minimal_outputs(xx, p, KEYS)
    return jax.jacfwd(dx)(x) @ J + jax.jacfwd(dz)(p)


Jk = attempt("solve_with_jacobian + chain", jac_via_kkt, p_base)
print("   J via KKT sensitivity:\n", None if Jk is None else np.asarray(Jk), flush=True)

print("== central finite differences of the EO solve (h = 0.02 C, 1e-3 rel, 5e-3)", flush=True)
hs = jnp.array([0.02, 1e-3 * float(p_base[1]), 5e-3])
cols = []
for j in range(3):
    e = jnp.zeros(3).at[j].set(hs[j])
    cols.append((fun(p_base + e) - fun(p_base - e)) / (2 * hs[j]))
Jfd = jnp.stack(cols, axis=1)
print("   J_fd:\n", np.asarray(Jfd), flush=True)
if Jr is not None:
    print("   max |J_rev - J_fd| / |J_fd| per column:",
          np.asarray(jnp.max(jnp.abs(Jr - Jfd) / (jnp.abs(Jfd) + 1e-12), axis=0)), flush=True)

print("== existing implementation: central FD of CatalyticReformer.solve (same levers)", flush=True)


def existing(p, tear):
    prm = ReformerParams().update(H2_HC=float(p[2])).with_wait(float(p[0]) + 273.15)
    r = ref.solve(feed.scaled(float(p[1]) / eo.base_rate), prm, tear_initial=tear, tol=1e-11, max_iter=400)
    o = r.outputs()
    return np.array([float(o[k]) for k in KEYS])


for label, p in (("base 500C", p_base), ("+5C 505C", p505)):
    cols = []
    for j in range(3):
        e = np.zeros(3)
        e[j] = float(hs[j])
        cols.append((existing(np.asarray(p) + e, base.tear) - existing(np.asarray(p) - e, base.tear)) / (2 * float(hs[j])))
    print(f"   existing FD at {label}: WAIT col {np.array(cols)[0]}, feed col {np.array(cols)[1]}, H2HC col {np.array(cols)[2]}",
          flush=True)
print(f"peak RSS {rss_mb():.0f} MB", flush=True)
