"""Stage 3 (rest): finite-difference references, then the KKT-sensitivity route last (it killed stage3.py once).

usage: stage3b.py NE K grading [kkt]
"""
from common import *  # noqa

NE, K, g = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
mesh = Collocation.make(NE, K, grading=g)
ref = CatalyticReformer(ReformerParams())
feed = lean_naphtha()
base = ref.solve(feed, tol=1e-11, max_iter=400)
eo = er.EOReformer(ref, feed, mesh)
z0 = eo.z_from_result(base)
f, gfun, bd = eo.bounds(z0, detect=True)
KEYS = ("reformate.RON", "h2.net_mol_s")
p_base = eo.p0
p505 = eo.p0.at[0].add(5.0)
hs = np.array([0.02, 1e-3 * float(p_base[1]), 5e-3])


def existing(p, tear):
    prm = ReformerParams().update(H2_HC=float(p[2])).with_wait(float(p[0]) + 273.15)
    r = ref.solve(feed.scaled(float(p[1]) / eo.base_rate), prm, tear_initial=tear, tol=1e-11, max_iter=400)
    o = r.outputs()
    return np.array([float(o[k]) for k in KEYS])


print("== existing implementation: central FD of CatalyticReformer.solve", flush=True)
for label, p in (("500C", p_base), ("505C", p505)):
    cols = []
    for j in range(3):
        e = np.zeros(3)
        e[j] = hs[j]
        cols.append((existing(np.asarray(p) + e, base.tear) - existing(np.asarray(p) - e, base.tear)) / (2 * hs[j]))
    C = np.array(cols).T
    print(f"existing FD at {label}: [WAIT, feed, H2HC] columns\n{C}", flush=True)

jp = differentiable_problem(f, gfun, bd, options=OPTS, sparse=True)
print("== central FD of the EO solve (pounce_problem re-solves at shifted p)", flush=True)


def eo_solve(p):
    p = jnp.asarray(p)
    return np.asarray(eo.minimal_outputs(jp.solve(p, jnp.asarray(z0)), p, KEYS))


for label, p in (("500C", p_base), ("505C", p505)):
    cols = []
    for j in range(3):
        e = np.zeros(3)
        e[j] = hs[j]
        cols.append((eo_solve(np.asarray(p) + e) - eo_solve(np.asarray(p) - e)) / (2 * hs[j]))
    print(f"EO FD at {label}:\n{np.array(cols).T}", flush=True)
print(f"peak RSS {rss_mb():.0f} MB", flush=True)

if len(sys.argv) > 4:
    t = time.time()
    x, duals, J = jp.solve_with_jacobian(p_base, jnp.asarray(z0), wrt_cols=[0])
    print(f"solve_with_jacobian wrt WAIT: {time.time()-t:.1f}s, J shape {J.shape}", flush=True)
    dxdp = J[:, 0]
    jac_x = jax.jacfwd(lambda xx: eo.minimal_outputs(xx, p_base, KEYS))(x)
    print("dout/dWAIT via KKT sensitivity:", np.asarray(jac_x @ dxdp + jax.jacfwd(lambda pp: eo.minimal_outputs(x, pp, KEYS))(p_base)[:, 0]), flush=True)
    print(f"peak RSS {rss_mb():.0f} MB", flush=True)
