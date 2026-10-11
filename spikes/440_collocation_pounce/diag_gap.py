"""Where does the 1.2e-3 RON gap between the EO solve and the existing solver come from?

Evaluate the EO residual blocks AT the existing solver's converged +5 C solution (beds sampled from diffrax at
rtol 1e-10), and re-run the existing solver with a tighter reactor rtol.
"""
from common import *  # noqa

mesh = Collocation.make(12, 3, grading=2.0)
ref = CatalyticReformer(ReformerParams())
feed = lean_naphtha()
base = ref.solve(feed, tol=1e-11, max_iter=400)
eo = er.EOReformer(ref, feed, mesh)
p505 = eo.p0.at[0].add(5.0)
r5 = CatalyticReformer(ReformerParams().with_wait(505.0 + 273.15)).solve(feed, tear_initial=base.tear, tol=1e-11, max_iter=400)
z = eo.z_from_result(r5, p505)
r = eo.residual(z, p505)
nb = eo.bed
print("EO residual blocks at the EXISTING +5C solution (beds from diffrax rtol 1e-10):")
for k in range(3):
    rk = r[k * nb:(k + 1) * nb].reshape(mesh.n_elements, mesh.K, eb.NV)
    print(f"  bed{k+1}: max|colloc resid| {float(jnp.max(jnp.abs(rk[..., :eb.NS+1]))):.2e}  enthalpy {float(jnp.max(jnp.abs(rk[..., eb.NS+1]))):.2e}")
o = 3 * nb
print(f"  flash block (EOSFlash fugacity equality + component balance): balance {float(jnp.max(jnp.abs(r[o:o+eb.NS]))):.2e}, "
      f"fugacity {float(jnp.max(jnp.abs(r[o+eb.NS:o+2*eb.NS]))):.2e}")
print(f"  recycle closure: {float(jnp.max(jnp.abs(r[o+2*eb.NS:]))):.2e}")
print(f"  (scales: flash fugacity rows are x*phi differences ~ mole fractions; max|Fl|={float(jnp.max(jnp.abs(eo.split(z)[2]))):.2f})")
# the existing solver with tighter reactor tolerance
for rtol in (1e-8, 1e-10, 1e-12):
    prm = ReformerParams(rtol=rtol).with_wait(505.0 + 273.15)
    t = time.time()
    rr = CatalyticReformer(prm).solve(feed, tear_initial=base.tear, tol=1e-11, max_iter=400)
    o = rr.outputs()
    print(f"existing at reactor rtol={rtol:g}: RON {float(o['reformate.RON']):.8f}  H2 {float(o['h2.net_mol_s']):.8f}  ({time.time()-t:.0f}s)", flush=True)
