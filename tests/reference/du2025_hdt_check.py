"""Du et al. (2025): the rigorous hydrotreater against Upd0.m22 / Upd1.m26 (#533).

Run by hand, one unit per call (each compiles for several minutes)::

    PYTHONPATH=src:. python tests/reference/du2025_hdt_check.py <data root> Upd1

``<data root>`` holds ``du/case1`` (the case files and a JSON export of the
solution). The script:

1. finds the crude bulk SG and sulfur whose 230-370 C straight-run cut has
   the benchmark's base feed SG and sulfur (``B``);
2. calibrates inlet T, the cracking-leak constant and the purge fraction to
   the benchmark's gas / naphtha / diesel yields (oil basis);
3. takes the AD Jacobian of the yields with respect to feed SG and sulfur
   (chain rule through the crude's bulk SG and sulfur) and prints it next to
   the published ``delta / Del``.

Results, recorded in the docs: the calibration does not reach the benchmark
(at every knob's bound the unit makes 0.4 % naphtha against 6.4 %), and the
sensitivities are reported as they come out.
"""
import time, sys, json, dataclasses, numpy as np, jax, jax.numpy as jnp, warnings
jax.config.update("jax_enable_x64", True); warnings.simplefilter('ignore')
import difflow_refinery as dr
from difflow_refinery.hydrotreating import Hydrotreater, HydrotreaterParams
from difflow_refinery.hydrotreating.feed import straight_run_cut, select_cuts
from difflow_refinery.hydroprocessing.reactor import ReactorOptions
from difflow_refinery.benchmarks.du2025 import load_case
from difflow_refinery.benchmarks.du2025.calibration import calibrate, unit_base, compare_delta
S=sys.argv[1]; which=sys.argv[2]  # noqa: N806
T0=time.time(); log=lambda *a: print(f"[{time.time()-T0:7.1f}s]",*a, flush=True)
D=load_case(S+'/du/case1')
u,m={'Upd0':('Upd0','m22'),'Upd1':('Upd1','m26')}[which]
b=unit_base(D,u,m)
TBP_PCT=[5,10,20,30,40,50,60,70,80,90,95]; TBP_A=[60,95,150,205,260,315,370,430,500,600,680]
LE={"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5}
def assay(sg, s): return dr.Assay(TBP_PCT, jnp.asarray([t+273.15 for t in TBP_A]), sg=sg, light_ends=LE, heavy_end=dr.HeavyEnd(), sulfur_wt=s, nitrogen_wppm=1500.0)
a0=assay(0.86,1.8); cps=dr.default_cut_points(a0) if hasattr(dr,'default_cut_points') else None
from difflow_refinery.assay import default_cut_points
cps=tuple(float(c) for c in default_cut_points(a0))
char0=dr.characterize(a0, cut_points=cps, composition=True)
cuts=select_cuts(char0, 230+273.15, 370+273.15)
def feed_of(c): 
    ch=dr.characterize(assay(c[0],c[1]), cut_points=cps, composition=True)
    return ch, straight_run_cut(ch, rate=50.0, cuts=cuts)
def feed_props(ch, f):
    idx=[ch.names.index(n) for n in cuts]; mw=jnp.asarray(ch.component_MW)[jnp.asarray(idx)]
    mol=jnp.asarray([f[f"F_{n}"] for n in cuts]); mass=mol*mw
    sg=jnp.asarray(ch.component_SG)[jnp.asarray(idx)]; S_=jnp.asarray(ch.sulfur)[jnp.asarray(idx)]
    return jnp.stack([mass.sum()/(mass/sg).sum(), 100*(mass*S_).sum()/mass.sum()])
# 1) crude (sg, S) giving the base feed (SG, S) = B
target_p=jnp.asarray([b.props['Q168'], b.props['Q0']])
c=jnp.asarray([0.86,1.8])
for it in range(8):
    fp=lambda c: feed_props(*feed_of(c))
    r=fp(c)-target_p; J=jax.jacfwd(fp)(c); c=c-jnp.linalg.solve(J,r)
log('crude for base feed', np.round(np.asarray(c),4), 'feed props', np.round(np.asarray(fp(c)),4), 'target', np.asarray(target_p))
chB, fB = feed_of(c)
p0=HydrotreaterParams(reactor=ReactorOptions(adjoint="forward"))
unit=Hydrotreater(chB, fB, p0)
r=unit.solve(fB); log('base solve', bool(r.converged))
o=r.outputs; log({k:round(float(o[k]),5) for k in ['product.yield','naphtha.yield','gas.yield','h2.chemical_wt','yields.total','feed.S_wppm','product.S_wppm','hds.conversion']})
main=b.inlets[int(np.argmax(b.gamma_in))]; gi=b.gamma_in.sum()/b.gamma_in.max()
y_oil=dict(zip(b.outlets, b.yields*gi))   # outlet per unit main feed
lumps={'Upd0':[('gas',['s140','s227']),('diesel',['s91'])], 'Upd1':[('gas',['s141','s228']),('naphtha',['s102']),('diesel',['s101'])]}[u]
target=np.array([sum(y_oil[s] for s in ss) for _,ss in lumps]); log('targets (on oil)', dict(zip([l for l,_ in lumps], np.round(target,4))))
def yields(o):
    gas=1.0+o['h2.chemical_wt']/100.0-o['product.yield']-o['naphtha.yield']
    if u=='Upd0': return jnp.stack([gas, o['product.yield']+o['naphtha.yield']])
    return jnp.stack([gas, o['naphtha.yield'], o['product.yield']])
knobs=('T_in','crack_k','purge'); lo=[590.0,3e-11,0.01]; hi=[660.0,3e-8,0.3]; th0=[613.15,3e-10,0.05]
def params_of(th, base=p0):
    kin=dataclasses.replace(base.kinetics, crack_k=th[1])
    return dataclasses.replace(base, T_in=(th[0],), purge=th[2], kinetics=kin)
def predict(th, ch=chB, f=fB): return yields(unit.solve(f, char=ch, params=params_of(th), warn=False).outputs)
log('predict at start', np.round(np.asarray(predict(jnp.asarray(th0))),5))
cal=calibrate(predict, th0, target, lower=lo, upper=hi, max_steps=15)
log('calibrated', dict(zip(knobs, cal.theta)), 'pred', np.round(cal.predicted,5), 'targ', np.round(cal.target,5), 'err', cal.max_abs_error, cal.within_tol)
# 2) AD delta: d yields(total feed) / d feed (SG, S) via crude (sg, S)
thc=jnp.asarray(cal.theta)
def y_of_c(cc):
    ch,f=feed_of(cc); return predict(thc, ch, f)
Jy=np.asarray(jax.jacfwd(y_of_c)(c)); Jp=np.asarray(jax.jacfwd(fp)(c))
dYdq=Jy@np.linalg.inv(Jp)      # per (SG, S wt%), on oil
log('dY/dq (oil basis)', np.round(dYdq,5))
# published, per lump, oil basis: delta/Del summed over lump outlets
pub=np.array([[sum(b.delta[q][b.outlets.index(s)] for s in ss) for q in ('Q168','Q0')] for _,ss in lumps])
log('published delta/Del (oil basis)', np.round(pub,5))
res=dict(unit=u,batch=m,knobs=dict(zip(knobs,[float(x) for x in cal.theta])),pred=cal.predicted.tolist(),targ=cal.target.tolist(),err=cal.max_abs_error,
         lumps=[l for l,_ in lumps], ad=dYdq.tolist(), published=pub.tolist(), crude=np.asarray(c).tolist())
json.dump(res, open(S+f'/hdt_{u}.json','w')); log('done')
