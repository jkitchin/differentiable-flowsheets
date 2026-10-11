"""Spike (#440): the whole catalytic reformer as ONE equation-oriented system.

Unknown vector ``z`` (all in one flat array)::

    [ Y_bed1 | Y_bed2 | Y_bed3 | R (recycle, 29) | Fl (separator liquid, 29) | Fv (vapour, 29) ]

* each bed: Radau collocation on finite elements (:mod:`eo_bed`), 31 unknowns
  per collocation point (29 fractions, coke, T);
* heaters are explicit (inlet T is a specification, so a bed's inlet is a
  function of the previous outlet / feed + recycle);
* separator: ``EOSFlash.eo_residuals`` (Peng-Robinson fugacity equality +
  component balance), 58 rows;
* recycle closure: ``R_i = (H2_HC * HC_feed / Fv_H2) * Fv_i`` (the splitter at
  its H2/HC specification, 29 rows);
* stabilizer, compressor, heater duties: explicit post-processing of the
  unknowns (they close no loop).

Parameters ``p = [WAIT (C), feed rate (kg/s), H2/HC]``.
The system is square (``m = n``) with a zero objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from difflow.dynamic.collocation import Collocation
from difflow.solvers import Bounds, detect_patterns, detect_jacobian_pattern
from difflow.solvers.nlp import _hessian_from_rows
from difflow.solvers.sparsity import SparsityDetectionError
from difflow_refinery.reforming import species as sp
from difflow_refinery.reforming import thermo as th
from difflow_refinery.reforming.reactor import stream_of
from difflow_refinery.reforming.unit import CatalyticReformer, ReformerParams, ReformerResult
import eo_bed as eb

jax.config.update("jax_enable_x64", True)
NS = sp.N_SPECIES


@dataclass
class EOReformer:
    reformer: CatalyticReformer
    feed: object
    mesh: Collocation

    def __post_init__(self):
        p = self.reformer.params
        self.n_rx = p.n_reactors
        self.bed = self.mesh.n_points * eb.NV
        self.n = self.n_rx * self.bed + 3 * NS
        self.m = self.n
        self.base_rate = float(self.feed.mass_flow)
        self.base_wait_C = float(p.wait) - 273.15
        self._flash = None
        from difflow.units.flash import EOSFlash, EOSFlashParams
        self._flash = EOSFlash(EOSFlashParams(species_order=list(sp.NAMES)), th.peng_robinson())
        self.p0 = jnp.array([self.base_wait_C, self.base_rate, float(p.H2_HC)])

    # ----- layout --------------------------------------------------------------
    def split(self, z):
        Ys = [z[k * self.bed:(k + 1) * self.bed].reshape(self.mesh.n_elements, self.mesh.K, eb.NV)
              for k in range(self.n_rx)]
        o = self.n_rx * self.bed
        return Ys, z[o:o + NS], z[o + NS:o + 2 * NS], z[o + 2 * NS:o + 3 * NS]

    def pack(self, Ys, R, Fl, Fv):
        return jnp.concatenate([Y.ravel() for Y in Ys] + [R, Fl, Fv])

    def conditions(self, p):
        """Per-parameter-vector data: inlet temperatures, feed flows, catalyst, H2 need."""
        rp = self.reformer.params
        wait_C, rate, h2hc = p[0], p[1], p[2]
        scale = rate / self.base_rate
        T_in = [jnp.asarray(t, dtype=float) + (wait_C + 273.15 - rp.wait) for t in rp.inlet_T]
        feed_F = jnp.asarray(self.feed.flows) * scale
        Ws = [W * scale for W in self.reformer.catalyst_masses(self.feed, rp)]
        hc = jnp.asarray(self.feed.hydrocarbon_moles) * scale
        return dict(T_in=T_in, feed_F=feed_F, Ws=Ws, need=h2hc * hc, P_rx=rp.P_reactor,
                    T_sep=jnp.asarray(rp.T_separator), P_sep=jnp.asarray(rp.P_separator))

    # ----- the system ----------------------------------------------------------
    def residual(self, z, p):
        c = self.conditions(p)
        rp = self.reformer.params
        Ys, R, Fl, Fv = self.split(z)
        res = []
        F_in = c["feed_F"] + R
        for k in range(self.n_rx):
            res.append(eb.bed_residual(Ys[k], F_in, c["T_in"][k], c["P_rx"], c["Ws"][k], rp.kinetics,
                                       self.mesh).ravel())
            F_in, _, _ = eb.outlet(Ys[k], F_in)
        eff = stream_of(F_in, Ys[-1][-1, -1, NS + 1], c["P_rx"])
        liq = stream_of(Fl, c["T_sep"], c["P_sep"])
        vap = stream_of(Fv, c["T_sep"], c["P_sep"])
        fl = self._flash.eo_residuals([eff], [liq, vap], T=c["T_sep"], P=c["P_sep"])[: 2 * NS]
        res.append(fl)
        iH = sp.INDEX["H2"]
        res.append(R - c["need"] * Fv / Fv[iH])
        return jnp.concatenate(res)

    def objective(self, z, p):
        return 0.0 * z[0]

    # ----- NLP handoff ---------------------------------------------------------
    def bounds(self, z0, detect=True):
        n = self.n
        lb = -jnp.inf * jnp.ones(n)
        ub = jnp.inf * jnp.ones(n)
        names = [f"z{i}" for i in range(n)]
        f = lambda x, p: self.objective(x, p)
        g = lambda x, p: self.residual(x, p)
        self.pattern_note = ""
        if detect:
            try:
                jac, hess = detect_patterns(self.objective, self.residual, z0, self.m, args=(self.p0,))
                self.pattern_note = "asdex jac + asdex hess"
            except SparsityDetectionError as exc:
                # asdex has no handler for one of the primitives in the Hessian pass (see report); the
                # Jacobian pattern is still graph-derived, and the Lagrangian Hessian of g is a subset of
                # the union of row-support outer products (the repo's own structural rule).
                jac = detect_jacobian_pattern(self.residual, z0, args=(self.p0,))
                rc = [set() for _ in range(self.m)]
                for r_, c_ in zip(*jac):
                    rc[int(r_)].add(int(c_))
                hess = _hessian_from_rows(rc, set(), n)
                self.pattern_note = f"asdex jac + row-support Hessian superset ({type(exc).__name__})"
        else:
            jac = hess = None
        bd = Bounds(lb=lb, ub=ub, cl=jnp.zeros(self.m), cu=jnp.zeros(self.m), x0=z0, p0=self.p0,
                    var_names=names, con_names=[f"r{i}" for i in range(self.m)], jac_pattern=jac,
                    hess_pattern=hess, n_decisions=0, layout=None, sparsity_source="global")
        return f, g, bd

    # ----- from z to the flowsheet's own objects -------------------------------
    def streams(self, z, p):
        c = self.conditions(p)
        rp = self.reformer.params
        Ys, R, Fl, Fv = self.split(z)
        S = {}
        S["naphtha"] = stream_of(c["feed_F"], jnp.asarray(self.feed.T), jnp.asarray(self.feed.P))
        S["recycle"] = stream_of(R, jnp.asarray(330.0), c["P_rx"])
        F_in = c["feed_F"] + R
        infos = []
        for k in range(self.n_rx):
            Fo, To, coke = eb.outlet(Ys[k], F_in)
            infos.append(dict(F_in=F_in, T_in=c["T_in"][k], F_out=Fo, T_out=To, coke=coke, W=c["Ws"][k]))
            S[f"rx{k+1}_in"] = stream_of(F_in, c["T_in"][k], c["P_rx"])
            S[f"rx{k+1}_out"] = stream_of(Fo, To, c["P_rx"])
            F_in = Fo
        S["sep_liquid"] = stream_of(Fl, c["T_sep"], c["P_sep"])
        S["sep_vapor"] = stream_of(Fv, c["T_sep"], c["P_sep"])
        return S, infos

    def products(self, z, p):
        """Flows of the product streams (cheap, no duties): reformate, LPG, fuel gas, net gas."""
        S, _ = self.streams(z, p)
        _, ops, _ = self._ops(p)
        ref, lpg, fg, _ = ops["stabilizer"](S["sep_liquid"])
        rec, net, _ = ops["splitter"](S["sep_vapor"])
        return dict(reformate=ref, lpg=lpg, fuel_gas=fg, net_gas=net, recycle_suction=rec, **S)

    def _ops(self, p):
        # the unit objects only carry scalars; build them at p's feed/params
        c = self.conditions(p)
        rp = self.reformer.params
        feedp = self.feed.scaled(p[1] / self.base_rate)
        rp2 = rp.update(H2_HC=p[2]) if hasattr(rp, "update") else rp
        fs, ops, Ws = self.reformer.build(feedp, rp2.with_wait(p[0] + 273.15))
        return fs, ops, Ws

    def result(self, z, p) -> ReformerResult:
        """A full :class:`ReformerResult` (all outputs()) from the solved ``z``."""
        p = jnp.asarray(p)
        S, infos = self.streams(z, p)
        fs, ops, Ws = self._ops(p)
        rp = self.reformer.params.update(H2_HC=p[2]).with_wait(p[0] + 273.15)
        feedp = self.feed.scaled(p[1] / self.base_rate)
        S["recycle_suction"], S["net_gas"], spl = ops["splitter"](S["sep_vapor"])
        S["recycle_out"], comp = ops["compressor"](S["recycle_suction"])
        S["recycle"] = S["recycle_out"]      # the tear is the compressor discharge (T matters for heater 1's duty)
        S["reformate"], S["lpg"], S["fuel_gas"], stab = ops["stabilizer"](S["sep_liquid"])
        info = {}
        for u in fs.units:
            if u.name.startswith("rx") or u.name in ("separator", "splitter", "compressor", "stabilizer"):
                continue
            info[u.name] = u.operation(*[S[n] for n in u.inlet_names])[-1]
        reactors = []
        for k, i in enumerate(infos):
            Fi = i["F_in"]
            reactors.append(dict(T_in=i["T_in"], T_out=i["T_out"], dT=i["T_out"] - i["T_in"], coke=i["coke"],
                                 W=i["W"], H_in=th.total_enthalpy_flash(Fi, i["T_in"], S[f"rx{k+1}_in"]["P"]),
                                 H_out=th.total_enthalpy_flash(i["F_out"], i["T_out"], S[f"rx{k+1}_in"]["P"])))
        heaters = [info[f"heater{k+1}"] for k in range(self.n_rx)]
        sepinfo = ops["separator"](S[f"rx{self.n_rx}_out"])[-1]
        return ReformerResult(streams=S, feed=feedp, params=rp, reactors=reactors, heaters=heaters,
                              separator=sepinfo, splitter=spl, compressor=comp, stabilizer=stab,
                              converged=True, iterations=None)

    def minimal_outputs(self, z, p, keys=("reformate.RON", "h2.net_mol_s")):
        """Differentiable-friendly subset: only what RON and net H2 need."""
        from difflow_refinery.reforming.products import reformate_properties
        pr = self.products(z, p)
        from difflow_refinery.reforming.reactor import flows_of
        out = {}
        if "reformate.RON" in keys:
            out["reformate.RON"] = reformate_properties(flows_of(pr["reformate"]))["RON"]
        if "h2.net_mol_s" in keys:
            out["h2.net_mol_s"] = flows_of(pr["net_gas"])[sp.INDEX["H2"]]
        return jnp.stack([out[k] for k in keys])

    # ----- starting points -----------------------------------------------------
    def z_from_result(self, res: ReformerResult, p=None):
        """Starting vector from an existing solve: bed profiles by diffrax at the converged inlets."""
        from difflow_refinery.reforming.reactor import integrate_bed, flows_of
        p = self.p0 if p is None else p
        c = self.conditions(p)
        rp = self.reformer.params
        Ys = []
        tt = jnp.asarray(self.mesh.t).ravel()
        for k in range(self.n_rx):
            s = res.streams[f"rx{k+1}_in"]
            F, T = flows_of(s), s["T"]
            F0 = jnp.sum(F)
            n_pts = self.mesh.n_points
            # one diffrax solve per collocation time: simple and robust for a start; not timed
            Yk = []
            for w in np.asarray(self.mesh.t).ravel():
                Fo, To, co, _ = integrate_bed(F, T, c["P_rx"], c["Ws"][k] * w, rp.kinetics, rtol=1e-10)
                Yk.append(jnp.concatenate([Fo / F0, jnp.atleast_1d(co), jnp.atleast_1d(To)]))
            Ys.append(jnp.stack(Yk).reshape(self.mesh.n_elements, self.mesh.K, eb.NV))
        R = res.flows("recycle")
        return self.pack(Ys, R, res.flows("sep_liquid"), res.flows("sep_vapor"))
