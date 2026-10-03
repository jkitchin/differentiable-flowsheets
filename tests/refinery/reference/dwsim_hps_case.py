"""The DWSIM high-pressure-separator case: inputs shared by
``dwsim_hps_generate.py`` and ``tests/refinery/test_dwsim_hps.py``.

The feed is a real hydrotreater reactor effluent: the diesel case of
``tests/refinery/test_hydrotreating.py`` (crude A, 230-370 C cut, 50 kg/s,
:class:`~difflow_refinery.hydrotreating.HydrotreaterParams` defaults) solved
by :class:`~difflow_refinery.hydrotreating.Hydrotreater`, its
``streams["reactor_out"]`` taken as the separator feed
(:func:`~difflow_refinery.hydroprocessing.separator.flash_components`: H2,
H2S, NH3, C1-nC4 and the treated pseudo-components; water is decanted, not
flashed, and the diesel effluent carries none). Components with no flow at
all (n-pentane and three empty light cuts) are left out; the rest, traces
included, are flashed.

It is flashed at the unit's own separator conditions (50 C, 47 bar) and at
the corners of 40-60 C x 30-60 bar with
:func:`~difflow_refinery.hydroprocessing.separator.pr_flash` on the
hydroprocessing PR (1976 kappa up to omega 0.491, Robinson-Peng 1978 above;
:data:`~difflow_refinery.hydroprocessing.thermo.DEFAULT_KIJ` plus 0.0333 for
H2S with every cut) and with DWSIM:

* (a) ``same`` -- DWSIM "Peng-Robinson 1978 (PR78)" (the same kappa branch
  at 0.491), every component a hypothetical compound on difflow's
  constants, difflow's kij;
* (b) ``dwsim_data`` -- the real gases as DWSIM database compounds with
  DWSIM's own kij (the cuts stay hypos on difflow's constants, kij 0 with
  everything: DWSIM has no value for a pseudo-component);
  ``dwsim_compounds`` -- DWSIM's gas constants with difflow's kij (separates
  the constants from the kij); ``pr76`` -- difflow's constants and kij on
  DWSIM's "Peng-Robinson (PR)", the 1976 kappa for every omega (what the
  kappa branch does to the heavy cuts).

The effluent and the constants are frozen into the JSON; a change to them
is a staleness failure, not a disagreement.
"""

from __future__ import annotations

TBP_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
TBP_A = [60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680]     # C; 0.86 SG, high sulfur
DIESEL = {"tbp_pct": TBP_PCT, "tbp_C": TBP_A, "sg": 0.86,
          "light_ends": {"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5},
          "sulfur_wt": 1.8, "nitrogen_wppm": 1500.0, "cut_C": [230.0, 370.0], "rate": 50.0}

#: (T K, P Pa): the unit's separator (50 C, 50 bar less the 3 bar loop drop)
#: and the corners of 40-60 C x 30-60 bar.
POINTS = [[323.15, 47e5], [313.15, 30e5], [313.15, 60e5], [333.15, 30e5], [333.15, 60e5]]

#: Gases whose solubility in the separator liquid is reported.
SOLUTES = ["hydrogen", "hydrogen_sulfide", "ammonia", "methane"]


def unit():
    """The diesel hydrotreater of ``test_hydrotreating.py`` and its feed."""
    import warnings

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import difflow_refinery as dr
    from difflow_refinery.hydrotreating import Hydrotreater, HydrotreaterParams, straight_run_cut
    from difflow_refinery.hydrotreating.feed import select_cuts

    d = DIESEL
    a = dr.Assay(d["tbp_pct"], jnp.asarray([t + 273.15 for t in d["tbp_C"]]), sg=d["sg"],
                 light_ends=d["light_ends"], heavy_end=dr.HeavyEnd(), sulfur_wt=d["sulfur_wt"],
                 nitrogen_wppm=d["nitrogen_wppm"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        char = dr.characterize(a, cut_points=dr.default_cut_points(a), composition=True)
    cuts = select_cuts(char, d["cut_C"][0] + 273.15, d["cut_C"][1] + 273.15)
    feed = straight_run_cut(char, rate=d["rate"], cuts=cuts)
    return Hydrotreater(char, feed, HydrotreaterParams()), feed


def constants(u, feed) -> dict:
    """The unit's flash component table (all components), as plain lists."""
    import numpy as np

    c = u._comps(u.theta(feed))
    out = {"names": list(c.names), "n_gas": int(c.n_gas)}
    for k in ("Tb", "Tc", "Pc", "omega", "hvap_nb", "SG", "MW"):
        out[k] = [float(v) for v in np.asarray(getattr(c, k))]
    out["cp_ig"] = [[float(v) for v in row] for row in np.asarray(c.cp_ig)]
    out["kij"] = np.asarray(c.kij).tolist()
    return out


def effluent() -> dict:
    """Solve the hydrotreater (about two minutes) and return the separator
    feed: ``flows`` (mol/s, every component), the unit's separator ``T``/``P``,
    and the solve's ``converged`` flag, with the full component table."""
    import numpy as np

    from difflow_refinery.hydroprocessing.separator import flash_components

    u, feed = unit()
    comp = constants(u, feed)
    r = u.solve(feed)
    z = np.asarray(flash_components(r.streams["reactor_out"], u.layout, u._comps(u.theta(feed))))
    return {"components": comp, "flows": [float(v) for v in z],
            "T": float(u.params.hps_T), "P": float(u.params.P - u.params.loop_dP),
            "converged": bool(r.converged)}


def present(eff: dict) -> list[int]:
    """Indices of the components with flow (the ones flashed)."""
    return [i for i, f in enumerate(eff["flows"]) if f > 0.0]


def flash_components_of(eff: dict):
    """:class:`~difflow_refinery.hydroprocessing.thermo.Components` on the
    frozen constants, restricted to the components present, and their ``z``."""
    import jax.numpy as jnp
    import numpy as np

    from difflow_refinery.hydroprocessing.thermo import Components

    c = eff["components"]
    idx = present(eff)
    names = tuple(c["names"][i] for i in idx)
    n_gas = sum(1 for i in idx if i < c["n_gas"])
    arr = lambda k: jnp.asarray(np.asarray(c[k])[idx])  # noqa: E731
    K = np.asarray(c["kij"])[np.ix_(idx, idx)]
    comps = Components(names=names, n_gas=n_gas, Tb=arr("Tb"), Tc=arr("Tc"), Pc=arr("Pc"),
                       omega=arr("omega"), hvap_nb=arr("hvap_nb"),
                       cp_ig=jnp.asarray(np.asarray(c["cp_ig"])[idx]), SG=arr("SG"),
                       MW=arr("MW"), kij=jnp.asarray(K))
    z = np.asarray(eff["flows"])[idx]
    return comps, z / z.sum()
