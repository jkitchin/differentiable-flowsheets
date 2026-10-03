"""Independent reference models for validating ``difflow_refinery`` (#298).

Nothing in this package is imported by the test suite at collection time
except :mod:`.case` and the committed JSON: the reference models need IDAES,
Pyomo and IPOPT, which CI does not install. ``generate.py`` runs them and
writes ``cdu_reference.json``; ``tests/refinery/test_validation.py`` asserts
difflow against that file.

* :mod:`.case` -- the crude, the column layout and the specs, as plain data,
  and the one function that calls difflow to characterise the crude. That is
  the only place the reference touches difflow's code: the pseudo-component
  constants are an *input* to both models.
* :mod:`.formulas` -- the thermodynamic model (Lee-Kesler vapour pressure,
  ideal-gas Cp, Watson latent heat, IAPWS-IF97 water saturation) written from
  the published equations over a pluggable math namespace, so the same text
  evaluates on floats and builds Pyomo expressions.
* :mod:`.idaes_thermo` -- IDAES generic property packages over the same
  constants: the ideal (Raoult + Lee-Kesler) package and Peng-Robinson.
* :mod:`.mesh` -- an equation-oriented MESH column in Pyomo, solved by IPOPT:
  stage network, side strippers, pumparounds, steam, total condenser with free
  water, furnace, over a general stage network.

The vacuum column (#294) has its own, on the same pattern
(``vdu_generate.py`` writes ``vdu_reference.json``; ``test_vdu_validation.py``
and ``test_vdu_validation_file.py`` read it):

* :mod:`.vdu_case` -- the residue (characterised by difflow, the shared
  input), the bed layout and the specs, and the call that runs difflow's
  column on them.
* :mod:`.vdu_formulas` -- Maxwell-Bonnell vapour pressure with the Watson-K
  correction, Kesler-Lee liquid Cp, Clausius-Clapeyron latent heat and NIST
  steam, from the published equations, over the same pluggable namespace.
* :mod:`.vdu_mesh` -- the vacuum column as Pyomo equations: packed beds as
  (Murphree) stages, pumparounds with their return temperature as an unknown,
  wash bed and overflash, entrainment routes, flash zone, bottom steam.
* :mod:`.vdu_generate` -- solves it from an engineering guess, the two
  model variants, a Murphree case and central differences, and writes the
  file.
"""
