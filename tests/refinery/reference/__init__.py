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
  water, furnace. Written for a general stage network, so the vacuum column
  (#294) can be posed on it.
"""
