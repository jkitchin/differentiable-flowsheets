"""Compatibility names for the vacuum column's correlations.

The correlations the vacuum column was written with -- Twu (1984) criticals,
Kesler-Lee acentric factor and liquid Cp, Maxwell-Bonnell vapour pressure --
now live in :mod:`difflow_refinery.correlations`, next to the crude unit's and
the blending pool's, so that one function answers for each property (#301).
This module keeps the names the vacuum code and its callers used:

- :func:`riazi_daubert_mw` is the *1987* molecular weight here (it is the
  1980 one in :mod:`difflow_refinery.characterization`; the two modules had
  the same name for different correlations, which is why the unified module
  spells the year).
- :func:`fit_antoine` is the Maxwell-Bonnell Antoine fit
  (:func:`~difflow_refinery.correlations.maxwell_bonnell_antoine`), not the
  Lee-Kesler one :mod:`difflow_refinery.assay` exports under the same name.
"""

from __future__ import annotations

from difflow_refinery.correlations import (  # noqa: F401
    ATM_MMHG,
    BTU_LB_R,
    MMHG,
    R_GAS,
    api_from_sg,
    kesler_lee_liquid_cp,
    kesler_lee_liquid_cp_coefficients,
    kesler_lee_liquid_enthalpy,
    lee_kesler_acentric,
    lee_kesler_psat,
    maxwell_bonnell_antoine as fit_antoine,
    maxwell_bonnell_boiling_point,
    maxwell_bonnell_psat,
    riazi_daubert_1987,
    sg_from_api,
    sg_from_watson_k,
    twu_critical_properties,
    watson_k,
)


def riazi_daubert_mw(Tb, SG):
    """Molecular weight, extended Riazi-Daubert (1987), Tb in K (MNL50 Eq. 2.51)."""
    return riazi_daubert_1987(Tb, SG)[0]
