"""IDAES generic property packages over the case's pseudo-components.

Two packages on the same constants (:func:`case.component_data`):

* :func:`ideal_config` -- the property model difflow states (Raoult over
  Lee-Kesler vapour pressures, ideal-gas enthalpy from the cubic Cp, Watson
  latent heat for the liquid), posed through IDAES's modular framework: its
  ``Ideal`` equation of state, ``SmoothVLE`` flash formulation and
  ``IdealBubbleDew``. The pure-component methods IDAES does not ship
  (Lee-Kesler, Watson) are written here from the published equations
  (:mod:`.formulas`). What this checks is everything *around* them: the
  K-values, mixture enthalpies, flash and bubble/dew points as IDAES
  assembles and solves them, against difflow's.
* :func:`pr_config` -- Peng-Robinson (1976) on the same Tc, Pc, omega and
  ideal-gas Cp, through IDAES's ``Cubic`` EOS with zero binary interaction
  parameters. A different property *model*: it measures how far the Raoult
  assumption is from a cubic EOS at this column's conditions, the difference
  the issue expects against a commercial simulator's PR or Grayson-Streed.

Water is a vapour-only component in the ideal package (``valid_phase_types``),
as it is on difflow's trays; the PR package is hydrocarbon-only, because PR
with zero kij says nothing useful about water-hydrocarbon liquid solubility.
"""

from __future__ import annotations

import numpy as np

from . import formulas as fm


class LeeKeslerPsat:
    """``pressure_sat_comp``: Lee-Kesler (1975) with the component's
    vapour-pressure acentric factor."""

    @staticmethod
    def build_parameters(cobj):
        from idaes.core.util.misc import set_param_from_config
        from pyomo.environ import Var
        from pyomo.environ import units as u

        cobj.omega_vp = Var(doc="Lee-Kesler vapour-pressure acentric factor", units=u.dimensionless)
        set_param_from_config(cobj, param="omega_vp")

    @staticmethod
    def return_expression(b, cobj, T, dT=False):
        from pyomo.environ import units as u

        Tr = u.convert(T, to_units=u.K) / cobj.temperature_crit
        if dT:
            m = fm.pyomo_math()
            Tc = cobj.temperature_crit
            w = cobj.omega_vp
            # d ln Pr / dT
            d0 = (6.09648 / Tr**2 - 1.28862 / Tr + 6 * 0.169347 * Tr**5) / Tc
            d1 = (15.6875 / Tr**2 - 13.4721 / Tr + 6 * 0.43577 * Tr**5) / Tc
            return LeeKeslerPsat.return_expression(b, cobj, T) * (d0 + w * d1)
        m = fm.pyomo_math()
        return cobj.pressure_crit * m.exp(fm.lee_kesler_ln_pr(m, Tr, cobj.omega_vp))


class IF97Psat:
    """``pressure_sat_comp`` for water: IAPWS-IF97 region 4."""

    @staticmethod
    def build_parameters(cobj):
        pass

    @staticmethod
    def return_expression(b, cobj, T, dT=False):
        from pyomo.environ import units as u

        if dT:
            raise NotImplementedError
        return fm.water_psat(fm.pyomo_math(), u.convert(T, to_units=u.K) / u.K) * u.Pa


class WatsonLiquidEnthalpy:
    """``enth_mol_liq_comp``: the ideal-gas enthalpy less a Watson latent
    heat anchored at ``Tb``. ``eps`` as in :func:`formulas.watson_base`."""

    eps = 0.0

    @staticmethod
    def build_parameters(cobj):
        from idaes.core.util.misc import set_param_from_config
        from pyomo.environ import Var
        from pyomo.environ import units as u

        cobj.hvap_nb = Var(doc="Heat of vaporisation at Tb", units=u.J / u.mol)
        set_param_from_config(cobj, param="hvap_nb")
        cobj.temperature_boil = Var(doc="Normal boiling point", units=u.K)
        set_param_from_config(cobj, param="temperature_boil")

    @classmethod
    def return_expression(cls, b, cobj, T):
        from idaes.models.properties.modular_properties.pure import RPP4
        from pyomo.environ import units as u

        m = fm.pyomo_math()
        hig = RPP4.enth_mol_ig_comp.return_expression(b, cobj, T)
        Tk = u.convert(T, to_units=u.K) / u.K
        eps = cls.eps
        # eps = 0 cannot go through max() in an expression; 1e-6 is exact to
        # far below the comparison's tolerance and keeps IPOPT on a C2 curve.
        lat = fm.latent_heat(m, Tk, cobj.temperature_boil / u.K, cobj.temperature_crit / u.K,
                             cobj.hvap_nb / (u.J / u.mol), eps if eps > 0 else 1e-6)
        return hig - lat * u.J / u.mol


class SGLiquidDensity:
    """``dens_mol_liq_comp`` from the standard specific gravity: only used by
    IDAES's ``(P - Pref)/rho`` term in the liquid enthalpy, which the
    comparisons cancel by setting ``pressure_ref`` to the state pressure."""

    @staticmethod
    def build_parameters(cobj):
        from idaes.core.util.misc import set_param_from_config
        from pyomo.environ import Var
        from pyomo.environ import units as u

        cobj.sg = Var(doc="Standard specific gravity, 60/60 F", units=u.dimensionless)
        set_param_from_config(cobj, param="sg")

    @staticmethod
    def return_expression(b, cobj, T):
        from pyomo.environ import units as u

        return 999.016 * cobj.sg / cobj.mw * u.kg / u.m**3


def ideal_config(comp: dict, water: bool = True, eps: float = 0.0) -> dict:
    """IDAES config for the ideal (Raoult + Lee-Kesler + Watson) model."""
    from idaes.core import Component, LiquidPhase, VaporPhase
    from idaes.core.base.phases import PhaseType as PT
    from idaes.models.properties.modular_properties.eos.ideal import Ideal
    from idaes.models.properties.modular_properties.phase_equil import SmoothVLE
    from idaes.models.properties.modular_properties.phase_equil.bubble_dew import IdealBubbleDew
    from idaes.models.properties.modular_properties.phase_equil.forms import fugacity
    from idaes.models.properties.modular_properties.pure import RPP4
    from idaes.models.properties.modular_properties.state_definitions import FTPx
    from pyomo.environ import units as u

    watson = type("WatsonEps", (WatsonLiquidEnthalpy,), {"eps": eps})
    cfg = {
        "components": {},
        "phases": {"Liq": {"type": LiquidPhase, "equation_of_state": Ideal},
                   "Vap": {"type": VaporPhase, "equation_of_state": Ideal}},
        "base_units": {"time": u.s, "length": u.m, "mass": u.kg, "amount": u.mol,
                       "temperature": u.K},
        "state_definition": FTPx,
        "state_bounds": {"flow_mol": (0, 100, 1e5, u.mol / u.s),
                         "temperature": (200, 400, 1100, u.K),
                         "pressure": (5e3, 1e5, 1e7, u.Pa)},
        "pressure_ref": (101325, u.Pa),
        "temperature_ref": (fm.T_REF, u.K),
        "include_enthalpy_of_formation": False,
        "phases_in_equilibrium": [("Vap", "Liq")],
        "phase_equilibrium_state": {("Vap", "Liq"): SmoothVLE},
        "bubble_dew_method": IdealBubbleDew,
    }
    for i, n in enumerate(comp["names"]):
        a, b, c, d = comp["cp_ig"][i]
        cfg["components"][n] = {
            "type": Component,
            "pressure_sat_comp": LeeKeslerPsat,
            "enth_mol_ig_comp": RPP4,
            "enth_mol_liq_comp": watson,
            "dens_mol_liq_comp": SGLiquidDensity,
            "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
            "parameter_data": {
                "mw": (comp["MW"][i] / 1000.0, u.kg / u.mol),
                "sg": comp["SG"][i],
                "pressure_crit": (comp["Pc"][i], u.Pa),
                "temperature_crit": (comp["Tc"][i], u.K),
                "temperature_boil": (comp["Tb"][i], u.K),
                "omega_vp": comp["omega_vp"][i],
                "hvap_nb": (comp["hvap_nb"][i], u.J / u.mol),
                "cp_mol_ig_comp_coeff": {"A": a, "B": b, "C": c, "D": d},
            },
        }
    if water:
        a, b, c, d = fm.WATER["cp_ig"]
        cfg["components"]["H2O"] = {
            "type": Component,
            "valid_phase_types": PT.vaporPhase,
            "enth_mol_ig_comp": RPP4,
            "parameter_data": {
                "mw": (fm.WATER["MW"] / 1000.0, u.kg / u.mol),
                "cp_mol_ig_comp_coeff": {"A": a, "B": b, "C": c, "D": d},
            },
        }
    return cfg


def pr_config(comp: dict) -> dict:
    """IDAES config for Peng-Robinson on the hydrocarbons, kij = 0."""
    from idaes.core import Component, LiquidPhase, VaporPhase
    from idaes.models.properties.modular_properties.eos.ceos import Cubic, CubicType
    from idaes.models.properties.modular_properties.phase_equil import SmoothVLE
    from idaes.models.properties.modular_properties.phase_equil.bubble_dew import LogBubbleDew
    from idaes.models.properties.modular_properties.phase_equil.forms import log_fugacity
    from idaes.models.properties.modular_properties.pure import RPP4
    from idaes.models.properties.modular_properties.state_definitions import FTPx
    from pyomo.environ import units as u

    names = comp["names"]
    eos = {"type": CubicType.PR}
    cfg = {
        "components": {},
        "phases": {"Liq": {"type": LiquidPhase, "equation_of_state": Cubic,
                           "equation_of_state_options": eos},
                   "Vap": {"type": VaporPhase, "equation_of_state": Cubic,
                           "equation_of_state_options": eos}},
        "base_units": {"time": u.s, "length": u.m, "mass": u.kg, "amount": u.mol,
                       "temperature": u.K},
        "state_definition": FTPx,
        "state_bounds": {"flow_mol": (0, 100, 1e5, u.mol / u.s),
                         "temperature": (200, 400, 1100, u.K),
                         "pressure": (5e3, 1e5, 1e7, u.Pa)},
        "pressure_ref": (101325, u.Pa),
        "temperature_ref": (fm.T_REF, u.K),
        "include_enthalpy_of_formation": False,
        "phases_in_equilibrium": [("Vap", "Liq")],
        "phase_equilibrium_state": {("Vap", "Liq"): SmoothVLE},
        "bubble_dew_method": LogBubbleDew,
        "parameter_data": {"PR_kappa": {(a, b): 0.0 for a in names for b in names}},
    }
    for i, n in enumerate(names):
        a, b, c, d = comp["cp_ig"][i]
        cfg["components"][n] = {
            "type": Component,
            "enth_mol_ig_comp": RPP4,
            "entr_mol_ig_comp": RPP4,
            # only for IDAES's bubble/dew initial guesses, not the equilibrium
            "pressure_sat_comp": LeeKeslerPsat,
            "phase_equilibrium_form": {("Vap", "Liq"): log_fugacity},
            "parameter_data": {
                "mw": (comp["MW"][i] / 1000.0, u.kg / u.mol),
                "pressure_crit": (comp["Pc"][i], u.Pa),
                "temperature_crit": (comp["Tc"][i], u.K),
                "omega": comp["omega_eos"][i],
                "omega_vp": comp["omega_vp"][i],
                "cp_mol_ig_comp_coeff": {"A": a, "B": b, "C": c, "D": d},
                "entr_mol_form_vap_comp_ref": (0.0, u.J / u.mol / u.K),
            },
        }
    return cfg


# -----------------------------------------------------------------------------
# State-point evaluation
# -----------------------------------------------------------------------------


class StatePoint:
    """One IDAES state block at fixed (F, z, T, P), re-used across points."""

    def __init__(self, cfg: dict, has_water: bool = False):
        from idaes.core import FlowsheetBlock
        from idaes.models.properties.modular_properties import GenericParameterBlock
        from pyomo.environ import ConcreteModel

        self.m = ConcreteModel()
        self.m.fs = FlowsheetBlock(dynamic=False)
        self.m.fs.props = GenericParameterBlock(**cfg)
        self.sb = self.m.fs.props.build_state_block([0], defined_state=True)
        self.m.fs.sb = self.sb
        self.s = self.sb[0]
        self.comps = list(self.m.fs.props.component_list)

    def solve(self, z: dict, T: float, P: float, flow: float = 1.0, init: bool = True):
        """Solve the state at fixed composition, T and P; True if optimal."""
        from idaes.core.solvers import get_solver

        s = self.s
        s.flow_mol.fix(flow)
        s.temperature.fix(T)
        s.pressure.fix(P)
        for j in self.comps:
            s.mole_frac_comp[j].fix(max(z.get(j, 0.0), 1e-12))
        self.m.fs.props.pressure_ref.set_value(P)
        if init:
            self.sb.initialize(outlvl=0)
        res = get_solver(options={"tol": 1e-10, "max_iter": 500}).solve(self.m)
        self.termination = str(res.solver.termination_condition)
        return self.termination == "optimal"

    def solve_two_phase(self, z: dict, T: float, P: float, seed: dict, flow: float = 1.0):
        """Solve a state known to be two-phase, seeded from an ideal flash.

        IDAES's own initialisation of a cubic package computes the bubble and
        dew points of the whole mixture first. For 28 pseudo-components
        spanning 230-1030 K boiling points the dew point sits where the
        heaviest cuts are near-critical and that routine does not finish.
        At a state well inside the envelope SmoothVLE's equilibrium
        temperature is ``T`` itself whatever the bubble and dew points are,
        so they are fixed at the ideal-model values (``seed``) and their
        equations deactivated; the caller checks the result is two-phase.

        Args:
            seed: ``beta``, ``x``, ``y`` (per component), ``T_bubble``,
                ``T_dew`` from the ideal model.
        """
        from idaes.core.solvers import get_solver

        s = self.s
        s.flow_mol.fix(flow)
        s.temperature.fix(T)
        s.pressure.fix(P)
        for j in self.comps:
            s.mole_frac_comp[j].fix(max(z.get(j, 0.0), 1e-12))
        self.m.fs.props.pressure_ref.set_value(P)
        b = seed["beta"]
        s.phase_frac["Vap"].set_value(b)
        s.phase_frac["Liq"].set_value(1 - b)
        s.flow_mol_phase["Vap"].set_value(flow * b)
        s.flow_mol_phase["Liq"].set_value(flow * (1 - b))
        for j in self.comps:
            s.mole_frac_phase_comp["Liq", j].set_value(max(seed["x"][j], 1e-14))
            s.mole_frac_phase_comp["Vap", j].set_value(max(seed["y"][j], 1e-14))
        for blk_name in ("temperature_bubble", "temperature_dew"):
            for v in getattr(s, blk_name).values():
                v.fix(seed["T_bubble"] if blk_name == "temperature_bubble" else seed["T_dew"])
        for cname in ("eq_temperature_bubble", "eq_mole_frac_tbub", "log_mole_frac_tbub_eqn",
                      "eq_temperature_dew", "eq_mole_frac_tdew", "log_mole_frac_tdew_eqn"):
            c = getattr(s, cname, None)
            if c is not None:
                c.deactivate()
        for cname in ("_mole_frac_tbub", "_mole_frac_tdew", "log_mole_frac_tbub", "log_mole_frac_tdew"):
            v = getattr(s, cname, None)
            if v is not None:
                v.fix()
        for v in s._teq.values():
            v.set_value(T)
        for cname in ("log_mole_frac_comp", "log_mole_frac_phase_comp"):
            v = getattr(s, cname, None)
            if v is not None:
                for k, vv in v.items():
                    base = s.mole_frac_comp[k] if cname == "log_mole_frac_comp" else s.mole_frac_phase_comp[k]
                    vv.set_value(float(np.log(max(base.value, 1e-300))))
        res = get_solver(options={"tol": 1e-10, "max_iter": 1000}).solve(self.m)
        self.termination = str(res.solver.termination_condition)
        return self.termination == "optimal"

    def value(self, expr):
        from pyomo.environ import value

        return value(expr)
