"""Chemical disinfection in a contact tank: Chick-Watson kinetics, tanks in series.

Kinetics: Chick's law (first order in organisms) with Watson's
concentration exponent,

    dN/dt = -k C^n N,

and first-order decay of the disinfectant residual, ``dC/dt = -k_d C``
[K1, K2: citation verified; the 1908 papers were not read here (their PMC
copies sit behind a bot check), so the forms are the standard statement of
the Chick-Watson law, (unverified) against them].

Hydraulics: ``n_tanks`` equal CSTRs in series of total volume ``V``
(``n_tanks = 1`` a single mixed tank; large ``n_tanks`` approaches plug
flow). Each tank, residence time ``tau = V / (n_tanks Q)``:

    C_i = C_{i-1} / (1 + k_d tau)
    N_i = N_{i-1} / (1 + k C_i^n tau)

(the steady state of the two rate laws in a mixed tank, at the tank's own
residual). Outputs: log inactivation ``-log10(N_out / N_in)`` and the CT
product ``sum C_i tau`` (mg min/L), the quantity regulatory credit tables
are written in. Units: ``k`` in ``(L/mg)^n / min``, concentrations in mg/L,
times in min.

Streams: the organism is ``F_<organism>`` (count/s, ``1 mol = 1
organism``), the disinfectant ``F_<disinfectant>`` with molar mass
``mw_disinfectant`` (default: free chlorine as Cl2). If the feed carries
no disinfectant, ``dose`` (mg/L) is added at the inlet. All defaults are
ILLUSTRATIVE: real rate constants depend on organism, disinfectant, pH and
temperature.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow_water.treatment.common import RHO_DEFAULT, volumetric_flow


@dataclass(repr=False)
class DisinfectionParams(ParamsMixin):
    """Contact tank and Chick-Watson kinetics (defaults ILLUSTRATIVE).

    Attributes:
        organism: Organism component name (``F_<organism>``, count/s).
        disinfectant: Disinfectant component name (``F_<disinfectant>``, mol/s).
        mw_disinfectant: Disinfectant molar mass (kg/mol; Cl2 by default).
        volume: Contact tank volume (m^3).
        n_tanks: Number of equal CSTRs in series (-).
        k: Chick-Watson rate constant ((L/mg)^n / min).
        n: Watson concentration exponent (-).
        k_decay: First-order disinfectant decay constant (1/min).
        dose: Disinfectant dosed at the inlet (mg/L), added to the feed's own.
        rho: Liquid density for the volumetric flow (kg/m^3).
    """

    organism: str = "Giardia"
    disinfectant: str = "Cl2"
    mw_disinfectant: float | Array = 70.906e-3
    volume: float | Array = 500.0
    n_tanks: int = 5
    k: float | Array = 0.2
    n: float | Array = 1.0
    k_decay: float | Array = 0.01
    dose: float | Array = 1.5
    rho: float | Array = RHO_DEFAULT


class Disinfection:
    """Chlorine (or other) contact tank: Chick-Watson inactivation in CSTRs in series.

    Inlet: water carrying the organism (and optionally the disinfectant).
    Outlet: the same stream with the surviving organisms and the decayed
    residual.

    Example:
        >>> from difflow import make_stream
        >>> feed = make_stream({"H2O": 5550.0, "Giardia": 1e3}, T=288.15, P=101325.0)
        >>> out, info = Disinfection(DisinfectionParams())(feed)
        >>> float(info["log_inactivation"]) > 0
        True
    """

    symbol = "CL"
    equations = [
        r"\frac{dN}{dt} = -k C^n N,\qquad \frac{dC}{dt} = -k_d C",
        r"C_i = \frac{C_{i-1}}{1 + k_d \tau},\qquad N_i = \frac{N_{i-1}}{1 + k C_i^n \tau},\qquad \tau = \frac{V}{n_t Q}",
        r"\log\text{-inactivation} = -\log_{10}(N_{out}/N_{in}),\qquad CT = \textstyle\sum_i C_i \tau",
    ]
    assumptions = [
        "Chick-Watson kinetics at each tank's residual; first-order residual decay.",
        "Equal CSTRs in series (n_tanks = 1: one mixed tank; many: near plug flow).",
        "Rate constants are inputs (ILLUSTRATIVE defaults).",
    ]
    references = [
        "Chick, J. Hyg. 8 (1908) 92-158, doi:10.1017/S0022172400006987.",
        "Watson, J. Hyg. 8 (1908) 536-542, doi:10.1017/S0022172400015928.",
    ]
    parameter_symbols = {"k": "k", "n": "n", "k_decay": "k_d", "volume": "V", "n_tanks": "n_t",
                         "dose": "C_0"}
    parameter_units = {"organism": "-", "disinfectant": "-", "mw_disinfectant": "kg/mol",
                       "volume": "m^3", "n_tanks": "-", "k": "(L/mg)^n/min", "n": "-",
                       "k_decay": "1/min", "dose": "mg/L", "rho": "kg/m^3"}
    outlet_roles = ["treated"]

    def __init__(self, params: DisinfectionParams):
        """Initialize.

        Args:
            params: :class:`DisinfectionParams`.
        """
        if int(params.n_tanks) < 1:
            raise ValueError(f"n_tanks must be at least 1, got {params.n_tanks}")
        self.params = params

    def __call__(self, feed: Stream) -> tuple[Stream, dict]:
        """Disinfect a feed.

        Returns:
            ``(treated, info)``; ``info`` has ``log_inactivation``, ``CT``
            (mg min/L), ``residual`` (mg/L at the outlet), ``contact_time``
            (min), ``concentration`` and ``survival`` per tank.
        """
        p = self.params
        org, dis = f"F_{p.organism}", f"F_{p.disinfectant}"
        if org not in feed:
            raise ValueError(f"the feed has no {org} (DisinfectionParams.organism)")
        Q = volumetric_flow(feed, {p.disinfectant: p.mw_disinfectant}, p.rho)   # m^3/s
        C0 = feed.get(dis, 0.0) * p.mw_disinfectant / Q * 1e3 + p.dose          # mg/L
        nt = int(p.n_tanks)
        tau = p.volume / (nt * Q) / 60.0                                          # min
        C = C0
        surv = jnp.asarray(1.0)
        Cs, Ss = [], []
        for _ in range(nt):
            C = C / (1.0 + p.k_decay * tau)
            s = 1.0 / (1.0 + p.k * C ** p.n * tau)
            surv = surv * s
            Cs.append(C), Ss.append(s)
        Cs = jnp.stack(Cs)
        out = dict(feed)
        out[org] = feed[org] * surv
        out[dis] = C * Q / 1e3 / p.mw_disinfectant
        info = {
            "log_inactivation": -jnp.log10(surv),
            "CT": jnp.sum(Cs) * tau,
            "residual": C,
            "dose_total": C0,
            "contact_time": nt * tau,
            "concentration": Cs,
            "survival": jnp.stack(Ss),
        }
        return out, info
