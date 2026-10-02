"""Product separator, recycle-gas splitter and stabilizer of the reformer.

Kept thin and local on purpose: the hydrotreater (#306) is building a shared
separator-and-recycle module, and these three are written directly on
difflow's :class:`~difflow.units.flash.EOSFlash` (Peng-Robinson) so that
they can be folded into it later without changing the reformer's numbers.

* :class:`ProductSeparator` -- the effluent cooled to the separator
  temperature and flashed (PR, :func:`difflow.eos.flash_TP_eos` through
  ``EOSFlash``) at the separator pressure. The vapour is the flash's
  ``V y``; the liquid is the feed less the vapour, so the component balance
  closes to round-off whatever the flash's own convergence. The cooler duty
  is the enthalpy change on the reformer's basis.
* :class:`RecycleSplitter` -- splits the separator vapour into recycle gas
  and net gas so that the recycle carries ``H2_HC`` mol of hydrogen per mol
  of naphtha hydrocarbon: the H2/HC ratio is the specification, the split
  fraction is what follows from it.
* :class:`Stabilizer` -- a DOCUMENTED SIMPLIFICATION. A real stabilizer is a
  debutanizer column; issue #312's gas-plant columns, which this would use,
  do not exist, and difflow's stage columns are not set up for a
  hydrogen-bearing feed. It is a component split instead: hydrogen,
  methane and ethane leave as fuel gas, propane and ``c4_recovery`` of the
  butanes as LPG, and the rest (with ``1 - c4_recovery`` of the butanes) as
  stabilized reformate, all at the stated product temperature. Its
  ``c4_recovery`` stands in for the RVP / C4-in-reformate spec. The
  reported duty is the net heat the split needs on the reformer's enthalpy
  basis (reboiler less condenser of an ideal sharp column; not a column
  design).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow.units.flash import EOSFlash, EOSFlashParams
from difflow_refinery.reforming import species as sp
from difflow_refinery.reforming import thermo as th
from difflow_refinery.reforming.reactor import flows_of, stream_of


@dataclass
class ProductSeparatorParams(ParamsMixin):
    """Parameters for :class:`ProductSeparator`.

    Attributes:
        T: Separator temperature (K), after the effluent cooler.
        P: Separator pressure (Pa).
    """

    T: float = 311.15
    P: float = 12.0e5


class ProductSeparator:
    """Effluent cooler and high-pressure separator (PR flash).

    Returns ``(liquid, vapor, info)``; ``info``: ``duty`` (W, negative:
    heat removed), ``V_frac``, ``H2_purity`` (mole fraction of the vapour).
    """

    symbol = "SEP"
    parameter_units = {"T": "K", "P": "Pa"}

    def __init__(self, params: ProductSeparatorParams):
        self.params = params

    def __call__(self, inlet: Stream):
        p = self.params
        F = flows_of(inlet)
        T = jnp.asarray(p.T, dtype=float)
        P = jnp.asarray(p.P, dtype=float)
        Fl, Fv, V_frac, H_in, H_out = _separator_core(F, inlet["T"], inlet["P"], T, P)
        out = {"duty": H_out - H_in, "V_frac": V_frac,
               "H2_purity": Fv[sp.INDEX["H2"]] / jnp.sum(Fv), "H_in": H_in, "H_out": H_out}
        return stream_of(Fl, T, P), stream_of(Fv, T, P), out


_FLASH = None


def _flash_unit() -> EOSFlash:
    global _FLASH
    if _FLASH is None:
        _FLASH = EOSFlash(EOSFlashParams(species_order=list(sp.NAMES)), th.peng_robinson())
    return _FLASH


@jax.jit
def _separator_core(F, T_in, P_in, T, P):
    _, vap, info = _flash_unit()(stream_of(F, T_in, P_in), T=T, P=P)
    Fv = flows_of(vap)
    Fl = F - Fv
    H_in = th.total_enthalpy_flash(F, T_in, P_in)
    H_out = th.total_enthalpy_flash(Fl, T, P) + th.total_enthalpy_flash(Fv, T, P)
    return Fl, Fv, info["V_frac"], H_in, H_out


@dataclass
class RecycleSplitterParams(ParamsMixin):
    """Parameters for :class:`RecycleSplitter`.

    Attributes:
        H2_HC: Recycle hydrogen per naphtha hydrocarbon (mol/mol).
        hydrocarbon_feed: Naphtha hydrocarbon rate the ratio refers to (mol/s).
        max_fraction: Cap on the recycled share of the vapour (a guard for
            early iterates; never active at a converged design point).
    """

    H2_HC: float
    hydrocarbon_feed: float
    max_fraction: float = 0.995


class RecycleSplitter:
    """Separator vapour to recycle gas and net gas, at the H2/HC spec.

    Returns ``(recycle, net_gas, info)``; ``info``: ``fraction`` recycled.
    """

    symbol = "SPL"
    parameter_units = {"H2_HC": "mol/mol", "hydrocarbon_feed": "mol/s", "max_fraction": "-"}

    def __init__(self, params: RecycleSplitterParams):
        self.params = params

    def __call__(self, vapor: Stream):
        p = self.params
        F = flows_of(vapor)
        H2 = F[sp.INDEX["H2"]]
        need = jnp.asarray(p.H2_HC) * jnp.asarray(p.hydrocarbon_feed)
        s = jnp.minimum(need / H2, p.max_fraction)
        return (stream_of(s * F, vapor["T"], vapor["P"]),
                stream_of((1.0 - s) * F, vapor["T"], vapor["P"]), {"fraction": s})


#: Species leaving the stabilizer as fuel gas, and as LPG.
FUEL_GAS: tuple[str, ...] = ("H2", "C1", "C2")
LPG: tuple[str, ...] = ("C3", "iC4", "nC4")


@dataclass
class StabilizerParams(ParamsMixin):
    """Parameters for :class:`Stabilizer`.

    Attributes:
        c4_recovery: Share of the butanes taken overhead to LPG (the rest
            stays in the reformate) -- the stand-in for an RVP spec.
        T: Product temperature (K).
        P: Product pressure (Pa).
    """

    c4_recovery: float = 0.95
    T: float = 311.15
    P: float = 10.0e5


class Stabilizer:
    """Component-split stabilizer (see the module docstring).

    Returns ``(reformate, lpg, fuel_gas, info)``; ``info``: ``duty`` (W).
    """

    symbol = "STAB"
    parameter_units = {"c4_recovery": "-", "T": "K", "P": "Pa"}

    def __init__(self, params: StabilizerParams):
        self.params = params

    def __call__(self, liquid: Stream):
        p = self.params
        F = flows_of(liquid)
        r = jnp.asarray(p.c4_recovery, dtype=float)
        fg = jnp.asarray([k in FUEL_GAS for k in sp.NAMES], dtype=float)
        c3 = jnp.asarray([k == "C3" for k in sp.NAMES], dtype=float)
        c4 = jnp.asarray([k in ("iC4", "nC4") for k in sp.NAMES], dtype=float)
        F_fg = F * fg
        F_lpg = F * (c3 + r * c4)
        F_ref = F - F_fg - F_lpg
        T, P = jnp.asarray(p.T, dtype=float), jnp.asarray(p.P, dtype=float)
        H_in = th.total_enthalpy_flash(F, liquid["T"], liquid["P"])
        H_out = sum(th.total_enthalpy_flash(x, T, P) for x in (F_ref, F_lpg, F_fg))
        return (stream_of(F_ref, T, P), stream_of(F_lpg, T, P), stream_of(F_fg, T, P),
                {"duty": H_out - H_in, "H_in": H_in, "H_out": H_out})
