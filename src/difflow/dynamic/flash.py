"""Dynamic EOS flash drum with liquid-level holdup (PR-consistent VLE).

difflow's steady-state :class:`~difflow.units.flash.EOSFlash` does an isothermal
TP flash with a cubic EOS (``flash_TP_eos`` -> vapor fraction, liquid/vapor
compositions). This module provides its *dynamic* companion for transient
flowsheets, using the same EOS VLE so the two agree at steady state.

Unlike :class:`~difflow.dynamic.dae.DynamicFlashDrum` (correlation K-values and a
Rachford-Rice/K algebraic block), :class:`DynamicEOSFlash` carries the drum's
total species holdup as ODE states and, each RHS evaluation, flashes the drum
contents with the real EOS at the drum ``(T, P)``. The vapor and liquid products
are drawn from the equilibrium phases by first-order residence-time laws::

    z = n / n_total
    V_frac, x, y = flash_TP_eos(eos, z, T, P)         # same VLE as EOSFlash
    M_L = (1 - V_frac) n_total,  M_V = V_frac n_total   # equilibrium holdups
    L  = M_L / tau_liquid,       Vg  = M_V / tau_vapor  # product draws
    dn_i/dt = F z_feed,i - L x_i - Vg y_i

Why the steady state is exact: at ``dn_i/dt = 0``, ``F z_feed,i = L x_i + Vg y_i``
with ``y_i = K_i x_i`` is, summed over species with ``beta = Vg/F``, precisely the
Rachford-Rice flash of the *feed* -- so ``x``, ``y`` and the product split match
:class:`EOSFlash` regardless of ``tau_liquid``/``tau_vapor``. Those time
constants set only the holdup magnitudes and the transient speed. ``tau_liquid``
is the dominant (slow) liquid-level time constant -- the natural handle for level
control; the vapor holdup is small (fast ``tau_vapor``).

The drum is isothermal at the feed temperature, matching ``EOSFlash``'s
isothermal-flash assumption (the steady-state model takes the separator
temperature from its feed). ``tau_liquid`` gives the drum a real liquid inventory
whose level responds to feed/draw imbalance -- the "holdup" this unit adds.
"""

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, get_flows, make_stream
from difflow.eos import flash_TP_eos
from difflow.dynamic.state import StateSpec, StateVar


class DynamicEOSFlash:
    """Dynamic isothermal TP flash drum with liquid-level holdup and EOS VLE.

    State:
        n_<species> -- total moles of each species held in the drum (mol).

    Inputs (``inputs`` dict): the feed stream under key ``"feed"`` (or the single
    provided stream). Outputs: ``{"liquid": Stream, "vapor": Stream}`` at the drum
    ``(T, P)``.
    """

    symbol = "Dynamic EOS flash"
    equations = [
        r"\frac{dn_i}{dt} = F z_{i,\mathrm{feed}} - L x_i - V y_i",
        r"V_\mathrm{frac}, x, y = \mathrm{flash\_TP\_eos}(z, T, P),\quad z_i = n_i/n_\mathrm{tot}",
        r"L = (1-V_\mathrm{frac})n_\mathrm{tot}/\tau_L,\qquad V = V_\mathrm{frac}\,n_\mathrm{tot}/\tau_V",
    ]
    assumptions = [
        "Isothermal at the feed temperature; instantaneous VLE at drum (T, P).",
        "Total species holdup as ODE states; products drawn by residence-time laws.",
        "Cubic-EOS VLE (flash_TP_eos), consistent with the steady-state EOSFlash.",
    ]
    references = [
        "Biegler, Grossmann, Westerberg. Systematic Methods of Chemical Process Design, 1997.",
        "Luyben, W.L. Process Modeling, Simulation, and Control for Chemical Engineers, 2e.",
    ]
    parameter_symbols = {"P": "P", "tau_liquid": r"\tau_L", "tau_vapor": r"\tau_V"}
    parameter_units = {"P": "Pa", "tau_liquid": "s", "tau_vapor": "s"}
    numerical_method = "ODE in species holdup; EOS TP flash (flash_TP_eos) each RHS eval."

    def __init__(
        self,
        eos,
        species_order: list[str],
        P: float | Array,
        tau_liquid: float | Array = 120.0,
        tau_vapor: float | Array = 10.0,
        k_ij: Array | None = None,
        name: str = "flash",
    ):
        """Initialize the dynamic flash drum.

        Args:
            eos: Cubic EOS (e.g. difflow.eos.PengRobinson) providing VLE via
                ``flash_TP_eos``; the same object the steady-state flash uses.
            species_order: Species names, matching the EOS ordering.
            P: Drum (flash) pressure (Pa).
            tau_liquid: Liquid-holdup time constant (s) -- the slow level state.
            tau_vapor: Vapor-holdup time constant (s) -- fast; keep small so the
                vapor holdup is a small fraction of the inventory.
            k_ij: Binary interaction parameters (optional).
            name: Unit name.
        """
        self.eos = eos
        self.species_order = species_order
        self.P = jnp.asarray(P)
        self.tau_liquid = jnp.asarray(tau_liquid)
        self.tau_vapor = jnp.asarray(tau_vapor)
        self.k_ij = k_ij
        self.name = name

    def state_spec(self) -> StateSpec:
        return StateSpec([
            StateVar(f"n_{s}", "moles", "mol", f"Drum holdup of {s}", bounds=(0.0, None))
            for s in self.species_order
        ])

    @staticmethod
    def _feed(inputs: dict[str, Stream]) -> Stream:
        return inputs.get("feed") or list(inputs.values())[0]

    def _split(self, n: Array, T: Array):
        """Flash the drum contents; return (x, y, L, Vg, V_frac, M_L)."""
        n_safe = jnp.maximum(n, 0.0)
        n_total = jnp.sum(n_safe) + 1e-30
        z = n_safe / n_total
        V_frac, x, y = flash_TP_eos(self.eos, z, T, self.P, self.k_ij)
        M_L = (1.0 - V_frac) * n_total
        M_V = V_frac * n_total
        L = M_L / self.tau_liquid
        Vg = M_V / self.tau_vapor
        return x, y, L, Vg, V_frac, M_L

    def derivatives(self, t: Array, state: Array, inputs: dict[str, Stream], params=None) -> Array:
        species = self.species_order
        feed = self._feed(inputs)
        T = jnp.asarray(feed["T"])
        feed_flows = get_flows(feed)
        F = jnp.array([feed_flows.get(s, 0.0) for s in species])

        x, y, L, Vg, _, _ = self._split(state, T)
        dn_dt = F - L * x - Vg * y
        return dn_dt

    def outputs(self, t: Array, state: Array, inputs: dict[str, Stream], params=None) -> dict[str, Stream]:
        species = self.species_order
        feed = self._feed(inputs)
        T = jnp.asarray(feed["T"])
        x, y, L, Vg, _, _ = self._split(state, T)
        liquid = make_stream({s: L * x[i] for i, s in enumerate(species)}, T, self.P)
        vapor = make_stream({s: Vg * y[i] for i, s in enumerate(species)}, T, self.P)
        return {"liquid": liquid, "vapor": vapor}

    def liquid_holdup(self, state: Array, inputs: dict[str, Stream]) -> Array:
        """Current liquid molar holdup M_L (mol) -- the level proxy."""
        feed = self._feed(inputs)
        T = jnp.asarray(feed["T"])
        _, _, _, _, _, M_L = self._split(state, T)
        return M_L

    def initial_state(self, inputs: dict[str, Stream], params=None) -> Array:
        """Start the drum with ~tau_liquid worth of feed inventory at feed
        composition (it relaxes to the correct split and holdup)."""
        species = self.species_order
        feed = self._feed(inputs)
        feed_flows = get_flows(feed)
        F = jnp.array([feed_flows.get(s, 0.0) for s in species])
        F_total = jnp.sum(F) + 1e-30
        z = F / F_total
        M0 = F_total * self.tau_liquid
        return M0 * z

    def __repr__(self) -> str:
        return (f"DynamicEOSFlash(name='{self.name}', P={float(self.P):.3e} Pa, "
                f"tau_liquid={float(self.tau_liquid):.1f} s, tau_vapor={float(self.tau_vapor):.1f} s)")
