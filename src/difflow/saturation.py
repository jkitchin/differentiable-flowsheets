"""Pure-component saturation temperature and pressure.

One explicit entry point for the question "at what temperature does this
species boil at ``P``?" (and its inverse), under either property model:

- ``model="ideal"``: the vapor-pressure correlation of an
  :class:`~difflow.thermo.IdealThermo` (Antoine, or Wagner where the
  species carries a Wagner set) -- :meth:`IdealThermo.saturation_T` and
  :meth:`IdealThermo.Psat`.
- ``model="eos"``: the cubic EOS's own saturation curve, equal fugacities of
  liquid and vapor -- :func:`difflow.eos.eos_bubble_T` and
  :func:`difflow.eos.eos_bubble_P` at a one-hot composition.

Both are differentiable (``jax.grad`` of ``T_sat`` with respect to ``P``
gives the Clausius-Clapeyron slope, positive). The two models agree only as
well as the EOS reproduces vapor pressures: Peng-Robinson is fitted to the
acentric factor, i.e. to ``Psat`` at ``Tr = 0.7``, and is typically within a
few percent of the correlation elsewhere.

Note :meth:`IdealThermo.vaporization_T` is something else: the fixed
temperature at which the ideal enthalpy path vaporizes a species (the lower
of 298.15 K and the normal boiling point), not a saturation temperature at
an arbitrary pressure.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = ["saturation_temperature", "saturation_pressure"]


def _ideal(species: str, thermo):
    if thermo is None:
        from difflow.database import get_species_data
        from difflow.thermo import IdealThermo
        thermo = IdealThermo({species: get_species_data(species)})
    if species not in thermo.species:
        raise KeyError(f"{species!r} is not in the thermo's species "
                       f"{list(thermo.species)}")
    return thermo


def _eos(species: str, eos):
    if eos is None:
        from difflow.database import get_critical_props
        from difflow.eos import PengRobinson
        eos = PengRobinson({species: get_critical_props(species)})
    order = list(eos.species_order)
    if species not in order:
        raise KeyError(f"{species!r} is not in the EOS's species {order}")
    x = jnp.zeros(len(order)).at[order.index(species)].set(1.0)
    return eos, x


def _check_model(model: str) -> None:
    if model not in ("ideal", "eos"):
        raise ValueError(f"model must be 'ideal' or 'eos', not {model!r}")


def saturation_temperature(species: str, P: Array | float, model: str = "ideal",
                           *, thermo=None, eos=None,
                           T0: Array | float | None = None) -> Array:
    """Saturation (boiling) temperature of a pure species at pressure ``P``.

    Args:
        species: Species name, as in the database or in ``thermo``/``eos``.
        P: Pressure (Pa).
        model: ``"ideal"`` (default): invert the vapor-pressure correlation,
            :meth:`~difflow.thermo.IdealThermo.saturation_T` (closed form
            for Antoine, Newton for Wagner). ``"eos"``: the cubic EOS
            saturation temperature, :func:`~difflow.eos.eos_bubble_T` at a
            one-hot liquid.
        thermo: The ``IdealThermo`` to use for ``model="ideal"``; default a
            one-species ``IdealThermo`` from :func:`~difflow.database.get_species_data`
            (Antoine).
        eos: The ``PengRobinson``/``SRK`` to use for ``model="eos"``; default
            Peng-Robinson from :func:`~difflow.database.get_critical_props`.
            It may hold other species; only ``species`` is present.
        T0: Optional initial guess (K).

    Returns:
        Saturation temperature (K); differentiable in ``P`` (and in the EOS
        or correlation parameters).

    Raises:
        KeyError: ``species`` is not in the given thermo/EOS (or database).
        ValueError: Unknown ``model``.

    Warns:
        AntoineRangeWarning: ``"ideal"`` result outside the correlation's
            range. BubbleDewConvergenceWarning: ``"eos"`` solve that did not
            converge (e.g. ``P`` at or above the critical pressure, where
            there is no saturation temperature).

    Example:
        >>> import jax
        >>> T = saturation_temperature("water", 101325.0)
        >>> abs(float(T) - 373.12) < 0.1
        True
        >>> dTdP = jax.grad(lambda p: saturation_temperature("water", p))(101325.0)
        >>> bool(dTdP > 0)
        True
    """
    _check_model(model)
    if model == "ideal":
        return _ideal(species, thermo).saturation_T(species, P, T0)
    from difflow.eos import eos_bubble_T
    eos, x = _eos(species, eos)
    return eos_bubble_T(eos, x, P, T_guess=T0)["T"]


def saturation_pressure(species: str, T: Array | float, model: str = "ideal",
                        *, thermo=None, eos=None,
                        P0: Array | float | None = None) -> Array:
    """Saturation (vapor) pressure of a pure species at temperature ``T``.

    The inverse of :func:`saturation_temperature`, with the same models:
    ``"ideal"`` is :meth:`~difflow.thermo.IdealThermo.Psat` (explicit);
    ``"eos"`` is :func:`~difflow.eos.eos_bubble_P` at a one-hot liquid.

    Args:
        species: Species name.
        T: Temperature (K).
        model: ``"ideal"`` (default) or ``"eos"``.
        thermo: ``IdealThermo`` for ``"ideal"`` (default from the database).
        eos: ``PengRobinson``/``SRK`` for ``"eos"`` (default Peng-Robinson).
        P0: Optional initial pressure for the EOS solve (Pa).

    Returns:
        Saturation pressure (Pa), differentiable in ``T``.

    Example:
        >>> P = saturation_pressure("water", 373.15)
        >>> abs(float(P) / 101325.0 - 1.0) < 0.01
        True
    """
    _check_model(model)
    if model == "ideal":
        return _ideal(species, thermo).Psat(species, T)
    from difflow.eos import eos_bubble_P
    eos, x = _eos(species, eos)
    return eos_bubble_P(eos, x, T, P_guess=P0)["P"]
