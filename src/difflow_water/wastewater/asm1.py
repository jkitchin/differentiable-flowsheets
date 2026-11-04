"""Activated Sludge Model No. 1 (ASM1), as the BSM1 report states it.

Thirteen state variables and eight processes, transcribed from the BSM1
technical report (Alex et al. 2008 [W5], Section 2.2: Table 1 variables,
the eight process rates, the observed conversion rates r1..r13, Tables 2-3
parameters at about 15 C): content verified, read against the report. ASM1
itself is Henze et al. (1987), restated in [W3] (not read here).

Concentrations in g/m^3 (COD, N) and mol/m^3 (alkalinity); rates per day.

Streams: plain difflow streams, ``F_H2O`` (mol/s, it carries the volume:
``Q = F_H2O M_w / 1000``) and ``F_<state>`` with ``1 mol = 1 g`` (COD or N;
for ``S_ALK`` a real mole), so ``F_<state> = Q C``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream
from difflow_water.chemistry.seawater import MW_WATER

#: ASM1 state variables, in the BSM1 report's order (Table 1)
STATES = ["S_I", "S_S", "X_I", "X_S", "X_BH", "X_BA", "X_P", "S_O", "S_NO", "S_NH",
          "S_ND", "X_ND", "S_ALK"]
#: particulate states (settle; carried in the settler's solids)
PARTICULATE = ["X_I", "X_S", "X_BH", "X_BA", "X_P", "X_ND"]
IDX = {s: i for i, s in enumerate(STATES)}
PART_MASK = jnp.array([s in PARTICULATE for s in STATES])
#: the suspended-solids states (X_ND is part of X_S, not counted)
SS_MASK = jnp.array([s in ("X_I", "X_S", "X_BH", "X_BA", "X_P") for s in STATES])
COD_TO_SS = 0.75   # 1 / fr_COD-SS, fr_COD-SS = 4/3 [W5]
RHO = 1000.0       # kg/m^3: Q = F_H2O M_w / RHO
S_O_SAT = 8.0      # g/m^3, oxygen saturation [W5]


@dataclass(repr=False)
class ASM1Params(ParamsMixin):
    """ASM1 parameters, BSM1 values at about 15 C ([W5] Tables 2-3).

    Attributes:
        Y_A: Autotrophic yield (g COD / g N).
        Y_H: Heterotrophic yield (g COD / g COD).
        f_P: Fraction of biomass to particulate products (-).
        i_XB: N content of biomass (g N / g COD).
        i_XP: N content of particulate products (g N / g COD).
        mu_H: Heterotrophic maximum growth rate (1/d).
        K_S: Half-saturation, readily biodegradable substrate (g COD/m^3).
        K_OH: Oxygen half-saturation, heterotrophs (g O2/m^3).
        K_NO: Nitrate half-saturation (g N/m^3).
        b_H: Heterotrophic decay (1/d).
        eta_g: Anoxic growth correction (-).
        eta_h: Anoxic hydrolysis correction (-).
        k_h: Hydrolysis rate (g COD / (g COD d)).
        K_X: Hydrolysis half-saturation (g COD / g COD).
        mu_A: Autotrophic maximum growth rate (1/d).
        K_NH: Ammonia half-saturation (g N/m^3).
        b_A: Autotrophic decay (1/d).
        K_OA: Oxygen half-saturation, autotrophs (g O2/m^3).
        k_a: Ammonification rate (m^3 / (g COD d)).
    """

    Y_A: float | Array = 0.24
    Y_H: float | Array = 0.67
    f_P: float | Array = 0.08
    i_XB: float | Array = 0.08
    i_XP: float | Array = 0.06
    mu_H: float | Array = 4.0
    K_S: float | Array = 10.0
    K_OH: float | Array = 0.2
    K_NO: float | Array = 0.5
    b_H: float | Array = 0.3
    eta_g: float | Array = 0.8
    eta_h: float | Array = 0.8
    k_h: float | Array = 3.0
    K_X: float | Array = 0.1
    mu_A: float | Array = 0.5
    K_NH: float | Array = 1.0
    b_A: float | Array = 0.05
    K_OA: float | Array = 0.4
    k_a: float | Array = 0.05


def process_rates(Z: Array, p: ASM1Params) -> Array:
    """The eight process rates rho_1..rho_8 ([W5] Section 2.2.2), g/m^3/d."""
    S_I, S_S, X_I, X_S, X_BH, X_BA, X_P, S_O, S_NO, S_NH, S_ND, X_ND, S_ALK = Z
    mon_O_H = S_O / (p.K_OH + S_O)
    inh_O_H = p.K_OH / (p.K_OH + S_O)
    mon_NO = S_NO / (p.K_NO + S_NO)
    r1 = p.mu_H * S_S / (p.K_S + S_S) * mon_O_H * X_BH
    r2 = p.mu_H * S_S / (p.K_S + S_S) * inh_O_H * mon_NO * p.eta_g * X_BH
    r3 = p.mu_A * S_NH / (p.K_NH + S_NH) * S_O / (p.K_OA + S_O) * X_BA
    r4 = p.b_H * X_BH
    r5 = p.b_A * X_BA
    r6 = p.k_a * S_ND * X_BH
    hyd = p.k_h * (X_S / X_BH) / (p.K_X + X_S / X_BH) * (mon_O_H + p.eta_h * inh_O_H * mon_NO) * X_BH
    r7 = hyd
    r8 = hyd * X_ND / X_S
    return jnp.stack([r1, r2, r3, r4, r5, r6, r7, r8])


def conversion_rates(Z: Array, p: ASM1Params) -> Array:
    """Observed conversion rates r_1..r_13 ([W5] Section 2.2.3), per day."""
    r1, r2, r3, r4, r5, r6, r7, r8 = process_rates(Z, p)
    Y_H, Y_A, f_P, i_XB, i_XP = p.Y_H, p.Y_A, p.f_P, p.i_XB, p.i_XP
    zero = 0.0 * r1
    return jnp.stack([
        zero,                                                    # S_I
        -r1 / Y_H - r2 / Y_H + r7,                                # S_S
        zero,                                                    # X_I
        (1 - f_P) * r4 + (1 - f_P) * r5 - r7,                    # X_S
        r1 + r2 - r4,                                            # X_BH
        r3 - r5,                                                 # X_BA
        f_P * r4 + f_P * r5,                                     # X_P
        -(1 - Y_H) / Y_H * r1 - (4.57 - Y_A) / Y_A * r3,         # S_O
        -(1 - Y_H) / (2.86 * Y_H) * r2 + r3 / Y_A,               # S_NO
        -i_XB * r1 - i_XB * r2 - (i_XB + 1 / Y_A) * r3 + r6,     # S_NH
        -r6 + r8,                                                # S_ND
        (i_XB - f_P * i_XP) * r4 + (i_XB - f_P * i_XP) * r5 - r8,  # X_ND
        -i_XB / 14 * r1 + ((1 - Y_H) / (14 * 2.86 * Y_H) - i_XB / 14) * r2
        - (i_XB / 14 + 1 / (7 * Y_A)) * r3 + r6 / 14,           # S_ALK
    ])


def stream_to_state(stream: Stream) -> tuple[Array, Array]:
    """``(Q m^3/d, Z)`` of a stream (states absent from it are zero)."""
    Q = stream["F_H2O"] * MW_WATER / RHO * 86400.0
    Z = jnp.stack([jnp.asarray(stream.get(f"F_{s}", 0.0), dtype=jnp.float64) for s in STATES]) \
        * 86400.0 / Q
    return Q, Z


def state_to_stream(Q, Z, T=288.15, P=101325.0) -> Stream:
    """A stream of flow Q (m^3/d) and concentrations Z (g/m^3; S_ALK mol/m^3)."""
    flows = {"H2O": Q / 86400.0 * RHO / MW_WATER}
    flows.update({s: Q * Z[i] / 86400.0 for i, s in enumerate(STATES)})
    return make_stream(flows, T=T, P=P)


def suspended_solids(Z: Array) -> Array:
    """Total suspended solids (g SS/m^3): 0.75 (X_S + X_P + X_I + X_BH + X_BA) [W5]."""
    return COD_TO_SS * jnp.sum(jnp.where(SS_MASK, Z, 0.0), axis=-1)


#: BSM1 stabilization influent ([W5] Table 5), g/m^3, S_ALK mol/m^3; Q0 = 18446 m^3/d
BSM1_INFLUENT = {"S_I": 30.0, "S_S": 69.5, "X_I": 51.2, "X_S": 202.32, "X_BH": 28.17,
                 "S_NH": 31.56, "S_ND": 6.95, "X_ND": 10.59, "S_ALK": 7.0}
BSM1_Q0 = 18446.0


def bsm1_influent(T=288.15, P=101325.0) -> Stream:
    """The BSM1 constant (stabilization) influent as a stream ([W5] Table 5)."""
    Z = jnp.array([BSM1_INFLUENT.get(s, 0.0) for s in STATES])
    return state_to_stream(BSM1_Q0, Z, T, P)
