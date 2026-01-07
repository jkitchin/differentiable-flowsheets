"""Numerical constants for difflow.

This module centralizes numerical constants used throughout the codebase
for consistent behavior and easy tuning.

Categories:
- Regularization: Small values to prevent division by zero
- Bounds: Physical limits for numerical stability
- Blending: Smoothing widths for differentiable transitions
- Defaults: Common default values for optional parameters
"""

# =============================================================================
# Regularization Constants
# =============================================================================

# Small value to prevent division by zero in general calculations
EPS = 1e-10

# Minimum temperature difference for heat transfer calculations (K)
MIN_DELTA_T = 1e-6

# Minimum relative volatility difference from 1.0 for Fenske equation
# When α < 1 + MIN_ALPHA_DIFF, mixture is essentially non-separable
MIN_ALPHA_DIFF = 0.01


# =============================================================================
# Physical Bounds
# =============================================================================

# K-value bounds for VLE calculations (prevent numerical issues)
K_MIN = 1e-6
K_MAX = 1e6

# Maximum number of theoretical stages for distillation
# Beyond this, the column is economically infeasible
MAX_STAGES = 500.0

# Maximum Gilliland Y parameter
# When Y → 1, N → ∞; capping at 0.95 limits N to ~20*N_min
MAX_GILLILAND_Y = 0.95

# Temperature bounds for reactor calculations (K)
T_MIN = 250.0
T_MAX = 800.0


# =============================================================================
# Blending Widths (for smooth/differentiable transitions)
# =============================================================================

# LMTD blending: smooth transition when dT1 ≈ dT2
# When |ln(dT1/dT2)| < this value, blend toward arithmetic mean
LMTD_BLEND_WIDTH = 0.1

# Capacity ratio blending: smooth Cr = 1 edge case
# Used in effectiveness-NTU calculations
CR_BLEND_WIDTH = 0.05

# Phase boundary transition width for VLE
# Smaller = sharper transitions, larger = smoother gradients
PHASE_TRANSITION_WIDTH = 0.02

# Temperature profile scaling for distillation
# Column ΔT ≈ TEMP_SCALE_FACTOR * T_feed for typical systems
DEFAULT_TEMP_SCALE = 0.05


# =============================================================================
# Default Values
# =============================================================================

# Default molar concentration for liquid streams (mol/m³)
# Used when estimating volumetric flow from molar flow
DEFAULT_CONCENTRATION = 50.0

# Default heat capacity for liquids (J/mol/K)
# Approximate value for organic liquids and water
DEFAULT_CP = 75.0

# Default rate constant for initialization (1/s)
DEFAULT_RATE_CONSTANT = 0.1
