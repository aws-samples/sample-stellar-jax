"""Named physical-floor constants — modeling choices that share syntax with
underflow guards but whose VALUE matters.

Pure underflow/div-by-zero guards (``x + 1e-30``, ``jnp.maximum(x, 1e-30)``)
are self-evident and stay inline.  The constants below are **different**: each
floor's magnitude is a deliberate modeling decision (it can bind physically or
change a derived quantity), so it gets a name and a rationale.

Placement follows the ``mesh/mesh_params.py`` precedent: subsystem-specific
thresholds in a dedicated module, SCREAMING_SNAKE_CASE, one-line rationale,
MESA grounding.

Convention (documented in ``docs/dev/coding-standards.md``):
  - Bare tiny literals in ``maximum()``/``clip()``/``+eps`` = underflow guards
    (magnitude immaterial, leave inline).
  - Any value that MATTERS (modeling choice / can bind physically) gets a NAMED
    constant imported from here.
  - **Never** interconvert ``x + eps`` / ``maximum(x, eps)`` / ``where(...)``
    — they have different boundary derivatives in AD.
"""

# === ∇_ad (adiabatic gradient) floor ===

# Floor on ∇_ad when it appears as a cp denominator:
#   cp = P·δ / (T·ρ·max(∇_ad, NAD_CP_FLOOR))
# CONSTRAINT (JAX differentiability): MESA does NOT floor grada at runtime
# (turb/private/mlt.f90, eos/private/eosdt_eval.f90 — grada flows unfloored).
# We floor it to keep cp finite for AD.  Value 0.05 is the lower bound of the
# CMS offline table builder's [0.1, 0.5] clamp (eosCMS_builder/src/cms_mixing.f90:244),
# well below any physically realized ∇_ad (ideal mono-atomic gas: 0.4; real
# stellar interior ≳0.15).
NAD_CP_FLOOR = 0.05

# Upper bound on EOS-interpolated ∇_ad (physical range ceiling).
# Ideal mono-atomic gas gives 0.4; 0.45 allows small deviations from ionisation
# and radiation effects while capping interpolation artifacts.
# Reference: MESA eosCMS_builder clamps to [0.1, 0.5].
NAD_UPPER = 0.45


# === Gas-pressure fraction floor ===

# P_gas = max(P − P_rad, PGAS_FRAC_FLOOR * P)
# Floors gas pressure at 0.1% of total pressure in radiation-dominated envelopes,
# keeping log₁₀(P_gas) and EOS lookups finite.  The 0.1% level is well below any
# stellar-interior β (even massive-star envelopes have β ≳ 0.01), so the floor
# never binds in physically realized models.
PGAS_FRAC_FLOOR = 1e-3


# === Nuclear-network abundance floors ===

# Hydrogen mass-fraction floor for pp-chain electron screening (ψ_pp):
#   psipp ∝ 1/max(X, X_MIN_PP) − 1.
# Keeps 1/X finite (and its AD gradient bounded) as H is exhausted in the core.
# Reference: Caughlan & Fowler 1988, ADNDT 40, 283 (ψ_pp correction).
X_MIN_PP = 1e-3


# === Structural hard limits (MESA delta_lgL / delta_lgTeff analog) ===

# MESA timestep.f90:1961-1987 (check_delta_lgL) compares |Δlog10(L_surf/L_surf_old)|
# against delta_lgL_hard_limit.  MESA defaults:
#   delta_lgL_limit = 0.10 (soft dt adjuster), delta_lgL_hard_limit = -1 (disabled)
#   (controls.defaults:10674)
# CONSTRAINT (JAX): we use hard rejection at a conservative 0.20 dex (2× MESA's
# soft limit) because stellar-jax lacks the full soft-limiter proportional feedback.
# |ΔlogL| > 0.20 means > 58% luminosity change in one step.
DELTA_LGL_HARD_LIMIT = 0.20   # |Δlog L| hard limit [dex]

# MESA timestep.f90:1265-1273 (check_dlgTeff_change).  MESA defaults:
#   delta_lgTeff_limit = 0.01 (soft), delta_lgTeff_hard_limit = -1 (disabled)
#   (controls.defaults:10657)
# CONSTRAINT (JAX): 0.05 dex (5× MESA's soft limit).
# |ΔlogTeff| > 0.05 means > 12% effective temperature change in a single step.
DELTA_LGTE_HARD_LIMIT = 0.05  # |Δlog Teff| hard limit [dex]
