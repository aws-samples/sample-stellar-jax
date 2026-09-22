"""Solar calibration results for stellar-jax.

These are the RESULTS of running the solar calibration procedure
(optimize alpha_MLT and Y_init until log(L/L_sun)=0 and log(R/R_sun)=0
at solar age). They change only when the solver physics changes (rare).

Provenance: see the block comment below for full recalibration history.
"""

# Solar calibration results (unified solver: Henyey MLT + Thoul/Burgers diffusion
# + Potekhin 2021 electron conduction opacity).
# Converged to tol=1e-7 with adaptive-FD Newton-Raphson.
#
# Recalibration history (newest last):
#   1. Atmosphere radiative branch: uses the energy transport equation
#      (nabla_rad * dlnP/dlntau) consistently with the interior, instead of
#      the KS derivative nabla_ks which overestimated T in the radiative
#      photosphere by ~17% (dlnP/dlntau ≈ 0.5, not 1, with varying kappa).
#   2. Bicubic opacity: blended bicubic T/R + quadratic Z at solar Z (Z=0.0188,
#      tz=0.88 in [0.01,0.02] bracket → w_blend=1.0).
#   3. Chugunov (2007) screening: regime-aware screening for PP+CNO increases
#      nuclear rates ~3-5% at solar-core conditions → requires higher alpha and
#      Y0 to maintain L=L_sun, R=R_sun at solar age.
#   4. Concentration-gradient diffusion: AX*dlnC/dr term added to diffusion
#      velocity (TBL 1994 eq. 21; MESA diffusion_support.f90:700-704). Slightly
#      reduces effective settling → small shift (d_alpha ≈ -3e-4, dY0 ≈ +2e-6).
#   5. Conservative zone-mass weighting (sum=1.0, MESA-faithful midpoint-boundary
#      cells). The legacy concat([diff, diff[-1:]]) summed to 1.015 — removing
#      the ~1.5% mass-weighting error shifts calibration by d_alpha ≈ +0.025,
#      dY0 ≈ -7e-5 (within expected sensitivity).
# 6. MESA-matched constants: G 6.67259e-8→6.67430e-8,
#      Msun 1.989e33→mu_sun/G, Lsun 3.826e33→3.828e33, Rsun 6.9599e10→6.957e10.
#      Higher G → stronger gravity → slightly higher alpha to maintain L=L☉.
#      Shift: d_alpha ≈ +0.004, dY0 ≈ -1.4e-5 (within expected sensitivity).
# 7. CNO rate + catalyst correction: rate 4.10e27→8.67e27 (MESA NACRE),
#      catalyst fallback 0.69*Z→0.251*Z (ZAMS CN+ON-eq). The rate constant
#      was 2× low vs MESA's default rate_n14pg_nacre (ratelib.f90:1506). The old
#      0.69*Z fallback included O16 as catalyst, overestimating MESA's actual
#      X_N14 by ~2.8× at ZAMS. The 0.251*Z matches MESA's ZAMS N14 (after PMS
#      CN+ON equilibration) to <1%. Net effect on solar calibration: the evolution
#      code uses tracked N14 (not the fallback), but the shooting solver and
#      Jacobian central BC use the fallback, so both the rate constant change and
#      the catalyst correction shift the solar model. At 1 Msun (~5% CNO),
#      net alpha shift is +0.001.
# Newton-iterated solar_residual to converge to |logL|,|logR| < 1e-8.
#
# WHY alpha > 2.11 (the KS-T(tau) literature reference, Salaris & Cassisi 2015):
#
# ALPHA_SOLAR (2.302) exceeds the KS reference by +0.192. This is explained
# by documented MESA divergences (measured; itemized below):
#
#   1. Deep atmosphere integration (CONSTRAINT, +0.144):
#      We integrate BOTH hydrostatic + energy transport (nabla * dlnP/dlntau,
#      nabla from MLT) from tau=1e-4 to tau=100 (structure.py:atmosphere_bc).
#      MESA integrates ONLY hydrostatic, T from analytic KS, tau_base=0.3122
#      (atm_t_tau_varying.f90:444; atm_t_tau_relations.f90:57).
#      WHY CONSTRAINT: end-to-end differentiability requires alpha to affect
#      the BC continuously (MESA's alpha-independent BC gives dBC/dalpha=0).
#      MEASURED: P 4.1x higher, T 1.9x higher at tau=100 vs tau=0.3122.
#
#   2. Bicubic + quadratic-Z opacity (BETTER, +0.060):
#      MESA uses linear Z (kap.defaults:268: cubic_interpolation_in_Z=.false.).
#      MEASURED: d^2(logkappa)/dZ^2 = -0.095 (concave-down -> linear
#      underestimates; Rogers & Iglesias 1996). At CZ base: +5% kappa.
#
# 3. Conservative zone-mass weighting (+0.025):
#      Fixing the legacy ~1.5% mass-weight error shifts calibration by +0.025.
#
# 4. CNO rate + catalyst correction (net +0.001):
#      Rate 4.10e27→8.67e27 (MESA NACRE), catalyst 0.69*Z→0.251*Z (ZAMS CN+ON-eq).
#      The evolution code uses tracked N14 (not the fallback), so the solar shift
#      comes from the rate constant change + the ZAMS construction fallback change.
#      At 1 Msun (~5% CNO): net alpha shifts +0.001.
#      MESA ref: ratelib.f90:1506, net_approx21.f90:1116.
#
# 5. Baryon-conserving burn: deposit burned H into He (ΔY = −ΔX),
#      matching MESA net_eval.f90:274. Previously Y was only modified by
#      diffusion; now burn_composition returns (X_new, Y_new) with
#      Y_new = Y_old + actual_dX. Shift: d_alpha ≈ +0.0013, dY0 ≈ -7.9e-5
#      (physical: increased Y in core from burn deposition → lower μ).
#
# 6. Smooth CZ-base diffusion taper (-0.040): replace hard 0/1
#      convective-zone mask with smooth cosine taper matching MESA's
#      limit_coeffs_face (diffusion_procs.f90:get_limit_coeffs L985-1039).
#      The smooth transition reduces the He settling drain at the CZ base
#      compared to the old replacement-upwind stencil, keeping slightly more
#      He in the envelope → higher μ → lower α to match L=L☉.
#      CONSTRAINT: we use Schwarzschild margin (nad-nrad) as proxy for MESA's
# phase(k); taper width Δ=0.02 validated by.
#      Shift: d_alpha ≈ -0.040, dY0 ≈ +1.6e-4.
#
# BUDGET: 2.11 + 0.144 + 0.060 + 0.025 + 0.001 + 0.001 - 0.040 = 2.301.
# Measured: 2.302. Residual: ~0.001.
# Guard: test_solar_alpha_physical_range (ceiling 2.35).
# Newton-iterated solar_residual to converge to |logL|,|logR| < 1e-8.
ALPHA_SOLAR = 2.30215985
Y0_SOLAR = 0.26692026

# Model S comparison:
# The Model S comparison uses the SINGLE calibration (ALPHA_SOLAR, Y0_SOLAR)
# at Z=0.0188 (present-day photospheric metallicity).
#
# DESIGN DECISION: One could instead calibrate at Z=0.0196 (Model S's initial Z)
# to achieve ~1.2% density agreement. We deliberately do NOT do this because:
# (1) It produces alpha=2.367 > 2.35 (the literature ceiling;).
#   (2) A second alpha used by one test is a code smell — it already caused the
# circular density-fit bug.
#   (3) The Z=0.0188 calibration is the SINGLE solar calibration used by all
#       other comparisons; consistency matters more than a tighter density match
#       that requires a bespoke alpha.
# The density residual from the Z-mismatch is the honest cost of this choice.
#
# Z-MISMATCH CAVEAT: Model S was computed with Z_initial=0.0196 (GN93 mixture;
# Christensen-Dalsgaard et al. 1996, Table 1). Our Z=0.0188 is lower by 4%.
# This introduces a ~5% radiative-zone opacity deficit vs Model S's interior,
# which appears as a ~2-4% density residual in the comparison. This is an
# EXPECTED physics difference from the Z mismatch — not a bug and not a reason
# to carry a second calibration. A single honestly-calibrated alpha < 2.35 (the
# literature ceiling) is correct; the density residual is a documented caveat.
# Closing M2b (<1% density) requires improved microphysics (diffusion
# coefficients, CZ-base opacity) —.
# Reference: Christensen-Dalsgaard et al. (1996); Basu & Antia (2004).
