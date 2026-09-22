"""NumPy mesh kernels for the adaptive forward pass.

Mesh constants, equidistribution, conservative remap, and remesh logic.
These are NumPy (not JAX) — they run in the Python outer loop, outside
lax.scan.

Moved from adaptive_forward.py into evolution/adaptive/ (#1115).

MESA references:
  - mesh_functions.f90:307-321 (do1_xa_function): gval = weight*log10(xa+param)
  - mesh_plan.f90:1081-1246 (pick1_dq): equidistribution on delta_gval_max
  - mesh_adjust.f90:do_mesh_adjust: piecewise-linear remap of state variables
  - evolve.f90:1882-1886: remesh in prepare_for_new_step, before solve
  - Paxton et al. 2011 ApJS 192, 3 §7; Paxton et al. 2013 ApJS 208, 4 §6.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# MESA mesh parameters (mesh_functions.f90:307-321, controls.defaults)
# =============================================================================
# MESA defaults for the xa_function (He4): weight=30, param=0.01.
# We use hydrogen (X) since the H-burning shell drives the mesh adaptation.
MESH_WEIGHT = 30.0
MESH_PARAM = 0.01
# MESA T_function1 (mesh_functions.f90:201-203, controls.defaults:6102):
#   gval += T_function1_weight * log10(T)
# Weight=110 concentrates zones where the temperature gradient is steep —
# this is the ENVELOPE (surface/ionization zone), where temperature drops
# from ~10^7 K (interior) to ~10^4 K (surface) over a small mass range.
# Without this, the envelope gets poor resolution → l≥1 modes are inaccurate.
MESH_T_FUNCTION1_WEIGHT = 110.0
# MESA P_function (mesh_functions.f90:196-198, controls.defaults:6089):
#   gval += P_function_weight * log10(P)
# Weight=40 concentrates where pressure gradient is steep.
MESH_P_FUNCTION_WEIGHT = 40.0
# Minimum cell width (fraction of total mass). MESA: min_dq (controls.defaults).
MESH_MIN_DQ = 1e-5
# Maximum allowed gval change per cell — controls resolution.
# Smaller → more zones at sharp features. MESA adjusts this per-zone;
# we use a fixed value tuned to give ~5-10 zones across the H-shell.
MESH_DELTA_GVAL_MAX = 0.5
# Smoothing iterations for mesh density.
# MESA smooths delta_gval_max (the allowed change per cell) with 3 iters of
# 1/3-1/3-1/3 (adjust_mesh_support.f90:157). Our equidistribution smooths the
# density function directly — a different approach (CONSTRAINT: global
# equidistribution vs MESA's split/merge). One pass removes single-cell noise
# while preserving the shell-density peak. Issue excessive smoothing (3
# iters) blurs the thin H-burning shell (dq~0.001-0.003) into surrounding
# zones, diluting the concentration signal and causing the logL≈1.66 plateau.
MESH_SMOOTH_ITERS = 1

# Mesh concentration factors (peak/floor density ratio).
# Issue the H-burning shell on the RGB has width dq~0.001-0.003.
# With max_concentration=100, only ~87-144 of N=600 zones reach the shell,
# causing the logL≈1.66 plateau. Increasing to 500 post-TAMS gives ~176-370
# shell zones. On the MS (no thin shell), 100 is sufficient and avoids
# creating unnecessarily thin cells that can cause solver instabilities.
# MESA ref: mesh_functions.f90:307-321, controls.defaults:5790.
MAX_CONCENTRATION_MS = 100.0   # MS: no thin shell, moderate concentration
MAX_CONCENTRATION_RGB = 500.0  # post-TAMS: thin H-shell needs aggressive conc


# =============================================================================
# Mesh functions: compute gval, equidistribute, remap
# =============================================================================

def compute_gval_numpy(X_profile, q_mesh, y_henyey=None):
    """Compute mesh function gval as the sum of MESA's default mesh functions.

    MESA uses three mesh functions simultaneously (controls.defaults):
      xa_function  = 30  * log10(X + 0.01)         [mesh_functions.f90:317-319]
      T_function1  = 110 * log10(T)                 [mesh_functions.f90:201-203]
      P_function   = 40  * log10(P)                 [mesh_functions.f90:196-198]

    The xa term concentrates at composition transitions (H-burning shell).
    The T term concentrates where the temperature gradient is steep — this is
    the ENVELOPE, where T drops from ~10^7 K (interior) to ~10^4 K (surface).
    The P term concentrates where the pressure gradient is steep.

    Without the T and P terms, the envelope gets poor resolution and l≥1 modes
    (which propagate in the g-mode cavity and are sensitive to near-surface
    structure) cannot be accurately computed.

    CONSTRAINT — gval combination method (sum vs MESA's per-function min):
      MESA treats each mesh function as a SEPARATE dimension and takes the
      per-function minimum cell size: for each new cell, it computes dq
      independently for each gval function via pick1_dq, then takes
      min(dq_xa, dq_T, dq_P) — the most demanding function controls
      cell size at each location (mesh_plan.f90:1068-1072, minloc(nxt_dqs)).

      We SUM the three functions into one scalar and equidistribute on that
      sum. This is a simplification forced by our 1D equidistribution
      algorithm (equidistribute_mesh_numpy operates on a single gval array).

      Effect on the MS: the T+P gradients concentrate zones in the
      envelope (where l≥1 modes propagate) while xa concentrates at the
      shell. The sum is monotone-preserving and works well because both
      regions need resolution on the MS.

      Effect on the RGB: the T+P gradients (~938 combined range) DOMINATE
      over xa (~46 range), leaving only ~5% of the gval budget for shell
      resolution. This is catastrophic: the H-burning shell needs tight
      resolution for the Henyey solver to converge and the star to ascend.
      MESA's min(dq) avoids this — each function gets independent resolution.

      Mitigation: the caller (_step_remesh) passes y_henyey=None post-TAMS,
      disabling T+P terms and restoring composition-only equidistribution
      on the RGB. This is adequate because (a) the envelope-resolution use
      case (l≥1 seismic modes) is an MS concern, and (b) the composition-only
      gval was already validated for RGB ascent on main.

      A per-function-min implementation would require multi-dimensional
      equidistribution or iterative refinement — a follow-up improvement
      that would enable T+P resolution throughout the full evolution.

    CONSTRAINT — xa_function species:
      MESA uses He4 (xa_function_species(1) = 'he4', controls.defaults:6283).
      We use hydrogen (X). Both concentrate at the H-burning shell where
      X drops and He4 rises — same physics target. Using X avoids carrying
      a separate He4 profile in the numpy forward pass.

    MESA reference:
      mesh_functions.f90:191-203 (E/P/T function computation)
      mesh_plan.f90:1060-1072 (pick_next_dq: per-function dq then minloc)
      controls.defaults:6089 (P_function_weight=40)
      controls.defaults:6102 (T_function1_weight=110)
      controls.defaults:6283-6287 (xa_function_species(1)='he4', weight=30)

    Parameters
    ----------
    X_profile : ndarray (N,) — hydrogen mass fraction on the mesh nodes
    q_mesh : ndarray (N,) — mass coordinate mesh nodes (for diagnostics)
    y_henyey : ndarray (N, 4) or None — Henyey state [ln_r, ln_P, ln_T, ell].
        When provided, the T_function1 and P_function terms are added to the
        gval using the structure from the last converged Henyey step. When None
        (at ZAMS before the first step), only the xa_function is used.

    Returns
    -------
    gval : ndarray (N,) — mesh function values at each node
    """
    # xa_function: weight=30, param=0.01 (composition term)
    gval = MESH_WEIGHT * np.log10(np.maximum(X_profile, 0.0) + MESH_PARAM)

    if y_henyey is not None:
        N = len(X_profile)
        N_y = y_henyey.shape[0]

        # Henyey state has N zones (matching mesh), but the state may have
        # been defined on the structure mesh (N+1 faces → N zone values).
        # Use as many points as match; interpolate if needed.
        if N_y >= N:
            ln_T = y_henyey[:N, 2]
            ln_P = y_henyey[:N, 1]
        else:
            # Shouldn't happen in practice, but handle gracefully
            ln_T = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, N_y),
                             y_henyey[:, 2])
            ln_P = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, N_y),
                             y_henyey[:, 1])

        # T_function1: gval += 110 * log10(T) = 110 * ln(T) / ln(10)
        # MESA: mesh_functions.f90:201-203
        logT = ln_T / np.log(10.0)
        gval += MESH_T_FUNCTION1_WEIGHT * logT

        # P_function: gval += 40 * log10(P) = 40 * ln(P) / ln(10)
        # MESA: mesh_functions.f90:196-198
        logP = ln_P / np.log(10.0)
        gval += MESH_P_FUNCTION_WEIGHT * logP

    return gval


def equidistribute_mesh_numpy(gval_old, q_old, N_new, mesh_config=None):
    """Equidistribute N_new nodes so |Δgval| is constant per cell.

    Parameters
    ----------
    mesh_config : dict or None — {min_dq, max_concentration, smooth_iters}.
        Defaults to module-level constants when None or keys absent.

    MESA analog: mesh_plan.f90:pick1_dq — each new cell spans at most
    delta_gval_max of the mesh function. Our fixed-N version distributes
    nodes so that each cell spans equal gval change.

    O2 mutation hook: if _STELLAR_MUTATION_COMP_EQUIDIST_DISABLED=1 is set,
    return a uniform grid (no equidistribution). This env-var signal survives
    reimports and subprocess boundaries — more robust than monkeypatch alone.

    Parameters
    ----------
    gval_old : ndarray (N_old,) — mesh function on old grid
    q_old : ndarray (N_old,) — mass coordinate positions [0, 1]
    N_new : int — number of nodes in new grid
    min_dq : float — minimum cell width
    max_concentration : float or None — maximum peak/floor density ratio.
        Controls how aggressively zones concentrate at composition features.
        None → use module default (MAX_CONCENTRATION_RGB = 500).
        On the MS (no H-shell), 100 is sufficient; on the RGB where the
        H-burning shell narrows to dq~0.001-0.003, 500 is needed to resolve
        eps_nuc. MESA ref: mesh_functions.f90:307-321, controls.defaults:5790.
    smooth_iters : int or None — number of density smoothing passes.
        None → use module default (MESH_SMOOTH_ITERS = 1).
        On the MS, 3 passes (matching MESA adjust_mesh_support.f90:157 default)
        smooths the density without penalty (no thin shell to preserve). Post-
        TAMS, 1 pass preserves the sharp shell-density peak that 3 passes would
        blur into neighbors. MESA smooths delta_gval_max (the
        allowed change per cell), not density directly — our direct-density
        smoothing with fewer passes approximates the same effect (CONSTRAINT).

    Returns
    -------
    q_new : ndarray (N_new,) — new equidistributed mesh positions in [0, 1]
    """
    if mesh_config is None:
        mesh_config = {}
    min_dq = mesh_config.get('min_dq', MESH_MIN_DQ)
    max_concentration = mesh_config.get('max_concentration', None)
    smooth_iters = mesh_config.get('smooth_iters', None)

    # O2 mutation hook: env-var signal disables equidistribution (returns uniform).
    # This is MORE ROBUST than monkeypatch alone — survives reimport/cache issues.
    import os as _os
    if _os.environ.get("_STELLAR_MUTATION_COMP_EQUIDIST_DISABLED") == "1":
        return np.linspace(0.0, 1.0, N_new)

    # Cumulative integral of |d(gval)/dq| — the equidistribution function
    dq = np.diff(q_old)
    dgval = np.abs(np.diff(gval_old))
    # Smooth the mesh density to prevent abrupt cell-size jumps
    # (MESA: mesh_delta_coeff_factor_smooth_iters, adjust_mesh_support.f90)
    density = dgval / np.maximum(dq, 1e-30)
    n_smooth = smooth_iters if smooth_iters is not None else MESH_SMOOTH_ITERS
    for _ in range(n_smooth):
        density_padded = np.concatenate([[density[0]], density, [density[-1]]])
        density = 0.25 * density_padded[:-2] + 0.5 * density_padded[1:-1] + 0.25 * density_padded[2:]
    # Floor: MESA uses max_dq=0.01 (controls.defaults:5877) to guarantee no
    # cell is wider than 1% of total mass. With equidistribution on density,
    # the equivalent is a floor that guarantees flat regions keep at least
    # ~N_new * max_dq fraction of the total zones. We implement this as:
    #   floor = peak_density / max_cell_ratio
    # where max_cell_ratio controls the maximum concentration factor.
    # MESA mesh_max_allowed_ratio=2.5 is an ADJACENT-cell constraint
    # (controls.defaults:5790); our max_concentration is a GLOBAL peak/floor
    # ratio — different mechanism (CONSTRAINT: fixed N_zones for JAX shape
    # stability), same goal (concentrate zones at composition features).
    #
    # Issue the H-burning shell on the RGB has width dq~0.001-0.003.
    # With max_concentration=100, only ~87-144 of N=600 zones reach the shell
    # — insufficient to resolve eps_nuc, causing the logL≈1.66 plateau.
    # Zone sweep confirms: N=600 (conc=100) → logL 1.66; N=2400 → logL 2.234.
    # Increasing to 500 gives ~176-370 shell zones, matching the effective
    # resolution MESA achieves with 1363-1642 total zones on the 1.5 M☉ RGB.
    # MESA ref: mesh_functions.f90:307-321 (do1_xa_function, weight=30).
    if max_concentration is None:
        max_concentration = MAX_CONCENTRATION_RGB
    peak_density = np.max(density)
    floor_value = max(peak_density / max_concentration, 1.0)
    density = np.maximum(density, floor_value)
    # Cumulative distribution
    G_vals = np.concatenate([[0.0], np.cumsum(density * dq)])
    # Equidistribute: place N_new nodes at equal G intervals
    G_target = np.linspace(0.0, G_vals[-1], N_new)
    q_new = np.interp(G_target, G_vals, q_old)
    # Enforce boundary conditions
    q_new[0] = 0.0
    q_new[-1] = 1.0
    # Enforce minimum cell width
    for i in range(1, N_new):
        if q_new[i] - q_new[i-1] < min_dq:
            q_new[i] = q_new[i-1] + min_dq
    # Re-normalize to [0, 1] if we pushed past the boundary
    if q_new[-1] > 1.0:
        q_new = q_new / q_new[-1]
    q_new[0] = 0.0
    q_new[-1] = 1.0
    return q_new


def _compute_slopes_mesa(profile, dq, q_nodes=None):
    """Compute limited linear slopes per cell (MESA's get1_lpp, order=1).

    MESA reference: mesh_adjust.f90:1720-1807 (get1_lpp with quad=.false.).

    For each interior cell k, compute left and right finite-difference slopes:
      sm1 = (profile[k-1] - profile[k]) / distance(k-1, k)
      s00 = (profile[k] - profile[k+1]) / distance(k, k+1)
    where distance is the spacing between node positions.

    CONSTRAINT: In MESA, cell values live at cell midpoints and `(dq[k-1]+dq[k])/2`
    IS the inter-cell-center distance. In our convention, values live at node
    positions q_nodes[k], so we use `q_nodes[k]-q_nodes[k-1]` as the distance.
    When q_nodes is None, falls back to MESA's formula (valid for uniform grids).

    If sm1*s00 <= 0 (local extremum): slope = 0 (piecewise constant).
    Otherwise: slope = average (sm1 + s00) / 2, with monotonicity fallback
    to minmod if the reconstruction overshoots at cell boundaries.

    Parameters
    ----------
    profile : ndarray (N,) — values at node positions
    dq : ndarray (N,) — cell widths (for boundary overshoot check)
    q_nodes : ndarray (N,) or None — node positions

    Returns
    -------
    c1 : ndarray (N,) — slope per unit q at each node
    """
    N = len(profile)
    c1 = np.zeros(N)

    for k in range(1, N - 1):
        # Distance between adjacent nodes
        if q_nodes is not None:
            dist_left = q_nodes[k] - q_nodes[k - 1]
            dist_right = q_nodes[k + 1] - q_nodes[k]
        else:
            dist_left = (dq[k - 1] + dq[k]) / 2
            dist_right = (dq[k] + dq[k + 1]) / 2

        # Slopes (positive when profile decreases with increasing k/q)
        sm1 = (profile[k - 1] - profile[k]) / max(dist_left, 1e-30)
        s00 = (profile[k] - profile[k + 1]) / max(dist_right, 1e-30)

        # At local extremum: flat
        if sm1 * s00 <= 0:
            continue

        # Average slope (MESA: "use average to smooth abundance transitions")
        c1[k] = (sm1 + s00) / 2

        # Monotonicity check at cell edges:
        # MESA (get1_lpp:1780-1790) uses dq(k)/2 from cell center to boundary.
        # Our node positions differ from cell midpoints on non-uniform grids
        # (actual distances: (q[k]-q[k-1])/2 outer, (q[k+1]-q[k])/2 inner).
        # Using dq[k]/2 is conservative: may trigger minmod fallback slightly
        # more/less often than exact distances, but does NOT affect linear-
        # profile exactness (those never trigger the fallback).
        dqhalf = dq[k] / 2
        vbdy_outer = profile[k] + c1[k] * dqhalf
        vbdy_inner = profile[k] - c1[k] * dqhalf

        overshoot_outer = (profile[k - 1] - vbdy_outer) * (vbdy_outer - profile[k]) < 0
        overshoot_inner = (profile[k] - vbdy_inner) * (vbdy_inner - profile[k + 1]) < 0

        if overshoot_outer or overshoot_inner:
            # Fall back to minmod
            if abs(sm1) <= abs(s00):
                c1[k] = sm1
            else:
                c1[k] = s00

    return c1


def conservative_remap_numpy(profile_old, q_old, q_new):
    """Mass-conserving order-1 (linear) remap from old grid to new grid.

    MESA reference: mesh_adjust.f90:do_mesh_adjust (lines 259-288 +
    get1_lpp lines 1720-1807 + get_xq_integral lines 1810-1924 +
    do_xa lines 1203-1324).

    Algorithm:
      1. Compute cell widths (dq) from cell boundaries (midpoints of nodes).
      2. Linear reconstruction: c0 = profile_old[k], c1 = limited slope.
         (MESA get1_lpp: average of left/right slopes, with monotonicity
         fallback to minmod, flat at boundaries/extrema.)
      3. For each new cell, integrate the linear profile across overlapping
         old cells (MESA get_xq_integral): avg = (f(left) + f(right)) / 2.
      4. Clip to [0, 1] and conserve total mass exactly by rescaling.

    The total species mass ∫ X dq is conserved to machine precision.

    Parameters
    ----------
    profile_old : ndarray (N_old,) — species mass fraction on old grid
    q_old : ndarray (N_old,) — old mesh node positions [0, 1]
    q_new : ndarray (N_new,) — new mesh node positions [0, 1]

    Returns
    -------
    profile_new : ndarray (N_new,) — remapped profile on new grid
    """
    N_old = len(q_old)
    N_new = len(q_new)

    # Cell boundaries: midpoints between nodes, pinned at 0 and 1
    mid_old = 0.5 * (q_old[:-1] + q_old[1:])
    bnd_old = np.concatenate([[0.0], mid_old, [1.0]])
    dq_old = np.diff(bnd_old)  # width of each old cell

    mid_new = 0.5 * (q_new[:-1] + q_new[1:])
    bnd_new = np.concatenate([[0.0], mid_new, [1.0]])
    dq_new = np.diff(bnd_new)  # width of each new cell

    # Cell midpoints in q-space (centers of mass coordinate intervals)
    # NOTE: In our convention, profile values are stored at NODE positions
    # q_old[k], not at cell midpoints. The linear reconstruction is centered
    # at the node: f(q) = c0[k] + c1[k] * (q_old[k] - q).
    # This differs from MESA where xa is the cell average (at cell midpoint).
    # CONSTRAINT: JAX/our mesh convention stores point values at nodes.

    # --- Step 1: Linear reconstruction (MESA get1_lpp, order=1) ---
    c0 = profile_old.copy()  # cell averages
    c1 = _compute_slopes_mesa(profile_old, dq_old, q_nodes=q_old)

    # --- Step 2: Conservative integration (MESA get_xq_integral, order=1) ---
    # For each new cell, integrate f(q) = c0[k] + c1[k] * (q - q_mid[k])
    # over the overlap with each old cell.
    profile_new = np.zeros(N_new)
    total_mass_old = np.sum(c0 * dq_old)

    # Use a pointer to track which old cell we're in
    k_old = 0
    for i in range(N_new):
        new_left = bnd_new[i]
        new_right = bnd_new[i + 1]
        integral = 0.0

        # Move pointer to the first overlapping old cell
        while k_old > 0 and bnd_old[k_old] > new_left:
            k_old -= 1

        k = k_old
        while k < N_old:
            old_left = bnd_old[k]
            old_right = bnd_old[k + 1]

            # Overlap region
            overlap_left = max(new_left, old_left)
            overlap_right = min(new_right, old_right)

            if overlap_right <= overlap_left:
                if old_left >= new_right:
                    break
                k += 1
                continue

            dq1 = overlap_right - overlap_left

            # Linear interpolant: f(q) = c0[k] + c1[k] * (q_old[k] - q)
            # where q_old[k] is the node position (where profile = c0[k]).
            # Average over [overlap_left, overlap_right]:
            #   avg = c0[k] + c1[k] * (q_old[k] - overlap_center)
            # MESA get_xq_integral lines 1884-1889 uses the same form with
            # cell midpoint as reference. We use node position because our
            # profile values are point values at nodes (CONSTRAINT: mesh convention).
            overlap_center = 0.5 * (overlap_left + overlap_right)
            avg_value = c0[k] + c1[k] * (q_old[k] - overlap_center)

            integral += dq1 * avg_value

            if old_right >= new_right:
                break
            k += 1

        # Update pointer for next new cell
        k_old = max(k - 1, 0) if k > 0 else 0

        profile_new[i] = integral / max(dq_new[i], 1e-30)

    # --- Step 3: Clip and conserve mass exactly ---
    # MESA (do_xa lines 1303-1322): clips to [0,1], normalizes sum to 1.
    # For single-species remap, we ensure non-negativity and rescale to
    # conserve total mass exactly (any clipping error is redistributed).
    profile_new = np.clip(profile_new, 0.0, 1.0)
    total_mass_new = np.sum(profile_new * dq_new)
    if total_mass_new > 0 and total_mass_old > 0:
        profile_new *= total_mass_old / total_mass_new

    return profile_new


def remap_henyey_state_numpy(y_old, q_old_struct, q_new_struct):
    """Remap the Henyey state vector from old structure mesh to new.

    Interpolates ln_r, ln_P, ln_T, ell at new q positions.
    Uses linear interpolation (the state is smooth between cells).

    Parameters
    ----------
    y_old : ndarray (N_old, 4) — [ln_r, ln_P, ln_T, ell] on old mesh
    q_old_struct : ndarray (N_old+1,) — old structure mesh (N+1 face positions)
    q_new_struct : ndarray (N_new+1,) — new structure mesh

    Returns
    -------
    y_new : ndarray (N_new, 4) — interpolated state on new mesh
    """
    N_old = y_old.shape[0]
    N_new = q_new_struct.shape[0] - 1

    # Old interface positions (where y is defined)
    q_old_iface = q_old_struct[:N_old]

    # New interface positions
    q_new_iface = q_new_struct[:N_new]

    y_new = np.zeros((N_new, 4))
    for col in range(4):
        y_new[:, col] = np.interp(q_new_iface, q_old_iface, y_old[:, col])
    return y_new


# =============================================================================
# Unified mesh generation
# =============================================================================

def make_unified_mesh(X_profile, N_zones, q_current=None,
                      mesh_config=None, y_henyey=None):
    """Generate a unified mesh adapted to the composition AND structure profiles.

    The mesh serves BOTH structure and composition: composition lives on the
    same cells that solve the structure equations. This eliminates the
    interpolation error of the two-mesh architecture.

    When y_henyey is provided (all steps after ZAMS), the gval includes MESA's
    T_function1 and P_function terms in addition to the xa_function, providing
    envelope resolution for l≥1 oscillation modes (issue #623).

    MESA reference: evolve.f90:1882-1886 — remesh in prepare_for_new_step,
    BEFORE the structure solve.

    Parameters
    ----------
    X_profile : ndarray (N_current,) — hydrogen profile on current mesh
    N_zones : int — target number of zones (may be the same as current)
    q_current : ndarray (N_current,) or None — current mesh positions
    max_concentration : float or None — peak/floor density ratio (see
        equidistribute_mesh_numpy). None → module default.
    smooth_iters : int or None — density smoothing passes (see
        equidistribute_mesh_numpy). None → module default.
    y_henyey : ndarray (N_current, 4) or None — Henyey state [ln_r, ln_P, ln_T, ell].
        When provided, T and P mesh functions are included in the gval for
        envelope resolution. None at ZAMS (before first step).

    Returns
    -------
    q_new : ndarray (N_zones,) — new cell-center positions in [0, 1]
    q_struct : ndarray (N_zones+1,) — structure mesh (face positions)
    """
    if q_current is None:
        # Initial static mesh (same as mesh.initial_lagrangian_mesh but numpy)
        xi = np.linspace(0.0, 1.0, N_zones + 1)
        q_struct = 0.30 * xi**3 + 0.15 * (1.0 - (1.0 - xi)**3) + 0.55 * xi
        q_struct[0] = 0.0
        q_struct[-1] = 1.0
        # Cell centers
        q_cells = 0.5 * (q_struct[:-1] + q_struct[1:])
        return q_cells, q_struct

    # Compute gval on current mesh (xa_function + T_function1 + P_function)
    gval = compute_gval_numpy(X_profile, q_current, y_henyey=y_henyey)

    # Equidistribute on the gval function
    q_cells_new = equidistribute_mesh_numpy(
        gval, q_current, N_zones, mesh_config=mesh_config)

    # Derive structure mesh (face positions) from cell centers
    # Face k sits at midpoint between cell k-1 and cell k
    faces = np.zeros(N_zones + 1)
    faces[0] = 0.0
    faces[-1] = 1.0
    faces[1:-1] = 0.5 * (q_cells_new[:-1] + q_cells_new[1:])
    q_struct_new = faces

    return q_cells_new, q_struct_new


# =============================================================================
# MESA-style remesh step (mesh adaptation + conservative remap)
# =============================================================================

def _step_remesh(composition, struct_state, mesh_state):
    """MESA-style mesh adaptation + conservative remap of all state.

    Parameters
    ----------
    composition : dict — {X, Y, Z, C12, C13, N14}
    struct_state : dict — {y_henyey, ln_T_prev, ln_P_prev, ln_rho_prev, Z_metal}
    mesh_state : dict — {q_cells, q_struct, N_zones}

    Called before each structure solve (MESA evolve.f90:1882-1886).
    Returns updated (q_cells, q_struct, compositions, y_henyey, prev arrays).
    """
    X_profile = composition['X']
    Y_profile = composition['Y']
    Z_profile = composition['Z']
    C12_profile = composition['C12']
    C13_profile = composition['C13']
    N14_profile = composition['N14']
    y_henyey = struct_state['y_henyey']
    ln_T_prev = struct_state['ln_T_prev']
    ln_P_prev = struct_state['ln_P_prev']
    ln_rho_prev = struct_state['ln_rho_prev']
    Z = struct_state['Z_metal']
    q_cells = mesh_state['q_cells']
    q_struct = mesh_state['q_struct']
    N_zones = mesh_state['N_zones']

    Xc_current = float(X_profile[0])
    post_tams = Xc_current < 0.05
    conc = MAX_CONCENTRATION_RGB if post_tams else MAX_CONCENTRATION_MS
    n_smooth = 1 if post_tams else 3
    # CONSTRAINT: disable T+P mesh functions post-TAMS.
    # Our sum-based gval (xa + T + P) dilutes shell concentration on the RGB:
    # the T+P gradients (~938 total range) dominate over xa (~46 range), leaving
    # only ~5% of the gval budget for shell resolution. MESA avoids this via
    # per-function min(dq) (mesh_plan.f90:1068-1072, minloc(nxt_dqs)) — each
    # function independently controls cell size. Our 1D equidistribution cannot
    # do this. Post-TAMS, the H-burning shell drives the RGB ascent and needs
    # tight resolution (the composition-only gval was already working for RGB
    # tests on main). The T+P terms are needed on the MS for envelope resolution
    # (l≥1 modes,) but are counterproductive on the RGB.
    y_henyey_for_mesh = y_henyey if not post_tams else None
    q_cells_new, q_struct_new = make_unified_mesh(
        X_profile, N_zones, q_current=q_cells,
        mesh_config={'max_concentration': conc, 'smooth_iters': n_smooth},
        y_henyey=y_henyey_for_mesh)

    mesh_changed = np.max(np.abs(q_cells_new - q_cells)) > 1e-8
    if not mesh_changed:
        return (q_cells, q_struct, X_profile, Y_profile, Z_profile,
                C12_profile, C13_profile, N14_profile,
                y_henyey, ln_T_prev, ln_P_prev, ln_rho_prev)

    # Conservative remap of composition (MESA mesh_adjust.f90:do_xa)
    X_profile = conservative_remap_numpy(X_profile, q_cells, q_cells_new)
    Y_profile = conservative_remap_numpy(Y_profile, q_cells, q_cells_new)
    Z_profile = conservative_remap_numpy(Z_profile, q_cells, q_cells_new)
    C12_profile = conservative_remap_numpy(C12_profile, q_cells, q_cells_new)
    C13_profile = conservative_remap_numpy(C13_profile, q_cells, q_cells_new)
    N14_profile = conservative_remap_numpy(N14_profile, q_cells, q_cells_new)

    # Normalize X+Y+Z=1 per cell (MESA do_xa:1318)
    total = np.maximum(X_profile + Y_profile + Z_profile, 1e-30)
    X_profile /= total
    Y_profile /= total
    Z_profile /= total

    # CN normalization: preserve initial CN total from Z (drift-free constant)
    CN_total_new = C12_profile + C13_profile + N14_profile
    CN_target = (0.170 + 0.170 / 89.0 + 0.079) * Z
    cn_ratio = np.where(CN_total_new > 1e-30, CN_target / CN_total_new, 1.0)
    C12_profile *= cn_ratio
    C13_profile *= cn_ratio
    N14_profile *= cn_ratio

    # Remap Henyey state + prev arrays
    y_henyey = remap_henyey_state_numpy(y_henyey, q_struct, q_struct_new)
    ln_T_prev = np.interp(q_struct_new[:N_zones], q_struct[:N_zones], ln_T_prev)
    ln_P_prev = np.interp(q_struct_new[:N_zones], q_struct[:N_zones], ln_P_prev)
    ln_rho_prev = np.interp(q_struct_new[:N_zones], q_struct[:N_zones], ln_rho_prev)

    return (q_cells_new, q_struct_new, X_profile, Y_profile, Z_profile,
            C12_profile, C13_profile, N14_profile,
            y_henyey, ln_T_prev, ln_P_prev, ln_rho_prev)
