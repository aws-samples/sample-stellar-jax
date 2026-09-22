"""Remap operations: conservative (composition) and linear (structure).

Conservative remap preserves integrated species mass: sum(profile * dm) is
conserved. Linear remap interpolates Henyey state variables in ln-space.

Neither function applies stop_gradient — that belongs in the orchestration
layer (structure_mesh.py / composition_mesh.py). These are pure remapping
operations; gradients flow through them normally.

Reference:
  - MESA mesh_adjust.f90:do_xa (piecewise-linear reconstruction + integration)
  - MESA mesh_adjust.f90:get1_lpp (lines 1720-1807) — slope computation
  - MESA mesh_adjust.f90:get_xq_integral (lines 1810-1924) — integration
  - Order-1 (linear) reconstruction + conservative integration.
    Cumulative species mass is piecewise-quadratic; evaluated at new boundaries
    via searchsorted + quadratic correction.

NOTE on reconstruction order: MESA passes mesh_adjust_use_quadratic=.true. to
do_xa (which sets order=2 in get_xq_integral), but the composition RECONSTRUCTION
coefficients are computed with quad=.false. (mesh_adjust.f90:258-260) and xa_c2=0
(line 267). The comment at line 256: "since we must adjust things to make the sum
of xa's = 1, only do linear reconstruction." So MESA's effective composition remap
is order-1 — our implementation MATCHES.

NOTE on per-zone X+Y+Z=1 normalization: MESA normalizes all species per cell
(mesh_adjust.f90:1322: xa(:,k) = xa(:,k) / xa_sum). This is CORRECT for MESA
because MESA tracks all species independently (H→He during burning keeps the
species sum ~1). Our burn step now also deposits H into He (ΔY = −ΔX, baryon
conservation — MESA net_eval.f90:274), so X+Y+Z=1 holds by
construction in burning zones. The CN normalization below handles the residual
from the CNO sub-cycle operator split.
"""
import jax.numpy as jnp


def _compute_slopes_jax(profile, dq, grid):
    """Compute limited linear slopes per cell (MESA get1_lpp, order=1).

    MESA reference: mesh_adjust.f90:1720-1807 (get1_lpp with quad=.false.).

    For each interior cell k:
      sm1 = (profile[k-1] - profile[k]) / ((dq[k-1]+dq[k])/2)
      s00 = (profile[k] - profile[k+1]) / ((dq[k]+dq[k+1])/2)
    If sm1*s00 <= 0: slope = 0 (local extremum).
    Otherwise: slope = (sm1 + s00) / 2, with minmod fallback if
    reconstruction overshoots at cell boundaries.

    CONSTRAINT (JAX): Vectorized operations, no Python loops. The limiter
    logic uses jnp.where for branchless execution. This matches MESA's
    algorithm exactly for the linear (non-quadratic) case.

    Parameters
    ----------
    profile : (N,) — values at node positions. DIFFERENTIABLE.
    dq : (N,) — cell widths.
    grid : (N,) — node positions (used for inter-node distances).

    Returns
    -------
    c1 : (N,) — slope per unit q at each node. DIFFERENTIABLE w.r.t. profile.
    """
    N = profile.shape[0]

    # Inter-node distances (using node positions, not dq midpoint formula)
    # CONSTRAINT: our convention stores point values at nodes, so distance
    # between values is grid[k] - grid[k-1], not (dq[k-1]+dq[k])/2.
    # For boundary cells (k=0, k=N-1), slope=0 (MESA: set_const).
    dist_left = jnp.zeros(N)
    dist_left = dist_left.at[1:].set(grid[1:] - grid[:-1])
    dist_left = jnp.maximum(dist_left, 1e-30)

    dist_right = jnp.zeros(N)
    dist_right = dist_right.at[:-1].set(grid[1:] - grid[:-1])
    dist_right = jnp.maximum(dist_right, 1e-30)

    # Left slope sm1 = (profile[k-1] - profile[k]) / dist_left[k]
    # Right slope s00 = (profile[k] - profile[k+1]) / dist_right[k]
    sm1 = jnp.zeros(N)
    sm1 = sm1.at[1:].set((profile[:-1] - profile[1:]) / dist_left[1:])

    s00 = jnp.zeros(N)
    s00 = s00.at[:-1].set((profile[:-1] - profile[1:]) / dist_right[:-1])

    # Average slope (MESA: "use average to smooth abundance transitions")
    c1_avg = (sm1 + s00) / 2

    # Monotonicity check at cell edges (MESA get1_lpp:1780-1790)
    # vbdy_outer = profile[k] + c1*dq[k]/2; vbdy_inner = profile[k] - c1*dq[k]/2
    dqhalf = dq / 2
    vbdy_outer = profile + c1_avg * dqhalf
    vbdy_inner = profile - c1_avg * dqhalf

    # Check outer boundary: (profile[k-1] - vbdy_outer) * (vbdy_outer - profile[k])
    # and inner boundary: (profile[k] - vbdy_inner) * (vbdy_inner - profile[k+1])
    prof_left = jnp.concatenate([profile[:1], profile[:-1]])   # profile[k-1], padded
    prof_right = jnp.concatenate([profile[1:], profile[-1:]])  # profile[k+1], padded

    overshoot_outer = (prof_left - vbdy_outer) * (vbdy_outer - profile) < 0
    overshoot_inner = (profile - vbdy_inner) * (vbdy_inner - prof_right) < 0
    overshoot = overshoot_outer | overshoot_inner

    # Minmod fallback: use the slope with smaller magnitude
    c1_minmod = jnp.where(jnp.abs(sm1) <= jnp.abs(s00), sm1, s00)

    # Apply limiter: use average slope, fall back to minmod on overshoot
    c1 = jnp.where(overshoot, c1_minmod, c1_avg)

    # Zero slope at extrema (sm1*s00 <= 0) and boundaries (k=0, k=N-1)
    is_extremum = sm1 * s00 <= 0
    is_boundary = jnp.zeros(N, dtype=bool)
    is_boundary = is_boundary.at[0].set(True)
    is_boundary = is_boundary.at[-1].set(True)
    c1 = jnp.where(is_extremum | is_boundary, 0.0, c1)

    return c1


def conservative_remap(profile, grid_old, grid_new):
    """Conservative order-1 (linear) remap of a composition profile.

    Preserves integrated species mass: sum(profile * dm) is conserved.

    Algorithm (MESA mesh_adjust.f90 order=1):
      1. Compute limited slopes via get1_lpp (vectorized in JAX).
      2. Compute cumulative species mass M(q) at old cell boundaries using
         the piecewise-linear reconstruction.
      3. For each new boundary q, find the old cell it falls in (via
         searchsorted), and compute the sub-cell integral analytically
         (piecewise-linear → quadratic antiderivative).
      4. Difference consecutive M values → species mass per new cell.
      5. Divide by new cell mass → new profile values.

    MESA reference: mesh_adjust.f90:get_xq_integral (lines 1810-1924).
    The linear interpolant is f(q) = c0[k] + c1[k]*(q_node[k] - q),
    so the integral from a to b is:
      c0[k]*(b-a) + c1[k]*(q_node[k]*(b-a) - (b²-a²)/2)

    Parameters
    ----------
    profile : (N,) — composition values on old mesh (mass fractions).
        DIFFERENTIABLE — gradients flow through this argument.
    grid_old : (N,) — old grid positions. Caller should stop_gradient.
    grid_new : (N,) — new grid positions. Caller should stop_gradient.

    Returns
    -------
    profile_new : (N,) — composition on new mesh, mass-conserving.
        DIFFERENTIABLE w.r.t. profile.
    """
    N = profile.shape[0]

    # Cell boundaries on old mesh (midpoints between nodes)
    bnd_old = jnp.zeros(N + 1)
    bnd_old = bnd_old.at[0].set(0.0)
    bnd_old = bnd_old.at[1:-1].set(0.5 * (grid_old[:-1] + grid_old[1:]))
    bnd_old = bnd_old.at[-1].set(1.0)

    # Cell widths on old mesh
    dq_old = jnp.diff(bnd_old)  # (N,)
    dq_old = jnp.maximum(dq_old, 1e-30)

    # --- Step 1: Linear reconstruction (MESA get1_lpp, order=1) ---
    c0 = profile  # cell "averages" (point values at nodes)
    c1 = _compute_slopes_jax(profile, dq_old, grid_old)

    # --- Step 2: Cumulative species mass at old boundaries ---
    # For cell k, the integral from bnd_old[k] to bnd_old[k+1] of
    # f(q) = c0[k] + c1[k]*(grid_old[k] - q) is:
    # = (c0[k] + c1[k]*(grid_old[k] - cell_mid[k])) * dq[k]
    cell_mid = 0.5 * (bnd_old[:-1] + bnd_old[1:])  # (N,) midpoints
    full_cell_mass = (c0 + c1 * (grid_old - cell_mid)) * dq_old  # (N,)
    cum_mass_at_bnd = jnp.concatenate([jnp.zeros(1), jnp.cumsum(full_cell_mass)])  # (N+1,)

    # --- Step 3: Evaluate cumulative mass at new cell boundaries ---
    bnd_new = jnp.zeros(N + 1)
    bnd_new = bnd_new.at[0].set(0.0)
    bnd_new = bnd_new.at[1:-1].set(0.5 * (grid_new[:-1] + grid_new[1:]))
    bnd_new = bnd_new.at[-1].set(1.0)

    # For each new boundary q_b, find the old cell index it falls in
    cell_idx = jnp.searchsorted(bnd_old, bnd_new, side='right') - 1
    cell_idx = jnp.clip(cell_idx, 0, N - 1)

    # Cumulative mass at the LEFT boundary of the cell containing q_b
    cum_at_cell_left = cum_mass_at_bnd[cell_idx]  # (N+1,)

    # Sub-cell integral from bnd_old[cell_idx] to bnd_new
    a = bnd_old[cell_idx]        # left boundary of containing cell
    b = bnd_new                  # the new boundary point
    delta = b - a                # width of partial cell
    k_idx = cell_idx             # old cell index

    c0_k = c0[k_idx]
    c1_k = c1[k_idx]
    node_k = grid_old[k_idx]

    # Sub-cell integral: (c0 + c1*(node - (a+b)/2)) * delta
    overlap_center = 0.5 * (a + b)
    sub_cell_integral = (c0_k + c1_k * (node_k - overlap_center)) * delta

    # Total cumulative mass at each new boundary
    cum_mass_at_new = cum_at_cell_left + sub_cell_integral  # (N+1,)

    # --- Step 4: Species mass in each new cell ---
    species_mass_new = jnp.diff(cum_mass_at_new)  # (N,)

    # New cell widths
    dm_new = jnp.diff(bnd_new)  # (N,)
    dm_new = jnp.maximum(dm_new, 1e-30)

    # --- Step 5: New profile = species mass / cell mass ---
    profile_new = species_mass_new / dm_new

    # --- Step 6: Enforce exact mass conservation (MESA do_xa:1318 analog) ---
    # The piecewise-linear integration computes a slightly different total mass
    # than sum(profile * dq) because the slope correction (c1*(node - cell_mid))
    # is non-zero when nodes don't coincide with cell midpoints. Rescale so
    # the standard mass metric sum(profile_new * dm_new) == sum(profile * dq_old)
    # is conserved exactly, matching what callers measure.
    mass_old = jnp.sum(profile * dq_old)
    mass_new = jnp.sum(profile_new * dm_new)
    scale = jnp.where(mass_new > 1e-30, mass_old / mass_new, 1.0)
    profile_new = profile_new * scale

    # NOTE: no clip to [0,1] here. The slope limiter (minmod + monotonicity
    # check in _compute_slopes_jax) already prevents overshoots beyond
    # neighboring values. The global rescaling preserves this bound because
    # scale ≈ 1.0 (it corrects only the node-vs-midpoint integration bias,
    # which is O(dq²) small). A clip would silently break per-zone
    # multi-species conservation (e.g. C12+C13+N14) by modifying individual
    # species asymmetrically — the CN normalization in remap_all_species
    # relies on unclipped values to restore per-zone sums correctly.

    return profile_new


def remap_linear(y_old, q_mesh_old, q_mesh_new):
    """Remap Henyey state from old mesh to new mesh via linear interpolation.

    Each variable in y = (ln_r, ln_P, ln_T, ell) is interpolated linearly
    in q. Linear interpolation in ln-space is equivalent to geometric
    interpolation in physical space — appropriate for quantities spanning
    many orders of magnitude.

    Parameters
    ----------
    y_old : (N_s, 4) — Henyey state on old mesh. DIFFERENTIABLE.
    q_mesh_old : (N_s+1,) — old mesh coordinates.
    q_mesh_new : (N_s+1,) — new mesh coordinates.

    Returns
    -------
    y_new : (N_s, 4) — Henyey state remapped to new mesh.
        DIFFERENTIABLE w.r.t. y_old.
    """
    N_s = y_old.shape[0]
    q_old = q_mesh_old[:N_s]  # interface positions on old mesh
    q_new = q_mesh_new[:N_s]  # interface positions on new mesh

    def interp_var(col):
        return jnp.interp(q_new, q_old, y_old[:, col],
                          left=y_old[0, col], right=y_old[-1, col])

    y_new = jnp.stack([interp_var(i) for i in range(4)], axis=-1)
    return y_new


def remap_all_species(X_prof, Y_prof, Z_prof, C12_prof, C13_prof, N14_prof,
                      grid_old, grid_new):
    """Conservatively remap all composition profiles to a new mesh.

    Each species is remapped independently via conservative_remap (order-1
    linear reconstruction + conservative integration). The remap preserves
    integrated mass of each species to machine precision (MESA do_xa analog).

    Per-zone CN normalization: the nonlinear slope limiter (minmod/extremum
    zeroing) is applied independently to C12, C13, N14. Since the limiter is
    nonlinear, limited_slope(C12) + limited_slope(C13) + limited_slope(N14)
    != limited_slope(C12+C13+N14), even when the CN total is spatially
    constant. Combined with the per-species global rescaling (Step 6 in
    conservative_remap), this causes per-zone C12+C13+N14 drift.

    Fix: after remapping sub-species, rescale them so their per-zone sum
    matches the per-zone sum from BEFORE the remap (conservatively projected
    to the new grid). We remap the CN_total as a single field (whose slope
    IS correctly limited as one entity), then force sub-species to match it.

    MESA analog: mesh_adjust.f90:do_xa (line 1203-1324) remaps each species
    independently with conservative integration, then normalizes so that
    sum(xa) = 1 per cell. Our CN normalization is the analogous constraint
    for the sub-species of Z.

    Parameters
    ----------
    X_prof, Y_prof, Z_prof, C12_prof, C13_prof, N14_prof : (N_COMP,)
        Composition profiles on old mesh. DIFFERENTIABLE.
    grid_old : (N_COMP,) — old grid positions.
    grid_new : (N_COMP,) — new grid positions.

    Returns
    -------
    Tuple of 6 remapped profiles (same shapes). DIFFERENTIABLE.
    """
    X_new = conservative_remap(X_prof, grid_old, grid_new)
    Y_new = conservative_remap(Y_prof, grid_old, grid_new)
    Z_new = conservative_remap(Z_prof, grid_old, grid_new)
    C12_new = conservative_remap(C12_prof, grid_old, grid_new)
    C13_new = conservative_remap(C13_prof, grid_old, grid_new)
    N14_new = conservative_remap(N14_prof, grid_old, grid_new)

    # --- Per-zone CN normalization ---
    # The CN cycle conserves C12+C13+N14 per zone (only redistributes among
    # isotopes). The order-1 remap with nonlinear slope limiters applied
    # independently to each isotope, plus independent per-species global
    # rescaling (Step 6 in conservative_remap), breaks this per-zone sum.
    #
    # Fix: remap the CN_total (= C12+C13+N14) as a SINGLE FIELD, then
    # normalize sub-species so their per-zone sum matches the remapped total.
    # Since CN_total is spatially nearly-constant (burn_cno conserves per zone,
    # mix homogenizes within CZ), its slopes are nearly zero and it remaps
    # with negligible per-zone error. This avoids the independent slope-limiter
    # nonlinearity that breaks the sum when remapping each isotope separately.
    #
    # MESA analog: mesh_adjust.f90:do_xa normalizes xa(:,k)/xa_sum per cell.
    CN_total_old = C12_prof + C13_prof + N14_prof
    CN_total_remapped = conservative_remap(CN_total_old, grid_old, grid_new)
    CN_total_new = C12_new + C13_new + N14_new

    cn_ratio = jnp.where(CN_total_new > 1e-30,
                         CN_total_remapped / CN_total_new, 1.0)
    C12_new = C12_new * cn_ratio
    C13_new = C13_new * cn_ratio
    N14_new = N14_new * cn_ratio

    return X_new, Y_new, Z_new, C12_new, C13_new, N14_new
