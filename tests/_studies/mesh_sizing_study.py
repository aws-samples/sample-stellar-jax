"""Issue #92: Fixed Lagrangian mass-mesh sizing study.

Demonstrates dL/dm error convergence for the recommended mass mesh at a
2 M☉ RGB H-shell burning structure. The mesh formula is:

    q_k = B(ξ_k; a, b)    where ξ_k = k/N, k=0..N

    B(ξ; a, b) = ξ^a * [1 + (b-1)*ξ*(1-ξ)]

normalized so q(0)=0, q(1)=1.  Parameter 'a' controls center-concentration
(a>1 gives denser spacing at q=0), and 'b' provides additional density at
intermediate q (the H-shell region).

Final recommended mesh uses a **three-segment** analytic formula:
    q(ξ) = w_c * ξ^a_c + w_s * [1-(1-ξ)^a_s] + w_m * ξ

with w_c + w_s + w_m = 1, choosing a_c=3 (center), a_s=3 (surface), and
weights tuned to place ~40 zones in the shell region q∈[0.15,0.20] for
a 2 M☉ RGB.

References:
  - Kippenhahn, Weigert & Weiss (2012), §11.2: Henyey method mesh placement
  - Paxton et al. (2011, 2013): MESA adaptive mesh functions
  - Christensen-Dalsgaard (2008, ApSS 316, 13): mesh resolution for oscillations

Usage:
    python3.11 studies/mesh_sizing_study.py
"""
import numpy as np


def mass_mesh(N, a_c=3.0, a_s=3.0, w_c=0.3, w_s=0.3):
    """Generate non-uniform mass mesh q(k) for k=0..N.

    Three-component formula:
      q(ξ) = w_c * ξ^a_c + w_s * [1-(1-ξ)^a_s] + (1 - w_c - w_s) * ξ

    Parameters
    ----------
    N : int
        Number of zones (N+1 mesh points).
    a_c : float
        Center concentration exponent (>1 denser at center).
    a_s : float
        Surface concentration exponent (>1 denser at surface).
    w_c : float
        Weight of center-concentrated component.
    w_s : float
        Weight of surface-concentrated component.

    Returns
    -------
    q : ndarray, shape (N+1,)
        Mass fractions m/M from 0 to 1.
    """
    xi = np.linspace(0.0, 1.0, N + 1)
    w_m = 1.0 - w_c - w_s
    q = w_c * xi**a_c + w_s * (1.0 - (1.0 - xi)**a_s) + w_m * xi
    q[0] = 0.0
    q[-1] = 1.0
    return q


def mock_rgb_shell_eps(q, q_shell=0.175, dq_shell=0.015, eps_peak=1e4):
    """Mock ε(q) profile for 2 M☉ RGB H-shell burning.

    CNO shell burning: ε ∝ exp(-(q-q_shell)^2 / (2*dq_shell^2))
    This is a Gaussian in mass coordinate, representing the thin shell.

    The shell width dq_shell ~ 0.015 corresponds to Δm ~ 0.03 M☉ for a
    2 M☉ star, consistent with MESA models (Paxton et al. 2013, Fig 8).
    """
    return eps_peak * np.exp(-0.5 * ((q - q_shell) / dq_shell)**2)


def luminosity_integral(q, eps):
    """Integrate L(q) = ∫₀^q ε(q') dq' using the trapezoidal rule.

    In Lagrangian coordinates, dL/dm = ε, so L(q) = M * ∫₀^q ε dq'.
    We normalize to L_total = ∫₀^1 ε dq.
    """
    L = np.zeros_like(q)
    for i in range(1, len(q)):
        L[i] = L[i-1] + 0.5 * (eps[i] + eps[i-1]) * (q[i] - q[i-1])
    return L


def dLdm_error(N, a_c=3.0, a_s=3.0, w_c=0.3, w_s=0.3,
               q_shell=0.175, dq_shell=0.015):
    """Compute max relative error in dL/dm at the H-shell for mesh with N zones.

    Method: compare the numerical derivative (ΔL/Δq at each zone) against the
    analytic ε(q) at zone centers. The relative error is |ΔL/Δq - ε_mid| / ε_peak.
    We report the maximum over zones in the shell region [q_shell-3σ, q_shell+3σ].
    """
    q = mass_mesh(N, a_c, a_s, w_c, w_s)
    eps = mock_rgb_shell_eps(q, q_shell, dq_shell)

    # Numerical dL/dq at zone centers
    dLdq_num = np.diff(luminosity_integral(q, eps)) / np.diff(q)

    # Analytic ε at zone centers
    q_mid = 0.5 * (q[:-1] + q[1:])
    eps_mid = mock_rgb_shell_eps(q_mid, q_shell, dq_shell)

    # Shell region: within 3σ of peak
    shell_mask = np.abs(q_mid - q_shell) < 3 * dq_shell
    if not np.any(shell_mask):
        return 1.0  # no zones in shell — worst case

    eps_peak = mock_rgb_shell_eps(np.array([q_shell]), q_shell, dq_shell)[0]
    rel_err = np.abs(dLdq_num[shell_mask] - eps_mid[shell_mask]) / eps_peak

    return np.max(rel_err)


def zones_in_shell(N, a_c=3.0, a_s=3.0, w_c=0.3, w_s=0.3,
                   q_shell=0.175, dq_shell=0.015):
    """Count zones within 2σ of the shell peak."""
    q = mass_mesh(N, a_c, a_s, w_c, w_s)
    q_mid = 0.5 * (q[:-1] + q[1:])
    return np.sum(np.abs(q_mid - q_shell) < 2 * dq_shell)


def convergence_study():
    """Run convergence study: dL/dm error vs N for recommended mesh params."""
    print("=" * 72)
    print("Issue #92: Mass-mesh sizing convergence study")
    print("Target: 2 M☉ RGB H-shell (CNO, q_shell=0.175, dq=0.015)")
    print("=" * 72)

    # Recommended parameters
    a_c, a_s, w_c, w_s = 3.0, 3.0, 0.30, 0.15

    Ns = [200, 400, 600, 800, 1000, 1200, 1600, 2400]

    print(f"\nMesh params: a_c={a_c}, a_s={a_s}, w_c={w_c}, w_s={w_s}")
    print(f"{'N':>6}  {'max|dL/dm err|':>14}  {'zones in shell':>14}  "
          f"{'min dq (shell)':>14}  {'min dq (surf)':>14}")
    print("-" * 72)

    results = []
    for N in Ns:
        err = dLdm_error(N, a_c, a_s, w_c, w_s)
        n_shell = zones_in_shell(N, a_c, a_s, w_c, w_s)

        q = mass_mesh(N, a_c, a_s, w_c, w_s)
        dq = np.diff(q)
        q_mid = 0.5 * (q[:-1] + q[1:])
        shell_mask = np.abs(q_mid - 0.175) < 2 * 0.015
        min_dq_shell = dq[shell_mask].min() if np.any(shell_mask) else 0
        min_dq_surf = dq[-5:].min()  # last 5 zones near surface

        print(f"{N:>6}  {err:>14.6f}  {n_shell:>14d}  "
              f"{min_dq_shell:>14.2e}  {min_dq_surf:>14.2e}")
        results.append((N, err, n_shell, min_dq_shell, min_dq_surf))

    # Compare with uniform mesh
    print(f"\n{'--- Uniform mesh (baseline) ---':^72}")
    print(f"{'N':>6}  {'max|dL/dm err|':>14}  {'zones in shell':>14}")
    print("-" * 72)
    for N in [800, 1000, 1200]:
        q_uni = np.linspace(0, 1, N + 1)
        eps_uni = mock_rgb_shell_eps(q_uni)
        dLdq_uni = np.diff(luminosity_integral(q_uni, eps_uni)) / np.diff(q_uni)
        q_mid_uni = 0.5 * (q_uni[:-1] + q_uni[1:])
        eps_mid_uni = mock_rgb_shell_eps(q_mid_uni)
        shell_mask = np.abs(q_mid_uni - 0.175) < 3 * 0.015
        eps_peak = mock_rgb_shell_eps(np.array([0.175]))[0]
        err_uni = np.max(np.abs(dLdq_uni[shell_mask] - eps_mid_uni[shell_mask]) / eps_peak)
        n_shell_uni = np.sum(np.abs(q_mid_uni - 0.175) < 2 * 0.015)
        print(f"{N:>6}  {err_uni:>14.6f}  {n_shell_uni:>14d}")

    # Compare with existing q=ξ³ mesh (composition grid)
    print(f"\n{'--- Current q=ξ³ mesh (N_COMP style) ---':^72}")
    print(f"{'N':>6}  {'max|dL/dm err|':>14}  {'zones in shell':>14}")
    print("-" * 72)
    for N in [800, 1000, 1200]:
        xi = np.linspace(0, 1, N + 1)
        q_cube = xi**3
        eps_cube = mock_rgb_shell_eps(q_cube)
        dLdq = np.diff(luminosity_integral(q_cube, eps_cube)) / np.diff(q_cube)
        q_mid_c = 0.5 * (q_cube[:-1] + q_cube[1:])
        eps_mid_c = mock_rgb_shell_eps(q_mid_c)
        shell_mask = np.abs(q_mid_c - 0.175) < 3 * 0.015
        eps_peak = mock_rgb_shell_eps(np.array([0.175]))[0]
        if np.any(shell_mask):
            err_c = np.max(np.abs(dLdq[shell_mask] - eps_mid_c[shell_mask]) / eps_peak)
        else:
            err_c = 1.0
        n_shell_c = np.sum(np.abs(q_mid_c - 0.175) < 2 * 0.015)
        print(f"{N:>6}  {err_c:>14.6f}  {n_shell_c:>14d}")

    # Final recommendation
    print("\n" + "=" * 72)
    print("RECOMMENDATION")
    print("=" * 72)
    # Find N where error < 0.01
    for N, err, n_shell, _, _ in results:
        if err < 0.01:
            print(f"\n  N = {N} zones achieves max|dL/dm err| = {err:.4f} < 1%")
            print(f"  with {n_shell} zones in the H-shell region.")
            print(f"\n  Recommended: N = 1000 (within 800-1200 target range)")
            break

    print(f"""
  Formula: q(ξ) = 0.30·ξ³ + 0.15·[1-(1-ξ)³] + 0.55·ξ
           where ξ = k/N, k = 0, 1, ..., N

  Properties at N=1000:
    - Center cell:  dq ~ {mass_mesh(1000, a_c, a_s, w_c, w_s)[1]:.2e}
    - Surface cell: dq ~ {1-mass_mesh(1000, a_c, a_s, w_c, w_s)[-2]:.2e}
    - Shell zones:  ~{zones_in_shell(1000, a_c, a_s, w_c, w_s)} in |q-0.175| < 0.03
    - Max dq:       {np.diff(mass_mesh(1000, a_c, a_s, w_c, w_s)).max():.4f}
""")

    return results


if __name__ == "__main__":
    results = convergence_study()
