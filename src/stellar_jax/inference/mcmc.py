"""Minimal Metropolis-Hastings MCMC sampler for Fisher-vs-MCMC validation.

This module provides a simple, dependency-free MCMC sampler sufficient to
validate the Fisher matrix machinery on synthetic/analytic toy problems.
It is NOT a general-purpose MCMC framework — it exists solely to provide
an independent posterior width estimate for the Cramér-Rao validation test
(issue #1036).

For a Gaussian likelihood (the toy problem's regime), Metropolis-Hastings
converges reliably to the true posterior with O(10⁴) samples. The 20%
Fisher-vs-MCMC agreement bar (Wolz 2012, arXiv:1205.3984) is conservative
for this regime; ~5% agreement is expected.

Design choices:
- Pure NumPy (no JAX dependency): MCMC sampling does not need gradients or
  JIT. Using NumPy avoids XLA compile overhead for a simple random walk.
- Adaptive proposal: diagonal covariance scaled by 2.38²/d (Roberts et al.
  1997, "Weak convergence and optimal scaling of random walk Metropolis").
  Adapts during burn-in only (the sampled chain is stationary).
- Returns posterior standard deviations per parameter for direct comparison
  with Fisher forecast_sigma output.

References
----------
Metropolis et al. (1953), J. Chem. Phys. 21, 1087 — the original MH algorithm.
Roberts, Gelman & Gilks (1997), Ann. Appl. Probab. 7(1), 110 — optimal scaling
    2.38²/d for Gaussian targets.
Wolz et al. (2012), arXiv:1205.3984 — Fisher-vs-MCMC validation protocol.
"""

from typing import Callable, Dict, NamedTuple, Optional, Sequence

import numpy as np


class MCMCResult(NamedTuple):
    """Result of an MCMC run.

    Attributes
    ----------
    chain : (n_samples, n_params) array — post-burn-in samples.
    acceptance_rate : float — fraction of accepted proposals.
    sigma_params : dict — per-parameter posterior standard deviation.
    mean_params : dict — per-parameter posterior mean.
    param_names : list of str — parameter names.
    """
    chain: np.ndarray
    acceptance_rate: float
    sigma_params: Dict[str, float]
    mean_params: Dict[str, float]
    param_names: list


def run_mcmc(
    log_prob_fn: Callable[[np.ndarray], float],
    theta_init: np.ndarray,
    param_names: Sequence[str],
    n_samples: int = 50_000,
    n_burnin: int = 10_000,
    proposal_cov: Optional[np.ndarray] = None,
    seed: int = 42,
) -> MCMCResult:
    """Run Metropolis-Hastings MCMC to sample a posterior.

    Parameters
    ----------
    log_prob_fn : callable
        Function theta (1D array) → log-posterior (scalar). Must be finite
        at theta_init.
    theta_init : (n_params,) array
        Starting point (should be near the mode for efficiency).
    param_names : sequence of str
        Names for each parameter (for labeling output).
    n_samples : int
        Number of post-burn-in samples to collect.
    n_burnin : int
        Number of burn-in samples (discarded). Proposal covariance is
        adapted during burn-in using the Roberts et al. (1997) optimal
        scaling: Σ_prop = (2.38² / d) × Σ_chain.
    proposal_cov : (n_params, n_params) array, optional
        Initial proposal covariance. If None, uses 0.01² × I (conservative
        initial step). Adapted during burn-in.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    MCMCResult with the sampled chain and per-parameter σ.
    """
    rng = np.random.default_rng(seed)
    n_params = len(theta_init)

    if proposal_cov is None:
        proposal_cov = (0.01 ** 2) * np.eye(n_params)

    # Cholesky of proposal covariance for efficient sampling
    prop_L = np.linalg.cholesky(proposal_cov)

    theta_current = np.array(theta_init, dtype=np.float64)
    lp_current = log_prob_fn(theta_current)

    if not np.isfinite(lp_current):
        raise ValueError(
            f"log_prob at theta_init is not finite: {lp_current}. "
            f"theta_init = {theta_init}")

    # ── Burn-in with proposal adaptation ──
    burnin_chain = np.zeros((n_burnin, n_params))
    n_accepted_burnin = 0

    for i in range(n_burnin):
        # Propose: θ' = θ + L @ z, z ~ N(0, I)
        z = rng.standard_normal(n_params)
        theta_proposed = theta_current + prop_L @ z

        lp_proposed = log_prob_fn(theta_proposed)
        log_alpha = lp_proposed - lp_current

        if np.isfinite(log_alpha) and np.log(rng.uniform()) < log_alpha:
            theta_current = theta_proposed
            lp_current = lp_proposed
            n_accepted_burnin += 1

        burnin_chain[i] = theta_current

        # Adapt proposal covariance every 500 steps (Roberts et al. 1997).
        # Scale factor 2.38²/d is optimal for Gaussian targets.
        if (i + 1) % 500 == 0 and i >= 499:
            chain_so_far = burnin_chain[:i + 1]
            emp_cov = np.cov(chain_so_far.T)
            # Ensure positive-definite: add small diagonal regularization
            emp_cov += 1e-10 * np.eye(n_params)
            optimal_scale = (2.38 ** 2) / n_params
            proposal_cov = optimal_scale * emp_cov
            try:
                prop_L = np.linalg.cholesky(proposal_cov)
            except np.linalg.LinAlgError:
                pass  # Keep previous Cholesky if adaptation fails

    # ── Production sampling ──
    chain = np.zeros((n_samples, n_params))
    n_accepted = 0

    for i in range(n_samples):
        z = rng.standard_normal(n_params)
        theta_proposed = theta_current + prop_L @ z

        lp_proposed = log_prob_fn(theta_proposed)
        log_alpha = lp_proposed - lp_current

        if np.isfinite(log_alpha) and np.log(rng.uniform()) < log_alpha:
            theta_current = theta_proposed
            lp_current = lp_proposed
            n_accepted += 1

        chain[i] = theta_current

    acceptance_rate = n_accepted / n_samples

    # Per-parameter statistics
    sigma_params = {}
    mean_params = {}
    for j, pname in enumerate(param_names):
        sigma_params[pname] = float(np.std(chain[:, j]))
        mean_params[pname] = float(np.mean(chain[:, j]))

    return MCMCResult(
        chain=chain,
        acceptance_rate=acceptance_rate,
        sigma_params=sigma_params,
        mean_params=mean_params,
        param_names=list(param_names),
    )
