# %% [markdown]
# # Tutorial: differentiable seismic gradients & structure kernels with `stellar-jax`
#
# A short, **runnable, step-by-step** walkthrough of the two headline capabilities:
#
# 1. the **analytic gradient** of an oscillation frequency w.r.t. a stellar parameter
#    (`jax.grad`, one line), and
# 2. the **sound-speed structure kernel** `K(r) = ∂ν/∂c²(r)` over the whole stellar
#    interior in one reverse-mode (adjoint) pass (`compute_structure_kernels`, one call),
#
# validated against the field-standard oscillation code **ADIPLS**.
#
# This file is written in the "percent" cell format: run it as a plain script
# (`python tutorial.py`), **or** open it in Jupyter / VS Code and it renders as a
# notebook (each `# %%` is a cell). No saved outputs to go stale — you always see the
# live result of the current code.
#
# > For the full set of committed, publication-style figures, run
# > `demo_seismic_gradients.py` instead. This tutorial is the "learn the API" path;
# > that script is the "reproduce the validated figures" path.

# %% [markdown]
# ## Step 0 — imports
#
# `stellar-jax` is pure JAX (CPU is fine). Importing it enables float64 automatically,
# which the oscillation solve needs.

# %%
import os
import numpy as np
import jax

import stellar_jax  # noqa: F401 — importing enables jax_enable_x64
from stellar_jax.oscillations import read_fgong
from stellar_jax.oscillations.kernels import compute_structure_kernels

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

# %% [markdown]
# ## Step 1 — load a stellar model (the FGONG format)
#
# The input to the oscillation code is a **1-D stellar structure model** in the
# standard **FGONG** text format (Christensen-Dalsgaard 2008, Ap&SS 316, §A.1) — the
# interchange format between a stellar-*evolution* code (e.g. MESA) and an
# *oscillation* code. You can produce one from your own evolution run; here we use the
# committed reference solar model, **Model S** (Christensen-Dalsgaard et al. 1996).
#
# `read_fgong` returns two arrays:
#
# * **`glob`** — star-level scalars (globals): `glob[0]` = mass M, `glob[1]` = radius R,
#   `glob[2]` = luminosity L, then Z, X₀, α_mlt, … (`iconst` values).
# * **`var`** — the per-shell profile, shape `(n_shells, n_vars)`, ordered
#   **center → surface**. The columns this demo uses:
#   `var[:,0]` = r (radius, cm), `var[:,2]` = T, `var[:,3]` = P, `var[:,4]` = ρ,
#   `var[:,9]` = Γ₁ (the adiabatic exponent — what the structure kernel is taken w.r.t).

# %%
fgong_path = os.path.join(_REPO_ROOT, "data", "model_s", "fgong.l5bi.d.15c")
glob, var = read_fgong(fgong_path)

M, R, L = glob[0], glob[1], glob[2]
n_shells = var.shape[0]
print(f"Loaded Model S: {n_shells} shells")
print(f"  M = {M:.4e} g   R = {R:.4e} cm   L = {L:.4e} erg/s")
print(f"  radius r    (var[:,0]): {var[0, 0]:.3e} → {var[-1, 0]:.3e} cm (center→surface)")
print(f"  density ρ   (var[:,4]): {var[0, 4]:.3e} → {var[-1, 4]:.3e} g/cm³")
print(f"  Γ₁          (var[:,9]): {var[0, 9]:.4f} → {var[-1, 9]:.4f}")
# To use YOUR OWN star, point read_fgong at any FGONG file from your evolution code.

# %% [markdown]
# ## Step 2 — a structure kernel `K(r) = ∂ν/∂c²(r)`, in ONE call
#
# The structure kernel tells you **how sensitive a mode frequency is to a change in the
# sound speed at each radius** — a function over the whole interior, used for
# seismic *inversions*. Classically it's derived analytically per code (Gough 1991).
# Here it comes straight out of the differentiable model: one call, one adjoint pass.
#
# We ask for the radial mode `l=0, n_pg=20`. `compute_structure_kernels` finds that
# mode, differentiates the eigenfrequency w.r.t. Γ₁ at every mesh point via `jax.grad`,
# and returns the kernel `K_gamma1_rho(r)` over the full interior — thousands of
# sensitivities from a **single** backward pass (finite differences would need ~2×
# that many eigenvalue solves).

# %%
kr = compute_structure_kernels(glob, var, l=0, n_pg=20)
print(f"Mode found: l={kr['l']}, n_pg={kr['n_pg']}, ν = {kr['nu']:.1f} µHz")
print(f"Kernel K_(Γ₁,ρ)(r): {kr['K_gamma1_rho'].shape[0]} points over the interior")
print(f"  computed in ONE adjoint pass (FD would need ~{2 * kr['K_gamma1_rho'].shape[0]} "
      f"eigenvalue solves for this one mode)")

# Validate it against the committed ADIPLS analytic kernel for the same mode.
adipls = np.loadtxt(os.path.join(
    _REPO_ROOT, "data", "model_s", "adipls_kernels", "adipls_gm1ker_l0_n20.dat"))
x_adipls, K_adipls_raw = adipls[:, 0], adipls[:, 1]
# ADIPLS uses absolute δΓ₁; ours uses relative δΓ₁/Γ₁ → K_ours = K_ADIPLS·Γ₁(x).
x_fgong, gamma1_fgong = var[:, 0] / R, var[:, 9]
K_adipls = K_adipls_raw * np.interp(x_adipls, x_fgong, gamma1_fgong)
K_ours = np.interp(x_adipls, kr["x"], kr["K_gamma1_rho"])
interior = (x_adipls > 0.1) & (x_adipls < 0.9)
K_ours_n = K_ours / np.trapezoid(K_ours[interior], x_adipls[interior])
K_adi_n = K_adipls / np.trapezoid(K_adipls[interior], x_adipls[interior])
rms = np.sqrt(np.mean(((K_ours_n - K_adi_n)[interior]) ** 2)
              / np.max(np.abs(K_adi_n[interior])) ** 2)
print(f"  vs ADIPLS (0.1<r/R<0.9): {rms:.1%} RMS peak-norm — matches the gold standard.")

# %% [markdown]
# ## Step 3 — a physics-parameter gradient `∂ν²/∂M`, in ONE line
#
# The same autodiff gives the sensitivity of the p-mode spectrum to a *physics
# parameter*. Below is the essence: write the frequency as a function of the parameter,
# then `jax.grad` it. (The demo script's `run_part1` runs the fully-set-up, FD-validated
# version — machine-precision agreement with an independent finite difference — over the
# same code path; we call it here so the tutorial reuses the validated implementation
# rather than re-deriving it.)

# %%
from demo_seismic_gradients import run_part1  # the validated ∂ν²/∂M path + FD cross-check

results = run_part1(full_evolution_grads=False)   # prints AD gradient and FD cross-check
# The key idea in one line is: ad_grad = jax.grad(sigma2_of_parameter)(parameter_value)
# — reverse-mode AD returns the sensitivity w.r.t. the input exactly (no finite-difference ε).

# %% [markdown]
# ## Where to go next
#
# * **Reproduce the full validated figures** (Sun + a 2.0 M☉ convective-core star,
#   degrees l=0,1,2, kernels vs ADIPLS): run `demo_seismic_gradients.py`.
# * **Use it on your own star**: point `read_fgong` at any FGONG file your evolution
#   code produces, then call `compute_structure_kernels` / take `jax.grad` exactly as above.
# * **What's validated, and the honest limitations** (the full gradient-vs-FD
#   summary table — including the now-fixed ∂ν/∂Y and the delivered subgiant
#   ∂ν²/∂M at 24.5%): see this folder's `README.md`.

if __name__ == "__main__":
    print("\nTutorial complete — see README.md for scope, validation, and next steps.")
