# Coding Standards — Numerical Guards and Floors

## The convention

Small-number literals appear throughout the codebase in three roles.
A reader must be able to tell them apart at a glance:

### 1. Underflow / div-by-zero guards — leave as bare literals

```python
x + 1e-30          # additive guard
jnp.maximum(x, 1e-30)   # clamp guard
```

The magnitude is **immaterial** — any tiny positive value prevents `log(0)`,
`1/0`, or `sqrt(negative)`. These are self-evident and stay inline. Do **not**
name them; doing so implies the value matters.

### 2. Physical floors — use a **named constant** from `config/physics_floors.py`

```python
from stellar_jax.config.physics_floors import PGAS_FRAC_FLOOR

P_gas = jnp.maximum(P - P_rad, PGAS_FRAC_FLOOR * P)
```

The value **matters**: it is a modeling choice that can bind physically or change
a derived quantity. Editing it should be a single-site change, not a 25-file
find-and-replace. Each constant carries:

- A one-line comment: **what** quantity it floors, **why**, and MESA grounding.
- SCREAMING_SNAKE_CASE, following the `mesh/mesh_params.py` precedent.

Current named floors (defined in `config/physics_floors.py`):

| Constant | Value | Role |
|---|---|---|
| `NAD_CP_FLOOR` | 0.05 | ∇_ad floor for cp denominators (CONSTRAINT: JAX; MESA unfloored) |
| `NAD_UPPER` | 0.45 | ∇_ad ceiling for EOS interpolation range |
| `PGAS_FRAC_FLOOR` | 1e-3 | P_gas ≥ 0.1% P_total in radiation-dominated envelopes |
| `X_MIN_PP` | 1e-3 | H mass-fraction floor for pp-chain screening (1/X finite) |
| `DELTA_LGL_HARD_LIMIT` | 0.20 | \|Δlog L\| per-step hard limit [dex] (drift guard, not a physical floor) |
| `DELTA_LGTE_HARD_LIMIT` | 0.05 | \|Δlog Teff\| per-step hard limit [dex] (drift guard, not a physical floor) |

### 3. Outlier-magnitude guards — add an inline comment

Guards with unusual magnitudes (`1e-300`, `1e-100`, `1e-50`) look arbitrary.
An inline comment explains **why this magnitude and not the usual 1e-30**:

```python
# Guard: log(0) has inf gradient in JAX; 1e-30 underflowed here (Bcubed ∝ Gamma³).
Zeta = jnp.clip(Gamma**3 / (Bcubed + 1e-300), 0.0, 1.0)
```

## The cardinal rule: never interconvert guard operators

```python
x + eps          # → grad(f(x + eps)) ≈ f'(x)   always
maximum(x, eps)  # → grad = 0 when x < eps       hard switch
where(cond, a, b)# → only the selected branch's grad flows
```

These have **different boundary derivatives**. In a differentiable code, a
mechanical swap (e.g. `x + eps` → `maximum(x, eps)`) **silently changes a
gradient** — a correctness defect with no test failure. Never convert between
them without understanding the AD implications.

The house pattern for deliberately controlling a divergent gradient is a
targeted `custom_jvp` (see `microphysics/safe_math.py`), not a blanket
guard wrapper.
