# Definition of Done

What it means for a capability to be **delivered** in stellar-jax. The bar is
physics-first and honesty-first: a capability counts as done only when its claims
are validated by green tests, its gradients are correct in sign and magnitude, and
its limitations are stated openly rather than hidden.

## Rule 1: No capability-critical gradient may be liveness-only

A capability cannot be claimed delivered while any capability-critical gradient or
observable is only checked for *liveness* — i.e. the test asserts a gradient is
non-zero ("both paths live") but not that it has the correct **sign** and
**magnitude** against an independent finite-difference.

### What "delivered" means for a gradient column

A Jacobian column ∂(observable)/∂(parameter) is **delivered** when its test:

1. Asserts `AD_grad · FD_grad > 0` (correct sign).
2. Asserts `|AD_grad - FD_grad| / |FD_grad| < tol` (correct magnitude).
3. Is gated by a **per-column** mutation that provably breaks exactly that
   column's test (not a shared machinery mutation).
4. Passes green (not expected-failure, not skipped).

A column that cannot yet meet this bar must be marked as an expected failure with a
reason, and documented as a known limitation. It is never counted as delivered.

## Rule 2: No proxy observable may pass as delivered

A reported uncertainty σ(param) must derive from a **real**
`d(observable)/d(parameter)`, not from a closed-form scaling of another σ (e.g.
`σ(age) = f_age × σ(M)/M`). A proxy must be marked as an expected failure with a
reason, documented as a **limitation**, and NOT counted as a delivered capability.

### The meta-test gate

`tests/test_validate.py::test_no_proxy_observable_passes_as_delivered`
AST-parses the forecast module and fails if a proxy pattern is detected while its
test passes green — a structural guard that a proxy cannot be merged as "delivered".

## Rule 3: Every "honest limitation" has a matching signal

A docstring or code comment that says "this is a limitation" / "this is
approximate" is **not sufficient** by itself. Every such disclosure must have a
**failing or expected-failure test** that makes the limitation visible. A
limitation documented only in a comment is invisible to the test suite — which is
exactly how a wrong-sign gradient can slip through behind a sign-forgiving test.

## Rule 4: Per-column mutation gating

Each capability-column test must be gated by its **own** mutation — one that
provably breaks exactly that column and no other. Shared mutations (e.g. zeroing
the entire backward pass) prove the machinery works, but not that a specific
column's gradient is correct. Per-column mutations:

- `flip_sign_jacobian_Y_init` — negates ∂σ²/∂Y_init.
- `flip_sign_jacobian_Z` — negates ∂σ²/∂Z.
- `flip_sign_jacobian_alpha_mlt` — negates ∂σ²/∂α_MLT.
- `f_ov_detach` — severs ∂σ²/∂f_ov via stop_gradient(f_ov) in mix_composition.
- `corrupt_age_jacobian` — zeros the f_age scaling.

## Checklist for delivering a capability

- [ ] Every capability-critical gradient column passes sign+tol (Rule 1).
- [ ] No column is an expected failure — all are green.
- [ ] No proxy observable passes as delivered (Rule 2).
- [ ] Every column has a per-column mutation (Rule 4).
- [ ] Every "limitation" docstring has a matching failing/expected-failure test (Rule 3).
- [ ] The meta-tests in `tests/test_validate.py` pass green.

## References

- `tests/test_validate.py`: the meta-test file enforcing Rules 1-3.
- `tests/mutations.py`: per-column mutation definitions (Rule 4).
- `docs/reference/gradient-accuracy.md`: gradient tier documentation.
