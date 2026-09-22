# demos/

Runnable, collaborator-facing demonstrations of what `stellar-jax` computes that a
non-differentiable stellar code does not.

- **[`differentiable-seismic-gradients/`](differentiable-seismic-gradients/)** — analytic gradients
  of oscillation observables (∂ν/∂physics) computed end-to-end through the physics in a single
  reverse-mode pass: across mass, opacity, nuclear rate, metallicity, helium (∂ν/∂Y, historically
  wrong-sign, now correct), mixing length (validated into the evolved regime), overshoot, and the
  surface-independent frequency ratios — plus sound-speed structure kernels ∂ν/∂c²(r) validated
  against ADIPLS on both a radiative-core and a convective-core star. Each gradient maps to a
  committed CI test.

The folder's `README.md` states the capability, why it is useful, the
grounding sources, the honest scope and limitations, and how every claim maps to a test.

A ready-to-run container is provided ([`Dockerfile`](Dockerfile)) so the figures reproduce with no
local setup.

Gradient-driven calibration of uncertain physics against a real star (16 Cygni A) is a related
capability: the calibration machinery is in the package and CI-tested, and a runnable real-star
demonstration is in progress.
