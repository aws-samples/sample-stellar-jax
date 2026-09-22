# Documentation

User-facing reference for stellar-jax. Start with the root [`README.md`](../README.md) for an
overview and quick-start; the pages below are the detailed reference.

## Orientation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — how the code is organized and how it works: the package map,
  the forward evolution loop, the structure solver, and the differentiability architecture. Read this
  first to understand or extend the codebase.

## Reference

- [`reference/capabilities.md`](reference/capabilities.md) — physics scope: what the solver models,
  and what is out of scope by design.
- [`reference/gradient-accuracy.md`](reference/gradient-accuracy.md) — what accuracy to expect from
  `jax.grad`: the trust budget, per-parameter/observable bounds, and the accuracy tiers.
- [`reference/validation-regimes.md`](reference/validation-regimes.md) — the validation regimes
  (MODE-A / MODE-B) and what is CI-gated against the MESA / Model S references.
- [`reference/starting-model.md`](reference/starting-model.md) — the ZAMS starting-model
  specification.
