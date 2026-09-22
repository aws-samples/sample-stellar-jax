# Ready-to-run container for the stellar-jax collaborator demos.
#
# Build (from the repo ROOT, so the whole package is in the build context):
#   docker build -f demos/Dockerfile -t stellar-jax-demo .
#
# Run the differentiable-seismic-gradients demo (~1–3 min, CPU-only):
#   docker run --rm stellar-jax-demo
#
# Run a specific part, or copy the generated figures out:
#   docker run --rm stellar-jax-demo python demos/differentiable-seismic-gradients/demo_seismic_gradients.py --part2
#   docker run --rm -v "$PWD/out:/out" stellar-jax-demo \
#       sh -c "python demos/differentiable-seismic-gradients/demo_seismic_gradients.py && cp demos/differentiable-seismic-gradients/figures/*.png /out/"
#
# Everything is pure-Python (JAX CPU); no GPU, no system libraries, no network at runtime.

FROM python:3.11-slim

# Non-interactive, unbuffered stdout so the demo's step-by-step output streams live.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Copy the whole package (data tables are baked into src/stellar_jax/data, so no downloads).
COPY . /app

# Install the package + its pinned deps (jax==0.10.2, jaxlib==0.10.2, numpy, matplotlib).
# `pip install -e .` reads pyproject.toml, which now lists matplotlib as a core dependency.
RUN pip install -e .

# Default: run the demo end-to-end (~1–3 min, no evolve_star XLA compile).
CMD ["python", "demos/differentiable-seismic-gradients/demo_seismic_gradients.py"]
