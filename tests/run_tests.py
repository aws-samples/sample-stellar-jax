"""Standard acceptance test for stellar-jax versions. Run on EC2.
Usage: python3.11 run_tests.py <version_dir>
Example: python3.11 run_tests.py ~/stellar-versions/v3
"""
import sys, os, time, importlib

if len(sys.argv) < 2:
    print("Usage: python3.11 run_tests.py <version_dir>")
    sys.exit(1)

version_dir = os.path.expanduser(sys.argv[1])
sys.path.insert(0, version_dir)

# Find the stellar module (handles hyphens in filenames)
stellar_file = [f for f in os.listdir(version_dir) if f.startswith('stellar_') and f.endswith('.py')][0]
module_name = stellar_file[:-3].replace('-', '_')

# Import via spec loader to handle hyphens
import importlib.util
spec = importlib.util.spec_from_file_location(module_name, os.path.join(version_dir, stellar_file))
stellar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stellar)

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

print(f"=== Testing: {stellar_file} ===\n")

# Test 1: Evolution for multiple masses
print("--- Test 1: Evolution (max_steps=200) ---")
for mass in [1.0, 1.2, 1.5, 2.0]:
    t0 = time.time()
    r = stellar.evolve_star(mass, Z=0.014, max_steps=200)
    dt = time.time() - t0
    print(f"  {mass} Msun: log_L={float(r['log_L'][-1]):.4f}, "
          f"log_Teff={float(r['log_Teff'][-1]):.4f}, "
          f"log_R={float(r['log_R'][-1]):.4f}, "
          f"age={float(r['star_age'][-1])/1e9:.2f} Gyr  ({dt:.1f}s)")

# Test 2: Gradient
print("\n--- Test 2: Gradient ---")
grad_fn = jax.grad(lambda m: stellar.evolve_star(m, Z=0.02, max_steps=5)['log_L'][-1])
_ = grad_fn(1.0)  # warm-up
t0 = time.time()
g = grad_fn(1.0)
dt = time.time() - t0
print(f"  d(log_L)/dM = {float(g):.4f}  ({dt*1000:.1f} ms post-JIT)")
assert jnp.isfinite(g) and g != 0.0, f"FAIL: gradient is {g}"

# Test 3: Timing (post-JIT, 5-step)
print("\n--- Test 3: Post-JIT timing ---")
# Forward
fwd_fn = jax.jit(lambda: stellar.evolve_star(1.0, Z=0.014, max_steps=5))
_ = fwd_fn()
t0 = time.time()
_ = fwd_fn()
dt_fwd = time.time() - t0

# Gradient
_ = grad_fn(1.0)
t0 = time.time()
_ = grad_fn(1.0)
dt_grad = time.time() - t0
print(f"  Forward (5 steps): {dt_fwd*1000:.1f} ms")
print(f"  Gradient (5 steps): {dt_grad*1000:.1f} ms")

print("\n=== ALL TESTS PASSED ===")
