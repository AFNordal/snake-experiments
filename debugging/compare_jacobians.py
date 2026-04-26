#!/usr/bin/env python3
"""
Compare the symbolic (lambdified) and analytical Jacobian implementations
for correctness and timing.

The symbolic approach (current):
  1. Pre-build J_func for all n_links contacts at init time.
  2. Each tick: inflate qe to full size, call J_func, then deflate to active contacts.

The analytical approach (proposed):
  - Compute J directly in numpy from cumulative angle sums. No inflation/deflation.
"""

import json
import pathlib
import time
import numpy as np
from SimSerpent.control.kinematics import symbolic_jacobian, jacobian

# --- Config ---
root_dir = pathlib.Path(__file__).parent.resolve()
config_root = root_dir / "configs"
with open(config_root / "snake_description.json") as f:
    snake_description = json.load(f)

n_links = snake_description["n_links"]
link_length = snake_description["link_length_m"]

# Build the symbolic J_func once (mirrors HPFCController.__init__)
print(f"Building symbolic J_func for n={n_links} links...", end="", flush=True)
t_build = time.perf_counter()
J_func_sym = symbolic_jacobian(
    n=n_links,
    nc=n_links,
    contact_links=list(range(n_links)),
    link_length=link_length,
)
print(f" {(time.perf_counter() - t_build)*1e3:.0f} ms")


def symbolic_J(phi: np.ndarray, contact_links: list[int], k: np.ndarray, phic: np.ndarray) -> np.ndarray:
    """Symbolic path: inflate → J_func → deflate (mirrors current tick())."""
    n = n_links
    nc = len(contact_links)
    link_arr = np.array(contact_links)

    inflated_qe = np.zeros(2 + n * 3)
    inflated_qe[2:2 + n] = phi
    inflated_qe[2 + n + link_arr] = k
    inflated_qe[2 + n * 2 + link_arr] = phic

    inflated_J = J_func_sym(*inflated_qe)

    row_idx = (link_arr[:, None] * 3 + np.arange(3)).ravel()
    col_idx = np.concatenate([
        np.arange(2 + n, dtype=int),
        2 + n + link_arr,
        2 + n * 2 + link_arr,
    ])
    return inflated_J[row_idx, :][:, col_idx]


# ── Correctness ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Correctness check (max |J_sym - J_ana| per trial)")
print(f"{'='*60}")

rng = np.random.default_rng(42)
all_passed = True
for trial in range(8):
    phi = rng.uniform(-0.5, 0.5, size=n_links)
    nc = rng.integers(1, 5)
    contact_links = sorted(rng.choice(n_links, size=nc, replace=False).tolist())
    k = rng.uniform(0.01, link_length * 0.9, size=nc)
    phic = rng.uniform(-np.pi, np.pi, size=nc)

    J_sym = symbolic_J(phi, contact_links, k, phic)
    J_ana = jacobian(phi, contact_links, k, link_length)

    max_err = np.max(np.abs(J_sym - J_ana))
    shapes_match = J_sym.shape == J_ana.shape
    passed = shapes_match and max_err < 1e-10
    all_passed = all_passed and passed
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] trial {trial+1}: nc={nc}, links={contact_links}, shape={J_ana.shape}, max|err|={max_err:.2e}")

print(f"\n  Overall: {'ALL PASS' if all_passed else 'FAILURES DETECTED'}")

# ── Timing ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Timing benchmark")
print(f"{'='*60}")

N_WARMUP = 20
N_BENCH = 500

for nc_target in [1, 3, 6, 10]:
    if nc_target > n_links:
        continue
    phi = rng.uniform(-0.5, 0.5, size=n_links)
    contact_links = sorted(rng.choice(n_links, size=nc_target, replace=False).tolist())
    k = np.full(nc_target, link_length * 0.5)
    phic = np.zeros(nc_target)

    for _ in range(N_WARMUP):
        symbolic_J(phi, contact_links, k, phic)
        jacobian(phi, contact_links, k, link_length)

    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        symbolic_J(phi, contact_links, k, phic)
    t_sym = (time.perf_counter() - t0) / N_BENCH * 1e3

    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        jacobian(phi, contact_links, k, link_length)
    t_ana = (time.perf_counter() - t0) / N_BENCH * 1e3

    print(f"  nc={nc_target:2d}: symbolic={t_sym:.3f} ms  analytical={t_ana:.3f} ms  speedup={t_sym/t_ana:.1f}x")
