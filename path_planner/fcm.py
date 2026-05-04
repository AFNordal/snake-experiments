"""
Vectorised computation of the 1st-order Form Closure Margin (FCM1).

Reference: Gravdahl et al. (2024), "Form Closure based Path Planning in
Obstacle-Aided Locomotion".

FCM1 definition (for a window of m contacts):
  For every pair (i, j) of contacts, find the intersection s_{i,j} of the two
  contact orthogonal lines (through p_i along n̂_i, and through p_j along n̂_j).
  For each valid pole s_{i,j}:
    r_k   = p_k − s_{i,j}
    δr_k  = (−r_y, r_x) / ‖r‖         (normalised perpendicular displacement)
    dot_k = δr_k · n̂_k
    t_ij  = −(max_k dot_k) · (min_k dot_k)   (pole twist index)
  FCM1 = min over all valid poles of t_{i,j}
       = −∞  if no valid pole exists (all contact normals parallel)

Two implementations are provided:
  * Numba JIT (primary): scalar loops, zero heap allocation, auto-parallelised
    over the B dimension.  Compiled on first call (result cached).
  * NumPy (fallback): used automatically when numba is not available.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Numba implementation (preferred)
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=True)
    def _fcm_batch_numba(window_points: np.ndarray,
                         normals_batch: np.ndarray,
                         eps: float) -> np.ndarray:
        """Numba-JIT FCM1 kernel.  All work is scalar — no heap allocations."""
        B = normals_batch.shape[0]
        m = normals_batch.shape[1]
        result = np.empty(B, dtype=np.float64)

        for b in prange(B):   # noqa: E741  — parallel over batch
            min_tij = np.inf
            has_pole = False

            for i in range(m):
                ni_x = normals_batch[b, i, 0]
                ni_y = normals_batch[b, i, 1]
                pi_x = window_points[i, 0]
                pi_y = window_points[i, 1]

                for j in range(i + 1, m):
                    nj_x = normals_batch[b, j, 0]
                    nj_y = normals_batch[b, j, 1]
                    pj_x = window_points[j, 0]
                    pj_y = window_points[j, 1]

                    det = nj_x * ni_y - ni_x * nj_y
                    if det * det < eps * eps:    # parallel normals — no pole
                        continue

                    dp_x = pj_x - pi_x
                    dp_y = pj_y - pi_y
                    t = (-nj_y * dp_x + nj_x * dp_y) / det
                    pole_x = pi_x + t * ni_x
                    pole_y = pi_y + t * ni_y

                    has_pole = True
                    dot_max = -np.inf
                    dot_min = np.inf

                    for kk in range(m):
                        r_kx = window_points[kk, 0] - pole_x
                        r_ky = window_points[kk, 1] - pole_y
                        r_norm = (r_kx * r_kx + r_ky * r_ky) ** 0.5
                        if r_norm < 1e-15:
                            r_norm = 1e-15
                        dr_x = -r_ky / r_norm
                        dr_y =  r_kx / r_norm
                        dot = (dr_x * normals_batch[b, kk, 0]
                               + dr_y * normals_batch[b, kk, 1])
                        if dot > dot_max:
                            dot_max = dot
                        if dot < dot_min:
                            dot_min = dot

                    tij = -(dot_max * dot_min)
                    if tij < min_tij:
                        min_tij = tij

            result[b] = min_tij if has_pole else -np.inf

        return result

    def compute_fcm_batch(
        window_points: np.ndarray,
        normals_batch: np.ndarray,
        eps: float = 1e-10,
    ) -> np.ndarray:
        """Compute FCM1 for a batch of normal assignments (numba backend).

        Parameters
        ----------
        window_points : (m, 2) float
        normals_batch : (B, m, 2) float — unit contact normals
        eps : float — parallel-normals threshold

        Returns
        -------
        (B,) float — FCM1 values; negative (or -inf) = no form closure
        """
        wp = np.ascontiguousarray(window_points, dtype=np.float64)
        nb = np.ascontiguousarray(normals_batch, dtype=np.float64)
        return _fcm_batch_numba(wp, nb, float(eps))

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# NumPy fallback (used when numba is absent)
# ---------------------------------------------------------------------------
if not _NUMBA_AVAILABLE:
    def compute_fcm_batch(   # type: ignore[misc]
        window_points: np.ndarray,
        normals_batch: np.ndarray,
        eps: float = 1e-10,
    ) -> np.ndarray:
        """Compute FCM1 for a batch of normal assignments (NumPy fallback).

        Parameters
        ----------
        window_points : (m, 2) float
        normals_batch : (B, m, 2) float — unit contact normals
        eps : float — parallel-normals threshold

        Returns
        -------
        (B,) float — FCM1 values; negative (or -inf) = no form closure
        """
        B, m, _ = normals_batch.shape
        running_min = np.full(B, np.inf)
        has_pole = np.zeros(B, dtype=bool)

        for i in range(m):
            for j in range(i + 1, m):
                pi = window_points[i]           # (2,)
                pj = window_points[j]           # (2,)
                ni = normals_batch[:, i, :]     # (B, 2)
                nj = normals_batch[:, j, :]     # (B, 2)
                dp = pj - pi                    # (2,)

                det = nj[:, 0] * ni[:, 1] - ni[:, 0] * nj[:, 1]   # (B,)
                parallel = np.abs(det) < eps

                safe_det = np.where(parallel, 1.0, det)
                t = (-nj[:, 1] * dp[0] + nj[:, 0] * dp[1]) / safe_det   # (B,)
                poles = pi + t[:, np.newaxis] * ni                         # (B, 2)

                r = window_points[np.newaxis, :, :] - poles[:, np.newaxis, :]   # (B, m, 2)
                r_norm = np.sqrt(r[:, :, 0] ** 2 + r[:, :, 1] ** 2)
                r_norm = np.maximum(r_norm, 1e-15)
                dr_x = -r[:, :, 1] / r_norm
                dr_y =  r[:, :, 0] / r_norm

                dots = dr_x * normals_batch[:, :, 0] + dr_y * normals_batch[:, :, 1]
                t_ij = -(dots.max(axis=1) * dots.min(axis=1))

                valid = ~parallel
                has_pole |= valid
                running_min = np.where(valid, np.minimum(running_min, t_ij), running_min)

        return np.where(has_pole, running_min, -np.inf)


# ---------------------------------------------------------------------------
# Scalar convenience wrapper
# ---------------------------------------------------------------------------
def compute_fcm(
    window_points: np.ndarray,
    normals: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """Compute FCM1 for a single window.

    Parameters
    ----------
    window_points : (m, 2) float
    normals : (m, 2) float — unit contact normals
    eps : float

    Returns
    -------
    float
    """
    return float(compute_fcm_batch(window_points, normals[np.newaxis, :, :], eps)[0])
