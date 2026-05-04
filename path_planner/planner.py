"""
FCM1-based path planner using dynamic programming with iterative refinement.

Finds the sequence of unit contact normals for a given ordered set of contact
points that maximises the Path Form Closure Margin (PFCM) — the minimum FCM1
over all sliding windows of m consecutive contacts.  Ties in PFCM are broken
by maximising total FCM1.

Optimisation strategy
---------------------
* Discretise the angle space to n_a candidates per contact per round.
* Forward DP: prune any (state, action) combination where FCM1 ≤ 0.
* DP state  = last (m−1) angle indices.
* DP value  = (min_fcm, total_fcm), compared lexicographically (max-min first).
* Lower-bound pruning: in refinement rounds, any state whose running min FCM1
  already falls below the PFCM from the previous round is pruned immediately.
  This keeps the number of live combinations small in rounds 2+.
* The (B, m, 2) normals tensor is never materialised in full; FCM is computed
  in chunks of chunk_size to bound peak memory.
* Numba JIT (if available) handles the inner FCM loop with zero heap allocation.
"""

from __future__ import annotations

import warnings
from typing import Optional
from tqdm import trange

import numpy as np

from .fcm import compute_fcm, compute_fcm_batch

_DEFAULT_CHUNK = 200_000


class FCMPathPlanner:
    """DP-based planner maximising the Path Form Closure Margin.

    Parameters
    ----------
    P : (nc, 2) or (2, nc) array
        Ordered contact point positions.
    m : int
        Sliding window size (simultaneous contacts). Must be ≥ 4.
    n_a : int
        Number of candidate angles per contact per DP round (default 18).
    n_iter : int
        Total rounds including the initial coarse pass (default 3).
    initial_normals : (m, 2) or (m,) array, optional
        Fixed unit normals (or angles in radians) for the first m contacts.
        These are never refined — they appear as single-candidate entries
        throughout all rounds.
    chunk_size : int
        Number of (state, action) combinations processed per FCM batch.
        Reducing this lowers peak memory (default 200 000).
    """

    def __init__(
        self,
        P: np.ndarray,
        m: int,
        n_a: int = 18,
        n_iter: int = 3,
        initial_normals: Optional[np.ndarray] = None,
        chunk_size: int = _DEFAULT_CHUNK,
    ) -> None:
        P = np.asarray(P, dtype=float)
        if P.ndim != 2:
            raise ValueError("P must be a 2-D array.")
        if P.shape[0] == 2 and P.shape[1] != 2:
            P = P.T
        self.P = P
        self.nc = P.shape[0]
        self.m = m
        self.n_a = n_a
        self.n_iter = max(1, n_iter)
        self.chunk_size = max(1, chunk_size)

        if m < 4:
            raise ValueError("m must be ≥ 4 for 1st-order form closure in 2D.")
        if self.nc < m:
            raise ValueError(f"Number of contacts nc={self.nc} must be ≥ m={m}.")

        state_space = n_a ** (m - 1)
        if state_space > 500_000:
            warnings.warn(
                f"Initial state space n_a^(m−1) = {n_a}^{m - 1} = {state_space:,}. "
                "This may be slow. Consider reducing n_a for the first round.",
                stacklevel=2,
            )

        self._init_normals: Optional[np.ndarray] = None
        if initial_normals is not None:
            arr = np.asarray(initial_normals, dtype=float)
            if arr.ndim == 1:
                arr = np.column_stack([np.cos(arr), np.sin(arr)])
            if arr.shape != (m, 2):
                raise ValueError(
                    f"initial_normals must have shape ({m}, 2) or ({m},), "
                    f"got {arr.shape}."
                )
            nrm = np.linalg.norm(arr, axis=1, keepdims=True)
            self._init_normals = arr / nrm   # (m, 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> np.ndarray:
        """Run DP with iterative refinement and return optimal normals.

        Returns
        -------
        (nc, 2) float
            Optimal unit contact normals in the same order as P.

        Raises
        ------
        RuntimeError
            If the first (coarse) round finds no feasible solution.
        """
        angle_cands = self._build_angle_sets_round1()
        best_normals = self._run_round(angle_cands, lower_bound=0.0)

        if best_normals is None:
            raise RuntimeError(
                "No feasible path with PFCM > 0 found in the initial round. "
                "Try a larger n_a, adjust initial_normals, or verify the "
                "contact geometry."
            )

        # PFCM of current solution — used as lower bound for subsequent rounds
        current_pfcm = self._pfcm(best_normals)
        print(current_pfcm)
        # print(angle_cands[0][1]-angle_cands[0][0])
        spread = np.pi
        for _ in range(self.n_iter - 1):
            spread *= 1. / self.m
            best_angles = np.arctan2(best_normals[:, 1], best_normals[:, 0])
            angle_cands = self._build_angle_sets_refine(best_angles, spread)
            refined = self._run_round(angle_cands, lower_bound=current_pfcm)
            if refined is not None:
                best_normals = refined
                current_pfcm = self._pfcm(best_normals)
            # print(angle_cands[0][1]-angle_cands[0][0])
            print(current_pfcm)

        return best_normals

    def _pfcm(self, normals: np.ndarray) -> float:
        """Compute the PFCM (minimum FCM1 over all windows) for a solution."""
        return min(
            compute_fcm(self.P[k : k + self.m], normals[k : k + self.m])
            for k in range(self.nc - self.m + 1)
        )

    # ------------------------------------------------------------------
    # Internal: angle candidate construction
    # ------------------------------------------------------------------

    def _build_angle_sets_round1(self) -> list[np.ndarray]:
        thetas = np.linspace(0.0, 2.0 * np.pi, self.n_a, endpoint=False)
        uniform = np.column_stack([np.cos(thetas), np.sin(thetas)])   # (n_a, 2)
        cands = []
        for i in range(self.nc):
            if self._init_normals is not None and i < self.m:
                cands.append(self._init_normals[i : i + 1])   # (1, 2) — pinned
            else:
                cands.append(uniform)
        return cands

    def _build_angle_sets_refine(
        self,
        best_angles: np.ndarray,
        spread: float,
    ) -> list[np.ndarray]:
        cands = []
        for i in range(self.nc):
            if self._init_normals is not None and i < self.m:
                cands.append(self._init_normals[i : i + 1])   # (1, 2) — pinned
            else:
                thetas = np.linspace(
                    best_angles[i] - spread, best_angles[i] + spread, self.n_a
                )
                cands.append(np.column_stack([np.cos(thetas), np.sin(thetas)]))
        return cands

    # ------------------------------------------------------------------
    # Internal: DP
    # ------------------------------------------------------------------

    def _run_round(
        self,
        angle_cands: list[np.ndarray],
        lower_bound: float = 0.0,
    ) -> Optional[np.ndarray]:
        """One full DP pass.

        Parameters
        ----------
        angle_cands : list of (n_a_i, 2) arrays, length nc
        lower_bound : float
            Any path whose running-min FCM1 falls below this value is pruned.
            Set to the PFCM of the previous round to skip provably sub-optimal
            branches early.

        Returns
        -------
        (nc, 2) normals or None if no feasible path survives pruning.
        """
        nc, m, n_a = self.nc, self.m, self.n_a
        chunk_size = self.chunk_size

        # Encoding: state = (a_{k-m+2}, ..., a_k) → int via mixed-radix with base n_a
        bases = n_a ** np.arange(m - 1, dtype=np.int64)   # (m-1,)

        # --- Initialise ---
        if self._init_normals is not None:
            live_states = np.zeros((1, m - 1), dtype=np.int32)
            live_values = np.zeros((1, 2), dtype=np.float64)
        else:
            grids = np.meshgrid(
                *[np.arange(n_a, dtype=np.int32) for _ in range(m - 1)],
                indexing="ij",
            )
            live_states = np.column_stack([g.ravel() for g in grids])
            live_values = np.zeros((len(live_states), 2), dtype=np.float64)

        back_ptrs: list[Optional[dict]] = [None] * nc

        # --- Forward DP ---
        for k in trange(m - 1, nc):
            K = len(live_states)
            if K == 0:
                return None

            n_a_k = len(angle_cands[k])
            B = K * n_a_k
            window_pts = self.P[k - m + 1 : k + 1]          # (m, 2)
            cands_win  = angle_cands[k - m + 1 : k + 1]     # list of m arrays
            first_win  = (k == m - 1)

            # Collect valid combinations from all chunks
            v_states   : list[np.ndarray] = []
            v_actions  : list[np.ndarray] = []
            v_new_min  : list[np.ndarray] = []
            v_new_total: list[np.ndarray] = []
            v_prev_enc : list[np.ndarray] = []

            for c_start in range(0, B, chunk_size):
                c_end = min(c_start + chunk_size, B)
                sz = c_end - c_start

                c_parent = np.arange(c_start, c_end, dtype=np.int32) // n_a_k
                c_action = np.arange(c_start, c_end, dtype=np.int32) % n_a_k
                c_state  = live_states[c_parent]   # (sz, m-1)

                # Build normals for this chunk: (sz, m, 2)
                normals_chunk = np.empty((sz, m, 2), dtype=np.float64)
                for j in range(m - 1):
                    normals_chunk[:, j, :] = cands_win[j][c_state[:, j]]
                normals_chunk[:, m - 1, :] = cands_win[m - 1][c_action]

                fcm_c = compute_fcm_batch(window_pts, normals_chunk)   # (sz,)

                # Running min and total for this chunk
                if first_win:
                    new_min_c   = fcm_c
                    new_total_c = fcm_c.copy()
                else:
                    c_prev_min   = live_values[c_parent, 0]
                    c_prev_total = live_values[c_parent, 1]
                    new_min_c   = np.minimum(c_prev_min, fcm_c)
                    new_total_c = c_prev_total + fcm_c

                # Prune: positive FCM AND running min meets the lower bound
                mask = (fcm_c > 0.0) & (new_min_c >= lower_bound - 1e-12)

                if np.any(mask):
                    v_states.append(c_state[mask])
                    v_actions.append(c_action[mask])
                    v_new_min.append(new_min_c[mask])
                    v_new_total.append(new_total_c[mask])
                    v_prev_enc.append(c_state[mask].astype(np.int64) @ bases)

            if not v_states:
                return None

            state_v   = np.concatenate(v_states,    axis=0)
            action_v  = np.concatenate(v_actions,   axis=0)
            new_min   = np.concatenate(v_new_min,   axis=0)
            new_total = np.concatenate(v_new_total, axis=0)
            prev_enc  = np.concatenate(v_prev_enc,  axis=0)

            # New state: drop oldest angle index, append new action
            new_states = np.concatenate(
                [state_v[:, 1:], action_v[:, np.newaxis]], axis=1
            )   # (V, m-1)
            new_enc = new_states.astype(np.int64) @ bases   # (V,)

            # Dedup: keep best (new_min DESC, new_total DESC) per unique new_enc
            sort_ord = np.lexsort((-new_total, -new_min, new_enc))
            sorted_enc = new_enc[sort_ord]
            _, first_occ = np.unique(sorted_enc, return_index=True)
            best_v = sort_ord[first_occ]

            live_states = new_states[best_v]
            live_values = np.column_stack([new_min[best_v], new_total[best_v]])

            # Back pointers (vectorised construction)
            bp_keys = new_enc[best_v].tolist()
            bp_vals = list(zip(prev_enc[best_v].tolist(), action_v[best_v].tolist()))
            back_ptrs[k] = dict(zip(bp_keys, bp_vals))

        return self._backtrack(back_ptrs, live_states, live_values, angle_cands, n_a, bases)

    # ------------------------------------------------------------------
    # Internal: backtracking
    # ------------------------------------------------------------------

    def _backtrack(
        self,
        back_ptrs: list,
        live_states: np.ndarray,
        live_values: np.ndarray,
        angle_cands: list[np.ndarray],
        n_a: int,
        bases: np.ndarray,
    ) -> np.ndarray:
        nc, m = self.nc, self.m

        # Best final state: lexicographic max of (min_fcm, total_fcm)
        best_k = int(np.lexsort((-live_values[:, 1], -live_values[:, 0]))[0])
        current_enc = int(live_states[best_k] @ bases)

        angle_indices = np.empty(nc, dtype=np.int32)

        # Trace from step nc-1 down to step m-1; action at step k = a_k
        for k in range(nc - 1, m - 2, -1):
            prev_enc, action = back_ptrs[k][current_enc]
            angle_indices[k] = action
            current_enc = prev_enc

        # Decode initial state (contacts 0..m-2) from the residual encoding
        enc = current_enc
        for j in range(m - 1):
            angle_indices[j] = enc % n_a
            enc //= n_a

        # Map indices → unit normals
        result = np.empty((nc, 2), dtype=np.float64)
        for i in range(nc):
            result[i] = angle_cands[i][angle_indices[i]]
        return result
