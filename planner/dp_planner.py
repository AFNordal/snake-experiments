import numpy as np
from tqdm import tqdm
import pathlib
import sys
root_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(root_dir.parent / "fc_tools"))
from fc_visualizer import FCM1, contacts_plot

def solve_snake_fcm_dp(points, first_m_normals, m, num_directions=12):
    """
    Optimizes snake robot normals using Index-Based DP.
    
    Fixes:
    - Uses integer indices for state hashing (solves float precision bugs).
    - Snaps initial normals to nearest discrete bin to ensure graph connectivity.
    - Adds shape assertions.
    """
    # --- 1. Shape & Input Validation ---
    points = np.asarray(points)
    first_m_normals = np.asarray(first_m_normals)
    
    # Ensure shape is (2, N)
    if points.shape[0] != 2 and points.shape[1] == 2:
        points = points.T
    if first_m_normals.shape[0] != 2 and first_m_normals.shape[1] == 2:
        first_m_normals = first_m_normals.T
        
    N = points.shape[1]
    assert N >= m, f"Total points N ({N}) must be >= window size m ({m})"

    # --- 2. Discretization & Index Mapping ---
    # Generate discrete directions
    angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
    discrete_normals = np.stack([np.cos(angles), np.sin(angles)], axis=0) # (2, D)
    
    # Helper: Find nearest discrete index for a given normal vector
    def get_nearest_idx(normal_vec):
        # Dot product to find highest cosine similarity
        scores = np.dot(normal_vec, discrete_normals)
        return np.argmax(scores)

    # Snap the fixed 'first_m_normals' to their nearest discrete indices
    # This ensures the starting state connects to the DP graph
    fixed_indices = [get_nearest_idx(first_m_normals[:, j]) for j in range(m)]

    # --- 3. DP Initialization ---
    # dp[i] keys will be TUPLES OF INDICES (integers), not floats.
    dp = [{} for _ in range(N)]

    # Base Case: Window 0 (indices 0 to m-1)
    # State signature: indices of normals at [1, ..., m-1]
    initial_history_indices = tuple(fixed_indices[1:])
    
    # Calculate initial FCM using the actual (snapped) discrete normals for consistency
    initial_window_normals = discrete_normals[:, fixed_indices]
    initial_fcm = FCM1(points[:, :m], initial_window_normals)
    
    dp[m-1][initial_history_indices] = (initial_fcm, None, None)

    # --- 4. DP Transitions ---
    # We decide the normal index for point i
    for i in tqdm(range(m, N), desc="Optimizing Normals"):
        
        # Pre-slice points for efficiency
        current_window_points = points[:, i-m+1 : i+1]
        
        # Iterate over all reachable previous states
        for prev_history_indices, (prev_min_fcm, _, _) in tqdm(dp[i-1].items(), leave=False):
            
            # prev_history_indices: (idx_{i-m+1}, ..., idx_{i-1})
            # We convert these indices back to vectors for FCM calculation
            prev_normals_vec = discrete_normals[:, prev_history_indices]
            
            for d_idx in range(num_directions):
                new_normal_vec = discrete_normals[:, d_idx]
                
                # Construct full window normals: prev_history + new_choice
                # Shape (2, m-1) + (2, 1) -> (2, m)
                current_window_normals = np.hstack([prev_normals_vec, new_normal_vec.reshape(2, 1)])
                
                # Calculate FCM
                current_fcm = FCM1(current_window_points, current_window_normals)
                
                # Update Bottleneck (Max-Min)
                # new_min_fcm = prev_min_fcm + current_fcm
                new_min_fcm = min(prev_min_fcm, current_fcm)
                
                # Create new state key: drop oldest index, add new index
                new_history_indices = prev_history_indices[1:] + (d_idx,)
                
                # Logic: If this state is new, OR we found a better path to it -> Store it
                if new_history_indices not in dp[i] or new_min_fcm > dp[i][new_history_indices][0]:
                    dp[i][new_history_indices] = (new_min_fcm, prev_history_indices, d_idx)

    # --- 5. Reconstruction ---
    final_states = dp[N-1]
    if not final_states:
        print("Optimization failed: No valid states reached end.")
        return None
    
    # Find best end state
    best_history_indices = max(final_states, key=lambda k: final_states[k][0])
    maximized_min_fcm = final_states[best_history_indices][0]
    
    # Backtrack
    optimized_indices = np.zeros(N, dtype=int)
    
    # Fill the tail (N-1 down to m)
    curr_history = best_history_indices
    for i in range(N-1, m-1, -1):
        _, prev_history, d_idx = dp[i][curr_history]
        optimized_indices[i] = d_idx
        curr_history = prev_history
        
    # Fill the head (0 to m-1) using the fixed initial indices
    optimized_indices[:m] = fixed_indices
    
    # Convert indices back to vectors
    optimized_normals = discrete_normals[:, optimized_indices]
    
    return optimized_normals, maximized_min_fcm


def contacts_to_json(P, N):
    """
    P: np.ndarray of shape (2, M)  - points as columns
    N: np.ndarray of shape (2, K)  - normals as columns

    Returns a dict ready to be serialized as JSON.
    """
    num_contacts = N.shape[1]
    contacts = []

    for i in range(num_contacts):
        point = P[:, i].tolist()
        normal = N[:, i].tolist()

        side = "r" if normal[1] > 0 else "l"

        contacts.append({
            "point": point,
            "normal": normal,
            "side": side
        })

    return {"contacts": contacts}

if __name__ == "__main__":
    import numpy as np

    P = np.array([
        [-2, -1.5, -1, -0.5, 0.0,  0.5,  0.75, 1.0,  1.5,  1.75, 2.0,  2.5,  2.75, 3.0,  3.5,  3.75, 4.0,  4.5],
        [0, 0.3, 0.3, 0, 1.0,  1.0,  0.5,  0.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0,  0.0,  0.5,  1.0,  1.0]
    ])
    N = np.array([
        [1.0, 0.0, 0.0, -1.0, 1],
        [-1.0, 1.0, 1.0, -1.0, -1]
    ])
    m = 5
    num_directions = 21
    print(f"FCM of first {m} points is {FCM1(P[:, :m], N[:, :m])}")
    
    N, pfcm = solve_snake_fcm_dp(P, N, m, num_directions)
    import json
    path_dir = root_dir / "paths"
    with open(path_dir / "path_{m}_{num_directions}.json", "w") as file:
        json.dump(contacts_to_json(P, N), file)
    print(f"Path's minimal fcm is {pfcm}")
    contacts_plot(P, N)
