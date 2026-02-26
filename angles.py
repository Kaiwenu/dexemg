import numpy as np

# --- Helper Utilities ---

def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > eps else v * 0.0

def _safe_acos(x: float) -> float:
    return float(np.arccos(np.clip(x, -1.0, 1.0)))

def _bend_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Standard 3-point flexion formula: pi - acos(dot(u1, u2))"""
    u1 = _normalize(v1)
    u2 = _normalize(v2)
    return float(np.pi - _safe_acos(np.dot(u1, u2)))

def project_onto_plane(v: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
    return v - np.dot(v, n_hat) * n_hat

# --- Config ---

BIASES = {"index": 0.08, "middle": 0.02, "ring": 0.06, "pinky": -0.02}
CLIPPING_LIMIT = np.radians(10.0)

# MediaPipe Indices for the 4 fingers: (MCP, PIP, DIP, TIP)
FINGER_MAP = {
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20)
}

WRIST = 0

def compute_finger_flexions(P: np.ndarray) -> dict:
    flex_results = {}
    for name, (mcp, pip, dip, tip) in FINGER_MAP.items():
        # MCP: Angle between the Metacarpal (wrist->mcp) and Proximal (mcp->pip)
        # PIP: Angle between Proximal (mcp->pip) and Intermediate (pip->dip)
        # DIP: Angle between Intermediate (pip->dip) and Distal (dip->tip)
        flex_results[name] = [
            _bend_angle(P[mcp] - P[0],   P[pip] - P[mcp]), # MCP
            _bend_angle(P[pip] - P[mcp], P[dip] - P[pip]), # PIP
            _bend_angle(P[dip] - P[pip], P[tip] - P[dip])  # DIP
        ]
    return flex_results

def compute_abduction(L: np.ndarray) -> dict:
    """Calculates clipped abduction for the 4 main fingers"""
    v0_5, v0_9, v0_13, v0_17 = _normalize(L[5]-L[0]), _normalize(L[9]-L[0]), _normalize(L[13]-L[0]), _normalize(L[17]-L[0])

    # Reference Normals
    n_idx = -_normalize(np.cross(v0_5, v0_9))
    # n_mid = -_normalize(np.cross(v0_9, v0_13) + np.cross(v0_13, v0_17))
    n_mid = -_normalize(np.cross(v0_5, v0_9) + np.cross(v0_9, v0_13))
    n_ring = -_normalize(np.cross(v0_9, v0_13) + np.cross(v0_13, v0_17))
    n_pky = -_normalize(np.cross(v0_13, v0_17))
    
    normals = {"index": n_idx, "middle": n_mid, "ring": n_ring, "pinky": n_pky}
    abd_results = {}

    for name, (mcp, pip, _, _) in FINGER_MAP.items():
        u = _normalize(L[mcp] - L[0])
        w = _normalize(np.cross(u, normals[name]))
        f = _normalize(L[pip] - L[mcp])
        
        raw_theta = -_safe_acos(np.dot(w, f))
        abd_results[name] = np.clip(raw_theta - BIASES[name], -CLIPPING_LIMIT, CLIPPING_LIMIT)
    
    return abd_results

def compute_thumb_angles(P: np.ndarray, is_right_hand=False) -> np.ndarray:
    """Calculates the 4 thumb angles"""
    side_factor = -1.0 if is_right_hand else 1.0

    # R_index frame
    x_idx = _normalize(P[5] - P[0])
    n_idx = side_factor * _normalize(np.cross(x_idx, _normalize(P[9] - P[0])))
    R_index = np.stack([x_idx, np.cross(n_idx, x_idx), n_idx], axis=1)

    # Thumb Axes
    a0 = R_index @ _normalize(np.array([3.0, 0.0, -1.0]))
    a1 = R_index @ _normalize(np.array([-0.2, -3.0, -0.6]))

    # CMC Angles
    v_ref, v_thumb = P[5]-P[0], P[3]-P[0]
    theta_0 = 1 - _safe_acos(np.dot(_normalize(project_onto_plane(v_ref, a0)), _normalize(project_onto_plane(v_thumb, a0))))
    theta_1 = np.radians(0.0) + _safe_acos(np.dot(_normalize(project_onto_plane(v_ref, a1)), _normalize(project_onto_plane(v_thumb, a1))))
    
    # Flexions (using points 1-2-3 and 2-3-4 from MediaPipe)
    angle_2 = _bend_angle(P[1]-P[2], P[3]-P[2])
    angle_3 = _bend_angle(P[2]-P[3], P[4]-P[3])
    
    return np.array([theta_0, theta_1, angle_2, angle_3]), (a0, a1, P[0])

def compute_all_hand_angles(landmarks: np.ndarray) -> np.ndarray:
    """Combines all logic into a single 20-element label vector."""
    thumb, debug = compute_thumb_angles(landmarks)
    flexions = compute_finger_flexions(landmarks)
    abductions = compute_abduction(landmarks)
    
    # Pack into vector: Thumb(4), Index(4), Middle(4), Ring(4), Pinky(4)
    y = list(thumb)
    for finger in ["index", "middle", "ring", "pinky"]:
        y.extend(flexions[finger])
        y.append(abductions[finger])
        
    return np.array(y, dtype=np.float32)