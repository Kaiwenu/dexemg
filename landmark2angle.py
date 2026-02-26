import math

import numpy as np

EPS = 1e-12

import numpy as np
import angles as ang
EPS = 1e-12

def normalize(v, eps=EPS):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n + eps)

def angle_at_B(A, B, C, eps=EPS):
    BA = A - B
    BC = C - B
    cos_theta = np.dot(BA, BC) / (
        (np.linalg.norm(BA) * np.linalg.norm(BC)) + eps
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def project_onto_plane(v, n):
    """Project vector v onto plane with normal n"""
    n_hat = normalize(n)
    return v - np.dot(v, n_hat) * n_hat


def compute_palm_frame(landmarks):
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    pinky_mcp = landmarks[17]

    palm_x = normalize(middle_mcp - wrist)
    palm_z = normalize(np.cross(palm_x, pinky_mcp - wrist))
    palm_y = np.cross(palm_z, palm_x)

    return palm_x, palm_y, palm_z


FINGER_MCP = {
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}

def index_reference_normal(landmarks):
    """
    n̂_index = -normalize( (L5 - L0) × (L9 - L0) )
    """
    L = np.asarray(landmarks, dtype=np.float64)
    n = np.cross(L[5] - L[0], L[9] - L[0])
    return -normalize(n)


def middle_reference_normal(landmarks):
    """
    n̂_middle = -normalize( (L5 - L0) × (L9 - L0)
                          + (L9 - L0) × (L13 - L0) )
    """
    L = np.asarray(landmarks, dtype=np.float64)
    n = (
        np.cross(L[5] - L[0], L[9] - L[0]) +
        np.cross(L[9] - L[0], L[13] - L[0])
    )
    return -normalize(n)


def ring_reference_normal(landmarks):
    """
    n̂_ring = -normalize( (L9 - L0) × (L13 - L0)
                        + (L13 - L0) × (L17 - L0) )
    """
    L = np.asarray(landmarks, dtype=np.float64)
    n = (
        np.cross(L[9] - L[0], L[13] - L[0]) +
        np.cross(L[13] - L[0], L[17] - L[0])
    )
    return -normalize(n)


def pinky_reference_normal(landmarks):
    """
    n̂_pinky = -normalize( (L13 - L0) × (L17 - L0) )
    """
    L = np.asarray(landmarks, dtype=np.float64)
    n = np.cross(L[13] - L[0], L[17] - L[0])
    return -normalize(n)

FLEXION_TRIPLETS = [
    # Index
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    # Middle
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    # Ring
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    # Pinky
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
    # Thumb MCP / IP handled later
]

def compute_flexion_angles(landmarks):
    angles = []
    for A, B, C in FLEXION_TRIPLETS:
        theta = np.pi - angle_at_B(
            landmarks[A], landmarks[B], landmarks[C]
        )
        angles.append(theta)
    return angles
mcp = [5, 9, 13, 17]
pip = [6, 10, 14, 18]

def compute_abduction_angles(landmarks, n_finger):
    ans = []
    L = np.asarray(landmarks, dtype=np.float64)
    for i in range(4):
        # Step 1: expected finger direction
        u_hat = normalize(L[mcp[i]] - L[0])

        # Step 2: perpendicular direction
        w_hat = normalize(np.cross(u_hat, n_finger[i]))

        # Step 3: actual finger direction
        f_hat = normalize(L[pip[i]] - L[mcp[i]])

        # Step 4: abduction angle
        sin_theta = np.clip(np.dot(w_hat, f_hat), -1.0, 1.0)
        theta_abduction = -np.arcsin(sin_theta)
        # print(theta_abduction)
        ans.append(theta_abduction)
    
    return ans


# def compute_thumb_angles(L):
#     x_hat = normalize(L[5] - L[0])  # normalize(L5 - L0)
#     z_hat = -index_reference_normal(L)
#     y_hat = normalize(np.cross(z_hat, x_hat))
#     x_hat = normalize(np.cross(y_hat, z_hat))  # re-orthogonalize
#     R_index = np.column_stack([x_hat, y_hat, z_hat])  # columns are basis axes

#     a0_local = np.array([3.0, 0.0, -1.0], dtype=np.float64)
#     a1_local = np.array([-0.2, -3.0, -0.6], dtype=np.float64)
#     a0 = R_index @ normalize(a0_local)
#     a1 = R_index @ normalize(a1_local)
#     u = L[5] - L[0]
#     v = L[3] - L[0]

#     def proj_perp_axis(v, a):
#         a_hat = normalize(a)
#         return v - np.dot(v, a_hat) * a_hat

#     def arccos_projected_dot(u, v, a):
#         up = proj_perp_axis(u, a)
#         vp = proj_perp_axis(v, a)
#         up_hat = normalize(up)
#         vp_hat = normalize(vp)
#         cos_t = np.clip(np.dot(up_hat, vp_hat), -1.0, 1.0)
#         return np.arccos(cos_t)

#     # def angle_projected_atan2(u, v, a):
#     #     # Project u and v onto plane perpendicular to axis a
#     #     a_hat = normalize(a)
#     #     up = u - np.dot(u, a_hat) * a_hat
#     #     vp = v - np.dot(v, a_hat) * a_hat

#     #     up_hat = normalize(up)
#     #     vp_hat = normalize(vp)

#     #     # signed angle around a_hat
#     #     sin_t = np.dot(np.cross(up_hat, vp_hat), a_hat)
#     #     cos_t = np.dot(up_hat, vp_hat)
#     #     return np.arctan2(sin_t, cos_t)  # stable, returns [-pi, pi]
    
#      # ----- CMC angles -----
#     #might be wrong?
#     theta0 = (-1.5 + arccos_projected_dot(u, v, a0))
#     theta1 = 0.3 - (arccos_projected_dot(u, v, a1))
#     # theta0 = (-1.5 + angle_projected_atan2(u, v, a0))
#     # theta1 = (0.3 - angle_projected_atan2(u, v, a1))
#     # print(theta0)
#     # print(theta1)

#     # ----- MCP & IP flexion -----
#     theta2 = np.pi - angle_at_B(L[1], L[2], L[3])
#     theta3 = np.pi - angle_at_B(L[2], L[3], L[4])

#     #first para control movement parallel to palm
#     #second para control movement perpendicular to  palm
#     return [theta0, theta1, theta2, theta3]


#from angles.py




def compute_thumb_angles(P: np.ndarray, is_right_hand=False) -> np.ndarray:
    """Calculates the 4 thumb angles"""
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

    def _project_onto_plane(v: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
        return v - np.dot(v, n_hat) * n_hat
    
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
    theta_0 = 1.5 - _safe_acos(np.dot(_normalize(_project_onto_plane(v_ref, a0)), _normalize(_project_onto_plane(v_thumb, a0))))
    theta_1 = np.radians(0.0) + _safe_acos(np.dot(_normalize(_project_onto_plane(v_ref, a1)), _normalize(_project_onto_plane(v_thumb, a1))))
    
    # Flexions (using points 1-2-3 and 2-3-4 from MediaPipe)
    angle_2 = _bend_angle(P[1]-P[2], P[3]-P[2])
    angle_3 = _bend_angle(P[2]-P[3], P[4]-P[3])
    
    return [theta_0, theta_1, angle_2, angle_3]


import numpy as np

class AngleEMASmoother:
    def __init__(self, alpha=0.2):
        self.alpha = float(alpha)
        self.prev = None  # shape (20,)

    def __call__(self, angles_20):
        angles_20 = np.asarray(angles_20, dtype=np.float64)

        if self.prev is None:
            self.prev = angles_20.copy()
            return angles_20

        smoothed = self.alpha * angles_20 + (1.0 - self.alpha) * self.prev
        self.prev = smoothed
        return smoothed
    
class RobustAngleSmoother:
    def __init__(self, alpha=0.08, max_jump=0.35):
        """
        alpha: EMA strength (smaller = smoother)
        max_jump: radians per frame allowed; bigger jumps are treated as spikes
        """
        self.alpha = float(alpha)
        self.max_jump = float(max_jump)
        self.prev = None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.prev is None:
            self.prev = x.copy()
            return x

        # spike rejection: clamp per-joint change
        delta = x - self.prev
        delta = np.clip(delta, -self.max_jump, self.max_jump)
        x_clamped = self.prev + delta

        y = self.alpha * x_clamped + (1 - self.alpha) * self.prev
        self.prev = y
        return y

import numpy as np

class LandmarkEMASmoother:
    def __init__(self, alpha=0.2):
        self.alpha = float(alpha)
        self.prev = None  # (21,3)

    def __call__(self, L):
        L = np.asarray(L, dtype=np.float64)
        assert L.shape == (21, 3)

        if self.prev is None:
            self.prev = L.copy()
            return L

        S = self.alpha * L + (1.0 - self.alpha) * self.prev
        self.prev = S
        return S

class LandmarkEMASmootherRobust:
    def __init__(self, alpha=0.2, max_step=0.02):
        self.alpha = float(alpha)
        self.max_step = float(max_step)
        self.prev = None

    def __call__(self, L):
        L = np.asarray(L, dtype=np.float64)
        assert L.shape == (21, 3)

        if self.prev is None:
            self.prev = L.copy()
            return L

        # clamp sudden jumps per coordinate
        delta = np.clip(L - self.prev, -self.max_step, self.max_step)
        L_clamped = self.prev + delta

        S = self.alpha * L_clamped + (1 - self.alpha) * self.prev
        self.prev = S
        return S
    
def compute_hand_angles(landmarks, smoother=None):
    landmarks = np.asarray(landmarks, dtype=np.float64)
    assert landmarks.shape == (21, 3)

    palm_x, palm_y, palm_z = compute_palm_frame(landmarks)
    n_finger = [
        index_reference_normal(landmarks),
        middle_reference_normal(landmarks),
        ring_reference_normal(landmarks),
        pinky_reference_normal(landmarks)
    ]

    angles = []
    thumb, debug = ang.compute_thumb_angles(landmarks, True)
    angles += thumb.tolist()              # 4
    flexion = compute_flexion_angles(landmarks)             # 12
    abduction = compute_abduction_angles(landmarks, n_finger)  # 4

    for i in range(4):
        angles.append(abduction[i])
        angles += flexion[i*3:i*3+3]

    angles = np.asarray(angles, dtype=np.float64)
    assert angles.shape == (20,)

    if smoother is not None:
        angles = smoother(angles)

    return angles

