import numpy as np

EPS = 1e-12

import numpy as np

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


def compute_thumb_angles(L):
    x_hat = normalize(L[5] - L[0])  # normalize(L5 - L0)
    z_hat = -index_reference_normal(L)
    y_hat = normalize(np.cross(z_hat, x_hat))
    x_hat = normalize(np.cross(y_hat, z_hat))  # re-orthogonalize
    R_index = np.column_stack([x_hat, y_hat, z_hat])  # columns are basis axes

    a0_local = np.array([3.0, 0.0, -1.0], dtype=np.float64)
    a1_local = np.array([-0.2, -3.0, -0.6], dtype=np.float64)
    a0 = R_index @ normalize(a0_local)
    a1 = R_index @ normalize(a1_local)
    u = L[5] - L[0]
    v = L[3] - L[0]

    def proj_perp_axis(v, a):
        a_hat = normalize(a)
        return v - np.dot(v, a_hat) * a_hat

    def arccos_projected_dot(u, v, a):
        up = proj_perp_axis(u, a)
        vp = proj_perp_axis(v, a)
        up_hat = normalize(up)
        vp_hat = normalize(vp)
        cos_t = np.clip(np.dot(up_hat, vp_hat), -1.0, 1.0)
        return np.arccos(cos_t)

     # ----- CMC angles -----
    theta0 = 1.5 - arccos_projected_dot(u, v, a0)
    theta1 = 0 + arccos_projected_dot(u, v, a1)
    # print(theta0)
    # print(theta1)

    # ----- MCP & IP flexion -----
    theta2 = np.pi - angle_at_B(L[1], L[2], L[3])
    theta3 = np.pi - angle_at_B(L[2], L[3], L[4])


    #take negative of theta0
    return [-theta0, theta1, theta2, theta3]


def compute_hand_angles(landmarks):
    landmarks = np.asarray(landmarks, dtype=np.float64)
    assert landmarks.shape == (21, 3)

    palm_x, palm_y, palm_z = compute_palm_frame(landmarks)
    # finger_normals = compute_finger_reference_normals(landmarks, palm_z)
    n_finger = [index_reference_normal(landmarks), middle_reference_normal(landmarks), ring_reference_normal(landmarks), pinky_reference_normal(landmarks)]


    angles = []
    angles += compute_thumb_angles(landmarks)       # 4
    # print(angles)
    flexion = compute_flexion_angles(landmarks)          # 12
    # print(flexion)
    # print(len(angles))
    abduction =  compute_abduction_angles(landmarks, n_finger)  # 4
    # print("fuck")
    # print(abduction)
    # print(len(angles))
    # print("fuck")
    # print(flexion)
    for i in range(4):
        angles.append(abduction[i])
        # print(flexion[i*3:i*3+3])
        angles += flexion[i*3:i*3+3]
        # angles += [0, 0, 0]


    # print(len(angles))

    angles = np.asarray(angles)
    # print(angles.shape)
    assert angles.shape == (20,)
    return angles

