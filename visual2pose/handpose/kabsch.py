import numpy as np

def kabsch(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    a0 = A.mean(axis=0)
    b0 = B.mean(axis=0)

    Ac = A - a0
    Bc = B - b0

    H = Ac.T @ Bc
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = b0 - a0 @ R
    return R.astype(np.float32), t.astype(np.float32)


def weighted_kabsch(A, B, w):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)

    ws = float(w.sum())
    if ws < 1e-12:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    w = w / ws

    a0 = (A * w[:, None]).sum(axis=0)
    b0 = (B * w[:, None]).sum(axis=0)

    Ac = A - a0
    Bc = B - b0

    H = (Ac * w[:, None]).T @ Bc
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = b0 - a0 @ R
    return R.astype(np.float32), t.astype(np.float32)
