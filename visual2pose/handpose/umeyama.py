import numpy as np

def umeyama_similarity(A, B, with_scale=True):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = A.shape[0]

    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    A0 = A - ca
    B0 = B - cb

    Sigma = (A0.T @ B0) / n
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = Vt.T @ S @ U.T

    if with_scale:
        varA = (A0 ** 2).sum() / n
        s = (np.trace(np.diag(D) @ S)) / (varA + 1e-12)
    else:
        s = 1.0

    t = cb - s * (ca @ R)
    return np.float32(s), R.astype(np.float32), t.astype(np.float32)
