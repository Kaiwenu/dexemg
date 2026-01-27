import numpy as np
from .umeyama import umeyama_similarity

def compute_rest_landmarks_from_mesh(mesh_vertices, lm_bone_idx, lm_bone_w, dense_bone_weights):
    V = mesh_vertices.shape[0]
    B = dense_bone_weights.shape[1]
    L = lm_bone_idx.shape[0]
    rest21 = np.zeros((L, 3), dtype=np.float32)

    Wv = np.asarray(dense_bone_weights, dtype=np.float64)
    Wv = Wv / (Wv.sum(axis=1, keepdims=True) + 1e-8)

    verts = np.asarray(mesh_vertices, dtype=np.float64)

    for li in range(L):
        bones = lm_bone_idx[li]
        bw = lm_bone_w[li]
        bw_sum = float(bw.sum())
        if bw_sum < 1e-8:
            rest21[li] = verts.mean(axis=0)
            continue

        bw = bw / bw_sum
        w_vert = np.zeros(V, dtype=np.float64)

        for j in range(bones.shape[0]):
            b = int(bones[j])
            if 0 <= b < B and bw[j] > 0:
                w_vert += bw[j] * Wv[:, b]

        wv_sum = float(w_vert.sum())
        if wv_sum < 1e-8:
            rest21[li] = verts.mean(axis=0)
        else:
            w_vert = w_vert / wv_sum
            rest21[li] = (w_vert[:, None] * verts).sum(axis=0)

    return rest21.astype(np.float32)


def compute_rest_corrective_transform(rest21, mp21_open, wrist_idx=0, palm_ids=[0,5,9,13,17]):
    rest_wc = rest21 - rest21[wrist_idx]
    mp_wc = mp21_open - mp21_open[wrist_idx]

    _, Rg, _ = umeyama_similarity(mp_wc[palm_ids], rest_wc[palm_ids], with_scale=False)

    R_corr = Rg.T.astype(np.float32)
    t_corr = (mp21_open[wrist_idx] - rest21[wrist_idx] @ R_corr).astype(np.float32)
    return R_corr, t_corr


def correct_wrist_offset(rest21, mp21_open, wrist_idx=0):
    mesh_wrist = rest21[wrist_idx]
    mp_wrist = mp21_open[wrist_idx]
    return (mp_wrist - mesh_wrist).astype(np.float32)
