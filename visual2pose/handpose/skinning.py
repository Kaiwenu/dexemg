import numpy as np

def apply_lbs(vertices, weights, R_list, t_list, rest_wrist):
    """
    Linear Blend Skinning (LBS)
    vertices: (V,3) base vertices in mesh space
    weights:  (V,B) skinning weights
    R_list:   list of (3,3) bone rotations
    t_list:   list of (3,)  bone translations
    rest_wrist: (3,) wrist center in mesh space

    Returns: (V,3) deformed vertices
    """
    V_wc = (vertices - rest_wrist).astype(np.float32)
    out_wc = np.zeros_like(V_wc, dtype=np.float32)

    B = weights.shape[1]

    for b in range(B):
        wv = weights[:, b:b+1]
        if float(wv.max()) < 1e-6:
            continue
        out_wc += wv * (V_wc @ R_list[b] + t_list[b])

    return out_wc + rest_wrist
