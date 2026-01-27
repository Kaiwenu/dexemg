import numpy as np

def apply_axis_map(mp21, swap_yz, flip_x, flip_z):
    p = np.asarray(mp21, dtype=np.float64).copy()
    p[:, 1] *= -1.0  # y-up
    if swap_yz:
        p = p[:, [0, 2, 1]]
    if flip_x:
        p[:, 0] *= -1.0
    if flip_z:
        p[:, 2] *= -1.0
    return p
