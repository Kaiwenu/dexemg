import numpy as np

MP_PARENT = {
    0: -1,
    1: 0, 2: 1, 3: 2, 4: 3,
    5: 0, 6: 5, 7: 6, 8: 7,
    9: 0, 10: 9, 11: 10, 12: 11,
    13: 0, 14: 13, 15: 14, 16: 15,
    17: 0, 18: 17, 19: 18, 20: 19
}

MP_CHILDREN = {i: [] for i in range(21)}
for c, p in MP_PARENT.items():
    if p >= 0:
        MP_CHILDREN[p].append(c)


def bone_membership_weights(lm_bone_idx, lm_bone_w, num_bones):
    membership = np.zeros((num_bones, lm_bone_idx.shape[0]), dtype=np.float32)
    L = lm_bone_idx.shape[0]
    for li in range(L):
        for j in range(lm_bone_idx.shape[1]):
            b = int(lm_bone_idx[li, j])
            if 0 <= b < num_bones:
                membership[b, li] += float(lm_bone_w[li, j])
    return membership
