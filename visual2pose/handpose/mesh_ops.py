import numpy as np

def apply_corrective_transform_to_mesh(verts, R_corr, t_corr):
    return (verts @ R_corr) + t_corr
