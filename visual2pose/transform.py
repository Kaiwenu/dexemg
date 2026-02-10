import numpy as np

def _angle(a, b, c, eps=1e-8):
    """Return angle ABC in radians."""
    a = np.asarray(a); b = np.asarray(b); c = np.asarray(c)
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba) + eps
    nbc = np.linalg.norm(bc) + eps
    cosang = np.dot(ba, bc) / (nba * nbc)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.arccos(cosang))

def _bend(a, b, c):
    theta = _angle(a, b, c)
    return float(np.pi - theta)



def mp_landmarks_to_emg2pose20(landmarks, tip_mode="zero"):
    """
    landmarks: (21,3) MediaPipe landmarks [x,y,z]
    tip_mode:  "zero" or "duplicate"  (for TIP joints)
    returns: (20,) vector in emg2pose joint order (radians)
    """
    L = np.asarray(landmarks, dtype=np.float32)
    assert L.shape == (21, 3)

    w = L[0]

    # Thumb indices
    t1, t2, t3, t4 = L[1], L[2], L[3], L[4]

    # Fingers (mcp,pip,dip,tip)
    idx = (L[5], L[6], L[7], L[8])
    mid = (L[9], L[10], L[11], L[12])
    rng = (L[13], L[14], L[15], L[16])
    pnk = (L[17], L[18], L[19], L[20])

    def finger_angles(f):
        mcp, pip, dip, tip = f
        mcp_bend = _bend(w, mcp, pip)       # approx MCP flexion
        pip_bend = _bend(mcp, pip, dip)     # PIP flexion
        dip_bend = _bend(pip, dip, tip)     # DIP flexion

        if tip_mode == "zero":
            tip_bend = 0.0
        elif tip_mode == "duplicate":
            tip_bend = dip_bend
        else:
            raise ValueError("tip_mode must be 'zero' or 'duplicate'")

        return [mcp_bend, pip_bend, dip_bend, tip_bend]

    # Thumb angles (approx)
    thumb_cmc = _bend(w,  t1, t2)      # wrist–CMC–MCP
    thumb_mcp = _bend(t1, t2, t3)      # CMC–MCP–IP
    thumb_ip  = _bend(t2, t3, t4)      # MCP–IP–TIP
    if tip_mode == "zero":
        thumb_tip = 0.0
    else:
        thumb_tip = thumb_ip

    out = [
        thumb_cmc, thumb_mcp, thumb_ip, thumb_tip,
        *finger_angles(idx),
        *finger_angles(mid),
        *finger_angles(rng),
        *finger_angles(pnk),
    ]
    return np.array(out, dtype=np.float32)
