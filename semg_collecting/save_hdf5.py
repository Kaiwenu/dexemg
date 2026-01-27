import numpy as np
import h5py
import time
from pathlib import Path

def save_emg2pose_hdf5(out_path, emg_samples, t_samples, user="local", side="right", split="train", stage="OneHandedFreeStyle"):
    """
    emg_samples: list of lists, each is EMG vector (len could be 8 or 16)
    t_ns_samples: list of int nanoseconds since start_time_ns (same length as emg_samples)
    """
    T = len(emg_samples)
    if T == 0:
        raise ValueError("No samples collected; nothing to save.")

    # Convert to arrays
    emg_raw = np.asarray(emg_samples, dtype=np.float32)   # (T, C_in)
    t = np.asarray(t_samples, dtype=np.int64)       # (T,)

    # --- Force EMG to (T,16) to match schema ---
    # If you only have 8 channels (typical Myo), we pad zeros to 16.
    # If you have >16, truncate.
    C_in = emg_raw.shape[1]
    emg = np.zeros((T, 16), dtype=np.float32)
    emg[:, :min(C_in, 16)] = emg_raw[:, :min(C_in, 16)]

    # --- joint_angles placeholder (T,20) ---
    joint_angles = np.full((T, 20), np.nan, dtype=np.float32)  # or zeros


    # Build compound dtype EXACTLY like the dataset you inspected
    ts_dtype = np.dtype([
        ("time", np.float64),
        ("joint_angles", np.float32, (20,)),
        ("emg", np.float32, (16,)),
    ])

    ts = np.empty((T,), dtype=ts_dtype)
    ts["time"] = t
    ts["joint_angles"] = joint_angles
    ts["emg"] = emg

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filename_stem = out_path.stem

    with h5py.File(out_path, "w") as f:
        g = f.create_group("emg2pose")

        # chunks=True helps windowed reading performance
        g.create_dataset("timeseries", data=ts, compression="gzip", chunks=True)

        # Metadata attrs (match keys style from the dataset you printed)
        g.attrs["filename"] = filename_stem
        g.attrs["session"] = filename_stem.split("-recording-")[0] if "-recording-" in filename_stem else filename_stem
        g.attrs["user"] = user
        g.attrs["side"] = side
        g.attrs["split"] = split
        g.attrs["stage"] = stage
        g.attrs["num_channels"] = 16
        g.attrs["sample_rate"] = 200.0  # schema expects this; if you want truth, set to your real rate
        g.attrs["start"] = float(t[0])
        g.attrs["end"] = float(t[-1])

        # Optional fields from your example (set if you use them)
        g.attrs["moving_hand"] = side
        g.attrs["held_out_user"] = False
        g.attrs["held_out_stage"] = False
        g.attrs["generalization"] = "user"

    return out_path
