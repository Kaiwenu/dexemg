import numpy as np
import matplotlib.pyplot as plt
from emg2pose.data import Emg2PoseSessionData

path = r"C:\Users\kaich\Desktop\dexemg\new_data\session_1772170323_left.hdf5"
# path = r"C:\Users\kaich\Downloads\Feb25_data-20260226T075008Z-1-001\five_gestures\spray_6.hdf5"
# path = r"C:\Users\kaich\Desktop\dexemg\new_data\newsingle.hdf5"
# path = r"C:\Users\kaich\Desktop\dexemg\new_data\new_single_2.hdf5"
data = Emg2PoseSessionData(hdf5_path=path)
session_window = data[:]                 # dict-like window
emg = np.asarray(session_window["emg"])  # expected shape (T, 8)

print("emg shape:", emg.shape, "dtype:", emg.dtype)

# ---- build time axis ----
native_fs = 200  # Myo is typically 200 Hz; change if yours differs
t = np.arange(emg.shape[0]) / native_fs

# ---- (optional) light downsample for faster plotting if long ----
# keep_every = 2   # e.g. 2 -> 100 Hz plot
# emg = emg[::keep_every]
# t = t[::keep_every]

# ---- plot 8 channels stacked ----
fig, axes = plt.subplots(8, 1, sharex=True, figsize=(14, 10))
for ch in range(8):
    axes[ch].plot(t, emg[:, ch], linewidth=0.8)
    axes[ch].set_ylabel(f"Ch {ch+1}")
    axes[ch].grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("sEMG (8 channels)", y=0.995)
plt.tight_layout()
plt.show()