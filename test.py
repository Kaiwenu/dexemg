import h5py
from emg2pose.utils import generate_hydra_config_from_overrides
import glob
import os
from emg2pose.data import Emg2PoseSessionData
import emg2pose.visualization as visualization
from emg2pose.utils import downsample
import numpy as np
from emg2pose.lightning import Emg2PoseModule
import torch
import mediapy


import numpy as np

import numpy as np

import numpy as np

T = 2000
a = -1
b = 0

joint_angles = np.zeros((T, 20))

mid = (a + b) / 2
amp = (b - a) / 2

for i in range(T):
    phase = 2 * np.pi * i / T
    joint_angles[i, 1] = mid + amp * np.sin(phase)
# print(joint_angles)
joint_angles_30hz = downsample(joint_angles, native_fs=200, target_fs=10)

temp = visualization.get_plotly_animation_for_joint_angles(joint_angles_30hz[:])
temp.show()

# frames = visualization.joint_angles_to_frames_parallel(joint_angles_30hz[:])
# frames = visualization.remove_alpha_channel(frames)
# mediapy.show_video(frames, width=800, fps=10, downsample=True)
# mediapy.write_video("oriori.mp4", frames, fps=10)
# print("Wrote pred.mp4")


# def check_array(name, arr):
#     arr = np.asarray(arr)
#     print(f"\nChecking {name}")
#     print("Shape:", arr.shape)
#     print("dtype:", arr.dtype)

#     nan_count = np.isnan(arr).sum()
#     inf_count = np.isinf(arr).sum()

#     print("NaN count:", nan_count)
#     print("Inf count:", inf_count)

#     if nan_count > 0:
#         print("⚠️  WARNING: Contains NaNs")
#     if inf_count > 0:
#         print("⚠️  WARNING: Contains Infs")

#     print("Min:", np.nanmin(arr))
#     print("Max:", np.nanmax(arr))
#     print("Mean:", np.nanmean(arr))
#     print("Std:", np.nanstd(arr))


# check_array("EMG", session_window["emg"])
# check_array("Joint Angles", session_window["joint_angles"])
