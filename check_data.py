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

# path = r"C:\Users\kaich\Desktop\dexemg\new_data\single_finger4.hdf5"
path = r"C:\Users\kaich\Desktop\dexemg\new_data\session_1772168949_left.hdf5"
with h5py.File(path, "r") as f:
    g = f["emg2pose"]
    
    print("Group attrs:")  
    print("start:", g.attrs["start"])
    print("end  :", g.attrs["end"])
    print("duration:", g.attrs["end"] - g.attrs["start"])
    ts = f["emg2pose"]["timeseries"]
    print("Number of samples:", len(ts))


data = Emg2PoseSessionData(hdf5_path=path)

session_window = data[:]

joint_angles = data["joint_angles"]
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
