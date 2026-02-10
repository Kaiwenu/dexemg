import h5py

path = r"C:\Users\kaich\Desktop\dexemg\data\session_1770708880_left.hdf5"

with h5py.File(path, "r") as f:
    ts = f["emg2pose/timeseries"]
    print("dtype:", ts.dtype)
    print("fields:", ts.dtype.fields.keys())
    print("emg field dtype:", ts.dtype.fields["emg"][0])
    print("joint_angles field dtype:", ts.dtype.fields["joint_angles"][0])
    print("time field dtype:", ts.dtype.fields["time"][0])
    print("shape:", ts.shape)
    print("group attrs:", dict(f["emg2pose"].attrs))
    # if exists:
    if "no_ik_failure" in f["emg2pose"]:
        print("no_ik_failure shape:", f["emg2pose/no_ik_failure"].shape)
