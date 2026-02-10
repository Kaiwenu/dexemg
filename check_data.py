import h5py

path = "data/session_1770701642_left.hdf5"

with h5py.File(path, "r") as f:
    print("Datasets:", list(f.keys()))
    d = f["data"]
    print("Dataset shape:", d.shape)
    print("Dataset dtype:", d.dtype)

with h5py.File(path, "r") as f:
    print("Attributes:")
    for k, v in f.attrs.items():
        print(f"  {k}: {v}")