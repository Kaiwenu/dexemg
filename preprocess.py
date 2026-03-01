"""
preprocess.py

Utility functions for EMG data preprocessing.

- preprocess_envelop(emg_int8, fs, cutoff_hz, clip)
    RAW -> float -> rectify -> smooth (EMG low-pass) -> soft clipping (optional)
- fit_channel_standarizer(x, eps)
    returns dict with 'mean' and 'std' shaped (C,)
- apply_channel_standarizer(x, stats)
    apply per-channel z-score standarization
- make_windows(x, win_s, hop_s, fs)
    slice a continuous sequence into overlapping windows

Usage:
    emg_raw = pd.read_csv("3_test_emg.csv")[[f"ch{i}" for i in range(1, 9)]].to_numpy()
    env = preprocess_envelope(emg_raw, fs=200, cutoff_hz=4.0)   # (T, 8)
    X = make_windows(env, win_s=0.25, hop_s=0.10, fs=200)       # (Nw, 50, 8) (50 = 0.25s/200hz)
    stats = fit_channel_standarizer(X)
    Xz = apply_channel_standarizer(X, stats)                    # (Nw, 50, 8)
    X_for_torch = np.transpose(Xz, (0, 2, 1))                   # (Nw, 8, 50)
"""

import numpy as np
import numpy.typing as npt

# ----------------------------
# Core helpers
# ----------------------------


def to_float_unit(emg_int8: np.ndarray) -> np.ndarray:
    """
    Convert Myo RAW EMG int8 [-128, 127] to float32 roughly in [-1, 1].

    emg_int8: (..., C) or (T, C) or (B, T, C)
    returns:  same shape, float32
    """
    x = emg_int8.astype(np.float32)
    # divide by 128 so -128 maps to -1.0 exactly; +127 maps to ~0.992
    return x / 128.0


def ema_lowpass(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    """
    Simple 1st-order IIR low-pass (exponential moving average) applied over time.
    Works well as a cheap envelope smoother.

    x: (T, C) float32
    fs: sampling rate in Hz (e.g., 200 for RAW mode)
    cutoff_hz: smoothing cutoff (static holds often 2–6 Hz)

    returns: (T, C)
    """
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be > 0")

    # RC low-pass equivalence: alpha = dt / (RC + dt), RC = 1/(2*pi*fc)
    dt = 1.0 / fs
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = dt / (rc + dt)

    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    for t in range(1, x.shape[0]):
        y[t] = y[t - 1] + alpha * (x[t] - y[t - 1])
    return y


def running_mean_highpass(x: np.ndarray, win_s: float, fs: float) -> np.ndarray:
    """
    Cheap drift / DC removal: subtract a running mean.
    Helpful for RAW waveform pipelines (less crucial if you rectify first).

    x: (T, C)
    win_s: window length in seconds (e.g., 0.5–1.0)
    fs: sampling rate in Hz
    returns: (T, C)
    """
    win = max(1, int(round(win_s * fs)))
    # causal running mean via cumulative sum
    csum = np.cumsum(x, axis=0, dtype=np.float64)
    y = np.empty_like(x, dtype=np.float32)
    for t in range(x.shape[0]):
        start = max(0, t - win + 1)
        count = t - start + 1
        mean = (csum[t] - (csum[start - 1] if start > 0 else 0.0)) / count
        y[t] = x[t] - mean.astype(np.float32)
    return y


# ----------------------------
# Recommended preprocessing for static holds (envelope)
# ----------------------------


def preprocess_envelope(
    emg_int8: np.ndarray,
    fs: float = 200.0,
    cutoff_hz: float = 4.0,
    clip: float | None = 3.0,
) -> np.ndarray:
    """
    Minimal, strong baseline for static pose classification:
      RAW -> float -> rectify -> smooth (EMA low-pass) -> (optional) soft clipping.

    emg_int8: (T, C) int8 from Myo RAW mode, C=8 typically
    returns:  (T, C) float32 envelope-like signal
    """
    x = to_float_unit(emg_int8)  # (T, C)
    x = np.abs(x)  # rectify -> (T, C)
    x = ema_lowpass(x, fs=fs, cutoff_hz=cutoff_hz)  # smooth -> (T, C)

    # Optional: tame occasional spikes (helps training stability)
    if clip is not None:
        x = np.clip(x, 0.0, clip).astype(np.float32)

    return x.astype(np.float32)


# ----------------------------
# Standardization (fit on training set, apply everywhere)
# ----------------------------


def fit_channel_standardizer(
    x: np.ndarray, eps: float = 1e-6
) -> dict[str, npt.NDArray]:
    """
    Fit per-channel mean/std on training data.

    x: (N, T, C) or (T, C)
    returns: dict with 'mean' and 'std' shaped (C,)
    """
    if x.ndim == 2:
        # (T, C)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    elif x.ndim == 3:
        # (N, T, C)
        mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
        std = x.reshape(-1, x.shape[-1]).std(axis=0)
    else:
        raise ValueError("x must be (T,C) or (N,T,C)")

    std = np.maximum(std, eps)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_channel_standardizer(x: np.ndarray, stats: dict[str, npt.NDArray]) -> np.ndarray:
    """
    Apply per-channel z-score standardization.

    x: (T, C) or (N, T, C)
    stats: output of fit_channel_standardizer
    """
    mean = stats["mean"]
    std = stats["std"]
    return ((x - mean) / std).astype(np.float32)


# ----------------------------
# Windowing (for classifier input)
# ----------------------------


def make_windows(
    x: np.ndarray,
    window_duration_s: float,
    step_duration_s: float,
    sampling_rate: float,
) -> np.ndarray:
    """
    Slice a continuous sequence into overlapping windows.

    x: (T, C)
    win_s: window length in seconds (static holds: 0.2–0.3s is common)
    hop_s: hop length in seconds (e.g., 0.05–0.1s)
    fs: sampling rate
    returns: (Nw, Tw, C)
    """
    T, C = x.shape
    Tw = int(round(window_duration_s * sampling_rate))
    hop = int(round(step_duration_s * sampling_rate))
    if Tw <= 0 or hop <= 0:
        raise ValueError("win_s and hop_s must yield positive sample counts")
    if T < Tw:
        return np.empty((0, Tw, C), dtype=np.float32)

    starts = np.arange(0, T - Tw + 1, hop)
    windows = np.stack([x[s : s + Tw] for s in starts], axis=0)  # (Nw, Tw, C)
    return windows.astype(np.float32)


# ----------------------------
# Example usage (static holds)
# ----------------------------
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("3_test_emg.csv")[[f"ch{i}" for i in range(1, 9)]]
    print(f"{df.shape=}")

    # emg_raw: (T, 8) int8 from mode 0x03
    emg_raw = df.to_numpy()
    print(f"{emg_raw.shape=}")

    fs = 200.0
    env = preprocess_envelope(emg_raw, fs=fs, cutoff_hz=4.0)  # (T, 8)
    print(f"{env.shape=}")

    # Create 250 ms windows every 100 ms
    X = make_windows(
        env, window_duration_s=0.25, step_duration_s=0.10, sampling_rate=fs
    )  # (Nw, 50, 8)
    print(f"{X.shape=}")

    # Fit/apply standardization (normally: fit on training only)
    stats = fit_channel_standardizer(X)  # mean/std per channel
    Xz = apply_channel_standardizer(X, stats)  # (Nw, 50, 8)
    print(f"{Xz.shape=}")

    # For PyTorch Conv1d you’ll often transpose: (Nw, C, T)
    X_for_torch = np.transpose(Xz, (0, 2, 1))  # (Nw, 8, 50)
    print(f"{X_for_torch.shape=}")
