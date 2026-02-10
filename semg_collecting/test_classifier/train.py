from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


import numpy as np

def window_view(x: np.ndarray, win: int, hop: int):
    """
    x: (T, C)
    returns windows: (N, win, C)
    """
    T, C = x.shape
    if T < win:
        return np.empty((0, win, C), dtype=x.dtype)
    n = 1 + (T - win) // hop
    out = np.zeros((n, win, C), dtype=x.dtype)
    for i in range(n):
        s = i * hop
        out[i] = x[s:s+win]
    return out

def emg_features(windows: np.ndarray) -> np.ndarray:
    """
    windows: (N, W, C)
    returns: (N, 3*C) features [RMS, MAV, WL] per channel
    """
    if windows.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    # remove DC per-window per-channel
    w = windows - windows.mean(axis=1, keepdims=True)

    absw = np.abs(w)
    rms = np.sqrt(np.mean(w * w, axis=1))                  # (N, C)
    mav = np.mean(absw, axis=1)                            # (N, C)
    wl  = np.sum(np.abs(np.diff(w, axis=1)), axis=1)       # (N, C)

    feats = np.concatenate([rms, mav, wl], axis=1).astype(np.float32)
    return feats


def build_dataset(trials, fs, win_ms=200, hop_ms=50):
    win = int(round(fs * win_ms / 1000))
    hop = int(round(fs * hop_ms / 1000))

    X_list, y_list = [], []
    for tr in trials:
        emg = tr["emg"].astype(np.float32)
        y = int(tr["label"])

        windows = window_view(emg, win=win, hop=hop)  # (N, win, C)
        if windows.shape[0] == 0:
            continue
        Xw = emg_features(windows)                    # (N, F)

        X_list.append(Xw)
        y_list.append(np.full((Xw.shape[0],), y, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y

def predict(clf, fs, emg):
    emg = np.atleast_2d(emg).astype(np.float32)  # (T, 8)
    win = int(round(fs * 200 / 1000))
    hop = int(round(fs * 50 / 1000))

    windows = window_view(emg, win=win, hop=hop)
    X = emg_features(windows)   # (N, F)
    pred_windows = clf.predict(X)   # array of 0/1
    final_label = int(np.round(pred_windows.mean()))

    label_name = "COUNT_UP" if final_label == 0 else "COUNT_DOWN"
    print("Predicted label:", label_name)


data0 = np.loadtxt("C:/Users/kaich/Desktop/dexemg/semg_collecting/test_classifier/one2five.csv", delimiter=",", skiprows=1)
data1 = np.loadtxt("C:/Users/kaich/Desktop/dexemg/semg_collecting/test_classifier/five2one.csv", delimiter=",", skiprows=1)
trials = [
  {"emg": data0, "label": 0},  # count up
  {"emg": data1, "label": 1},  # count down
]
fs = 50  # example


X, y = build_dataset(trials, fs=fs)

# IMPORTANT: split by trials ideally; as a quick start, we split by windows.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000))
])

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, pred))
# print("Confusion matrix:\n", confusion_matrix(y_test, pred))
# print(classification_report(y_test, pred, target_names=["COUNT_UP", "COUNT_DOWN"]))


emg = np.loadtxt("new_trial_down.csv", delimiter=",", skiprows=1)
predict(clf, fs, emg)