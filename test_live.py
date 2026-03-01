# live_4panel_emg2pose_ui_overlay.py
#
# 4-panel UI (2x2) + tiny overlays:
#   Top-left:    Webcam + FPS overlay
#   Top-right:   "GT" mesh from MediaPipe landmark2angle + update rate overlay
#   Bottom-left: Live EMG plot + EMG Hz overlay
#   Bottom-right:Live inference mesh + pred FPS + buffer fill % overlay
#
# Run:
#   (emg2pose) python live_4panel_emg2pose_ui_overlay.py
#
# You must edit:
#   CHECKPOINT_PATH below

import os
import time
import multiprocessing as mp
from collections import deque

import numpy as np
import cv2
import torch

from PyQt5 import QtCore, QtGui, QtWidgets

import mediapipe as mp_hands_lib
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from pyomyo import Myo, emg_mode

from emg2pose.utils import generate_hydra_config_from_overrides
from emg2pose.lightning import Emg2PoseModule
import emg2pose.visualization as visualization

import landmark2angle as landmark2angle


# =========================
# 1) Simple envelope preprocess (self-contained)
# =========================
def preprocess_envelope(emg_tc: np.ndarray, fs: float = 200.0, cutoff_hz: float = 4.0, clip: float = 3.0):
    x = np.asarray(emg_tc, dtype=np.float32)
    x = np.abs(x)

    dt = 1.0 / float(fs)
    rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    alpha = dt / (rc + dt)

    y = np.empty_like(x)
    state = np.zeros((x.shape[1],), dtype=np.float32)
    for i in range(x.shape[0]):
        state = (1.0 - alpha) * state + alpha * x[i]
        y[i] = state

    if clip is not None:
        y = np.clip(y, -float(clip), float(clip))
    return y


# =========================
# 2) Myo worker (separate process)
# =========================
def myo_worker(q: mp.Queue, mode):
    m = Myo(mode=mode)
    m.connect()

    def on_emg(emg, movement):
        q.put((time.perf_counter_ns(), np.asarray(emg, dtype=np.float32)))

    m.add_emg_handler(on_emg)
    m.set_leds([128, 128, 0], [128, 128, 0])
    m.vibrate(1)

    while True:
        m.run()


# =========================
# 3) Mesh panel helper
# =========================
class MeshPanel:
    def __init__(self, glview: gl.GLViewWidget):
        self.glview = glview
        self._mesh_item = None
        self._faces = None

        grid = gl.GLGridItem()
        grid.scale(20, 20, 1)
        self.glview.addItem(grid)

        # Rate tracking
        self._last_update_t = None
        self._ema_hz = 0.0

    def update_from_angles(self, ja20: np.ndarray, opacity=0.55):
        now = time.perf_counter()
        if self._last_update_t is not None:
            dt = max(now - self._last_update_t, 1e-6)
            hz = 1.0 / dt
            # EMA to stabilize
            self._ema_hz = 0.9 * self._ema_hz + 0.1 * hz
        self._last_update_t = now

        hand_mesh = visualization.generate_hand_mesh_from_joint_angles(
            joint_angles=np.asarray(ja20, dtype=np.float32),
            color="lightpink",
            flip=False,
            opacity=float(opacity),
        )

        x = np.asarray(hand_mesh.x, dtype=np.float32)
        y = np.asarray(hand_mesh.y, dtype=np.float32)
        z = np.asarray(hand_mesh.z, dtype=np.float32)
        verts = np.stack([x, y, z], axis=1)

        if self._faces is None:
            i = np.asarray(hand_mesh.i, dtype=np.int32)
            j = np.asarray(hand_mesh.j, dtype=np.int32)
            k = np.asarray(hand_mesh.k, dtype=np.int32)
            self._faces = np.stack([i, j, k], axis=1)

        if self._mesh_item is None:
            mesh_color = QtGui.QColor(255, 182, 193, int(255 * float(opacity)))
            self._mesh_item = gl.GLMeshItem(
                vertexes=verts,
                faces=self._faces,
                smooth=True,
                drawEdges=True,
                color=mesh_color,
            )
            self.glview.addItem(self._mesh_item)
        else:
            self._mesh_item.setMeshData(vertexes=verts, faces=self._faces)

    @property
    def hz(self) -> float:
        return float(self._ema_hz)


# =========================
# 4) Main app
# =========================
class Live4PanelApp(QtWidgets.QWidget):
    def __init__(
        self,
        checkpoint_path: str,
        cam_index: int = 0,
        webcam_target_fps: float = 21.0,
        show_mediapipe_landmarks: bool = True,
        emg_fs: float = 200.0,
        window_len: int = 2190,
        stride: int = 20,
    ):
        super().__init__()
        self.setWindowTitle("4-Panel + Overlay: Webcam | GT Mesh | EMG | Pred Mesh")
        self.resize(1400, 1040)

        # ---- config ----
        self.cam_index = cam_index
        self.webcam_target_fps = float(webcam_target_fps)
        self.show_mediapipe_landmarks = bool(show_mediapipe_landmarks)

        self.emg_fs = float(emg_fs)
        self.window_len = int(window_len)
        self.stride = int(stride)

        # ---- stats ----
        self.mean = np.array(
            [0.04573015, 0.08782841, 0.12609805, 0.04755059,
             0.03398797, 0.05191861, 0.04411899, 0.04901559],
            dtype=np.float32
        )
        self.std = np.array(
            [0.03806118, 0.09254788, 0.08249861, 0.02404639,
             0.02373608, 0.05819574, 0.0381004, 0.05719165],
            dtype=np.float32
        )

        # ---- model ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module = self._load_emg2pose_module(checkpoint_path).to(self.device).eval()

        # ---- webcam/mediapipe ----
        self.cap = cv2.VideoCapture(self.cam_index)
        self.hands = mp_hands_lib.solutions.hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
            max_num_hands=1,
            model_complexity=1,
        )
        self.drawer = mp_hands_lib.solutions.drawing_utils

        # ---- layout: 2x2 ----
        root = QtWidgets.QGridLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setHorizontalSpacing(10)
        root.setVerticalSpacing(10)

        # TL: webcam (QLabel)
        self.view = QtWidgets.QLabel()
        self.view.setFixedSize(640, 480)
        self.view.setStyleSheet("background:#111;")
        root.addWidget(self.view, 0, 0)

        # TR: GT mesh
        self.gl_gt = gl.GLViewWidget()
        self.gl_gt.setFixedSize(640, 480)
        self.gl_gt.opts["distance"] = 300
        root.addWidget(self.gl_gt, 0, 1)
        self.gt_mesh = MeshPanel(self.gl_gt)

        # BL: EMG plot
        self.emg_plot = pg.PlotWidget(title="Live sEMG (RAW + EMA envelope)")
        self.emg_plot.setFixedSize(640, 480)
        self.emg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.emg_plot.setLabel("bottom", "Time", units="s")
        self.emg_plot.setLabel("left", "Amplitude (offset per channel)")
        self.emg_plot.addLegend()
        root.addWidget(self.emg_plot, 1, 0)

        # BR: Pred mesh
        self.gl_pred = gl.GLViewWidget()
        self.gl_pred.setFixedSize(640, 480)
        self.gl_pred.opts["distance"] = 300
        root.addWidget(self.gl_pred, 1, 1)
        self.pred_mesh = MeshPanel(self.gl_pred)

        # Overlays (tiny labels inside each panel)
        self.ov_webcam = self._make_overlay_label(self.view, "Webcam: -- FPS")
        self.ov_gt = self._make_overlay_label(self.gl_gt, "GT mesh: -- Hz")
        self.ov_emg = self._make_overlay_label(self.emg_plot, "EMG: -- Hz")
        self.ov_pred = self._make_overlay_label(self.gl_pred, "Pred: -- Hz | Buffer: --%")

        # Status bar (optional)
        self.status = QtWidgets.QLabel("Starting…")
        self.status.setStyleSheet("font-size:14px;")
        root.addWidget(self.status, 2, 0, 1, 2)

        # ---- EMG plot buffers ----
        self.emg_window_s = 5.0
        self.emg_plot_fps = 30
        self.emg_plot_downsample = 2

        self.emg_offsets = np.arange(8, dtype=np.float32) * 300.0
        self.emg_gain = 1.0

        self._emg_buf_len = int(self.emg_window_s * self.emg_fs)
        self._emg_buf = np.zeros((self._emg_buf_len, 8), dtype=np.float32)
        self._emg_buf_n = 0
        self._emg_buf_write = 0

        self.emg_env_alpha = 0.15
        self._emg_env_state = np.zeros((8,), dtype=np.float32)

        self._emg_raw_curves = []
        self._emg_env_curves = []
        for ch in range(8):
            raw_c = self.emg_plot.plot(pen=pg.mkPen(width=1), name=f"Ch {ch+1} raw")
            env_c = self.emg_plot.plot(pen=pg.mkPen(width=2), name=f"Ch {ch+1} env")
            self._emg_raw_curves.append(raw_c)
            self._emg_env_curves.append(env_c)

        # ---- Online inference buffers ----
        self.emg_q = mp.Queue()
        self.emg_proc = None

        self._raw_buf = deque(maxlen=max(10000, self.window_len * 4))
        self._sample_count = 0
        self._next_emit = None

        # ---- rate tracking ----
        self._webcam_last_t = None
        self._webcam_ema_fps = 0.0

        self._emg_last_count = 0
        self._emg_last_t = time.perf_counter()
        self._emg_ema_hz = 0.0

        # ---- timers ----
        self.cam_timer = QtCore.QTimer()
        self.cam_timer.timeout.connect(self.on_frame)
        self.cam_timer.start(int(1000 / self.webcam_target_fps))

        self.emg_drain_timer = QtCore.QTimer()
        self.emg_drain_timer.timeout.connect(self.drain_emg_queue)
        self.emg_drain_timer.start(5)

        self.emg_plot_timer = QtCore.QTimer()
        self.emg_plot_timer.timeout.connect(self.update_emg_plot)
        self.emg_plot_timer.start(int(1000 / self.emg_plot_fps))

        # Update overlay texts regularly (independent)
        self.overlay_timer = QtCore.QTimer()
        self.overlay_timer.timeout.connect(self.update_overlays)
        self.overlay_timer.start(200)

        # start EMG
        self.start_emg()

    # ---------- overlay helper ----------
    def _make_overlay_label(self, parent_widget: QtWidgets.QWidget, text: str) -> QtWidgets.QLabel:
        lab = QtWidgets.QLabel(parent_widget)
        lab.setText(text)
        lab.setStyleSheet(
            "QLabel {"
            " background: rgba(0,0,0,140);"
            " color: white;"
            " padding: 3px 6px;"
            " border-radius: 4px;"
            " font-size: 11px;"
            "}"
        )
        lab.adjustSize()
        lab.move(8, 8)  # top-left corner
        lab.raise_()
        return lab

    # ---------- model loading ----------
    def _load_emg2pose_module(self, checkpoint_path: str) -> Emg2PoseModule:
        config = generate_hydra_config_from_overrides(
            overrides=["experiment=tracking_vemg2pose", f"checkpoint={checkpoint_path}"]
        )
        module = Emg2PoseModule.load_from_checkpoint(
            config.checkpoint,
            network=config.network,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
        )
        return module

    # ---------- EMG controls ----------
    def start_emg(self):
        MODE = emg_mode.RAW
        self.emg_proc = mp.Process(target=myo_worker, args=(self.emg_q, MODE))
        self.emg_proc.start()
        self.status.setText("EMG started. Filling model buffer for first prediction…")

    def stop_emg(self):
        if self.emg_proc is not None:
            try:
                self.emg_proc.terminate()
                self.emg_proc.join()
            finally:
                self.emg_proc = None

    # ---------- webcam + GT ----------
    def on_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        # FPS tracking (EMA)
        now = time.perf_counter()
        if self._webcam_last_t is not None:
            dt = max(now - self._webcam_last_t, 1e-6)
            fps = 1.0 / dt
            self._webcam_ema_fps = 0.9 * self._webcam_ema_fps + 0.1 * fps
        self._webcam_last_t = now

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        angles20 = None
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            if self.show_mediapipe_landmarks:
                self.drawer.draw_landmarks(frame, lm, mp_hands_lib.solutions.hands.HAND_CONNECTIONS)

            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            angles = landmark2angle.compute_hand_angles(pts, smoother=None)
            angles20 = np.asarray(angles, dtype=np.float32)

        if angles20 is not None:
            self.gt_mesh.update_from_angles(angles20, opacity=0.55)

        self.display_frame(frame)

    def display_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.view.size(), QtCore.Qt.KeepAspectRatio)
        self.view.setPixmap(pix)

    # ---------- EMG drain (plot + inference) ----------
    def drain_emg_queue(self):
        got_any = False
        n_new = 0

        while not self.emg_q.empty():
            t_ns, emg8 = self.emg_q.get()
            emg8 = np.asarray(emg8, dtype=np.float32)
            if emg8.shape != (8,):
                continue

            # plot ring buffer
            self._emg_buf[self._emg_buf_write] = emg8
            self._emg_buf_write = (self._emg_buf_write + 1) % self._emg_buf_len
            self._emg_buf_n = min(self._emg_buf_n + 1, self._emg_buf_len)

            # envelope state (for display)
            rect = np.abs(emg8)
            a = float(self.emg_env_alpha)
            self._emg_env_state = (1.0 - a) * self._emg_env_state + a * rect

            # inference buffer
            self._raw_buf.append(emg8)
            self._sample_count += 1

            got_any = True
            n_new += 1

        if not got_any:
            return

        # EMG Hz estimation based on counts
        now = time.perf_counter()
        dt = max(now - self._emg_last_t, 1e-6)
        hz_inst = n_new / dt
        self._emg_ema_hz = 0.9 * self._emg_ema_hz + 0.1 * hz_inst
        self._emg_last_t = now

        # init next_emit
        if self._next_emit is None and self._sample_count >= self.window_len:
            self._next_emit = int(np.ceil(self.window_len / self.stride) * self.stride)

        # predict at stride
        did_pred = False
        while self._next_emit is not None and self._sample_count >= self._next_emit:
            if len(self._raw_buf) < self.window_len:
                break

            raw_window_tc = np.asarray(self._raw_buf, dtype=np.float32)[-self.window_len:]  # (T,8)
            ja20 = self.predict_latest_joint_angles(raw_window_tc)
            self.pred_mesh.update_from_angles(ja20, opacity=0.55)

            self._next_emit += self.stride
            did_pred = True

        if did_pred:
            secs = self._sample_count / self.emg_fs
            self.status.setText(
                f"EMG: {self._sample_count} samples (~{secs:.1f}s) | "
                f"Pred target: {self.emg_fs/self.stride:.1f} Hz | device={self.device.type}"
            )

    def predict_latest_joint_angles(self, raw_window_tc: np.ndarray) -> np.ndarray:
        env_tc = preprocess_envelope(raw_window_tc, fs=self.emg_fs, cutoff_hz=4.0, clip=3.0)
        env_norm_tc = ((env_tc - self.mean) / self.std).astype(np.float32)

        emg_bct = torch.from_numpy(env_norm_tc.T).unsqueeze(0).float().to(self.device)
        T = emg_bct.shape[-1]
        no_ik_failure = torch.ones((1, 1, T), dtype=torch.float32, device=self.device)

        batch = {
            "emg": emg_bct,
            "joint_angles": torch.zeros((1, 20, T), dtype=torch.float32, device=self.device),
            "no_ik_failure": no_ik_failure,
        }

        with torch.no_grad():
            preds, _, _ = self.module.forward(batch)

        return preds[0, :, -1].detach().cpu().numpy().astype(np.float32)

    # ---------- plot update ----------
    def update_emg_plot(self):
        if self._emg_buf_n < 2:
            return

        n = self._emg_buf_n
        end = self._emg_buf_write
        start = (end - n) % self._emg_buf_len
        if start < end:
            y = self._emg_buf[start:end]
        else:
            y = np.vstack([self._emg_buf[start:], self._emg_buf[:end]])

        ds = max(1, int(self.emg_plot_downsample))
        if ds > 1:
            y = y[::ds]

        t = np.arange(len(y), dtype=np.float32) * (ds / float(self.emg_fs))
        if t[-1] > self.emg_window_s:
            t = t - (t[-1] - self.emg_window_s)

        y_raw = y * self.emg_gain + self.emg_offsets[None, :]
        rect = np.abs(y)

        env = np.empty_like(rect)
        env_state = self._emg_env_state.copy()
        a = float(self.emg_env_alpha)
        for i in range(rect.shape[0]):
            env_state = (1.0 - a) * env_state + a * rect[i]
            env[i] = env_state
        y_env = env * self.emg_gain + self.emg_offsets[None, :]

        for ch in range(8):
            self._emg_raw_curves[ch].setData(t, y_raw[:, ch])
            self._emg_env_curves[ch].setData(t, y_env[:, ch])

        self.emg_plot.setXRange(max(0, t[-1] - self.emg_window_s), t[-1], padding=0)

    # ---------- overlays ----------
    def update_overlays(self):
        # Webcam
        self.ov_webcam.setText(f"Webcam: {self._webcam_ema_fps:.1f} FPS")
        self.ov_webcam.adjustSize()

        # GT mesh
        self.ov_gt.setText(f"GT mesh: {self.gt_mesh.hz:.1f} Hz")
        self.ov_gt.adjustSize()

        # EMG
        self.ov_emg.setText(f"EMG: {self._emg_ema_hz:.0f} Hz")
        self.ov_emg.adjustSize()

        # Pred
        fill = min(1.0, (len(self._raw_buf) / float(self.window_len)) if self.window_len > 0 else 0.0)
        fill_pct = int(fill * 100)
        pred_hz = self.pred_mesh.hz
        self.ov_pred.setText(f"Pred: {pred_hz:.1f} Hz | Buffer: {fill_pct}%")
        self.ov_pred.adjustSize()

    # ---------- cleanup ----------
    def closeEvent(self, event):
        try:
            self.stop_emg()
            if self.cap:
                self.cap.release()
        finally:
            super().closeEvent(event)


def main():
    mp.freeze_support()

    # =========================
    # ✅ EDIT THIS
    # =========================
    CHECKPOINT_PATH = r"C:\Users\kaich\Downloads\last.ckpt"

    if not os.path.exists(CHECKPOINT_PATH):
        print("ERROR: checkpoint not found:", CHECKPOINT_PATH)
        print("Edit CHECKPOINT_PATH to your real .ckpt file.")
        return

    app = QtWidgets.QApplication([])
    w = Live4PanelApp(
        checkpoint_path=CHECKPOINT_PATH,
        cam_index=0,
        webcam_target_fps=21.0,
        show_mediapipe_landmarks=True,
        emg_fs=200.0,
        window_len=2190,
        stride=20,
    )
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()