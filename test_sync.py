# integrated_emg_angles_record.py
import os
import time
import json
import multiprocessing as mp
import numpy as np
import cv2
import h5py

from PyQt5 import QtCore, QtGui, QtWidgets

import mediapipe as mp_hands_lib
from pyomyo import Myo, emg_mode
# from save_hdf5 import save_emg2pose_hdf5

from pathlib import Path

import pyqtgraph.opengl as gl
import numpy as np
import emg2pose.visualization as visualization
import pyqtgraph as pg
# ---------------------------
# Replace with your function:
# landmark2angle.compute_hand_angles(pts21x3) -> (20,) angles
# ---------------------------
import landmark2angle as landmark2angle
import angles as ang

# ============================================================
# 1) Myo worker (separate process)
# ============================================================
def myo_worker(q: mp.Queue, mode):
    m = Myo(mode=mode)
    m.connect()

    def on_emg(emg, movement):
        # emg is length-8
        q.put((time.perf_counter_ns(), list(emg)))

    m.add_emg_handler(on_emg)
    m.set_leds([128, 128, 0], [128, 128, 0])
    m.vibrate(1)

    while True:
        m.run()


# ============================================================
# 4) Qt app: camera + mediapipe + record control + EMG draining
# ============================================================
class RecorderApp(QtWidgets.QWidget):
    def __init__(self, cam_index=0, fps=24):
        super().__init__()
        self.setWindowTitle("EMG + Angles Recorder (Qt + Myo)")

        # ---- camera / mediapipe ----
        self.cap = cv2.VideoCapture(cam_index)
        self.video_writer = None
        self.fps = fps

        self.hands = mp_hands_lib.solutions.hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
            max_num_hands=1,
            model_complexity=1,
        )
        self.drawer = mp_hands_lib.solutions.drawing_utils

        # -----------------------
        # Layout: 2x2 grid
        # -----------------------
        root = QtWidgets.QGridLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setHorizontalSpacing(10)
        root.setVerticalSpacing(10)

        # ---------- Top-left: camera view (landmarks) ----------
        self.view = QtWidgets.QLabel()
        self.view.setFixedSize(640, 480)
        self.view.setStyleSheet("background:#111;")
        root.addWidget(self.view, 0, 0)

        # ---------- Top-right: GL hand mesh ----------
        self.glview = gl.GLViewWidget()
        self.glview.setFixedSize(640, 480)
        self.glview.opts["distance"] = 300  # adjust as needed for your mesh scale
        root.addWidget(self.glview, 0, 1)

        grid = gl.GLGridItem()
        grid.scale(20, 20, 1)
        self.glview.addItem(grid)

        self._mesh_item = None
        self._mesh_faces = None
        self._mesh_tick = 0
        self._mesh_update_every = 2  # set to 2 or 3 if slow

        # ---------- Bottom-left: live sEMG plot ----------
        # Live sEMG signal (RAW Myo defaults tuned for RAW)
        self.emg_fs = 200
        self.emg_window_s = 5.0
        self.emg_plot_fps = 30
        self.emg_plot_downsample = 2  # plot at 100 Hz

        self.emg_plot = pg.PlotWidget(title="Live sEMG (RAW) - 8 channels")
        self.emg_plot.setFixedSize(640, 480)
        self.emg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.emg_plot.setLabel("bottom", "Time", units="s")
        self.emg_plot.setLabel("left", "Amplitude (offset per channel)")
        self.emg_plot.addLegend()

        # RAW is roughly [-128,127]; offsets ~250-350 look good
        self.emg_offsets = np.arange(8, dtype=np.float32) * 300.0
        self.emg_gain = 1.0

        # ring buffer (last window)
        self._emg_buf_len = int(self.emg_window_s * self.emg_fs)
        self._emg_buf = np.zeros((self._emg_buf_len, 8), dtype=np.float32)
        self._emg_buf_n = 0
        self._emg_buf_write = 0

        # EMA envelope state (per-channel)
        self.emg_env_alpha = 0.15  # bigger = smoother (0.1~0.25 nice)
        self._emg_env = np.zeros((8,), dtype=np.float32)

        # curves: raw (faint) + envelope (bold)
        self._emg_raw_curves = []
        self._emg_env_curves = []
        for ch in range(8):
            raw_c = self.emg_plot.plot(pen=pg.mkPen(width=1), name=f"Ch {ch+1} raw")
            env_c = self.emg_plot.plot(pen=pg.mkPen(width=2), name=f"Ch {ch+1} env")
            self._emg_raw_curves.append(raw_c)
            self._emg_env_curves.append(env_c)

        root.addWidget(self.emg_plot, 1, 0)

        # ---------- Bottom-right: controls + status ----------
        controls = QtWidgets.QFrame()
        controls.setFrameShape(QtWidgets.QFrame.StyledPanel)
        controls.setMinimumSize(320, 240)
        root.addWidget(controls, 1, 1)

        ctrl = QtWidgets.QVBoxLayout(controls)
        ctrl.setContentsMargins(12, 12, 12, 12)
        ctrl.setSpacing(10)

        title = QtWidgets.QLabel("Controls")
        title.setStyleSheet("font-size:22px; font-weight:700;")
        ctrl.addWidget(title)

        # Record button
        self.btn_record = QtWidgets.QPushButton("Start Recording")
        self.btn_record.setFixedHeight(44)
        self.btn_record.clicked.connect(self.toggle_recording)
        ctrl.addWidget(self.btn_record)

        # Status text
        self.status = QtWidgets.QLabel("Idle")
        self.status.setStyleSheet("font-size:16px;")
        ctrl.addWidget(self.status)

        # recording timer
        self.timer_label = QtWidgets.QLabel("00:00.0")
        self.timer_label.setStyleSheet("font-size:22px; font-weight:bold; color:#1976d2;")
        ctrl.addWidget(self.timer_label)

        # phase UI (single pair)
        self.phase_label = QtWidgets.QLabel("Phase: -")
        self.phase_label.setStyleSheet("font-size:18px; font-weight:bold;")
        ctrl.addWidget(self.phase_label)

        self.phase_hint = QtWidgets.QLabel("")
        self.phase_hint.setWordWrap(True)
        self.phase_hint.setStyleSheet("font-size:14px; color:#444;")
        ctrl.addWidget(self.phase_hint)

        # spacer
        ctrl.addStretch(1)

        # Hints block
        hint = QtWidgets.QLabel(
            "Hints:\n"
            "• Keep forearm still for cleaner EMG\n"
            "• If mesh is jittery, enable landmark smoothing\n"
            "• RAW EMG is noisy — envelope shows activation\n\n"
            "Saved sessions are written to ./new_data by default."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size:13px; color:#555;")
        ctrl.addWidget(hint)

        # ------------------------
        # shared state, timers, buffers
        # ------------------------
        # NEW: track last phase for beeps
        self._last_phase_key = None

        self.record_start_time = None
        self.record_timer = QtCore.QTimer()
        self.record_timer.timeout.connect(self.update_timer)
        self.record_timer.setInterval(100)  # update every 100 ms

        # ---- recording buffers ----
        self.is_recording = False

        self.angles_x = []
        self.angles_t_ns = []

        self.emg_x = []
        self.emg_t_ns = []

        # ---- EMG process + queue ----
        self.emg_q = mp.Queue()
        self.emg_proc = None

        # Drain EMG queue while Qt loop runs
        self.emg_drain_timer = QtCore.QTimer()
        self.emg_drain_timer.timeout.connect(self.drain_emg_queue)
        self.emg_drain_timer.setInterval(5)  # ms

        # Camera timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_frame)
        self.timer.start(int(1000 / self.fps))

        self.video_path = None
        self.video_frame_count = 0
        self.video_start_time = None
        self.video_writer_fps = None

        # smoother instances (keep existing defaults)
        self.smoother = landmark2angle.RobustAngleSmoother(alpha=0.8, max_jump=0.1)
        self.lm_smoother = landmark2angle.LandmarkEMASmootherRobust(alpha=0.2, max_step=0.2)

        # plot refresh timer (queue drain is separate)
        self.emg_plot_timer = QtCore.QTimer()
        self.emg_plot_timer.timeout.connect(self.update_emg_plot)
        self.emg_plot_timer.start(int(1000 / self.emg_plot_fps))

        # start EMG process / draining on app start
        self.start_emg()

        # give the grid flexible expansion behavior
        root.setColumnStretch(0, 1)
        root.setColumnStretch(1, 1)
        root.setRowStretch(0, 1)
        root.setRowStretch(1, 1)

    def update_live_hand_mesh(self, ja20: np.ndarray):
        """
        ja20: (20,) joint angles float32
        Uses emg2pose Plotly Mesh3d trace and renders it live via GLMeshItem.
        """
        self._mesh_tick += 1
        if (self._mesh_tick % self._mesh_update_every) != 0:
            return

        # Create Plotly Mesh3d
        hand_mesh = visualization.generate_hand_mesh_from_joint_angles(
            joint_angles=ja20,
            color="lightpink",
            flip=False,
            opacity=0.5,
        )

        # Extract vertices
        x = np.asarray(hand_mesh.x, dtype=np.float32)
        y = np.asarray(hand_mesh.y, dtype=np.float32)
        z = np.asarray(hand_mesh.z, dtype=np.float32)
        verts = np.stack([x, y, z], axis=1)  # (V,3)

        # Cache faces (topology should be constant)
        if self._mesh_faces is None:
            i = np.asarray(hand_mesh.i, dtype=np.int32)
            j = np.asarray(hand_mesh.j, dtype=np.int32)
            k = np.asarray(hand_mesh.k, dtype=np.int32)
            self._mesh_faces = np.stack([i, j, k], axis=1)  # (F,3)

        # Create/update GL mesh
        if self._mesh_item is None:
            mesh_color = QtGui.QColor(255, 182, 193, 200)  # R,G,B,Alpha
            self._mesh_item = gl.GLMeshItem(
                vertexes=verts,
                faces=self._mesh_faces,
                smooth=True,
                drawEdges=True,
                color=mesh_color
            )
            self.glview.addItem(self._mesh_item)
        else:
            self._mesh_item.setMeshData(vertexes=verts, faces=self._mesh_faces)
    def _phase_from_elapsed(self, t: float):
        """
        Returns (phase_key, action_text, hint_text, seconds_to_next_change)
        Schedule:
          0-5s    REST BASELINE
          5-60s   repeat 7s cycle: OPEN(2) -> CLOSE(2) -> HOLD(1) -> OPEN(2)
          60-65s  REST BASELINE
          >=65s   STOP
        """
        # 0:00-0:05
        if t < 10.0:
            return ("REST0", "REST BASELINE", "Hand relaxed, no movement", 5.0 - t)

        # 1:00-1:05
        if 60.0 <= t < 70.0:
            return ("REST1", "REST BASELINE", "Hand flat on table, relaxed", 65.0 - t)

        # stop
        if t >= 70.0:
            return ("STOP", "STOP RECORDING", "Auto-stopping…", 0.0)

        # 0:10-1:00 repeating cycle
        cycle_t = (t - 10.0) % 7.0  # [0,7)
        # OPEN 2s
        if cycle_t < 2.0:
            return ("OPEN", "OPEN HAND", "Spread fingers wide, hold", 2.0 - cycle_t)
        # CLOSE 2s
        if cycle_t < 4.0:
            return ("CLOSE", "CLOSE TO GRASP", "Curl fingers to loose fist (like holding egg)", 4.0 - cycle_t)
        # HOLD 1s
        if cycle_t < 5.0:
            return ("HOLD", "HOLD GRASP", "Maintain grasp position", 5.0 - cycle_t)
        # OPEN 2s
        return ("OPEN2", "OPEN HAND", "Spread fingers wide, hold", 7.0 - cycle_t)

    def _maybe_beep_on_phase_change(self, phase_key: str):
        if phase_key != self._last_phase_key:
            self._last_phase_key = phase_key
            QtWidgets.QApplication.beep()

    # ---------------------
    # EMG control
    # ---------------------
    def start_emg(self):
        # MODE = emg_mode.PREPROCESSED
        MODE = emg_mode.RAW
        self.emg_proc = mp.Process(target=myo_worker, args=(self.emg_q, MODE))
        self.emg_proc.start()
        self.emg_drain_timer.start()

    def stop_emg(self):
        self.emg_drain_timer.stop()
        self.drain_emg_queue()  # final drain

        if self.emg_proc is not None:
            self.emg_proc.terminate()
            self.emg_proc.join()
            self.emg_proc = None

    def drain_emg_queue(self):
        while not self.emg_q.empty():
            t_ns, emg = self.emg_q.get()

            # keep your original recording buffers
            self.emg_t_ns.append(t_ns)
            self.emg_x.append(emg)

            # ---- live plot feeding ----
            emg = np.asarray(emg, dtype=np.float32)
            if emg.shape[0] != 8:
                continue

            # store RAW
            self._emg_buf[self._emg_buf_write] = emg
            self._emg_buf_write = (self._emg_buf_write + 1) % self._emg_buf_len
            self._emg_buf_n = min(self._emg_buf_n + 1, self._emg_buf_len)

            # update EMA envelope (rectified)
            rect = np.abs(emg)
            a = float(self.emg_env_alpha)
            self._emg_env = (1.0 - a) * self._emg_env + a * rect

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

        # time axis (0..window)
        t = np.arange(len(y), dtype=np.float32) * (ds / float(self.emg_fs))
        # show only last window seconds
        if t[-1] > self.emg_window_s:
            t = t - (t[-1] - self.emg_window_s)

        # raw + per-channel offsets
        y_raw = y * self.emg_gain + self.emg_offsets[None, :]

        # envelope curve: use latest EMA value as a flat reference + also plot an env history
        # (cheap approximation: EMA of abs on the shown segment)
        rect = np.abs(y)
        env = np.empty_like(rect)
        env_state = self._emg_env.copy()
        a = float(self.emg_env_alpha)
        # run EMA forward on the displayed segment to get a smooth env trace
        for i in range(rect.shape[0]):
            env_state = (1.0 - a) * env_state + a * rect[i]
            env[i] = env_state
        y_env = env * self.emg_gain + self.emg_offsets[None, :]

        for ch in range(8):
            self._emg_raw_curves[ch].setData(t, y_raw[:, ch])
            self._emg_env_curves[ch].setData(t, y_env[:, ch])

        self.emg_plot.setXRange(max(0, t[-1] - self.emg_window_s), t[-1], padding=0)
    # ---------------------
    # Recording toggle
    # ---------------------
    def toggle_recording(self):
        if not self.is_recording:
            # start
            self.video_frame_count = 0
            self.video_path = None
            self.video_start_time = time.perf_counter()
            self.video_writer_fps = None

            self.is_recording = True
            self.btn_record.setText("Stop & Save")
            self.btn_record.setStyleSheet("background-color:#d32f2f; color:white; font-weight:bold;")

            self.record_start_time = time.perf_counter()
            self.record_timer.start()
            self.timer_label.setText("00:00.0")

            self.angles_x.clear()
            self.angles_t_ns.clear()
            self.emg_x.clear()
            self.emg_t_ns.clear()

            # self.start_emg()
            # self.drain_emg_queue()
            self.status.setText("Recording...")

            self._last_phase_key = None
            self.phase_label.setText("Phase: -")
            self.phase_hint.setText("")
        else:
            # stop
            if self.video_start_time is not None:
                dur = time.perf_counter() - self.video_start_time
                eff_fps = self.video_frame_count / max(dur, 1e-6)
                print(f"[VIDEO] frames={self.video_frame_count} dur={dur:.2f}s eff_fps={eff_fps:.2f}")


            self.record_timer.stop()
            self.record_start_time = None

            self.is_recording = False
            self.btn_record.setText("Start Recording")
            self.btn_record.setStyleSheet("")
            self.status.setText("Saving...")



            self.stop_emg()
            self.save_session()

            # Save Video
            if self.video_writer:
                
                self.video_writer.release()
                self.video_writer = None

            # --- AUTO REMUX ---
            if self.video_start_time is not None and self.video_path is not None:
                dur = time.perf_counter() - self.video_start_time
                eff_fps = self.video_frame_count / max(dur, 1e-6)
                print(f"[VIDEO] frames={self.video_frame_count} dur={dur:.2f}s eff_fps={eff_fps:.2f}")

                remux_path = self.video_path.replace(".mp4", "_remux.mp4")
                ok = self.remux_video_with_fps(self.video_path, remux_path, eff_fps)

                if ok:
                    # replace original with remuxed
                    import os
                    os.replace(remux_path, self.video_path)
                    print("[REMUX] replaced original:", self.video_path)
            self.status.setText("Saved. Idle.")

            

    # ---------------------
    # Main camera loop
    # ---------------------
    def on_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        angles20 = None
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            self.drawer.draw_landmarks(frame, lm, mp_hands_lib.solutions.hands.HAND_CONNECTIONS)

            
            # landmarks_smooth = self.lm_smoother(pts)  # (21,3)
            # # each frame:
            # angles = landmark2angle.compute_hand_angles(landmarks_smooth, smoother=self.smoother)
            angles = landmark2angle.compute_hand_angles(pts, smoother=None)
            # Compute angles
            # angles = ang.compute_all_hand_angles(pts)
            angles20 = np.asarray(angles, dtype=np.float32)

            angles20 = np.asarray(angles, dtype=np.float32)

            # live 3D mesh from joint angles
            self.update_live_hand_mesh(angles20)

        # --- ALWAYS write video frames while recording ---
        if self.is_recording:
            # Initialize VideoWriter if needed (use first available frame size)
            if self.video_writer is None:
                timestamp = int(QtCore.QDateTime.currentMSecsSinceEpoch())
                self.base_name = f"capture_{timestamp}"
                data_dir = os.path.join(os.getcwd(), "new_data")
                os.makedirs(data_dir, exist_ok=True)
                self.video_path = os.path.join(data_dir, f"{self.base_name}.mp4")  # <-- store

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (w, h))

            # Write frame NO MATTER WHAT (even if no hand detected)
            self.video_writer.write(frame)
            self.video_frame_count += 1

        # --- angles recording stays conditional on having a hand ---
        if self.is_recording and angles20 is not None:
            self.angles_t_ns.append(time.perf_counter_ns())
            self.angles_x.append(angles20)

        # display
        self.display_frame(frame)

    def display_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.view.size(), QtCore.Qt.KeepAspectRatio)
        self.view.setPixmap(pix)

    def update_timer(self):
        if self.record_start_time is None:
            return

        elapsed = time.perf_counter() - self.record_start_time

        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        tenths = int((elapsed - int(elapsed)) * 10)
        self.timer_label.setText(f"{minutes:02d}:{seconds:02d}.{tenths}")

        phase_key, action, hint, t_next = self._phase_from_elapsed(elapsed)

        # Update phase UI
        self.phase_label.setText(f"Phase: {action}")
        if t_next > 0:
            self.phase_hint.setText(f"{hint}   (next in {t_next:.1f}s)")
        else:
            self.phase_hint.setText(hint)

        # Beep when phase changes (optional)
        self._maybe_beep_on_phase_change(phase_key)

        # Auto-stop at 1:05 (optional)
        if phase_key == "STOP" and self.is_recording:
            QtCore.QTimer.singleShot(0, self.toggle_recording)

    def remux_video_with_fps(self, in_path: str, out_path: str, fps_out: float):
        import cv2, os

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            print("[REMUX] Failed to open:", in_path)
            return False

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, float(fps_out), (w, h))
        if not writer.isOpened():
            cap.release()
            print("[REMUX] Failed to open writer:", out_path)
            return False

        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            n += 1

        cap.release()
        writer.release()

        print(f"[REMUX] wrote {n} frames @ {fps_out:.2f} fps -> {out_path}")
        return True
    
    def save_session(self):
        """
        Save EXACTLY in Emg2PoseSessionData format:

        /emg2pose (group with attrs)
            /timeseries  shape (N,) compound dtype fields:
            time: float64
            joint_angles: float32 (20,)
            emg: float32 (C,)  where C = num_channels (8 or 16)

        NOTE:
        - Metadata MUST be stored on the /emg2pose group (not root).
        - Dataset MUST be named 'timeseries' (not 'data').
        """
        import os, time
        import numpy as np
        import h5py

        if len(self.angles_x) < 5:
            print("Not enough angle frames to save.")
            return
        if len(self.emg_x) < 10:
            print("Not enough EMG samples to save.")
            return

        # --- raw buffers ---
        emg_x = np.asarray(self.emg_x, dtype=np.float32)        # (N,C)
        emg_t = np.asarray(self.emg_t_ns, dtype=np.int64)       # (N,)
        ang_x = np.asarray(self.angles_x, dtype=np.float32)     # (T,20)
        ang_t = np.asarray(self.angles_t_ns, dtype=np.int64)    # (T,)

        # --- sort by time (safety) ---
        e_order = np.argsort(emg_t)
        emg_t = emg_t[e_order]
        emg_x = emg_x[e_order]

        a_order = np.argsort(ang_t)
        ang_t = ang_t[a_order]
        ang_x = ang_x[a_order]

        # --- convert timestamps to seconds (float64) ---
        emg_t_s = emg_t.astype(np.float64) / 1e9
        ang_t_s = ang_t.astype(np.float64) / 1e9

        # --- align angles onto EMG timeline (N,20): nearest-neighbor ---
        # guard for very short angle streams
        if len(ang_t_s) < 2:
            print("Not enough angle timestamps to align.")
            return

        idx = np.searchsorted(ang_t_s, emg_t_s, side="left")
        idx = np.clip(idx, 1, len(ang_t_s) - 1)
        left = idx - 1
        right = idx
        choose_right = (ang_t_s[right] - emg_t_s) < (emg_t_s - ang_t_s[left])
        ang_idx = np.where(choose_right, right, left)
        angles_aligned = ang_x[ang_idx].astype(np.float32)      # (N,20)

        # --- build compound array (N,) with REQUIRED field names ---
        N, C = emg_x.shape
        dt = np.dtype([
            ("time", np.float64),
            ("joint_angles", np.float32, (20,)),
            ("emg", np.float32, (C,)),
        ])
        ts = np.empty((N,), dtype=dt)
        ts["time"] = emg_t_s
        ts["joint_angles"] = angles_aligned
        ts["emg"] = emg_x

        # --- choose output path that Hydra actually reads ---
        # IMPORTANT: save into C:\Users\kaich\emg2pose_dataset_mini
        # out_dir = r"C:\Users\kaich\emg2pose_dataset_mini"
        out_dir = r"C:\Users\kaich\Desktop\dexemg\new_data"
        os.makedirs(out_dir, exist_ok=True)

        unix = int(time.time())
        side = getattr(self, "side", "left")
        stage = getattr(self, "stage", "ThumbsUpDownThumbRotationSCWCWP")
        user = getattr(self, "user", "kaich")
        session = getattr(self, "session", f"session_{unix}_{side}")  # must match split naming
        dataset_name = getattr(self, "dataset", "custom")
        sample_rate = float(getattr(self, "sample_rate", 200))

        out_path = os.path.join(out_dir, f"{session}.hdf5")

        # --- write HDF5 in Emg2PoseSessionData schema ---
        with h5py.File(out_path, "w") as f:
            g = f.create_group("emg2pose")

            # required dataset name: timeseries
            g.create_dataset(
                "timeseries",
                data=ts,
                compression="gzip",
                chunks=True,
                shuffle=True,
            )

            # required: metadata on group attrs
            g.attrs["session"] = str(session)
            g.attrs["user"] = str(user)
            g.attrs["side"] = str(side)
            g.attrs["stage"] = str(stage)
            g.attrs["start"] = float(emg_t_s[0])
            g.attrs["end"] = float(emg_t_s[-1])
            g.attrs["num_channels"] = int(C)
            g.attrs["sample_rate"] = sample_rate
            g.attrs["dataset"] = str(dataset_name)

        print("Saved (Emg2PoseSessionData format):", out_path)


    def closeEvent(self, event):
        try:
            if self.is_recording:
                self.stop_emg()
            if self.cap:
                self.cap.release()
        finally:
            super().closeEvent(event)


def main():
    # IMPORTANT on Windows: multiprocessing needs this guard
    mp.freeze_support()
    app = QtWidgets.QApplication([])
    w = RecorderApp(cam_index=0, fps=21.1)
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
