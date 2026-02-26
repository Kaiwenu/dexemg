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
# 2) Align EMG to frame timestamps (nearest neighbor)
# ============================================================
# def align_nearest(emg_t_ns: np.ndarray, emg_x: np.ndarray, frame_t_ns: np.ndarray) -> np.ndarray:
#     """
#     emg_t_ns: (N,) int64 sorted by time
#     emg_x:    (N,8) float32
#     frame_t_ns:(T,) int64
#     return:   (T,8) float32, nearest emg sample for each frame
#     """
#     if len(emg_t_ns) < 2:
#         raise ValueError("Not enough EMG samples to align.")

#     idx = np.searchsorted(emg_t_ns, frame_t_ns, side="left")
#     idx = np.clip(idx, 1, len(emg_t_ns) - 1)

#     left = idx - 1
#     right = idx

#     choose_right = (emg_t_ns[right] - frame_t_ns) < (frame_t_ns - emg_t_ns[left])
#     out_idx = np.where(choose_right, right, left)

#     return emg_x[out_idx]


# def align_angles_to_emg_times(emg_t_s: np.ndarray, ang_t_s: np.ndarray, ang_x: np.ndarray) -> np.ndarray:
#     """
#     Nearest-neighbor align angles (T,20) onto EMG timeline (N,).
#     Returns angles_aligned (N,20).
#     """
#     emg_t_s = np.asarray(emg_t_s, dtype=np.float64)
#     ang_t_s = np.asarray(ang_t_s, dtype=np.float64)
#     ang_x   = np.asarray(ang_x,   dtype=np.float32)

#     idx = np.searchsorted(ang_t_s, emg_t_s, side="left")
#     idx = np.clip(idx, 1, len(ang_t_s) - 1)

#     left = idx - 1
#     right = idx
#     choose_right = (ang_t_s[right] - emg_t_s) < (emg_t_s - ang_t_s[left])
#     out_idx = np.where(choose_right, right, left)

#     return ang_x[out_idx]  # (N,20)

# ============================================================
# 3) Save one HDF5
# ============================================================
# def save_hdf5(out_path: str,
#               emg_x: np.ndarray, emg_t_ns: np.ndarray,
#               angles_x: np.ndarray, angles_t_ns: np.ndarray,
#               emg_aligned: np.ndarray):
#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
#     with h5py.File(out_path, "w") as f:
#         f.create_dataset("emg/x", data=emg_x, compression="gzip")
#         f.create_dataset("emg/t_ns", data=emg_t_ns, compression="gzip")

#         f.create_dataset("angles/x", data=angles_x, compression="gzip")
#         f.create_dataset("angles/t_ns", data=angles_t_ns, compression="gzip")

#         f.create_dataset("emg_aligned_to_angles/x", data=emg_aligned, compression="gzip")

#         # optional metadata
#         meta = {
#             "user": "kaich",
#             "side": "right",
#             "stage": "OneHandedFreeStyle",
#             "aligned_by": "nearest",
#             "created_unix_s": time.time(),
#         }
#         f.create_dataset("meta/json", data=np.string_(json.dumps(meta)))
# def save_emg2pose_hdf5(
#     out_path: str,
#     emg_x: np.ndarray,          # (N,16) float32
#     emg_t_s: np.ndarray,        # (N,)   float64 seconds (absolute unix time recommended)
#     ang_x: np.ndarray,          # (T,20) float32
#     ang_t_s: np.ndarray,        # (T,)   float64 seconds (same time base as emg_t_s)
#     *,
#     session: str,
#     user: str,
#     side: str,
#     stage: str,
#     dataset: str = "custom",
#     num_channels: int = 8,
#     sample_rate: float = 50,
# ):
#     """
#     Writes an HDF5 file compatible with Emg2PoseSessionData:

#       /emg2pose (group with attrs)
#         /timeseries (compound dataset with fields: time, joint_angles, emg)

#     Notes:
#     - emg_t_s should be monotonic non-decreasing.
#     - Joint angles are aligned to each EMG timestamp -> stored at EMG rate.
#     """

#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     emg_x = np.asarray(emg_x, dtype=np.float32)
#     emg_t_s = np.asarray(emg_t_s, dtype=np.float64)
#     ang_x = np.asarray(ang_x, dtype=np.float32)
#     ang_t_s = np.asarray(ang_t_s, dtype=np.float64)

#     assert emg_x.ndim == 2 and emg_x.shape[1] == num_channels, \
#         f"emg_x must be (N,{num_channels}), got {emg_x.shape}"
#     assert emg_t_s.ndim == 1 and emg_t_s.shape[0] == emg_x.shape[0], \
#         "emg_t_s must be (N,) matching emg_x"
#     assert ang_x.ndim == 2 and ang_x.shape[1] == 20, \
#         f"ang_x must be (T,20), got {ang_x.shape}"
#     assert ang_t_s.ndim == 1 and ang_t_s.shape[0] == ang_x.shape[0], \
#         "ang_t_s must be (T,) matching ang_x"

#     # Emg2PoseSessionData.timestamps assumes monotonic timestamps
#     if len(emg_t_s) >= 2:
#         assert np.all(np.diff(emg_t_s) >= 0), "emg_t_s must be monotonic non-decreasing"

#     # Align joint angles to each EMG timestamp -> (N,20)
#     # You already have this helper; it should return joint angles in the same timebase.
#     ang_aligned = align_angles_to_emg_times(emg_t_s, ang_t_s, ang_x).astype(np.float32)

#     N = emg_x.shape[0]

#     # Must match Emg2PoseSessionData constants:
#     # TIMESERIES="timeseries", fields: EMG="emg", JOINT_ANGLES="joint_angles", TIMESTAMPS="time"
#     ts_dtype = np.dtype([
#         ("time", np.float64),
#         ("joint_angles", np.float32, (20,)),
#         ("emg", np.float32, (num_channels,)),
#     ])

#     ts = np.empty((N,), dtype=ts_dtype)
#     ts["time"] = emg_t_s
#     ts["joint_angles"] = ang_aligned
#     ts["emg"] = emg_x

#     with h5py.File(out_path, "w") as f:
#         g = f.create_group("emg2pose")

#         # Create exactly /emg2pose/timeseries
#         # chunks=True improves random window reads; choose chunk size if you want
#         g.create_dataset(
#             "timeseries",
#             data=ts,
#             compression="gzip",
#             chunks=True,
#             shuffle=True,
#         )

#         # Metadata lives on the GROUP (this is what Emg2PoseSessionData loads)
#         g.attrs["session"] = str(session)
#         g.attrs["user"] = str(user)
#         g.attrs["side"] = str(side)
#         g.attrs["stage"] = str(stage)

#         g.attrs["start"] = float(emg_t_s[0]) if N > 0 else 0.0
#         g.attrs["end"] = float(emg_t_s[-1]) if N > 0 else 0.0

#         g.attrs["num_channels"] = int(num_channels)
#         g.attrs["sample_rate"] = float(sample_rate)

#         # This key exists in the dataclass: DATASET_NAME="dataset"
#         g.attrs["dataset"] = str(dataset)

#     print("Saved (emg2pose format):", str(out_path))
    # # --- exact compound dtype from your screenshot ---
    # dt = np.dtype([
    #     ("time",        "<f8"),
    #     ("joint_angles","<f4", (20,)),
    #     ("emg",         "<f4", (16,))
    # ])

    # data = np.empty((N,), dtype=dt)
    # data["time"] = emg_t_s
    # data["joint_angles"] = ang_aligned
    # data["emg"] = emg_x

    # with h5py.File(out_path, "w") as f:
    #     # dataset name can be whatever your loader expects; common choices: "data" or "samples"
    #     dset = f.create_dataset("data", data=data, compression="gzip")

    #     # --- group/root attrs like in your screenshot ---
    #     f.attrs["start"] = float(emg_t_s[0])
    #     f.attrs["end"] = float(emg_t_s[-1])
    #     f.attrs["sample_rate"] = float(sample_rate)
    #     f.attrs["num_channels"] = int(emg_x.shape[1])

    #     f.attrs["user"] = user
    #     f.attrs["side"] = side
    #     f.attrs["split"] = split
    #     f.attrs["stage"] = stage

    #     f.attrs["moving_hand"] = moving_hand
    #     f.attrs["generalization"] = generalization
    #     f.attrs["held_out_user"] = bool(held_out_user)
    #     f.attrs["held_out_stage"] = bool(held_out_stage)

    #     f.attrs["session"] = session
    #     f.attrs["filename"] = filename



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

        # ---- UI ----
        self.view = QtWidgets.QLabel()
        self.view.setFixedSize(640, 480)

        self.btn_record = QtWidgets.QPushButton("Start Recording")
        self.btn_record.clicked.connect(self.toggle_recording)

        self.status = QtWidgets.QLabel("Idle")



        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.view)
        layout.addWidget(self.btn_record)
        layout.addWidget(self.status)

        # ---- recording timer ----
        self.timer_label = QtWidgets.QLabel("00:00.0")
        self.timer_label.setStyleSheet("font-size:18px; font-weight:bold; color:#1976d2;")

        layout.addWidget(self.timer_label)
        
        # NEW: phase labels
        self.phase_label = QtWidgets.QLabel("Phase: -")
        self.phase_label.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(self.phase_label)

        self.phase_hint = QtWidgets.QLabel("")
        self.phase_hint.setStyleSheet("font-size:14px; color:#444;")
        layout.addWidget(self.phase_hint)

        # NEW: track last phase for beeps
        self._last_phase_key = None
        # NEW: phase labels
        self.phase_label = QtWidgets.QLabel("Phase: -")
        self.phase_label.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(self.phase_label)

        self.phase_hint = QtWidgets.QLabel("")
        self.phase_hint.setStyleSheet("font-size:14px; color:#444;")
        layout.addWidget(self.phase_hint)

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

        #live
        # ---- Live 3D mesh view ----
        # ---- Live 3D Mesh widget ----
        self.glview = gl.GLViewWidget()
        self.glview.setFixedSize(640, 480)
        self.glview.opts["distance"] = 300  # your mesh coords are ~[-30,30], so distance should be big-ish
        layout.addWidget(self.glview)

        grid = gl.GLGridItem()
        grid.scale(20, 20, 1)
        self.glview.addItem(grid)

        self._mesh_item = None
        self._mesh_faces = None
        self._mesh_tick = 0
        self._mesh_update_every = 1  # set to 2 or 3 if slow

        #smoother
        self.smoother = landmark2angle.RobustAngleSmoother(alpha=0.8, max_jump=0.1)
        self.lm_smoother = landmark2angle.LandmarkEMASmootherRobust(alpha=0.2, max_step=0.2)
        self.start_emg()

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
        # drain quickly
        while not self.emg_q.empty():
            t_ns, emg = self.emg_q.get()
            self.emg_t_ns.append(t_ns)
            self.emg_x.append(emg)

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
    # ---------------------
    # Save + align
    # ---------------------
    # def save_session(self):
    #     if len(self.angles_x) < 5:
    #         print("Not enough angle frames to save.")
    #         return
    #     if len(self.emg_x) < 10:
    #         print("Not enough EMG samples to save.")
    #         return

    #     emg_x = np.asarray(self.emg_x, dtype=np.float32)         # (N,8)
    #     emg_t = np.asarray(self.emg_t_ns, dtype=np.int64)        # (N,)
    #     ang_x = np.asarray(self.angles_x, dtype=np.float32)      # (T,20)
    #     ang_t = np.asarray(self.angles_t_ns, dtype=np.int64)     # (T,)

    #     # Ensure EMG is time-sorted (should already be, but just in case)
    #     order = np.argsort(emg_t)
    #     emg_t = emg_t[order]
    #     emg_x = emg_x[order]

    #     emg_aligned = align_nearest(emg_t, emg_x, ang_t)         # (T,8)

    #     out_dir = os.path.join(os.getcwd(), "data")
    #     os.makedirs(out_dir, exist_ok=True)
    #     out_path = os.path.join(out_dir, f"session_{int(time.time())}.hdf5")

    #     save_hdf5(out_path, emg_x, emg_t, ang_x, ang_t, emg_aligned)
    #     print("Saved:", out_path)

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
