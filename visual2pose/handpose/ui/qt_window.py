import cv2
import json
import numpy as np
import mediapipe as mp
import os
from PyQt5 import QtGui, QtWidgets, QtCore
from handpose.ui.plotter import HandPlotter
import landmark2angle

# ============================================================
# QtCapture: Handles video input + MediaPipe processing
# ============================================================
class QtCapture(QtWidgets.QWidget):
    pose_data_ready = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, source=0, fps=24):
        super().__init__()

        self.cap = cv2.VideoCapture(source)
        self.fps = fps

        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.recorded_data = [] # Stores landmarks for JSON

        self._init_mediapipe()
        self._init_ui()

        self.parent_slider = None
        self.timer = None

    # ------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------
    def _init_mediapipe(self):
        self.mp_hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
            max_num_hands=1,
            model_complexity=1,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def _init_ui(self):
        self.annotated_view = QtWidgets.QLabel()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.annotated_view)

    # ------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------
    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        if not ret:
            # Clean up writer if we reach end of video while recording
            self.stop_recording()
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb)

        current_pts = None
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            pts = np.array([[l.x, l.y, l.z] for l in lm.landmark], dtype=np.float32)
            # print(pts.shape)
            current_pts = pts
            self.pose_data_ready.emit(pts)
            self.mp_draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)

         # Recording Logic
        if self.is_recording:
            timestamp = int(QtCore.QDateTime.currentMSecsSinceEpoch())
            
            # Initialize VideoWriter if needed
            if self.video_writer is None:
                self.base_name = f"capture_{timestamp}"
                data_dir = os.path.join(os.getcwd(), "data")
                os.makedirs(data_dir, exist_ok=True)
                video_path = os.path.join(data_dir, f"{self.base_name}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
            
            self.video_writer.write(frame)
            
            # Log Landmarks to memory
            # We store a dict with frame index and points
            data_entry = {
                "frame": int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "landmarks": current_pts.tolist() if current_pts is not None else None
            }
            self.recorded_data.append(data_entry)

        if self.parent_slider:
            frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.parent_slider.setValue(frame_idx)

        self._display_image(frame)

    # def stop_recording(self):
    #     """Finalize video file and save JSON log"""
    #     if not self.is_recording:
    #         return
            
    #     self.is_recording = False
        
    #     # Save Video
    #     if self.video_writer:
    #         self.video_writer.release()
    #         self.video_writer = None
            
    #     # Save JSON
    #     if self.recorded_data:
    #         with open(f"{self.base_name}_data.json", "w") as f:
    #             json.dump(self.recorded_data, f, indent=4)
    #         self.recorded_data = []
        
    #     print(f"Recording saved as {self.base_name}.mp4 and .json")

    def convert2angles(self):
        # Take first 100 frames
        landmarks = [frame["landmarks"] for frame in self.recorded_data]
        
        angles = []
        for landmark in landmarks:
            angles.append(landmark2angle.compute_hand_angles(landmark).tolist())
        return angles

    def stop_recording(self):
        """Finalize video file and save JSON log"""
        if not self.is_recording:
            return

        self.is_recording = False

        # Ensure /data directory exists
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)

        # Save Video
        if self.video_writer:
            
            self.video_writer.release()
            self.video_writer = None

        # Save JSON
        if self.recorded_data:
            json_path = os.path.join(data_dir, f"{self.base_name}_landmarks.json")
            with open(json_path, "w") as f:
                json.dump(self.recorded_data, f, indent=4)

            json_path = os.path.join(data_dir, f"{self.base_name}_angles.json")
            angles = self.convert2angles()
            with open(json_path, "w") as f:
                json.dump(angles, f, indent=4)

            self.recorded_data = []

        print(f"Recording saved to {data_dir}/{self.base_name}.mp4 and _data.json")
    # ------------------------------------------------------------
    # Display helper
    # ------------------------------------------------------------
    def _display_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape

        qimg = QtGui.QImage(rgb.data, w, h, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            420, 320, QtCore.Qt.KeepAspectRatio
        )
        self.annotated_view.setPixmap(pix)

    # ------------------------------------------------------------
    # Start capture
    # ------------------------------------------------------------
    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(int(1000.0 / self.fps))


# ============================================================
# ControlWindow: UI for controlling capture + plotter
# ============================================================
class ControlWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.plotter = self._load_plotter()
        self.capture_widget = None

        self._init_ui()

    # ------------------------------------------------------------
    # Load model + plotter
    # ------------------------------------------------------------
    def _load_plotter(self):
        with open("C:/Users/kaich/Desktop/dexemg/visual2pose/handpose/ui/generic_hand_model.json", "r") as f:
            model_data = json.load(f)

        plotter = HandPlotter(model_data)
        plotter.show()
        return plotter

    # ------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------
    def _init_ui(self):
        self.setWindowTitle("Hand Control Panel")
        layout = QtWidgets.QVBoxLayout(self)

        # Buttons
        self.btn_live = QtWidgets.QPushButton("Start Live")
        self.btn_load = QtWidgets.QPushButton("Load Video")
        self.btn_calib_pose = QtWidgets.QPushButton("Calibrate Pose (Snap)")
        self.btn_calib_align = QtWidgets.QPushButton("Calibrate Alignment (Open Hand)")
        self.btn_reset_pose = QtWidgets.QPushButton("Reset Pose Calibration")
        self.btn_record = QtWidgets.QPushButton("Start Recording")

        # Toggles
        self.ghost_check = QtWidgets.QCheckBox("Show Ghost Reference")
        self.ghost_check.setChecked(True)
        self.ghost_check.stateChanged.connect(
            lambda s: self.plotter.toggle_ghost(s == QtCore.Qt.Checked)
        )

        self.swapyz_check = QtWidgets.QCheckBox("Swap Y/Z")
        self.swapyz_check.setChecked(True)
        self.swapyz_check.stateChanged.connect(
            lambda s: setattr(self.plotter, "swap_yz", s == QtCore.Qt.Checked)
        )

        self.flipx_check = QtWidgets.QCheckBox("Flip X (Mirror)")
        self.flipx_check.setChecked(True)
        self.flipx_check.stateChanged.connect(
            lambda s: setattr(self.plotter, "flip_x", s == QtCore.Qt.Checked)
        )

        self.flipz_check = QtWidgets.QCheckBox("Flip Z (Depth)")
        self.flipz_check.setChecked(False)
        self.flipz_check.stateChanged.connect(
            lambda s: setattr(self.plotter, "flip_z", s == QtCore.Qt.Checked)
        )

        # Playback speed
        self.lbl_speed = QtWidgets.QLabel("Playback Speed: 1.0x")
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setRange(10, 300)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self._change_speed)

        # Timeline
        self.timeline = QtWidgets.QSlider(QtCore.Qt.Horizontal)

        # Add widgets
        for w in [
            self.btn_live,
            self.btn_load,
            self.btn_calib_pose,
            self.btn_calib_align,
            self.btn_reset_pose,
            self.ghost_check,
            self.swapyz_check,
            self.flipx_check,
            self.flipz_check,
            self.lbl_speed,
            self.speed_slider,
            self.timeline,
            self.btn_live,
            self.btn_load,
            self.btn_record
        ]:
            layout.addWidget(w)

        # Connect buttons
        self.btn_live.clicked.connect(self._start_live)
        self.btn_load.clicked.connect(self._load_video)
        self.btn_calib_pose.clicked.connect(self.plotter.calibrate_pose_snap)
        self.btn_calib_align.clicked.connect(self.plotter.calibrate_alignment_from_last)
        self.btn_record.clicked.connect(self._toggle_recording)

    # ------------------------------------------------------------
    # Playback speed
    # ------------------------------------------------------------
    def _change_speed(self, val):
        speed = val / 100.0
        self.lbl_speed.setText(f"Playback Speed: {speed:.1f}x")

        if self.capture_widget and self.capture_widget.timer:
            interval = int(1000.0 / (24 * speed))
            self.capture_widget.timer.setInterval(interval)

    # ------------------------------------------------------------
    # Live capture
    # ------------------------------------------------------------
    def _start_live(self):
        self._setup_capture(0)

    def _toggle_recording(self):
        if not self.capture_widget:
            return

        if not self.capture_widget.is_recording:
            self.capture_widget.is_recording = True
            self.btn_record.setText("Stop Recording & Save JSON")
            self.btn_record.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold;")
        else:
            self.capture_widget.stop_recording()
            self.btn_record.setText("Start Recording")
            self.btn_record.setStyleSheet("")

    # ------------------------------------------------------------
    # Load video
    # ------------------------------------------------------------
    def _load_video(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video")
        if not f:
            return

        self._setup_capture(f)

        total = int(self.capture_widget.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.timeline.setRange(0, max(total - 1, 0))

    # ------------------------------------------------------------
    # Capture setup
    # ------------------------------------------------------------
    def _setup_capture(self, src):
        if self.capture_widget:
            self.capture_widget.deleteLater()

        self.capture_widget = QtCapture(src)
        self.capture_widget.pose_data_ready.connect(self.plotter.update_3d_pose)
        self.capture_widget.parent_slider = self.timeline

        self.capture_widget.show()
        self.capture_widget.start()
