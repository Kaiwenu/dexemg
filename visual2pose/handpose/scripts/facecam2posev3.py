import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import pyqtgraph.opengl as gl
from PyQt5 import QtGui, QtWidgets, QtCore

class HandPlotter(gl.GLViewWidget):
    def __init__(self, model_data):
        super().__init__()
        self.json_data = model_data
        
        # 1. Mesh Normalization
        raw_verts = np.array(model_data['mesh_vertices'])
        self.base_verts = raw_verts - np.mean(raw_verts, axis=0)
        self.base_verts /= (np.max(np.abs(self.base_verts)) + 1e-6)
        
        self.mesh_faces = np.array(model_data['mesh_triangles'], dtype=np.int32).reshape(-1, 3)
        self.weights = np.array(model_data['dense_bone_weights'])
        
        # 2. Rest Pose Normalization
        indices_17 = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20]
        raw_rest = np.array(model_data['landmark_rest_positions'])[indices_17]
        self.rest_17 = raw_rest - raw_rest[0]
        self.rest_17 /= (np.max(np.abs(self.rest_17)) + 1e-6)

        # 3. GL Items
        self.mesh_item = gl.GLMeshItem(vertexes=self.base_verts, faces=self.mesh_faces, shader='shaded', color=(0.4, 0.4, 0.7, 1.0))
        self.addItem(self.mesh_item)
        self.joint_dots = gl.GLScatterPlotItem(pos=np.zeros((17, 3)), color=(1, 0, 0, 1), size=12)
        self.addItem(self.joint_dots)

        # Calibration State
        self.is_calibrated = False
        self.calibration_offset = np.zeros((17, 3))
        self.current_live_norm = None

        self.depth_boost = 6.0

    def calibrate(self):
        if self.current_live_norm is not None:
            self.calibration_offset = self.current_live_norm - self.rest_17
            self.is_calibrated = True

    def reset_calibration(self):
        self.calibration_offset = np.zeros((17, 3))
        self.is_calibrated = False

    def update_3d_pose(self, mp_landmarks):
        # 1. FIX SIDEWAYS ORIENTATION
        # We swap X and Y to rotate the entire hand 90 degrees
        raw_points = mp_landmarks.copy()
        corrected = np.zeros_like(raw_points)
        corrected[:, 0] = raw_points[:, 0]   # X stays X
        corrected[:, 1] = -raw_points[:, 2]  # Y becomes -Z (brings depth to height)
        corrected[:, 2] = -raw_points[:, 1]  # Z becomes -Y (brings height to depth)
        
        # 2. Slice and Normalize
        indices_17 = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20]
        live_17 = corrected[indices_17]
        live_17_centered = live_17 - live_17[0]
        self.current_live_norm = live_17_centered / (np.max(np.abs(live_17_centered)) + 1e-6)

        # 3. Apply Calibration & Skinning
        final_displacement = (self.current_live_norm - self.rest_17) - self.calibration_offset
        vertex_shifts = self.weights @ final_displacement 
        
        new_vertices = self.base_verts + vertex_shifts
        self.mesh_item.setMeshData(vertexes=new_vertices.astype(np.float32), faces=self.mesh_faces)
        self.joint_dots.setData(pos=self.current_live_norm)

class QtCapture(QtWidgets.QWidget):
    pose_data_ready = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, source=0):
        super().__init__()
        self.cap = cv2.VideoCapture(source)
        self.fps = 24
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # UI Views
        self.raw_view = QtWidgets.QLabel("Raw")
        self.annotated_view = QtWidgets.QLabel("Annotated")
        self.pose_view = QtWidgets.QLabel("Pose Only")

        layout = QtWidgets.QHBoxLayout()
        for v in [self.raw_view, self.annotated_view, self.pose_view]:
            layout.addWidget(v)
        self.setLayout(layout)

        self.isCapturing = False
        self.video_writer = None
        self.parent_slider = None

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        if not ret:
            if hasattr(self, 'timer'): self.timer.stop()
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)

        annotated_img = frame.copy()
        pose_only_img = np.zeros((h, w, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark])
                self.pose_data_ready.emit(points) # Update 3D

                self.mp_draw.draw_landmarks(annotated_img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                self.mp_draw.draw_landmarks(pose_only_img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Timeline logic
        if self.parent_slider:
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.parent_slider.blockSignals(True)
            self.parent_slider.setValue(current_pos)
            self.parent_slider.blockSignals(False)

        # Recording logic
        if self.isCapturing:
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(f"capture_{int(time.time())}.mp4", fourcc, self.fps, (w, h))
            self.video_writer.write(annotated_img)

        self.display_image(frame, self.raw_view)
        self.display_image(annotated_img, self.annotated_view)
        self.display_image(pose_only_img, self.pose_view)

    def display_image(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(320, 240, QtCore.Qt.KeepAspectRatio))

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(int(1000./self.fps))

class ControlWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        with open('generic_hand_model.json', 'r') as f:
            self.model_data = json.load(f)
        
        self.plotter = HandPlotter(self.model_data)
        self.plotter.show()
        self.capture_widget = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Hand Tracking & Replay')
        layout = QtWidgets.QVBoxLayout(self)

        self.btn_live = QtWidgets.QPushButton('Start Live Camera')
        self.btn_load = QtWidgets.QPushButton('Load Video for Replay')
        self.btn_record = QtWidgets.QPushButton('Start/Stop Recording')
        self.btn_calib = QtWidgets.QPushButton('Calibrate (Relative Reset)')
        self.btn_reset = QtWidgets.QPushButton('Reset to JSON Default')
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)

        for btn in [self.btn_live, self.btn_load, self.btn_record, self.btn_calib, self.btn_reset, self.slider]:
            layout.addWidget(btn)

        self.btn_live.clicked.connect(self.startLive)
        self.btn_load.clicked.connect(self.loadVideo)
        self.btn_record.clicked.connect(self.toggleRecord)
        self.btn_calib.clicked.connect(self.plotter.calibrate)
        self.btn_reset.clicked.connect(self.plotter.reset_calibration)
        self.slider.sliderMoved.connect(self.seek_video)

        self.lbl_speed = QtWidgets.QLabel("Playback Speed: 1.0x")
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setRange(1, 200) # 0.1x to 2.0x
        self.speed_slider.setValue(100)
        
        layout.addWidget(self.lbl_speed)
        layout.addWidget(self.speed_slider)
        
        self.speed_slider.valueChanged.connect(self.update_speed)

    def update_speed(self, value):
        speed = value / 100.0
        self.lbl_speed.setText(f"Playback Speed: {speed:.1f}x")
        if self.capture_widget and hasattr(self.capture_widget, 'timer'):
            # Adjust the timer interval based on speed
            new_interval = int(1000. / (self.capture_widget.fps * speed))
            self.capture_widget.timer.setInterval(max(1, new_interval))

    def startLive(self):
        self.setup_capture(0)
        self.slider.setEnabled(False)

    def loadVideo(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Video', '', "Video files (*.mp4 *.avi)")
        if fname:
            self.setup_capture(fname)
            total_frames = int(self.capture_widget.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setRange(0, total_frames - 1)
            self.slider.setEnabled(True)

    def setup_capture(self, source):
        if self.capture_widget: self.capture_widget.deleteLater()
        self.capture_widget = QtCapture(source)
        self.capture_widget.pose_data_ready.connect(self.plotter.update_3d_pose)
        if source != 0: self.capture_widget.parent_slider = self.slider
        self.capture_widget.show()
        self.capture_widget.start()

    def seek_video(self, position):
        if self.capture_widget:
            self.capture_widget.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            self.capture_widget.nextFrameSlot()

    def toggleRecord(self):
        if self.capture_widget:
            self.capture_widget.isCapturing = not self.capture_widget.isCapturing
            self.btn_record.setStyleSheet(f"background-color: {'red' if self.capture_widget.isCapturing else 'none'};")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = ControlWindow()
    main.show()
    sys.exit(app.exec_())