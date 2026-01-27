import sys
import cv2
import mediapipe as mp
import numpy as np
import time
from PyQt5 import QtGui, QtWidgets, QtCore

class QtCapture(QtWidgets.QWidget):
    def __init__(self, source=0):
        super(QtWidgets.QWidget, self).__init__()
        self.cap = cv2.VideoCapture(source)
        self.fps = 24
        
        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # 3-Window UI
        self.raw_view = QtWidgets.QLabel("Raw")
        self.annotated_view = QtWidgets.QLabel("Annotated")
        self.pose_view = QtWidgets.QLabel("Pose Only")

        layout = QtWidgets.QHBoxLayout()
        for v in [self.raw_view, self.annotated_view, self.pose_view]:
            layout.addWidget(v)
        self.setLayout(layout)

        self.isCapturing = False
        self.video_writer = None
        self.parent_slider = None # Linked from ControlWindow

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        if not ret:
            # If we reach the end of a file, stop the timer
            if hasattr(self, 'timer'): self.timer.stop()
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Process Pose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)

        annotated_img = frame.copy()
        pose_only_img = np.zeros((h, w, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Custom colors (BGR format: Blue, Green, Red)
                # Example: Neon Green (0, 255, 0) for lines, Light Blue (255, 200, 0) for dots
                dot_style = self.mp_draw.DrawingSpec(color=(255, 200, 0), thickness=8, circle_radius=8)
                line_style = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=6)

                # Update the draw calls for both windows
                self.mp_draw.draw_landmarks(
                    annotated_img, 
                    hand_landmarks, 
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=dot_style,
                    connection_drawing_spec=line_style
                )
                
                self.mp_draw.draw_landmarks(
                    pose_only_img, 
                    hand_landmarks, 
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=dot_style,
                    connection_drawing_spec=line_style
                )

        # Update Slider Position if playing a file
        if self.parent_slider:
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.parent_slider.blockSignals(True) # Prevent feedback loop
            self.parent_slider.setValue(current_pos)
            self.parent_slider.blockSignals(False)

        # Recording Logic
        if self.isCapturing:
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(f"capture_{int(time.time())}.mp4", fourcc, self.fps, (w, h))
            
            # Timestamp on recording
            ts = time.strftime("%H:%M:%S")
            cv2.putText(annotated_img, ts, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
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

    def stop(self):
        if hasattr(self, 'timer'): self.timer.stop()

class ControlWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.capture_widget = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Hand Tracking & Replay')
        layout = QtWidgets.QVBoxLayout(self)

        self.btn_live = QtWidgets.QPushButton('Start Live Camera')
        self.btn_load = QtWidgets.QPushButton('Load Video for Replay')
        self.btn_record = QtWidgets.QPushButton('Start/Stop Recording')
        
        # --- Timeline Slider ---
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.seek_video) # The scrubbing logic

        layout.addWidget(self.btn_live)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.slider)
        layout.addWidget(self.btn_record)

        self.btn_live.clicked.connect(self.startLive)
        self.btn_load.clicked.connect(self.loadVideo)
        self.btn_record.clicked.connect(self.toggleRecord)

    def startLive(self):
        if self.capture_widget: self.capture_widget.deleteLater()
        self.capture_widget = QtCapture(0)
        self.capture_widget.show()
        self.capture_widget.start()
        self.slider.setEnabled(False)

    def loadVideo(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Video', '', "Video files (*.mp4 *.avi)")
        if fname:
            if self.capture_widget: self.capture_widget.deleteLater()
            self.capture_widget = QtCapture(fname)
            self.capture_widget.parent_slider = self.slider
            
            # Set slider range based on file
            total_frames = int(self.capture_widget.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setRange(0, total_frames - 1)
            self.slider.setEnabled(True)
            
            self.capture_widget.show()
            self.capture_widget.start()

    def seek_video(self, position):
        """Scrub through the video file"""
        if self.capture_widget:
            # Tell OpenCV to jump to the frame
            self.capture_widget.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            # Update the view immediately
            self.capture_widget.nextFrameSlot()

    def toggleRecord(self):
        if self.capture_widget:
            self.capture_widget.isCapturing = not self.capture_widget.isCapturing
            color = "red" if self.capture_widget.isCapturing else "none"
            self.btn_record.setStyleSheet(f"background-color: {color};")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = ControlWindow()
    main.show()
    sys.exit(app.exec_())