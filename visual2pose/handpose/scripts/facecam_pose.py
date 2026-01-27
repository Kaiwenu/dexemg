import sys
import cv2
import mediapipe as mp
import numpy as np
import time
from PyQt5 import QtGui, QtWidgets, QtCore

class QtCapture(QtWidgets.QWidget):
    def __init__(self, *args):
        super(QtWidgets.QWidget, self).__init__()

        self.fps = 24
        self.cap = cv2.VideoCapture(*args)
        
        # 1. MediaPipe Setup
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 2. UI Setup: Horizontal layout with 3 windows
        self.raw_view = QtWidgets.QLabel("Raw Video")
        self.annotated_view = QtWidgets.QLabel("Tracking Overlay")
        self.pose_view = QtWidgets.QLabel("Skeleton Only")

        layout = QtWidgets.QHBoxLayout()
        for view in [self.raw_view, self.annotated_view, self.pose_view]:
            layout.addWidget(view)
        self.setLayout(layout)

        # 3. Video Recording Logic
        self.isCapturing = False
        self.video_writer = None

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Process Hand Pose
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

        # --- Recording Logic with Timestamp ---
        if self.isCapturing:
            if self.video_writer is None:
                # Initialize writer (mp4v codec)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                filename = f"hand_track_{int(time.time())}.mp4"
                self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))
            
            # Add Timestamp to the recorded frame
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            cv2.putText(annotated_img, timestamp, (10, h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            self.video_writer.write(annotated_img)

        # Update the UI
        self.display_image(frame, self.raw_view)
        self.display_image(annotated_img, self.annotated_view)
        self.display_image(pose_only_img, self.pose_view)

    def display_image(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(400, 300, QtCore.Qt.KeepAspectRatio))

    def capture(self):
        """Toggle video recording"""
        if not self.isCapturing:
            self.isCapturing = True
            print("Recording started...")
        else:
            self.isCapturing = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print("Recording stopped and saved.")

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(int(1000./self.fps))

    def stop(self):
        if hasattr(self, 'timer'): self.timer.stop()
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def deleteLater(self):
        self.stop()
        self.cap.release()
        self.mp_hands.close()
        super(QtWidgets.QWidget, self).deleteLater()

class ControlWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.capture_widget = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Hand Tracking Control Panel')
        self.setGeometry(100, 100, 300, 250)

        layout = QtWidgets.QVBoxLayout(self)

        self.btn_start = QtWidgets.QPushButton('1. Open Camera / Start Feed')
        self.btn_stop = QtWidgets.QPushButton('2. Stop Camera')
        self.btn_record = QtWidgets.QPushButton('3. Start/Stop Recording Video')
        self.btn_quit = QtWidgets.QPushButton('4. Quit Application')

        self.btn_start.clicked.connect(self.startVideo)
        self.btn_stop.clicked.connect(self.stopVideo)
        self.btn_record.clicked.connect(self.toggleRecord)
        self.btn_quit.clicked.connect(self.quitApp)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_record)
        layout.addWidget(self.btn_quit)

    def startVideo(self):
        if not self.capture_widget:
            self.capture_widget = QtCapture(0)
            self.capture_widget.setWindowTitle("Three-Stream Hand Pose Visualization")
        self.capture_widget.start()
        self.capture_widget.show()

    def stopVideo(self):
        if self.capture_widget:
            self.capture_widget.stop()

    def toggleRecord(self):
        if self.capture_widget:
            self.capture_widget.capture()
            # Change button color to indicate recording status
            if self.capture_widget.isCapturing:
                self.btn_record.setStyleSheet("background-color: red; color: white;")
            else:
                self.btn_record.setStyleSheet("")

    def quitApp(self):
        if self.capture_widget:
            self.capture_widget.deleteLater()
        self.close()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # Set a clean style
    app.setStyle('Fusion')
    
    main_window = ControlWindow()
    main_window.show()
    
    sys.exit(app.exec_())