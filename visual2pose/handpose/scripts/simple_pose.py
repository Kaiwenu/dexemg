import sys
import cv2
import mediapipe as mp
import numpy as np
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal

# MediaPipe Hand Connection Map
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),       # Thumb
    (0,5), (5,6), (6,7), (7,8),       # Index
    (5,9), (9,10), (10,11), (11,12),  # Middle
    (9,13), (13,14), (14,15), (15,16),# Ring
    (13,17), (0,17), (17,18), (18,19), (19,20) # Pinky
]

class CaptureThread(QThread):
    data_received = pyqtSignal(np.ndarray)

    def run(self):
        cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.7
        )

        while cap.isOpened():
            success, image = cap.read()
            if not success: continue

            # Flip for mirror effect and process
            image = cv2.flip(image, 1)
            results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract 21 points as (x, y, z)
                    points = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark])
                    self.data_received.emit(points)
        cap.release()

class HandPlotter:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.view = gl.GLViewWidget()
        self.view.show()
        self.view.setWindowTitle('3D Hand Pose')

        # Add a grid for perspective
        grid = gl.GLGridItem()
        self.view.addItem(grid)

        # Skeleton segments (using GLLinePlotItem)
        self.lines = []
        for _ in HAND_CONNECTIONS:
            line = gl.GLLinePlotItem(mode='lines', width=2, antialias=True)
            self.view.addItem(line)
            self.lines.append(line)

        self.thread = CaptureThread()
        self.thread.data_received.connect(self.update_plot)
        self.thread.start()

    def update_plot(self, points):
        # 1. Zero-Center at the wrist
        wrist_pos = points[0]
        pts = points - wrist_pos
        
        # 2. Scale
        pts = pts * 10 

        # 3. Rotate 90 degrees around X-axis
        # Logic: New Y = Old Z, New Z = -Old Y
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        
        # Reconstruct rotated points
        # We swap Y and Z to change the orientation from "depth" to "up"
        pts = np.column_stack((x, z, -y))

        # 4. Update the lines
        for i, (start, end) in enumerate(HAND_CONNECTIONS):
            line_pts = np.array([pts[start], pts[end]])
            self.lines[i].setData(pos=line_pts, color=(0, 1, 0, 1)) # Green

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == '__main__':
    plotter = HandPlotter()
    plotter.run()