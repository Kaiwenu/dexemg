from PyQt5.QtWidgets import QApplication
import sys
from handpose.ui.qt_window import ControlWindow
import transform
import json
import numpy as np
import landmark2angle
import emg2pose.visualization as visualization

import mediapy


def record():
    app = QApplication(sys.argv)
    main = ControlWindow()
    main.show()
    sys.exit(app.exec_())
if __name__ == "__main__":
    record()




    # #test
    # # with open("C:/Users/kaich/Desktop/dexemg/visual2pose/capture_1770019842699_data.json", "r") as f:
    # #     data = json.load(f)
    # with open("C:/Users/kaich/Desktop/dexemg/visual2pose/capture_1770324785306_data.json", "r") as f:
    #     data = json.load(f)
    # # landmarks = np.array(data[50]['landmarks'], dtype=np.float32)
    # # print(landmarks.shape)   # (21, 3)
    # # ja = transform.mp_landmarks_to_emg2pose20(landmarks)
    # # print(ja)
    # # with open("C:/Users/kaich/Desktop/dexemg/visual2pose/capture_1770018524204_data.json", "r") as f:
    # #     data = json.load(f)
    # # with open("C:/Users/kaich/Desktop/dexemg/visual2pose/capture_1769923625725_data.json", "r") as f:
    # #     data = json.load(f)
    # # landmarks = np.array(data[80]['landmarks'], dtype=np.float32)
    # # angles = landmark2angle.compute_hand_angles(landmarks)
    # # fig = visualization.plot_hand_mesh(angles, auto_range=False)
    # # fig.show()

    # # Take first 100 frames
    # landmarks = [frame["landmarks"] for frame in data[0:170]]
    
    # angles = []
    # for landmark in landmarks:
    #     angles.append(landmark2angle.compute_hand_angles(landmark))

    
    # video = visualization.get_plotly_animation_for_joint_angles(angles)
    # video.show()

    # # frames = visualization.joint_angles_to_frames_parallel(angles)
    # # frames = visualization.remove_alpha_channel(frames)
    # # mediapy.show_video(frames, width=800, fps=30, downsample=True)