import sys
import cv2
import json
import numpy as np
import mediapipe as mp
import pyqtgraph.opengl as gl
from PyQt5 import QtGui, QtWidgets, QtCore












class HandPlotter(gl.GLViewWidget):
    """
    Pipeline:
      1) Take raw MediaPipe 21 (MP order)
      2) Apply axis mapping (y-up + optional swap_yz + flips)
      3) Reorder MP -> MODEL landmark order
      4) Wrist-center using MODEL wrist (index 20)
      5) Global similarity (Umeyama) using palm anchors in MODEL order
      6) Per-bone rigid fits (optional) in the SAME wrist-centered mesh space
      7) LBS blend vertices and render
    """

    def __init__(self, model_data):
        super().__init__()
        self.model_data = model_data

        # ---------------------------
        # Axis / convention toggles (these will be auto-fit)
        # ---------------------------
        self.swap_yz = True
        self.flip_x = True
        self.flip_z = False

        # ---------------------------
        # Mesh setup + normalization transform
        # ---------------------------
        raw_verts = np.array(model_data["mesh_vertices"], dtype=np.float32)
        self.vert_center = raw_verts.mean(axis=0)
        verts_centered = raw_verts - self.vert_center
        self.vert_scale = float(np.max(np.abs(verts_centered)) + 1e-6)

        self.base_verts = verts_centered / self.vert_scale  # (V,3)
        self.mesh_faces = np.array(model_data["mesh_triangles"], dtype=np.int32).reshape(-1, 3)

        # Weights (V,17)
        self.weights = np.array(model_data["dense_bone_weights"], dtype=np.float32)
        self.num_bones = self.weights.shape[1]

        # Normalize weights per-vertex (row sum -> 1)
        row_sum = self.weights.sum(axis=1, keepdims=True)
        self.weights = self.weights / (row_sum + 1e-8)

        # (Optional) sharpen weights to reduce mushiness
        pw = 2.0
        W = np.power(self.weights, pw)
        W = W / (W.sum(axis=1, keepdims=True) + 1e-8)
        self.weights = W.astype(np.float32)

        # ---------------------------
        # Landmarks: rest in mesh space
        # ---------------------------
        # IMPORTANT: JSON landmarks are in MODEL order (your MP_INDEX_FOR_MODEL maps MP->MODEL)
        # Landmarks: rest in mesh space, **assume MP order (0..20)**
        # Landmarks: recompute rest positions from mesh + bone weights
        raw_verts = np.array(model_data["mesh_vertices"], dtype=np.float32)
        self.vert_center = raw_verts.mean(axis=0)
        verts_centered = raw_verts - self.vert_center
        self.vert_scale = float(np.max(np.abs(verts_centered)) + 1e-6)
        self.base_verts = verts_centered / self.vert_scale

        self.mesh_faces = np.array(model_data["mesh_triangles"], dtype=np.int32).reshape(-1, 3)

        self.weights = np.array(model_data["dense_bone_weights"], dtype=np.float32)
        self.num_bones = self.weights.shape[1]

        # Recompute rest landmarks from mesh + bone weights
        self.lm_bone_idx = np.array(model_data["landmark_rest_bone_indices"], dtype=np.int32)    # (21,3)
        self.lm_bone_w   = np.array(model_data["landmark_rest_bone_weights"], dtype=np.float32) # (21,3)

        rest21_raw = compute_rest_landmarks_from_mesh(
            raw_verts,
            self.lm_bone_idx,
            self.lm_bone_w,
            self.weights,
        )  # (21,3) in raw mesh space

        # Normalize landmarks the same way as the mesh
        self.rest21 = (rest21_raw - self.vert_center) / self.vert_scale


        

        # No MP->MODEL remap: MODEL == MP
        self.MP_INDEX_FOR_MODEL = np.arange(21, dtype=np.int32)

        # Wrist is MP 0
        self.model_wrist = 0
        
        self.rest_wrist = self.rest21[self.model_wrist]

        # Wrist-centered rest landmarks (still MP order)
        self.rest_wc = self.rest21 - self.rest_wrist

        # Palm anchors in MP order: wrist + MCPs
        self.palm_ids_model = np.array([0, 5, 9, 13, 17], dtype=np.int32)


        # ---------------------------
        # Landmark -> bones (from JSON)
        # ---------------------------
        self.lm_bone_idx = np.array(model_data["landmark_rest_bone_indices"], dtype=np.int32)    # (21,3)
        self.lm_bone_w = np.array(model_data["landmark_rest_bone_weights"], dtype=np.float32)   # (21,3)

        # Per-bone landmark membership weights (used if you choose weighted Kabsch)
        self.lm_w_per_bone = bone_membership_weights(self.lm_bone_idx, self.lm_bone_w, self.num_bones)

        # ---------------------------
        # Build a more stable per-bone fitting cluster using MP graph neighbors
        # ---------------------------
        # Primary landmark per bone = landmark with strongest assignment to that bone
        self.bone_primary_landmark = [-1] * self.num_bones
        for li in range(21):
            j = int(np.argmax(self.lm_bone_w[li]))
            b = int(self.lm_bone_idx[li, j])
            if 0 <= b < self.num_bones and self.bone_primary_landmark[b] == -1:
                self.bone_primary_landmark[b] = li

        # Bone fit sets (MODEL landmark indices), ensure >=3 if possible
        self.bone_fit_ids = [[] for _ in range(self.num_bones)]
        for b in range(self.num_bones):
            li = self.bone_primary_landmark[b]
            if li < 0:
                continue

            ids = {li}

            p = MP_PARENT.get(li, -1)
            if p >= 0:
                ids.add(p)

            kids = MP_CHILDREN.get(li, [])
            if len(kids) > 0:
                ids.add(kids[0])

            if len(ids) < 3 and p >= 0:
                gp = MP_PARENT.get(p, -1)
                if gp >= 0:
                    ids.add(gp)

            # For the two palm/root-ish bones, force palm anchors
            if b in (0, 1):
                ids = set(self.palm_ids_model.tolist())

            self.bone_fit_ids[b] = sorted(ids)

        print("per-bone fit sizes:", [len(x) for x in self.bone_fit_ids])

        # ---------------------------
        # GL items
        # ---------------------------
        self.mesh_item = gl.GLMeshItem(
            vertexes=self.base_verts.copy(),
            faces=self.mesh_faces,
            shader="shaded",
            color=(0.4, 0.4, 0.7, 1.0),
            smooth=True,
            drawEdges=False,
        )
        self.addItem(self.mesh_item)

        self.ghost_mesh = gl.GLMeshItem(
            vertexes=self.base_verts.copy(),
            faces=self.mesh_faces,
            shader="shaded",
            color=(1, 1, 1, 0.15),
            smooth=True,
            drawEdges=False,
        )
        self.addItem(self.ghost_mesh)

        self.joint_dots = gl.GLScatterPlotItem(
            pos=np.zeros((21, 3), dtype=np.float32),
            color=(1, 0, 0, 1),
            size=8,
        )
        self.addItem(self.joint_dots)

        self.setCameraPosition(distance=3.0, elevation=15, azimuth=45)

        # ---------------------------
        # Calibration / state
        # ---------------------------
        self._last_mp21_raw = None

        # Optional: freeze global similarity (not required, but handy)
        self.use_fixed_global = False
        self.fixed_s = 1.0
        self.fixed_R = np.eye(3, dtype=np.float32)
        self.fixed_t = np.zeros(3, dtype=np.float32)

        self.snap_live = None  # optional

    # ---------------------------
    # Axis mapping + auto-fit
    # ---------------------------
    

    def _preprocess_mp(self, mp21: np.ndarray) -> np.ndarray:
        """Apply current toggles."""
        return self._apply_axis_map(mp21, self.swap_yz, self.flip_x, self.flip_z).astype(np.float32)

    def _fit_axis_map_to_rest(self, mp21_raw: np.ndarray):
        """
        Try all (swap_yz, flip_x, flip_z) combos and choose the one that best aligns
        the palm anchors after a global similarity fit.
        """
        best = None
        best_err = 1e18

        rest_wc = self.rest_wc.astype(np.float64)
        palm = self.palm_ids_model

        for swap_yz in (False, True):
            for flip_x in (False, True):
                for flip_z in (False, True):
                    live = self._apply_axis_map(mp21_raw, swap_yz, flip_x, flip_z)  # MP order
                    live = live[self.MP_INDEX_FOR_MODEL]                           # MODEL order
                    live = live - live[self.model_wrist]                           # wrist-center (MODEL wrist)

                    s, Rg, tg = _umeyama_similarity(live[palm], rest_wc[palm], with_scale=True)
                    live_mesh = (s * (live @ Rg) + tg)

                    err = float(np.mean(np.linalg.norm(live_mesh[palm] - rest_wc[palm], axis=1)))
                    if err < best_err:
                        best_err = err
                        best = (swap_yz, flip_x, flip_z)

        self.swap_yz, self.flip_x, self.flip_z = best
        print("Axis map chosen:", best, "palm err:", best_err)

    # ---------------------------
    # UI actions
    # ---------------------------
    def toggle_ghost(self, visible: bool):
        self.ghost_mesh.setVisible(visible)

    def reset_pose_calibration(self):
        self.snap_live = None
        self.use_fixed_global = False

    def calibrate_pose_snap(self):
        if self._last_mp21_raw is None:
            return
        live = self._preprocess_mp(self._last_mp21_raw).astype(np.float64)
        live = live[self.MP_INDEX_FOR_MODEL]
        live = live - live[self.model_wrist]
        self.snap_live = live.astype(np.float32).copy()

    def calibrate_alignment_from_last(self):
        if self._last_mp21_raw is None:
            return

        # 1) Choose best axis mapping
        self._fit_axis_map_to_rest(self._last_mp21_raw)

        # 2) Preprocess MP landmarks (MODEL order, wrist-centered)
        live = self._preprocess_mp(self._last_mp21_raw).astype(np.float64)
        live = live[self.MP_INDEX_FOR_MODEL]
        live = live - live[self.model_wrist]

        # ---------------------------------------------------------
        # 3) APPLY CORRECTIVE TRANSFORM **ONLY ONCE**
        # ---------------------------------------------------------
        if not hasattr(self, "rest_corrected") or not self.rest_corrected:

            # Compute corrective transform (rotation + translation)
            R_corr, t_corr = compute_rest_corrective_transform(
                self.rest21,
                live.astype(np.float32),
                wrist_idx=self.model_wrist,
                palm_ids=self.palm_ids_model.tolist()
            )

            # Apply corrective transform to rest landmarks
            self.rest21 = (self.rest21 @ R_corr) + t_corr

            # Compute wrist offset
            offset = correct_wrist_offset(
                self.rest21,
                live.astype(np.float32),
                wrist_idx=self.model_wrist
            )

            # Apply wrist offset
            self.rest21 += offset

            # Recompute wrist-centered rest pose
            self.rest_wrist = self.rest21[self.model_wrist]
            self.rest_wc = self.rest21 - self.rest_wrist

            # Mark as corrected so we never do this again
            self.rest_corrected = True

        # ---------------------------------------------------------
        # 4) Freeze global similarity (s, R, t)
        # ---------------------------------------------------------
        s, Rg, tg = _umeyama_similarity(
            live[self.palm_ids_model],
            self.rest_wc[self.palm_ids_model],
            with_scale=True
        )

        self.fixed_s = float(s)
        self.fixed_R = Rg.astype(np.float32)
        self.fixed_t = tg.astype(np.float32)
        self.use_fixed_global = True

        print("Global similarity frozen (s,R,t).")


    # ---------------------------
    # Frame update
    # ---------------------------
    def update_3d_pose(self, mp_landmarks21: np.ndarray):
        # Save raw MP for buttons
        self._last_mp21_raw = mp_landmarks21.copy()

        # 1) Axis-map using current toggles
        live = self._preprocess_mp(mp_landmarks21).astype(np.float64)  # MP order
        live = live[self.MP_INDEX_FOR_MODEL]  # becomes no-op
        live = live - live[self.model_wrist]


        # 4) Global similarity on palm anchors (MODEL order)
        if self.use_fixed_global:
            s, Rg, tg = self.fixed_s, self.fixed_R.astype(np.float64), self.fixed_t.astype(np.float64)
        else:
            s, Rg, tg = _umeyama_similarity(live[self.palm_ids_model], self.rest_wc[self.palm_ids_model], with_scale=True)
            Rg = Rg.astype(np.float64)
            tg = tg.astype(np.float64)

        live_mesh_wc = (float(s) * (live @ Rg) + tg)  # wrist-centered mesh space
        # ---------------------------------------------------------
        # DEBUG: GLOBAL-ONLY ALIGNMENT TEST
        # ---------------------------------------------------------
        # Show only the globally aligned landmarks (no bone fitting, no skinning)
        # self.joint_dots.setData(
        #     pos=(live_mesh_wc + self.rest_wrist.astype(np.float64)).astype(np.float32)
        # )
        # return
        # ---------------------------------------------------------


    
   


        # 5) Per-bone transforms in wrist-centered mesh space
        R_list = [np.eye(3, dtype=np.float32) for _ in range(self.num_bones)]
        t_list = [np.zeros(3, dtype=np.float32) for _ in range(self.num_bones)]

        rest_wc = self.rest_wc.astype(np.float64)

        for b in range(self.num_bones):
            ids = self.bone_fit_ids[b]
            if len(ids) < 3:
                continue

            A = rest_wc[ids]
            B = live_mesh_wc[ids]

            # Use weights if you want (often helps when some points are noisy)
            wb = self.lm_w_per_bone[b, ids].astype(np.float64)
            if float(wb.sum()) > 1e-8:
                R, t = _weighted_kabsch(A, B, wb)
            else:
                R, t = _kabsch(A, B)

            R_list[b] = R
            t_list[b] = t

        # 6) LBS on wrist-centered vertices
        V_wc = (self.base_verts - self.rest_wrist).astype(np.float32)  # wrist-centered mesh verts
        out_wc = np.zeros_like(V_wc, dtype=np.float32)

        for b in range(self.num_bones):
            wv = self.weights[:, b:b + 1]
            if float(wv.max()) < 1e-6:
                continue
            out_wc += wv * (V_wc @ R_list[b] + t_list[b])

        out = out_wc + self.rest_wrist  # back to absolute mesh space

        self.mesh_item.setMeshData(vertexes=out, faces=self.mesh_faces)

        # Dots: convert live from wrist-centered mesh space back to absolute mesh space
        self.joint_dots.setData(pos=(live_mesh_wc + self.rest_wrist.astype(np.float64)).astype(np.float32))

        # Minimal debug (keep it lightweight)
        # MODEL order tips: thumb tip=0, index tip=1, wrist=20
        if False:
            print("MODEL thumb tip live_wc:", live[0], "rest_wc:", rest_wc[0])
            print("MODEL index tip live_wc:", live[1], "rest_wc:", rest_wc[1])
            print("MODEL wrist live_wc:", live[20], "rest_wc:", rest_wc[20])

    



class QtCapture(QtWidgets.QWidget):
    pose_data_ready = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, source=0):
        super().__init__()
        self.cap = cv2.VideoCapture(source)
        self.fps = 24

        self.mp_hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
            max_num_hands=1,
            model_complexity=1,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.annotated_view = QtWidgets.QLabel()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.annotated_view)

        self.parent_slider = None
        self.timer = None

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            points = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark], dtype=np.float32)

            self.pose_data_ready.emit(points)
            self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        if self.parent_slider:
            self.parent_slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

        self.display_image(frame, self.annotated_view)

    def display_image(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(420, 320, QtCore.Qt.KeepAspectRatio))

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(int(1000.0 / self.fps))


class ControlWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        model_path = "visual2pose/generic_hand_model.json"
        with open(model_path, "r") as f:
            model_data = json.load(f)

        self.plotter = HandPlotter(model_data)
        self.plotter.show()

        self.capture_widget = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Hand Control Panel")
        layout = QtWidgets.QVBoxLayout(self)

        self.btn_live = QtWidgets.QPushButton("Start Live")
        self.btn_load = QtWidgets.QPushButton("Load Video")

        self.btn_calib_pose = QtWidgets.QPushButton("Calibrate Pose (Snap)")
        self.btn_calib_align = QtWidgets.QPushButton("Calibrate Alignment (Open Hand)")
        self.btn_reset_pose = QtWidgets.QPushButton("Reset Pose Calibration")

        self.ghost_check = QtWidgets.QCheckBox("Show Ghost Reference")
        self.ghost_check.setChecked(True)
        self.ghost_check.stateChanged.connect(lambda s: self.plotter.toggle_ghost(s == QtCore.Qt.Checked))

        # Axis toggles (still usable, but "Calibrate Alignment" will overwrite them)
        self.swapyz_check = QtWidgets.QCheckBox("Swap Y/Z")
        self.swapyz_check.setChecked(True)
        self.swapyz_check.stateChanged.connect(self.toggle_swapyz)

        self.flipx_check = QtWidgets.QCheckBox("Flip X (Mirror)")
        self.flipx_check.setChecked(True)
        self.flipx_check.stateChanged.connect(self.toggle_flipx)

        self.flipz_check = QtWidgets.QCheckBox("Flip Z (Depth)")
        self.flipz_check.setChecked(False)
        self.flipz_check.stateChanged.connect(self.toggle_flipz)

        self.lbl_speed = QtWidgets.QLabel("Playback Speed: 1.0x")
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setRange(10, 300)
        self.speed_slider.setValue(100)

        self.timeline = QtWidgets.QSlider(QtCore.Qt.Horizontal)

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
        ]:
            layout.addWidget(w)

        self.btn_live.clicked.connect(self.startLive)
        self.btn_load.clicked.connect(self.loadVideo)

        self.btn_calib_pose.clicked.connect(self.plotter.calibrate_pose_snap)
        self.btn_calib_align.clicked.connect(self.plotter.calibrate_alignment_from_last)
        self.btn_reset_pose.clicked.connect(self.plotter.reset_pose_calibration)

        self.speed_slider.valueChanged.connect(self.change_speed)

    def toggle_swapyz(self, state):
        self.plotter.swap_yz = (state == QtCore.Qt.Checked)

    def toggle_flipx(self, state):
        self.plotter.flip_x = (state == QtCore.Qt.Checked)

    def toggle_flipz(self, state):
        self.plotter.flip_z = (state == QtCore.Qt.Checked)

    def change_speed(self, val):
        speed = val / 100.0
        self.lbl_speed.setText(f"Playback Speed: {speed:.1f}x")
        if self.capture_widget and self.capture_widget.timer:
            self.capture_widget.timer.setInterval(int(1000.0 / (24 * speed)))

    def startLive(self):
        self.setup_cap(0)

    def loadVideo(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video")
        if f:
            self.setup_cap(f)
            total = int(self.capture_widget.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.timeline.setRange(0, max(total - 1, 0))

    def setup_cap(self, src):
        if self.capture_widget:
            self.capture_widget.deleteLater()

        self.capture_widget = QtCapture(src)
        self.capture_widget.pose_data_ready.connect(self.plotter.update_3d_pose)
        self.capture_widget.parent_slider = self.timeline
        self.capture_widget.show()
        self.capture_widget.start()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = ControlWindow()
    main.show()
    sys.exit(app.exec_())