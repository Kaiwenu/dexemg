import numpy as np
import pyqtgraph.opengl as gl

from ..axis_map import apply_axis_map
from ..kabsch import kabsch, weighted_kabsch
from ..umeyama import umeyama_similarity
from ..rest_pose import (
    compute_rest_landmarks_from_mesh,
    compute_rest_corrective_transform,
)
from ..mesh_ops import apply_corrective_transform_to_mesh
from ..bones import MP_PARENT, MP_CHILDREN, bone_membership_weights

# Standard MediaPipe hand skeleton connections
MP_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15),
    (15,16), (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

class HandPlotter(gl.GLViewWidget):

    # ============================================================
    # Initialization
    # ============================================================
    def __init__(self, model_data):
        super().__init__()

        self.model_data = model_data
        self._init_state()

        self._load_mesh(model_data)
        self._load_weights(model_data)
        self._load_landmark_bindings(model_data)

        self._compute_rest_from_mesh()
        self._build_bone_fit_sets()
        self._init_gl_items()

        self.setCameraPosition(distance=3.0, elevation=15, azimuth=45)

    # ------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------
    def _init_state(self):
        self.swap_yz = False
        self.flip_x = True
        self.flip_z = True

        self.MP_INDEX_FOR_MODEL = np.arange(21, dtype=np.int32)
        self.model_wrist = 0
        self.palm_ids_model = np.array([0, 5, 9, 13, 17], dtype=np.int32)

        self._last_mp21_raw = None
        self.rest_corrected = False

        self.use_fixed_global = False
        self.fixed_s = 1.0
        self.fixed_R = np.eye(3, dtype=np.float32)
        self.fixed_t = np.zeros(3, dtype=np.float32)

        self.smooth_pos = None

    # ============================================================
    # Mesh / Weights / Bindings
    # ============================================================
    def _load_mesh(self, model_data):
        raw = np.array(model_data["mesh_vertices"], dtype=np.float32)

        # We define 'center' as the wrist landmark position in the raw JSON
        # In most MediaPipe-based models, index 0 is the wrist.
        self.vert_center = raw[self.model_wrist].copy()

        centered = raw - self.vert_center
        self.vert_scale = float(np.max(np.abs(centered)) + 1e-6)
        self.base_verts = centered / self.vert_scale
        self.mesh_faces = np.array(model_data["mesh_triangles"], dtype=np.int32).reshape(-1, 3)

    def _load_weights(self, model_data):
        W = np.array(model_data["dense_bone_weights"], dtype=np.float32)
        self.num_bones = W.shape[1]

        W = W / (W.sum(axis=1, keepdims=True) + 1e-8)
        W = np.power(W, 1.0)
        W = W / (W.sum(axis=1, keepdims=True) + 1e-8)

        self.weights = W.astype(np.float32)

    def _load_landmark_bindings(self, model_data):
        self.lm_bone_idx = np.array(model_data["landmark_rest_bone_indices"], dtype=np.int32)
        self.lm_bone_w = np.array(model_data["landmark_rest_bone_weights"], dtype=np.float32)

        self.lm_w_per_bone = bone_membership_weights(
            self.lm_bone_idx, self.lm_bone_w, self.num_bones
        )

    # ============================================================
    # Rest pose computation
    # ============================================================
    def _compute_rest_from_mesh(self):
        raw = self.base_verts * self.vert_scale + self.vert_center

        rest21_raw = compute_rest_landmarks_from_mesh(
            raw, self.lm_bone_idx, self.lm_bone_w, self.weights
        )

        self.rest21 = (rest21_raw - self.vert_center) / self.vert_scale
        self.rest_wrist = self.rest21[self.model_wrist].copy()
        self.rest_wc = self.rest21 - self.rest_wrist

    # ============================================================
    # Bone fitting clusters
    # ============================================================
    def _build_bone_fit_sets(self):
        self.bone_primary_landmark = [-1] * self.num_bones

        for li in range(21):
            j = int(np.argmax(self.lm_bone_w[li]))
            b = int(self.lm_bone_idx[li, j])
            if 0 <= b < self.num_bones and self.bone_primary_landmark[b] == -1:
                self.bone_primary_landmark[b] = li

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
            if kids:
                ids.add(kids[0])

            if len(ids) < 3 and p >= 0:
                gp = MP_PARENT.get(p, -1)
                if gp >= 0:
                    ids.add(gp)

            if b in (0, 1):
                ids = set(self.palm_ids_model.tolist())

            self.bone_fit_ids[b] = sorted(ids)

    # ============================================================
    # GL Items
    # ============================================================
    def _init_gl_items(self):
        self.mesh_item = gl.GLMeshItem(
            vertexes=self.base_verts.copy(),
            faces=self.mesh_faces,
            shader="shaded",
            color=(0.4, 0.4, 0.7, 1.0),
            smooth=True,
            drawEdges=True,
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

        # Initialize the skeleton lines
        # We use 'lines' mode which expects pairs of points (Start, End, Start, End...)
        self.skeleton_lines = gl.GLLinePlotItem(
            pos=np.zeros((len(MP_CONNECTIONS) * 2, 3), dtype=np.float32),
            color=(0, 1, 0, 1), # Bright green to match your "green picture"
            width=2,
            mode='lines'
        )
        self.addItem(self.skeleton_lines)

    def toggle_ghost(self, visible: bool):
        """Show or hide the ghost reference mesh."""
        self.ghost_mesh.setVisible(visible)


    # ============================================================
    # Axis mapping
    # ============================================================
    def _preprocess_mp(self, mp21):
        return apply_axis_map(
            mp21,
            swap_yz=self.swap_yz,
            flip_x=self.flip_x,
            flip_z=self.flip_z,
        ).astype(np.float32)
    
    def _fit_axis_map_to_rest(self, mp21_raw):
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

                    live = apply_axis_map(mp21_raw, swap_yz, flip_x, flip_z)
                    live = live[self.MP_INDEX_FOR_MODEL]
                    live = live - live[self.model_wrist]

                    s, Rg, tg = umeyama_similarity(
                        live[palm], rest_wc[palm], with_scale=True
                    )
                    live_mesh = (s * (live @ Rg) + tg)

                    err = float(np.mean(
                        np.linalg.norm(live_mesh[palm] - rest_wc[palm], axis=1)
                    ))

                    if err < best_err:
                        best_err = err
                        best = (swap_yz, flip_x, flip_z)

        self.swap_yz, self.flip_x, self.flip_z = best
        print("Axis map chosen:", best, "palm err:", best_err)


    # ============================================================
    # Calibration
    # ============================================================
    def calibrate_pose_snap(self):
        if self._last_mp21_raw is None:
            return

        live = self._compute_live_wc(self._last_mp21_raw)
        self.snap_live = live.astype(np.float32).copy()

    def calibrate_alignment_from_last(self):
        if self._last_mp21_raw is None:
            return

        self._fit_axis_map_to_rest(self._last_mp21_raw)
        live = self._compute_live_wc(self._last_mp21_raw)

        if not self.rest_corrected:
            R_corr, t_corr = compute_rest_corrective_transform(
                self.rest21,
                live.astype(np.float32),
                wrist_idx=self.model_wrist,
                palm_ids=self.palm_ids_model.tolist(),
            )

            self.base_verts = apply_corrective_transform_to_mesh(
                self.base_verts, R_corr, t_corr
            )

            self._compute_rest_from_mesh()
            self.rest_corrected = True

        self._freeze_global_similarity(live)

    def _freeze_global_similarity(self, live):
        s, Rg, tg = umeyama_similarity(
            live[self.palm_ids_model],
            self.rest_wc[self.palm_ids_model],
            with_scale=True,
        )

        self.fixed_s = float(s)
        self.fixed_R = Rg.astype(np.float32)
        self.fixed_t = tg.astype(np.float32)
        self.use_fixed_global = True

    # ============================================================
    # Frame update
    # ============================================================
    def update_3d_pose(self, mp21):
        self._last_mp21_raw = mp21.copy()

        live = self._compute_live_wc(mp21)
        s, Rg, tg = self._compute_global_alignment(live)

        live_mesh_wc = float(s) * (live @ Rg) + tg
        full_pos = (live_mesh_wc).astype(np.float32)

        # 2. Update the Skeleton Line Data
        line_pts = []
        for start_idx, end_idx in MP_CONNECTIONS:
            line_pts.append(full_pos[start_idx])
            line_pts.append(full_pos[end_idx])
            
        self.skeleton_lines.setData(pos=np.array(line_pts))

        R_list, t_list = self._compute_bone_transforms(live_mesh_wc)
        out = self._apply_skinning(R_list, t_list)

        if self.smooth_pos is None:
            self.smooth_pos = full_pos
        else:
            # 70% old position, 30% new position to kill the "jitter"
            self.smooth_pos = self.smooth_pos * 0.7 + full_pos * 0.3
            
        self.mesh_item.setMeshData(vertexes=out, faces=self.mesh_faces)
        # self.joint_dots.setData(pos=(live_mesh_wc + self.rest_wrist).astype(np.float32))
        self.joint_dots.setData(pos=self.smooth_pos)

    # ============================================================
    # Helpers
    # ============================================================
    def _compute_live_wc(self, mp21):
        live = self._preprocess_mp(mp21).astype(np.float64)
        live = live[self.MP_INDEX_FOR_MODEL]
        return live - live[self.model_wrist]

    def _compute_global_alignment(self, live):
        if self.use_fixed_global:
            return self.fixed_s, self.fixed_R.astype(np.float64), self.fixed_t.astype(np.float64)

        s, Rg, tg = umeyama_similarity(
            live[self.palm_ids_model],
            self.rest_wc[self.palm_ids_model],
            with_scale=True,
        )
        return float(s), Rg.astype(np.float64), tg.astype(np.float64)

    def _compute_bone_transforms(self, live_mesh_wc):
        R_list = [np.eye(3, dtype=np.float32) for _ in range(self.num_bones)]
        t_list = [np.zeros(3, dtype=np.float32) for _ in range(self.num_bones)]

        # We iterate through the joints based on your MP_PARENT hierarchy
        # This ensures parent bones (knuckles) are calculated before child bones (tips)
        for child_idx, parent_idx in MP_PARENT.items():
            if parent_idx == -1: continue # Skip the wrist for now
            
            # 1. Get the direction vector in 'Rest' pose
            v_rest = self.rest_wc[child_idx] - self.rest_wc[parent_idx]
            v_rest_norm = v_rest / (np.linalg.norm(v_rest) + 1e-8)
            
            # 2. Get the direction vector in 'Live' pose (MediaPipe)
            v_live = live_mesh_wc[child_idx] - live_mesh_wc[parent_idx]
            v_live_norm = v_live / (np.linalg.norm(v_live) + 1e-8)
            
            # 3. Calculate rotation from rest vector to live vector
            # This is the "Look-At" math UmeTrack uses
            dot = np.dot(v_rest_norm, v_live_norm)
            axis = np.cross(v_rest_norm, v_live_norm)
            axis_len = np.linalg.norm(axis)
            
            if axis_len < 1e-8:
                R = np.eye(3)
            else:
                axis = axis / axis_len
                # Rodrigues formula for rotation matrix
                K = np.array([[0, -axis[2], axis[1]],
                            [axis[2], 0, -axis[0]],
                            [-axis[1], axis[0], 0]])
                R = np.eye(3) + K * axis_len + (K @ K) * (1 - dot)

            # 4. Map back to the specific bone index in your model
            # You'll need a mapping between MP landmarks and your JSON bone indices
            bone_idx = self.lm_bone_idx[child_idx, 0] 
            R_list[bone_idx] = R.astype(np.float32)
            
            # t is essentially the position of the parent joint
            t_list[bone_idx] = live_mesh_wc[parent_idx].astype(np.float32)

        return R_list, t_list

    def _apply_skinning(self, R_list, t_list):
        V_wc = (self.base_verts).astype(np.float32)
        out_wc = np.zeros_like(V_wc, dtype=np.float32)

        for b in range(self.num_bones):
            wv = self.weights[:, b:b + 1]
            if float(wv.max()) < 1e-6:
                continue
            out_wc += wv * (V_wc @ R_list[b] + t_list[b])

        return out_wc + self.rest_wrist
