import sys
import numpy as np
from loop_rate_limiters import RateLimiter
import pyrealsense2 as rs
import cv2
import sapien.core as sapien
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.limits import ConfigurationLimit
import mediapipe as mp
from dex_retargeting.retargeting_config import RetargetingConfig

sys.path.append("./example/vector_retargeting")
from single_hand_detector import SingleHandDetector

rate = RateLimiter(frequency=30.0)

# Mediapipe Hands

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# URDF Path
urdf_path = "./assets/robots/assembly/ur5e_shadow/ur5e_shadow_right_hand_glb.urdf" # Options -> ur5e_shadow, xarm7_ability

# --- Retargeter Setup ---
retargeting_config = RetargetingConfig.load_from_file("./src/dex_retargeting/configs/teleop/shadow_hand_right_dexpilot.yml") # Replace shadow_hand w/ hand
retargeter = retargeting_config.build()

# --- SAPIEN Setup ---
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene()
scene.set_timestep(1/240)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, -1, -1], [1, 1, 1])
scene.add_ground(altitude=0)

loader = scene.create_urdf_loader()
loader.fix_root_link = True
loader.load_nonconvex_collisions = False
robot = loader.load(urdf_path)
robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

viewer = scene.create_viewer()
viewer.set_camera_xyz(x=1.5, y=0.0, z=1.5)
viewer.set_camera_rpy(r=0, p=-0.5, y=3.14)

# --- Joint mapping ---
sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
retargeting_joint_names = retargeter.joint_names
retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names if name in retargeting_joint_names]).astype(int)

# --- Pink Setup ---
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()
configuration_limit = ConfigurationLimit(model)

# Default Position
q_init = pin.neutral(model)
q_init[0] = 0
q_init[1] = -np.pi/2
q_init[2] = np.pi/2
q_init[3] = np.pi
q_init[4] = -np.pi/2
q_init[5] = np.pi/2

configuration = pink.Configuration(model, data, q_init)
configuration.update(q_init)
robot.set_qpos(q_init[:robot.dof])

# Shadow Hand -> "palm"
# Ability Hand -> "base"
palm_name = "palm" # Change This
palm_task = FrameTask(palm_name, position_cost=1.0, orientation_cost=0.5)
palm_task.set_target(configuration.get_transform_frame_to_world(palm_name))
palm_down_rotation = configuration.get_transform_frame_to_world(palm_name).rotation.copy()

posture_task = PostureTask(cost=1e-1)
posture_task.set_target(q_init)

# --- Hand Detector ---
detector = SingleHandDetector(hand_type="Right", selfie=False)

# --- RealSense Setup ---
pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(rs_config)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

# --- PnP Setup For Rotation ---
MODEL_POINTS_3D = np.array([
    [  0.0,    0.0,   0.0],
    [ 36.0,   10.0,  20.0],
    [ 52.0,   25.0,  25.0],
    [ 62.0,   45.0,  20.0],
    [ 68.0,   60.0,  15.0],
    [ 30.0,   80.0,   0.0],
    [ 30.0,  115.0,   0.0],
    [ 30.0,  135.0,   0.0],
    [ 30.0,  150.0,   0.0],
    [ 10.0,   85.0,   0.0],
    [ 10.0,  125.0,   0.0],
    [ 10.0,  145.0,   0.0],
    [ 10.0,  160.0,   0.0],
    [-10.0,   80.0,   0.0],
    [-10.0,  115.0,   0.0],
    [-10.0,  135.0,   0.0],
    [-10.0,  147.0,   0.0],
    [-28.0,   72.0,   0.0],
    [-28.0,  100.0,   0.0],
    [-28.0,  115.0,   0.0],
    [-28.0,  125.0,   0.0],
], dtype=np.float64)

PALM_INDICES = [0, 1, 5, 9, 13, 17]
PALM_POINTS_3D = MODEL_POINTS_3D[PALM_INDICES]

camera_matrix = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))

def solve_hand_pnp(pts_2d):
    palm_2d = pts_2d[PALM_INDICES].astype(np.float64)
    success, rvec, tvec = cv2.solvePnP(
        PALM_POINTS_3D,
        palm_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    return success, rvec, tvec

# Camera to sim rotation mapping
R_cam_to_sim = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0]
])

# --- Workspace ---
# Edit these variables to make changes in loop
sim_x_fixed = 0.4
sim_y_range = (-0.3, 0.3)
sim_z_range = (0.2, 0.6)
smoothed_pos = np.array([0.4, 0.0, 0.5])
alpha = 0.05
smoothed_rotation = palm_down_rotation.copy()
rotation_alpha = 0.1  # lower = smoother
hand_qpos_reordered = None
current_rotation = palm_down_rotation.copy()
last_good_pos = smoothed_pos.copy()
last_good_rotation = palm_down_rotation.copy()
pnp_initialized = False
pnp_frame_count = 0

# --- Main Loop ---
while not viewer.closed:
    frames = pipeline.wait_for_frames(timeout_ms=15000)
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data()) * 0.001
    color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    results = hands.process(color_rgb)
    _, joint_pos, keypoint_2d, _ = detector.detect(color_rgb)
    color_image = detector.draw_skeleton_on_image(color_image, keypoint_2d, style="default")
    cv2.imshow("Hand Tracking", color_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        wrist = landmarks.landmark[0]

        # --- Position ---
        sim_y = np.interp(wrist.x, [0.2, 0.8], [sim_y_range[0], sim_y_range[1]])
        sim_z = np.interp(wrist.y, [0.2, 0.8], [sim_z_range[1], sim_z_range[0]])
        target_pos = np.array([sim_x_fixed, sim_y, sim_z])
        smoothed_pos = alpha * target_pos + (1 - alpha) * smoothed_pos

        # --- Rotation via PnP ---
        pts_2d = np.array([[lm.x * 640, lm.y * 480] for lm in landmarks.landmark], dtype=np.float64)
        success, rvec, tvec = solve_hand_pnp(pts_2d)

        if success:
            R_cam, _ = cv2.Rodrigues(rvec)
            R_sim = R_cam_to_sim @ R_cam
            smoothed_rotation = rotation_alpha * R_sim + (1 - rotation_alpha) * smoothed_rotation # Smooth the rotation
            U, _, Vt = np.linalg.svd(smoothed_rotation)
            current_rotation = U @ Vt

            # Only start using PnP rotation after it has stabilized
            if pnp_frame_count > 30:
                pnp_initialized = True
        
        if not pnp_initialized:
            current_rotation = palm_down_rotation # fallback

        # Save last good values
        last_good_pos = smoothed_pos.copy()
        last_good_rotation = current_rotation.copy()
        target_pose = pin.SE3.Identity()
        target_pose.translation = smoothed_pos.copy()
        target_pose.rotation = current_rotation
        palm_task.set_target(target_pose)

          # --- Finger Retargeting ---
        if joint_pos is not None:
            indices = retargeter.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            hand_qpos = retargeter.retarget(ref_value)
            hand_qpos_reordered = hand_qpos[retargeting_to_sapien]
    else:
        target_pose = pin.SE3.Identity()
        target_pose.translation = last_good_pos.copy()
        target_pose.rotation = last_good_rotation.copy()
        palm_task.set_target(target_pose)

    # Inverse Kinematics
    velocity = solve_ik(
        configuration,
        [palm_task, posture_task],
        rate.dt,
        solver="quadprog",
        limits=[configuration_limit]
    )
    configuration.integrate_inplace(velocity, rate.dt)

    # Set Pose
    full_qpos = configuration.q.copy()
    if hand_qpos_reordered is not None:
        full_qpos[6:] = hand_qpos_reordered
    robot.set_qpos(full_qpos)

    # Update Step & Viewer
    scene.step()
    scene.update_render()
    viewer.render()
    rate.sleep()