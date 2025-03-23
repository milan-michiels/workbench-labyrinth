import importlib
from typing import Optional

import cv2
import mujoco
import numpy as np
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
from numpy._typing import NDArray
from sheeprl.utils.lab_path import closest_point_on_path, distance_along_path, path_coords, find_path_index, \
    get_next_targets
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class LabyrinthEnv(MujocoEnv, utils.EzPickle):
    metadata = {'render_modes': ['human', 'rgb_array', 'depth_array'], 'render_fps': 50}

    def __init__(self, episode_length=500, resolution=(64, 64), intermediate_goals=5, evaluation_vid=False, padding=120,
                 n_envs=1,
                 target_points=5, demo=False,
                 **kwargs):
        utils.EzPickle.__init__(self, resolution, episode_length * n_envs, **kwargs)

        self.start_episode_length = episode_length
        self.episode_length = episode_length
        self.resolution = resolution
        self.observation_space = spaces.Dict(
            image=spaces.Box(low=0, high=256, shape=(3, resolution[0], resolution[1]), dtype=np.uint8),
            states=spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            targets=spaces.Box(low=-np.inf, high=np.inf, shape=(target_points * 2,), dtype=np.float32),
            progress=spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)
        )

        self.step_number = 0
        self.evaluation_vid = evaluation_vid
        self.padding = padding

        self.prev_distance = None
        self.succes = 0
        self.intermediate_goals = intermediate_goals
        self.target_points = target_points
        # self.end_steps = 1200
        # self.decay_rate = 0.00005
        self.n_envs = n_envs
        # self.max_steps = max_steps
        # self.initial_threshold = 40
        # self.final_threshold = 8
        self.total_steps = 0

        MujocoEnv.__init__(self, observation_space=self.observation_space,
                           model_path="./assets/mujoco/labyrinth_wo_meshes_actuators.xml",
                           camera_name="top_view",
                           frame_skip=1, **kwargs)

        end_goal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_goal")  # Get site ID
        self.end_goal_pos = self.model.site_pos[end_goal_id, :2]  # Extract (x, y) position
        self.end_goal_size = self.model.site_size[end_goal_id, :2]  # Half extents
        self.last_pos_path = None
        self.last_action = None
        self.last_reward = None
        self.tot_reward = 0
        self.demo = demo
        self.obs = True
        self.last_index = None

        if self.demo:
            self.pyautogui = importlib.import_module('pyautogui')
        else:
            self.pyautogui = None

        for i in range(1, self.intermediate_goals + 1):
            goal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"goal_{i}")
            intermediate_goal_pos = self.model.site_pos[goal_id, :2]
            intermediate_goal_size = self.model.site_size[goal_id, :2]
            setattr(self, f"intermediate_goal_{i}_pos", intermediate_goal_pos)
            setattr(self, f"intermediate_goal_{i}_reached", False)
            setattr(self, f"intermediate_goal_{i}_size", intermediate_goal_size)

    def _get_obs(self):
        return {
            "image": self._get_image(),
            "states": self._get_state(),
            "progress": np.array(self._compute_distances(), dtype=np.float32),
            "targets": np.array(self._get_target_vectors(), dtype=np.float32)
        }

    def _get_target_vectors(self):
        goal_vectors = []

        next_targets_coords = get_next_targets(self.last_pos_path, last_known_index=self.last_index,
                                               num_next_points=self.target_points)

        ball_coords = np.array(self.data.body("ball").xpos[:2])

        for target in next_targets_coords:
            target = np.array(target)
            target_vector = target - ball_coords
            goal_vectors.append(target_vector)

        while len(goal_vectors) < self.target_points:
            goal_vectors.append(np.array([0.0, 0.0]))

        return np.concatenate(goal_vectors, axis=0)

    def _get_state(self):
        ball_x_y = self.data.body("ball").xpos[:2].astype(np.float32)
        tilt_x_angle = self.data.joint("tilt_x").qfrc_actuator[0]
        tilt_y_angle = self.data.joint("tilt_y").qfrc_actuator[0]
        return np.concatenate([ball_x_y, [tilt_x_angle, tilt_y_angle]]).astype(np.float32)

    def _get_image(self):
        self.obs = True
        img = self.render()  # Get full RGB image (H, W, 3)
        img = np.ascontiguousarray(img)
        img_rgb = img.copy()

        # (x, y) position of the ball in world coordinates (meters)
        ball_pos = self.data.geom("ball_geom").xpos[:2]

        # Get image dimensions
        img_height, img_width, _ = img_rgb.shape

        ball_x_px, ball_y_px = self.world_to_pixel(ball_pos, img_width, img_height)

        crop_half = 80

        # Adjusted cropping: Shift when near edges
        x1 = max(0, ball_x_px - crop_half)
        x2 = x1 + crop_half * 2
        if x2 > img_width:  # If it exceeds the right edge, shift left
            x2 = img_width
            x1 = max(0, x2 - crop_half * 2)

        y1 = max(0, ball_y_px - crop_half)
        y2 = y1 + crop_half * 2
        if y2 > img_height:  # If it exceeds the bottom edge, shift up
            y2 = img_height
            y1 = max(0, y2 - crop_half * 2)

        img_rgb = self.draw_path_from_point(self.last_pos_path, img_rgb)

        # Crop and resize
        cropped_img = img_rgb[y1:y2, x1:x2]

        obs = cv2.resize(cropped_img, self.resolution, interpolation=cv2.INTER_LINEAR)
        self.obs = False
        return obs

    def draw_path_from_point(self, point_on_path, frame, num_segments=3):
        """ Draw path from the given point onwards for a limited number of segments. """
        frame = frame.astype(np.uint8)
        img_height, img_width, _ = frame.shape
        closest_index = find_path_index(point_on_path, self.last_index)

        start_px = self.world_to_pixel(point_on_path, img_width, img_height)
        index = min(closest_index, len(path_coords) - 1)
        end_world = path_coords[index]
        end_px = self.world_to_pixel(end_world, img_width, img_height)

        cv2.line(frame, tuple(start_px), tuple(end_px), (0, 255, 0), 2)

        for i in range(closest_index, min(closest_index + num_segments, len(path_coords) - 1)):
            end_world = path_coords[i + 1]
            end_px = self.world_to_pixel(end_world, img_width, img_height)

            start_world = path_coords[i]

            start_px = self.world_to_pixel(start_world, img_width, img_height)

            # Draw the segment
            cv2.line(frame, tuple(start_px), tuple(end_px), (0, 255, 0), 2)

        return frame

    def world_to_pixel(self, world_pos, img_width, img_height, camera_name="top_view"):
        """
        Convert a 3D point in world coordinates to pixel coordinates in the image.
        """
        # Get the camera id
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found in the model.")

        # Extract camera parameters:
        cam_pos = self.model.cam_pos[cam_id]  # shape (3,)
        # cam_mat0 is stored as a flat array; reshape it to (3,3)
        cam_mat = self.model.cam_mat0[cam_id].reshape(3, 3)
        fovy = self.model.cam_fovy[cam_id]  # Field-of-view in degrees

        # --- Compute the View Matrix ---
        # In MuJoCo, cam_mat0 gives the rotation from camera to world.
        # The view (camera) transformation matrix is the inverse of that,
        # which is its transpose (since it's orthonormal).
        R = cam_mat.T  # view rotation
        # The translation is -R * cam_pos.
        view = np.eye(4)
        view[:3, :3] = R
        view[:3, 3] = -R @ cam_pos

        # --- Compute the Projection Matrix ---
        # Choose near and far clipping planes:
        near = 0.1
        far = 100.0
        fovy_rad = np.deg2rad(fovy)
        f = 1.0 / np.tan(fovy_rad / 2)
        aspect = img_width / img_height
        proj = np.zeros((4, 4))
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0

        # --- Prepare the World Coordinate ---
        # Ensure that world_pos is a 3D point in homogeneous coordinates.
        if len(world_pos) == 2:
            # If only (x, y) is provided, append the ball's current z coordinate.
            ball_z = self.data.body("ball").xpos[2]
            p_world = np.array([world_pos[0], world_pos[1], ball_z, 1.0])
        else:
            p_world = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0])

        # --- Transform the World Coordinate ---
        p_camera = view @ p_world  # From world to camera coordinates
        p_clip = proj @ p_camera  # From camera to clip space
        if p_clip[3] == 0:
            raise ValueError("Invalid projection: w component is zero.")
        p_ndc = p_clip[:3] / p_clip[3]  # Normalize to get NDC (range -1 to 1)

        # --- Map Normalized Device Coordinates to Pixel Coordinates ---
        # In NDC, x and y are in [-1, 1]. We map them to pixel space.
        pixel_x = int((p_ndc[0] + 1) / 2 * img_width)
        # Note: In image coordinates, y increases downward.
        pixel_y = int((1 - p_ndc[1]) / 2 * img_height)

        return pixel_x, pixel_y

    def render(self):
        frame = super().render()
        if self.evaluation_vid and not self.obs:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if self.last_pos_path is not None:
                frame = self.draw_path_from_point(self.last_pos_path, frame)

            h, w, _ = frame.shape
            padding = self.padding
            new_frame = np.zeros((h + padding, w, 3), dtype=np.uint8)

            new_frame[padding:, :] = frame
            if self.last_action is not None and self.last_reward is not None:
                tot_distance, closest_dist_to_path = self._compute_distances()
                cv2.putText(new_frame, f"Action: {self.last_action}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(new_frame, f"Total Reward: {self.tot_reward:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(new_frame, f"DTP: {closest_dist_to_path:.4f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(new_frame, f"Distance: {tot_distance:.4f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        if self.demo and not self.obs:
            demo_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Get screen resolution
            screen_width, screen_height = self.pyautogui.size()
            h, w = demo_frame.shape[:2]

            # Scale while maintaining aspect ratio
            scale = min(screen_width / w, screen_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_frame = cv2.resize(demo_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Create a black canvas with screen size
            canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            # Calculate top-left corner to center the image
            x_offset = (screen_width - new_w) // 2
            y_offset = (screen_height - new_h) // 2

            # Place resized frame onto black canvas (centered)
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame

            cv2.namedWindow("Evaluation", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Evaluation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Evaluation', canvas)

            cv2.waitKey(1)

        return frame

    # def define_episode_length(self):
    #     return self.end_steps + ((self.start_episode_length * self.n_envs) - self.end_steps) * np.exp(
    #         -self.decay_rate * self.total_steps)

    def step(
            self, action: NDArray[np.float32]
    ):
        self.do_simulation(action, self.frame_skip)
        self.step_number += 1
        self.total_steps += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        self.tot_reward += reward

        if self.step_number % 10 == 0 or self.last_action is None:
            self.last_action = action
        self.last_reward = reward

        done = bool(self._goal_reached("end_goal"))
        episode_length = self.episode_length
        if self.evaluation_vid:
            episode_length = self.start_episode_length
        truncated = self.step_number >= episode_length
        if done:
            self.succes += 1
        return (
            obs, reward, done, truncated, {}
        )

    def reset_model(self):

        for i in range(1, self.intermediate_goals + 1):
            setattr(self, f"intermediate_goal_{i}_reached", False)

        self.step_number = 0
        self.prev_distance = None
        self.tot_reward = 0

        logger.info(f"Episode length: {self.episode_length}")

        # Copy the initial state so we don't modify the original
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Only randomize the ball's state (assume ball state is in indices 2 onward)
        ball_start = 2
        # Randomize qpos for the ball
        qpos[ball_start:] = self.init_qpos[ball_start:] + self.np_random.uniform(
            size=qpos[ball_start:].shape, low=-0.01, high=0.08
        )
        # Optionally, you can randomize the ball's velocities as well:
        qvel[ball_start:] = self.np_random.uniform(
            size=qvel[ball_start:].shape, low=-0.01, high=0.01
        )

        # Set the new state; the board remains at its initial state
        self.set_state(qpos, qvel)

        ball_pos = self.data.body("ball").xpos[:2]
        self.last_pos_path = closest_point_on_path(ball_pos[0], ball_pos[1], path_coords[0], 0)[0]

        return self._get_obs()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)
        return self._get_obs(), {}

    def _compute_reward(self):
        distance_to_goal, closest_dist_to_path = self._compute_distances()

        end_goal_reward = 5.0 if self._goal_reached("end_goal") else 0.0

        intermediate_goal_reward = 0.0

        for i in range(1, self.intermediate_goals + 1):
            if self._goal_reached(f"intermediate_goal_{i}"):
                setattr(self, f"intermediate_goal_{i}_reached", True)
                intermediate_goal_reward = 1.0

        if self.prev_distance is None:
            self.prev_distance = distance_to_goal

        distance_reward = self.prev_distance - distance_to_goal

        self.prev_distance = distance_to_goal

        # time_penalty = 0.0
        # progress = min(self.total_steps / self.max_steps, 1.0)
        # current_threshold = self.final_threshold + (self.initial_threshold - self.final_threshold) * (0.5 ** (self.decay_rate * progress))
        #
        # if self.data.time > current_threshold:
        #     time_penalty = self.data.time * 0.001

        if closest_dist_to_path > 0.15:
            distance_reward -= closest_dist_to_path * 0.02

        return end_goal_reward + distance_reward + intermediate_goal_reward

    def _compute_distances(self):
        ball_pos = self.data.body("ball").xpos[:2]
        closest_point, closest_dist_to_path, last_index = closest_point_on_path(ball_pos[0], ball_pos[1],
                                                                                self.last_pos_path, self.last_index)
        self.last_pos_path = closest_point
        self.last_index = last_index
        distance_to_goal = distance_along_path(closest_point, last_index)
        return distance_to_goal, closest_dist_to_path

    def _goal_reached(self, site) -> bool:
        ball_pos = self.data.body("ball").xpos[:2]

        sensor_plate_pos = getattr(self, f"{site}_pos")
        sensor_plate_size = getattr(self, f"{site}_size")

        # Compute the bounding box of the sensor plate
        x_min = sensor_plate_pos[0] - sensor_plate_size[0]
        x_max = sensor_plate_pos[0] + sensor_plate_size[0]
        y_min = sensor_plate_pos[1] - sensor_plate_size[1]
        y_max = sensor_plate_pos[1] + sensor_plate_size[1]

        if site != "end_goal":
            if getattr(self, f"{site}_reached"):
                return False

        # Check if ball is within the bounds
        return (x_min <= ball_pos[0] <= x_max) and (y_min <= ball_pos[1] <= y_max)
