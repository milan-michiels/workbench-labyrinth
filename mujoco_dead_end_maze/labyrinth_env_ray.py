from typing import Optional

import cv2
import mujoco
import numpy as np
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
from numpy._typing import NDArray


class LabyrinthRayEnv(MujocoEnv, utils.EzPickle):
    metadata = {'render_modes': ['human', 'rgb_array', 'depth_array'], 'render_fps': 100}

    def __init__(self, config=None, **kwargs):

        default_config = {
            "episode_length": 500,
            "resolution": (64, 64),
            "intermediate_goals": 5
        }

        if config is None:
            config = {}
        self.config = {**default_config, **config}

        utils.EzPickle.__init__(self, self.config)

        self.episode_length = self.config["episode_length"]
        self.resolution = self.config["resolution"]
        # self.observation_space = spaces.Dict(
        #     image=spaces.Box(low=-1, high=1, shape=(self.resolution[0], self.resolution[1], 3), dtype=np.float32),
        #     states=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        #     goal=spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
        #     progress=spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        # )

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.resolution[0], self.resolution[1], 3),
                                            dtype=np.float32)

        self.step_number = 0

        self.prev_distance = None
        self.succes = 0
        self.intermediate_goals = self.config["intermediate_goals"]

        MujocoEnv.__init__(self, observation_space=self.observation_space,
                           model_path="./resources/labyrinth_wo_meshes_actuators.xml", camera_name="top_view",
                           frame_skip=5, **kwargs)

        end_goal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_goal")  # Get site ID
        self.end_goal_pos = self.model.site_pos[end_goal_id, :2]  # Extract (x, y) position
        self.end_goal_size = self.model.site_size[end_goal_id, :2]  # Half extents

        for i in range(1, self.intermediate_goals + 1):
            goal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"goal_{i}")
            intermediate_goal_pos = self.model.site_pos[goal_id, :2]
            intermediate_goal_size = self.model.site_size[goal_id, :2]
            setattr(self, f"intermediate_goal_{i}_pos", intermediate_goal_pos)
            setattr(self, f"intermediate_goal_{i}_reached", False)
            setattr(self, f"intermediate_goal_{i}_size", intermediate_goal_size)

    def _get_obs(self):
        # return {
        #     "image": self._get_image(),
        #     "states": self.data.body("ball").xpos[:2].astype(np.float32),
        #     "progress": np.array([self._compute_distance()], dtype=np.float32),
        #     "goal": np.array(self._get_goal_coordinates(), dtype=np.float32)
        # }
        return self._get_image()

    def _get_goal_coordinates(self):
        goal_coordinates = []

        for i in range(1, self.intermediate_goals + 1):
            goal_coordinates.append(getattr(self, f"intermediate_goal_{i}_pos"))

        goal_coordinates.append(self.end_goal_pos)

        return np.concatenate(goal_coordinates, axis=0)

    def _get_image(self):
        img = self.render()  # Get full RGB image (H, W, 3)
        img = np.ascontiguousarray(img)

        img_rgb = img.copy()

        # (x, y) position of the ball in world coordinates (meters)
        ball_pos = self.data.geom("ball_geom").xpos[:2]

        # Get image dimensions
        img_height, img_width, _ = img_rgb.shape

        ball_x_px, ball_y_px = self.world_to_pixel(ball_pos, img_width, img_height)

        crop_half = 80

        # Calculate crop boundaries with padding
        x1 = max(0, ball_x_px - crop_half)
        y1 = max(0, ball_y_px - crop_half)
        x2 = min(img_width, ball_x_px + crop_half)
        y2 = min(img_height, ball_y_px + crop_half)

        # Crop and resize
        cropped_img = img_rgb[y1:y2, x1:x2]

        obs = cv2.resize(cropped_img, self.resolution, interpolation=cv2.INTER_LINEAR)
        obs = obs.astype(np.float32) / 127.5 - 1.0
        return obs

    def world_to_pixel(self, world_pos, img_width, img_height, camera_name="top_view"):
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

    def step(
            self, action: NDArray[np.float32]
    ):
        self.do_simulation(action, self.frame_skip)
        self.step_number += 1
        obs = self._get_obs()
        reward = self._compute_reward()

        done = bool(self._goal_reached("end_goal"))
        truncated = self.step_number >= self.episode_length
        if done:
            self.succes += 1
        return obs, reward, done, truncated, {}

    def reset_model(self):

        for i in range(1, self.intermediate_goals + 1):
            setattr(self, f"intermediate_goal_{i}_reached", False)

        self.step_number = 0
        self.prev_distance = None

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

        return self._get_obs()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)
        return self.reset_model(), {}

    def _compute_reward(self):
        distance = self._compute_distance()

        end_goal_reward = 10.0 if self._goal_reached("end_goal") else 0.0

        intermediate_goal_reward = 0.0

        for i in range(1, self.intermediate_goals + 1):
            if self._goal_reached(f"intermediate_goal_{i}"):
                setattr(self, f"intermediate_goal_{i}_reached", True)
                intermediate_goal_reward = 2.0

        if self.prev_distance is None:
            self.prev_distance = distance

        distance_reward = self.prev_distance - distance
        self.prev_distance = distance

        time_penalty = -0.002

        return end_goal_reward + distance_reward + intermediate_goal_reward + time_penalty

    def _compute_distance(self):
        ball_pos = self.data.body("ball").xpos[:2]
        return np.linalg.norm(ball_pos - self.end_goal_pos)

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
