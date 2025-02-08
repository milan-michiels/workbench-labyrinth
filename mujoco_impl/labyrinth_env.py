import cv2
import mujoco
import numpy as np
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
from numpy._typing import NDArray


class LabyrinthEnv(MujocoEnv, utils.EzPickle):
    metadata = {'render_modes': ['human', 'rgb_array', 'depth_array'], 'render_fps': 100}

    def __init__(self, episode_length=500, resolution=(8, 8), **kwargs):
        utils.EzPickle.__init__(self, resolution, episode_length, **kwargs)

        self.episode_length = episode_length
        self.resolution = resolution
        self.observation_space = spaces.Box(low=0, high=255, shape=(resolution[0] * resolution[1],), dtype=np.uint8)

        self.step_number = 0

        self.prev_distance = None

        MujocoEnv.__init__(self, observation_space=self.observation_space,
                           model_path="./resources/labyrinth_wo_meshes_actuators.xml", frame_skip=5, **kwargs)

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "sensor_plate_site")  # Get site ID
        self.goal_pos = self.model.site_pos[site_id, :2]  # Extract (x, y) position

    def _get_obs(self):
        """Render the MuJoCo environment from the top camera and process it into a grayscale observation."""
        img = self.render()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        obs = cv2.resize(img_gray, self.resolution, interpolation=cv2.INTER_AREA)  # Resize to desired resolution
        return obs.flatten()

    def step(
            self, action: NDArray[np.float32]
    ):
        self.do_simulation(action, self.frame_skip)
        self.do_simulation(action, self.frame_skip)
        self.step_number += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        done = bool(self._is_done())
        truncated = self.step_number >= self.episode_length
        return obs, reward, done, truncated, {}

    def reset_model(self) -> NDArray[np.float64]:
        self.step_number = 0
        self.prev_distance = None

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _compute_reward(self):
        ball_pos = self.data.qpos[:2]

        distance = np.linalg.norm(ball_pos - self.goal_pos)

        goal_reward = 10.0 if self._is_done() else 0.0

        if self.prev_distance is None:
            self.prev_distance = distance

        distance_reward = self.prev_distance - distance
        self.prev_distance = distance

        time_penalty = -0.01

        return goal_reward + distance_reward + time_penalty

    def _is_done(self) -> bool:
        return self.data.sensordata[0] > 0.5
