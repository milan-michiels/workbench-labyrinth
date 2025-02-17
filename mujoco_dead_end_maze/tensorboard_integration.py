from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True,
                 log_interval=4,
                 verbose=0):
        super().__init__(verbose)
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self._log_interval = log_interval

    def _on_step(self) -> bool:
        env = self.training_env
        distance = env.env_method("_compute_distances")
        distance = distance[0][0]
        ep_distance_mean = np.mean(distance)
        self.logger.record('trajectory/avg_distance', ep_distance_mean)

        successes = env.get_attr("succes")[0]
        self.logger.record('eval/success_rate', (successes / self._log_interval) * 100)
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                # We expect `render()` to return a uint8 array with values in [0, 255] or a float array
                # with values in [0, 1], as described in
                # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video
                screen = self._eval_env.render()
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.from_numpy(np.asarray([screens])), fps=30),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True
