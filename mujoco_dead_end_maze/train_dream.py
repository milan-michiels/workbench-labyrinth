import logging
import os

from gymnasium.wrappers import RecordVideo
from ray import tune
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config

from labyrinth_env_ray import LabyrinthRayEnv

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def env_creator(env_config):
    return LabyrinthRayEnv(env_config)


tune.register_env("LabyrinthEnv", env_creator)

config = (
    DreamerV3Config()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=False
    )
    .framework("tf2")
    .environment("LabyrinthEnv", env_config={"episode_length": 6000, "render_mode": "rgb_array"})
    .env_runners(num_env_runners=2, num_envs_per_env_runner=2)
)

dreamer = config.build_algo()

for i in range(1_000_00):
    results = dreamer.train()
    logger.info(f"Iter: {i}; avg. results={results['env_runners']}")

    if i % 10000 == 0:
        dreamer.save_to_path(f"./output/checkpoints/dreamer_labyrinth_{i}")

dreamer.save_to_path("./output/dreamer_labyrinth")

eval_env = LabyrinthRayEnv(episode_length=6000, render_mode='rgb_array')
eval_env = RecordVideo(eval_env, video_folder="./output/vids", name_prefix="eval",
                       episode_trigger=lambda x: x % 10 == 0)

n_episodes = 100
succes = 0

for episode_num in range(n_episodes):
    obs, info = eval_env.reset()
    episode_over = False
    while not episode_over:
        action = dreamer.compute_single_action(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated:
            succes += 1
        episode_over = terminated or truncated

eval_env.close()
logger.info(f"Success rate: {(succes / n_episodes) * 100}%")
