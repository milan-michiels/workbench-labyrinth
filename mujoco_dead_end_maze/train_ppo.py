import logging

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from labyrinth_env import LabyrinthEnv
from tensorboard_integration import TensorboardCallback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    eval_env = LabyrinthEnv(episode_length=7000, render_mode='rgb_array')
    eval_env = Monitor(eval_env)

    vec_env = make_vec_env(LabyrinthEnv, n_envs=6, env_kwargs={"episode_length": 7000, "render_mode": "rgb_array"})
    model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log="./output/ppo_labyrinth_tensorboard",
                device="cuda")
    tensor_callback = TensorboardCallback(eval_env, render_freq=5000, log_interval=20)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./output/checkpoints",
                                             name_prefix="ppo_labyrinth")
    model.learn(total_timesteps=2000000, log_interval=20, callback=[tensor_callback, checkpoint_callback])
    model.save("./output/ppo_labyrinth")

    eval_env = RecordVideo(eval_env, video_folder="./output/vids", name_prefix="eval-ppo",
                           episode_trigger=lambda x: x % 10 == 0)

    model = PPO.load("./output/ppo_labyrinth")
    n_episodes = 100
    succes = 0
    for episode_num in range(n_episodes):
        obs, info = eval_env.reset()
        episode_over = False
        while not episode_over:
            action, _states = model.predict(obs)
            action = action[0]
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated:
                succes += 1
            episode_over = terminated or truncated
    eval_env.close()
    logger.info(f"Success rate: {(succes / n_episodes) * 100}%")


if __name__ == "__main__":
    main()
