from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC

from labyrinth_env import LabyrinthEnv


def main():
    env = LabyrinthEnv(episode_length=1000, render_mode='rgb_array')

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_labyrinth_tensorboard")
    model.learn(total_timesteps=100000, log_interval=4)

    eval_env = LabyrinthEnv(render_mode='human')
    eval_env = RecordVideo(eval_env, video_folder="./vids", name_prefix="eval", episode_trigger=lambda x: x % 5 == 0)

    n_episodes = 10
    for episode_num in range(n_episodes):
        obs, info = eval_env.reset()
        episode_over = False
        while not episode_over:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_over = terminated or truncated


if __name__ == "__main__":
    main()
