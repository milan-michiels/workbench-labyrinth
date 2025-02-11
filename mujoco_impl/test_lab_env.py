from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from labyrinth_env import LabyrinthEnv
from tensorboard_integration import TensorboardCallback


def main():
    # env = LabyrinthEnv(episode_length=1000, render_mode='rgb_array')

    eval_env = LabyrinthEnv(episode_length=1500, render_mode='rgb_array')
    eval_env = Monitor(eval_env)

    vec_env = make_vec_env(LabyrinthEnv, n_envs=4, env_kwargs={"episode_length": 1000, "render_mode": "rgb_array"})

    model = SAC("MultiInputPolicy", vec_env, verbose=1, tensorboard_log="./output/sac_labyrinth_tensorboard")
    tensor_callback = TensorboardCallback(eval_env, render_freq=5000)
    model.learn(total_timesteps=200000, log_interval=4, callback=tensor_callback)
    model.save("./output/sac_labyrinth")

    eval_env = RecordVideo(eval_env, video_folder="./output/vids", name_prefix="eval",
                           episode_trigger=lambda x: x % 10 == 0)

    model = SAC.load("./output/sac_labyrinth")
    n_episodes = 20
    for episode_num in range(n_episodes):
        obs, info = eval_env.reset()
        episode_over = False
        while not episode_over:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_over = terminated or truncated
    eval_env.close()


if __name__ == "__main__":
    main()
