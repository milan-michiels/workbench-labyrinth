import gymnasium
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from stable_baselines3 import SAC
from labyrinth_env import LabyrinthEnv


def main():
    # Register the environment
    gymnasium.register(
        id="gymnasium_env/GridWorld-v0",
        entry_point=LabyrinthEnv
    )

    # Create the training environment (no human rendering to speed up training)
    train_env = gymnasium.make("gymnasium_env/GridWorld-v0", render_mode="rgb_array", max_episode_steps=1000)
    train_env = RecordEpisodeStatistics(train_env)

    # Create the evaluation environment (with rendering or video recording if desired)
    eval_env = gymnasium.make("gymnasium_env/GridWorld-v0", render_mode="rgb_array", max_episode_steps=1000)
    eval_env = RecordVideo(eval_env, video_folder="vids", name_prefix="eval", episode_trigger=lambda x: x % 10 == 0)

    # Create and train the model with a larger number of timesteps
    model = SAC('MlpPolicy',train_env , verbose=1)
    model.learn(total_timesteps=100_000)  # Increase total timesteps

    # Evaluate the trained agent
    n_episodes = 20
    for episode_num in range(n_episodes):
        obs, info = eval_env.reset()
        episode_over = False
        while not episode_over:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_over = terminated or truncated

    eval_env.close()

    # Print recorded statistics
    print(f'Episode time taken: {train_env.time_queue}')
    print(f'Episode total rewards: {train_env.return_queue}')
    print(f'Episode lengths: {train_env.length_queue}')


if __name__ == "__main__":
    main()
