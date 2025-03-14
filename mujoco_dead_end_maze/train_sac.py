import logging

import cv2
from labyrinth_env_no_intermediate_rewards_with_velocity_obs import LabyrinthEnv
from stable_baselines3 import SAC

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    eval_env = LabyrinthEnv(episode_length=4000, render_mode='rgb_array', evaluation_vid=True, demo=True)
    # eval_env = Monitor(eval_env)
    #
    # vec_env = make_vec_env(LabyrinthEnv, n_envs=6, env_kwargs={"episode_length": 4000, "render_mode": "rgb_array"})
    model = SAC("MultiInputPolicy", eval_env, verbose=1, tensorboard_log="./output/sac_labyrinth_tensorboard",
                device="cuda")
    # tensor_callback = TensorboardCallback(eval_env, render_freq=5000, log_interval=20)
    # checkpoint_callback = CheckpointCallback(save_freq=1_000_000, save_path="./output/checkpoints",
    #                                          name_prefix="sac_labyrinth")
    # model.learn(total_timesteps=10_000_000, log_interval=10_000, callback=[tensor_callback, checkpoint_callback])
    # model.save("./output/sac_labyrinth")
    #
    # eval_env = RecordVideo(eval_env, video_folder="./output/vids", name_prefix="eval-sac",
    #                        episode_trigger=lambda x: x % 10 == 0)

    # model = SAC.load("./output/sac_labyrinth")
    n_episodes = 1
    succes = 0
    for episode_num in range(n_episodes):
        obs, info = eval_env.reset()
        episode_over = False
        while not episode_over:
            action, _states = model.predict(obs)
            # action = action[0]
            obs, reward, terminated, truncated, info = eval_env.step(action)
            should_exit = eval_env.render()
            if should_exit.dtype == bool and should_exit:
                break

            if terminated:
                succes += 1
            episode_over = terminated or truncated
    eval_env.close()
    cv2.destroyAllWindows()
    logger.info(f"Success rate: {(succes / n_episodes) * 100}%")


if __name__ == "__main__":
    main()
