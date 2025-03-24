# Milan-Workbench

## Overview

This repository contains the implementation of Reinforcement Learning environments and agents in the Labyrinth digital
twin. The environments are implemented in Mujoco and Gymnasium. There are different iterations before we actually build
a labyrinth environment. In each environment, there will be an environment class which implements the actions,
observations and reward function. Then there will be a training script which will train the agent in the environment.
Depending on the environment, there will be different utilities scripts which helped the agent. We used different
algorithms like PPO, SAC, etc. to train the agent. Because DreamerV3 is only implemented in very few toolkits, we had to
clone the whole sheeprl repository and integrate our environments in it. Because the whole sheeprl isn't fully relevant
we only showed the relevant parts in the project structure. The repo has its own README so feel free to read it to
understand the sheeprl toolkit.

## Project Structure

```sh
└── milan-workbench/
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    
    # Environment Implementations
    ├── bare_gym
    │   ├── explantion_training_vids.md
    │   ├── labyrinth_env.py
    │   └── test_lab_env.py
    ├── mujoco_dead_end_maze
    │   ├── __init__.py
    │   ├── labyrinth_env_intermediate_rewards.py
    │   ├── labyrinth_env_no_intermediate_rewards.py
    │   ├── labyrinth_env_no_intermediate_rewards_with_time_penalty.py
    │   ├── labyrinth_env_no_intermediate_rewards_with_velocity_obs.py
    │   ├── model_viewer.py
    │   ├── path.py
    │   ├── resources
    │   │   └── labyrinth_no_visible_rewards.xml
    │   ├── tensorboard_integration.py
    │   ├── train_ppo.py
    │   └── train_sac.py
    ├── mujoco_plane
    │   ├── labyrinth_env.py
    │   ├── model_viewer.py
    │   ├── resources
    │   │   └── labyrinth_wo_meshes_actuators.xml
    │   ├── tensorboard_integration.py
    │   └── test_lab_env_sac.py
    ├── mujoco_simple_maze
    │   ├── labyrinth_env.py
    │   ├── model_viewer.py
    │   ├── resources
    │   │   └── labyrinth_wo_meshes_actuators.xml
    │   ├── tensorboard_integration.py
    │   └── test_lab_env_sac.py
    
    # SheepRL Integration
    └── sheeprl_lab
        ├── README.md
        ├── assets
        │   └── mujoco
        │       ├── labyrinth_wo_meshes_actuators.xml
        │       └── labyrinth_wo_meshes_actuators_no_intermediate_goals.xml
        ├── sheeprl
        │   ├── __init__.py
        │   ├── __main__.py
        │   ├── configs
        │   │   ├── __init__.py
        │   │   ├── env
        │   │   └── exp
        │   ├── envs
        │   │   ├── __init__.py
        │   │   ├── labyrinth.py
        │   │   └── labyrinth_no_intermediate.py
        │   └── utils
        │       ├── __init__.py
        │       └── lab_path.py
        ├── sheeprl.py
        └── sheeprl_eval.py
```