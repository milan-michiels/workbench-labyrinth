from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils


class LabyrinthEnv(MujocoEnv, utils.EzPickle):
    def __init__(self):
        MujocoEnv.__init__(self, './resources/labyrinth_wo_meshes_actuators.xml', 2)
        utils.EzPickle.__init__(self)