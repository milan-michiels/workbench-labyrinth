import time

from stable_baselines3.common.callbacks import BaseCallback


class CheckPointCallback(BaseCallback):
    """
    Custom callback for saving the model every step.
    """

    def __init__(self, save_freq, save_path,
                 verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path + str(self.n_calls) + "_" + str(time.time()))
        return True
