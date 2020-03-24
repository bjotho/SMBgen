import tensorflow as tf
import numpy as np
from source.mario_gym.mario_env import MarioEnv as ME


class curiosity(ME):
    def __init__(self, ext_rew, int_rew,  int_rew_coeff, ext_rew_coeff):
        self.nenvs = None
        self.int_rew = int_rew
        self.ext_rew = ext_rew
        self.cur_reward = lambda ext_rew, int_rew: ext_rew_coeff *np.clip(ext_rew, -1., 1.) + int_rew_coeff * int_rew
        self.buf_rews = np.empty((nenvs, self.))


    def calculate_curiosity(self):
        int_rew = self.ME.calculate_loss(ob = self.buf_obs, last_ob = ME(self.last_observation), acs= self.buf_acs)
        self.buf_rews[:] = self.cur_reward(ext_rew=ME(self.step(self, action=self.setup_action_to_keys())), )