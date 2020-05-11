import ray
import tensorflow as tf
import numpy as np
import gym
from source.mario_gym import mario_env
import functools


class Curiosity:
    def __init__(self, ob_space, ac_space):
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.last_observation = None

        self.reward_fun = lambda ext_rew, int_rew: ext_rew_coeff * np.clip(ext_rew, -1., 1.) + int_rew_coeff * int_rew

        self.step_count = 0

    def step_counter(self):
        self.step_count += 1

    def get_step_count(self):
        return self.step_count

    def last_obs(self):
        self.ob_space = self.last_observation

    def calculate_reward(self):
        int_rew = mario_env.MarioEnv.calculate_loss(ob=self.ob_space,
                                               last_ob=self.buf_obs_last,
                                               acs=self.ac_space)
        self.buf_rews[:] = self.reward_fun(int_rew=int_rew, ext_rew=self.buf_ext_rews)