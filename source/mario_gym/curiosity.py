import ray
import tensorflow as tf
import numpy as np
import gym
from source.mario_gym import mario_env
import functools


class Rollout:
    def __init__(self, ob_space, ac_space, nenvs, nsteps_per_seg, nsegs_per_env, nlumps, envs, policy,
                 int_rew_coeff, ext_rew_coeff, record_rollouts, dynamics, scope='feature_extractor'):
        self.scope = scope
        self.nenvs = nenvs
        self.nsteps_per_seg = nsteps_per_seg
        self.observation = mario_env.MarioEnv.get_observation()
        self.nsegs_per_env = nsegs_per_env
        self.nsteps = self.nsteps_per_seg * self.nsegs_per_env
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.nlumps = nlumps
        self.lump_stride = nenvs // self.nlumps

        self.reward_fun = lambda ext_rew, int_rew: ext_rew_coeff * np.clip(ext_rew, -1., 1.) + int_rew_coeff * int_rew

        self.buf_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_ext_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_acs = np.empty((nenvs, self.nsteps, *self.ac_space.shape), self.ac_space.dtype)
        self.buf_obs = np.empty((nenvs, self.nsteps, *self.ob_space.shape), self.ob_space.dtype)
        self.buf_obs_last = np.empty((nenvs, self.nsegs_per_env, *self.ob_space.shape), np.float32)

        self.step_count = 0

        with tf.variable_scope(scope):
            self.last_ob = tf.placeholder(dtype=tf.int32,
                                          shape=(None, 1) + self.ob_space.shape, name='last_ob')
            self.next_ob = tf.concat([self.obs[:, 1:], self.last_ob], 1)

            if features_shared_with_policy:
                self.features = self.policy.features
                self.last_features = self.policy.get_features(self.last_ob, reuse=True)
            else:
                self.features = self.get_features(self.obs, reuse=False)
                self.last_features = self.get_features(self.last_ob, reuse=True)
            self.next_features = tf.concat([self.features[:, 1:], self.last_features], 1)

            self.ac = self.policy.ph_ac
            self.scope = scope

            self.loss = self.get_loss()

    def step_counter(self):
        self.step_count += 1

    def step_count(self):
        return self.step_count

    def calculate_reward(self):
        int_rew = mario_env.MarioEnv.calculate_loss(ob=self.buf_obs,
                                               last_ob=self.buf_obs_last,
                                               acs=self.buf_acs)
        self.buf_rews[:] = self.reward_fun(int_rew=int_rew, ext_rew=self.buf_ext_rews)


    def _set_env_vars(self):
        env = self.make_env(0, add_monitor=False)
        self.ob_space, self.ac_space = env.observation_space, env.action_space
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        del env
        self.envs = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]

