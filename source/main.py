from source.mario_gym.mario_env import MarioEnv
from source.mario_gym.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from . import constants as c
import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo


def env_creator(env_config):
    # Use mode='human' as argument to enable keyboard input
    return MarioEnv(actions=SIMPLE_MOVEMENT)


def main():
    register_env(c.ENV_NAME, env_creator)
    print("registered", c.ENV_NAME)

    ray.init()
    trainer = ppo.PPOTrainer(env=c.ENV_NAME)
    while True:
        print(trainer.train())

    # EPISODES = 1000

    # for ep in range(EPISODES):
    #
    #     print("Episode:", ep)
    #     current_state = env.reset()
    #     done = False
    #
    #     while not done:
    #         action = env.action_space.sample()
    #         new_state, reward, done, info = env.step(action)
    #         env.render()
    #         current_state = new_state
    #         if info['x_btn']:
    #             return
