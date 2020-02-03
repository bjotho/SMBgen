import pygame as pg
from source.mario_gym.joypad_space import JoypadSpace
from source.mario_gym.mario_env import MarioEnv
from source.mario_gym.actions import COMPLEX_MOVEMENT

env = MarioEnv()
env = JoypadSpace(env, COMPLEX_MOVEMENT)
# print("action_space:", env.get_keys_to_action())
EPISODES = 1000


def main():

    for ep in range(EPISODES):

        print("Episode:", ep)
        current_state = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            env.render()
            current_state = new_state
            if info['x_btn']:
                return
