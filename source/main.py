__author__ = 'marble_xu'

from random import random
import pygame
import gym as gym
import pygame as pg
from . import setup, tools
from . import constants as c
from .states import main_menu, load_screen, level


class MarioEnv(gym.Env):

    def __init__(self):

        self.game = tools.Control()
        state_dict = {c.MAIN_MENU: main_menu.Menu(),
                      c.LOAD_SCREEN: load_screen.LoadScreen(),
                      c.LEVEL: level.Level(),
                      c.GAME_OVER: load_screen.GameOver(),
                      c.TIME_OUT: load_screen.TimeOut()}
        self.game.setup_states(state_dict, c.MAIN_MENU)

    def step(self, action):
        # 0 = left
        # 1 = left + up

        # TODO - Translate action => keypress
        self.game.event_loop()
        self.game.keys = action
        self.game.update()
        pg.display.update()
        self.game.clock.tick(self.game.fps)

        return None, None, None, {}

    def reset(self):
        # TODO - RESET game
        return self.get_state()

    def get_state(self):
        # TODO - read state
        return 0

def main():

    env = MarioEnv()

    EPISODES = 1000

    for episode in range(EPISODES):

        state = env.reset()
        terminal = False

        while not terminal:

            action = pygame.key.get_pressed()
            state_1, reward, terminal, _ = env.step(action)
            state = state_1
