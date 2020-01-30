import gym
import pygame as pg
from . import setup, tools
from . import constants as c
from .actions import COMPLEX_MOVEMENT
from .states import main_menu, load_screen, level


class MarioEnv(gym.Env):

    def __init__(self):
        self.last_keypress = None

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

        if action != self.last_keypress:
            print(action)
            print(self.game.done)
            self.last_keypress = action

        # TODO - Translate action => keypress
        self.game.event_loop()
        self.game.keys = action
        self.game.update()
        self.game.clock.tick(self.game.fps)

        # returns State, reward, done, info
        return None, None, self.game.done, {}

    def reset(self):
        # TODO - RESET game
        return self.get_state()

    def get_state(self):
        # TODO - read state
        return 0

    def render(self, mode='human'):
        if mode == 'human':
            pg.display.update()

def main():

    env = MarioEnv()

    # Training params
    EPISODES = 1000

    for ep in range(EPISODES):

        print("Episode:", ep)
        state = env.reset()
        done = False

        while not done:
            action = pg.key.get_pressed()
            state_1, reward, done, info = env.step(action)
            env.render()
            state = state_1
