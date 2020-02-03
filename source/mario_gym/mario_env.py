import gym
import pygame as pg
from source import tools
from source import constants as c
from source.states import main_menu, load_screen, level

_ACTION_TO_KEYS = {}


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
        action = pg.key.get_pressed()
        # if action not in _ACTION_TO_KEYS.keys():
        #     print("action:", action)
        #     buttons = []
        #     for i in range(c.ACTION_KEYS):
        #         if action[i]:
        #             buttons.append(i)
        #     self.setup_action_to_keys(buttons)
        #     self.last_keypress = action

        self.game.event_loop()
        self.game.keys = action
        self.game.update()
        self.game.clock.tick(self.game.fps)

        # returns State, reward, done, info
        return None, None, self.game.done, {"x_btn": self.game.x_btn}

    # TODO - Translate action => keypress
    def setup_action_to_keys(self, buttons):
        """Map action to keyboard keys"""

        key = [0 for i in range(c.ACTION_KEYS)]
        for i in buttons:
            key[i] = 1

    def reset(self):
        # TODO - RESET game
        return self.get_state()

    def get_state(self):
        # TODO - read state
        return 0

    def render(self, mode='human'):
        if mode == 'human':
            pg.display.update()
