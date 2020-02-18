import sys
import gym
import pygame as pg
import numpy as np
from .. import tools
from .. import constants as c
from ..states import main_menu, load_screen, level_state
if c.GENERATE_MAP:
    from ..states import level_gen as level
else:
    from ..states import level


class MarioEnv(gym.Env):

    def __init__(self, mode='bot'):
        if mode == 'human':
            c.HUMAN_PLAYER = True

        self.done = False
        self.mario_x_last = c.DEBUG_START_X
        self.game = tools.Control()
        state_dict = {c.MAIN_MENU: main_menu.Menu(),
                      c.LOAD_SCREEN: load_screen.LoadScreen(),
                      c.LEVEL: level.Level(),
                      c.GAME_OVER: load_screen.GameOver(),
                      c.TIME_OUT: load_screen.TimeOut()}
        self.game.setup_states(state_dict, c.MAIN_MENU)
        if c.HUMAN_PLAYER:
            self.game.main()
            sys.exit(0)

    def step(self, action):
        self.game.event_loop()
        self.game.keys = action
        self.game.update()
        self.game.clock.tick(self.game.fps)

        observation = None
        reward = 0
        if self.game.state == self.game.state_dict[c.LEVEL]:
            observation = self.get_observation()
            reward = self._reward()
            if self.game.state_dict[c.LEVEL].done or self.game.state_dict[c.LEVEL].player.dead:
                self.done = True

        info = self.game.state.persist
        info['x_btn'] = self.game.x_btn

        # returns observation, reward, done, info
        return observation, reward, self.done, info

    def _reward(self):
        current_x = self.game.state_dict[c.LEVEL].player.rect.x
        reward = current_x - self.mario_x_last
        self.mario_x_last = current_x
        return reward

    def _will_reset(self):
        """Handle any hacking before a reset occurs."""
        if self.game.state.next == c.GAME_OVER:
            self.game.state.persist = {
                c.COIN_TOTAL: 0,
                c.SCORE: 0,
                c.LIVES: 3,
                c.TOP_SCORE: 0,
                c.CURRENT_TIME: 0.0,
                c.LEVEL_NUM: 1,
                c.PLAYER_NAME: c.PLAYER_MARIO
            }

    def _did_reset(self):
        """Handle any hacking after a reset occurs."""
        self.mario_x_last = c.DEBUG_START_X

    def reset(self):
        self._will_reset()
        self.game.flip_state(force=c.LEVEL)
        self._did_reset()
        self.done = False
        return self.get_observation()

    def get_observation(self):
        return level_state.get_observation(self.game.state_dict[c.LEVEL].player)

    @staticmethod
    def render(mode='human'):
        if mode == 'human':
            pg.display.update()
