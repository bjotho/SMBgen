import sys
import gym
import pygame as pg
from pygame import K_RIGHT, K_LEFT, K_DOWN, K_UP, K_RETURN, K_s, K_a, KMOD_NONE
import numpy as np
from .. import tools
from .. import constants as c
from ..states import main_menu, load_screen, level_state
if c.GENERATE_MAP:
    from ..states import level_gen as level
else:
    from ..states import level


class MarioEnv(gym.Env):

    def __init__(self, actions, mode='bot'):
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

        # a mapping of buttons to binary values
        self._button_map = {
            'right': K_RIGHT,
            'left': K_LEFT,
            'down': K_DOWN,
            'up': K_UP,
            'start': K_RETURN,
            'select': K_RETURN,
            'B': K_s,
            'A': K_a,
            'NOOP': KMOD_NONE,
        }

        self._ACTION_TO_KEYS = {}
        self._TILE_MAP = {}
        self.setup_spaces(actions)
        print("observation space in", c.ENV_NAME + ":", self.observation_space)

    def buttons(self) -> list:
        """Return the buttons that can be used as actions."""
        return list(self._button_map.keys())

    def setup_spaces(self, actions: list):
        """Setup binary to discrete action space converter.
           Setup observation space."""

        # create the new action space
        self.action_space = gym.spaces.Discrete(len(actions))
        # create the new observation space
        _obs_size = 2 * c.OBSERVATION_RADIUS + 1
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(_obs_size, _obs_size))
        # create the action map from the list of discrete actions
        self._action_map = {}
        self._action_meanings = {}
        # iterate over all the actions (as button lists)
        for action, button_list in enumerate(actions):
            # the value of this action's bitmap
            action_list = []
            # iterate over the buttons in this button list
            for button in button_list:
                action_list.append(self._button_map[button])
            # set this action maps value to the byte action value
            self._action_map[action] = action_list
            self._action_meanings[action] = ' '.join(button_list)

        self.setup_action_to_keys()
        self.tokenize_tiles()

    def tokenize_tiles(self):
        token_step = 1.0 / (len(c.TILES) - 1)
        for n, tile in enumerate(c.TILES):
            self._TILE_MAP[tile] = n * token_step

    def setup_action_to_keys(self):
        """Map action to keyboard keys"""
        for action in self._action_map.items():
            keyboard = [0 for i in range(c.ACTION_KEYS)]
            for button in action[-1]:
                keyboard[button] = 1

            self._ACTION_TO_KEYS[action[0]] = tuple(keyboard)

    def get_action_to_keys(self):
        return self._ACTION_TO_KEYS

    def get_action_meanings(self):
        """Return a list of actions meanings."""
        actions = sorted(self._action_meanings.keys())
        return [self._action_meanings[action] for action in actions]

    def step(self, action):
        action = self._ACTION_TO_KEYS[action]
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
        """Reset the environment and return the initial observation."""
        self._will_reset()
        self.game.flip_state(force=c.LEVEL)
        self._did_reset()
        self.done = False
        return self.get_observation()

    def get_observation(self):
        return np.array(self._TILE_MAP[tile] for tile in level_state.get_observation(self.game.state_dict[c.LEVEL].player))

    @staticmethod
    def render(mode='human'):
        if mode == 'human':
            pg.display.update()
