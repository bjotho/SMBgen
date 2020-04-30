import random
import gym
import os
import numpy as np

from pygame import K_RIGHT, K_LEFT, K_DOWN, K_UP, K_RETURN, K_s, K_a, KMOD_NONE
from source import tools
from source import constants as c
from source.mario_gym.actions import COMPLEX_MOVEMENT
from source.states import main_menu, load_screen, level_state
if c.GENERATE_MAP:
    from source.states import level_gen as level
else:
    from source.states import level

# from source.mario_gym import curiosity


class MarioEnv(gym.Env):

    def __init__(self, config, mode='agent'):
        # TODO - Fix so that the window does not show up while training (window=false)
        has_window = "window" in config and config["window"]
        fps = 60 if "fps" not in config else config["fps"]
        actions = COMPLEX_MOVEMENT if "actions" not in config else config["actions"]

        if not has_window:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        import pygame

        self.pg = pygame
        # self.rollout = curiosity.Rollout()

        self.has_window = has_window
        self.last_observation = None
        self.observation = None
        self.done = False
        self.mario_x_last = c.DEBUG_START_X
        self.clock_last = c.GAME_TIME_OUT
        self.game = tools.Control()
        self.game.fps = fps
        state_dict = {c.MAIN_MENU: main_menu.Menu(),
                      c.LOAD_SCREEN: load_screen.LoadScreen(),
                      c.LEVEL: level.Level(),
                      c.GAME_OVER: load_screen.GameOver(),
                      c.TIME_OUT: load_screen.TimeOut()}
        self.game.setup_states(state_dict, c.MAIN_MENU)

        # a mapping of buttons to pygame values
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
        self._TILE_MAP, self._CHAR_MAP = self.tokenize_tiles(c.TILES)
        # print("mario_env._TILE_MAP:")
        # self.print_dict(self._TILE_MAP)
        # print("mario_env._CHAR_MAP:")
        # self.print_dict(self._CHAR_MAP)
        self.setup_spaces(actions)

    @staticmethod
    def print_dict(_dict):
        for pair in _dict.items():
            print(pair)

    def buttons(self) -> list:
        """Return the buttons that can be used as actions."""
        return list(self._button_map.keys())

    def tokenize_tiles(self, tiles: list):
        # Tokenize tiles and populate tile_map dict with (tile_id: token) pairs,
        # and populate char_map dict with (token: tile_id) pairs.
        tile_map = {}
        char_map = {}
        token_step = 1.0 / (len(tiles) - 1)
        for n, tile in enumerate(tiles):
            tile_map[tile] = n * token_step
            char_map[n * token_step] = tile

        return tile_map, char_map

    def setup_spaces(self, actions: list):
        """Setup binary to discrete action space converter.
           Setup observation space."""

        # create the new action space
        self.action_space = gym.spaces.Discrete(len(actions))

        # create the new observation space
        self.obs_counter = 0
        self.observation = None
        self.observation_frames = np.zeros(shape=(c.OBS_FRAMES, c.OBS_SIZE, c.OBS_SIZE))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(c.OBS_FRAMES, c.OBS_SIZE, c.OBS_SIZE))

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
        # self.game.event_loop()
        self.game.keys = action
        self.game.update()
        self.game.clock.tick(self.game.fps)

        reward = 0
        if self.game.state == self.game.state_dict[c.LEVEL]:
            self.observation = self.get_observation()
            reward = self._reward()
            if self.game.state_dict[c.LEVEL].done or self.game.state_dict[c.LEVEL].player.dead:
                self.done = True

        info = self.game.state.persist

        if c.PRINT_OBSERVATION:
            if not self.obs_counter % 10:
                print("observation:")
                print("[")
                for frame in self.observation_frames:
                    print("  [")
                    for col in frame:
                        print("   ", [self._CHAR_MAP[tile] for tile in col])

                    print("  ]")

                print("]")

            self.obs_counter += 1
            self.obs_counter = self.obs_counter % 10

        if self.has_window:
            self.render()

        # self.rollout.inc_step_count()

        self.last_observation = self.observation
        # returns observation, reward, done, info
        return self.observation, reward, self.done, info

    def _reward(self):
        """Mario reward function"""

        # Current x-value of mario
        current_x = self.game.state_dict[c.LEVEL].player.rect.x

        # Difference in current x-value and last x-value
        reward = current_x - self.mario_x_last

        # Update last x-value
        self.mario_x_last = current_x

        # Time left on game clock
        clock_now = self.game.state_dict[c.LEVEL].overhead_info.time

        # Difference in remaining time. Clock counts down from 300,
        # hence no clock tick: reward += 0, clock tick: reward += 1
        reward += self.clock_last - clock_now

        # Update last clock value
        self.clock_last = clock_now

        # If mario is dead, set reward to -15
        if self.game.state_dict[c.LEVEL].player.dead:
            reward = -15

        return reward

    def _will_reset(self):
        """Handle any hacking before a reset occurs."""
        try:
            game = self.game.state_dict[c.LEVEL]

            # Decay epsilon
            if game.generator.epsilon > c.MIN_EPSILON:
                game.generator.epsilon *= c.EPSILON_DECAY
                game.generator.epsilon = max(c.MIN_EPSILON, game.generator.epsilon)

            # Insert penalty for generator at transition (generation) mario was unable to traverse.
            if not game.mario_done and game.insert_zero_index:
                print("zero_index:", game.zero_reward_index)
                gen = game.gen_list[game.zero_reward_index]
                gen[c.REWARD] = -1
                game.generator.update_replay_memory(gen)
        except AttributeError:
            pass

        if self.game.state.next == c.GAME_OVER:
            self.game.state.persist = {
                c.BASE_FPS: 60,
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
        self.clock_last = c.GAME_TIME_OUT

        # Initialize frame stack
        self.step(0)
        for _ in range(c.OBS_FRAMES):
            self.observation_frames = np.delete(self.observation_frames, 0, axis=0)
            self.observation_frames = np.concatenate((self.observation_frames, self.observation))

    def reset(self):
        """Reset the environment and return the initial observation."""
        self._will_reset()
        self.game.flip_state(force=c.LEVEL)
        self._did_reset()
        self.done = False
        return self.get_observation()

    def get_observation(self):
        raw_observation = level_state.get_observation(self.game.state_dict[c.LEVEL].player)
        self.observation = np.ndarray(shape=(1, c.OBS_SIZE, c.OBS_SIZE))
        for m, i in enumerate(raw_observation):
            for n, j in enumerate(i):
                self.observation[0][m][n] = self._TILE_MAP[j]
        self.observation_frames = np.delete(self.observation_frames, 0, axis=0)
        self.observation_frames = np.concatenate((self.observation_frames, self.observation))
        return self.observation_frames

    def render(self, mode='human', close=False):
        if mode == 'human':
            self.pg.display.update()
