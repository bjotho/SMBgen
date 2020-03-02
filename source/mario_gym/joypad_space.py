"""An environment wrapper to convert binary to discrete action space."""
import gym
from gym import Env
from pygame import K_RIGHT, K_LEFT, K_DOWN, K_UP, K_RETURN, K_s, K_a, KMOD_NONE
from source import constants as c


class JoypadSpace(gym.Env):
    """Convert binary to discrete action space."""

    # a mapping of buttons to binary values
    _button_map = {
        'right':  K_RIGHT,
        'left':   K_LEFT,
        'down':   K_DOWN,
        'up':     K_UP,
        'start':  K_RETURN,
        'select': K_RETURN,
        'B':      K_s,
        'A':      K_a,
        'NOOP':   KMOD_NONE,
    }

    _ACTION_TO_KEYS = {}

    @classmethod
    def buttons(cls) -> list:
        """Return the buttons that can be used as actions."""
        return list(cls._button_map.keys())

    def __init__(self, env: Env, actions: list):
        """
        Initialize a new binary to discrete action space wrapper.

        Args:
            env: the environment to wrap
            actions: an ordered list of actions (as lists of buttons).
                The index of each button list is its discrete coded value

        Returns:
            None

        """
        super().__init__(env)
        # create the new action space
        self.action_space = gym.spaces.Discrete(len(actions))
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


# explicitly define the outward facing API of this module
__all__ = [JoypadSpace.__name__]
