import gym
import pygame as pg
from source import tools
from source import constants as c
from source.states import main_menu, load_screen
if c.GENERATE_MAP:
    from ..states import level_gen as level
else:
    from ..states import level


class MarioEnv(gym.Env):

    def __init__(self, mode='bot'):
        self.mode = mode
        self.game = tools.Control()
        state_dict = {c.MAIN_MENU: main_menu.Menu(),
                      c.LOAD_SCREEN: load_screen.LoadScreen(),
                      c.LEVEL: level.Level(),
                      c.GAME_OVER: load_screen.GameOver(),
                      c.TIME_OUT: load_screen.TimeOut()}
        self.game.setup_states(state_dict, c.MAIN_MENU)

    def step(self, action):
        self.game.event_loop()
        if self.mode == 'bot':
            self.game.keys = action
        self.game.update()
        self.game.clock.tick(self.game.fps)

        # returns State, reward, done, info
        return None, None, self.game.state_dict[c.LEVEL].done, {"x_btn": self.game.x_btn}

    def reset(self):
        # TODO - RESET game
        return self.get_state()

    def get_state(self):
        # TODO - read state
        return 0

    def render(self, display='human'):
        if display == 'human':
            pg.display.update()
