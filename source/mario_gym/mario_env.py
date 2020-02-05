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
        self.done = False
        self.mario_x_last = c.DEBUG_START_X
        self.game = tools.Control()
        state_dict = {c.MAIN_MENU: main_menu.Menu(),
                      c.LOAD_SCREEN: load_screen.LoadScreen(),
                      c.LEVEL: level.Level(),
                      c.GAME_OVER: load_screen.GameOver(),
                      c.TIME_OUT: load_screen.TimeOut()}
        self.game.setup_states(state_dict, c.MAIN_MENU)
        if mode == 'human':
            self.game.main()

    def step(self, action):
        self.game.event_loop()
        self.game.keys = action
        self.game.update()
        self.game.clock.tick(self.game.fps)

        if self.game.state == self.game.state_dict[c.LEVEL]:
            if self.game.state_dict[c.LEVEL].done:
                self.done = True
            elif self.game.state_dict[c.LEVEL].player.dead:
                self.done = True

        info = self.game.state.persist
        info['x_btn'] = self.game.x_btn

        reward = 0
        if self.game.state == self.game.state_dict[c.LEVEL]:
            reward = self._reward()

        # returns State, reward, done, info
        return None, reward, self.done, info

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
        pass

    def reset(self):
        if c.SKIP_BORING_ACTIONS:
            self._will_reset()
            self.game.flip_state(force=c.LEVEL)
            self._did_reset()
            self.done = False
        return self.get_state()

    def get_state(self):
        # TODO - read state
        return 0

    def render(self, display='human'):
        if display == 'human':
            pg.display.update()
