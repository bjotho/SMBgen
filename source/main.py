from source.mario_gym.joypad_space import JoypadSpace
from source.mario_gym.mario_env import MarioEnv
from source.mario_gym.actions import COMPLEX_MOVEMENT

env = MarioEnv(human_input=True)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
EPISODES = 4


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
