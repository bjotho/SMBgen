from source.mario_gym.joypad_space import JoypadSpace
from source.mario_gym.mario_env import MarioEnv
from source.mario_gym.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT


def main():
    # Use mode='human' as argument to enable keyboard input
    env = MarioEnv(mode='human')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    EPISODES = 100

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
