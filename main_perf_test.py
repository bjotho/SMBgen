from source.actions import COMPLEX_MOVEMENT
from source.mario_gym.mario_env import MarioEnv

if __name__ == "__main__":

    config = dict(
        window=False,
        fps=60000
    )
    env = MarioEnv(config, actions=COMPLEX_MOVEMENT)

    while True:
        current_state = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            env.render()
            current_state = new_state
            if info['x_btn']:
                continue
