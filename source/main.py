import shutil
from threading import Thread

from source.mario_gym.mario_env import MarioEnv
from source.mario_gym.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from source import constants as c
import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo, dqn
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    checkpoint_dir = os.path.join(dir_path, "checkpoints")
    checkpoint_all = os.path.join(dir_path, "checkpoints", "all")
    checkpoint_latest = os.path.join(checkpoint_dir, "latest.pkl")
    os.makedirs(checkpoint_dir, exist_ok=True)



    register_env(c.ENV_NAME, lambda c: MarioEnv(c, actions=COMPLEX_MOVEMENT))

    def test(trainer):
        config = dict(
            window=False,
            fps=60000
        )
        env = MarioEnv(config, actions=COMPLEX_MOVEMENT)

        while True:
            current_state = env.reset()
            done = False

            while not done:
                action = trainer.compute_action(current_state)
                new_state, reward, done, info = env.step(action)
                env.render()
                current_state = new_state
                if info['x_btn']:
                    return

    ray.init()
    trainer = dqn.DQNTrainer(env=c.ENV_NAME, config={
        "num_workers": 5,
        "monitor": False,
    })
    try:
        trainer.restore(checkpoint_all)
    except:
        pass

    save_interval = 10
    save_counter = 0

    eval_thread = Thread(target=test, args=(trainer, ))
    eval_thread.daemon = True
    eval_thread.start()
    while True:
        trainer.train()
        if save_counter % save_interval == 1:
            checkpoint = trainer.save(checkpoint_all)
            try:
                os.remove(checkpoint_latest)
            except FileNotFoundError:
                pass
            shutil.copy(checkpoint, checkpoint_latest)
            print("checkpoint saved at", checkpoint)
        save_counter += 1
