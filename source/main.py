import shutil
from threading import Thread

from source.mario_gym.mario_env import MarioEnv
from source.mario_gym.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from source import constants as c
import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo, dqn, impala
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    checkpoint_dir = os.path.join(dir_path, "checkpoints")
    checkpoint_all = os.path.join(dir_path, "checkpoints", "all")

    os.makedirs(checkpoint_dir, exist_ok=True)

    register_env(c.ENV_NAME, lambda c: MarioEnv(c, actions=COMPLEX_MOVEMENT))

    def find_latest_checkpoint():
        largest = -1
        for chkpath in os.listdir(checkpoint_all):
            checkpoint_id = int(chkpath.split("_")[1])

            if checkpoint_id > largest:
                largest = checkpoint_id

        if largest == -1:
            return None
        else:
            ret_path = os.path.join(checkpoint_all, f"checkpoint_{str(largest)}" + f"/checkpoint-{str(largest)}")
            return ret_path

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

    trainer = impala.ImpalaTrainer(env=c.ENV_NAME, config={
        "num_gpus": 2,
        "num_workers": 8,
        #"train_batch_size": 2048,
        "monitor": False,
    })
    latest_checkpoint = find_latest_checkpoint()
    if latest_checkpoint:
        trainer.restore(latest_checkpoint)

    save_interval = 10
    save_counter = 0

    eval_thread = Thread(target=test, args=(trainer, ))
    eval_thread.daemon = True
    eval_thread.start()
    while True:
        trainer.train()
        if save_counter % save_interval == 1:
            checkpoint = trainer.save(checkpoint_all)
            print(checkpoint)
            """_chkp_path = os.path.dirname(os.path.abspath(checkpoint))
            try:
                os.remove(checkpoint_latest)
            except FileNotFoundError:
                pass
            os.symlink(_chkp_path, checkpoint_latest)
            #shutil.copy(checkpoint, checkpoint_latest)
            print("checkpoint saved at", checkpoint)"""
        save_counter += 1
